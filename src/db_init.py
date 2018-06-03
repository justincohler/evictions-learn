import os
import sys
import logging
from db_client import DBClient
import db_statements
import pandas as pd

logger = logging.getLogger('evictionslog')
sh = logging.StreamHandler(sys.stdout)
logger.addHandler(sh)
logger.setLevel(logging.INFO)


class DBInit():
    """Clear and initialize the evictions database."""

    def __init__(self):
        self.db = DBClient()

    def evictions_init(self, level):
        """Clear and initialize the evictions table(s)."""

        logger.info("Dropping table {}...".format(level))
        self.db.write([db_statements.DROP_TABLE_EV_LAB.format(level)])
        logger.info("Creating table {}...".format(level))
        self.db.write([db_statements.CREATE_TABLE_EV_LAB.format(level)])
        logger.info("Copying table...")
        self.db.copy('data/raw/eviction_lab/{}.csv'.format(level),
                     db_statements.COPY_CSV_EVICTIONS.format(level))
        logger.info("Records committed.")
        logger.info("Creating indexes...")
        self.db.write([db_statements.IDX_STATE_YEAR.format(level, level),
                       db_statements.IDX_YEAR.format(level, level),
                       db_statements.IDX_STATE.format(level, level),
                       db_statements.IDX_EVICTIONS.format(level, level),
                       db_statements.IDX_STATE_YEAR.format(level, level),
                       db_statements.IDX_GEOID.format(level, level),
                       db_statements.IDX_GEOID_YEAR.format(level, level),
                       ])
        logger.info("Indexes created.")
        logger.info("Adding sub-geography columns...")
        self.db.write([db_statements.CREATE_VAR_STATE.format(level),
                       db_statements.CREATE_VAR_COUNTY.format(level),
                       db_statements.UPDATE_VAR_STATE.format(level),
                       db_statements.UPDATE_VAR_COUNTY.format(level)
                       ])
        if level == "blockgroup":
            self.db.write([db_statements.CREATE_VAR_TRACT,
                           db_statements.UPDATE_VAR_TRACT])
        logger.info("Sub-geography columns added...")
        logger.info("Dropping records outside SA/ENC Divisions...")
        self.db.write([db_statements.DROP_STATE.format(level)])
        logger.info("{} table completed.".format(level))

    def geo_init(self):
        """Clear and initialize Postgis."""

        self.db.write([
            db_statements.CREATE_EXT_POSTGIS,
            db_statements.CREATE_EXT_FUZZY,
            db_statements.CREATE_EXT_TIGER,
            db_statements.CREATE_EXT_POSTGIS_TOP,
            db_statements.DROP_F_EXEC,
            db_statements.CREATE_F_EXEC,
            db_statements.ALTER_SPATIAL_REF_SYS.format(self.db.DB_USER)
        ])

    def census_shp(self, geography):
        """Read shapes for a given geography."""
        DROP_TABLE_SHP = db_statements.DROP_TABLE_SHP.format(geography)
        self.db.write([DROP_TABLE_SHP])

        shp_read = "shp2pgsql -s 4269:4326 -W 'latin1' data/tl_2010_us_{}10/tl_2010_us_{}10.shp evictions.census_{}_shp | psql {} -U {} -W {} -p {} -h {}".format(
            geography, geography, geography, 'evictions', self.db.DB_USER, self.db.DB_PASSWORD, self.db.DB_PORT, self.db.DB_HOST)
        os.system(shp_read)

    def create_n_year_average(self, source_table, source_col, target_table, lag):
        """Create n-year (lag) average for given attribute, source table, and target table"""

        logger.info("Adding {} year average to {} for feature {}".format(
            lag, target_table, source_col))
        target_col = '{}_avg_{}yr'.format(source_col, lag)
        DROP_COLUMN = db_statements.DROP_COLUMN.format(target_table, target_col)
        ADD_COLUMN = db_statements.ADD_COLUMN.format(
            target_table, target_col, "FLOAT4")
        INSERT_N_YEAR_AVG = db_statements.INSERT_N_YEAR_AVG.format(
            target_table, target_col, source_col, source_table, source_table, lag)

        logger.info("Running:")
        logger.debug(INSERT_N_YEAR_AVG)
        try:
            self.db.write([
                DROP_COLUMN,
                ADD_COLUMN,
                INSERT_N_YEAR_AVG
            ])
        except Exception as e:
            logger.error(e)
            return False

        logger.info("Added {} year average to {} for feature {}".format(
            lag, target_table, source_col))
        return True

    def create_n_year_pct_change(self, source_table, source_col, target_table, lag):
        """Create n-year (lag) percentage change for given attribute, source table, and target table"""

        logger.info("Adding {} year pct change to {} for feature {}".format(
            lag, target_table, source_col))
        target_col = '{}_pct_change_{}yr'.format(source_col, lag)
        DROP_COLUMN = db_statements.DROP_COLUMN.format(target_table, target_col)
        ADD_COLUMN = db_statements.ADD_COLUMN.format(
            target_table, target_col, "FLOAT")
        INSERT_N_YEAR_PCT_CHANGE = db_statements.INSERT_N_YEAR_PCT_CHANGE \
            .format(target_table, target_col,
                    source_col, source_col,
                    source_col, source_col,
                    source_col, source_col,
                    source_col, source_col, source_col,
                    source_table, source_table,
                    lag)

        logger.info("Running:")
        logger.debug(INSERT_N_YEAR_PCT_CHANGE)
        try:
            self.db.write([
                DROP_COLUMN,
                ADD_COLUMN,
                INSERT_N_YEAR_PCT_CHANGE
            ])
        except Exception as e:
            logger.error(e)
            return False

        logger.info("Added {} year pct change to {} for feature {}".format(
            lag, target_table, source_col))
        return True

    def create_geo_features_table(self):
        """Create geographic features table"""

        self.db.write([
            db_statements.DROP_TABLE_URBAN,
            db_statements.CREATE_TABLE_URBAN])

        logger.info("Create urban table")

        df = pd.read_csv(
            '/Users/alenastern/Documents/Spring2018/Machine_Learning/evictions-learn/src/data/Urban_County_2010.csv', header=0)
        df = df[['UA', 'STATE', 'COUNTY', 'GEOID']]
        df.to_csv('/Users/alenastern/Documents/Spring2018/Machine_Learning/evictions-learn/src/data/Urban_County_2010_sub.csv', index=False)

        self.db.copy('/Users/alenastern/Documents/Spring2018/Machine_Learning/evictions-learn/src/data/Urban_County_2010_sub.csv',
                     db_statements.COPY_CSV_URBAN)

        logger.info("Creating geo table...")
        self.db.write([
            db_statements.DROP_TABLE_GEOGRAPHIC,
            db_statements.CREATE_TABLE_GEOGRAPHIC])

        logger.info("Updating geo table...")
        self.db.write([
            db_statements.ALTER_TABLE_GEOGRAPHIC])

        logger.info("Geo table created.")

        logger.info("Creating indexes...")
        self.db.write([
            db_statements.IDX_COUNTY_GEO,
            db_statements.IDX_STATE_GEO])

        logger.info("Updating table...")

        self.db.write([
            db_statements.UPDATE_VAR_DIV_NE,
            db_statements.UPDATE_VAR_DIV_MA,
            db_statements.UPDATE_VAR_DIV_ENC,
            db_statements.UPDATE_VAR_DIV_WNC,
            db_statements.UPDATE_VAR_DIV_SA,
            db_statements.UPDATE_VAR_DIV_ESC,
            db_statements.UPDATE_VAR_DIV_WSC,
            db_statements.UPDATE_VAR_DIV_MNT,
            db_statements.UPDATE_VAR_DIV_PAC])

        logger.info("updating urban")
        self.db.write([db_statements.UPDATE_VAR_URBAN
                       ])

        logger.info("Regional dummies and urban updated.")

    def create_outcome_table(self, start_year, end_year):
        """Create outcomes table with different outcomes measures"""

        DROP_TABLE_OUTCOME = db_statements.DROP_TABLE_OUTCOME
        CREATE_TABLE_OUTCOME = db_statements.CREATE_TABLE_OUTCOME

        write_list = []

        for year in range(start_year, end_year):
            INSERT_OUTCOMES = db_statements.INSERT_OUTCOMES.format(year, year)
            write_list.append(INSERT_OUTCOMES)

        logger.debug(INSERT_OUTCOMES)
        try:
            self.db.write(write_list)
        except Exception as e:
            logger.error(e)
            return False

        return True

    def update_outcome_change_cat(self, col_name, col_type, existing_col, zero_to_one=True):
        """Update outcomes table to add category change variables"""

        DROP_COLUMN = db.statements.DROP_COLUMN.format('outcome', col_name)
        ADD_COLUMN = db_statements.ADD_COLUMN.format(
            'outcome', col_name, col_type)

        if zero_to_one:
            OUTCOME_CAT_CHANGE = db_statements.OUTCOME_CAT_CHANGE_0_1.format(
                col_name, existing_col, existing_col)
        else:
            OUTCOME_CAT_CHANGE = db_statements.OUTCOME_CAT_CHANGE_1_0.format(
                col_name, existing_col, existing_col)

        logger.debug(OUTCOME_CAT_CHANGE)
        try:
            self.db.write([DROP_COLUMN, ADD_COLUMN, OUTCOME_CAT_CHANGE])
        except Exception as e:
            logger.error(e)
            return False

        return True

    # TODO RENAME
    def create_ntile_discretization(self, source_col, target_table, col_type, num_buckets=4):
        """discretize a given column into a given number of buckets"""
        target_col = '{}_{}tiles'.format(source_col, num_buckets)
        DROP_COLUMN = db_statements.DROP_COLUMN.format(target_table, target_col)
        ADD_COLUMN = db_statements.ADD_COLUMN.format(
            target_table, target_col, col_type)
        INSERT_NTILE_DISCRETIZATION = db_statements.INSERT_NTILE_DISCRETIZATION.format(
            target_table, target_col, num_buckets, source_col, target_col)

        logger.debug(INSERT_NTILE_DISCRETIZATION)
        try:
            self.db.write([
                DROP_COLUMN,
                ADD_COLUMN,
                INSERT_NTILE_DISCRETIZATION
            ])
        except Exception as e:
            logger.error(e)
            return False

        return True

    def permit_import(self):
        """Create permits table and import permit data from csv"""
        self.db.write([db_statements.DROP_TABLE_PERMITS,
                       db_statements.CREATE_TABLE_PERMITS])
        self.db.copy('/Users/alenastern/Documents/Spring2018/Machine_Learning/evictions-learn/src/data/permits.csv',
                     db_statements.COPY_CSV_PERMITS)

        return True

    def hhsize_import(self):
        """Create household size table and import household size data from csv"""
        self.db.write([db_statements.DROP_TABLE_HHSIZE,
                       db_statements.CREATE_TABLE_HHSIZE])
        self.db.copy('/Users/alenastern/Documents/Spring2018/Machine_Learning/evictions-learn/src/data/census/hs_final.csv',
                     db_statements.COPY_CSV_HHSIZE)
        self.db.write([db_statements.CREATE_VAR_HHSIZE,
                       db_statements.UPDATE_VAR_HHSIZE, db_statements.DROP_TABLE_HHSIZE])
        return True

    def ev_lag_tr(self):
        """Create lagged outcome features (one year lag) for tract table"""
        self.db.write(["drop table if exists ev_lag_tr;"])
        self.db.write([db_statements.CREATE_EV_TABLE.format("tr", "tr")
                       ])
        self.db.write(["create index lag_gy on ev_lag_tr (geo_id, year);"])
        self.db.write(["create index lag_y on ev_lag_tr (year);"])
        self.db.write([db_statements.UPDATE_COLS_LAG_TR])
        self.db.write(["drop table ev_lag_tr;"])

    def ev_lag_bg(self):
        """Create lagged outcome features (one year lag) for blockgroup table"""
        self.db.write(["drop table if exists ev_lag_blockgroup;"])
        self.db.write([db_statements.CREATE_EV_TABLE.format("blockgroup", "blockgroup")
                       ])
        self.db.write(
            ["create index lag_gy_bg on ev_lag_blockgroup (geo_id, year);"])
        self.db.write(["create index lag_y_bg on ev_lag_blockgroup (year);"])
        self.db.write([db_statements.UPDATE_COLS_LAG_BG])
        self.db.write(["drop table ev_lag_blockgroup;"])

    def rem_9_ev(self, lag, col, tr):
        """Remove 999999 outlier values from evictions pct change columns, replace with max for given year"""
        logger.info(
            "removing 999999 from lagged {} for {} yr pct change where tract is {}".format(col, lag, tr))
        try:
            self.db.write([db_statements.REM_999999_ev.format(
                col, lag, tr, col, lag, tr, col, lag, tr, col, lag, tr)])
        except Exception as e:
            logger.error(e)
            return False
        logger.info(
            "removed 999999 from lagged {} for {} yr pct change".format(col, lag))
        return True

    def rem_9(self, lag, col, tr):
        """Remove 999999 outlier values from pct change columns, replace with max for given year"""
        logger.info(
            "removing 999999 from {} for {} yr pct change where tract is {}".format(col, lag, tr))
        try:
            self.db.write([db_statements.REM_999999.format(
                col, lag, tr, col, lag, tr, col, lag, tr, col, lag, tr)])
        except Exception as e:
            logger.error(e)
            return False
        logger.info(
            "removed 999999 from {} for {} yr pct change".format(col, lag))
        return True


evcols = [
    "eviction_filings",
    "evictions",
    "eviction_rate",
    "eviction_filing_rate",
    "conversion_rate"]

cols = [
    "population",
    "poverty_rate",
    "median_gross_rent",
    "median_household_income",
    "median_property_value",
    "rent_burden",
    "pct_white",
    "pct_af_am",
    "pct_hispanic",
    "pct_am_ind",
    "pct_asian",
    "pct_nh_pi",
    "pct_multiple",
    "pct_other",
    "renter_occupied_households",
    "pct_renter_occupied",
    "avg_hh_size"]

permits = [
    "total_bldg",
    "total_units",
    "total_value"]

if __name__ == "__main__":
    initializer = DBInit()

    for level in ["blockgroup", "tr"]:
        initializer.evictions_init()
    initializer.geo_init()

    for geo in ["blck_grp", "state"]:
        initializer.census_shp(geo)

    initializer.permit_import()
    initializer.hhsize_import()
    initializer.create_geo_features_table()
    initializer.create_outcome_table()
    initializer.update_outcome_change_cat()

    # Generate lagged eviction avg/pct change features for blockgroup and tract tables
    for table in ["blockgroup", "tr"]:
        lags = [1, 3, 5]
        for lag in lags:
            for col in evcols:
                res = initializer.create_n_year_average(col, table, lag)
                if not res:
                    break
                res = initializer.create_n_year_pct_change(
                    table, col, table, lag)
                if not res:
                    break

    # Generate demographic avg/pct change features for blockgroup and tract tables
    for table in ["blockgroup", "tr"]:
        lags = [5]
        for lag in lags:
            for col in cols:
                res = initializer.create_n_year_average(col, table, lag)
                if not res:
                    break
                res = initializer.create_n_year_pct_change(
                    table, col, table, lag)
                if not res:
                    break

    # Generate permits avg/pct change features
    for table in ["permits"]:
        lags = [1, 3, 5]
        for lag in lags:
            for col in permits:
                res = initializer.create_n_year_average(col, table, lag)
                if not res:
                    break
                res = initializer.create_n_year_pct_change(
                    table, col, table, lag)
                if not res:
                    break

    initializer.ev_lag_tr()
    initializer.ev_lag_bg()

    # replace instances in demographic features, which has 5 year data
    lag = 5
    for col in cols:
        for tr in [True, False]:
            res = init.rem_9(lag, col, tr)
            if not res:
                break

    # replace instances in evictions features, which has 1, 3, and 5-year data
    lags = [1, 3, 5]
    for col in evcols:
        for lag in lags:
            for tr in [True, False]:
                init.rem_9_ev(lag, col, tr)
                if not res:
                    break

    # replace instances in permit features, which has 1, 3, and 5-year data
    lags = [1, 3, 5]
    for col in permits:
        for lag in lags:
            init.rem_9_ev(lag, col, tr)
            if not res:
                break
