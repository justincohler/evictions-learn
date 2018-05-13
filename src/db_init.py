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

    def evictions_init(self):
        """Clear and initialize the evictions table(s)."""

        logger.info("Dropping tables...")
        self.db.write([db_statements.DROP_TABLE_BLOCKGROUP])

        logger.info("Creating tables...")
        self.db.write([db_statements.CREATE_TABLE_BLOCKGROUP])

        logger.info("Creating indexes...")
        self.db.write([
            db_statements.IDX_YEAR,
            db_statements.IDX_STATE,
            db_statements.IDX_STATE_YEAR,
            db_statements.IDX_GEOID,
            db_statements.IDX_GEOID_YEAR,
            db_statements.IDX_EVICTIONS
        ])

        logger.info("Tables & indexes committed.")

        logger.info("Copying CSV data to evictions db...")
        self.db.copy('C:/Users/Justin Cohler/output.csv', db_statements.COPY_CSV_BLOCKGROUP)
        logger.info("Records committed.")

    def geo_init(self):
        """Clear and initialize the geospatial table(s)."""

        self.db.write([
            db_statements.CREATE_EXT_POSTGIS,
            db_statements.CREATE_EXT_FUZZY,
            db_statements.CREATE_EXT_TIGER,
            db_statements.CREATE_EXT_POSTGIS_TOP,
            db_statements.DROP_F_EXEC,
            db_statements.CREATE_F_EXEC,
            db_statements.ALTER_SPATIAL_REF_SYS.format(self.db.DB_USER),
            db_statements.RENAME_VAR_STATE,
            db_statements.CREATE_VAR_STATE,
            db_statements.CREATE_VAR_COUNTY,
            db_statements.CREATE_VAR_TRACT,
            db_statements.UPDATE_VAR_STATE,
            db_statements.UPDATE_VAR_COUNTY,
            db_statements.UPDATE_VAR_TRACT
        ])

    def census_shp(self, geography):
        """Read shapes for a given geography."""
        DROP_TABLE_SHP = db_statements.DROP_TABLE_SHP.format(geography)
        self.db.write([DROP_TABLE_SHP])

        shp_read = "shp2pgsql -s 4269:4326 -W 'latin1' data/tl_2010_us_{}10/tl_2010_us_{}10.shp evictions.census_{}_shp | psql {} -U {} -W {} -p {} -h {}".format(geography, geography, geography,'evictions', self.db.DB_USER, self.db.DB_PASSWORD, self.db.DB_PORT, self.db.DB_HOST)
        os.system(shp_read)

    def group_by_geo(self, geo):
        """Clear and initialize the evictions_state table."""
        DROP_TABLE_EVICTIONS_GEO = db_statements.DROP_TABLE_EVICTIONS_GEO.format(geo)
        CREATE_TABLE_EVICTIONS_GEO = db_statements.CREATE_TABLE_EVICTIONS_GEO.format(geo, geo, geo)
        INSERT_EVICTIONS_GEO = db_statements.INSERT_EVICTIONS_GEO.format(geo, geo, geo, geo, geo, geo, geo)

        self.db.write([DROP_TABLE_EVICTIONS_GEO,
        	CREATE_TABLE_EVICTIONS_GEO,
        	INSERT_EVICTIONS_GEO])


    def create_n_year_average(self,source_col, target_table, lag):
        target_col = '{}_avg_{}yr'.format(source_col, lag)
        DROP_COLUMN = db_statements.DROP_COLUMN.format(target_table, target_col)
        ADD_COLUMN = db_statements.ADD_COLUMN.format(target_table, target_col, "FLOAT")
        INSERT_N_YEAR_AVG = db_statements.ALTER_N_YEAR_AVG.format(target_table, target_col, source_col, lag)

        logger.debug(INSERT_N_YEAR_AVG)
        try:
            self.db.write([
                DROP_COLUMN,
                INSERT_N_YEAR_AVG
            ])
        except Exception as e:
            logger.error(e)
            return False

        return True

    def create_n_year_pct_change(self, source_col, target_table, lag):
        target_col = '{}_pct_change_{}yr'.format(source_col, lag)
        DROP_COLUMN = db_statements.DROP_COLUMN.format(target_table, target_col)
        ADD_COLUMN = db_statements.ADD_COLUMN.format(target_table, target_col, "FLOAT")
        INSERT_N_YEAR_PCT_CHANGE = db_statements.INSERT_N_YEAR_PCT_CHANGE.format(target_table, target_col, source_col, source_col, source_col, lag, source_col, source_col)

        logger.debug(INSERT_N_YEAR_PCT_CHANGE)
        try:
            self.db.write([
                DROP_COLUMN,
                INSERT_N_YEAR_PCT_CHANGE
            ])
        except Exception as e:
            logger.error(e)
            return False

        return True

    def geo_features_table(self):
        INSERT_GEO_COLS = db_statements.INSERT_GEO_COLS.format('evictions.geography')
        self.db.write([
            db_statements.DROP_TABLE_URBAN,
            db_statements.CREATE_TABLE_URBAN])

        df = pd.read_csv('/Users/alenastern/Documents/Spring2018/Machine_Learning/evictions-learn/src/data/Urban_County_2010.csv', header = 0)
        df = df[['UA', 'STATE', 'COUNTY', 'GEOID']]
        df.to_csv('/Users/alenastern/Documents/Spring2018/Machine_Learning/evictions-learn/src/data/Urban_County_2010_sub.csv', index = False)

        self.db.copy('/Users/alenastern/Documents/Spring2018/Machine_Learning/evictions-learn/src/data/Urban_County_2010_sub.csv', db_statements.COPY_CSV_URBAN)
            
        self.db.write([
        db_statements.DROP_TABLE_GEOGRAPHIC,
        db_statements.CREATE_TABLE_GEOGRAPHIC,
        db_statements.INSERT_GEO_COLS,
        db_statements.UPDATE_VAR_DIV_NE,
        db_statements.UPDATE_VAR_DIV_MA,
        db_statements.UPDATE_VAR_DIV_ENC,
        db_statements.UPDATE_VAR_DIV_WNC,
        db_statements.UPDATE_VAR_DIV_SA,
        db_statements.UPDATE_VAR_DIV_ESC,
        db_statements.UPDATE_VAR_DIV_WSC,
        db_statements.UPDATE_VAR_DIV_MNT,
        db_statements.UPDATE_VAR_DIV_PAC,
        db_statements.UPDATE_VAR_URBAN
        ])
        
    def avg_bordering_block_groups(self):
        var_list = ['evictions', 'evict_rate', 'population', 'poverty_rate', 'pct_renter_occupied', 'median_gross_rent',
        'median_household_income', 'median_property_value', 'rent_burden', 'pct_white', 'pct_af_am', 'pct_hispanic', 'pct_am_ind',
        'pct_asian', 'pct_nh_pi', 'pct_multiple', 'pct_other', 'renter_occupied_households']

        try:
            self.db.write(db_statements.CREATE_TMP_AVG_BBG)
        except Exception as e:
                logger.error(e)
                return False

        for var in var_list:
            UPDATE_GEOGRAPHIC_BBG = db_statemetns.UPDATE_GEOGRAPHIC_BBG.format(var, var)
            try:
                self.db.write(UPDATE_GEOGRAPHIC_BBG)
            except Exception as e:
                logger.error(e)
                return False

            return True

   

    def create_outcome_table(self,start_year, end_year):
        DROP_TABLE_OUTCOME = db_statements.DROP_TABLE_OUTCOME
        CREATE_TABLE_OUTCOME = db_statements.CREATE_TABLE_OUTCOME

        write_list = [DROP_TABLE_OUTCOME, CREATE_TABLE_OUTCOME]

        for year in range(start_year, end_year):
            INSERT_OUTCOMES = db_statements.INSERT_OUTCOMES.format(year,year)
            #INSERT_CONVERSION = db_statements.INSERT_CONVERSION.format(year)
            write_list.append(INSERT_OUTCOMES)
            #write_list.append(INSERT_CONVERSION)

        logger.debug(INSERT_OUTCOMES)
        #logger.debug(INSERT_CONVERSION)
        try:
            self.db.write(write_list)
        except Exception as e:
            logger.error(e)
            return False

        return True

    def create_n_year_pct_change(self, source_col, target_table, col_type, num_buckets=4):
        target_col = '{}_{}tiles'.format(source_col, num_buckets)
        DROP_COLUMN = db_statements.DROP_COLUMN.format(target_table, target_col)
        ADD_COLUMN = db_statements.ADD_COLUMN.format(target_table, target_col, col_type)
        INSERT_NTILE_DISCRETIZATION = db_statements.INSERT_NTILE_DISCRETIZATION.format(target_table, target_col, num_buckets, source_col, target_col)

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

if __name__=="__main__":
    initializer = DBInit()
    #initializer.evictions_init()
    #initializer.geo_init()
    # "state", "county",
    #for geog in ["state", "county", "tract"]:
    #    initializer.group_by_geo(geog)

    #for geo in ["blck_grp"]:
    #	initializer.census_shp(geo)
    	#if geo != "blck_grp":
    	#	initializer.group_by_geo(geo)

    initializer.geo_features_table()

    #initializer.create_n_year_average("rent_burden", "demographic", 3)
