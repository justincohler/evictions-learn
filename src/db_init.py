import os
import logging
from src.db_client import DBClient
import src.db_statements

logger = logging.getLogger('evictionslog')

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
            db_statements.INSERT_SPATIAL_REF_SYS,
            db_statements.RENAME_VAR_STATE,
            db_statements.CREATE_VAR_STATE,
            db_statements.CREATE_VAR_COUNTY,
            db_statements.CREATE_VAR_TRACT,
            db_statements.UPDATE_VAR_STATE,
            db_statements.UPDATE_VAR_COUNTY,
            db_statements.UPDATE_VAR_TRACT
        ])

    def census_shp(geography):
        """Read shapes for a given geography."""

        shp_read = "shp2pgsql -s 4269:4326 -W 'latin1' data/tl_2010_us_{}10/tl_2010_us_{}10.shp evictions.census_{}_shp | psql {} -U {} -W {} -p {} -h {}".format(geography, geography, geography,'evictions', self.db.DB_USER, self.db.DB_PASSWORD, self.db.DB_PORT, self.db.DB_HOST)
        os.system(shp_read)

    def group_by_geo():
        """Clear and initialize the evictions_state table."""
        DROP_TABLE_EVICTIONS_GEO = db_statements.DROP_TABLE_EVICTIONS_GEO.format(geo)
        CREATE_TABLE_EVICTIONS_GEO = db_statements.CREATE_TABLE_EVICTIONS_STATE.format(geo, geo)
        INSERT_EVICTIONS_GEO = db_statements.INSERT_EVICTIONS_STATE.format(geo, geo, geo, geo, geo, geo)

        self.db.write([DROP_TABLE_EVICTIONS_GEO,
        	CREATE_TABLE_EVICTIONS_GEO,
        	INSERT_EVICTIONS_GEO
        ])


    def create_n_year_average(source_col, target_table, lag):
    	"""Create a column and populate with the average of the last n years of data from the source.

        The default source table is "blockgroup".

    	Inputs:
    		- source_col (str): The table name from which the new column will be aggegated
    		- target_table (str): The table name to which the column will be generated
    		- lag (int): The number of years to aggregate on

    	Returns:
    		- True/False (Success/Fail)

    	"""
        ALTER_N_YEAR_AVG = db_statements.ALTER_N_YEAR_AVG
                            .format(target_table, source_col, lag, source_col, lag)

        logger.debug(ALTER_N_YEAR_AVG)
        try:
            self.db.write([
                ALTER_N_YEAR_AVG
            ])
        except Exception as e:
            logger.error(e)
            return False

        return True

    def create_n_year_pct_change(source_col, target_table, lag):
    	"""Create a column and populate with the % change from n years ago from the source.

        The default source table is "blockgroup".

    	Inputs:
    		- source_col (str): The table name from which the new column will be aggegated
    		- target_table (str): The table name to which the column will be generated
    		- lag (int): The number of years to aggregate on

    	Returns:
    		- True/False (Success/Fail)

    	"""
    	ALTER_N_YEAR_PCT_CHANGE = db_statements.ALTER_N_YEAR_PCT_CHANGE
                            .format(target_table, source_col, lag, source_col,
                            source_col, source_col, lag, source_col, source_col)

        logger.debug(ALTER_N_YEAR_PCT_CHANGE)
        try:
            self.db.write([
                ALTER_N_YEAR_PCT_CHANGE
            ])
        except Exception as e:
            logger.error(e)
            return False

        return True

if __name__=="__main__":
    initializer = DBInit()
    initializer.evictions_init()
    initializer.geo_init()
    for geo in ["state", "county", "tract", "blck_grp"]:
    	initializer.census_shp(geo)
    	if geo != "blck_grp":
    		initializer.group_by_geo(geo)
