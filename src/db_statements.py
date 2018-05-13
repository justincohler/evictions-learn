"""DB Statements for the evictions-learn project."""

'''============================================================================
    SCHEMA
============================================================================'''
SET_SCHEMA = "set schema 'evictions';"

'''============================================================================
GEO INIT
============================================================================'''

CREATE_EXT_POSTGIS = "CREATE EXTENSION IF NOT EXISTS postgis;"
CREATE_EXT_FUZZY = "create extension IF NOT EXISTS fuzzystrmatch;"
CREATE_EXT_TIGER = "create extension IF NOT EXISTS postgis_tiger_geocoder;"
CREATE_EXT_POSTGIS_TOP = "create extension IF NOT EXISTS postgis_topology;"

DROP_F_EXEC = "drop function if exists exec(text);"
CREATE_F_EXEC = "CREATE FUNCTION exec(text) returns text language plpgsql volatile AS $f$ BEGIN EXECUTE $1; RETURN $1; END; $f$;"
ALTER_SPATIAL_REF_SYS = "ALTER TABLE spatial_ref_sys OWNER TO {};"

'''============================================================================
    BLOCKGROUP TABLE
============================================================================='''
DROP_TABLE_BLOCKGROUP = "DROP TABLE IF EXISTS blockgroup;"

CREATE_TABLE_BLOCKGROUP = """CREATE TABLE blockgroup
(
    _id SERIAL PRIMARY KEY,
    state CHAR(2),
    geo_id CHAR(12),
    year SMALLINT,
    name VARCHAR(10),
    parent_location VARCHAR(100),
    population FLOAT,
    poverty_rate FLOAT,
    pct_renter_occupied FLOAT,
    median_gross_rent FLOAT,
    median_household_income FLOAT,
    median_property_value FLOAT,
    rent_burden FLOAT,
    pct_white FLOAT,
    pct_af_am FLOAT,
    pct_hispanic FLOAT,
    pct_am_ind FLOAT,
    pct_asian FLOAT,
    pct_nh_pi FLOAT,
    pct_multiple FLOAT,
    pct_other FLOAT,
    renter_occupied_households FLOAT,
    eviction_filings FLOAT,
    evictions FLOAT,
    eviction_rate FLOAT,
    eviction_filing_rate FLOAT,
    imputed BOOLEAN,
    subbed BOOLEAN
);"""

IDX_STATE_YEAR = "CREATE INDEX idx_state_year ON blockgroup (state, year);"
IDX_YEAR = "CREATE INDEX idx_year ON blockgroup (year);"
IDX_STATE = "CREATE INDEX idx_state ON blockgroup (state);"
IDX_EVICTIONS = "CREATE INDEX idx_evictions ON blockgroup (evictions);"
IDX_GEOID = "CREATE INDEX idx_geoid on blockgroup (geo_id);"
IDX_GEOID_YEAR = "CREATE INDEX idx_geoid on blockgroup (geo_id, year);"

COPY_CSV_BLOCKGROUP = """COPY evictions.blockgroup(
    state, geo_id, year, name, parent_location,population, poverty_rate, pct_renter_occupied,
    median_gross_rent, median_household_income, median_property_value, rent_burden,
    pct_white, pct_af_am, pct_hispanic, pct_am_ind, pct_asian, pct_nh_pi, pct_multiple,
    pct_other, renter_occupied_households, eviction_filings, evictions, eviction_rate,
    eviction_filing_rate, imputed, subbed)
    FROM stdin WITH CSV HEADER DELIMITER as ','
"""
RENAME_VAR_STATE = "ALTER TABLE evictions.blockgroup RENAME COLUMN state TO state_code;"
CREATE_VAR_STATE = "ALTER TABLE evictions.blockgroup add column state CHAR(2);"
CREATE_VAR_TRACT = "ALTER TABLE evictions.blockgroup add column tract CHAR(11);"
CREATE_VAR_COUNTY = "ALTER TABLE evictions.blockgroup add column county CHAR(5);"

UPDATE_VAR_STATE = "UPDATE evictions.blockgroup set state = substring(geo_id from 1 for 2);"
UPDATE_VAR_TRACT = "UPDATE evictions.blockgroup set tract = substring(geo_id from 1 for 11);"
UPDATE_VAR_COUNTY = "UPDATE evictions.blockgroup set county = substring(geo_id from 1 for 5);"

'''==========================================================================
SHAPEFILE LOAD + GROUP BY GEO
=========================================================================='''

DROP_TABLE_EVICTIONS_GEO = "DROP TABLE IF EXISTS evictions_{};"
DROP_TABLE_SHP = "DROP TABLE IF EXISTS census_{}_shp;"

CREATE_TABLE_EVICTIONS_GEO = """CREATE TABLE evictions_{} ({} VARCHAR(12),
   year SMALLINT,
   sum_evict FLOAT,
   avg_evict_rate FLOAT,
   avg_population FLOAT,
   avg_poverty_rate FLOAT,
   avg_pct_renter_occupied FLOAT,
   avg_median_gross_rent FLOAT,
   avg_median_household_income FLOAT,
   avg_median_property_value FLOAT,
   avg_rent_burden FLOAT,
   avg_pct_white FLOAT,
   avg_pct_af_am FLOAT,
   avg_pct_hispanic FLOAT,
   avg_pct_am_ind FLOAT,
   avg_pct_asian FLOAT,
   avg_pct_nh_pi FLOAT,
   avg_pct_multiple FLOAT,
   avg_pct_other FLOAT,
   avg_renter_occupied_households FLOAT,
   PRIMARY KEY({}, year));"""

INSERT_EVICTIONS_GEO = """INSERT INTO evictions.evictions_{} ({}, year, sum_evict, avg_evict_rate, avg_population,
   avg_poverty_rate, avg_pct_renter_occupied, avg_median_gross_rent, avg_median_household_income,
   avg_median_property_value, avg_rent_burden, avg_pct_white, avg_pct_af_am, avg_pct_hispanic,
   avg_pct_am_ind, avg_pct_asian, avg_pct_nh_pi, avg_pct_multiple, avg_pct_other, avg_renter_occupied_households)
   SELECT {}, year, sum(evictions) as sum_evict, avg(eviction_rate) as avg_evict_rate, avg(population) as avg_population,
   avg(poverty_rate) as avg_poverty_rate, avg(pct_renter_occupied) as avg_pct_renter_occupied, avg(median_gross_rent) as avg_median_gross_rent, 
   avg(median_household_income) as avg_median_household_income, avg(median_property_value) as avg_median_property_value, 
   avg(rent_burden) as avg_rent_burden, avg(pct_white) as avg_pct_white, avg(pct_af_am) as avg_pct_af_am, avg(pct_hispanic) as avg_pct_hispanic,
   avg(pct_am_ind) as avg_pct_am_ind, avg(pct_asian) as avg_pct_asian, avg(pct_nh_pi) as avg_pct_nh_pi, avg(pct_multiple) as avg_pct_multiple, 
   avg(pct_other) as avg_pct_other, avg(renter_occupied_households) as avg_renter_occupied_households
   FROM evictions.blockgroup
   GROUP BY {}, year;"""


'''============================================================================
    DEMOGRAPHIC TABLE
============================================================================'''

DROP_TABLE_DEMOGRAPHIC = "DROP TABLE IF EXISTS demographic;"

CREATE_TABLE_DEMOGRAPHIC = """CREATE TABLE demographic (
    geo_id CHAR(12) NOT NULL,
    year SMALLINT NOT NULL,
    PRIMARY KEY(geo_id, year)
);"""

DROP_TABLE_OUTCOME = "DROP TABLE IF EXISTS outcome;"
DROP_COLUMN = "ALTER TABLE {} DROP COLUMN IF EXISTS {};"


'''============================================================================
GEOGRAPHIC TABLE
============================================================================'''

DROP_TABLE_URBAN = "DROP TABLE IF EXISTS urban;"
DROP_TABLE_GEOGRAPHIC = "DROP TABLE IF EXISTS geographic;"


CREATE_TABLE_URBAN = """CREATE TABLE urban (UA int, 
    STATE int,
    COUNTY int,
    GEOID int);"""

CREATE_TABLE_GEOGRAPHIC = """CREATE TABLE geographic (_id int,
state varchar(2),
geo_id varchar(12),
year int,
county varchar(5),
tract varchar(11),
urban int DEFAULT(0),
div_ne int DEFAULT(0),
div_ma int DEFAULT(0),
div_enc int DEFAULT(0),
div_wnc int DEFAULT(0),
div_sa int DEFAULT(0),
div_esc int DEFAULT(0),
div_wsc int DEFAULT(0),
div_mnt int DEFAULT(0),
div_pac int DEFAULT(0),
bbg_sum_evict FLOAT,
bbg_avg_evict_rate FLOAT,
bbg_avg_population FLOAT,
bbg_avg_poverty_rate FLOAT,
bbg_avg_pct_renter_occupied FLOAT,
bbg_avg_median_gross_rent FLOAT,
bbg_avg_median_household_income FLOAT,
bbg_avg_median_property_value FLOAT,
bbg_avg_rent_burden FLOAT,
bbg_avg_pct_white FLOAT,
bbg_avg_pct_af_am FLOAT,
bbg_avg_pct_hispanic FLOAT,
bbg_avg_pct_am_ind FLOAT,
bbg_avg_pct_asian FLOAT,
bbg_avg_pct_nh_pi FLOAT,
bbg_avg_pct_multiple FLOAT,
bbg_avg_pct_other FLOAT,
bbg_sum_evict_1 FLOAT,
bbg_avg_evict_rate_1 FLOAT,
bbg_avg_population_1 FLOAT,
bbg_avg_poverty_rate_1 FLOAT,
bbg_avg_pct_renter_occupied_1 FLOAT,
bbg_avg_median_gross_rent_1 FLOAT,
bbg_avg_median_household_income_1 FLOAT,
bbg_avg_median_property_value_1 FLOAT,
bbg_avg_rent_burden_1 FLOAT,
bbg_avg_pct_white_1 FLOAT,
bbg_avg_pct_af_am_1 FLOAT,
bbg_avg_pct_hispanic_1 FLOAT,
bbg_avg_pct_am_ind_1 FLOAT,
bbg_avg_pct_asian_1 FLOAT,
bbg_avg_pct_nh_pi_1 FLOAT,
bbg_avg_pct_multiple_1 FLOAT,
bbg_avg_pct_other_1 FLOAT,
bbg_sum_evict_3 FLOAT,
bbg_avg_evict_rate_3 FLOAT,
bbg_avg_population_3 FLOAT,
bbg_avg_poverty_rate_3 FLOAT,
bbg_avg_pct_renter_occupied_3 FLOAT,
bbg_avg_median_gross_rent_3 FLOAT,
bbg_avg_median_household_income_3 FLOAT,
bbg_avg_median_property_value_3 FLOAT,
bbg_avg_rent_burden_3 FLOAT,
bbg_avg_pct_white_3 FLOAT,
bbg_avg_pct_af_am_3 FLOAT,
bbg_avg_pct_hispanic_3 FLOAT,
bbg_avg_pct_am_ind_3 FLOAT,
bbg_avg_pct_asian_3 FLOAT,
bbg_avg_pct_nh_pi_3 FLOAT,
bbg_avg_pct_multiple_3 FLOAT,
bbg_avg_pct_other_3 FLOAT,
bbg_sum_evict_1pct FLOAT,
bbg_avg_evict_rate_1pct FLOAT,
bbg_avg_population_1pct FLOAT,
bbg_avg_poverty_rate_1pct FLOAT,
bbg_avg_pct_renter_occupied_1pct FLOAT,
bbg_avg_median_gross_rent_1pct FLOAT,
bbg_avg_median_household_income_1pct FLOAT,
bbg_avg_median_property_value_1pct FLOAT,
bbg_avg_rent_burden_1pct FLOAT,
bbg_avg_pct_white_1pct FLOAT,
bbg_avg_pct_af_am_1pct FLOAT,
bbg_avg_pct_hispanic_1pct FLOAT,
bbg_avg_pct_am_ind_1pct FLOAT,
bbg_avg_pct_asian_1pct FLOAT,
bbg_avg_pct_nh_pi_1pct FLOAT,
bbg_avg_pct_multiple_1pct FLOAT,
bbg_avg_pct_other_1pct FLOAT,
bbg_sum_evict_3pct FLOAT,
bbg_avg_evict_rate_3pct FLOAT,
bbg_avg_population_3pct FLOAT,
bbg_avg_poverty_rate_3pct FLOAT,
bbg_avg_pct_renter_occupied_3pct FLOAT,
bbg_avg_median_gross_rent_3pct FLOAT,
bbg_avg_median_household_income_3pct FLOAT,
bbg_avg_median_property_value_3pct FLOAT,
bbg_avg_rent_burden_3pct FLOAT,
bbg_avg_pct_white_3pct FLOAT,
bbg_avg_pct_af_am_3pct FLOAT,
bbg_avg_pct_hispanic_3pct FLOAT,
bbg_avg_pct_am_ind_3pct FLOAT,
bbg_avg_pct_asian_3pct FLOAT,
bbg_avg_pct_nh_pi_3pct FLOAT,
bbg_avg_pct_multiple_3pct FLOAT,
bbg_avg_pct_other_3pct FLOAT,
PRIMARY KEY(geo_id, year));"""

INSERT_GEO_COLS = """INSERT into evictions.geographic (_id, state, geo_id, year, county, tract) 
                     SELECT _id, state, geo_id, year, county, tract 
                     FROM evictions.blockgroup;
                  """

COPY_CSV_URBAN = """COPY evictions.urban(UA, STATE, COUNTY, GEOID) 
      from stdin with CSV HEADER DELIMITER as ',';"""


UPDATE_VAR_URBAN = '''update evictions.geographic set urban = 1
                      from urban t1
                      join evictions.blockgroup t2 on t1.GEOID::varchar(5) = t2.county'''

UPDATE_VAR_DIV_NE = '''UPDATE evictions.geographic set div_ne = 1 
  WHERE state = '09' OR state = '23'
  OR state = '25' OR state = '33' OR
  state = '44' OR state = '50';'''

UPDATE_VAR_DIV_MA = '''UPDATE evictions.geographic set div_ma = 1 
  WHERE state = '34' OR state = '36'
  OR state = '42';'''

UPDATE_VAR_DIV_ENC = '''UPDATE evictions.geographic set div_enc = 1 
  WHERE state = '17' OR state = '18'
  OR state = '26' OR state = '39'
  OR state = '55';'''

UPDATE_VAR_DIV_WNC = '''UPDATE evictions.geographic set div_wnc = 1 
  WHERE state = '19' OR state = '20'
  OR state = '27' OR state = '29'
  OR state = '31' OR state = '38'
  OR state = '46';'''

UPDATE_VAR_DIV_SA = '''UPDATE evictions.geographic set div_wnc = 1 
  WHERE state = '10' OR state = '11'
  OR state = '12' OR state = '13'
  OR state = '24' OR state = '37'
  OR state = '45' OR state = '51'
  OR state = '54';'''

UPDATE_VAR_DIV_MNT = '''UPDATE evictions.geographic set div_wnc = 1 
  WHERE state = '04' OR state = '08'
  OR state = '16' OR state = '30'
  OR state = '32' OR state = '35'
  OR state = '49' OR state = '56';'''

UPDATE_VAR_DIV_ESC = '''UPDATE evictions.geographic set div_mnt = 1 
  WHERE state = '01' OR state = '21'
  OR state = '28' OR state = '47';'''

UPDATE_VAR_DIV_WSC = '''UPDATE evictions.geographic set div_mnt = 1 
  WHERE state = '05' OR state = '22'
  OR state = '40' OR state = '48';'''

UPDATE_VAR_DIV_PAC = '''UPDATE evictions.geographic set div_pac = 1 
  WHERE state = '02' OR state = '06'
  OR state = '15' OR state = '41'
  OR state = '53';'''


UPDATE_GEOGRAPHIC_BBG = """update geographic set {} = tmp.{} 
                                    where tmp.geo_id = geographic.geo_id and tmp.year = geographic.year"""
                        

CREATE_TMP_AVG_BBG = """CREATE TEMPORARY TABLE tmp (
                                select b1.geo_id, b1.year, sum(b2.evictions) as bbg_sum_evict, avg(b2.evict_rate) as bbg_avg_evict_rate,
                                avg(b2.population) as bbg_avg_population, avg(b2.poverty_rate) as bbg_avg_poverty_rate, avg(b2.pct_renter_occupied) as bbg_avg_pct_renter_occupied
                                avg(b2.rent_burden) as bbg_avg_rent_burden, avg(b2.pct_white) as bbg_avg_pct_white, avg(b2.pct_af_am) as bbg_avg_pct_af_am,
                                avg(b2.pct_hispanic) as bbg_avg_pct_hispanic, avg(b2.pct_am_ind) as bbg_avg_pct_am_ind, avg(b2.pct_asian) as bbg_avg_pct_asian,
                                avg(b2.pct_nh_pi) as bbg_avg_pct_nh_pi, avg(b2.pct_multiple) as bbg_avg_pct_multiple, avg(b2.pct_other) as bbg_avg_pct_other,
                                avg(b2.renter_occupied_households) as bbg_avg_renter_occupied_households
                                from (SELECT * from blockgroup join census_blk_grp_shp on blockgroup.geo_id = census_blk_grp_shp.geoid) b1 
                                join (SELECT * from blockgroup join census_blk_grp_shp on blockgroup.geo_id = census_blk_grp_shp.geoid) b2
                                  on ST_Intersects(b1.geom, b2.geom)
                                  and b1.year = b2.year
                                  group by (b1.geo_id, b1.year));"""


'''============================================================================
 OUTCOME TABLE
============================================================================'''

CREATE_TABLE_OUTCOME = """CREATE TABLE outcome (
    geo_id CHAR(12) NOT NULL,
    year SMALLINT NOT NULL,
    top20_num SMALLINT,
    top20_rate SMALLINT
PRIMARY KEY (geo_id, year)
);"""

INSERT_OUTCOMES = """WITH tmp AS (SELECT geo_id, year, 
                            ntile(5) over(ORDER BY evictions DESC) AS num_quint, 
                            ntile(5) over(ORDER BY eviction_rate DESC) AS rate_quint
                        FROM blockgroup
                        WHERE year = {}
                        AND evictions IS NOT NULL
                        )
                    INSERT INTO outcome (geo_id, year, top20_num, top20_rate)
                        SELECT geo_id, year,
                        CASE
                            WHEN tmp.num_quint = 1 THEN 1
                            ELSE 0
                        END 
                        AS top20_num,
                        CASE
                            WHEN tmp.rate_quint = 1 THEN 1
                            ELSE 0
                        END
                        AS top20_rate
                        FROM tmp;
                """


'''============================================================================
    FUNCTIONS & EXTENSIONS
============================================================================'''

ADD_COLUMN = "ALTER TABLE {} add column {} {};"

INSERT_N_YEAR_AVG = """INSERT into {}(geo_id, year, {})
                        select b1.geo_id, b1.year, avg(b2.{})
                            from blockgroup b1 join blockgroup b2
                              on b1.geo_id=b2.geo_id
                              and b2.year between (b1.year - {}) and (b1.year - 1)
                            group by (b1.geo_id, b1.year);
                    """


INSERT_N_YEAR_AVG = """INSERT into {}(geo_id, year, {})
                        select b1.geo_id, b1.year, avg(b2.{})
                            from blockgroup b1 join blockgroup b2
                              on b1.geo_id=b2.geo_id
                              and b2.year between (b1.year - {}) and (b1.year - 1)
                            group by (b1.geo_id, b1.year);
                    """

INSERT_N_YEAR_AVG = """INSERT into {}(geo_id, year, {})
                        select b1.geo_id, b1.year, avg(b2.{})
                            from blockgroup b1 join blockgroup b2
                              on b1.geo_id=b2.geo_id
                              and b2.year between (b1.year - {}) and (b1.year - 1)
                            group by (b1.geo_id, b1.year);"""

INSERT_N_YEAR_PCT_CHANGE = """INSERT into {}(geo_id, year, {})
                            select b1.geo_id, b1.year, (b1.{} - b2.{})/b2.{}
                            from blockgroup b1 join blockgroup b2
                              on b1.geo_id=b2.geo_id
                              and b2.year = b1.year-{}
                            where b2.{} is not null and b2.{} != 0;
                            """


INSERT_NTILE_DISCRETIZATION = """INSERT into {}(geo_id, year, {})
                                SELECT geo_id, year, ntile({}) over (order by {} desc) as {}
                                FROM blockgroup;
                            """

INSERT_LAG_CONVERSION = """INSERT INTO {}(geo_id, year, {})
                            SELECT 
                            CASE 
                                WHEN b1.eviction_filings IS NOT NULL 
                                AND b1.eviction_filings != 0
                                THEN b1.evictions/b1.eviction_filings
                                ELSE 0
                            END AS conversion_rate
                            FROM blockgroup b1 join blockgroup b2
                              ON b1.geo_id=b2.geo_id
                              AND b2.year-1 = b1.year;
                        """



