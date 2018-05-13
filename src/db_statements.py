"""DB Statements for the evictions-learn project."""

'''============================================================================
    SCHEMA
============================================================================'''
SET_SCHEMA = "set schema 'evictions';"

'''============================================================================
    DROPS
============================================================================'''
DROP_TABLE_BLOCKGROUP = "DROP TABLE IF EXISTS blockgroup;"
DROP_TABLE_URBAN = "DROP TABLE IF EXISTS urban;"
DROP_TABLE_GEOGRAPHIC = "DROP TABLE IF EXISTS geographic;"
DROP_TABLE_EVICTIONS_GEO = "DROP TABLE IF EXISTS evictions_{};"
DROP_TABLE_SHP = "DROP TABLE IF EXISTS census_{}_shp;"
DROP_TABLE_DEMOGRAPHIC = "DROP TABLE IF EXISTS demographic;"
DROP_TABLE_OUTCOME = "DROP TABLE IF EXISTS outcome;"
DROP_COLUMN = "ALTER TABLE {} DROP COLUMN IF EXISTS {};"


'''============================================================================
    TABLES
============================================================================'''
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
    median_property_value	FLOAT,
    rent_burden	FLOAT,
    pct_white	FLOAT,
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
    imputed	BOOLEAN,
    subbed BOOLEAN
);"""

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

CREATE_TABLE_DEMOGRAPHIC = """CREATE TABLE demographic (
    geo_id CHAR(12) NOT NULL,
    year SMALLINT NOT NULL,
    PRIMARY KEY(geo_id, year)
);"""

CREATE_TABLE_URBAN = """CREATE TABLE urban (UA int, 
    UANAME text,
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
bbg_avg_pct_renter_occupied ,FLOAT
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
bbg_avg_pct_renter_occupied_1 ,FLOAT
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
bbg_avg_pct_renter_occupied_3 ,FLOAT
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
bbg_avg_pct_renter_occupied_1pct ,FLOAT
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
bbg_avg_pct_renter_occupied_3pct ,FLOAT
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
PRIMARY KEY(geo_id, year)
;)"""

CREATE_TABLE_OUTCOME = """CREATE TABLE outcome (
    geo_id CHAR(12),
    year smallint,
    top20_num int,
    top20_rate int);"""


'''============================================================================
    INDEXES
============================================================================'''
IDX_STATE_YEAR = "CREATE INDEX idx_state_year ON blockgroup (state, year);"
IDX_YEAR = "CREATE INDEX idx_year ON blockgroup (year);"
IDX_STATE = "CREATE INDEX idx_state ON blockgroup (state);"
IDX_EVICTIONS = "CREATE INDEX idx_evictions ON blockgroup (evictions);"
IDX_GEOID = "CREATE INDEX idx_geoid on blockgroup (geo_id);"
IDX_GEOID_YEAR = "CREATE INDEX idx_geoid on blockgroup (geo_id, year);"

'''============================================================================
    COPY, INSERTS, & UPDATES
============================================================================'''
COPY_CSV_BLOCKGROUP = """COPY evictions.blockgroup(
    state, geo_id, year, name, parent_location,population, poverty_rate, pct_renter_occupied,
    median_gross_rent, median_household_income, median_property_value, rent_burden,
    pct_white, pct_af_am, pct_hispanic, pct_am_ind, pct_asian, pct_nh_pi, pct_multiple,
    pct_other, renter_occupied_households, eviction_filings, evictions, eviction_rate,
    eviction_filing_rate, imputed, subbed)
    FROM stdin WITH CSV HEADER DELIMITER as ','
"""

COPY_CSV_URBAN = """COPY evictions.urban(UA, UANAME, STATE, COUNTY, GEOID) 
      from 'data/URBAN_COUNTY_2010.csv' with CSV HEADER DELIMITER as ',';"""

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



UPDATE_VAR_STATE = "UPDATE evictions.blockgroup set state = substring(geo_id from 1 for 2);"
UPDATE_VAR_TRACT = "UPDATE evictions.blockgroup set tract = substring(geo_id from 1 for 11);"
UPDATE_VAR_COUNTY = "UPDATE evictions.blockgroup set county = substring(geo_id from 1 for 5);"
UPDATE_VAR_URBAN = '''update evictions.geographic set evictions.geographoc.urban = 1
                      from urban t1
                      join evictions.blockgroup t2 on t1.GEOID = t2.county'''

INSERT_N_YEAR_AVG = """INSERT into {}(geo_id, year, {})
                        select b1.geo_id, b1.year, avg(b2.{})
                            from blockgroup b1 join blockgroup b2
                            	on b1.geo_id=b2.geo_id
                            	and b2.year between (b1.year - {}) and (b1.year - 1)
                            group by (b1.geo_id, b1.year);
                    """

INSERT_N_YEAR_PCT_CHANGE = """INSERT into {}(geo_id, year, {})
                            select b1.geo_id, b1.year, (b1.{} - b2.{})/b2.{}
                            from blockgroup b1 join blockgroup b2
                            	on b1.geo_id=b2.geo_id
                            	and b2.year = b1.year-{}
                            where b2.{} is not null and b2.{} != 0;
                            """
INSERT_GEO_COLS = """INSERT into {}(_id, state, geoid, year, county, tract) 
                     SELECT _id, state, geoid, year, county, tract 
                     FROM evictions.blockgroup;
                  """

UPDATE_VAR_DIV_NE = """UPDATE evicitions.geographic set evictions.geographic.div_ne = 1 
  WHERE evictions.geographic.state = "09" OR evictions.geographic.state = "23"
  OR evictions.geographic.state = "25" OR evictions.geographic.state = "33" OR
  evictions.geographic.state = "44" OR evictions.geographic.state = "50";"""

UPDATE_VAR_DIV_MA = """UPDATE evicitions.geographic set evictions.geographic.div_ma = 1 
  WHERE evictions.geographic.state = "34" OR evictions.geographic.state = "36"
  OR evictions.geographic.state = "42";"""

UPDATE_VAR_DIV_ENC = """UPDATE evicitions.geographic set evictions.geographic.div_enc = 1 
  WHERE evictions.geographic.state = "17" OR evictions.geographic.state = "18"
  OR evictions.geographic.state = "26" OR evictions.geographic.state = "39"
  OR evictions.geographic.state = "55";"""

UPDATE_VAR_DIV_WNC = """UPDATE evicitions.geographic set evictions.geographic.div_wnc = 1 
  WHERE evictions.geographic.state = "19" OR evictions.geographic.state = "20"
  OR evictions.geographic.state = "27" OR evictions.geographic.state = "29"
  OR evictions.geographic.state = "31" OR evictions.geographic.state = "38"
  OR evictions.geographic.state = "46";"""

UPDATE_VAR_DIV_SA = """UPDATE evicitions.geographic set evictions.geographic.div_wnc = 1 
  WHERE evictions.geographic.state = "10" OR evictions.geographic.state = "11"
  OR evictions.geographic.state = "12" OR evictions.geographic.state = "13"
  OR evictions.geographic.state = "24" OR evictions.geographic.state = "37"
  OR evictions.geographic.state = "45" OR evictions.geographic.state = "51"
  OR evictions.geographic.state = "54";"""

UPDATE_VAR_DIV_MNT = """UPDATE evicitions.geographic set evictions.geographic.div_wnc = 1 
  WHERE evictions.geographic.state = "04" OR evictions.geographic.state = "08"
  OR evictions.geographic.state = "16" OR evictions.geographic.state = "30"
  OR evictions.geographic.state = "32" OR evictions.geographic.state = "35"
  OR evictions.geographic.state = "49" OR evictions.geographic.state = "56";"""

UPDATE_VAR_DIV_ESC = """UPDATE evicitions.geographic set evictions.geographic.div_mnt = 1 
  WHERE evictions.geographic.state = "01" OR evictions.geographic.state = "21"
  OR evictions.geographic.state = "28" OR evictions.geographic.state = "47";"""

UPDATE_VAR_DIV_WSC = """UPDATE evicitions.geographic set evictions.geographic.div_mnt = 1 
  WHERE evictions.geographic.state = "05" OR evictions.geographic.state = "22"
  OR evictions.geographic.state = "40" OR evictions.geographic.state = "48";"""

UPDATE_VAR_DIV_PAC = """UPDATE evicitions.geographic set evictions.geographic.div_pac = 1 
  WHERE evictions.geographic.state = "02" OR evictions.geographic.state = "06"
  OR evictions.geographic.state = "15" OR evictions.geographic.state = "41"
  OR evictions.geographic.state = "53";"""

INSERT_NTILE_DISCRETIZATION = """INSERT into {}(geo_id, year, {})
                                SELECT geo_id, year, ntile({}) over (order by {} desc) as {}
                                FROM blockgroup;
                            """

INSERT_OUTCOMES = """WITH tmp AS (
                        select geo_id, year, 
                            ntile(5) over(order by evictions desc) as num_quint, 
                            ntile(5) over(order by eviction_rate desc) as rate_quint
                        from blockgroup
                        where year = {}
                        and evictions is not NULL)
                    insert into outcome (geo_id, year, top20_num, top20_rate)
                        select geo_id, year, 
                        case
                            when tmp.num_quint = 1 then 1
                            else 0
                        end 
                        as top20_num,
                        case
                            when tmp.rate_quint = 1 then 1
                            else 0
                        end
                        as top20_rate
                        from tmp;
                """

INSERT_N_YEAR_AVG = """INSERT into {}(geo_id, year, {})
                        select b1.geo_id, b1.year, avg(b2.{})
                            from blockgroup b1 join blockgroup b2
                              on b1.geo_id=b2.geo_id
                              and b2.year between (b1.year - {}) and (b1.year - 1)
                            group by (b1.geo_id, b1.year);
                    """

"""IDENTIFY_BORDERING_GEOM = '''INSERT INTO  (gid, holc_grade_b, holc_grade_a, geom) 
          
          SELECT b.gid, b.holc_grade as holc_a, a.holc_grade as holc_b, 
          geometry(ST_Intersection(ST_Buffer(CAST(a.geom AS geography), 402), 
          b.geom)) FROM redline a, redline b  
          WHERE a.holc_grade = '{}' AND b.holc_grade != '{}' AND 
          ST_Intersects(ST_Buffer(CAST(a.geom AS geography), 402), 
          b.geom);'''.format(grade, grade)"""

'''============================================================================
    FUNCTIONS & EXTENSIONS
============================================================================'''
CREATE_EXT_POSTGIS = "CREATE EXTENSION IF NOT EXISTS postgis;"
CREATE_EXT_FUZZY = "create extension IF NOT EXISTS fuzzystrmatch;"
CREATE_EXT_TIGER = "create extension IF NOT EXISTS postgis_tiger_geocoder;"
CREATE_EXT_POSTGIS_TOP = "create extension IF NOT EXISTS postgis_topology;"

DROP_F_EXEC = "drop function if exists exec(text);"
CREATE_F_EXEC = "CREATE FUNCTION exec(text) returns text language plpgsql volatile AS $f$ BEGIN EXECUTE $1; RETURN $1; END; $f$;"
ALTER_SPATIAL_REF_SYS = "ALTER TABLE spatial_ref_sys OWNER TO {};"
INSERT_SPATIAL_REF_SYS = """INSERT into spatial_ref_sys (srid, auth_name, auth_srid, proj4text, srtext)
                            VALUES ( 102003, 'esri', 102003, '+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs ', 'PROJCS["USA_Contiguous_Albers_Equal_Area_Conic",GEOGCS["GCS_North_American_1983",DATUM["North_American_Datum_1983",SPHEROID["GRS_1980",6378137,298.257222101]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["longitude_of_center",-96],PARAMETER["Standard_Parallel_1",29.5],PARAMETER["Standard_Parallel_2",45.5],PARAMETER["latitude_of_center",37.5],UNIT["Meter",1],AUTHORITY["EPSG","102003"]]');"""

'''============================================================================
    ALTERS
============================================================================'''
RENAME_VAR_STATE = "ALTER TABLE evictions.blockgroup RENAME COLUMN state TO state_code;"
CREATE_VAR_STATE = "ALTER TABLE evictions.blockgroup add column state CHAR(2);"
CREATE_VAR_TRACT = "ALTER TABLE evictions.blockgroup add column tract CHAR(11);"
CREATE_VAR_COUNTY = "ALTER TABLE evictions.blockgroup add column county CHAR(5);"
ADD_COLUMN = "ALTER TABLE {} add column {} {};"
