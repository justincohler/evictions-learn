"""DB Statements for the evictions-learn project."""
# East North Central: Indiana (18) Illinois (17) Michigan (26) Ohio (39) Wisconsin (55)
# South Atlantic: Delaware (10) District of Columbia (11) Florida  (12) Georgia (13)
#   Maryland  (24) North Carolina (37) South Carolina (45) Virginia (51) West Virginia  (54)

# psql -d evictions -f shp_dump.sql
# pg_dump -h evictions.cuvbjnrvbddj.us-east-1.rds.amazonaws.com -p 5432 -U ev_user -f shp_dump.sql -d evictions -v  --table=census_blck_grp_shp | psql -d evictions

#pg_dump -h evictions.cuvbjnrvbddj.us-east-1.rds.amazonaws.com -p 5432 -U ev_user -f geographic.sql -d evictions -v  --table=geographic | psql -d evictions


Drop_State = "delete from evictions_tract where state not in ('18', '17', '26', '39', '55', '10', '11', '12', '13', '24', '37', '45', '51', '54');"


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

CREATE_TABLE_EVICTIONS = """CREATE TABLE tr
(
    geo_id CHAR(12),
    year SMALLINT,
    name VARCHAR(50),
    parent_location VARCHAR(100),
    population FLOAT4,
    poverty_rate FLOAT4,
    pct_renter_occupied FLOAT4,
    median_gross_rent FLOAT4,
    median_household_income FLOAT4,
    median_property_value FLOAT4,
    rent_burden FLOAT4,
    pct_white FLOAT4,
    pct_af_am FLOAT4,
    pct_hispanic FLOAT4,
    pct_am_ind FLOAT4,
    pct_asian FLOAT4,
    pct_nh_pi FLOAT4,
    pct_multiple FLOAT4,
    pct_other FLOAT4,
    renter_occupied_households FLOAT4,
    eviction_filings FLOAT4,
    evictions FLOAT4,
    eviction_rate FLOAT4,
    eviction_filing_rate FLOAT4,
    imputed BOOLEAN,
    subbed BOOLEAN,
    low_flag BOOLEAN
);"""

IDX_STATE_YEAR = "CREATE INDEX idx_state_year ON blockgroup (state, year);"
IDX_YEAR = "CREATE INDEX idx_year ON blockgroup (year);"
IDX_STATE = "CREATE INDEX idx_state ON blockgroup (state);"
IDX_EVICTIONS = "CREATE INDEX idx_evictions ON blockgroup (evictions);"
IDX_GEOID = "CREATE INDEX idx_geoid on blockgroup (geo_id);"
IDX_GEOID_YEAR = "CREATE INDEX idx_geoid on blockgroup (geo_id, year);"
IDX_STATE_TR = "create index idx_state_tr on evictions_tract (state);"
IDX_STATE_CT = "create index idx_state_ct on evictions_county (state);"
IDX_STATE_GEO = "create index idx_state_geo on geographic (state);"

COPY_CSV_EVICTIONS = """COPY {}
    FROM stdin WITH CSV HEADER DELIMITER as ','
"""
RENAME_VAR_STATE = "ALTER TABLE evictions.blockgroup RENAME COLUMN state TO state_code;"
CREATE_VAR_STATE = "ALTER TABLE evictions.blockgroup add column state CHAR(2);"
CREATE_VAR_TRACT = "ALTER TABLE evictions.blockgroup add column tract CHAR(11);"
CREATE_VAR_COUNTY = "ALTER TABLE evictions.blockgroup add column county CHAR(5);"

UPDATE_VAR_STATE = "UPDATE evictions.blockgroup set state = substring(geo_id from 1 for 2);"
UPDATE_VAR_TRACT = "UPDATE evictions.blockgroup set tract = substring(geo_id from 1 for 11);"
UPDATE_VAR_COUNTY = "UPDATE evictions.blockgroup set county = substring(geo_id from 1 for 5);"

ALTER TABLE evictions.tr add column state CHAR(2);
UPDATE evictions.tr set state = substring(geo_id from 1 for 2);

# delete from evictions_tract where state not in ('18', '17', '26', '39',
'''==========================================================================
SHAPEFILE LOAD + GROUP BY GEO
=========================================================================='''

DROP_TABLE_EVICTIONS_GEO = "DROP TABLE IF EXISTS evictions_{};"
DROP_TABLE_SHP = "DROP TABLE IF EXISTS census_{}_shp;"

IDX_GEOID = "CREATE INDEX idx_geoid on blockgroup (geo_id);"

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
   SELECT {}, year, sum(evictions *b2.population)/sum(b2.population) as sum_evict, avg(eviction_rate *b2.population)/sum(b2.population) as avg_evict_rate, avg(population *b2.population)/sum(b2.population) as avg_population,
   avg(poverty_rate *b2.population)/sum(b2.population) as avg_poverty_rate, avg(pct_renter_occupied *b2.population)/sum(b2.population) as avg_pct_renter_occupied, avg(median_gross_rent *b2.population)/sum(b2.population) as avg_median_gross_rent,
   avg(median_household_income *b2.population)/sum(b2.population) as avg_median_household_income, avg(median_property_value *b2.population)/sum(b2.population) as avg_median_property_value,
   avg(rent_burden *b2.population)/sum(b2.population) as avg_rent_burden, avg(pct_white *b2.population)/sum(b2.population) as avg_pct_white, avg(pct_af_am *b2.population)/sum(b2.population) as avg_pct_af_am, avg(pct_hispanic *b2.population)/sum(b2.population) as avg_pct_hispanic,
   avg(pct_am_ind *b2.population)/sum(b2.population) as avg_pct_am_ind, avg(pct_asian *b2.population)/sum(b2.population) as avg_pct_asian, avg(pct_nh_pi *b2.population)/sum(b2.population) as avg_pct_nh_pi, avg(pct_multiple *b2.population)/sum(b2.population) as avg_pct_multiple,
   avg(pct_other *b2.population)/sum(b2.population) as avg_pct_other, avg(renter_occupied_households *b2.population)/sum(b2.population) as avg_renter_occupied_households
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
DROP_TABLE = "DROP TABLE IF EXISTS {};"
DROP_TABLE_GEOGRAPHIC = "DROP TABLE IF EXISTS geographic;"


CREATE_TABLE_URBAN = """CREATE TABLE urban (UA int,
    STATE int,
    COUNTY int,
    GEOID int);"""


CREATE_TABLE_GEOGRAPHIC = """CREATE TABLE geographic as SELECT _id, state, geo_id, year, county, tract FROM evictions.blockgroup;"""

ALTER_TABLE_GEOGRAPHIC = """ALTER TABLE geographic
ADD COLUMN urban int DEFAULT(0),
ADD COLUMN div_ne int DEFAULT(0),
ADD COLUMN div_ma int DEFAULT(0),
ADD COLUMN div_enc int DEFAULT(0),
ADD COLUMN div_wnc int DEFAULT(0),
ADD COLUMN div_sa int DEFAULT(0),
ADD COLUMN div_esc int DEFAULT(0),
ADD COLUMN div_wsc int DEFAULT(0),
ADD COLUMN div_mnt int DEFAULT(0),
ADD COLUMN div_pac int DEFAULT(0),
ADD COLUMN bbg_sum_evict FLOAT,
ADD COLUMN bbg_avg_evict_rate FLOAT,
ADD COLUMN bbg_avg_population FLOAT,
ADD COLUMN bbg_avg_poverty_rate FLOAT,
ADD COLUMN bbg_avg_pct_renter_occupied FLOAT,
ADD COLUMN bbg_avg_median_gross_rent FLOAT,
ADD COLUMN bbg_avg_median_household_income FLOAT,
ADD COLUMN bbg_avg_median_property_value FLOAT,
ADD COLUMN bbg_avg_rent_burden FLOAT,
ADD COLUMN bbg_avg_pct_white FLOAT,
ADD COLUMN bbg_avg_pct_af_am FLOAT,
ADD COLUMN bbg_avg_pct_hispanic FLOAT,
ADD COLUMN bbg_avg_pct_am_ind FLOAT,
ADD COLUMN bbg_avg_pct_asian FLOAT,
ADD COLUMN bbg_avg_pct_nh_pi FLOAT,
ADD COLUMN bbg_avg_pct_multiple FLOAT,
ADD COLUMN bbg_avg_pct_other FLOAT,
ADD COLUMN bbg_sum_evict_1 FLOAT,
ADD COLUMN bbg_avg_evict_rate_1 FLOAT,
ADD COLUMN bbg_avg_population_1 FLOAT,
ADD COLUMN bbg_avg_poverty_rate_1 FLOAT,
ADD COLUMN bbg_avg_pct_renter_occupied_1 FLOAT,
ADD COLUMN bbg_avg_median_gross_rent_1 FLOAT,
ADD COLUMN bbg_avg_median_household_income_1 FLOAT,
ADD COLUMN bbg_avg_median_property_value_1 FLOAT,
ADD COLUMN bbg_avg_rent_burden_1 FLOAT,
ADD COLUMN bbg_avg_pct_white_1 FLOAT,
ADD COLUMN bbg_avg_pct_af_am_1 FLOAT,
ADD COLUMN bbg_avg_pct_hispanic_1 FLOAT,
ADD COLUMN bbg_avg_pct_am_ind_1 FLOAT,
ADD COLUMN bbg_avg_pct_asian_1 FLOAT,
ADD COLUMN bbg_avg_pct_nh_pi_1 FLOAT,
ADD COLUMN bbg_avg_pct_multiple_1 FLOAT,
ADD COLUMN bbg_avg_pct_other_1 FLOAT,
ADD COLUMN bbg_sum_evict_3 FLOAT,
ADD COLUMN bbg_avg_evict_rate_3 FLOAT,
ADD COLUMN bbg_avg_population_3 FLOAT,
ADD COLUMN bbg_avg_poverty_rate_3 FLOAT,
ADD COLUMN bbg_avg_pct_renter_occupied_3 FLOAT,
ADD COLUMN bbg_avg_median_gross_rent_3 FLOAT,
ADD COLUMN bbg_avg_median_household_income_3 FLOAT,
ADD COLUMN bbg_avg_median_property_value_3 FLOAT,
ADD COLUMN bbg_avg_rent_burden_3 FLOAT,
ADD COLUMN bbg_avg_pct_white_3 FLOAT,
ADD COLUMN bbg_avg_pct_af_am_3 FLOAT,
ADD COLUMN bbg_avg_pct_hispanic_3 FLOAT,
ADD COLUMN bbg_avg_pct_am_ind_3 FLOAT,
ADD COLUMN bbg_avg_pct_asian_3 FLOAT,
ADD COLUMN bbg_avg_pct_nh_pi_3 FLOAT,
ADD COLUMN bbg_avg_pct_multiple_3 FLOAT,
ADD COLUMN bbg_avg_pct_other_3 FLOAT,
ADD COLUMN bbg_sum_evict_1pct FLOAT,
ADD COLUMN bbg_avg_evict_rate_1pct FLOAT,
ADD COLUMN bbg_avg_population_1pct FLOAT,
ADD COLUMN bbg_avg_poverty_rate_1pct FLOAT,
ADD COLUMN bbg_avg_pct_renter_occupied_1pct FLOAT,
ADD COLUMN bbg_avg_median_gross_rent_1pct FLOAT,
ADD COLUMN bbg_avg_median_household_income_1pct FLOAT,
ADD COLUMN bbg_avg_median_property_value_1pct FLOAT,
ADD COLUMN bbg_avg_rent_burden_1pct FLOAT,
ADD COLUMN bbg_avg_pct_white_1pct FLOAT,
ADD COLUMN bbg_avg_pct_af_am_1pct FLOAT,
ADD COLUMN bbg_avg_pct_hispanic_1pct FLOAT,
ADD COLUMN bbg_avg_pct_am_ind_1pct FLOAT,
ADD COLUMN bbg_avg_pct_asian_1pct FLOAT,
ADD COLUMN bbg_avg_pct_nh_pi_1pct FLOAT,
ADD COLUMN bbg_avg_pct_multiple_1pct FLOAT,
ADD COLUMN bbg_avg_pct_other_1pct FLOAT,
ADD COLUMN bbg_sum_evict_3pct FLOAT,
ADD COLUMN bbg_avg_evict_rate_3pct FLOAT,
ADD COLUMN bbg_avg_population_3pct FLOAT,
ADD COLUMN bbg_avg_poverty_rate_3pct FLOAT,
ADD COLUMN bbg_avg_pct_renter_occupied_3pct FLOAT,
ADD COLUMN bbg_avg_median_gross_rent_3pct FLOAT,
ADD COLUMN bbg_avg_median_household_income_3pct FLOAT,
ADD COLUMN bbg_avg_median_property_value_3pct FLOAT,
ADD COLUMN bbg_avg_rent_burden_3pct FLOAT,
ADD COLUMN bbg_avg_pct_white_3pct FLOAT,
ADD COLUMN bbg_avg_pct_af_am_3pct FLOAT,
ADD COLUMN bbg_avg_pct_hispanic_3pct FLOAT,
ADD COLUMN bbg_avg_pct_am_ind_3pct FLOAT,
ADD COLUMN bbg_avg_pct_asian_3pct FLOAT,
ADD COLUMN bbg_avg_pct_nh_pi_3pct FLOAT,
ADD COLUMN bbg_avg_pct_multiple_3pct FLOAT,
ADD COLUMN bbg_avg_pct_other_3pct FLOAT,
ADD PRIMARY KEY(geo_id, year);"""


COPY_CSV_URBAN = """COPY evictions.urban(UA, STATE, COUNTY, GEOID)
      from stdin with CSV HEADER DELIMITER as ',';"""


IDX_COUNTY_GEO = "CREATE INDEX idx_county_geo ON geographic (county);"
IDX_STATE_GEO = "CREATE INDEX idx_state_geo ON geographic (state);"


UPDATE_VAR_URBAN = '''UPDATE evictions.geographic
                      SET urban = 1
                      WHERE county IN (SELECT GEOID::varchar(5) FROM evictions.urban);'''

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


UPDATE_VAR_DIV_SA = '''UPDATE evictions.geographic set div_sa = 1
  WHERE state = '10' OR state = '11'
  OR state = '12' OR state = '13'
  OR state = '24' OR state = '37'
  OR state = '45' OR state = '51'
  OR state = '54';'''


UPDATE_VAR_DIV_MNT = '''UPDATE evictions.geographic set div_mnt = 1
  WHERE state = '04' OR state = '08'
  OR state = '16' OR state = '30'
  OR state = '32' OR state = '35'
  OR state = '49' OR state = '56';'''

UPDATE_VAR_DIV_ESC = '''UPDATE evictions.geographic set div_esc = 1
  WHERE state = '01' OR state = '21'
  OR state = '28' OR state = '47';'''

UPDATE_VAR_DIV_WSC = '''UPDATE evictions.geographic set div_wsc = 1
  WHERE state = '05' OR state = '22'
  OR state = '40' OR state = '48';'''

UPDATE_VAR_DIV_PAC = '''UPDATE evictions.geographic set div_pac = 1
  WHERE state = '02' OR state = '06'
  OR state = '15' OR state = '41'
  OR state = '53';'''


ADD_FILINGS = "ALTER TABLE evictions.geographic add column bbg_avg_eviction_filings float8;"
ADD_FILING_RATE = "ALTER TABLE evictions.geographic add column bbg_avg_filing_rate float8;"
ADD_HHSIZE = "ALTER TABLE evictions.geographic add column bbg_avg_hh_size float8;"
ADD_HHSIZE = "ALTER TABLE evictions.geographic add column bbg_avg_conversion_rate float8;"
ADD_RENT_OCC_HH = "ALTER TABLE evictions.geographic add column bbg_avg_renter_occupied_households float8;"


#UPDATE_GEOGRAPHIC_BBG = """update geographic set {} = tmp.{}
                                    #where tmp.geo_id = geographic.geo_id and tmp.year = geographic.year"""


CREATE_IDX_Shape = "CREATE INDEX idx_geoid10 on census_blck_grp_shp (geoid10);"

CREATE_TMP = """WITH tmp_bbg AS (SELECT * from evictions.blockgroup join evictions.census_blck_grp_shp on blockgroup.geo_id = census_blck_grp_shp.geoid10::varchar(12)
                )
                CREATE TABLE evictions_BBG as (
                    select b1.geo_id, b1.year, sum(b2.evictions *b2.population)/sum(b2.population) as bbg_sum_evict, avg(b2.eviction_rate *b2.population)/sum(b2.population) as bbg_avg_evict_rate,
                        avg(b2.population *b2.population)/sum(b2.population) as bbg_avg_population, avg(b2.poverty_rate *b2.population)/sum(b2.population) as bbg_avg_poverty_rate, avg(b2.pct_renter_occupied *b2.population)/sum(b2.population) as bbg_avg_pct_renter_occupied,
                        avg(b2.rent_burden *b2.population)/sum(b2.population) as bbg_avg_rent_burden, avg(b2.pct_white *b2.population)/sum(b2.population) as bbg_avg_pct_white, avg(b2.pct_af_am *b2.population)/sum(b2.population) as bbg_avg_pct_af_am,
                        avg(b2.pct_hispanic *b2.population)/sum(b2.population) as bbg_avg_pct_hispanic, avg(b2.pct_am_ind *b2.population)/sum(b2.population) as bbg_avg_pct_am_ind, avg(b2.pct_asian *b2.population)/sum(b2.population) as bbg_avg_pct_asian,
                        avg(b2.pct_nh_pi *b2.population)/sum(b2.population) as bbg_avg_pct_nh_pi, avg(b2.pct_multiple *b2.population)/sum(b2.population) as bbg_avg_pct_multiple, avg(b2.pct_other *b2.population)/sum(b2.population) as bbg_avg_pct_other,
                        avg(b2.renter_occupied_households *b2.population)/sum(b2.population) as bbg_avg_renter_occupied_households, avg(b2.avg_hh_size *b2.population)/sum(b2.population) as bbg_avg_hh_size
                        from tmp_bbg b1
                        join tmp_bbg b2
                        on ST_Intersects(b1.geom, b2.geom)
                        and b1.year = b2.year
                        group by b1.geo_id, b1.year);
                    """
CREATE_BBG = """WITH tmp_bbg AS (SELECT * from evictions.blockgroup join evictions.census_blck_grp_shp on blockgroup.geo_id = census_blck_grp_shp.geoid10::varchar(12)
                )
insert into evictions.geographic (bbg_sum_evict, bbg_avg_evict_rate, bbg_avg_population, bbg_avg_poverty_rate, bbg_avg_pct_renter_occupied, bbg_avg_median_gross_rent,
  bbg_avg_median_household_income, bbg_avg_median_property_value, bbg_avg_rent_burden, bbg_avg_pct_white, bbg_avg_pct_af_am, bbg_avg_pct_hispanic, bbg_avg_pct_am_ind,
  bbg_avg_pct_nh_pi, bbg_avg_pct_other, bbg_avg_renter_occupied_households, bbg_avg_eviction_filings, bbg_avg_filing_rate, bbg_avg_conversion_rate, bbg_avg_hh_size)
                    select sum(b2.evictions *b2.population)/sum(b2.population) as bbg_sum_evict, avg(b2.eviction_rate *b2.population)/sum(b2.population) as bbg_avg_evict_rate,
                        avg(b2.population *b2.population)/sum(b2.population) as bbg_avg_population, avg(b2.poverty_rate *b2.population)/sum(b2.population) as bbg_avg_poverty_rate, avg(b2.pct_renter_occupied *b2.population)/sum(b2.population) as bbg_avg_pct_renter_occupied,
                        avg(b2.rent_burden *b2.population)/sum(b2.population) as bbg_avg_rent_burden, avg(b2.pct_white *b2.population)/sum(b2.population) as bbg_avg_pct_white, avg(b2.pct_af_am *b2.population)/sum(b2.population) as bbg_avg_pct_af_am,
                        avg(b2.pct_hispanic *b2.population)/sum(b2.population) as bbg_avg_pct_hispanic, avg(b2.pct_am_ind *b2.population)/sum(b2.population) as bbg_avg_pct_am_ind, avg(b2.pct_asian *b2.population)/sum(b2.population) as bbg_avg_pct_asian,
                        avg(b2.pct_nh_pi *b2.population)/sum(b2.population) as bbg_avg_pct_nh_pi, avg(b2.pct_multiple *b2.population)/sum(b2.population) as bbg_avg_pct_multiple, avg(b2.pct_other *b2.population)/sum(b2.population) as bbg_avg_pct_other,
                        avg(b2.renter_occupied_households *b2.population)/sum(b2.population) as bbg_avg_renter_occupied_households, avg(b2.eviction_filings *b2.population)/sum(b2.population) as bbg_avg_eviction_filings,
                        avg(b2.eviction_filing_rate *b2.population)/sum(b2.population) as bbg_avg_filing_rate, avg(b2.conversion_rate *b2.population)/sum(b2.population) as bbg_avg_conversion_rate, avg(b2.avg_hh_size *b2.population)/sum(b2.population) as bbg_avg_hh_size

                    select sum(b2.evictions) as bbg_sum_evict, avg(b2.eviction_rate) as bbg_avg_evict_rate,
                        avg(b2.population) as bbg_avg_population, avg(b2.poverty_rate) as bbg_avg_poverty_rate, avg(b2.pct_renter_occupied) as bbg_avg_pct_renter_occupied,
                        avg(b2.rent_burden) as bbg_avg_rent_burden, avg(b2.pct_white) as bbg_avg_pct_white, avg(b2.pct_af_am) as bbg_avg_pct_af_am,
                        avg(b2.pct_hispanic) as bbg_avg_pct_hispanic, avg(b2.pct_am_ind) as bbg_avg_pct_am_ind, avg(b2.pct_asian) as bbg_avg_pct_asian,
                        avg(b2.pct_nh_pi) as bbg_avg_pct_nh_pi, avg(b2.pct_multiple) as bbg_avg_pct_multiple, avg(b2.pct_other) as bbg_avg_pct_other,
                        avg(b2.renter_occupied_households) as bbg_avg_renter_occupied_households, avg(b2.eviction_filings) as bbg_avg_eviction_filings,
                        avg(b2.eviction_filing_rate) as bbg_avg_filing_rate, avg(b2.conversion_rate) as bbg_avg_conversion_rate, avg(b2.avg_hh_size) as bbg_avg_hh_size
                        from tmp_bbg b1
                        join tmp_bbg b2
                        on ST_Intersects(b1.geom, b2.geom)
                        and b1.year = b2.year
                        group by b1.geo_id, b1.year
                        where geographic.geo_id = b1.geo_id and geographic.year = b1.year
                        ;
"""

BBG_UPDATE = """
UPDATE evictions.geographic SET
bbg_sum_evict = t.bbg_sum_evict,
bbg_avg_evict_rate = t.bbg_avg_evict_rate,
bbg_avg_population = t.bbg_avg_population,
bbg_avg_poverty_rate = t.bbg_avg_poverty_rate,
bbg_avg_pct_renter_occupied = t.bbg_avg_pct_renter_occupied,
bbg_avg_median_gross_rent = t.bbg_avg_median_gross_rent,
bbg_avg_median_household_income = t.bbg_avg_median_household_income,
bbg_avg_median_property_value = t.bbg_avg_median_property_value,
bbg_avg_rent_burden= t.bbg_avg_rent_burden,
bbg_avg_pct_white= t.bbg_avg_pct_white,
bbg_avg_pct_af_am= t.bbg_avg_pct_af_am,
bbg_avg_pct_hispanic = t.bbg_avg_pct_hispanic,
bbg_avg_pct_am_ind = t.bbg_avg_pct_am_ind,
bbg_avg_pct_nh_pi = t.bbg_avg_pct_nh_pi,
bbg_avg_pct_asian = t.bbg_avg_pct_asian
bbg_avg_pct_multiple = t.bbg_avg_pct_multiple,
bbg_avg_pct_other = t.bbg_avg_pct_other,
bbg_avg_renter_occupied_households = t.bbg_avg_renter_occupied_households,
bbg_avg_eviction_filings = t.bbg_avg_eviction_filings,
bbg_avg_filing_rate = t.bbg_avg_filing_rate,
bbg_avg_conversion_rate = t.bbg_avg_conversion_rate,
bbg_avg_hh_size = t.bbg_avg_hh_size
FROM bbg t
where geographic.geo_id = t.gid and geographic.year = t.yr
;
"""
CREATE_TMP_BBG = '''CREATE TABLE tmp_bbg AS (
                    SELECT blockgroup.geo_id, blockgroup.year, evictions, eviction_rate,
                    population, poverty_rate, pct_renter_occupied, median_gross_rent,
                    median_household_income, median_property_value, rent_burden, pct_white,
                    pct_af_am, pct_hispanic, pct_am_ind, pct_asian, pct_nh_pi, pct_multiple, pct_other,
                    renter_occupied_households, eviction_filings, eviction_filing_rate, conversion_rate,
                    avg_hh_size, census_blck_grp_shp.geom
                    from evictions.blockgroup join evictions.census_blck_grp_shp
                    on blockgroup.geo_id = census_blck_grp_shp.geoid10
                );'''

CREATE_IDX_TMP_GIDYR = "CREATE INDEX idx_gid_yr on tmp_bbg (geo_id, year);"


CREATE_TABLE_AS_BBG = """CREATE TABLE bbg as (SELECT b1.geo_id as gid, b1.year as yr,
  sum(b2.evictions *b2.population)/sum(b2.population) as bbg_sum_evict,
  sum(b2.eviction_rate *b2.population)/sum(b2.population) as bbg_avg_evict_rate,
  sum(b2.population *b2.population)/sum(b2.population) as bbg_avg_population,
  sum(b2.poverty_rate *b2.population)/sum(b2.population) as bbg_avg_poverty_rate,
  sum(b2.pct_renter_occupied *b2.population)/sum(b2.population) as bbg_avg_pct_renter_occupied,
  sum(b2.median_gross_rent *b2.population)/sum(b2.population) as bbg_avg_median_gross_rent,
  sum(b2.median_household_income *b2.population)/sum(b2.population) as bbg_avg_median_household_income,
  sum(b2.median_property_value *b2.population)/sum(b2.population) as bbg_avg_median_property_value,
  sum(b2.rent_burden *b2.population)/sum(b2.population) as bbg_avg_rent_burden,
  sum(b2.pct_white *b2.population)/sum(b2.population) as bbg_avg_pct_white,
  sum(b2.pct_af_am *b2.population)/sum(b2.population) as bbg_avg_pct_af_am,
  sum(b2.pct_hispanic *b2.population)/sum(b2.population) as bbg_avg_pct_hispanic,
  sum(b2.pct_am_ind *b2.population)/sum(b2.population) as bbg_avg_pct_am_ind,
  sum(b2.pct_asian *b2.population)/sum(b2.population) as bbg_avg_pct_asian,
  sum(b2.pct_nh_pi *b2.population)/sum(b2.population) as bbg_avg_pct_nh_pi,
  sum(b2.pct_multiple *b2.population)/sum(b2.population) as bbg_avg_pct_multiple,
  sum(b2.pct_other *b2.population)/sum(b2.population) as bbg_avg_pct_other,
  sum(b2.renter_occupied_households *b2.population)/sum(b2.population) as bbg_avg_renter_occupied_households,
  sum(b2.eviction_filings *b2.population)/sum(b2.population) as bbg_avg_eviction_filings,
  sum(b2.eviction_filing_rate *b2.population)/sum(b2.population) as bbg_avg_filing_rate,
  sum(b2.conversion_rate *b2.population)/sum(b2.population) as bbg_avg_conversion_rate,
  sum(b2.avg_hh_size *b2.population)/sum(b2.population) as bbg_avg_hh_size"""

CREATE_TEMP_BG = '''CREATE tmp_bbg AS (SELECT blockgroup.geo_id, blockgroup.year, evictions, eviction_rate, population,
                    poverty_rate, pct_renter_occupied, median_gross_rent, median_household_income, median_property_value,
                    rent_burden, pct_white, pct_af_am, pct_hispanic, pct_am_ind, pct_asian, pct_nh_pi, pct_multiple,
                    pct_other, pct_renter_occupied_households, renter_occupied_households, eviction_filings, eviction_filing_rate,
                    conversion_rate, avg_hh_size, census_blck_grp_shp.geom from evictions.blockgroup join evictions.census_blck_grp_shp on blockgroup.geo_id = census_blck_grp_shp.geoid10::varchar(12)
                    );'''


CREATE_TABLE_AS_BBG = """CREATE TABLE bbg as (SELECT b1.geo_id as gid, b1.year as yr,
  sum(b2.evictions*b2.population*b2.population)/sum(b2.population) as bbg_sum_evict,
  sum(b2.eviction_rate*b2.population)/sum(b2.population) as bbg_avg_evict_rate,
  sum(b2.population*b2.population)/sum(b2.population) as bbg_avg_population,
  sum(b2.poverty_rate*b2.population)/sum(b2.population) as bbg_avg_poverty_rate,
  sum(b2.pct_renter_occupied*b2.population)/sum(b2.population) as bbg_avg_pct_renter_occupied,
  sum(b2.median_gross_rent*b2.population)/sum(b2.population) as bbg_avg_median_gross_rent,
  sum(b2.median_household_income*b2.population)/sum(b2.population) as bbg_avg_median_household_income,
  sum(b2.median_property_value*b2.population)/sum(b2.population) as bbg_avg_median_property_value,
  sum(b2.rent_burden*b2.population)/sum(b2.population) as bbg_avg_rent_burden,
  sum(b2.pct_white*b2.population)/sum(b2.population) as bbg_avg_pct_white,
  sum(b2.pct_af_am*b2.population)/sum(b2.population) as bbg_avg_pct_af_am,
  sum(b2.pct_hispanic*b2.population)/sum(b2.population) as bbg_avg_pct_hispanic,
  sum(b2.pct_am_ind*b2.population)/sum(b2.population) as bbg_avg_pct_am_ind,
  sum(b2.pct_asian*b2.population)/sum(b2.population) as bbg_avg_pct_asian,
  sum(b2.pct_nh_pi*b2.population)/sum(b2.population) as bbg_avg_pct_nh_pi,
  sum(b2.pct_multiple*b2.population)/sum(b2.population) as bbg_avg_pct_multiple,
  sum(b2.pct_other*b2.population)/sum(b2.population) as bbg_avg_pct_other,
  sum(b2.renter_occupied_households*b2.population)/sum(b2.population) as bbg_avg_renter_occupied_households,
  sum(b2.eviction_filings*b2.population)/sum(b2.population) as bbg_avg_eviction_filings,
  sum(b2.eviction_filing_rate*b2.population)/sum(b2.population) as bbg_avg_filing_rate,
  sum(b2.conversion_rate*b2.population)/sum(b2.population) as bbg_avg_conversion_rate,
  sum(b2.avg_hh_size*b2.population)/sum(b2.population) as bbg_avg_hh_size
                        from county_bbg b1
                        join county_bbg b2
                        on ST_Intersects(b1.geom, b2.geom)
                        and b1.year = b2.year
                        group by b1.geo_id, b1.year);"""

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
                            ntile(5) over(ORDER BY evictions DESC *b2.population)/sum(b2.population) as num_quint,
                            ntile(5) over(ORDER BY eviction_rate DESC *b2.population)/sum(b2.population) as rate_quint
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

INSERT_N_YEAR_AVG = """UPDATE  {} bg
                        set {} = tmp.aver
                        from (
                                select b1.geo_id, b1.year, avg(b2.{}) as aver
                                from {} b1 join {} b2
                                  on b1.geo_id=b2.geo_id
                                  and b2.year between (b1.year - {}) and (b1.year - 1)
                                group by (b1.geo_id, b1.year)
                            ) as tmp
                            where bg.geo_id=tmp.geo_id
                            and bg.year=tmp.year;"""



                        UPDATE  evictions_tract bg
                        set conversion_rate_avg_3yr = tmp.aver
                        from (
                                select b1.geo_id, b1.year, avg(b2.conversion_rate) as aver
                                from evictions_tract b1 join evictions_tract b2
                                  on b1.geo_id=b2.geo_id
                                  and b2.year between (b1.year - 3) and (b1.year - 1)
                                group by (b1.geo_id, b1.year)
                            ) as tmp
                            where bg.geo_id=tmp.geo_id
                            and bg.year=tmp.year;"""

INSERT_N_YEAR_PCT_CHANGE = """
                            UPDATE {} bg
                                SET {} = tmp.pct_change
                                FROM (
                                  select b1.geo_id, b1.year,
                                  case
                                    when b1.{} = 0 and b2.{} = 0 then 0
                                    when b1.{} > 0 and b2.{} = 0 then 999999
                                    when b1.{} < 0 and b2.{} = 0 then -999999
                                    else (b1.{} - b2.{})/b2.{}*100
                                  end as pct_change
                                from {} b1
                                join {} b2 on b1.geo_id=b2.geo_id
                                    and b2.year = b1.year-{}
                                  ) as tmp
                            where bg.geo_id=tmp.geo_id
                            and bg.year=tmp.year;
                            """


# Note: may need to be called in a loop over all years
INSERT_NTILE_DISCRETIZATION = """INSERT into {}(geo_id, year, {})
                                SELECT geo_id, year, ntile({}) over (order by {} desc *b2.population)/sum(b2.population) as {}
                                FROM blockgroup
                                WHERE year = {};
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

OUTCOME_CAT_CHANGE_0_1 = """UPDATE outcome o
                            SET {} = val
                            from (select o1.geo_id as geo_id, o1.year as year, case
                                          when o1.{} - o2.{}} = 1 then 1
                                          else 0
                                      end as val
                                      from outcome as o1
                                      join outcome as o2
                                      on o1.geo_id = o2.geo_id
                                      and o1.year-1 = o2.year
                                       *b2.population)/sum(b2.population) as l
                          where o.geo_id = l.geo_id
                          and o.year = l.year;
                          """

OUTCOME_CAT_CHANGE_1_0 = """UPDATE outcome o
                            SET {} = val
                            from (select o1.geo_id as geo_id, o1.year as year, case
                                          when o1.{} - o2.{}} = -1 then 1
                                          else 0
                                      end as val
                                      from outcome as o1
                                      join outcome as o2
                                      on o1.geo_id = o2.geo_id
                                      and o1.year-1 = o2.year
                                       *b2.population)/sum(b2.population) as l
                          where o.geo_id = l.geo_id
                          and o.year = l.year;
                          """
'''=========================================================================
    Permits table
==========================================================================='''

DROP_TABLE_PERMITS = "DROP TABLE if exists permits;"

CREATE_TABLE_PERMITS = """CREATE TABLE permits (
    year SMALLINT,
    state varchar(2),
    county varchar(3),
    region varchar(1),
    division varchar(1),
    total_bldg int,
    total_units int,
    total_value FLOAT,
    geo_id varchar(5),
    PRIMARY KEY (year, geo_id)
    );"""

COPY_CSV_PERMITS = """COPY evictions.permits (
    year, state, county, region, division, total_bldg, total_units, total_value, geo_id)
    FROM stdin WITH CSV HEADER DELIMITER as ',' """


'''=========================================================================
    Household Size Table
==========================================================================='''

DROP_TABLE_HHSIZE = "DROP TABLE if exists hhsize;"

CREATE_TABLE_HHSIZE = """CREATE TABLE hhsize (
    year SMALLINT,
    geo_id varchar(12),
    avg_size FLOAT,
    PRIMARY KEY (year, geo_id)
    );"""

COPY_CSV_HHSIZE = """COPY evictions.hhsize (
    year, geo_id, avg_size)
    FROM stdin WITH CSV HEADER DELIMITER as ',' """

CREATE_IDX_HH = "CREATE INDEX hh_year_gid on hhsize (geo_id, year);"

CREATE_VAR_HHSIZE = "ALTER TABLE evictions.blockgroup add column avg_hh_size FLOAT;"

UPDATE_VAR_HHSIZE = """UPDATE evictions.blockgroup set avg_hh_size =  hhsize.avg_size from hhsize
                      where hhsize.geo_id = blockgroup.geo_id and hhsize.year =blockgroup.year;"""


'''============================================================================
    Evictions Lags
============================================================================'''

ADD_CONVERSION_LAGS = "ALTER TABLE evictions.evictions_tract add column conversion_rate float8;"

UPDATE_CONVERSION_RATE = """
update evictions_tract et
    set conversion_rate = l.conversion_rate
    from (select year, geo_id, evictions, eviction_filings, 
    case when eviction_filings !=0 and eviction_filings is not null then evictions/eviction_filings 
         when eviction_filings = 0 then 0
         else null
   end as conversion_rate
   from evictions_tract) as l
where et.geo_id = l.geo_id
and et.year = l.year;"""


CREATE_EV_TABLE = """CREATE TABLE ev_{} as (SELECT geo_id, year, eviction_filings, evictions, eviction_rate, eviction_filing_rate, 
                                            conversion_rate FROM {});"""

ADD_TRACT_COLS_BG = """ALTER TABLE blockgroup 
  ADD COLUMN eviction_filings_lag float8,
  ADD COLUMN evictions_lag float8,
  ADD COLUMN eviction_rate_lag float8,
  ADD COLUMN eviction_filing_rate_lag float8,
  ADD COLUMN conversion_rate_lag float8,
  ADD COLUMN eviction_filings_avg_3yr_lag float8,
  ADD COLUMN evictions_avg_3yr_lag float8,
  ADD COLUMN eviction_rate_avg_3yr_lag float8,
  ADD COLUMN eviction_filing_rate_avg_3yr_lag float8,
  ADD COLUMN conversion_rate_avg_3yr_lag float8,
  ADD COLUMN eviction_filings_avg_5yr_lag float8,
  ADD COLUMN evictions_avg_5yr_lag float8,
  ADD COLUMN eviction_rate_avg_5yr_lag float8,
  ADD COLUMN eviction_filing_rate_avg_5yr_lag float8,
  ADD COLUMN conversion_rate_avg_5yr_lag float8,
  ADD COLUMN eviction_filings_pct_change_1yr_lag float8,
  ADD COLUMN evictions_pct_change_1yr_lag float8,
  ADD COLUMN eviction_rate_pct_change_1yr_lag float8,
  ADD COLUMN eviction_filing_rate_pct_change_1yr_lag float8,
  ADD COLUMN conversion_rate_pct_change_1yr_lag float8,
  ADD COLUMN eviction_filings_pct_change_3yr_lag float8,
  ADD COLUMN evictions_pct_change_3yr_lag float8,
  ADD COLUMN eviction_rate_pct_change_3yr_lag float8,
  ADD COLUMN eviction_filing_rate_pct_change_3yr_lag float8,
  ADD COLUMN conversion_rate_pct_change_3yr_lag float8,
  ADD COLUMN eviction_filings_pct_change_5yr_lag float8,
  ADD COLUMN evictions_pct_change_5yr_lag float8,
  ADD COLUMN eviction_rate_pct_change_5yr_lag float8,
  ADD COLUMN eviction_filing_rate_pct_change_5yr_lag float8,
  ADD COLUMN conversion_rate_pct_change_5yr_lag float8,
  ADD COLUMN eviction_filings_avg_3yr_lag_tr float8,
  ADD COLUMN evictions_avg_3yr_lag_tr float8,
  ADD COLUMN eviction_rate_avg_3yr_lag_tr float8,
  ADD COLUMN eviction_filing_rate_avg_3yr_lag_tr float8,
  ADD COLUMN conversion_rate_avg_3yr_lag_tr float8,

  ADD COLUMN population_avg_5yr_tr float8,
  ADD COLUMN poverty_rate_avg_5yr_tr float8,
  ADD COLUMN median_gross_rent_avg_5yr_tr float8, 
  ADD COLUMN median_household_income_avg_5yr_tr float8,
  ADD COLUMN median_property_value_avg_5yr_tr float8,
  ADD COLUMN rent_burden_avg_5yr_tr float8,
  ADD COLUMN pct_white_avg_5yr_tr float8,
  ADD COLUMN pct_af_am_avg_5yr_tr float8,
  ADD COLUMN pct_hispanic_avg_5yr_tr float8,
  ADD COLUMN pct_am_ind_avg_5yr_tr float8,
  ADD COLUMN pct_asian_avg_5yr_tr float8,
  ADD COLUMN pct_nh_pi_avg_5yr_tr float8,
  ADD COLUMN pct_multiple_avg_5yr_tr float8,
  ADD COLUMN pct_other_avg_5yr_tr float8,
  ADD COLUMN renter_occupied_households_avg_5yr_tr float8,
  ADD COLUMN pct_renter_occupied_avg_5yr_tr float8,
  ADD COLUMN avg_hh_size_avg_5yr_tr float8,
  ADD COLUMN eviction_filings_avg_5yr_lag_tr float8,
  ADD COLUMN evictions_avg_5yr_lag_tr float8,
  ADD COLUMN eviction_rate_avg_5yr_lag_tr float8,
  ADD COLUMN eviction_filing_rate_avg_5yr_lag_tr float8,
  ADD COLUMN conversion_rate_avg_5yr_lag_tr float8,

  ADD COLUMN eviction_filings_pct_change_3yr_lag_tr float8,
  ADD COLUMN evictions_pct_change_3yr_lag_tr float8,
  ADD COLUMN eviction_rate_pct_change_3yr_lag_tr float8,
  ADD COLUMN eviction_filing_rate_pct_change_3yr_lag_tr float8,
  ADD COLUMN conversion_rate_pct_change_3yr_lag_tr float8,
  ADD COLUMN population_pct_change_5yr_tr float8,
  ADD COLUMN poverty_rate_pct_change_5yr_tr float8,
  ADD COLUMN pct_renter_occupied_pct_change_5yr_tr float8,
  ADD COLUMN median_gross_rent_pct_change_5yr_tr float8, 
  ADD COLUMN median_household_income_pct_change_5yr_tr float8,
  ADD COLUMN median_property_value_pct_change_5yr_tr float8,
  ADD COLUMN rent_burden_pct_change_5yr_tr float8,
  ADD COLUMN pct_white_pct_change_5yr_tr float8,
  ADD COLUMN pct_af_am_pct_change_5yr_tr float8,
  ADD COLUMN pct_hispanic_pct_change_5yr_tr float8,
  ADD COLUMN pct_am_ind_pct_change_5yr_tr float8,
  ADD COLUMN pct_asian_pct_change_5yr_tr float8,
  ADD COLUMN pct_nh_pi_pct_change_5yr_tr float8,
  ADD COLUMN pct_multiple_pct_change_5yr_tr float8,
  ADD COLUMN pct_other_pct_change_5yr_tr float8,
  ADD COLUMN renter_occupied_households_pct_change_5yr_tr float8,
  ADD COLUMN avg_hh_size_pct_change_5yr_tr float8,

  ADD COLUMN eviction_filings_pct_change_5yr_lag_tr float8,
  ADD COLUMN evictions_pct_change_5yr_lag_tr float8,
  ADD COLUMN eviction_rate_pct_change_5yr_lag_tr float8,
  ADD COLUMN eviction_filing_rate_pct_change_5yr_lag_tr float8,
  ADD COLUMN conversion_rate_pct_change_5yr_lag_tr float8,
  ADD COLUMN eviction_filings_pct_change_1yr_lag_tr float8,
  ADD COLUMN evictions_pct_change_1yr_lag_tr float8,
  ADD COLUMN eviction_rate_pct_change_1yr_lag_tr float8,
  ADD COLUMN eviction_filing_rate_pct_change_1yr_lag_tr float8,
  ADD COLUMN conversion_rate_pct_change_1yr_lag_tr float8,
  ADD COLUMN total_bldg_pct_change_1yr float8,
  ADD COLUMN total_bldg_pct_change_3yr float8,
  ADD COLUMN total_bldg_pct_change_5yr float8,
  ADD COLUMN total_bldg_avg_3yr float8,
  ADD COLUMN total_bldg_avg_5yr float8,
  ADD COLUMN total_units_pct_change_1yr float8,
  ADD COLUMN total_units_pct_change_3yr float8,
  ADD COLUMN total_units_pct_change_5yr float8,
  ADD COLUMN total_units_avg_3yr float8,
  ADD COLUMN total_units_avg_5yr float8,
  ADD COLUMN total_value_pct_change_1yr float8,
  ADD COLUMN total_value_pct_change_3yr float8,
  ADD COLUMN total_value_pct_change_5yr float8,
  ADD COLUMN total_value_avg_3yr float8,
  ADD COLUMN total_value_avg_5yr float8,
  ADD COLUMN urban int,
  ADD COLUMN div_sa int,
  ADD COLUMN div_enc int;
"""


UPDATE_COLS_TR = """
UPDATE blockgroup b SET
b.population_avg_5yr_tr = t.population_avg_5yr,
b.poverty_rate_avg_5yr_tr = t.poverty_rate_avg_5yr,
b.pct_renter_occupied_5yr_tr = t.pct_renter_occupied_5yr,
b.median_gross_rent_avg_5yr_tr = t.median_gross_rent_avg_5yr_tr, 
b.median_household_income_avg_5yr_tr =t.median_household_income_avg_5yr_tr ,
b.median_property_value_avg_5yr_tr = t.median_property_value_avg_5yr_tr,
b.rent_burden_avg_5yr_tr = t.rent_burden_avg_5yr_tr,
b.pct_white_avg_5yr_tr = t.pct_white_avg_5yr_tr,
b.pct_af_am_avg_5yr_tr = t.pct_af_am_avg_5yr_tr,
b.pct_hispanic_avg_5yr_tr = t.pct_hispanic_avg_5yr_tr,
b.pct_am_ind_avg_5yr_tr = t.pct_am_ind_avg_5yr_tr,
b.pct_asian_avg_5yr_tr = t.pct_asian_avg_5yr_tr,
b.pct_nh_pi_avg_5yr_tr = t.pct_nh_pi_avg_5yr_tr,
b.pct_multiple_avg_5yr_tr = t.pct_multiple_avg_5yr_tr,
b.pct_other_avg_5yr_tr = t.pct_other_avg_5yr_tr,
b.renter_occupied_households_avg_5yr_tr = t.renter_occupied_households_avg_5yr_tr,
b.pct_renter_occupied_avg_5yr_tr = t.pct_renter_occupied_avg_5yr_tr,
b.avg_hh_size_avg_5yr_tr = t.avg_hh_size_avg_5yr_tr,

b.population_pct_change_5yr_tr = t.population_pct_change_5yr,
b.poverty_rate_pct_change_5yr_tr = t.poverty_rate_pct_change_5yr,
b.pct_renter_occupied_5yr_tr = t.pct_renter_occupied_5yr,
b.median_gross_rent_pct_change_5yr_tr = t.median_gross_rent_pct_change_5yr_tr, 
b.median_household_income_pct_change_5yr_tr =t.median_household_income_pct_change_5yr_tr ,
b.median_property_value_pct_change_5yr_tr = t.median_property_value_pct_change_5yr_tr,
b.rent_burden_pct_change_5yr_tr = t.rent_burden_pct_change_5yr_tr,
b.pct_white_pct_change_5yr_tr = t.pct_white_pct_change_5yr_tr,
b.pct_af_am_pct_change_5yr_tr = t.pct_af_am_pct_change_5yr_tr,
b.pct_hispanic_pct_change_5yr_tr = t.pct_hispanic_pct_change_5yr_tr,
b.pct_am_ind_pct_change_5yr_tr = t.pct_am_ind_pct_change_5yr_tr,
b.pct_asian_pct_change_5yr_tr = t.pct_asian_pct_change_5yr_tr,
b.pct_nh_pi_pct_change_5yr_tr = t.pct_nh_pi_pct_change_5yr_tr,
b.pct_multiple_pct_change_5yr_tr = t.pct_multiple_pct_change_5yr_tr,
b.pct_other_pct_change_5yr_tr = t.pct_other_pct_change_5yr_tr,
b.renter_occupied_households_pct_change_5yr_tr = t.renter_occupied_households_pct_change_5yr_tr,
b.pct_renter_occupied_pct_change_5yr_tr = t.pct_renter_occupied_pct_change_5yr_tr,
b.avg_hh_size_pct_change_5yr_tr = t.avg_hh_size_pct_change_5yr_tr

FROM evictions_tract t 
where b.tract = t.geo_id and b.year = t.year;
"""


UPDATE_COLS_GEO = """
UPDATE blockgroup SET
urban = g.urban,
div_sa = g.div_sa,
div_enc = g.div_enc
from geographic g
where blockgroup.geo_id =g.geo_id and blockgroup.year = g.year;"""

UPDATE_COLS_LAG_BG = """
UPDATE blockgroup b SET

b.eviction_filings_lag = e.eviction_filings,
b.evictions_lag = e.evictions,
b.eviction_rate_lag = e.eviction_rate,
b.eviction_filing_rate_lag = e.eviction_rate,
b.conversion_rate_lag = e.conversion_rate,

b.eviction_filings_avg_3yr_lag = e.eviction_filings_avg_3yr,
b.evictions_avg_3yr_lag = e.evictions_avg_3yr,
b.eviction_rate_avg_3yr_lag = e.eviction_rate_avg_3yr,
b.eviction_filing_rate_avg_3yr_lag = e.eviction_rate_avg_3yr,
b.conversion_rate_avg_3yr_lag = e.conversion_rate_avg_3yr,

b.eviction_filings_avg_5yr_lag = e.eviction_filings_avg_5yr,
b.evictions_avg_5yr_lag = e.evictions_avg_5yr,
b.eviction_rate_avg_5yr_lag = e.eviction_rate_avg_5yr,
b.eviction_filing_rate_avg_5yr_lag = e.eviction_rate_avg_5yr,
b.conversion_rate_avg_5yr_lag = e.conversion_rate_avg_5yr,

b.eviction_filings_pct_change_1yr_lag = e.eviction_filings_pct_change_1yr,
b.evictions_pct_change_1yr_lag = e.evictions_pct_change_1yr,
b.eviction_rate_pct_change_1yr_lag = e.eviction_rate_pct_change_1yr,
b.eviction_filing_rate_pct_change_1yr_lag = e.eviction_rate_pct_change_1yr,
b.conversion_rate_pct_change_1yr_lag = e.conversion_rate_pct_change_1yr,

b.eviction_filings_pct_change_3yr_lag = e.eviction_filings_pct_change_3yr,
b.evictions_pct_change_3yr_lag = e.evictions_pct_change_3yr,
b.eviction_rate_pct_change_3yr_lag = e.eviction_rate_pct_change_3yr,
b.eviction_filing_rate_pct_change_3yr_lag = e.eviction_rate_pct_change_3yr,
b.conversion_rate_pct_change_3yr_lag = e.conversion_rate_pct_change_3yr,

b.eviction_filings_pct_change_5yr_lag = e.eviction_filings_pct_change_5yr,
b.evictions_pct_change_5yr_lag = e.evictions_pct_change_5yr,
b.eviction_rate_pct_change_5yr_lag = e.eviction_rate_pct_change_5yr,
b.eviction_filing_rate_pct_change_5yr_lag = e.eviction_rate_pct_change_5yr,
b.conversion_rate_pct_change_5yr_lag = e.conversion_rate_pct_change_5yr

FROM ev_blockgroup e
where b.geo_id = e.geo_id and b.year = e.year - 1;





"""


'''============================================================================
    ML Incremental Cursor
============================================================================'''
CREATE_BG_CURSOR = """declare bg_cursor CURSOR WITHOUT HOLD FOR
	                     select * from blockgroup bg
                            inner join outcome o on o.geo_id=bg.geo_id and o.year=bg.year;
                    """

DROP_FUNCTION_BG_CURSOR = "drop function if exists bg_get_chunk(refcursor);"

CREATE_FUNCTION_BG_CURSOR = """create function bg_get_chunk(refcursor) returns refcursor as $$
                            	begin
                            		open $1 for
                            			select * from blockgroup bg
                            			inner join outcome o on o.geo_id=bg.geo_id and o.year=bg.year;
                            		return $1;
                            	end;
                            	$$ language plpgsql;
                            """
