""""DB Statements for the evictions-learn project."""

"""==============================================================
    NOTES
=============================================================="""
# Geographic sample includes East North Central and South Atlantic regions
# East North Central: Indiana (18) Illinois (17) Michigan (26) Ohio (39) Wisconsin (55)
# South Atlantic: Delaware (10) District of Columbia (11) Florida  (12) Georgia (13)
# Maryland  (24) North Carolina (37) South Carolina (45) Virginia (51) West Virginia  (54)


Drop_State = "delete from {} where {} not in ('18', '17', '26', '39', '55', '10', '11', '12', '13', '24', '37', '45', '51', '54');"


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
    population float4,
    poverty_rate float4,
    pct_renter_occupied float4,
    median_gross_rent float4,
    median_household_income float4,
    median_property_value float4,
    rent_burden float4,
    pct_white float4,
    pct_af_am float4,
    pct_hispanic float4,
    pct_am_ind float4,
    pct_asian float4,
    pct_nh_pi float4,
    pct_multiple float4,
    pct_other float4,
    renter_occupied_households float4,
    eviction_filings float4,
    evictions float4,
    eviction_rate float4,
    eviction_filing_rate float4,
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
CREATE_VAR_STATE = "ALTER TABLE evictions.blockgroup state CHAR(2);"
CREATE_VAR_TRACT = "ALTER TABLE evictions.blockgroup tract CHAR(11);"
CREATE_VAR_COUNTY = "ALTER TABLE evictions.blockgroup county CHAR(5);"

UPDATE_VAR_STATE = "UPDATE evictions.blockgroup set state = substring(geo_id from 1 for 2);"
UPDATE_VAR_TRACT = "UPDATE evictions.blockgroup set tract = substring(geo_id from 1 for 11);"
UPDATE_VAR_COUNTY = "UPDATE evictions.blockgroup set county = substring(geo_id from 1 for 5);"

'''==========================================================================
SHAPEFILE LOAD 
=========================================================================='''

DROP_TABLE_EVICTIONS_GEO = "DROP TABLE IF EXISTS evictions_{};"
DROP_TABLE_SHP = "DROP TABLE IF EXISTS census_{}_shp;"

CREATE_IDX_Shape = "CREATE INDEX idx_geoid10 on census_blck_grp_shp (geoid10);"

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
ADD PRIMARY KEY(geo_id, year);"""


COPY_CSV_URBAN = """COPY evictions.urban(UA, STATE, COUNTY, GEOID)
      from '/Users/alenastern/Documents/Spring2018/Machine_Learning/evictions-learn/src/data/Urban_County_2010_states.csv' with CSV HEADER DELIMITER as ',';"""


IDX_COUNTY_GEO = "CREATE INDEX idx_county_geo ON geographic (county);"
IDX_STATE_GEO = "CREATE INDEX idx_state_geo ON geographic (state);"


UPDATE_VAR_URBAN = '''UPDATE evictions.blockgroup
                      SET urban = 1
                      WHERE county IN (SELECT GEOID::varchar(5) FROM evictions.urban);'''

UPDATE_VAR_DIV_NE = '''UPDATE evictions.geographic set div_ne = 1
  WHERE state = '09' OR state = '23'g
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

ADD_COLUMN = "ALTER TABLE {} ADD COLUMN {} {};"

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
    total_value float4,
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
    avg_size float4,
    PRIMARY KEY (year, geo_id)
    );"""

COPY_CSV_HHSIZE = """COPY evictions.hhsize (
    year, geo_id, avg_size)
    FROM stdin WITH CSV HEADER DELIMITER as ',' """

CREATE_IDX_HH = "CREATE INDEX hh_year_gid on hhsize (geo_id, year);"

CREATE_VAR_HHSIZE = "ALTER TABLE evictions.blockgroup avg_hh_size FLOAT;"

UPDATE_VAR_HHSIZE = """UPDATE evictions.blockgroup set avg_hh_size =  hhsize.avg_size from hhsize
                      where hhsize.geo_id = blockgroup.geo_id and hhsize.year =blockgroup.year;"""


'''============================================================================
    Evictions Lags
============================================================================'''

ADD_CONVERSION_LAGS = "ALTER TABLE evictions.evictions_tract ADD COLUMN conversion_rate float4;"

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


CREATE_EV_TABLE = """CREATE TABLE ev_lag_{} as (SELECT geo_id, year, eviction_filings, evictions, eviction_rate, eviction_filing_rate,
                                            conversion_rate, eviction_filings_avg_3yr, evictions_avg_3yr, eviction_rate_avg_3yr, eviction_filing_rate_avg_3yr,
                                            conversion_rate_avg_3yr, eviction_filings_avg_5yr, evictions_avg_5yr, eviction_rate_avg_5yr, eviction_filing_rate_avg_5yr,
                                            conversion_rate_avg_5yr, eviction_filings_pct_change_1yr, evictions_pct_change_1yr, eviction_rate_pct_change_1yr, eviction_filing_rate_pct_change_1yr,
                                            conversion_rate_pct_change_1yr, eviction_filings_pct_change_3yr, evictions_pct_change_3yr, eviction_rate_pct_change_3yr, eviction_filing_rate_pct_change_3yr,
                                            conversion_rate_pct_change_3yr, eviction_filings_pct_change_5yr, evictions_pct_change_5yr, eviction_rate_pct_change_5yr, eviction_filing_rate_pct_change_5yr,
                                            conversion_rate_pct_change_5yr FROM {});"""



#Clean add column, float8

ADD_LAG_COLS_BG = """ALTER TABLE blockgroup 
  ADD COLUMN eviction_filings_lag float4,
  ADD COLUMN evictions_lag float4,
  ADD COLUMN eviction_rate_lag float4,
  ADD COLUMN eviction_filing_rate_lag float4,
  ADD COLUMN conversion_rate_lag float4,
  ADD COLUMN eviction_filings_avg_3yr_lag float4,
  ADD COLUMN evictions_avg_3yr_lag float4,
  ADD COLUMN eviction_rate_avg_3yr_lag float4,
  ADD COLUMN eviction_filing_rate_avg_3yr_lag float4,
  ADD COLUMN conversion_rate_avg_3yr_lag float4,
  ADD COLUMN eviction_filings_avg_5yr_lag float4,
  ADD COLUMN evictions_avg_5yr_lag float4,
  ADD COLUMN eviction_rate_avg_5yr_lag float4,
  ADD COLUMN eviction_filing_rate_avg_5yr_lag float4,
  ADD COLUMN conversion_rate_avg_5yr_lag float4,
  ADD COLUMN eviction_filings_pct_change_1yr_lag float4,
  ADD COLUMN evictions_pct_change_1yr_lag float4,
  ADD COLUMN eviction_rate_pct_change_1yr_lag float4,
  ADD COLUMN eviction_filing_rate_pct_change_1yr_lag float4,
  ADD COLUMN conversion_rate_pct_change_1yr_lag float4,
  ADD COLUMN eviction_filings_pct_change_3yr_lag float4,
  ADD COLUMN evictions_pct_change_3yr_lag float4,
  ADD COLUMN eviction_rate_pct_change_3yr_lag float4,
  ADD COLUMN eviction_filing_rate_pct_change_3yr_lag float4,
  ADD COLUMN conversion_rate_pct_change_3yr_lag float4,
  ADD COLUMN eviction_filings_pct_change_5yr_lag float4,
  ADD COLUMN evictions_pct_change_5yr_lag float4,
  ADD COLUMN eviction_rate_pct_change_5yr_lag float4,
  ADD COLUMN eviction_filing_rate_pct_change_5yr_lag float4,
  ADD COLUMN conversion_rate_pct_change_5yr_lag float4;"""

ADD_TRACT_COLS_BG = """ALTER TABLE blockgroup 
  ADD COLUMN population_tr float4,
  ADD COLUMN poverty_rate_tr float4,
  ADD COLUMN median_gross_rent_tr float4, 
  ADD COLUMN median_household_income_tr float4,
  ADD COLUMN median_property_value_tr float4,
  ADD COLUMN rent_burden_tr float4,
  ADD COLUMN pct_white_tr float4,
  ADD COLUMN pct_af_am_tr float4,
  ADD COLUMN pct_hispanic_tr float4,
  ADD COLUMN pct_am_ind_tr float4,
  ADD COLUMN pct_asian_tr float4,
  ADD COLUMN pct_nh_pi_tr float4,
  ADD COLUMN pct_multiple_tr float4,
  ADD COLUMN pct_other_tr float4,
  ADD COLUMN renter_occupied_households_tr float4,
  ADD COLUMN pct_renter_occupied_tr float4,
  ADD COLUMN avg_hh_size_tr float4,
  ADD COLUMN eviction_filings_lag_tr float4,
  ADD COLUMN evictions_lag_tr float4,
  ADD COLUMN eviction_rate_lag_tr float4,
  ADD COLUMN eviction_filing_rate_lag_tr float4,
  ADD COLUMN conversion_rate_lag_tr float4,
  ADD COLUMN eviction_filings_avg_3yr_lag_tr float4,
  ADD COLUMN evictions_avg_3yr_lag_tr float4,
  ADD COLUMN eviction_rate_avg_3yr_lag_tr float4,
  ADD COLUMN eviction_filing_rate_avg_3yr_lag_tr float4,
  ADD COLUMN conversion_rate_avg_3yr_lag_tr float4,
  ADD COLUMN eviction_filings_avg_5yr_lag_tr float4,
  ADD COLUMN evictions_avg_5yr_lag_tr float4,
  ADD COLUMN eviction_rate_avg_5yr_lag_tr float4,
  ADD COLUMN eviction_filing_rate_avg_5yr_lag_tr float4,
  ADD COLUMN conversion_rate_avg_5yr_lag_tr float4,
  ADD COLUMN eviction_filings_pct_change_1yr_lag_tr float4,
  ADD COLUMN evictions_pct_change_1yr_lag_tr float4,
  ADD COLUMN eviction_rate_pct_change_1yr_lag_tr float4,
  ADD COLUMN eviction_filing_rate_pct_change_1yr_lag_tr float4,
  ADD COLUMN conversion_rate_pct_change_1yr_lag_tr float4,
  ADD COLUMN eviction_filings_pct_change_3yr_lag_tr float4,
  ADD COLUMN evictions_pct_change_3yr_lag_tr float4,
  ADD COLUMN eviction_rate_pct_change_3yr_lag_tr float4,
  ADD COLUMN eviction_filing_rate_pct_change_3yr_lag_tr float4,
  ADD COLUMN conversion_rate_pct_change_3yr_lag_tr float4,
  ADD COLUMN eviction_filings_pct_change_5yr_lag_tr float4,
  ADD COLUMN evictions_pct_change_5yr_lag_tr float4,
  ADD COLUMN eviction_rate_pct_change_5yr_lag_tr float4,
  ADD COLUMN eviction_filing_rate_pct_change_5yr_lag_tr float4,
  ADD COLUMN conversion_rate_pct_change_5yr_lag_tr float4
  ADD COLUMN population_avg_5yr_tr,
  ADD COLUMN poverty_rate_avg_5yr_tr,
  ADD COLUMN median_gross_rent_avg_5yr_tr,
  ADD COLUMN median_household_income_avg_5yr_tr,
  ADD COLUMN median_property_value_avg_5yr_tr,
  ADD COLUMN rent_burden_avg_5yr_tr,
  ADD COLUMN pct_white_avg_5yr_tr,
  ADD COLUMN pct_af_am_avg_5yr_tr,
  ADD COLUMN pct_hispanic_avg_5yr_tr,
  ADD COLUMN pct_am_ind_avg_5yr_tr,
  ADD COLUMN pct_asian_avg_5yr_tr,
  ADD COLUMN pct_nh_pi_avg_5yr_tr,
  ADD COLUMN pct_multiple_avg_5yr_tr,
  ADD COLUMN pct_other_avg_5yr_tr,
  ADD COLUMN renter_occupied_households_avg_5yr_tr,
  ADD COLUMN pct_renter_occupied_avg_5yr_tr,
  ADD COLUMN avg_hh_size_avg_5yr_tr,
  ADD COLUMN population_pct_change_5yr_tr,
  ADD COLUMN poverty_rate_pct_change_5yr_tr,
  ADD COLUMN pct_renter_occupied_pct_change_5yr_tr,
  ADD COLUMN median_gross_rent_pct_change_5yr_tr,
  ADD COLUMN median_household_income_pct_change_5yr_tr,
  ADD COLUMN median_property_value_pct_change_5yr_tr,
  ADD COLUMN rent_burden_pct_change_5yr_tr,
  ADD COLUMN pct_white_pct_change_5yr_tr,
  ADD COLUMN pct_af_am_pct_change_5yr_tr,
  ADD COLUMN pct_hispanic_pct_change_5yr_tr,
  ADD COLUMN pct_am_ind_pct_change_5yr_tr,
  ADD COLUMN pct_asian_pct_change_5yr_tr,
  ADD COLUMN pct_nh_pi_pct_change_5yr_tr,
  ADD COLUMN pct_multiple_pct_change_5yr_tr,
  ADD COLUMN pct_other_pct_change_5yr_tr,
  ADD COLUMN renter_occupied_households_pct_change_5yr_tr,
  ADD COLUMN avg_hh_size_pct_change_5yr_tr;"""

ADD_ECON_COLS = """
  ADD COLUMN total_bldg int,
  ADD COLUMN total_units int,
  ADD COLUMN total_value float4,
  ADD COLUMN total_bldg_pct_change_1yr,
  ADD COLUMN total_bldg_pct_change_3yr
  ADD COLUMN total_bldg_pct_change_5yr,
  ADD COLUMN total_bldg_avg_3yr,
  ADD COLUMN total_bldg_avg_5yr,
  ADD COLUMN total_units_pct_change_1yr,
  ADD COLUMN total_units_pct_change_3yr,
  ADD COLUMN total_units_pct_change_5yr,
  ADD COLUMN total_units_avg_3yr,
  ADD COLUMN total_units_avg_5yr,
  ADD COLUMN total_value_pct_change_1yr,
  ADD COLUMN total_value_pct_change_3yr,
  ADD COLUMN total_value_pct_change_5yr,
  ADD COLUMN total_value_avg_3yr,
  ADD COLUMN total_value_avg_5yr,
  urban int,
  div_sa int,
  div_enc int;"""



UPDATE_DEM_TR_COLS_BG = """
UPDATE blockgroup SET
population_tr = t.population,
poverty_rate_tr = t.poverty_rate,
median_gross_rent_tr = t.median_gross_rent, 
median_household_income_tr =t.median_household_income,
median_property_value_tr = t.median_property_value,
rent_burden_tr = t.rent_burden,
pct_white_tr = t.pct_white,
pct_af_am_tr = t.pct_af_am,
pct_hispanic_tr = t.pct_hispanic,
pct_am_ind_tr = t.pct_am_ind,
pct_asian_tr = t.pct_asian,
pct_nh_pi_tr = t.pct_nh_pi,
pct_multiple_tr = t.pct_multiple,
pct_other_tr = t.pct_other,
renter_occupied_households_tr = t.renter_occupied_households,
pct_renter_occupied_tr = t.pct_renter_occupied,
avg_hh_size_tr = t.avg_hh_size
population_avg_5yr_tr = t.population_avg_5yr,
poverty_rate_avg_5yr_tr = t.poverty_rate_avg_5yr,
median_gross_rent_avg_5yr_tr = t.median_gross_rent_avg_5yr, 
median_household_income_avg_5yr_tr =t.median_household_income_avg_5yr,
median_property_value_avg_5yr_tr = t.median_property_value_avg_5yr,
rent_burden_avg_5yr_tr = t.rent_burden_avg_5yr,
pct_white_avg_5yr_tr = t.pct_white_avg_5yr,
pct_af_am_avg_5yr_tr = t.pct_af_am_avg_5yr,
pct_hispanic_avg_5yr_tr = t.pct_hispanic_avg_5yr,
pct_am_ind_avg_5yr_tr = t.pct_am_ind_avg_5yr,
pct_asian_avg_5yr_tr = t.pct_asian_avg_5yr,
pct_nh_pi_avg_5yr_tr = t.pct_nh_pi_avg_5yr,
pct_multiple_avg_5yr_tr = t.pct_multiple_avg_5yr,
pct_other_avg_5yr_tr = t.pct_other_avg_5yr,
renter_occupied_households_avg_5yr_tr = t.renter_occupied_households_avg_5yr,
pct_renter_occupied_avg_5yr_tr = t.pct_renter_occupied_avg_5yr,
avg_hh_size_avg_5yr_tr = t.avg_hh_size_avg_5yr,
population_pct_change_5yr_tr = t.population_pct_change_5yr,
poverty_rate_pct_change_5yr_tr = t.poverty_rate_pct_change_5yr,
median_gross_rent_pct_change_5yr_tr = t.median_gross_rent_pct_change_5yr, 
median_household_income_pct_change_5yr_tr =t.median_household_income_pct_change_5yr,
median_property_value_pct_change_5yr_tr = t.median_property_value_pct_change_5yr,
rent_burden_pct_change_5yr_tr = t.rent_burden_pct_change_5yr,
pct_white_pct_change_5yr_tr = t.pct_white_pct_change_5yr,
pct_af_am_pct_change_5yr_tr = t.pct_af_am_pct_change_5yr,
pct_hispanic_pct_change_5yr_tr = t.pct_hispanic_pct_change_5yr,
pct_am_ind_pct_change_5yr_tr = t.pct_am_ind_pct_change_5yr,
pct_asian_pct_change_5yr_tr = t.pct_asian_pct_change_5yr,
pct_nh_pi_pct_change_5yr_tr = t.pct_nh_pi_pct_change_5yr,
pct_multiple_pct_change_5yr_tr = t.pct_multiple_pct_change_5yr,
pct_other_pct_change_5yr_tr = t.pct_other_pct_change_5yr,
renter_occupied_households_pct_change_5yr_tr = t.renter_occupied_households_pct_change_5yr,
pct_renter_occupied_pct_change_5yr_tr = t.pct_renter_occupied_pct_change_5yr,
avg_hh_size_pct_change_5yr_tr = t.avg_hh_size_pct_change_5yr

FROM tr t 
where blockgroup.tract = t.geo_id and blockgroup.year = t.year;"""

UPDATE_COLS_GEO = """
UPDATE blockgroup SET
urban = g.urban,
div_sa = g.div_sa,
div_enc = g.div_enc
from geographic g
where blockgroup.geo_id =g.geo_id and blockgroup.year = g.year;"""

UPDATE_COLS_LAG_BG = """
UPDATE blockgroup SET
eviction_filings_lag = e.eviction_filings,
evictions_lag = e.evictions,
eviction_rate_lag = e.eviction_rate,
eviction_filing_rate_lag = e.eviction_rate,
conversion_rate_lag = e.conversion_rate,

eviction_filings_avg_3yr_lag = e.eviction_filings_avg_3yr,
evictions_avg_3yr_lag = e.evictions_avg_3yr,
eviction_rate_avg_3yr_lag = e.eviction_rate_avg_3yr,
eviction_filing_rate_avg_3yr_lag = e.eviction_rate_avg_3yr,
conversion_rate_avg_3yr_lag = e.conversion_rate_avg_3yr,

eviction_filings_avg_5yr_lag = e.eviction_filings_avg_5yr,
evictions_avg_5yr_lag = e.evictions_avg_5yr,
eviction_rate_avg_5yr_lag = e.eviction_rate_avg_5yr,
eviction_filing_rate_avg_5yr_lag = e.eviction_rate_avg_5yr,
conversion_rate_avg_5yr_lag = e.conversion_rate_avg_5yr,

eviction_filings_pct_change_1yr_lag = e.eviction_filings_pct_change_1yr,
evictions_pct_change_1yr_lag = e.evictions_pct_change_1yr,
eviction_rate_pct_change_1yr_lag = e.eviction_rate_pct_change_1yr,
eviction_filing_rate_pct_change_1yr_lag = e.eviction_rate_pct_change_1yr,
conversion_rate_pct_change_1yr_lag = e.conversion_rate_pct_change_1yr,

eviction_filings_pct_change_3yr_lag = e.eviction_filings_pct_change_3yr,
evictions_pct_change_3yr_lag = e.evictions_pct_change_3yr,
eviction_rate_pct_change_3yr_lag = e.eviction_rate_pct_change_3yr,
eviction_filing_rate_pct_change_3yr_lag = e.eviction_rate_pct_change_3yr,
conversion_rate_pct_change_3yr_lag = e.conversion_rate_pct_change_3yr,

eviction_filings_pct_change_5yr_lag = e.eviction_filings_pct_change_5yr,
evictions_pct_change_5yr_lag = e.evictions_pct_change_5yr,
eviction_rate_pct_change_5yr_lag = e.eviction_rate_pct_change_5yr,
eviction_filing_rate_pct_change_5yr_lag = e.eviction_rate_pct_change_5yr,
conversion_rate_pct_change_5yr_lag = e.conversion_rate_pct_change_5yr

FROM ev_lag_blockgroup e
where blockgroup.geo_id = e.geo_id and blockgroup.year = e.year + 1;"""

UPDATE_COLS_LAG_TR = """
UPDATE blockgroup SET

eviction_filings_lag_tr = e.eviction_filings,
evictions_lag_tr = e.evictions,
eviction_rate_lag_tr = e.eviction_rate,
eviction_filing_rate_lag_tr = e.eviction_rate,
conversion_rate_lag_tr = e.conversion_rate,

eviction_filings_avg_3yr_lag_tr = e.eviction_filings_avg_3yr,
evictions_avg_3yr_lag_tr = e.evictions_avg_3yr,
eviction_rate_avg_3yr_lag_tr = e.eviction_rate_avg_3yr,
eviction_filing_rate_avg_3yr_lag_tr = e.eviction_rate_avg_3yr,
conversion_rate_avg_3yr_lag_tr = e.conversion_rate_avg_3yr,

eviction_filings_avg_5yr_lag_tr = e.eviction_filings_avg_5yr,
evictions_avg_5yr_lag_tr = e.evictions_avg_5yr,
eviction_rate_avg_5yr_lag_tr = e.eviction_rate_avg_5yr,
eviction_filing_rate_avg_5yr_lag_tr = e.eviction_rate_avg_5yr,
conversion_rate_avg_5yr_lag_tr = e.conversion_rate_avg_5yr,

eviction_filings_pct_change_1yr_lag_tr = e.eviction_filings_pct_change_1yr,
evictions_pct_change_1yr_lag_tr = e.evictions_pct_change_1yr,
eviction_rate_pct_change_1yr_lag_tr = e.eviction_rate_pct_change_1yr,
eviction_filing_rate_pct_change_1yr_lag_tr = e.eviction_rate_pct_change_1yr,
conversion_rate_pct_change_1yr_lag_tr = e.conversion_rate_pct_change_1yr,

eviction_filings_pct_change_3yr_lag_tr = e.eviction_filings_pct_change_3yr,
evictions_pct_change_3yr_lag_tr = e.evictions_pct_change_3yr,
eviction_rate_pct_change_3yr_lag_tr = e.eviction_rate_pct_change_3yr,
eviction_filing_rate_pct_change_3yr_lag_tr = e.eviction_rate_pct_change_3yr,
conversion_rate_pct_change_3yr_lag_tr = e.conversion_rate_pct_change_3yr,

eviction_filings_pct_change_5yr_lag_tr = e.eviction_filings_pct_change_5yr,
evictions_pct_change_5yr_lag_tr = e.evictions_pct_change_5yr,
eviction_rate_pct_change_5yr_lag_tr = e.eviction_rate_pct_change_5yr,
eviction_filing_rate_pct_change_5yr_lag_tr = e.eviction_rate_pct_change_5yr,
conversion_rate_pct_change_5yr_lag_tr = e.conversion_rate_pct_change_5yr

FROM ev_lag_tr e
where blockgroup.tract = e.geo_id and blockgroup.year = e.year + 1;"""





"""

### Tract HH_Size is the average of the bg hh sizes for each tract

"""

ADD_AVG_HH_SIZE_TR = "ALTER TABLE evictions.tr ADD COLUMN avg_hh_size float4;"

UPDATE_AVG_HH_SIZE_TR = """
UPDATE evictions.tr set avg_hh_size = t.avg_hh_size from (
select tract, year, avg(avg_size) as avg_hh_size from
hhsize group by tract, year) t
where t.tract = tr.geo_id and t.year = tr.year;"""


UPDATE_PERMITS = """
UPDATE evictions.blockgroup set 
total_units = permits.total_units,
total_bldg = permits.total_bldg,
total_value = permits.total_value,
total_units = permits.total_units
total_bldg = permits.total_bldg
total_value = permits.total_value
total_bldg_pct_change_1yr = permits.total_bldg_pct_change_1yr,
total_bldg_pct_change_3yr = permits.total_bldg_pct_change_3yr,
total_bldg_pct_change_5yr = permits.total_bldg_pct_change_5yr,
total_bldg_avg_3yr = permits.total_bldg_avg_3yr,
total_bldg_avg_5yr = permits.total_bldg_avg_5yr,
total_units_pct_change_1yr = permits.total_units_pct_change_1yr,
total_units_pct_change_3yr = permits.total_units_pct_change_3yr,
total_units_pct_change_5yr = permits.total_units_pct_change_5yr,
total_units_avg_3yr = permits.total_units_avg_3yr,
total_units_avg_5yr = permits.total_units_avg_5yr,
total_value_pct_change_1yr = permits.total_value_pct_change_1yr,
total_value_pct_change_3yr = permits.total_value_pct_change_3yr,
total_value_pct_change_5yr = permits.total_value_pct_change_5yr,
total_value_avg_3yr = permits.total_value_avg_3yr,
total_value_avg_5yr = permits.total_value_avg_5yr
from permits where blockgroup.county = permits.geo_id and blockgroup.year = permits.year;
"""

REM_999999 = """UPDATE blockgroup set {}_pct_change_{}yr{} = max from (select year, max({}_pct_change_{}yr{}) as max 
from blockgroup where {}_pct_change_{}yr{} != 999999 group by year) as tmp
where tmp.year = blockgroup.year and {}_pct_change_{}yr{} = 999999;"""

REM_999999_ev = """UPDATE blockgroup set {}_pct_change_{}yr_lag{} = max from (select year, max({}_pct_change_{}yr_lag{}) as max 
from blockgroup where {}_pct_change_{}yr_lag{} != 999999 group by year) as tmp
where tmp.year = blockgroup.year and {}_pct_change_{}yr_lag{} = 999999;"""

'''============================================================================
    ML Incremental Cursor
============================================================================'''
CREATE_BG_CURSOR = """declare bg_cursor CURSOR WITHOUT HOLD FOR
                         select * from blockgroup bg
                        	inner join outcomes o on o.geo_id=bg.geo_id and o.year=bg.year
                        	where bg.geo_id in
                        		(select geos.geo_id from
                        			(select distinct bg.geo_id
                        				from blockgroup bg
                        		     ) geos
                        	     	order by random() limit 7500
                                );
                    """

DROP_FUNCTION_BG_CURSOR = "drop function if exists bg_get_chunk(refcursor);"

CREATE_FUNCTION_BG_CURSOR = """create function bg_get_chunk(refcursor) returns refcursor as $$
                            	begin
                            		open $1 for
                            			select * from blockgroup bg
											inner join outcomes o on o.geo_id=bg.geo_id and o.year=bg.year
											where bg.geo_id in
												(select geos.geo_id from
													(select distinct bg.geo_id
														from blockgroup bg
												     ) geos
											     	order by random() limit 7500
										        );
                            		return $1;
                            	end;
                            	$$ language plpgsql;
                            """
