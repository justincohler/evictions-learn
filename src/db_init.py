"""Clear and initialize the blockgroup table in the evictions database."""
import json
import csv
import psycopg2
import os

conn = psycopg2.connect(database="evictions"
                        , user=os.environ['DB_USER']
                        , password=os.environ['DB_PASSWORD']
                        , host=os.environ['DB_HOST']
                        , port=os.environ['DB_PORT']
                        , options=f'-c search_path=evictions')
cur = conn.cursor()

# Drop the TABLE
try:
    cur.execute("DROP TABLE evictions.blockgroup;")
    conn.commmit()
except:
    pass

# Create the TABLE
create_table_block_group = """CREATE TABLE evictions.blockgroup
(
  _id SERIAL PRIMARY KEY,
  state CHAR(2),
  geo_id CHAR(12),
  year SMALLINT,
  name VARCHAR(10),
  parent_location VARCHAR(100),
  population DECIMAL,
  poverty_rate DECIMAL,
  pct_renter_occupied DECIMAL,
  median_gross_rent DECIMAL,
  median_household_income DECIMAL,
  median_property_value	DECIMAL,
  rent_burden	DECIMAL,
  pct_white	DECIMAL,
  pct_af_am DECIMAL,
  pct_hispanic DECIMAL,
  pct_am_ind DECIMAL,
  pct_asian DECIMAL,
  pct_nh_pi DECIMAL,
  pct_multiple DECIMAL,
  pct_other DECIMAL,
  renter_occupied_households DECIMAL,
  eviction_filings DECIMAL,
  evictions DECIMAL,
  eviction_rate DECIMAL,
  eviction_filing_rate DECIMAL,
  imputed	BOOLEAN,
  subbed BOOLEAN
);"""

idx_state_year = "CREATE INDEX idx_state_year ON evictions.blockgroup (state, year);"
idx_year = "CREATE INDEX idx_year ON evictions.blockgroup (year);"
idx_state = "CREATE INDEX idx_state ON evictions.blockgroup (state);"
idx_evictions = "CREATE INDEX idx_evictions ON evictions.blockgroup(evictions);"

cur.execute(create_table_block_group)
cur.execute(idx_state_year)
cur.execute(idx_year)
cur.execute(idx_state)
cur.execute(idx_evictions)

# INSERT all rows from dump
with open('C:/Users/Justin Cohler/output.csv', 'r') as f:
    copy_sql = """COPY evictions.blockgroup(
    state, geo_id, year, name, parent_location,population, poverty_rate, pct_renter_occupied,
    median_gross_rent, median_household_income, median_property_value, rent_burden,
    pct_white, pct_af_am, pct_hispanic, pct_am_ind, pct_asian, pct_nh_pi, pct_multiple,
    pct_other, renter_occupied_households, eviction_filings, evictions, eviction_rate,
    eviction_filing_rate, imputed, subbed)
    FROM stdin WITH CSV HEADER DELIMITER as ','
    """
    cur.copy_expert(sql=copy_sql, file=f)

conn.commit()
