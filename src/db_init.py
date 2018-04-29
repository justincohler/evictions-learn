"""Clear and initialize the blockgroup table in the evictions database."""
import json
import csv
import psycopg2
import os


def db_connect():

    if 'DB_USER' not in os.environ:
        resources_dir = os.path.dirname(__file__)
        secrets_file = os.path.join(resources_dir, '../resources/secrets.json')
        env = json.load(secrets_file)
        DB_USER = env['DB_USER']
        DB_PASSWORD = env['DB_PASSWORD']
        DB_HOST = env['DB_HOST']
        DB_PORT = env['DB_PORT']
    else:
        DB_USER = os.environ['DB_USER']
        DB_PASSWORD = os.environ['DB_PASSWORD']
        DB_HOST = os.environ['DB_HOST']
        DB_PORT = os.environ['DB_PORT']

    conn = psycopg2.connect(database="evictions"
                            , user= DB_USER
                            , password=DB_PASSWORD
                            , host=DB_HOST
                            , port=DB_PORT
                            , options=f'-c search_path=evictions')
    cur = conn.cursor()

    return cur, conn

def db_init():
    cur,_ = db_connect()
    # Drop the TABLE
    try:
        cur.execute("DROP TABLE evictions.blockgroup;")
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
      population NUMERIC,
      poverty_rate NUMERIC,
      pct_renter_occupied NUMERIC,
      median_gross_rent NUMERIC,
      median_household_income NUMERIC,
      median_property_value	NUMERIC,
      rent_burden	NUMERIC,
      pct_white	NUMERIC,
      pct_af_am NUMERIC,
      pct_hispanic NUMERIC,
      pct_am_ind NUMERIC,
      pct_asian NUMERIC,
      pct_nh_pi NUMERIC,
      pct_multiple NUMERIC,
      pct_other NUMERIC,
      renter_occupied_households NUMERIC,
      eviction_filings NUMERIC,
      evictions NUMERIC,
      eviction_rate NUMERIC,
      eviction_filing_rate NUMERIC,
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

if __name__=="__main__":
    db_init()
