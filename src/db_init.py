"""Clear and initialize the blockgroup table in the evictions database."""
import json
import csv
import psycopg2
import os


def db_connect():

    if 'DB_USER' not in os.environ:
        resources_dir = os.path.dirname(__file__)
        secrets_file = os.path.join(resources_dir, '../resources/secrets.json')
        with open(secrets_file) as f:
            env = json.load(f)
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
    cur,conn = db_connect()
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

    idx_state_year = "CREATE INDEX idx_state_year ON evictions.blockgroup (state, year);"
    idx_year = "CREATE INDEX idx_year ON evictions.blockgroup (year);"
    idx_state = "CREATE INDEX idx_state ON evictions.blockgroup (state);"
    idx_evictions = "CREATE INDEX idx_evictions ON evictions.blockgroup (evictions);"


    print("Creating table...")
    cur.execute(create_table_block_group)
    print("Creating indexes...")
    cur.execute(idx_state_year)
    cur.execute(idx_year)
    cur.execute(idx_state)
    cur.execute(idx_evictions)

    conn.commit()
    print("Tables & indexes committed.")

    # INSERT all rows from dump
    print("\n\nCopying CSV data to evictions db...")

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
    print("Committed records.")

if __name__=="__main__":
    db_init()
