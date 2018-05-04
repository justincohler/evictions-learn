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

def geo_init():
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

    cur,conn = db_connect()
    cur.execute("set schema 'evictions';")
    cur.execute("CREATE EXTENSION postgis;")
    cur.execute("create extension fuzzystrmatch;")
    cur.execute("create extension postgis_tiger_geocoder;")
    cur.execute("create extension postgis_topology;")
    cur.execute("drop function if exists exec(text);")
    cur.execute("CREATE FUNCTION exec(text) returns text language plpgsql volatile AS $f$ BEGIN EXECUTE $1; RETURN $1; END; $f$;")
    permission_str = "ALTER TABLE spatial_ref_sys OWNER TO {};".format(DB_USER)
    cur.execute(permission_str)
    cur.execute('''INSERT into spatial_ref_sys (srid, auth_name, auth_srid, proj4text, srtext) values ( 102003, 'esri', 102003, '+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs ', 'PROJCS["USA_Contiguous_Albers_Equal_Area_Conic",GEOGCS["GCS_North_American_1983",DATUM["North_American_Datum_1983",SPHEROID["GRS_1980",6378137,298.257222101]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["longitude_of_center",-96],PARAMETER["Standard_Parallel_1",29.5],PARAMETER["Standard_Parallel_2",45.5],PARAMETER["latitude_of_center",37.5],UNIT["Meter",1],AUTHORITY["EPSG","102003"]]');''')
    conn.commit()

    #"-U username --password -p 5432 -h reallylonghostnametoamazonaws.com dbname"


def census_shp(geography):
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
    shp_read = "shp2pgsql -s 102003:4326  data/tl_2010_us_{}10/tl_2010_us_{}10.shp evictions.census_{}_shp | psql {} -U {} -W {} -p {} -h {}".format(geography, geography, geography,'evictions', DB_USER, 
        DB_PASSWORD, DB_PORT, DB_HOST)
    os.system(shp_read)

def group_by_state():
    create = '''CREATE TABLE evictions_state (state CHAR(2),
       stusps10 char(2), 
       year SMALLINT,
       sum_evict FLOAT,
       avg_evict_rate float8,
       geom geometry);'''

    insert = '''INSERT INTO evictions.evictions_state (state, stusps10, year, sum_evict, avg_evict_rate, geom)
                SELECT state, stusps10, year, sum_evict, avg_evict_rate, geom from (
                    SELECT state, year, sum(evictions) as sum_evict, avg(eviction_rate) as avg_evict_rate
                    FROM evictions.blockgroup
                    GROUP BY state, year
                    ) as t1
                    JOIN (SELECT stusps10, geom FROM evictions.census_state_shp) as t2 ON t1.state = t2.stusps10;'''

    cur,conn = db_connect()
    cur.execute(create)
    cur.execute(insert)
    conn.commit()
    cur.close()

if __name__=="__main__":
    db_init()
