import psycopg2
import os
import json
import pandas as pd
import sys
sys.path.insert(0, '/Users/alenastern/Documents/Spring2018/Machine_Learning/evictions-learn/src/')
from db_init import db_connect
def outlier_table(cur = db_connect()[0]):
    cols = ["population", "poverty_rate", "pct_renter_occupied", "median_gross_rent", "median_household_income",
            "median_property_value", "rent_burden", "pct_white", "pct_af_am", "pct_hispanic", "pct_am_ind",
            "pct_asian", "pct_nh_pi", "pct_multiple", "pct_other", "renter_occupied_households", "eviction_filings",
            "evictions", "eviction_rate", "eviction_filing_rate"]

    table = []
    for col in cols:
        cur.execute("ROLLBACK;")
        #drop temp table if it exists already
        try:
            cur.execute("DROP TABLE tmp;")
        except:
            pass

        #create tmp table for vairable
        tmp = '''CREATE TEMP TABLE tmp as SELECT avg({}) as mean, stddev({})*3 as out_there 
                FROM evictions.blockgroup;'''.format(col, col)

        #identify count and average of high and low outliers
        query_high = "SELECT count({}), avg({}) FROM evictions.blockgroup WHERE {} > (select mean + out_there from tmp);".format(col, col, col)
        query_low = "SELECT count({}), avg({}) FROM evictions.blockgroup WHERE {} < (select mean - out_there from tmp);".format(col, col, col)
        
        #execute queries and build output table
        cur.execute(tmp)
        cur.execute(query_high)
        row = [col] + list(cur.fetchone())
        cur.execute(query_low)
        row += list(cur.fetchone())
        table.append(row)

    df = pd.DataFrame(table)
    df.columns = ["variable", "count high", "avg high", "count low", "avg low"]
    df.to_csv("outlier_table.csv")

    return df

