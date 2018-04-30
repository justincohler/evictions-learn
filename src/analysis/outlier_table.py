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
		query_high = '''SELECT count({}), avg({}) FROM (SELECT {} from evictions.blockgroup) as t1
				   WHERE {} > ((SELECT avg({}) FROM evictions.blockgroup) + 
				   (SELECT stddev({}) FROM evictions.blockgroup) * 3);'''.format(col, col, col, col, col, col)

		query_low = '''SELECT count({}), avg({}) FROM (SELECT {} from evictions.blockgroup) as t1
				   WHERE {} > ((SELECT avg({}) FROM evictions.blockgroup) - 
				   (SELECT stddev({}) FROM evictions.blockgroup) * 3);'''.format(col, col, col, col, col, col)

		cur.execute(query_high)
		row = [col] + list(cur.fetchone())
		cur.execute(query_low)
		row += list(cur.fetchone())
		table.append(row)

	df = pd.DataFrame(table)
	df.columns = ["variable", "count high", "avg high", "count low", "avg low"]

	return df

