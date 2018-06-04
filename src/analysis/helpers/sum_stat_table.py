import psycopg2
import os
import json
import pandas as pd
from collections import OrderedDict

import sys
sys.path.insert(0, '/Users/alenastern/Documents/Spring2018/Machine_Learning/evictions-learn/src/')
from db_init import db_connect

def sum_stat_table(cur = db_connect()[0]):
	cols = ["population", "poverty_rate", "pct_renter_occupied", "median_gross_rent", "median_household_income",
			"median_property_value", "rent_burden", "pct_white", "pct_af_am", "pct_hispanic", "pct_am_ind",
			"pct_asian", "pct_nh_pi", "pct_multiple", "pct_other", "renter_occupied_households", "eviction_filings",
			"evictions", "eviction_rate", "eviction_filing_rate"]

	cur.execute('''SELECT avg(population), avg(poverty_rate), avg(pct_renter_occupied), avg(median_gross_rent), avg(median_household_income),
			avg(median_property_value), avg(rent_burden), avg(pct_white), avg(pct_af_am), avg(pct_hispanic), avg(pct_am_ind),
			avg(pct_asian), avg(pct_nh_pi), avg(pct_multiple), avg(pct_other), avg(renter_occupied_households), avg(eviction_filings),
			avg(evictions), avg(eviction_rate), avg(eviction_filing_rate) FROM evictions.blockgroup;''')
	avg_row = list(cur.fetchone())

	cur.execute('''SELECT min(population), min(poverty_rate), min(pct_renter_occupied), min(median_gross_rent), min(median_household_income),
			min(median_property_value), min(rent_burden), min(pct_white), min(pct_af_am), min(pct_hispanic), min(pct_am_ind),
			min(pct_asian), min(pct_nh_pi), min(pct_multiple), min(pct_other), min(renter_occupied_households), min(eviction_filings),
			min(evictions), min(eviction_rate), min(eviction_filing_rate) FROM evictions.blockgroup;''')
	min_row = list(cur.fetchone())

	cur.execute('''SELECT max(population), max(poverty_rate), max(pct_renter_occupied), max(median_gross_rent), max(median_household_income),
			max(median_property_value), max(rent_burden), max(pct_white), max(pct_af_am), max(pct_hispanic), max(pct_am_ind),
			max(pct_asian), max(pct_nh_pi), max(pct_multiple), max(pct_other), max(renter_occupied_households), max(eviction_filings),
			max(evictions), max(eviction_rate), max(eviction_filing_rate) FROM evictions.blockgroup;''')
	max_row = list(cur.fetchone())

	cur.execute('''SELECT stddev(population), stddev(poverty_rate), stddev(pct_renter_occupied), stddev(median_gross_rent), stddev(median_household_income),
			stddev(median_property_value), stddev(rent_burden), stddev(pct_white), stddev(pct_af_am), stddev(pct_hispanic), stddev(pct_am_ind),
			stddev(pct_asian), stddev(pct_nh_pi), stddev(pct_multiple), stddev(pct_other), stddev(renter_occupied_households), stddev(eviction_filings),
			stddev(evictions), stddev(eviction_rate), stddev(eviction_filing_rate) FROM evictions.blockgroup;''')
	stddev_row = list(cur.fetchone())

	df = pd.DataFrame(OrderedDict({"variable": cols, "avg" : avg_row, "min": min_row, "max": max_row, "std": stddev_row}))
	df.to_csv("sum_stat_table.csv")
	return df


