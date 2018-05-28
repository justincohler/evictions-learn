from db_init_local import 

# questions - do we want 3 year averages/ pct changes for non-eviction census data
# For blockgroup, tract
	# add conversion rate pct change and average features (questions for J about function)
	# create table as for eviction columns
	# drop/add lag columns
	# update lag columns from table
# Create permits, hh_size lags, hh_size at tract level
# Merge in permits, tracts
#### fix urban
# Should I drop the evictions cols from tracts/ blockgroup? No 
# Why diff # obs across tables?


init = DBInit()
evcols = [
    "eviction_filings",
    "evictions",
    "eviction_rate",
    "eviction_filing_rate",
    "conversion_rate"
]

cols = [
    "population",
    "poverty_rate",
    "pct_renter_occupied",
    "median_gross_rent",
    "median_household_income",
    "median_property_value",
    "rent_burden",
    "pct_white",
    "pct_af_am",
    "pct_hispanic",
    "pct_am_ind",
    "pct_asian",
    "pct_nh_pi",
    "pct_multiple",
    "pct_other",
    "renter_occupied_households",
    "pct_renter_occupied"
    ]

  times = [
  "_avg_5yr",
  "_pct_change_5yr",
  "_avg_3yr",
  "_pct_change_3yr"
  ]

  ev_times = [
  "_avg_5yr",
  "_pct_change_5yr",
  "_avg_3yr",
  "_pct_change_3yr"
  "_pct_change_1yr"
  ]


"""update blockgroup bg set
	population_avg_3yr=t.population_avg_3yr,
	poverty_rate_avg_3yr=t.poverty_rate_avg_3yr,
	pct_renter_occupied_3yr=t.pct_renter_occupied_3yr,
	median_gross_rent_avg_3yr=t.median_gross_rent_avg_3yr, 
	median_household_income_avg_3yr=t.median_household_income_avg_3yr,
	median_property_value_avg_3yr=t.median_property_value_avg_3yr,
	rent_burden_avg_3yr=t.rent_burden_avg_3yr,
	pct_white_avg_3yr=t.pct_white_avg_3yr,
	pct_af_am_avg_3yr=t.pct_af_am_avg_3yr,
	pct_hispanic_avg_3yr=t.pct_hispanic_avg_3yr,
	pct_am_ind_avg_3yr=t.pct_am_ind_avg_3yr,
	pct_asian_avg_3yr=t.pct_asian_avg_3yr,
	pct_nh_pi_avg_3yr=t.pct_nh_pi_avg_3yr,
	pct_multiple_avg_3yr=t.pct_multiple_avg_3yr,
	pct_other_avg_3yr=t.pct_other_avg_3yr,
	renter_occupied_households_avg_3yr=t.renter_occupied_households_avg_3yr,
	pct_renter_occupied_avg_3yr=t.pct_renter_occupied_avg_3yr,
	eviction_filings_avg_3yr=t.eviction_filings_avg_3yr,
	evictions_avg_3yr=t.evictions_avg_3yr,
	eviction_rate_avg_3yr=t.eviction_rate_avg_3yr,
	eviction_filing_rate_avg_3yr=t.eviction_filing_rate_avg_3yr
from blockgroup_3yr t 
where bg.geo_id=t.geo_id and bg.year=t.year;"""

# update conversion_rate



for table in ["blockgroup", "evictions_tract"]:
    

        print("Adding {} year pct change to {} for feature {}".format(lag, table, col))
        res = init.create_n_year_pct_change(table, col, table, lag)
        if not res:
            break
        else:
            print("Added {} year pct change to {} for feature {}".format(lag, table, col))
