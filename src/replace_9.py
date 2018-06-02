from db_init_local import DBInit

init = DBInit()

"""
Removes instances of 999999 from data inserted in percentage change calculations to avoid skew. Replaces
999999 entries with the maximum value for the feature in the given year.

Note that no -999999 instances were present in our data, so we did not need to replcae these values.

"""
cols = ["population",
		"poverty_rate",
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
	    "pct_renter_occupied",
	    "avg_hh_size"]

evcols = [
    "eviction_filings",
    "evictions",
    "eviction_rate",
    "eviction_filing_rate",
    "conversion_rate"
    ]

# replace instances in demographic features, which only has 5-year data
lag = 5
for col in ["avg_hh_size"]:
	for tr in [True, False]:
	    print("removing 999999 from {} for {} yr pct change where tract is {}".format(col, lag, tr))
	    res = init.rem_9(lag, col, tr)
	    if not res:
	        break
	    else:
	        print("removed 999999 from {} for {} yr pct change".format(col, lag))

# replace instances in evictions features, which has 1, 3, and 5-year data
lags = [1, 3, 5]
for col in evcols:
	for lag in lags:
		for tr in [True, False]:
		    print("removing 999999 from {} for {} yr pct change where tract is {}".format(col, lag, tr))
		    init.rem_9_ev(lag, col, tr)
		    if not res:
		        break
		    else:
		        print("removed 999999 from {} for {} yr pct change".format(col, lag))




