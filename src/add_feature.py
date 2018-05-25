from db_init import DBInit

init = DBInit()

cols1yr = [
    "renter_occupied_households",
    "pct_renter_occupied",
    "eviction_filings",
    "evictions",
    "eviction_rate",
    "eviction_filing_rate"
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
    "pct_renter_occupied",
    "eviction_filings",
    "evictions",
    "eviction_rate",
    "eviction_filing_rate"
]
for table in ["evictions_state", "evictions_tract", "evictions_county", "blockgroup"]:
    lags = [3, 5]
    for lag in lags:
        for col in cols:
            print("Adding {} year average to {} for feature {}".format(lag, table, col))
            res = init.create_n_year_average(table, col, table, lag)
            if not res:
                break
            else:
                print("Added {} year average to {} for feature {}".format(lag, table, col))


for table in ["blockgroup"]:
    lags = [1]
    for lag in lags:
        for col in cols1yr:
            print("Adding {} year pct change to {} for feature {}".format(lag, table, col))
            res = init.create_n_year_pct_change(table, col, table, lag)
            if not res:
                break
            else:
                print("Added {} year pct change to {} for feature {}".format(lag, table, col))
