from db_init_local import DBInit

init = DBInit()

evcols = [
    "eviction_filings",
    "evictions",
    "eviction_rate",
    "eviction_filing_rate",
    "conversion_rate"]

cols = [
    "population",
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

permits = [
"total_bldg",
"total_units",
"total_value"]

for table in ["blockgroup", "tr"]:
    lags = [1, 3, 5]
    for lag in lags:
        for col in evcols:
            print("Adding {} year average to {} for feature {}".format(lag, table, col))
            res = init.create_n_year_average(col, table, lag)
            if not res:
                break
            else:
                print("Added {} year average to {} for feature {}".format(lag, table, col))

            print("Adding {} year pct change to {} for feature {}".format(lag, table, col))
            res = init.create_n_year_pct_change(table, col, table, lag)
            if not res:
                break
            else:
                print("Added {} year pct change to {} for feature {}".format(lag, table, col))


for table in ["blockgroup", "tr"]:
    lags = [5]
    for lag in lags:
        for col in cols:
            print("Adding {} year average to {} for feature {}".format(lag, table, col))
            res = init.create_n_year_average(col, table, lag)
            if not res:
                break
            else:
                print("Added {} year average to {} for feature {}".format(lag, table, col))

            print("Adding {} year pct change to {} for feature {}".format(lag, table, col))
            res = init.create_n_year_pct_change(table, col, table, lag)
            if not res:
                break
            else:
                print("Added {} year pct change to {} for feature {}".format(lag, table, col))


for table in ["permits"]:
    lags = [1, 3, 5]
    for lag in lags:
        for col in permits:
            print("Adding {} year average to {} for feature {}".format(lag, table, col))
            res = init.create_n_year_average(col, table, lag)
            if not res:
                break
            else:
                print("Added {} year average to {} for feature {}".format(lag, table, col))

            print("Adding {} year pct change to {} for feature {}".format(lag, table, col))
            res = init.create_n_year_pct_change(table, col, table, lag)
            if not res:
                break
            else:
                print("Added {} year pct change to {} for feature {}".format(lag, table, col))



