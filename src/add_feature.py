from db_init import DBInit

init = DBInit()

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
    "eviction_filings",
    "evictions",
    "eviction_rate",
    "eviction_filing_rate"
]

for col in cols:
    res = init.create_n_year_pct_change(col, "blockgroup", 5)
    if not res:
        break
    else:
        print("Added 5 year pct change for {}".format(col))