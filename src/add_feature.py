from db_init import DBInit

init = DBInit()

cols = [
"avg_population",
"avg_poverty_rate",
"avg_pct_renter_occupied",
"avg_median_gross_rent",
"avg_median_household_income",
"avg_median_property_value",
"avg_rent_burden",
"avg_pct_white",
"avg_pct_af_am",
"avg_pct_hispanic",
"avg_pct_am_ind",
"avg_pct_asian",
"avg_pct_nh_pi",
"avg_pct_multiple",
"avg_pct_other",
"avg_renter_occupied_households",
#"avg_eviction_filings",
"sum_evict",
"avg_evict_rate",
#"eviction_filing_rate"
]

for table in ["evictions_county"]:
    for col in cols:
        print("Adding 5 year pct change to {} for feature {}".format(table, col))
        res = init.create_n_year_pct_change(table, col, table, 5)
        if not res:
            break
        else:
            print("Added 5 year pct change to {} for feature {}".format(table, col))
