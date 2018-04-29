import seaborn
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd

def plot_corr(cursor):

    query = """SELECT year,
                sum(population), avg(poverty_rate),
                avg(pct_renter_occupied), avg(median_gross_rent),
                avg(median_household_income), avg(median_property_value),
                avg(rent_burden), avg(pct_white), avg(pct_af_am),
                avg(pct_hispanic), avg(pct_am_ind), avg(pct_asian),
                avg(pct_nh_pi), avg(pct_multiple), avg(pct_other),
                sum(renter_occupied_households),
                sum(eviction_filings), sum(evictions),
                avg(eviction_rate), avg(eviction_filing_rate)
            FROM evictions.blockgroup
            GROUP BY year order by year ASC;"""

    cursor.execute(query)
    res = cursor.fetchall()

    columns = ("Year", "Population", "Poverty Rate", "% Renter Occupied"
                                                , "Median Gross Rent", "Medium Household Income", "Median Property Value"
                                                , "Rent Burden", "% White", "% Af Am", "% Hispanic", "% Am Ind"
                                                , "% Asian", "% NH PI", "% Multiple", "% Other", "Renter Occupied Households"
                                                , "Eviction Filings", "Evictions", "Eviction Rate", "Eviction Filing Rate" )

    df = pd.DataFrame.from_records(res, columns=columns)
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))
    plt.xticks(rotation=90)

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
               square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.yticks(rotation=0)
