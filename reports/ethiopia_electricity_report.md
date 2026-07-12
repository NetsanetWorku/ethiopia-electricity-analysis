# Introduction

This report presents an analysis of Ethiopia's electricity sector using publicly available data from the World Bank. Two key indicators are examined: electric power consumption per capita (kWh) and access to electricity as a percentage of the total population. The analysis covers all available years and aims to identify long-term trends that can inform policy discussion and academic research on Ethiopia's energy development trajectory.


# Data Sources

All data were obtained from the **World Bank Open Data** platform via the World Bank JSON REST API (`https://api.worldbank.org/v2`).

| Field | Detail |
|-------|--------|
| Country | Ethiopia (country code: `ET`) |
| Indicator 1 | Electric Power Consumption — `EG.USE.ELEC.KH.PC` |
| Indicator 2 | Access to Electricity — `EG.ELC.ACCS.ZS` |
| Access Date | 2026-07-13 |

**Full citation:**

> World Bank (various years). *Electric power consumption (kWh per capita)* [EG.USE.ELEC.KH.PC] and *Access to electricity (% of population)* [EG.ELC.ACCS.ZS]. Washington, D.C.: World Bank. Retrieved 2026-07-13 from https://data.worldbank.org/indicator/EG.USE.ELEC.KH.PC and https://data.worldbank.org/indicator/EG.ELC.ACCS.ZS.


# Methodology

The analysis pipeline consists of four sequential steps:

1. **Data Collection** — Annual indicator values for Ethiopia were fetched from the World Bank API using the `data_collector` module. Raw responses were saved as CSV files in `data/raw/` for reproducibility. If the API was unreachable, local fallback files were used.

2. **Data Cleaning** — The `data_cleaner` module merged the two indicator datasets on the `year` column (outer join), renamed columns to human-readable names, cast `year` to integer and value columns to float, dropped rows where both indicator values were missing, and sorted data in ascending year order. The cleaned dataset was saved to `data/processed/ethiopia_electricity_cleaned.csv`.

3. **Trend Analysis** — The `trend_analyzer` module computed descriptive statistics (mean, median, min, max, standard deviation) and year-over-year percentage changes for each indicator. Where at least five non-NaN data points were available, a linear regression (using `scipy.stats.linregress`) was fitted to quantify the overall trend slope. Summary statistics were saved to `data/processed/ethiopia_electricity_summary.csv`.

4. **Visualization** — The `visualizer` module produced line charts for each indicator over time, with trend lines overlaid where regression was available. Charts were saved as PNG files in the `charts/` directory.


# Key Findings

## Electric Power Consumption (kWh per capita)

- **Mean:** 47.8851

- **Trend Slope:** 2.636169 per year

- **Trend Direction:** increasing

- **Peak Year:** 2023



## Access to Electricity (% of population)

- **Mean:** 30.1125

- **Trend Slope:** 2.104826 per year

- **Trend Direction:** increasing

- **Peak Year:** 2022




# Visualizations


## Electric Power Consumption (kWh per capita)


![Electric Power Consumption (kWh per capita)](../charts/consumption_kwh_per_capita.png)


## Access to Electricity (% of population)


![Access to Electricity (% of population)](../charts/access_pct_of_population.png)



# Conclusion

This analysis provides a data-driven overview of Ethiopia's electricity sector over the available historical period. The findings reveal the trajectory of both per capita electricity consumption and population-level access to electricity, highlighting long-term trends driven by economic growth, infrastructure expansion, and rural electrification efforts. The linear trend analysis offers a quantitative basis for projecting future developments. These insights can support evidence-based policy decisions aimed at achieving universal electricity access and sustainable energy consumption patterns in Ethiopia.
