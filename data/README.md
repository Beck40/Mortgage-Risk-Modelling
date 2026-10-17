# Data Guide

Due to confidentiality and file size limits, no raw data or generated synthetic portfolios are committed to this repository. To run the synthetic data generator, you must manually download the public macroeconomic datasets.

## Directory Structure
Ensure your local `data` directory matches this structure:

```text
data/
├── raw_macro/
│   ├── fca-mlar-long-run.xlsx
│   ├── MOJ_LA_CSV.csv
│   ├── nomis_claimant_counts.csv
│   ├── nomis_model_estimate_unemployment.csv
│   └── pp-<year>.csv
├── synthetic_portfolio/
│   └── .gitkeep
└── README.md
```

##  Required Public Datasets

Ministry of Justice (MoJ) - Possession Claims 
- Source: https://www.gov.uk/government/statistics/mortgage-and-landlord-possession-statistics-april-to-june-2026
- Action: Download the "Mortgage and landlord possession statistical data" ZIP file. Extract LA_CSV.csv and rename it to MOJ_LA_CSV.csv.

ONS / Nomis - Local Unemployment & Claimant Counts
- Source URL: https://www.nomisweb.co.uk/
- Extract the 'model-based estimates of unemployment' at the Local Authority District level. Save as nomis_model_estimate_unemployment.csv.
- Extract the 'Claimant count by sex and age' at the Local Authority level. Save as nomis_claimant_counts.csv.

FCA - Mortgage Lending and Administration Return (MLAR)
- Source URL: https://www.fca.org.uk/data/mortgage-lending-statistics
- Action: Download the Long-run historical MLAR statistics spreadsheet. Save as fca-mlar-long-run.xlsx.

HM Land Registry - Price Paid Data
- Source URL: https://www.gov.uk/government/statistical-data-sets/price-paid-data-yearly-file
- Action: Download the yearly files from 2019 to 2026.