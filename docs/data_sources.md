# Data Sources

## MSCI Index Data

Monthly index-level data is obtained from the MSCI Index Data Search.

The analysis uses the following developed-market indices:

- MSCI World Index
- MSCI World Enhanced Value Index
- MSCI World Momentum Index
- MSCI World Quality Index
- MSCI World Minimum Volatility Index
- MSCI World Equal Weighted Index

All series use monthly USD net return index levels.

Raw and processed MSCI data files are excluded from the public repository due to licensing restrictions.

Detailed metadata for each series, including launch and download dates, is stored in `data/metadata/index_metadata.csv`.

Some histories begin before the official index launch. These observations are treated as backtested history and are discussed separately from the common live period.

## Data Processing

The raw index-level files are processed in `notebooks/01_data_preparation.ipynb`.

The notebook creates:

- `data/processed/monthly_index_levels.csv`
- `data/processed/monthly_returns_full.csv`
- `data/processed/monthly_returns_common.csv`

The import checks and processing steps are documented in the notebook.
