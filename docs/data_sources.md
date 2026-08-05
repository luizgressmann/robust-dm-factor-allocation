# Data Sources

## MSCI Index Data

Monthly index-level data is obtained from the MSCI Index Data Search.

The analysis uses following DM indices:

- MSCI World Index
- MSCI World Enhanced Value Index
- MSCI World Momentum Index
- MSCI World Quality Index
- MSCI World Minimum Volatility Index
- MSCI World Equal Weighted Index

The index series are downloaded constantly using NET retrun-type, monthly frequency and USD currency.

Raw and processed MSCI data files are excluded from the public repository due to licensing restrictions.

Detailed metadata for eac hseries is stored in
`data/metadata/index_metadata.csv`.

## Data Processing

The raw index-level files are processed in
`notebooks/01_data_preparation.ipynb`.

Further infos and the processing pipeline are described as comments within the notebook.