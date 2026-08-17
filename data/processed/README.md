# Processed data

Notebook 01 writes three local files here:

- `monthly_index_levels.csv`
- `monthly_returns_full.csv`
- `monthly_returns_common.csv`

The common file puts Value, Momentum, Quality, Small Cap and the MSCI World benchmark on the same complete monthly sample. The other notebooks read this file through the package loader.

These CSV files are derived from licensed inputs and are ignored by Git. Re-run Notebook 01 after changing a raw workbook or the metadata snapshot.
