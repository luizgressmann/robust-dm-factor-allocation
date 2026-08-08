# Processed Data

This directory contains cleaned index levels and calculated monthly return series used in the analysis.

The processed data is generated from the raw MSCI files by `notebooks/01_data_preparation.ipynb`:

- `monthly_index_levels.csv`: aligned monthly index levels
- `monthly_returns_full.csv`: monthly returns with each series' available history
- `monthly_returns_common.csv`: complete common sample used for portfolio comparisons

Processed datasets derived from MSCI index data are not included in the public repository. See `../../DATA_NOTICE.md`.
