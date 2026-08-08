# Raw Data

This project expects monthly index-level files obtained independently from the [MSCI Index Data Search](https://www-cdn.msci.com/web/msci/index-tools/end-of-day-index-data-search).

The underlying files are not included in this repository. Anyone using the pipeline is responsible for obtaining the data and complying with the applicable terms. See `../../DATA_NOTICE.md`.

The required index series, data settings and relevant metadata are documented in `data/metadata/index_metadata.csv`.

Place the downloaded files in this directory with the following names:

- `msci_world_usd_net.xls`
- `msci_world_enhanced_value_usd_net.xls`
- `msci_world_momentum_usd_net.xls`
- `msci_world_quality_usd_net.xls`
- `msci_world_minimum_volatility_usd_net.xls`
- `msci_world_equal_weighted_usd_net.xls`

The files should contain monthly USD net return index levels. They are read by `notebooks/01_data_preparation.ipynb`.
