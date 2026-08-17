# Raw MSCI files

This folder is empty in Git. Download the monthly USD net return index-level workbooks from the [MSCI index data search](https://www-cdn.msci.com/web/msci/index-tools/end-of-day-index-data-search) and use these filenames:

| Series | MSCI code | Local filename |
| --- | ---: | --- |
| MSCI World | 990100 | `msci_world_usd_net.xls` |
| MSCI World Enhanced Value | 105868 | `msci_world_enhanced_value_usd_net.xls` |
| MSCI World Momentum | 703755 | `msci_world_momentum_usd_net.xls` |
| MSCI World Quality | 702787 | `msci_world_quality_usd_net.xls` |
| MSCI World Small Cap | 106230 | `msci_world_small_cap_usd_net.xls` |

The metadata records one fixed download snapshot. Four files run through 2026-06-30; the Small Cap workbook runs through 2026-04-30. Notebook 01 checks these endpoints against `data/metadata/index_metadata.csv`.

If you download a newer snapshot, update `last_observation` and `download_date` in the metadata file first. Do not commit the workbooks or the processed CSV files.
