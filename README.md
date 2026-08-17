# Robust Developed-Market Factor Allocation

[![CI](https://github.com/luizgressmann/robust-dm-factor-allocation/actions/workflows/ci.yml/badge.svg)](https://github.com/luizgressmann/robust-dm-factor-allocation/actions/workflows/ci.yml)

I built this project to see how different allocation rules behave across developed-market equity factors. The focus is on Value, Momentum and Quality. MSCI World Small Cap adds a direct size sleeve instead of using a broad equal-weighted index as a proxy.

The backtest runs month by month and only uses past data. Weights estimated at one month-end are first used in the following month. Turnover costs are deducted at each rebalance.

The MSCI files are licensed and are not included. A small synthetic example lets the project run without them. Its results stay local.

## Factor set

| Sleeve | Index | Role |
| --- | --- | --- |
| Value | MSCI World Enhanced Value | Main style factor |
| Momentum | MSCI World Momentum | Main style factor |
| Quality | MSCI World Quality | Main style factor |
| Small Cap | MSCI World Small Cap | Size sleeve |

MSCI World is used as the benchmark. It is not included in the optimizer.

## Portfolio rules

The package compares five long-only portfolios:

- equal weight
- inverse volatility
- sample-covariance minimum variance
- OAS-shrinkage minimum variance
- equal risk contribution

The main run uses a 120-month window, quarterly rebalancing, a 40% weight cap and 10 basis points of one-way turnover cost. Notebook 04 varies the window, rebalance interval, cap and cost for OAS-GMV only.

## Synthetic example

Run the synthetic example from the repository root:

```powershell
python -m pip install -e .
python -m robust_dm_factor_allocation.demo --output-dir results/demo
```

The example uses synthetic returns with a fixed seed. It runs the same backtest code, but it is not a performance estimate. It writes two CSV files and two charts to the ignored `results/demo/` folder.

## Tests

```powershell
python -m pip install -e ".[dev]"
ruff check .
ruff format --check --exclude notebooks .
coverage run -m unittest discover -s tests -v
coverage report --fail-under=85
```

CI runs these checks on the minimum and current Python versions. It also checks that committed notebooks have no saved outputs.

## Research notebooks

To run the MSCI analysis, place the five workbooks in `data/raw/`. The filenames and snapshot dates are listed in [`data/raw/README.md`](data/raw/README.md).

```powershell
python -m pip install -e ".[dev,research]"
python -m jupyter lab
```

Run the notebooks in this order:

1. [`01_data_preparation.ipynb`](notebooks/01_data_preparation.ipynb)
2. [`02_exploratory_analysis.ipynb`](notebooks/02_exploratory_analysis.ipynb)
3. [`03_walk_forward_analysis.ipynb`](notebooks/03_walk_forward_analysis.ipynb)
4. [`04_robustness_analysis.ipynb`](notebooks/04_robustness_analysis.ipynb)

Generated MSCI-based tables and charts stay local and are ignored by Git.

## Project layout

```text
src/robust_dm_factor_allocation/  package code
tests/                            unit and integration tests
notebooks/                        four research notebooks
config/config.yaml                main assumptions
data/metadata/                    source and launch-date metadata
results/README.md                local-output instructions
docs/methodology.md               formulas and implementation details
```

## Limits

This is a historical comparison, not a strategy ready to trade. Parts of the Momentum and Quality histories were backtested by the index provider. I also made the project choices after seeing the available data. Costs are simplified, and the analysis leaves out taxes, market impact and practical trading limits.

See [`DATA_NOTICE.md`](DATA_NOTICE.md) before using MSCI files or sharing derived results.

## Author and license

Luiz Gressmann | [GitHub](https://github.com/luizgressmann)

The code is released under the [MIT License](LICENSE). MSCI data and MSCI-derived results are not covered by that license.
