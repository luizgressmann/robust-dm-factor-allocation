# Robust Developed-Market Factor Allocation

A historical walk-forward study of risk-based allocation across developed-market equity factor indices.

## Research Question

**Do risk-based allocation methods provide a robust improvement over equal weighting when combining developed-market equity factor indices under walk-forward estimation, portfolio constraints and transaction costs?**

Portfolio optimization can improve diversification in theory, but the necessary risk estimates are themselves uncertain. This project studies that trade-off in a small universe of developed-market equity factor indices.

The main comparison is against a simple equal-weighted portfolio. It tests whether using estimated information about the historical risk structure produces an improvement that remains stable once weights are estimated only from past data and implementation assumptions are varied.

This is related to the broader estimation-error problem studied by DeMiguel, Garlappi and Uppal (2009). A more direct motivation comes from Dichtl, Drobetz and Wendt (2021), who compare different ways of combining factor portfolios and find that more sophisticated allocation methods do not reliably outperform an equal-weighted factor portfolio.

## Portfolio Universe

The allocation universe consists of four long-only MSCI World indices representing value, momentum, quality and size sleeves:

| Sleeve | Index |
| --- | --- |
| Value | MSCI World Enhanced Value |
| Momentum | MSCI World Momentum |
| Quality | MSCI World Quality |
| Size | MSCI World Small Cap |

MSCI World is used as a broad-market benchmark and is excluded from the allocation universe.

The indices are treated as long-only factor/style sleeves rather than academic long-short factor returns. In particular, MSCI World Small Cap is used as a size sleeve and should not be interpreted as an SMB factor.

The underlying historical index data were obtained from the MSCI End of Day Index Data Search. The source files and MSCI-derived outputs are not included in the public repository.

## Allocation Strategies

The project compares five long-only allocation strategies under the same constraints and walk-forward schedule. Equal Weight serves as the estimation-free baseline. The remaining methods require estimated risk inputs, while the covariance-aware methods also depend on the estimated relationship between the sleeves.

### Equal Weight

A 1/N allocation that does not require estimated volatility or covariance inputs. It serves as the estimation-free baseline.

### Inverse Volatility

Weights are inversely proportional to estimated standalone volatility. The method uses individual risk estimates but does not use the correlation structure between sleeves.

### Sample GMV

A global minimum-variance portfolio estimated from the historical sample covariance matrix. The approach follows the mean-variance portfolio framework introduced by Markowitz (1952).

### OAS-GMV

The same minimum-variance problem is solved using an Oracle Approximating Shrinkage covariance estimate instead of the raw sample covariance matrix.

The OAS estimator follows Chen et al. (2010) and provides a regularized alternative to the sample covariance estimate.

### Equal Risk Contribution

Weights are chosen so that the four sleeves contribute approximately equally to estimated portfolio variance. The method is based on the Equal Risk Contribution framework discussed by Maillard, Roncalli and Teiletche (2010).

## Walk-Forward Design

Portfolio weights are estimated using only observations available at the end of each estimation period and are first applied to the following month's return.

The timing is therefore:

```text
returns through month t
        ↓
estimate portfolio weights
        ↓
apply weights from month t+1
```

Between rebalances, portfolio weights drift with the underlying index returns.

The main specification uses:

| Parameter | Main specification |
| --- | ---: |
| Estimation window | 120 months |
| Rebalancing | Quarterly |
| Maximum sleeve weight | 40% |
| Portfolio constraints | Long-only, fully invested |
| Transaction cost | 10 bps one-way turnover cost |
| Implementation lag | 1 month |

Turnover is measured against the drifted pre-trade portfolio rather than against the previous target weights. Transaction costs are deducted when the portfolio is rebalanced.

This is a historical walk-forward study rather than a true live out-of-sample experiment because the index universe and research design were chosen with the historical dataset already available.

## Evaluation

The main portfolio comparison reports:

- annualized return,
- annualized volatility,
- Sharpe ratio,
- Sortino ratio,
- maximum drawdown,
- active return,
- tracking error,
- information ratio,
- turnover,
- transaction-cost drag.

MSCI World is used for the benchmark-relative measures.

The project does not forecast factor returns or attempt to time factors. The portfolio methods differ mainly in how they use historical risk and covariance information to construct the allocation.

## Robustness Analysis

The robustness analysis focuses on OAS-GMV and changes several implementation assumptions.

### Estimation and portfolio settings

| Parameter | Values |
| --- | --- |
| Estimation window | 60, 120 months |
| Rebalancing | Monthly, quarterly, annually |
| Maximum sleeve weight | 30%, 40%, 50% |

The different specifications are evaluated over a common period so that the comparison is not driven by different starting dates.

### Transaction costs

The main OAS-GMV specification is also tested with:

- 0 bps,
- 10 bps,
- 25 bps

of one-way turnover cost.

### Post-launch period

Parts of the Momentum and Quality index histories were backtested by the index provider before their official launch dates. The robustness analysis therefore also reports results for a post-launch evaluation period.

### Bootstrap

A circular block bootstrap with 12-month blocks is used to obtain a rough confidence interval for realized annualized active returns.

The bootstrap resamples the realized active-return series. It does not re-estimate the complete portfolio strategy inside every bootstrap draw, so the interval should be interpreted accordingly.

## Implementation

The project is implemented as a small Python package rather than entirely inside notebooks.

The reusable package code handles:

- data validation and return preparation,
- covariance estimation,
- portfolio optimization,
- walk-forward accounting,
- performance metrics,
- bootstrap inference,
- plotting and result export.

The notebooks are used for the research workflow:

1. `01_data_preparation.ipynb` — import, clean and align the raw index data
2. `02_exploratory_analysis.ipynb` — inspect sample coverage, return properties and correlations
3. `03_walk_forward_analysis.ipynb` — compare the five allocation strategies
4. `04_robustness_analysis.ipynb` — test OAS-GMV sensitivity and bootstrap intervals

Automated tests cover portfolio constraints, timing conventions, performance calculations and the synthetic demonstration pipeline.

## Synthetic Demo

Because the underlying MSCI data are licensed, the repository includes a deterministic synthetic example that runs through the same backtesting code without requiring the original market data.

From the repository root:

```bash
python -m pip install -e .
python -m robust_dm_factor_allocation.demo --output-dir results/demo
```

The demo generates 180 synthetic monthly observations, applies all five portfolio methods and exports example tables and figures.

The synthetic results are only a reproducibility and implementation check. Their strategy rankings and metric values are properties of the generated sample and should not be interpreted as evidence of historical or expected investment performance.

## Running the Empirical Analysis

Place the required MSCI source files in `data/raw/`. The expected filenames and source metadata are documented in the repository.

Install the research dependencies:

```bash
python -m pip install -e ".[dev,research]"
python -m jupyter lab
```

Run the notebooks in numerical order.

## Tests

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check --exclude notebooks .
coverage run -m unittest discover -s tests -v
coverage report --fail-under=85
```

## Scope and Limitations

The project is an empirical study of portfolio construction, not a trading strategy or evidence of investable alpha.

The analysis uses a small investment universe and simplified transaction costs. Taxes, fund fees, market impact and detailed implementation constraints are not modeled.

Some underlying index histories predate their official launch dates, and the research design was developed with knowledge of the historical dataset.

With only four portfolio sleeves and relatively long estimation windows, the project should also not be interpreted as a general test of whether covariance shrinkage is superior to sample covariance estimation. OAS-GMV is included as one regularized portfolio specification within this particular setting.

The main question is deliberately narrower: whether additional estimation and optimization produce improvements that are stable enough to justify the additional portfolio complexity in this historical walk-forward setting.

## Data and License

The source code is released under the MIT License.

MSCI source data, processed MSCI returns and MSCI-derived empirical outputs are not distributed with the repository and are not covered by the software license.

The public synthetic demo does not use MSCI data.

## References

Markowitz, H. (1952). *Portfolio Selection.* *The Journal of Finance, 7*(1), 77–91. DOI: 10.1111/j.1540-6261.1952.tb01525.x.

DeMiguel, V., Garlappi, L. & Uppal, R. (2009). *Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?* *The Review of Financial Studies, 22*(5), 1915–1953. DOI: 10.1093/rfs/hhm075.

Dichtl, H., Drobetz, W. & Wendt, V.-S. (2021). *How to build a factor portfolio: Does the allocation strategy matter?* *European Financial Management, 27*(1), 20–58. DOI: 10.1111/eufm.12264.

Chen, Y., Wiesel, A., Eldar, Y. C. & Hero, A. O. (2010). *Shrinkage Algorithms for MMSE Covariance Estimation.* *IEEE Transactions on Signal Processing, 58*(10), 5016–5029. DOI: 10.1109/TSP.2010.2053029.

Maillard, S., Roncalli, T. & Teiletche, J. (2010). *On the Properties of Equally-Weighted Risk Contributions Portfolios.* *The Journal of Portfolio Management, 36*(4), 60–70. DOI: 10.3905/jpm.2010.36.4.060.

## Author

Luiz Gressmann
