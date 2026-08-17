# Methodology

This page explains the main choices behind the notebooks.

## Data

The analysis uses monthly USD net return index levels for:

- MSCI World Enhanced Value
- MSCI World Momentum
- MSCI World Quality
- MSCI World Small Cap
- MSCI World as the benchmark

Value, Momentum and Quality are the main style factors. Small Cap is included as the size sleeve. A broad equal-weighted index is not used as a size proxy because it still holds large- and mid-cap names and adds an equal-weighting effect of its own.

The raw workbooks are converted to simple monthly returns:

```text
r[t] = level[t] / level[t-1] - 1
```

For each series, Notebook 01 keeps the last valid observation in every calendar month. Before writing a CSV, it checks the date range, missing months, duplicate dates and non-positive levels.

## Covariance estimates

The sample covariance uses the usual unbiased estimate with `T - 1` in the denominator.

The OAS estimate shrinks the maximum-likelihood covariance matrix toward a scaled identity matrix:

```text
Sigma[OAS] = (1 - rho) * S[ML] + rho * mu * I
mu         = trace(S[ML]) / N
```

The code uses the finite-sample OAS shrinkage intensity from Chen et al. (2010), equation 23. This is not Ledoit-Wolf shrinkage.

## Portfolio methods

Every portfolio is long-only, fully invested and subject to the configured maximum weight.

### Equal weight

Each sleeve receives `1 / N`. This is the main baseline.

### Inverse volatility

Weights are proportional to inverse sample volatility. Correlations do not enter the rule.

### Minimum variance

The optimizer solves:

```text
minimize    w' Sigma w
subject to  sum(w) = 1
            0 <= w[i] <= cap
```

`sample_gmv` uses the sample covariance and `oas_gmv` uses OAS. Projected gradient descent solves the problem on the capped simplex. The function only returns weights after a convergence check.

### Equal risk contribution

For covariance matrix `Sigma`, normalized risk contributions are:

```text
RC[i] = w[i] * (Sigma w)[i] / (w' Sigma w)
```

ERC makes all four contributions equal. If the exact solution breaks the cap, `erc_weights` raises `ExactERCCapError`. That run is then marked as unavailable.

## Walk-forward timing

At an estimation month `t`, the optimizer sees only the trailing window ending at `t`. The new weights are first applied to the return in `t + 1`.

```text
returns through t -> estimate weights -> apply from t+1
```

Weights drift with the sleeve returns between rebalances. At a rebalance, one-way turnover is measured against the drifted pre-trade weights:

```text
turnover = 0.5 * sum(abs(target - pretrade))
cost     = turnover * cost_bps / 10,000
```

The cost is deducted from the first return under the new target weights.

## Main setup and checks

The default specification is:

- 120-month estimation window
- quarterly rebalancing
- one-month implementation lag
- 40% maximum sleeve weight
- 10 basis points of one-way turnover cost

All five portfolio methods use this setup. Notebook 04 then changes the OAS-GMV settings only:

- 60 or 120 months of history
- monthly, quarterly or annual rebalancing
- 30%, 40% or 50% cap
- 0, 10 or 25 basis points of cost in a separate cost check

The evaluation sample starts after the longest warm-up so the 60- and 120-month runs cover the same return months.

A circular block bootstrap with 12-month blocks gives a rough interval for annualized mean active return. It resamples the active-return series but does not rerun the optimizer for each draw.

## Launch dates and interpretation

Momentum and Quality contain provider-backtested observations from before their launch dates. The Small Cap file starts one business day before its formal launch. The notebooks also show post-launch periods. These are still not live tests because I chose the indices and setup with hindsight.

The results leave out taxes, market impact, fund fees and benchmark-tracking limits. I treat them as a comparison of portfolio rules, not as an investable performance record.

## References

- Harry Markowitz, "Portfolio Selection," *The Journal of Finance* 7(1), 1952, pp. 77-91. [doi:10.1111/j.1540-6261.1952.tb01525.x](https://doi.org/10.1111/j.1540-6261.1952.tb01525.x)
- Yilun Chen, Ami Wiesel, Yonina C. Eldar and Alfred O. Hero III, "Shrinkage Algorithms for MMSE Covariance Estimation," *IEEE Transactions on Signal Processing* 58(10), 2010, pp. 5016-5029. [doi:10.1109/TSP.2010.2053029](https://doi.org/10.1109/TSP.2010.2053029)
- Sebastien Maillard, Thierry Roncalli and Jerome Teiletche, "The Properties of Equally Weighted Risk Contribution Portfolios," *The Journal of Portfolio Management* 36(4), 2010, pp. 60-70. [doi:10.3905/jpm.2010.36.4.060](https://doi.org/10.3905/jpm.2010.36.4.060)
- Dimitris N. Politis and Joseph P. Romano, "A Circular Block-Resampling Procedure for Stationary Data," in *Exploring the Limits of Bootstrap*, John Wiley & Sons, 1992, pp. 263-270.
