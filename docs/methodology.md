# Methodology

The analysis uses monthly simple returns calculated from USD net return index levels. Portfolio comparisons use the common period for which all series are available.

The factor allocation contains Value, Momentum, Quality, Minimum Volatility and Equal Weight. MSCI World is kept outside the allocation and used as the market benchmark. Equal Weight is only an approximate size tilt.

Static full-sample allocations are descriptive. In the walk-forward analysis, each weight is estimated with data available up to month t and applied to returns in month t + 1. Total-return covariance is used for the main portfolio view. Active returns relative to MSCI World are used to report tracking error, information ratio and market-relative diversification.

The optimized portfolios are compared with simple rules such as equal weighting and inverse-volatility weighting. Robustness checks vary the estimation window, rebalancing frequency and portfolio constraints. Results from the full set of specifications should be reported rather than selecting the best one after the test.

Sharpe and Sortino ratios use a zero return target because a separate monthly risk-free series is not included in the dataset.

Definitions and formulas for the individual calculations are included in the relevant notebooks.
