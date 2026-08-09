# Limitations

Several MSCI factor series begin before their official launch dates. These earlier observations are provider backtests and may be affected by methodology and backfill bias. The common post-launch period is much shorter than the full sample.

The post-launch check uses returns observed after all selected indices had launched, but its rolling estimation windows can still contain pre-launch history. It is therefore a post-launch return check rather than a fully live-data portfolio test.

The walk-forward analysis prevents weights from using future returns directly, but it is still a historical pseudo-out-of-sample exercise. The data, factor set and model choices were available while the project was developed.

Index returns are theoretical. They do not include fund fees, trading costs or the full implementation cost of the underlying index rebalances. Portfolio-level turnover and reasonable cost assumptions should therefore be reported separately.

The reported Sharpe and Sortino ratios use a zero return target. They should not be interpreted as excess-return ratios relative to an observed USD risk-free rate.

MSCI World Equal Weighted is not a pure size factor. It combines a tilt toward smaller constituents with rebalancing, sector, country and other effects. Results should also be checked without this proxy.

The study uses one index provider, one base currency and a global developed-market aggregate. Conclusions may not carry over to other providers, regions, currencies or investable products.
