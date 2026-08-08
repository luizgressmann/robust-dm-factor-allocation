import numpy as np
import pandas as pd


def _clean_series(values):
    series = pd.Series(values, copy=False).dropna().astype(float)
    if series.empty:
        raise ValueError("returns must contain at least one observation")
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError("returns must be finite")
    return series


def _return_statistic(returns, function):
    if isinstance(returns, pd.DataFrame):
        return returns.apply(function)
    return float(function(_clean_series(returns)))


def annualized_return(returns, periods_per_year=12):
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    def calculate(series):
        series = _clean_series(series)
        if (series <= -1).any():
            raise ValueError("returns must be greater than -1")
        growth = np.prod(1 + series.to_numpy())
        return growth ** (periods_per_year / len(series)) - 1

    return _return_statistic(returns, calculate)


def annualized_volatility(returns, periods_per_year=12):
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    def calculate(series):
        series = _clean_series(series)
        if len(series) < 2:
            return np.nan
        return series.std(ddof=1) * np.sqrt(periods_per_year)

    return _return_statistic(returns, calculate)


def wealth_index(returns):
    if isinstance(returns, pd.DataFrame):
        frame = returns.astype(float)
        if not np.isfinite(frame.to_numpy()).all() or (frame <= -1).any().any():
            raise ValueError("returns must be finite and greater than -1")
        return (1 + frame).cumprod()
    series = _clean_series(returns)
    if (series <= -1).any():
        raise ValueError("returns must be greater than -1")
    return (1 + series).cumprod()


def maximum_drawdown(returns):
    def calculate(series):
        series = _clean_series(series)
        if (series <= -1).any():
            raise ValueError("returns must be greater than -1")
        wealth = np.r_[1.0, np.cumprod(1 + series.to_numpy())]
        drawdowns = wealth / np.maximum.accumulate(wealth) - 1
        return drawdowns.min()

    return _return_statistic(returns, calculate)


def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=12):
    if risk_free_rate <= -1:
        raise ValueError("risk_free_rate must be greater than -1")
    periodic_rate = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    def calculate(series):
        series = _clean_series(series) - periodic_rate
        if len(series) < 2 or series.std(ddof=1) == 0:
            return np.nan
        return series.mean() / series.std(ddof=1) * np.sqrt(periods_per_year)

    return _return_statistic(returns, calculate)


def downside_deviation(returns, target_return=0.0, periods_per_year=12):
    def calculate(series):
        series = _clean_series(series)
        downside = np.minimum(series.to_numpy() - target_return, 0)
        return np.sqrt(np.mean(downside**2)) * np.sqrt(periods_per_year)

    return _return_statistic(returns, calculate)


def sortino_ratio(returns, target_return=0.0, periods_per_year=12):
    def calculate(series):
        series = _clean_series(series)
        downside = np.minimum(series.to_numpy() - target_return, 0)
        risk = np.sqrt(np.mean(downside**2))
        if risk == 0:
            return np.nan
        return (series.mean() - target_return) / risk * np.sqrt(periods_per_year)

    return _return_statistic(returns, calculate)


def _active_returns(portfolio_returns, benchmark_returns):
    portfolio = pd.Series(portfolio_returns, copy=False, name="portfolio").astype(float)
    benchmark = pd.Series(benchmark_returns, copy=False, name="benchmark").astype(float)
    aligned = pd.concat([portfolio, benchmark], axis=1, join="inner").dropna()
    if aligned.empty:
        raise ValueError("portfolio and benchmark do not overlap")
    if not np.isfinite(aligned.to_numpy()).all():
        raise ValueError("returns must be finite")
    return aligned["portfolio"] - aligned["benchmark"]


def annualized_active_return(portfolio_returns, benchmark_returns, periods_per_year=12):
    active = _active_returns(portfolio_returns, benchmark_returns)
    return float(active.mean() * periods_per_year)


def tracking_error(portfolio_returns, benchmark_returns, periods_per_year=12):
    active = _active_returns(portfolio_returns, benchmark_returns)
    if len(active) < 2:
        return np.nan
    return float(active.std(ddof=1) * np.sqrt(periods_per_year))


def information_ratio(portfolio_returns, benchmark_returns, periods_per_year=12):
    active_return = annualized_active_return(
        portfolio_returns,
        benchmark_returns,
        periods_per_year,
    )
    error = tracking_error(portfolio_returns, benchmark_returns, periods_per_year)
    if not np.isfinite(error) or error == 0:
        return np.nan
    return active_return / error


def active_metrics(portfolio_returns, benchmark_returns, periods_per_year=12):
    return pd.Series(
        {
            "annualized_active_return": annualized_active_return(
                portfolio_returns,
                benchmark_returns,
                periods_per_year,
            ),
            "tracking_error": tracking_error(
                portfolio_returns,
                benchmark_returns,
                periods_per_year,
            ),
            "information_ratio": information_ratio(
                portfolio_returns,
                benchmark_returns,
                periods_per_year,
            ),
        }
    )


def performance_metrics(returns, periods_per_year=12):
    return pd.Series(
        {
            "annualized_return": annualized_return(returns, periods_per_year),
            "annualized_volatility": annualized_volatility(returns, periods_per_year),
            "maximum_drawdown": maximum_drawdown(returns),
            "sharpe_ratio": sharpe_ratio(returns, periods_per_year=periods_per_year),
            "sortino_ratio": sortino_ratio(returns, periods_per_year=periods_per_year),
        }
    )
