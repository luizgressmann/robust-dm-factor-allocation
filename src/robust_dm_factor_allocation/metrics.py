"""Portfolio performance metrics."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from numbers import Integral, Real

import numpy as np
import pandas as pd

from ._validation import (
    MissingPolicy,
    finite_real,
    missing_policy,
)
from ._validation import (
    periods_per_year as validate_periods_per_year,
)
from .config import DEFAULT_PERIODS_PER_YEAR


def _clean_series(values: Iterable[float] | pd.Series, missing: MissingPolicy) -> pd.Series:
    try:
        series = pd.Series(values, copy=False).astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("returns must be numeric") from error
    array = series.to_numpy(dtype=float)
    if np.isinf(array).any():
        raise ValueError("returns must be finite")
    if np.isnan(array).any():
        if missing == "raise":
            raise ValueError("returns must not contain missing values")
        series = series.dropna()
    if series.empty:
        raise ValueError("returns must contain at least one observation")
    return series


def _clean_frame(values: pd.DataFrame, missing: MissingPolicy) -> pd.DataFrame:
    if not values.columns.is_unique:
        raise ValueError("returns columns must be unique")
    try:
        frame = values.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("returns must be numeric") from error
    array = frame.to_numpy(dtype=float)
    if np.isinf(array).any():
        raise ValueError("returns must be finite")
    if np.isnan(array).any():
        if missing == "raise":
            raise ValueError("returns must not contain missing values")
        frame = frame.dropna(axis=0, how="any")
    if frame.empty:
        raise ValueError("returns must contain at least one complete observation")
    return frame


def _return_statistic(
    returns: Iterable[float] | pd.Series | pd.DataFrame,
    function: Callable[[pd.Series], float],
    missing: MissingPolicy,
) -> float | pd.Series:
    if isinstance(returns, pd.DataFrame):
        return _clean_frame(returns, missing).apply(function)
    return float(function(_clean_series(returns, missing)))


def _validate_simple_returns(series: pd.Series) -> None:
    if (series <= -1).any():
        raise ValueError("returns must be greater than -1")


def annualized_return(
    returns: Iterable[float] | pd.Series | pd.DataFrame,
    periods_per_year: Integral = DEFAULT_PERIODS_PER_YEAR,
    *,
    missing: MissingPolicy = "raise",
) -> float | pd.Series:
    """Geometrically annualize periodic simple returns."""
    frequency = validate_periods_per_year(periods_per_year)
    policy = missing_policy(missing)

    def calculate(series: pd.Series) -> float:
        _validate_simple_returns(series)
        growth = float(np.prod(1 + series.to_numpy(dtype=float)))
        return growth ** (frequency / len(series)) - 1

    return _return_statistic(returns, calculate, policy)


def annualized_volatility(
    returns: Iterable[float] | pd.Series | pd.DataFrame,
    periods_per_year: Integral = DEFAULT_PERIODS_PER_YEAR,
    *,
    missing: MissingPolicy = "raise",
) -> float | pd.Series:
    """Annualized sample standard deviation of periodic returns."""
    frequency = validate_periods_per_year(periods_per_year)
    policy = missing_policy(missing)

    def calculate(series: pd.Series) -> float:
        if len(series) < 2:
            return float("nan")
        return float(series.std(ddof=1) * np.sqrt(frequency))

    return _return_statistic(returns, calculate, policy)


def wealth_index(
    returns: Iterable[float] | pd.Series | pd.DataFrame,
    *,
    missing: MissingPolicy = "raise",
) -> pd.Series | pd.DataFrame:
    """Compound simple returns into a wealth index starting at the first period."""
    policy = missing_policy(missing)
    if isinstance(returns, pd.DataFrame):
        frame = _clean_frame(returns, policy)
        if (frame.to_numpy(dtype=float) <= -1).any():
            raise ValueError("returns must be greater than -1")
        return (1 + frame).cumprod()
    series = _clean_series(returns, policy)
    _validate_simple_returns(series)
    return (1 + series).cumprod()


def maximum_drawdown(
    returns: Iterable[float] | pd.Series | pd.DataFrame,
    *,
    missing: MissingPolicy = "raise",
) -> float | pd.Series:
    """Largest peak-to-trough loss, including initial wealth as a peak."""
    policy = missing_policy(missing)

    def calculate(series: pd.Series) -> float:
        _validate_simple_returns(series)
        wealth = np.r_[1.0, np.cumprod(1 + series.to_numpy(dtype=float))]
        drawdowns = wealth / np.maximum.accumulate(wealth) - 1
        return float(drawdowns.min())

    return _return_statistic(returns, calculate, policy)


def sharpe_ratio(
    returns: Iterable[float] | pd.Series | pd.DataFrame,
    risk_free_rate: Real = 0.0,
    periods_per_year: Integral = DEFAULT_PERIODS_PER_YEAR,
    *,
    missing: MissingPolicy = "raise",
) -> float | pd.Series:
    """Annualized Sharpe ratio using an annual effective risk-free rate."""
    frequency = validate_periods_per_year(periods_per_year)
    annual_rate = finite_real(risk_free_rate, "risk_free_rate")
    if annual_rate <= -1:
        raise ValueError("risk_free_rate must be greater than -1")
    periodic_rate = (1 + annual_rate) ** (1 / frequency) - 1
    policy = missing_policy(missing)

    def calculate(series: pd.Series) -> float:
        excess = series - periodic_rate
        if len(excess) < 2:
            return float("nan")
        volatility = float(excess.std(ddof=1))
        if np.isclose(volatility, 0.0):
            return float("nan")
        return float(excess.mean() / volatility * np.sqrt(frequency))

    return _return_statistic(returns, calculate, policy)


def downside_deviation(
    returns: Iterable[float] | pd.Series | pd.DataFrame,
    target_return: Real = 0.0,
    periods_per_year: Integral = DEFAULT_PERIODS_PER_YEAR,
    *,
    missing: MissingPolicy = "raise",
) -> float | pd.Series:
    """Annualized lower partial deviation around a periodic target return."""
    frequency = validate_periods_per_year(periods_per_year)
    target = finite_real(target_return, "target_return")
    policy = missing_policy(missing)

    def calculate(series: pd.Series) -> float:
        downside = np.minimum(series.to_numpy(dtype=float) - target, 0)
        return float(np.sqrt(np.mean(downside**2)) * np.sqrt(frequency))

    return _return_statistic(returns, calculate, policy)


def sortino_ratio(
    returns: Iterable[float] | pd.Series | pd.DataFrame,
    target_return: Real = 0.0,
    periods_per_year: Integral = DEFAULT_PERIODS_PER_YEAR,
    *,
    missing: MissingPolicy = "raise",
) -> float | pd.Series:
    """Annualized arithmetic excess return per unit of downside deviation."""
    frequency = validate_periods_per_year(periods_per_year)
    target = finite_real(target_return, "target_return")
    policy = missing_policy(missing)

    def calculate(series: pd.Series) -> float:
        downside = np.minimum(series.to_numpy(dtype=float) - target, 0)
        risk = float(np.sqrt(np.mean(downside**2)))
        if np.isclose(risk, 0.0):
            return float("nan")
        return float((series.mean() - target) / risk * np.sqrt(frequency))

    return _return_statistic(returns, calculate, policy)


def _active_returns(
    portfolio_returns: Iterable[float] | pd.Series,
    benchmark_returns: Iterable[float] | pd.Series,
    missing: MissingPolicy,
) -> pd.Series:
    """Pair labeled series by index and all other inputs by position."""
    try:
        portfolio = pd.Series(portfolio_returns, copy=False, name="portfolio").astype(float)
        benchmark = pd.Series(benchmark_returns, copy=False, name="benchmark").astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("portfolio and benchmark returns must be numeric") from error

    both_labeled = isinstance(portfolio_returns, pd.Series) and isinstance(
        benchmark_returns, pd.Series
    )
    if both_labeled:
        if portfolio.index.has_duplicates or benchmark.index.has_duplicates:
            raise ValueError("portfolio and benchmark indexes must be unique")
        if not portfolio.index.equals(benchmark.index):
            raise ValueError("portfolio and benchmark indexes must match exactly")
        aligned = pd.concat([portfolio, benchmark], axis=1)
    else:
        if len(portfolio) != len(benchmark):
            raise ValueError("portfolio and benchmark returns must have the same length")
        aligned = pd.DataFrame(
            {
                "portfolio": portfolio.to_numpy(dtype=float),
                "benchmark": benchmark.to_numpy(dtype=float),
            }
        )
    if aligned.empty:
        raise ValueError("portfolio and benchmark must contain paired observations")
    array = aligned.to_numpy(dtype=float)
    if np.isinf(array).any():
        raise ValueError("portfolio and benchmark returns must be finite")
    if np.isnan(array).any():
        if missing == "raise":
            raise ValueError("portfolio and benchmark returns must not contain missing values")
        aligned = aligned.dropna(axis=0, how="any")
    if aligned.empty:
        raise ValueError("portfolio and benchmark have no complete overlap")
    return aligned["portfolio"] - aligned["benchmark"]


def annualized_active_return(
    portfolio_returns: Iterable[float] | pd.Series,
    benchmark_returns: Iterable[float] | pd.Series,
    periods_per_year: Integral = DEFAULT_PERIODS_PER_YEAR,
    *,
    missing: MissingPolicy = "raise",
) -> float:
    """Annualize the arithmetic mean portfolio-minus-benchmark return."""
    frequency = validate_periods_per_year(periods_per_year)
    active = _active_returns(portfolio_returns, benchmark_returns, missing_policy(missing))
    return float(active.mean() * frequency)


def tracking_error(
    portfolio_returns: Iterable[float] | pd.Series,
    benchmark_returns: Iterable[float] | pd.Series,
    periods_per_year: Integral = DEFAULT_PERIODS_PER_YEAR,
    *,
    missing: MissingPolicy = "raise",
) -> float:
    """Annualized sample volatility of arithmetic active returns."""
    frequency = validate_periods_per_year(periods_per_year)
    active = _active_returns(portfolio_returns, benchmark_returns, missing_policy(missing))
    if len(active) < 2:
        return float("nan")
    return float(active.std(ddof=1) * np.sqrt(frequency))


def information_ratio(
    portfolio_returns: Iterable[float] | pd.Series,
    benchmark_returns: Iterable[float] | pd.Series,
    periods_per_year: Integral = DEFAULT_PERIODS_PER_YEAR,
    *,
    missing: MissingPolicy = "raise",
) -> float:
    """Annualized active return divided by tracking error."""
    frequency = validate_periods_per_year(periods_per_year)
    policy = missing_policy(missing)
    active = _active_returns(portfolio_returns, benchmark_returns, policy)
    if len(active) < 2:
        return float("nan")
    error = float(active.std(ddof=1) * np.sqrt(frequency))
    if np.isclose(error, 0.0):
        return float("nan")
    return float(active.mean() * frequency / error)


def active_metrics(
    portfolio_returns: Iterable[float] | pd.Series,
    benchmark_returns: Iterable[float] | pd.Series,
    periods_per_year: Integral = DEFAULT_PERIODS_PER_YEAR,
    *,
    missing: MissingPolicy = "raise",
) -> pd.Series:
    """Calculate benchmark-relative performance metrics."""
    frequency = validate_periods_per_year(periods_per_year)
    policy = missing_policy(missing)
    active = _active_returns(portfolio_returns, benchmark_returns, policy)
    annual_active = float(active.mean() * frequency)
    error = float(active.std(ddof=1) * np.sqrt(frequency)) if len(active) >= 2 else float("nan")
    ratio = annual_active / error if np.isfinite(error) and not np.isclose(error, 0.0) else np.nan
    return pd.Series(
        {
            "annualized_active_return": annual_active,
            "tracking_error": error,
            "information_ratio": ratio,
        },
        dtype=float,
    )


def performance_metrics(
    returns: Iterable[float] | pd.Series | pd.DataFrame,
    periods_per_year: Integral = DEFAULT_PERIODS_PER_YEAR,
    *,
    missing: MissingPolicy = "raise",
) -> pd.Series | pd.DataFrame:
    """Calculate standard performance metrics."""
    frequency = validate_periods_per_year(periods_per_year)
    policy = missing_policy(missing)
    values = {
        "annualized_return": annualized_return(returns, frequency, missing=policy),
        "annualized_volatility": annualized_volatility(returns, frequency, missing=policy),
        "maximum_drawdown": maximum_drawdown(returns, missing=policy),
        "sharpe_ratio": sharpe_ratio(returns, periods_per_year=frequency, missing=policy),
        "sortino_ratio": sortino_ratio(returns, periods_per_year=frequency, missing=policy),
    }
    if isinstance(returns, pd.DataFrame):
        return pd.DataFrame(values)
    return pd.Series(values, dtype=float)
