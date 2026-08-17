"""Walk-forward portfolio backtest."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Integral, Real
from typing import TypedDict

import numpy as np
import pandas as pd

from ._validation import MissingPolicy, finite_real, positive_integer
from .config import (
    DEFAULT_LAG_MONTHS,
    DEFAULT_LOOKBACK_MONTHS,
    DEFAULT_MAX_WEIGHT,
    DEFAULT_METHOD,
    DEFAULT_REBALANCE_MONTHS,
    DEFAULT_TRANSACTION_COST_BPS,
)
from .data import FACTOR_COLUMNS, MARKET_COLUMN, validate_returns
from .optimization import _validate_cap, estimate_weights, project_weights

WeightFunction = Callable[[pd.DataFrame, float], Sequence[float] | np.ndarray | pd.Series]


class BacktestResult(TypedDict):
    """Backtest output tables."""

    returns: pd.DataFrame
    weights: pd.DataFrame
    pretrade_weights: pd.DataFrame
    target_weights: pd.DataFrame
    end_weights: pd.DataFrame


def _target_weights(
    window: pd.DataFrame,
    method: str,
    max_weight: float,
    weight_function: WeightFunction | None,
) -> np.ndarray:
    if weight_function is None:
        weights = estimate_weights(window, method, max_weight)
    else:
        weights = weight_function(window.copy(), max_weight)
    if isinstance(weights, pd.Series):
        if not weights.index.is_unique:
            raise ValueError("optimizer returned duplicate weight labels")
        if len(window.columns.difference(weights.index, sort=False)) or len(
            weights.index.difference(window.columns, sort=False)
        ):
            raise ValueError("optimizer weight labels must match factor columns")
        values = weights.reindex(window.columns).to_numpy(dtype=float)
    else:
        values = np.asarray(weights, dtype=float)
    if values.shape != (window.shape[1],) or not np.isfinite(values).all():
        raise ValueError("optimizer returned invalid weights")
    if (values < -1e-10).any() or (values > max_weight + 1e-10).any():
        raise ValueError("optimizer violated weight bounds")
    if not np.isclose(values.sum(), 1, atol=1e-9, rtol=0):
        raise ValueError("optimizer weights must sum to one")
    return project_weights(values, max_weight)


def walk_forward_backtest(
    returns: pd.DataFrame,
    method: str = DEFAULT_METHOD,
    lookback: Integral = DEFAULT_LOOKBACK_MONTHS,
    rebalance_months: Integral = DEFAULT_REBALANCE_MONTHS,
    lag: Integral = DEFAULT_LAG_MONTHS,
    max_weight: Real = DEFAULT_MAX_WEIGHT,
    transaction_cost_bps: Real = DEFAULT_TRANSACTION_COST_BPS,
    factor_columns: Sequence[str] = FACTOR_COLUMNS,
    market_column: str = MARKET_COLUMN,
    weight_function: WeightFunction | None = None,
    *,
    missing: MissingPolicy = "raise",
) -> BacktestResult:
    """Run a rolling-window backtest.

    Weights only use data available before the trade month. Between rebalances,
    they drift with returns. Trading costs are charged on turnover.
    """
    frame = validate_returns(
        returns,
        factor_columns,
        market_column,
        missing=missing,
    )
    factors = tuple(factor_columns)
    window_length = positive_integer(lookback, "lookback", minimum=2)
    rebalance_interval = positive_integer(rebalance_months, "rebalance_months")
    execution_lag = positive_integer(lag, "lag")
    cap = _validate_cap(len(factors), max_weight)
    costs_bps = finite_real(transaction_cost_bps, "transaction_cost_bps")
    if costs_bps < 0:
        raise ValueError("transaction_cost_bps must be nonnegative")
    if costs_bps >= 10_000:
        raise ValueError("transaction_cost_bps must be less than 10000")
    if weight_function is not None and not callable(weight_function):
        raise TypeError("weight_function must be callable")
    first_position = window_length + execution_lag - 1
    if first_position >= len(frame):
        raise ValueError("not enough observations for the requested lookback and lag")

    asset_returns = frame.loc[:, factors]
    result_rows: list[dict[str, object]] = []
    applied_rows: list[np.ndarray] = []
    pretrade_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    end_rows: list[np.ndarray] = []
    result_index: list[pd.Timestamp] = []
    current_end_weights: np.ndarray | None = None
    cost_rate = costs_bps / 10_000

    for position in range(first_position, len(frame)):
        date = frame.index[position]
        is_rebalance = (position - first_position) % rebalance_interval == 0
        estimation_end_date = pd.NaT
        if is_rebalance:
            estimation_end = position - execution_lag + 1
            estimation_start = estimation_end - window_length
            window = asset_returns.iloc[estimation_start:estimation_end]
            target = _target_weights(
                window,
                method,
                cap,
                weight_function,
            )
            estimation_end_date = window.index[-1]
            if current_end_weights is None:
                pretrade = target.copy()
                turnover = 0.0
            else:
                pretrade = current_end_weights.copy()
                turnover = float(0.5 * np.abs(target - pretrade).sum())
            applied = target
            target_row = target.copy()
        else:
            if current_end_weights is None:  # pragma: no cover
                raise RuntimeError("backtest has no portfolio weights")
            pretrade = current_end_weights.copy()
            turnover = 0.0
            applied = pretrade.copy()
            target_row = np.full(len(factors), np.nan)

        period_returns = asset_returns.iloc[position].to_numpy(dtype=float)
        gross_return = float(applied @ period_returns)
        cost = float(turnover * cost_rate)
        net_return = float((1 - cost) * (1 + gross_return) - 1)
        growth = applied * (1 + period_returns)
        current_end_weights = growth / growth.sum()
        benchmark_return = float(frame.iloc[position][market_column])

        result_rows.append(
            {
                "gross_return": gross_return,
                "net_return": net_return,
                "benchmark_return": benchmark_return,
                "active_return": net_return - benchmark_return,
                "turnover": turnover,
                "cost": cost,
                "rebalance": is_rebalance,
                "estimation_end": estimation_end_date,
            }
        )
        result_index.append(date)
        applied_rows.append(applied)
        pretrade_rows.append(pretrade)
        target_rows.append(target_row)
        end_rows.append(current_end_weights.copy())

    index = pd.DatetimeIndex(result_index, name=frame.index.name)
    results = pd.DataFrame(result_rows, index=index)
    return {
        "returns": results,
        "weights": pd.DataFrame(applied_rows, index=index, columns=factors),
        "pretrade_weights": pd.DataFrame(pretrade_rows, index=index, columns=factors),
        "target_weights": pd.DataFrame(target_rows, index=index, columns=factors),
        "end_weights": pd.DataFrame(end_rows, index=index, columns=factors),
    }
