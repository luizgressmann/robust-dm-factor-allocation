import numpy as np
import pandas as pd

from .data import FACTOR_COLUMNS, MARKET_COLUMN, validate_returns
from .optimization import estimate_weights


def _target_weights(window, method, max_weight, shrinkage, weight_function):
    if weight_function is None:
        weights = estimate_weights(window, method, max_weight, shrinkage)
    else:
        weights = weight_function(window.copy(), max_weight)
    if isinstance(weights, pd.Series):
        weights = weights.reindex(window.columns).to_numpy(dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
    if weights.shape != (window.shape[1],) or not np.isfinite(weights).all():
        raise ValueError("optimizer returned invalid weights")
    if (weights < -1e-10).any() or (weights > max_weight + 1e-10).any():
        raise ValueError("optimizer violated weight bounds")
    if not np.isclose(weights.sum(), 1, atol=1e-9):
        raise ValueError("optimizer weights must sum to one")
    return np.clip(weights, 0, max_weight) / np.clip(weights, 0, max_weight).sum()


def walk_forward_backtest(
    returns,
    method="equal_weight",
    lookback=60,
    rebalance_months=3,
    lag=1,
    max_weight=1.0,
    transaction_cost_bps=0.0,
    shrinkage=None,
    factor_columns=FACTOR_COLUMNS,
    market_column=MARKET_COLUMN,
    weight_function=None,
):
    frame = validate_returns(returns, factor_columns, market_column)
    factor_columns = tuple(factor_columns)
    lookback = int(lookback)
    rebalance_months = int(rebalance_months)
    lag = int(lag)
    if lookback < 2:
        raise ValueError("lookback must be at least two")
    if rebalance_months < 1:
        raise ValueError("rebalance_months must be positive")
    if lag < 1:
        raise ValueError("lag must be at least one")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be nonnegative")
    first_position = lookback + lag - 1
    if first_position >= len(frame):
        raise ValueError("not enough observations for the requested lookback and lag")
    asset_returns = frame.loc[:, factor_columns]
    result_rows = []
    applied_rows = []
    pretrade_rows = []
    target_rows = []
    end_rows = []
    result_index = []
    current_end_weights = None
    cost_rate = transaction_cost_bps / 10000
    for position in range(first_position, len(frame)):
        date = frame.index[position]
        is_rebalance = (position - first_position) % rebalance_months == 0
        estimation_end_date = pd.NaT
        if is_rebalance:
            estimation_end = position - lag + 1
            estimation_start = estimation_end - lookback
            window = asset_returns.iloc[estimation_start:estimation_end]
            target = _target_weights(
                window,
                method,
                max_weight,
                shrinkage,
                weight_function,
            )
            estimation_end_date = window.index[-1]
            if current_end_weights is None:
                pretrade = target.copy()
                turnover = 0.0
            else:
                pretrade = current_end_weights.copy()
                turnover = 0.5 * np.abs(target - pretrade).sum()
            applied = target
            target_row = target.copy()
        else:
            pretrade = current_end_weights.copy()
            turnover = 0.0
            applied = pretrade.copy()
            target_row = np.full(len(factor_columns), np.nan)
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
        "weights": pd.DataFrame(applied_rows, index=index, columns=factor_columns),
        "pretrade_weights": pd.DataFrame(
            pretrade_rows,
            index=index,
            columns=factor_columns,
        ),
        "target_weights": pd.DataFrame(
            target_rows,
            index=index,
            columns=factor_columns,
        ),
        "end_weights": pd.DataFrame(end_rows, index=index, columns=factor_columns),
    }
