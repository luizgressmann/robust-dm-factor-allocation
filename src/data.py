from pathlib import Path

import numpy as np
import pandas as pd


FACTOR_COLUMNS = (
    "value",
    "momentum",
    "quality",
    "min_volatility",
    "size_proxy",
)
MARKET_COLUMN = "market"


def validate_returns(
    returns,
    factor_columns=FACTOR_COLUMNS,
    market_column=MARKET_COLUMN,
    require_complete_months=True,
):
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    factor_columns = tuple(factor_columns)
    if len(factor_columns) < 2 or len(set(factor_columns)) != len(factor_columns):
        raise ValueError("at least two unique factor columns are required")
    required_columns = [*factor_columns, market_column]
    missing_columns = [column for column in required_columns if column not in returns]
    if missing_columns:
        raise ValueError(f"missing return columns: {missing_columns}")
    frame = returns.loc[:, required_columns].copy()
    try:
        frame = frame.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("returns must be numeric") from error
    if not isinstance(frame.index, pd.DatetimeIndex):
        try:
            frame.index = pd.to_datetime(frame.index, errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError("returns index must contain dates") from error
    if frame.index.has_duplicates:
        raise ValueError("returns index contains duplicate dates")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("returns index must be increasing")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("returns must be finite and complete")
    if (values <= -1).any():
        raise ValueError("returns must be greater than -1")
    months = frame.index.to_period("M")
    if months.has_duplicates:
        raise ValueError("returns index contains duplicate months")
    if require_complete_months and len(months):
        expected = pd.period_range(months[0], months[-1], freq="M")
        if not months.equals(expected):
            raise ValueError("returns index contains missing months")
    frame.index.name = returns.index.name or "date"
    return frame


def load_monthly_returns(
    file_path,
    factor_columns=FACTOR_COLUMNS,
    market_column=MARKET_COLUMN,
):
    path = Path(file_path)
    frame = pd.read_csv(path, index_col="date", parse_dates=True)
    return validate_returns(frame, factor_columns, market_column)


def load_config(file_path):
    import yaml

    path = Path(file_path)
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("config must contain a mapping")
    return config
