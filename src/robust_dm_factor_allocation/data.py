"""Monthly return data."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ._validation import MissingPolicy, missing_policy
from .config import DEFAULT_BENCHMARK, DEFAULT_FACTOR_COLUMNS, ProjectConfig

FACTOR_COLUMNS = DEFAULT_FACTOR_COLUMNS
MARKET_COLUMN = DEFAULT_BENCHMARK


def _return_columns(
    factor_columns: Sequence[str],
    market_column: str,
) -> tuple[tuple[str, ...], str]:
    if isinstance(factor_columns, (str, bytes)):
        raise TypeError("factor_columns must be a sequence of column names")
    factors = tuple(factor_columns)
    if len(factors) < 2:
        raise ValueError("at least two factor columns are required")
    if any(not isinstance(column, str) or not column for column in factors):
        raise ValueError("factor columns must be nonempty strings")
    if len(set(factors)) != len(factors):
        raise ValueError("factor columns must be unique")
    if not isinstance(market_column, str) or not market_column:
        raise ValueError("market_column must be a nonempty string")
    if market_column in factors:
        raise ValueError("market_column must not also be a factor column")
    return factors, market_column


def validate_returns(
    returns: pd.DataFrame,
    factor_columns: Sequence[str] = FACTOR_COLUMNS,
    market_column: str = MARKET_COLUMN,
    require_complete_months: bool = True,
    *,
    missing: MissingPolicy = "raise",
) -> pd.DataFrame:
    """Check and order monthly factor and market returns.

    Missing values raise an error by default. To drop incomplete months, use
    ``missing="drop"`` with ``require_complete_months=False``.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if not isinstance(require_complete_months, (bool, np.bool_)):
        raise TypeError("require_complete_months must be boolean")
    policy = missing_policy(missing)
    factors, benchmark = _return_columns(factor_columns, market_column)
    if not returns.columns.is_unique:
        raise ValueError("returns columns must be unique")

    required_columns = [*factors, benchmark]
    missing_columns = [column for column in required_columns if column not in returns.columns]
    if missing_columns:
        raise ValueError(f"missing return columns: {missing_columns}")
    frame = returns.loc[:, required_columns].copy()
    try:
        frame = frame.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("returns must be numeric") from error

    values = frame.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("returns must be finite")
    if np.isnan(values).any():
        if policy == "raise":
            raise ValueError("returns must not contain missing values")
        frame = frame.dropna(axis=0, how="any")
    if frame.empty:
        raise ValueError("returns must contain at least one complete observation")

    if not isinstance(frame.index, pd.DatetimeIndex):
        try:
            frame.index = pd.to_datetime(frame.index, errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError("returns index must contain dates") from error
    if frame.index.tz is not None:
        raise ValueError("returns index must be timezone-naive")
    if frame.index.has_duplicates:
        raise ValueError("returns index contains duplicate dates")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("returns index must be increasing")
    if (frame.to_numpy(dtype=float) <= -1).any():
        raise ValueError("returns must be greater than -1")

    months = frame.index.to_period("M")
    month_ends = months.to_timestamp("M")
    if not frame.index.equals(month_ends):
        raise ValueError("returns index must use calendar month-end labels")
    if months.has_duplicates:
        raise ValueError("returns index contains duplicate months")
    if require_complete_months:
        expected = pd.period_range(months[0], months[-1], freq="M")
        if not months.equals(expected):
            raise ValueError("returns index contains missing months")
    frame.index.name = returns.index.name or "date"
    return frame


def load_monthly_returns(
    file_path: str | Path,
    factor_columns: Sequence[str] = FACTOR_COLUMNS,
    market_column: str = MARKET_COLUMN,
    *,
    missing: MissingPolicy = "raise",
) -> pd.DataFrame:
    """Load and check a monthly return CSV."""
    path = Path(file_path)
    frame = pd.read_csv(path, index_col="date", parse_dates=True)
    return validate_returns(
        frame,
        factor_columns,
        market_column,
        missing=missing,
    )


def load_config(file_path: str | Path) -> ProjectConfig:
    """Load project settings from YAML."""
    import yaml

    path = Path(file_path)
    with path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError("config must contain a mapping")
    return ProjectConfig.from_mapping(raw, config_path=path)
