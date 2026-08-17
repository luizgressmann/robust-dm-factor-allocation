"""Input checks used across the package."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Literal

import numpy as np

MissingPolicy = Literal["raise", "drop"]


def finite_real(value: Real, name: str) -> float:
    """Convert a finite real number to ``float``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def positive_integer(value: Integral, name: str, *, minimum: int = 1) -> int:
    """Check that an integer meets a lower bound."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def periods_per_year(value: Integral) -> int:
    """Check an annualization frequency."""
    return positive_integer(value, "periods_per_year")


def missing_policy(value: str) -> MissingPolicy:
    """Check the missing-value setting."""
    if value not in {"raise", "drop"}:
        raise ValueError("missing_policy must be 'raise' or 'drop'")
    return value
