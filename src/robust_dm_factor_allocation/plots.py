"""Portfolio plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from .metrics import wealth_index

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_wealth(
    returns: pd.Series | pd.DataFrame,
    ax: Axes | None = None,
    log_scale: bool = False,
) -> Axes:
    """Plot compounded wealth for one or more return series."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))
    wealth = wealth_index(returns)
    wealth.plot(ax=ax, logy=log_scale)
    ax.set_xlabel("Date")
    ax.set_ylabel("Wealth index")
    ax.grid(alpha=0.3)
    return ax


def plot_weights(weights: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot a stacked portfolio-weight history."""
    if not isinstance(weights, pd.DataFrame) or weights.empty:
        raise ValueError("weights must be a nonempty DataFrame")
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))
    weights.plot.area(ax=ax, stacked=True)
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    return ax
