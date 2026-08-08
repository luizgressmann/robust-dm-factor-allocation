import matplotlib.pyplot as plt

from .metrics import wealth_index


def plot_wealth(returns, ax=None, log_scale=False):
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))
    wealth = wealth_index(returns)
    wealth.plot(ax=ax, logy=log_scale)
    ax.set_xlabel("Date")
    ax.set_ylabel("Wealth Index")
    ax.grid(alpha=0.3)
    return ax


def plot_weights(weights, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))
    weights.plot.area(ax=ax, stacked=True)
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    return ax
