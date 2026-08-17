"""Tools for developed-market factor allocation."""

from .backtest import BacktestResult, walk_forward_backtest
from .config import ProjectConfig, RobustnessConfig
from .data import load_config, load_monthly_returns, validate_returns
from .inference import circular_block_bootstrap_mean
from .metrics import (
    active_metrics,
    annualized_active_return,
    annualized_return,
    annualized_volatility,
    downside_deviation,
    information_ratio,
    maximum_drawdown,
    performance_metrics,
    sharpe_ratio,
    sortino_ratio,
    tracking_error,
    wealth_index,
)
from .optimization import (
    METHODS,
    ExactERCCapError,
    equal_weight,
    erc_weights,
    estimate_weights,
    gmv_weights,
    inverse_volatility_weights,
    oas_covariance,
    oas_gmv_weights,
    project_weights,
    risk_contributions,
    sample_covariance,
    sample_gmv_weights,
)
from .plots import plot_wealth, plot_weights

__version__ = "1.0.0"

__all__ = [
    "BacktestResult",
    "ExactERCCapError",
    "METHODS",
    "ProjectConfig",
    "RobustnessConfig",
    "active_metrics",
    "annualized_active_return",
    "annualized_return",
    "annualized_volatility",
    "circular_block_bootstrap_mean",
    "downside_deviation",
    "equal_weight",
    "erc_weights",
    "estimate_weights",
    "gmv_weights",
    "information_ratio",
    "inverse_volatility_weights",
    "load_config",
    "load_monthly_returns",
    "maximum_drawdown",
    "oas_covariance",
    "oas_gmv_weights",
    "performance_metrics",
    "plot_wealth",
    "plot_weights",
    "project_weights",
    "risk_contributions",
    "sample_covariance",
    "sample_gmv_weights",
    "sharpe_ratio",
    "sortino_ratio",
    "tracking_error",
    "validate_returns",
    "walk_forward_backtest",
    "wealth_index",
]
