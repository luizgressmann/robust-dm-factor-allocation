"""Run the allocation code on synthetic monthly returns."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .backtest import BacktestResult, walk_forward_backtest
from .config import DEFAULT_FACTOR_COLUMNS
from .metrics import active_metrics, performance_metrics
from .optimization import METHODS

plt.switch_backend("Agg")


DISPLAY_NAMES = {
    "equal_weight": "Equal weight",
    "inverse_volatility": "Inverse volatility",
    "sample_gmv": "Sample GMV",
    "oas_gmv": "OAS GMV",
    "erc": "ERC",
    "market": "Synthetic market",
}
COLORS = {
    "equal_weight": "#4477AA",
    "inverse_volatility": "#66CCEE",
    "sample_gmv": "#228833",
    "oas_gmv": "#CC6677",
    "erc": "#AA3377",
    "market": "#555555",
}


def generate_synthetic_returns(*, periods: int = 180, seed: int = 19) -> pd.DataFrame:
    """Create synthetic monthly returns with calm and stressed periods."""
    if isinstance(periods, bool) or not isinstance(periods, int):
        raise TypeError("periods must be an integer")
    if periods < 180:
        raise ValueError("periods must be at least 180 for the demo")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    generator = np.random.default_rng(seed)
    positions = np.arange(periods)
    stressed = ((positions >= 84) & (positions < 114)) | ((positions >= 204) & (positions < 234))
    volatility_scale = np.where(stressed, 1.55, 0.85)

    market_mean = 0.0055
    market = market_mean + 0.038 * volatility_scale * generator.standard_normal(periods)
    for position, shock in ((96, -0.12), (218, -0.15)):
        if position < periods:
            market[position] += shock

    betas = np.array([0.98, 0.92, 0.82, 1.04])
    target_means = np.array([0.0060, 0.0063, 0.0061, 0.0060])
    idiosyncratic_volatility = np.array([0.017, 0.019, 0.013, 0.018])
    residuals = generator.standard_normal((periods, len(betas)))
    residuals *= volatility_scale[:, None] * idiosyncratic_volatility
    factor_returns = target_means + betas * (market[:, None] - market_mean) + residuals

    values = np.column_stack([factor_returns, market])
    values = np.clip(values, -0.80, 0.50)
    index = pd.date_range("2000-01-31", periods=periods, freq=pd.offsets.MonthEnd())
    frame = pd.DataFrame(
        values,
        index=index,
        columns=[*DEFAULT_FACTOR_COLUMNS, "market"],
    )
    frame.index.name = "date"
    return frame


def _run_method_comparison(returns: pd.DataFrame) -> dict[str, BacktestResult]:
    return {
        method: walk_forward_backtest(
            returns,
            method=method,
            lookback=60,
            rebalance_months=3,
            lag=1,
            max_weight=0.40,
            transaction_cost_bps=10,
        )
        for method in METHODS
    }


def _return_table(backtests: dict[str, BacktestResult]) -> pd.DataFrame:
    table = pd.DataFrame(
        {method: result["returns"]["net_return"] for method, result in backtests.items()}
    )
    first_result = next(iter(backtests.values()))
    table["market"] = first_result["returns"]["benchmark_return"]
    return table


def _summary_table(
    return_table: pd.DataFrame,
    backtests: dict[str, BacktestResult],
) -> pd.DataFrame:
    summary = return_table.apply(performance_metrics).T
    benchmark = return_table["market"]
    active = pd.DataFrame(
        {method: active_metrics(return_table[method], benchmark) for method in METHODS}
    ).T
    summary = summary.join(active)
    summary["annualized_turnover"] = np.nan
    for method, result in backtests.items():
        summary.loc[method, "annualized_turnover"] = result["returns"]["turnover"].mean() * 12
    summary.index.name = "method"
    return summary


def _robustness_table(returns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    common_start = returns.index[120]
    for cap in (0.30, 0.40, 0.50):
        for lookback in (60, 120):
            for rebalance_months in (3, 12):
                result = walk_forward_backtest(
                    returns,
                    method="oas_gmv",
                    lookback=lookback,
                    rebalance_months=rebalance_months,
                    lag=1,
                    max_weight=cap,
                    transaction_cost_bps=10,
                )
                result_returns = result["returns"].loc[common_start:]
                metrics = active_metrics(
                    result_returns["net_return"],
                    result_returns["benchmark_return"],
                )
                rows.append(
                    {
                        "method": "oas_gmv",
                        "maximum_weight": cap,
                        "lookback_months": lookback,
                        "rebalance_months": rebalance_months,
                        "annualized_active_return": metrics["annualized_active_return"],
                        "tracking_error": metrics["tracking_error"],
                        "information_ratio": metrics["information_ratio"],
                        "annualized_turnover": result_returns["turnover"].mean() * 12,
                    }
                )
    return pd.DataFrame(rows)


def _plot_wealth(return_table: pd.DataFrame, path: Path) -> None:
    selected = return_table.loc[:, [*METHODS, "market"]]
    starting_date = selected.index[0] - pd.offsets.MonthEnd(1)
    wealth = pd.concat(
        [
            pd.DataFrame(1.0, index=[starting_date], columns=selected.columns),
            (1 + selected).cumprod(),
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for method in selected.columns:
        ax.plot(
            wealth.index,
            wealth[method],
            color=COLORS[method],
            linewidth=2.1 if method in {"oas_gmv", "market"} else 1.5,
            label=DISPLAY_NAMES[method],
        )
    ax.set_yscale("log")
    ticks = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"{y:g}")
    )
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_title("Synthetic walk-forward growth simulation of different strategies")
    ax.set_ylabel("Cummulative wealth (initial value = 1)")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_cap_sensitivity(robustness: pd.DataFrame, path: Path) -> None:
    caps = (0.30, 0.40, 0.50)
    color = "#4477AA"
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for x_position, cap in enumerate(caps, start=1):
        values = robustness.loc[
            np.isclose(robustness["maximum_weight"], cap), "information_ratio"
        ].to_numpy()
        offsets = np.linspace(-0.08, 0.08, len(values))
        ax.scatter(
            np.full(len(values), x_position) + offsets,
            values,
            color=color,
            alpha=0.72,
            s=34,
            zorder=3,
        )
        median = float(np.median(values))
        ax.plot(
            [x_position - 0.18, x_position + 0.18],
            [median, median],
            color="#222222",
            linewidth=2.5,
            zorder=4,
        )
    ax.axhline(0, color="#777777", linewidth=1, linestyle="--")
    ax.set_xticks(range(1, len(caps) + 1), [f"{cap:.0%}" for cap in caps])
    ax.set_xlabel("Maximum target weight")
    ax.set_ylabel("Information ratio")
    ax.set_title("OAS-GMV sensitivity across windows and rebalance schedules")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_demo(output_dir: str | Path, *, seed: int = 19) -> Path:
    """Run the synthetic example and return its output directory."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    returns = generate_synthetic_returns(seed=seed)
    backtests = _run_method_comparison(returns)
    return_table = _return_table(backtests)
    summary = _summary_table(return_table, backtests)
    robustness = _robustness_table(returns)

    summary.to_csv(destination / "summary.csv", lineterminator="\n")
    robustness.to_csv(destination / "robustness.csv", index=False, lineterminator="\n")
    _plot_wealth(return_table, destination / "wealth.png")
    _plot_cap_sensitivity(robustness, destination / "cap_sensitivity.png")

    return destination


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/demo"),
        help="Directory for synthetic results (default: results/demo)",
    )
    parser.add_argument("--seed", type=int, default=19, help="Synthetic generator seed")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    destination = run_demo(arguments.output_dir, seed=arguments.seed)
    print(f"Synthetic demo written to {destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
