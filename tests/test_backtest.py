import unittest

import numpy as np
import pandas as pd

from robust_dm_factor_allocation.backtest import walk_forward_backtest
from robust_dm_factor_allocation.data import FACTOR_COLUMNS


class BacktestTest(unittest.TestCase):
    def make_returns(self, periods=12):
        index = pd.date_range(
            "2020-01-31",
            periods=periods,
            freq=pd.offsets.MonthEnd(),
        )
        frame = pd.DataFrame(0.0, index=index, columns=[*FACTOR_COLUMNS, "market"])
        frame.index.name = "date"
        return frame

    def test_estimation_window_is_lagged(self):
        returns = self.make_returns()
        seen_windows = []

        def optimizer(window, max_weight):
            seen_windows.append(window.copy())
            return np.full(len(FACTOR_COLUMNS), 1 / len(FACTOR_COLUMNS))

        result = walk_forward_backtest(
            returns,
            method="equal_weight",
            lookback=3,
            rebalance_months=3,
            lag=1,
            weight_function=optimizer,
        )
        rebalance_dates = result["returns"].index[result["returns"]["rebalance"]]
        self.assertEqual(len(seen_windows), len(rebalance_dates))
        for window, date in zip(seen_windows, rebalance_dates, strict=True):
            self.assertEqual(len(window), 3)
            self.assertLess(window.index[-1], date)

    def test_quarterly_rebalancing(self):
        result = walk_forward_backtest(
            self.make_returns(),
            lookback=3,
            rebalance_months=3,
        )
        positions = np.flatnonzero(result["returns"]["rebalance"].to_numpy())
        np.testing.assert_array_equal(np.diff(positions), np.full(len(positions) - 1, 3))

    def test_turnover_uses_drifted_weights_and_costs(self):
        returns = self.make_returns(10)
        returns.iloc[3, returns.columns.get_loc(FACTOR_COLUMNS[0])] = 1.0
        result = walk_forward_backtest(
            returns,
            method="equal_weight",
            lookback=3,
            rebalance_months=3,
            transaction_cost_bps=100,
        )
        second_rebalance = result["returns"].index[3]
        expected_turnover = 0.15
        expected_cost = expected_turnover * 0.01
        self.assertAlmostEqual(
            result["returns"].loc[second_rebalance, "turnover"],
            expected_turnover,
        )
        self.assertAlmostEqual(result["returns"].loc[second_rebalance, "cost"], expected_cost)
        self.assertAlmostEqual(
            result["returns"].loc[second_rebalance, "net_return"],
            -expected_cost,
        )

    def test_optimizer_series_is_aligned_by_factor_name(self):
        returns = self.make_returns(7)

        def optimizer(window, max_weight):
            return pd.Series(
                [0.1, 0.2, 0.3, 0.4],
                index=list(reversed(FACTOR_COLUMNS)),
            )

        result = walk_forward_backtest(
            returns,
            lookback=3,
            max_weight=0.5,
            weight_function=optimizer,
        )
        first = result["weights"].iloc[0]
        self.assertAlmostEqual(first[FACTOR_COLUMNS[0]], 0.4)
        self.assertAlmostEqual(first[FACTOR_COLUMNS[-1]], 0.1)

    def test_optimizer_series_requires_exact_labels(self):
        def optimizer(window, max_weight):
            return pd.Series(
                np.full(len(FACTOR_COLUMNS), 1 / len(FACTOR_COLUMNS)),
                index=[*FACTOR_COLUMNS[:-1], "wrong"],
            )

        with self.assertRaisesRegex(ValueError, "labels"):
            walk_forward_backtest(
                self.make_returns(7),
                lookback=3,
                weight_function=optimizer,
            )

    def test_nonfinite_costs_and_fractional_integers_raise(self):
        with self.assertRaises(ValueError):
            walk_forward_backtest(self.make_returns(), lookback=3, transaction_cost_bps=np.nan)
        with self.assertRaises(TypeError):
            walk_forward_backtest(self.make_returns(), lookback=3.5)
        with self.assertRaises(TypeError):
            walk_forward_backtest(self.make_returns(), lookback=3, lag=True)

    def test_missing_returns_are_rejected(self):
        returns = self.make_returns()
        returns.iloc[2, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "missing"):
            walk_forward_backtest(returns, lookback=3)

    def test_result_columns(self):
        result = walk_forward_backtest(self.make_returns(7), lookback=3)
        self.assertEqual(
            result["returns"].columns.tolist(),
            [
                "gross_return",
                "net_return",
                "benchmark_return",
                "active_return",
                "turnover",
                "cost",
                "rebalance",
                "estimation_end",
            ],
        )
        np.testing.assert_allclose(
            result["returns"]["active_return"],
            result["returns"]["net_return"] - result["returns"]["benchmark_return"],
        )

    def test_custom_optimizer_validation(self):
        returns = self.make_returns(7)

        def duplicate_labels(window, max_weight):
            return pd.Series([0.25, 0.25, 0.25, 0.25], index=["a", "a", "b", "c"])

        def wrong_shape(window, max_weight):
            return np.array([0.5, 0.5])

        def outside_bounds(window, max_weight):
            return np.array([0.5, 1 / 6, 1 / 6, 1 / 6])

        def wrong_sum(window, max_weight):
            return np.full(len(FACTOR_COLUMNS), 0.1)

        cases = [duplicate_labels, wrong_shape, outside_bounds, wrong_sum]
        for optimizer in cases:
            with self.subTest(optimizer=optimizer):
                with self.assertRaises(ValueError):
                    walk_forward_backtest(
                        returns,
                        lookback=3,
                        max_weight=0.4,
                        weight_function=optimizer,
                    )

    def test_invalid_backtest_parameters(self):
        returns = self.make_returns()
        invalid_arguments = [
            {"transaction_cost_bps": -1},
            {"transaction_cost_bps": 10_000},
            {"weight_function": 4},
            {"lookback": len(returns)},
        ]
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises((TypeError, ValueError)):
                    walk_forward_backtest(returns, **({"lookback": 3} | arguments))


if __name__ == "__main__":
    unittest.main()
