import unittest

import numpy as np
import pandas as pd

from src.backtest import walk_forward_backtest
from src.data import FACTOR_COLUMNS


class BacktestTest(unittest.TestCase):
    def make_returns(self, periods=12):
        index = pd.date_range("2020-01-31", periods=periods, freq="ME")
        frame = pd.DataFrame(0.0, index=index, columns=[*FACTOR_COLUMNS, "market"])
        frame.index.name = "date"
        return frame

    def test_estimation_window_is_lagged(self):
        returns = self.make_returns()
        seen_windows = []

        def optimizer(window, max_weight):
            seen_windows.append(window.copy())
            return np.full(5, 0.2)

        result = walk_forward_backtest(
            returns,
            lookback=3,
            rebalance_months=3,
            lag=1,
            weight_function=optimizer,
        )
        rebalance_dates = result["returns"].index[result["returns"]["rebalance"]]
        self.assertEqual(len(seen_windows), len(rebalance_dates))
        for window, date in zip(seen_windows, rebalance_dates):
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
            lookback=3,
            rebalance_months=3,
            transaction_cost_bps=100,
        )
        second_rebalance = result["returns"].index[3]
        expected_turnover = 2 / 15
        expected_cost = expected_turnover * 0.01
        self.assertAlmostEqual(
            result["returns"].loc[second_rebalance, "turnover"],
            expected_turnover,
        )
        self.assertAlmostEqual(
            result["returns"].loc[second_rebalance, "cost"],
            expected_cost,
        )
        self.assertAlmostEqual(
            result["returns"].loc[second_rebalance, "net_return"],
            -expected_cost,
        )


if __name__ == "__main__":
    unittest.main()
