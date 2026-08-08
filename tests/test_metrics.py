import unittest

import numpy as np
import pandas as pd

from src.metrics import (
    active_metrics,
    annualized_return,
    maximum_drawdown,
)


class MetricsTest(unittest.TestCase):
    def test_annualized_return(self):
        returns = pd.Series(np.full(12, 0.01))
        self.assertAlmostEqual(annualized_return(returns), 1.01**12 - 1)

    def test_empty_annualized_return_raises(self):
        with self.assertRaises(ValueError):
            annualized_return(pd.Series(dtype=float))

    def test_maximum_drawdown_includes_initial_wealth(self):
        returns = pd.Series([-0.2, 0.1])
        self.assertAlmostEqual(maximum_drawdown(returns), -0.2)

    def test_active_metrics(self):
        portfolio = pd.Series([0.02, 0.00, 0.03, -0.01])
        benchmark = pd.Series([0.01, 0.01, 0.01, 0.00])
        metrics = active_metrics(portfolio, benchmark)
        active = portfolio - benchmark
        expected_return = active.mean() * 12
        expected_error = active.std(ddof=1) * np.sqrt(12)
        self.assertAlmostEqual(metrics["annualized_active_return"], expected_return)
        self.assertAlmostEqual(metrics["tracking_error"], expected_error)
        self.assertAlmostEqual(
            metrics["information_ratio"],
            expected_return / expected_error,
        )


if __name__ == "__main__":
    unittest.main()
