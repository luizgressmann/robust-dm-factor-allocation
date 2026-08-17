import unittest

import numpy as np
import pandas as pd

from robust_dm_factor_allocation.metrics import (
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

    def test_active_metrics_use_paired_arithmetic_returns(self):
        portfolio = pd.Series([0.02, 0.00, 0.03, -0.01])
        benchmark = pd.Series([0.01, 0.01, 0.01, 0.00])
        metrics = active_metrics(portfolio, benchmark)
        active = portfolio - benchmark
        expected_return = active.mean() * 12
        expected_error = active.std(ddof=1) * np.sqrt(12)
        self.assertAlmostEqual(metrics["annualized_active_return"], expected_return)
        self.assertAlmostEqual(metrics["tracking_error"], expected_error)
        self.assertAlmostEqual(metrics["information_ratio"], expected_return / expected_error)

    def test_missing_values(self):
        returns = pd.Series([0.01, np.nan, 0.02])
        with self.assertRaises(ValueError):
            annualized_return(returns)
        expected = (1.01 * 1.02) ** 6 - 1
        self.assertAlmostEqual(annualized_return(returns, missing="drop"), expected)

    def test_dataframe_drop_policy_uses_a_common_sample(self):
        returns = pd.DataFrame(
            {
                "a": [0.01, np.nan, 0.03],
                "b": [0.02, 0.50, 0.04],
            }
        )
        result = annualized_return(returns, periods_per_year=2, missing="drop")
        self.assertAlmostEqual(result["a"], 1.01 * 1.03 - 1)
        self.assertAlmostEqual(result["b"], 1.02 * 1.04 - 1)

    def test_active_drop_policy_drops_pairs(self):
        portfolio = pd.Series([0.02, np.nan, 0.03])
        benchmark = pd.Series([0.01, 0.50, 0.02])
        self.assertAlmostEqual(
            annualized_active_return(portfolio, benchmark, missing="drop"),
            0.12,
        )

    def test_active_metrics_require_identical_series_indexes(self):
        portfolio = pd.Series([0.01, 0.02], index=["a", "b"])
        benchmark = pd.Series([0.00, 0.01], index=["b", "c"])
        with self.assertRaisesRegex(ValueError, "match exactly"):
            active_metrics(portfolio, benchmark)

    def test_active_inputs_pair_by_position(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            active_metrics([0.01, 0.02], [0.00])

        portfolio = pd.Series([0.02, 0.01], index=["later", "earlier"])
        result = annualized_active_return(portfolio, [0.01, 0.00])
        self.assertAlmostEqual(result, 0.12)

    def test_invalid_frequency(self):
        returns = pd.Series([0.01, -0.01, 0.02])
        benchmark = pd.Series([0.0, 0.0, 0.0])
        calls = [
            lambda value: annualized_return(returns, value),
            lambda value: annualized_volatility(returns, value),
            lambda value: sharpe_ratio(returns, periods_per_year=value),
            lambda value: downside_deviation(returns, periods_per_year=value),
            lambda value: sortino_ratio(returns, periods_per_year=value),
            lambda value: annualized_active_return(returns, benchmark, value),
            lambda value: tracking_error(returns, benchmark, value),
            lambda value: information_ratio(returns, benchmark, value),
            lambda value: active_metrics(returns, benchmark, value),
        ]
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, ValueError)):
                    call(12.5)
                with self.assertRaises((TypeError, ValueError)):
                    call(0)

    def test_invalid_rate_parameters(self):
        returns = pd.Series([0.01, -0.01])
        with self.assertRaises(ValueError):
            sharpe_ratio(returns, risk_free_rate=np.inf)
        with self.assertRaises(TypeError):
            sharpe_ratio(returns, risk_free_rate=True)
        with self.assertRaises(ValueError):
            downside_deviation(returns, target_return=np.nan)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            annualized_return(["not-a-number"])
        with self.assertRaises(ValueError):
            annualized_return([0.01, np.inf])

        duplicate_columns = pd.DataFrame([[0.01, 0.02]], columns=["a", "a"])
        with self.assertRaisesRegex(ValueError, "unique"):
            annualized_return(duplicate_columns)
        with self.assertRaises(ValueError):
            annualized_return(pd.DataFrame({"a": ["bad", "data"]}))
        with self.assertRaises(ValueError):
            annualized_return(pd.DataFrame({"a": [0.01, np.inf]}))
        with self.assertRaisesRegex(ValueError, "complete observation"):
            annualized_return(
                pd.DataFrame({"a": [np.nan], "b": [np.nan]}),
                missing="drop",
            )

    def test_simple_return_bounds_and_wealth_index(self):
        with self.assertRaisesRegex(ValueError, "greater than -1"):
            annualized_return([0.01, -1.0])
        with self.assertRaisesRegex(ValueError, "greater than -1"):
            wealth_index(pd.DataFrame({"a": [0.01, -1.0]}))

        series = pd.Series([0.10, -0.05], index=["first", "second"])
        expected = pd.Series([1.10, 1.045], index=series.index)
        pd.testing.assert_series_equal(wealth_index(series), expected)
        frame = pd.DataFrame({"a": [0.10, -0.05], "b": [0.00, 0.02]})
        pd.testing.assert_frame_equal(wealth_index(frame), (1 + frame).cumprod())

    def test_ratios_and_downside_metrics(self):
        self.assertTrue(np.isnan(annualized_volatility([0.01])))
        with self.assertRaisesRegex(ValueError, "greater than -1"):
            sharpe_ratio([0.01, 0.02], risk_free_rate=-1.0)
        self.assertTrue(np.isnan(sharpe_ratio([0.01])))
        self.assertTrue(np.isnan(sharpe_ratio([0.01, 0.01])))

        returns = pd.Series([-0.02, 0.01, -0.01])
        expected_downside = np.sqrt(np.mean(np.array([-0.02, 0.0, -0.01]) ** 2)) * np.sqrt(12)
        self.assertAlmostEqual(downside_deviation(returns), expected_downside)
        self.assertTrue(np.isnan(sortino_ratio([0.01, 0.02])))
        self.assertTrue(np.isfinite(sortino_ratio(returns)))

    def test_active_metric_alignment_and_short_samples(self):
        duplicate = pd.Series([0.01, 0.02], index=["same", "same"])
        with self.assertRaisesRegex(ValueError, "unique"):
            active_metrics(duplicate, duplicate)
        with self.assertRaisesRegex(ValueError, "numeric"):
            active_metrics(["bad"], [0.0])
        with self.assertRaisesRegex(ValueError, "paired observations"):
            active_metrics([], [])
        with self.assertRaisesRegex(ValueError, "finite"):
            active_metrics([np.inf, 0.01], [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "missing"):
            active_metrics([np.nan, 0.01], [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "complete overlap"):
            active_metrics([np.nan], [np.nan], missing="drop")

        self.assertTrue(np.isnan(tracking_error([0.01], [0.0])))
        self.assertTrue(np.isnan(information_ratio([0.01], [0.0])))
        self.assertTrue(np.isnan(information_ratio([0.01, 0.01], [0.0, 0.0])))
        one_period = active_metrics([0.01], [0.0])
        self.assertTrue(np.isnan(one_period["tracking_error"]))
        self.assertTrue(np.isnan(one_period["information_ratio"]))

    def test_dataframe_performance_metrics(self):
        returns = pd.DataFrame(
            {
                "a": [0.01, -0.02, 0.03],
                "b": [0.00, 0.01, -0.01],
            }
        )
        summary = performance_metrics(returns)
        self.assertEqual(summary.index.tolist(), ["a", "b"])
        self.assertEqual(
            summary.columns.tolist(),
            [
                "annualized_return",
                "annualized_volatility",
                "maximum_drawdown",
                "sharpe_ratio",
                "sortino_ratio",
            ],
        )

    def test_unknown_missing_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing_policy"):
            annualized_return([0.01], missing="guess")


if __name__ == "__main__":
    unittest.main()
