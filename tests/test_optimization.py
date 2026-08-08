import unittest

import numpy as np
import pandas as pd

from src.optimization import (
    METHODS,
    erc_weights,
    estimate_weights,
    risk_contributions,
)


class OptimizationTest(unittest.TestCase):
    def setUp(self):
        generator = np.random.default_rng(7)
        common = generator.normal(0, 0.03, 120)
        values = np.column_stack(
            [common + generator.normal(0, scale, 120) for scale in np.linspace(0.01, 0.03, 5)]
        )
        self.returns = pd.DataFrame(values, columns=list("abcde"))

    def test_all_methods_respect_constraints(self):
        for method in METHODS:
            with self.subTest(method=method):
                weights = np.asarray(
                    estimate_weights(self.returns, method, max_weight=0.3),
                    dtype=float,
                )
                self.assertAlmostEqual(weights.sum(), 1.0, places=9)
                self.assertGreaterEqual(weights.min(), -1e-10)
                self.assertLessEqual(weights.max(), 0.3 + 1e-10)

    def test_singular_covariance_is_supported(self):
        base = np.linspace(-0.02, 0.03, 80)
        returns = pd.DataFrame({column: base for column in list("abcde")})
        for method in ("sample_gmv", "shrinkage_gmv"):
            with self.subTest(method=method):
                weights = np.asarray(estimate_weights(returns, method), dtype=float)
                self.assertTrue(np.isfinite(weights).all())
                self.assertAlmostEqual(weights.sum(), 1.0, places=9)

    def test_erc_equalizes_risk_on_diagonal_covariance(self):
        covariance = np.diag([0.01, 0.02, 0.03, 0.04, 0.05])
        weights = np.asarray(erc_weights(covariance), dtype=float)
        contributions = np.asarray(
            risk_contributions(weights, covariance),
            dtype=float,
        )
        np.testing.assert_allclose(contributions, np.full(5, 0.2), atol=1e-7)

    def test_infeasible_max_weight_raises(self):
        with self.assertRaises(ValueError):
            estimate_weights(self.returns, "equal_weight", max_weight=0.19)


if __name__ == "__main__":
    unittest.main()
