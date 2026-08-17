import unittest

import numpy as np
import pandas as pd

from robust_dm_factor_allocation import ExactERCCapError
from robust_dm_factor_allocation.optimization import (
    METHODS,
    _capped_proportions,
    equal_weight,
    erc_weights,
    estimate_weights,
    gmv_weights,
    inverse_volatility_weights,
    oas_covariance,
    project_weights,
    risk_contributions,
    sample_covariance,
    sample_gmv_weights,
)


class OptimizationTest(unittest.TestCase):
    def setUp(self):
        generator = np.random.default_rng(7)
        common = generator.normal(0, 0.03, 120)
        values = np.column_stack(
            [common + generator.normal(0, scale, 120) for scale in np.linspace(0.01, 0.03, 4)]
        )
        self.returns = pd.DataFrame(values, columns=list("abcd"))

    def test_methods_respect_constraints(self):
        self.assertEqual(
            METHODS,
            ("equal_weight", "inverse_volatility", "sample_gmv", "oas_gmv", "erc"),
        )
        for method in METHODS:
            with self.subTest(method=method):
                weights = np.asarray(
                    estimate_weights(self.returns, method, max_weight=0.4),
                    dtype=float,
                )
                self.assertAlmostEqual(weights.sum(), 1.0, places=9)
                self.assertGreaterEqual(weights.min(), -1e-10)
                self.assertLessEqual(weights.max(), 0.4 + 1e-10)

    def test_gmv_with_singular_covariance(self):
        base = np.linspace(-0.02, 0.03, 80)
        returns = pd.DataFrame({column: base for column in list("abcd")})
        for method in ("sample_gmv", "oas_gmv"):
            with self.subTest(method=method):
                weights = np.asarray(estimate_weights(returns, method), dtype=float)
                self.assertTrue(np.isfinite(weights).all())
                self.assertAlmostEqual(weights.sum(), 1.0, places=9)

    def test_gmv_matches_closed_form_for_diagonal_covariance(self):
        covariance = np.diag([1.0, 2.0, 4.0])
        weights = np.asarray(gmv_weights(covariance), dtype=float)
        expected = np.array([1.0, 0.5, 0.25]) / 1.75
        np.testing.assert_allclose(weights, expected, atol=1e-8)

    def test_gmv_matches_closed_form_and_kkt_for_nondiagonal_spd_covariance(self):
        covariance = np.array(
            [
                [0.040, 0.006, 0.004],
                [0.006, 0.030, 0.005],
                [0.004, 0.005, 0.025],
            ]
        )
        inverse_times_one = np.linalg.solve(covariance, np.ones(3))
        expected = inverse_times_one / inverse_times_one.sum()
        self.assertGreater(expected.min(), 0)

        weights = np.asarray(gmv_weights(covariance), dtype=float)

        np.testing.assert_allclose(weights, expected, atol=1e-8)
        marginal_variances = covariance @ weights
        np.testing.assert_allclose(
            marginal_variances,
            np.full(3, marginal_variances.mean()),
            atol=1e-9,
        )

    def test_erc_equalizes_risk_on_diagonal_covariance(self):
        covariance = np.diag([0.01, 0.02, 0.03, 0.04])
        weights = np.asarray(erc_weights(covariance), dtype=float)
        contributions = np.asarray(risk_contributions(weights, covariance), dtype=float)
        np.testing.assert_allclose(contributions, np.full(4, 0.25), atol=1e-7)

    def test_erc_equalizes_contributions_for_nondiagonal_spd_covariance(self):
        covariance = np.array(
            [
                [0.040, 0.006, 0.004],
                [0.006, 0.030, 0.005],
                [0.004, 0.005, 0.025],
            ]
        )
        self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0)
        self.assertTrue(np.any(covariance - np.diag(np.diag(covariance))))

        weights = np.asarray(erc_weights(covariance), dtype=float)
        contributions = np.asarray(risk_contributions(weights, covariance), dtype=float)

        np.testing.assert_allclose(contributions, np.full(3, 1 / 3), atol=1e-8)

    def test_binding_cap_raises_for_erc(self):
        covariance = np.diag([0.0001, 1.0, 1.0, 1.0])
        self.assertTrue(issubclass(ExactERCCapError, ValueError))
        with self.assertRaisesRegex(ExactERCCapError, "binds the exact ERC"):
            erc_weights(covariance, max_weight=0.4)

    def test_risk_contributions_align_series_labels(self):
        covariance = pd.DataFrame(np.diag([1.0, 4.0]), index=["a", "b"], columns=["a", "b"])
        weights = pd.Series({"b": 0.8, "a": 0.2})
        contributions = risk_contributions(weights, covariance)
        expected = np.array([0.04, 2.56]) / 2.60
        self.assertEqual(contributions.index.tolist(), ["a", "b"])
        np.testing.assert_allclose(contributions.to_numpy(), expected)

    def test_risk_contributions_reject_label_mismatch(self):
        covariance = pd.DataFrame(np.eye(2), index=["a", "b"], columns=["a", "b"])
        with self.assertRaises(ValueError):
            risk_contributions(pd.Series({"a": 0.5, "c": 0.5}), covariance)

    def test_covariance_must_be_symmetric(self):
        with self.assertRaisesRegex(ValueError, "symmetric"):
            gmv_weights(np.array([[1.0, 0.3], [0.1, 1.0]]))

    def test_oas_formula_and_labels(self):
        returns = pd.DataFrame(
            [[0.01, 0.02], [0.03, -0.01], [-0.02, 0.04], [0.00, 0.01]],
            columns=["left", "right"],
        )
        estimate = oas_covariance(returns)
        centered = returns.to_numpy() - returns.to_numpy().mean(axis=0)
        sample = centered.T @ centered / len(centered)
        n_assets = sample.shape[0]
        trace = np.trace(sample)
        trace_squared_covariance = np.trace(sample @ sample)
        mu = trace / n_assets
        coefficient = min(
            ((1 - 2 / n_assets) * trace_squared_covariance + trace**2)
            / (
                (len(centered) + 1 - 2 / n_assets)
                * (trace_squared_covariance - trace**2 / n_assets)
            ),
            1,
        )
        expected = (1 - coefficient) * sample + coefficient * np.eye(2) * mu
        self.assertEqual(estimate.index.tolist(), ["left", "right"])
        np.testing.assert_allclose(estimate.to_numpy(), expected)

    def test_invalid_weight_parameters(self):
        with self.assertRaises(ValueError):
            estimate_weights(self.returns, "equal_weight", max_weight=0.19)
        with self.assertRaises(ValueError):
            estimate_weights(self.returns, "equal_weight", max_weight=np.nan)

    def test_integer_parameters(self):
        with self.assertRaises(TypeError):
            equal_weight(3.5)
        with self.assertRaises(TypeError):
            gmv_weights(np.eye(3), max_iterations=1.5)
        with self.assertRaises(TypeError):
            erc_weights(np.eye(3), max_iterations=True)

    def test_gmv_nonconvergence(self):
        covariance = np.array([[1.0, 0.8, 0.1], [0.8, 1.0, 0.2], [0.1, 0.2, 4.0]])
        with self.assertRaises(RuntimeError):
            gmv_weights(covariance, tolerance=1e-14, max_iterations=1)

    def test_projection_and_capped_proportion_edge_cases(self):
        projected = project_weights([1.2, -0.1, 0.3], max_weight=0.6)
        self.assertAlmostEqual(projected.sum(), 1.0)
        self.assertLessEqual(projected.max(), 0.6)
        with self.assertRaises(ValueError):
            project_weights([])
        with self.assertRaises(ValueError):
            project_weights([0.5, np.nan])
        with self.assertRaises(ValueError):
            project_weights([0.5, 0.5], max_weight=0.0)
        with self.assertRaises(ValueError):
            project_weights([0.5, 0.5], tolerance=0.0)
        with self.assertRaises(RuntimeError):
            project_weights(
                [3.0, -2.0, 1.0],
                max_weight=0.6,
                tolerance=1e-20,
                max_iterations=1,
            )

        np.testing.assert_allclose(_capped_proportions([1.0, 0.0, 0.0], 0.5), [0.5, 0.25, 0.25])
        with self.assertRaises(ValueError):
            _capped_proportions([[1.0, 2.0]], 1.0)
        with self.assertRaises(ValueError):
            _capped_proportions([1.0, -1.0], 1.0)
        with self.assertRaises(ValueError):
            _capped_proportions([0.0, 0.0], 1.0)

    def test_covariance_validation(self):
        duplicate = pd.DataFrame(np.eye(2), index=["a", "a"], columns=["a", "a"])
        mismatched = pd.DataFrame(np.eye(2), index=["a", "b"], columns=["b", "a"])
        nonnumeric = pd.DataFrame([["bad", 0], [0, "bad"]], index=["a", "b"], columns=["a", "b"])
        invalid = [
            duplicate,
            mismatched,
            nonnumeric,
            [["bad", 0], [0, "bad"]],
            np.ones((2, 3)),
            np.array([[1.0, np.inf], [np.inf, 1.0]]),
            np.array([[1.0, 2.0], [2.0, 1.0]]),
        ]
        for covariance in invalid:
            with self.subTest(covariance=covariance):
                with self.assertRaises(ValueError):
                    gmv_weights(covariance)

        almost_psd = np.diag([1.0, -1e-12])
        weights = gmv_weights(almost_psd)
        self.assertAlmostEqual(np.asarray(weights).sum(), 1.0)

    def test_return_matrix_validation(self):
        duplicate = pd.DataFrame([[0.01, 0.02], [0.02, 0.01]], columns=["a", "a"])
        nonnumeric = pd.DataFrame([["bad", 0.01], [0.02, 0.03]])
        invalid = [
            duplicate,
            nonnumeric,
            [["bad", 0.01], [0.02, 0.03]],
            [[0.01, 0.02]],
            [[0.01, np.nan], [0.02, 0.03]],
        ]
        for returns in invalid:
            with self.subTest(returns=returns):
                with self.assertRaises(ValueError):
                    estimate_weights(returns, "equal_weight")

        values = self.returns.to_numpy()
        self.assertIsInstance(sample_covariance(values), np.ndarray)
        self.assertIsInstance(oas_covariance(values), np.ndarray)
        self.assertIsInstance(sample_gmv_weights(values), np.ndarray)

    def test_inverse_volatility_requires_positive_variances(self):
        with self.assertRaisesRegex(ValueError, "variances"):
            inverse_volatility_weights(np.zeros((2, 2)))

    def test_risk_contribution_inputs(self):
        with self.assertRaisesRegex(ValueError, "labels"):
            equal_weight(2, labels=["a"])
        covariance = pd.DataFrame(np.eye(2), index=["a", "b"], columns=["a", "b"])
        duplicate_weights = pd.Series([0.5, 0.5], index=["a", "a"])
        with self.assertRaisesRegex(ValueError, "unique"):
            risk_contributions(duplicate_weights, covariance)
        with self.assertRaises(TypeError):
            risk_contributions([0.5, 0.5], covariance, normalized="yes")
        with self.assertRaises(ValueError):
            risk_contributions([1.0], covariance)
        with self.assertRaisesRegex(ValueError, "variance"):
            risk_contributions([0.0, 0.0], covariance)

        raw = risk_contributions([0.25, 0.75], covariance, normalized=False)
        np.testing.assert_allclose(raw.to_numpy(), [0.0625, 0.5625])

    def test_erc_degenerate_and_nonconvergent_cases(self):
        with self.assertRaisesRegex(ValueError, "positive asset variances"):
            erc_weights(np.diag([1.0, 0.0]))
        singular = np.ones((3, 3))
        weights = np.asarray(erc_weights(singular))
        np.testing.assert_allclose(weights, np.full(3, 1 / 3), atol=1e-7)
        covariance = np.array([[1.0, 0.8, 0.1], [0.8, 1.0, 0.2], [0.1, 0.2, 4.0]])
        with self.assertRaises(RuntimeError):
            erc_weights(covariance, tolerance=1e-14, max_iterations=1)

    def test_dispatch_rejects_invalid_method_names(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            estimate_weights(self.returns, " ")
        with self.assertRaisesRegex(ValueError, "unknown"):
            estimate_weights(self.returns, "not-a-method")


if __name__ == "__main__":
    unittest.main()
