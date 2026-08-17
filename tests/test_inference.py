import unittest

import numpy as np
import pandas as pd

from robust_dm_factor_allocation.inference import circular_block_bootstrap_mean


class CircularBlockBootstrapTest(unittest.TestCase):
    def test_fixed_seed_and_result_fields(self):
        values = pd.Series([0.01, -0.02, 0.03, 0.00, 0.02, -0.01])
        arguments = {
            "block_months": 2,
            "repetitions": 500,
            "seed": 7,
            "periods_per_year": 12,
            "confidence_level": 0.90,
        }
        first = circular_block_bootstrap_mean(values, **arguments)
        second = circular_block_bootstrap_mean(values, **arguments)
        self.assertEqual(
            first.index.tolist(),
            ["annualized_mean", "lower_bound", "upper_bound"],
        )
        pd.testing.assert_series_equal(first, second)
        self.assertAlmostEqual(first["annualized_mean"], values.mean() * 12)
        self.assertLessEqual(first["lower_bound"], first["upper_bound"])

    def test_full_sample_blocks_preserve_the_mean(self):
        values = [0.01, 0.02, -0.01, 0.00]
        result = circular_block_bootstrap_mean(
            values,
            block_months=4,
            repetitions=20,
            seed=3,
        )
        expected = np.mean(values) * 12
        self.assertAlmostEqual(result["lower_bound"], expected)
        self.assertAlmostEqual(result["upper_bound"], expected)

    def test_invalid_inputs(self):
        valid = [0.01, 0.02, -0.01]
        invalid_calls = [
            {"values": ["bad"], "block_months": 1, "repetitions": 10, "seed": 1},
            {"values": [], "block_months": 1, "repetitions": 10, "seed": 1},
            {"values": [0.01, np.nan], "block_months": 1, "repetitions": 10, "seed": 1},
            {"values": valid, "block_months": 4, "repetitions": 10, "seed": 1},
            {"values": valid, "block_months": 1.5, "repetitions": 10, "seed": 1},
            {"values": valid, "block_months": 1, "repetitions": 0, "seed": 1},
            {"values": valid, "block_months": 1, "repetitions": 10, "seed": -1},
            {
                "values": valid,
                "block_months": 1,
                "repetitions": 10,
                "seed": 1,
                "periods_per_year": 12.5,
            },
            {
                "values": valid,
                "block_months": 1,
                "repetitions": 10,
                "seed": 1,
                "confidence_level": np.inf,
            },
            {
                "values": valid,
                "block_months": 1,
                "repetitions": 10,
                "seed": 1,
                "confidence_level": 1.0,
            },
        ]
        for arguments in invalid_calls:
            with self.subTest(arguments=arguments):
                with self.assertRaises((TypeError, ValueError)):
                    circular_block_bootstrap_mean(**arguments)


if __name__ == "__main__":
    unittest.main()
