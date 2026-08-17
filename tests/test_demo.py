import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from robust_dm_factor_allocation.demo import (
    _parse_arguments,
    generate_synthetic_returns,
    main,
    run_demo,
)


class DemoTest(unittest.TestCase):
    def test_synthetic_returns_repeat_with_same_seed(self):
        first = generate_synthetic_returns(periods=180, seed=23)
        second = generate_synthetic_returns(periods=180, seed=23)

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.index.name, "date")
        self.assertEqual(
            first.columns.tolist(),
            ["value", "momentum", "quality", "small_cap", "market"],
        )
        self.assertEqual(first.shape, (180, 5))
        self.assertFalse(first.isna().any(axis=None))
        self.assertTrue((first > -1).all(axis=None))

    def test_demo_writes_four_results(self):
        expected = {
            "cap_sensitivity.png",
            "robustness.csv",
            "summary.csv",
            "wealth.png",
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = run_demo(directory, seed=29)
            self.assertEqual({path.name for path in destination.iterdir()}, expected)
            robustness = pd.read_csv(destination / "robustness.csv")
            self.assertEqual(len(robustness), 12)
            self.assertEqual(set(robustness["method"]), {"oas_gmv"})

            summary = pd.read_csv(destination / "summary.csv", index_col="method")
            self.assertEqual(
                summary.index.tolist(),
                ["equal_weight", "inverse_volatility", "sample_gmv", "oas_gmv", "erc", "market"],
            )
            for image_name in ("wealth.png", "cap_sensitivity.png"):
                self.assertGreater((Path(directory) / image_name).stat().st_size, 10_000)

    def test_generator_validation(self):
        invalid_arguments = [
            {"periods": True},
            {"periods": 179},
            {"seed": True},
        ]
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises((TypeError, ValueError)):
                    generate_synthetic_returns(**arguments)

    def test_cli_argument_parsing_and_dispatch(self):
        with patch("sys.argv", ["demo", "--output-dir", "custom", "--seed", "31"]):
            arguments = _parse_arguments()
        self.assertEqual(arguments.output_dir, Path("custom"))
        self.assertEqual(arguments.seed, 31)

        destination = Path("synthetic-output")
        with (
            patch("sys.argv", ["demo", "--output-dir", str(destination), "--seed", "31"]),
            patch("robust_dm_factor_allocation.demo.run_demo", return_value=destination) as run,
            patch("builtins.print") as output,
        ):
            self.assertEqual(main(), 0)
        run.assert_called_once_with(destination, seed=31)
        output.assert_called_once()


if __name__ == "__main__":
    unittest.main()
