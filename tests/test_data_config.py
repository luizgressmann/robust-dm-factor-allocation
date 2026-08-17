import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from robust_dm_factor_allocation.config import ProjectConfig, RobustnessConfig
from robust_dm_factor_allocation.data import (
    FACTOR_COLUMNS,
    load_config,
    load_monthly_returns,
    validate_returns,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class DataTest(unittest.TestCase):
    def make_returns(self):
        index = pd.date_range("2020-01-31", periods=4, freq=pd.offsets.MonthEnd())
        columns = ["extra", *FACTOR_COLUMNS, "market"]
        return pd.DataFrame(0.0, index=index, columns=columns)

    def test_validation_orders_required_columns_and_drops_extras(self):
        result = validate_returns(self.make_returns())
        self.assertEqual(result.columns.tolist(), [*FACTOR_COLUMNS, "market"])
        self.assertEqual(result.index.name, "date")

    def test_factor_and_benchmark_names(self):
        returns = self.make_returns()
        with self.assertRaises(ValueError):
            validate_returns(returns, [*FACTOR_COLUMNS, FACTOR_COLUMNS[0]])
        with self.assertRaises(ValueError):
            validate_returns(returns, FACTOR_COLUMNS, FACTOR_COLUMNS[0])
        duplicate = pd.concat([returns, returns[["market"]]], axis=1)
        with self.assertRaises(ValueError):
            validate_returns(duplicate)

    def test_missing_values_and_month_gaps(self):
        returns = self.make_returns()
        returns.iloc[1, 1] = np.nan
        with self.assertRaises(ValueError):
            validate_returns(returns)
        dropped = validate_returns(
            returns,
            require_complete_months=False,
            missing="drop",
        )
        self.assertEqual(len(dropped), 3)
        with self.assertRaisesRegex(ValueError, "missing months"):
            validate_returns(returns, missing="drop")

    def test_return_schema_errors_are_rejected(self):
        returns = self.make_returns()
        invalid_calls = [
            lambda: validate_returns([0.01]),
            lambda: validate_returns(returns, "value"),
            lambda: validate_returns(returns, ["value"]),
            lambda: validate_returns(returns, ["value", ""]),
            lambda: validate_returns(returns, FACTOR_COLUMNS, ""),
            lambda: validate_returns(returns, require_complete_months="yes"),
            lambda: validate_returns(returns.drop(columns="market")),
        ]
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, ValueError)):
                    call()

    def test_return_values_and_dates_are_validated(self):
        returns = self.make_returns()
        nonnumeric = returns.astype(object)
        nonnumeric.iloc[0, 1] = "bad"
        infinite = returns.copy()
        infinite.iloc[0, 1] = np.inf
        total_missing = returns.copy()
        total_missing.loc[:, FACTOR_COLUMNS[0]] = np.nan
        below_floor = returns.copy()
        below_floor.iloc[0, 1] = -1.0
        duplicate_date = returns.copy()
        duplicate_date.index = [returns.index[0], returns.index[0], *returns.index[2:]]
        unordered = returns.iloc[::-1]
        duplicate_month = returns.copy()
        duplicate_month.index = pd.to_datetime(
            ["2020-01-01", "2020-01-31", "2020-02-29", "2020-03-31"]
        )

        for frame in (
            nonnumeric,
            infinite,
            below_floor,
            duplicate_date,
            unordered,
            duplicate_month,
        ):
            with self.subTest(frame=frame):
                with self.assertRaises(ValueError):
                    validate_returns(frame)
        with self.assertRaisesRegex(ValueError, "complete observation"):
            validate_returns(total_missing, require_complete_months=False, missing="drop")

        string_dates = returns.copy()
        string_dates.index = string_dates.index.strftime("%Y-%m-%d")
        converted = validate_returns(string_dates)
        self.assertIsInstance(converted.index, pd.DatetimeIndex)
        invalid_dates = returns.copy()
        invalid_dates.index = ["not-a-date", "also-bad", "still-bad", "bad"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with self.assertRaisesRegex(ValueError, "contain dates"):
                validate_returns(invalid_dates)

        month_starts = returns.copy()
        month_starts.index = pd.date_range(
            "2020-01-01", periods=len(month_starts), freq=pd.offsets.MonthBegin()
        )
        with self.assertRaisesRegex(ValueError, "month-end"):
            validate_returns(month_starts)

        timezone_aware = returns.copy()
        timezone_aware.index = timezone_aware.index.tz_localize("UTC")
        with self.assertRaisesRegex(ValueError, "timezone-naive"):
            validate_returns(timezone_aware)

    def test_csv_loader_uses_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "returns.csv"
            frame = self.make_returns().drop(columns="extra")
            frame.index.name = "date"
            frame.to_csv(path)
            loaded = load_monthly_returns(path)
            pd.testing.assert_frame_equal(loaded, frame, check_freq=False)


class ConfigTest(unittest.TestCase):
    def test_repository_config(self):
        config = load_config(REPOSITORY_ROOT / "config" / "config.yaml")
        self.assertIsInstance(config, ProjectConfig)
        self.assertTrue(config.data_file.is_absolute())
        self.assertEqual(config.data_file.name, "monthly_returns_common.csv")
        self.assertEqual(
            config.factor_columns,
            ("value", "momentum", "quality", "small_cap"),
        )
        self.assertEqual(config.robustness.lookback_months, (60, 120))
        self.assertFalse(hasattr(config, "to_dict"))

    def test_invalid_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(
                "factor_columns: [a, b]\nbenchmark: a\nmaximum_weight: 0.5\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "benchmark"):
                load_config(path)

    def test_unknown_config_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("mystery_setting: 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown"):
                load_config(path)

    def test_non_mapping_yaml_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mapping"):
                load_config(path)

    def test_robustness_grid_validation(self):
        invalid_settings = [
            {"unknown": [1]},
            {"lookback_months": "60"},
            {"lookback_months": []},
            {"lookback_months": [60, 60]},
            {"maximum_weight": [0.0]},
            {"maximum_weight": [1.1]},
            {"transaction_cost_bps": [-1.0]},
            {"transaction_cost_bps": [10_000.0]},
        ]
        for settings in invalid_settings:
            with self.subTest(settings=settings):
                with self.assertRaises((TypeError, ValueError)):
                    RobustnessConfig.from_mapping(settings)

        robustness = RobustnessConfig.from_mapping(None)
        self.assertEqual(robustness.rebalance_frequency_months, (1, 3, 12))

    def test_project_config_rejects_invalid_fields(self):
        invalid_mappings = [
            {"data_file": 4},
            {"benchmark": ""},
            {"factor_columns": "value"},
            {"factor_columns": ["a"]},
            {"factor_columns": ["a", ""]},
            {"factor_columns": ["a", "a"]},
            {"maximum_weight": 0.0},
            {"maximum_weight": 0.1},
            {"transaction_cost_bps": -1},
            {"transaction_cost_bps": 10_000},
            {"random_seed": True},
            {"random_seed": -1},
            {"lookback_months": 1},
            {"rebalance_frequency_months": 0},
            {"lag_months": 0},
            {"periods_per_year": 0},
            {"bootstrap_block_months": 0},
            {"bootstrap_repetitions": 0},
            {"missing_policy": "guess"},
        ]
        for mapping in invalid_mappings:
            with self.subTest(mapping=mapping):
                with self.assertRaises((TypeError, ValueError)):
                    ProjectConfig.from_mapping(
                        mapping,
                        config_path=REPOSITORY_ROOT / "config" / "test.yaml",
                    )

    def test_config_paths_and_grid_feasibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            data_path = root / "returns.csv"
            config = ProjectConfig.from_mapping(
                {"data_file": data_path},
                config_path=root / "settings.yaml",
            )
            self.assertEqual(config.data_file, data_path)

        with self.assertRaisesRegex(ValueError, "robustness"):
            ProjectConfig.from_mapping(
                {
                    "factor_columns": ["a", "b"],
                    "maximum_weight": 0.5,
                    "robustness": {"maximum_weight": [0.4]},
                },
                config_path=REPOSITORY_ROOT / "config" / "test.yaml",
            )


if __name__ == "__main__":
    unittest.main()
