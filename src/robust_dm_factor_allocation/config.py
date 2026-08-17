"""Project settings and defaults."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path

from ._validation import finite_real, missing_policy, periods_per_year, positive_integer

DEFAULT_FACTOR_COLUMNS = (
    "value",
    "momentum",
    "quality",
    "small_cap",
)
DEFAULT_BENCHMARK = "market"
DEFAULT_METHOD = "oas_gmv"
DEFAULT_LOOKBACK_MONTHS = 120
DEFAULT_REBALANCE_MONTHS = 3
DEFAULT_LAG_MONTHS = 1
DEFAULT_MAX_WEIGHT = 0.40
DEFAULT_TRANSACTION_COST_BPS = 10.0
DEFAULT_PERIODS_PER_YEAR = 12
DEFAULT_BOOTSTRAP_BLOCK_MONTHS = 12
DEFAULT_BOOTSTRAP_REPETITIONS = 2_000
DEFAULT_RANDOM_SEED = 7
DEFAULT_MISSING_POLICY = "raise"


def _column_names(values: Sequence[object], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of column names")
    if any(not isinstance(column, str) or not column.strip() for column in values):
        raise ValueError(f"{name} must contain nonempty strings")
    columns = tuple(column.strip() for column in values if isinstance(column, str))
    if len(columns) < 2:
        raise ValueError(f"{name} must contain at least two columns")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{name} must not contain duplicates")
    return columns


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _integer_tuple(values: object, name: str, *, minimum: int = 1) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(positive_integer(value, name, minimum=minimum) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _float_tuple(
    values: object,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(finite_real(value, name) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(value < minimum or value > maximum for value in result):
        raise ValueError(f"{name} values must lie in [{minimum}, {maximum}]")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class RobustnessConfig:
    """Sensitivity grid for OAS-GMV."""

    lookback_months: tuple[int, ...] = (60, 120)
    rebalance_frequency_months: tuple[int, ...] = (1, 3, 12)
    maximum_weight: tuple[float, ...] = (0.30, 0.40, 0.50)
    transaction_cost_bps: tuple[float, ...] = (0.0, 10.0, 25.0)

    def __post_init__(self) -> None:
        lookbacks = _integer_tuple(
            self.lookback_months,
            "robustness.lookback_months",
            minimum=2,
        )
        rebalances = _integer_tuple(
            self.rebalance_frequency_months,
            "robustness.rebalance_frequency_months",
        )
        caps = _float_tuple(
            self.maximum_weight,
            "robustness.maximum_weight",
            minimum=0.0,
            maximum=1.0,
        )
        if any(cap <= 0 for cap in caps):
            raise ValueError("robustness.maximum_weight values must be positive")
        costs = _float_tuple(
            self.transaction_cost_bps,
            "robustness.transaction_cost_bps",
            minimum=0.0,
            maximum=10_000.0,
        )
        if any(cost >= 10_000 for cost in costs):
            raise ValueError("robustness.transaction_cost_bps values must be less than 10000")
        object.__setattr__(self, "lookback_months", lookbacks)
        object.__setattr__(self, "rebalance_frequency_months", rebalances)
        object.__setattr__(self, "maximum_weight", caps)
        object.__setattr__(self, "transaction_cost_bps", costs)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None) -> RobustnessConfig:
        """Create settings from the YAML robustness section."""
        if values is None:
            return cls()
        if not isinstance(values, Mapping):
            raise TypeError("robustness must be a mapping")
        allowed = {definition.name for definition in fields(cls)}
        unknown = set(values).difference(allowed)
        if unknown:
            raise ValueError(f"unknown robustness settings: {sorted(unknown)}")
        return cls(**dict(values))


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Project settings."""

    data_file: Path
    benchmark: str = DEFAULT_BENCHMARK
    factor_columns: tuple[str, ...] = DEFAULT_FACTOR_COLUMNS
    lookback_months: int = DEFAULT_LOOKBACK_MONTHS
    rebalance_frequency_months: int = DEFAULT_REBALANCE_MONTHS
    lag_months: int = DEFAULT_LAG_MONTHS
    maximum_weight: float = DEFAULT_MAX_WEIGHT
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR
    bootstrap_block_months: int = DEFAULT_BOOTSTRAP_BLOCK_MONTHS
    bootstrap_repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS
    random_seed: int = DEFAULT_RANDOM_SEED
    missing_policy: str = DEFAULT_MISSING_POLICY
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.data_file, (str, Path)):
            raise TypeError("data_file must be a path")
        data_file = Path(self.data_file).resolve()

        benchmark = _nonempty_string(self.benchmark, "benchmark")
        factors = _column_names(self.factor_columns, "factor_columns")
        if benchmark in factors:
            raise ValueError("benchmark must not also be a factor column")

        cap = finite_real(self.maximum_weight, "maximum_weight")
        if not 0 < cap <= 1:
            raise ValueError("maximum_weight must be in (0, 1]")
        if len(factors) * cap < 1 - 1e-12:
            raise ValueError("maximum_weight is infeasible for factor_columns")

        costs = finite_real(self.transaction_cost_bps, "transaction_cost_bps")
        if not 0 <= costs < 10_000:
            raise ValueError("transaction_cost_bps must be in [0, 10000)")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise TypeError("random_seed must be an integer")
        if self.random_seed < 0:
            raise ValueError("random_seed must be nonnegative")
        if not isinstance(self.robustness, RobustnessConfig):
            raise TypeError("robustness must be a RobustnessConfig")
        if any(len(factors) * value < 1 - 1e-12 for value in self.robustness.maximum_weight):
            raise ValueError("a robustness maximum_weight is infeasible for factor_columns")

        object.__setattr__(self, "data_file", data_file)
        object.__setattr__(self, "benchmark", benchmark)
        object.__setattr__(self, "factor_columns", factors)
        object.__setattr__(
            self,
            "lookback_months",
            positive_integer(self.lookback_months, "lookback_months", minimum=2),
        )
        object.__setattr__(
            self,
            "rebalance_frequency_months",
            positive_integer(self.rebalance_frequency_months, "rebalance_frequency_months"),
        )
        object.__setattr__(
            self,
            "lag_months",
            positive_integer(self.lag_months, "lag_months"),
        )
        object.__setattr__(self, "maximum_weight", cap)
        object.__setattr__(self, "transaction_cost_bps", costs)
        object.__setattr__(self, "periods_per_year", periods_per_year(self.periods_per_year))
        object.__setattr__(
            self,
            "bootstrap_block_months",
            positive_integer(self.bootstrap_block_months, "bootstrap_block_months"),
        )
        object.__setattr__(
            self,
            "bootstrap_repetitions",
            positive_integer(self.bootstrap_repetitions, "bootstrap_repetitions"),
        )
        object.__setattr__(self, "missing_policy", missing_policy(self.missing_policy))

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object],
        *,
        config_path: str | Path,
    ) -> ProjectConfig:
        """Create settings from YAML and resolve relative data paths."""
        if not isinstance(values, Mapping):
            raise TypeError("config must be a mapping")
        allowed = {definition.name for definition in fields(cls)}
        unknown = set(values).difference(allowed)
        if unknown:
            raise ValueError(f"unknown configuration settings: {sorted(unknown)}")

        data = dict(values)
        path = Path(config_path).resolve()
        default_root = path.parent.parent if path.parent.name == "config" else path.parent

        data_value = data.get("data_file", "data/processed/monthly_returns_common.csv")
        if not isinstance(data_value, (str, Path)):
            raise TypeError("data_file must be a path")
        data_file = Path(data_value)
        if not data_file.is_absolute():
            data_file = default_root / data_file

        robustness_value = data.get("robustness")
        robustness = (
            robustness_value
            if isinstance(robustness_value, RobustnessConfig)
            else RobustnessConfig.from_mapping(robustness_value)
        )
        data["data_file"] = data_file
        data["robustness"] = robustness
        return cls(**data)
