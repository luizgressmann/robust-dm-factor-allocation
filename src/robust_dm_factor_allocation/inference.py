"""Bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral, Real

import numpy as np
import pandas as pd

from ._validation import finite_real, positive_integer
from ._validation import periods_per_year as validate_periods_per_year
from .config import DEFAULT_PERIODS_PER_YEAR


def circular_block_bootstrap_mean(
    values: Iterable[float] | pd.Series,
    *,
    block_months: Integral,
    repetitions: Integral,
    seed: Integral,
    periods_per_year: Integral = DEFAULT_PERIODS_PER_YEAR,
    confidence_level: Real = 0.95,
) -> pd.Series:
    """Estimate an annualized mean and interval with circular blocks."""
    try:
        series = pd.Series(values, copy=False).astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("values must be numeric") from error
    observations = series.to_numpy(dtype=float)
    if observations.ndim != 1 or observations.size == 0:
        raise ValueError("values must contain at least one observation")
    if not np.isfinite(observations).all():
        raise ValueError("values must be finite and complete")

    block_length = positive_integer(block_months, "block_months")
    if block_length > len(observations):
        raise ValueError("block_months must not exceed the sample length")
    draws = positive_integer(repetitions, "repetitions")
    random_seed = positive_integer(seed, "seed", minimum=0)
    frequency = validate_periods_per_year(periods_per_year)
    confidence = finite_real(confidence_level, "confidence_level")
    if not 0 < confidence < 1:
        raise ValueError("confidence_level must be in (0, 1)")

    generator = np.random.default_rng(random_seed)
    blocks_per_sample = int(np.ceil(len(observations) / block_length))
    starts = generator.integers(
        0,
        len(observations),
        size=(draws, blocks_per_sample),
    )
    offsets = np.arange(block_length)
    indices = (starts[:, :, None] + offsets) % len(observations)
    indices = indices.reshape(draws, -1)[:, : len(observations)]
    bootstrap_means = observations[indices].mean(axis=1) * frequency

    tail_probability = (1 - confidence) / 2
    lower, upper = np.quantile(
        bootstrap_means,
        [tail_probability, 1 - tail_probability],
    )
    return pd.Series(
        {
            "annualized_mean": float(observations.mean() * frequency),
            "lower_bound": float(lower),
            "upper_bound": float(upper),
        },
        dtype=float,
    )
