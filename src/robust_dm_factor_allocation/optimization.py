"""Long-only portfolio weights."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
import pandas as pd

from ._validation import finite_real, positive_integer
from .config import DEFAULT_MAX_WEIGHT

ArrayLike: TypeAlias = Sequence[float] | np.ndarray | pd.Series
MatrixLike: TypeAlias = Sequence[Sequence[float]] | np.ndarray | pd.DataFrame

METHODS = (
    "equal_weight",
    "inverse_volatility",
    "sample_gmv",
    "oas_gmv",
    "erc",
)


class ExactERCCapError(ValueError):
    """Raised when a weight cap prevents exact ERC."""


def _validate_cap(n_assets: int, max_weight: Real) -> float:
    cap = finite_real(max_weight, "max_weight")
    if not 0 < cap <= 1:
        raise ValueError("max_weight must be in (0, 1]")
    if n_assets * cap < 1 - 1e-12:
        raise ValueError("max_weight is infeasible")
    return cap


def _positive_tolerance(value: Real) -> float:
    tolerance = finite_real(value, "tolerance")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    return tolerance


def project_weights(
    weights: ArrayLike,
    max_weight: Real = 1.0,
    *,
    tolerance: Real = 1e-12,
    max_iterations: Integral = 200,
) -> np.ndarray:
    """Project weights onto a capped, long-only simplex."""
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("weights must be a finite one-dimensional array")
    cap = _validate_cap(values.size, max_weight)
    convergence_tolerance = _positive_tolerance(tolerance)
    iterations = positive_integer(max_iterations, "max_iterations")

    lower = float(np.min(values - cap))
    upper = float(np.max(values))
    projected = np.clip(values, 0, cap)
    for _ in range(iterations):
        midpoint = (lower + upper) / 2
        projected = np.clip(values - midpoint, 0, cap)
        total = float(projected.sum())
        if abs(total - 1) <= convergence_tolerance:
            break
        if total > 1:
            lower = midpoint
        else:
            upper = midpoint
    else:
        raise RuntimeError("capped-simplex projection did not converge")

    residual = 1 - float(projected.sum())
    if residual > 0:
        room = cap - projected
        for index in np.flatnonzero(room > 0):
            addition = min(residual, float(room[index]))
            projected[index] += addition
            residual -= addition
            if residual <= convergence_tolerance:
                break
    elif residual < 0:
        for index in np.flatnonzero(projected > 0):
            removal = min(-residual, float(projected[index]))
            projected[index] -= removal
            residual += removal
            if residual >= -convergence_tolerance:
                break
    if not np.isclose(projected.sum(), 1.0, atol=10 * convergence_tolerance, rtol=0):
        raise RuntimeError("capped-simplex projection produced an invalid sum")
    if projected.min() < -convergence_tolerance or projected.max() > cap + convergence_tolerance:
        raise RuntimeError("capped-simplex projection violated its bounds")
    return np.clip(projected, 0, cap)


def _capped_proportions(scores: ArrayLike, max_weight: Real) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a nonempty one-dimensional array")
    cap = _validate_cap(values.size, max_weight)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("scores must be finite and nonnegative")
    if values.sum() <= 0:
        raise ValueError("at least one score must be positive")

    weights = np.zeros_like(values)
    active = np.ones(values.size, dtype=bool)
    remaining = 1.0
    while active.any():
        active_scores = values[active]
        if active_scores.sum() == 0:
            proposed = np.full(active.sum(), remaining / active.sum())
        else:
            proposed = remaining * active_scores / active_scores.sum()
        over_cap = proposed > cap + 1e-14
        active_indices = np.flatnonzero(active)
        if not over_cap.any():
            weights[active_indices] = proposed
            break
        capped_indices = active_indices[over_cap]
        weights[capped_indices] = cap
        active[capped_indices] = False
        remaining = 1 - float(weights.sum())
    return project_weights(weights, cap)


def _covariance_values(
    covariance: MatrixLike,
) -> tuple[np.ndarray, pd.Index | None]:
    labels: pd.Index | None = None
    if isinstance(covariance, pd.DataFrame):
        if not covariance.index.is_unique or not covariance.columns.is_unique:
            raise ValueError("covariance labels must be unique")
        if not covariance.index.equals(covariance.columns):
            raise ValueError("covariance index and columns must match in the same order")
        labels = covariance.columns.copy()
        try:
            values = covariance.to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("covariance must be numeric") from error
    else:
        try:
            values = np.asarray(covariance, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("covariance must be numeric") from error
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] == 0:
        raise ValueError("covariance must be a nonempty square matrix")
    if not np.isfinite(values).all():
        raise ValueError("covariance must be finite")

    scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    if not np.allclose(values, values.T, rtol=1e-10, atol=1e-12 * scale):
        raise ValueError("covariance must be symmetric")
    values = (values + values.T) / 2
    eigenvalues = np.linalg.eigvalsh(values)
    if eigenvalues.min() < -1e-10 * scale:
        raise ValueError("covariance must be positive semidefinite")
    if eigenvalues.min() < 0:
        values = values + np.eye(values.shape[0]) * (-float(eigenvalues.min()))
    return values, labels


def _return_weights(
    weights: np.ndarray,
    labels: Sequence[object] | pd.Index | None,
) -> np.ndarray | pd.Series:
    if labels is None:
        return weights
    index = pd.Index(labels)
    if len(index) != len(weights) or not index.is_unique:
        raise ValueError("labels must be unique and match n_assets")
    return pd.Series(weights, index=index, name="weight")


def _returns_values(
    returns: Sequence[Sequence[float]] | np.ndarray | pd.DataFrame,
) -> tuple[np.ndarray, pd.Index | None]:
    labels: pd.Index | None = None
    if isinstance(returns, pd.DataFrame):
        if not returns.columns.is_unique:
            raise ValueError("returns columns must be unique")
        labels = returns.columns.copy()
        try:
            values = returns.to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("returns must be numeric") from error
    else:
        try:
            values = np.asarray(returns, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("returns must be numeric") from error
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError("returns must contain at least two rows and one column")
    if not np.isfinite(values).all():
        raise ValueError("returns must be finite and complete")
    return values, labels


def sample_covariance(
    returns: Sequence[Sequence[float]] | np.ndarray | pd.DataFrame,
) -> np.ndarray | pd.DataFrame:
    """Calculate the unbiased sample covariance."""
    values, labels = _returns_values(returns)
    covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    if labels is None:
        return covariance
    return pd.DataFrame(covariance, index=labels, columns=labels)


def oas_covariance(
    returns: Sequence[Sequence[float]] | np.ndarray | pd.DataFrame,
) -> np.ndarray | pd.DataFrame:
    """Estimate Oracle Approximating Shrinkage covariance toward ``mu * I``.

    The calculation uses centered returns and the ``1 / n`` covariance required
    by the OAS formula.
    """
    values, labels = _returns_values(returns)
    centered = values - values.mean(axis=0)
    covariance = centered.T @ centered / len(centered)
    n_assets = covariance.shape[0]
    trace = float(np.trace(covariance))
    trace_squared_covariance = float(np.trace(covariance @ covariance))
    mean_variance = trace / n_assets
    denominator = (len(centered) + 1 - 2 / n_assets) * (
        trace_squared_covariance - trace**2 / n_assets
    )
    if denominator <= 0:
        shrinkage = 1.0
    else:
        shrinkage = min(
            ((1 - 2 / n_assets) * trace_squared_covariance + trace**2) / denominator,
            1.0,
        )
    target = np.eye(n_assets) * mean_variance
    estimate = (1 - shrinkage) * covariance + shrinkage * target
    estimate = (estimate + estimate.T) / 2
    if labels is None:
        return estimate
    return pd.DataFrame(estimate, index=labels, columns=labels)


def equal_weight(
    n_assets: Integral,
    max_weight: Real = 1.0,
    labels: Sequence[object] | pd.Index | None = None,
) -> np.ndarray | pd.Series:
    """Return equal weights under a common cap."""
    assets = positive_integer(n_assets, "n_assets")
    cap = _validate_cap(assets, max_weight)
    weights = project_weights(np.full(assets, 1 / assets), cap)
    return _return_weights(weights, labels)


def inverse_volatility_weights(
    covariance: MatrixLike,
    max_weight: Real = 1.0,
) -> np.ndarray | pd.Series:
    """Weight assets by inverse volatility."""
    values, labels = _covariance_values(covariance)
    variances = np.diag(values)
    if (variances <= 0).any():
        raise ValueError("variances must be positive")
    weights = _capped_proportions(1 / np.sqrt(variances), max_weight)
    return _return_weights(weights, labels)


def gmv_weights(
    covariance: MatrixLike,
    max_weight: Real = 1.0,
    tolerance: Real = 1e-10,
    max_iterations: Integral = 20_000,
) -> np.ndarray | pd.Series:
    """Solve the capped, long-only GMV problem with projected gradient descent."""
    values, labels = _covariance_values(covariance)
    n_assets = values.shape[0]
    cap = _validate_cap(n_assets, max_weight)
    convergence_tolerance = _positive_tolerance(tolerance)
    iterations = positive_integer(max_iterations, "max_iterations")

    largest_eigenvalue = float(np.linalg.eigvalsh(values).max())
    if largest_eigenvalue <= np.finfo(float).eps:
        return _return_weights(np.full(n_assets, 1 / n_assets), labels)

    weights = np.full(n_assets, 1 / n_assets)
    for _ in range(iterations):
        candidate = project_weights(
            weights - (values @ weights) / largest_eigenvalue,
            cap,
        )
        if np.linalg.norm(candidate - weights, ord=np.inf) <= convergence_tolerance:
            weights = candidate
            break
        weights = candidate
    else:
        raise RuntimeError("GMV optimizer did not converge")

    residual = np.linalg.norm(
        weights - project_weights(weights - (values @ weights) / largest_eigenvalue, cap),
        ord=np.inf,
    )
    if residual > max(10 * convergence_tolerance, 1e-9):
        raise RuntimeError("GMV optimizer failed its convergence check")
    return _return_weights(weights, labels)


def sample_gmv_weights(
    returns: Sequence[Sequence[float]] | np.ndarray | pd.DataFrame,
    max_weight: Real = 1.0,
) -> np.ndarray | pd.Series:
    """Calculate GMV weights from the sample covariance."""
    return gmv_weights(sample_covariance(returns), max_weight)


def oas_gmv_weights(
    returns: Sequence[Sequence[float]] | np.ndarray | pd.DataFrame,
    max_weight: Real = 1.0,
) -> np.ndarray | pd.Series:
    """Calculate GMV weights from an OAS covariance estimate."""
    return gmv_weights(oas_covariance(returns), max_weight)


def risk_contributions(
    weights: ArrayLike,
    covariance: MatrixLike,
    normalized: bool = True,
) -> np.ndarray | pd.Series:
    """Calculate component variance contributions."""
    values, labels = _covariance_values(covariance)
    if not isinstance(normalized, (bool, np.bool_)):
        raise TypeError("normalized must be boolean")
    if isinstance(weights, pd.Series) and labels is not None:
        if not weights.index.is_unique:
            raise ValueError("weight labels must be unique")
        missing_labels = labels.difference(weights.index)
        extra_labels = weights.index.difference(labels)
        if len(missing_labels) or len(extra_labels):
            raise ValueError("weight labels must match covariance labels")
        weight_values = weights.reindex(labels).to_numpy(dtype=float)
    else:
        weight_values = np.asarray(weights, dtype=float)
    if weight_values.shape != (values.shape[0],) or not np.isfinite(weight_values).all():
        raise ValueError("weights do not match covariance")
    contributions = weight_values * (values @ weight_values)
    if normalized:
        total = float(contributions.sum())
        if total <= 0:
            raise ValueError("portfolio variance must be positive")
        contributions = contributions / total
    return _return_weights(contributions, labels)


def erc_weights(
    covariance: MatrixLike,
    max_weight: Real = 1.0,
    tolerance: Real = 1e-11,
    max_iterations: Integral = 10_000,
) -> np.ndarray | pd.Series:
    """Calculate exact equal-risk-contribution weights.

    If the exact solution exceeds ``max_weight``, the function raises
    ``ExactERCCapError``.
    """
    values, labels = _covariance_values(covariance)
    n_assets = values.shape[0]
    cap = _validate_cap(n_assets, max_weight)
    convergence_tolerance = _positive_tolerance(tolerance)
    iterations = positive_integer(max_iterations, "max_iterations")

    scale = max(float(np.trace(values) / n_assets), np.finfo(float).tiny)
    if (np.diag(values) <= np.finfo(float).eps * scale).any():
        raise ValueError("ERC requires strictly positive asset variances")
    minimum_eigenvalue = float(np.linalg.eigvalsh(values).min())
    if minimum_eigenvalue <= np.finfo(float).eps * scale:
        values = values + np.eye(n_assets) * scale * 1e-12

    budgets = np.full(n_assets, 1 / n_assets)
    solution = np.sqrt(budgets / np.diag(values))
    converged = False
    for _ in range(iterations):
        previous = solution.copy()
        for index in range(n_assets):
            variance = values[index, index]
            cross_term = values[index] @ solution - variance * solution[index]
            discriminant = cross_term**2 + 4 * variance * budgets[index]
            solution[index] = (-cross_term + np.sqrt(max(discriminant, 0))) / (2 * variance)
        relative_change = np.linalg.norm(solution - previous, ord=1) / max(
            np.linalg.norm(previous, ord=1),
            np.finfo(float).tiny,
        )
        if relative_change <= convergence_tolerance:
            converged = True
            break
    if not converged:
        raise RuntimeError("ERC optimizer did not converge")

    weights = solution / solution.sum()
    contributions = weights * (values @ weights)
    contributions = contributions / contributions.sum()
    if np.max(np.abs(contributions - budgets)) > max(100 * convergence_tolerance, 1e-8):
        raise RuntimeError("ERC optimizer failed its risk-contribution check")
    if weights.max() > cap + 10 * convergence_tolerance:
        raise ExactERCCapError("max_weight binds the exact ERC solution")
    return _return_weights(weights, labels)


def estimate_weights(
    returns: Sequence[Sequence[float]] | np.ndarray | pd.DataFrame,
    method: str,
    max_weight: Real = DEFAULT_MAX_WEIGHT,
) -> np.ndarray | pd.Series:
    """Estimate portfolio weights with the selected method."""
    values, labels = _returns_values(returns)
    frame: np.ndarray | pd.DataFrame
    frame = pd.DataFrame(values, columns=labels) if labels is not None else values
    if not isinstance(method, str) or not method.strip():
        raise ValueError("method must be a nonempty string")
    name = method.strip().lower()
    if name not in METHODS:
        raise ValueError(f"unknown method: {method}; expected one of {METHODS}")

    if name == "equal_weight":
        return equal_weight(values.shape[1], max_weight, labels)
    covariance = sample_covariance(frame)
    if name == "inverse_volatility":
        return inverse_volatility_weights(covariance, max_weight)
    if name == "sample_gmv":
        return gmv_weights(covariance, max_weight)
    if name == "oas_gmv":
        return oas_gmv_weights(frame, max_weight)
    return erc_weights(covariance, max_weight)
