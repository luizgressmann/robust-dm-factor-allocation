from itertools import product

import numpy as np
import pandas as pd


METHODS = (
    "equal_weight",
    "inverse_volatility",
    "sample_gmv",
    "shrinkage_gmv",
    "erc",
)


def _validate_cap(n_assets, max_weight):
    max_weight = float(max_weight)
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1]")
    if n_assets * max_weight < 1 - 1e-12:
        raise ValueError("max_weight is infeasible")
    return max_weight


def project_weights(weights, max_weight=1.0):
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("weights must be a finite one-dimensional array")
    cap = _validate_cap(values.size, max_weight)
    lower = float(np.min(values - cap))
    upper = float(np.max(values))
    for _ in range(100):
        midpoint = (lower + upper) / 2
        candidate = np.clip(values - midpoint, 0, cap)
        if candidate.sum() > 1:
            lower = midpoint
        else:
            upper = midpoint
    projected = np.clip(values - (lower + upper) / 2, 0, cap)
    residual = 1 - projected.sum()
    if residual > 0:
        for index in np.argsort(values)[::-1]:
            addition = min(residual, cap - projected[index])
            projected[index] += addition
            residual -= addition
            if residual <= 1e-14:
                break
    elif residual < 0:
        for index in np.argsort(values):
            removal = min(-residual, projected[index])
            projected[index] -= removal
            residual += removal
            if residual >= -1e-14:
                break
    return projected


def _capped_proportions(scores, max_weight):
    scores = np.asarray(scores, dtype=float)
    cap = _validate_cap(scores.size, max_weight)
    if scores.ndim != 1 or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("scores must be finite and nonnegative")
    if scores.sum() <= 0:
        raise ValueError("at least one score must be positive")
    weights = np.zeros_like(scores)
    active = np.ones(scores.size, dtype=bool)
    remaining = 1.0
    while active.any():
        active_scores = scores[active]
        if active_scores.sum() == 0:
            proposed = np.full(active.sum(), remaining / active.sum())
        else:
            proposed = remaining * active_scores / active_scores.sum()
        over = proposed > cap + 1e-14
        active_indices = np.flatnonzero(active)
        if not over.any():
            weights[active_indices] = proposed
            break
        capped_indices = active_indices[over]
        weights[capped_indices] = cap
        active[capped_indices] = False
        remaining = 1 - weights.sum()
    return project_weights(weights, cap)


def _covariance_values(covariance):
    labels = None
    if isinstance(covariance, pd.DataFrame):
        if not covariance.index.equals(covariance.columns):
            raise ValueError("covariance index and columns must match")
        labels = covariance.columns
        values = covariance.to_numpy(dtype=float)
    else:
        values = np.asarray(covariance, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] == 0:
        raise ValueError("covariance must be a nonempty square matrix")
    if not np.isfinite(values).all():
        raise ValueError("covariance must be finite")
    values = (values + values.T) / 2
    scale = max(float(np.max(np.abs(values))), 1.0)
    eigenvalues = np.linalg.eigvalsh(values)
    if eigenvalues.min() < -1e-10 * scale:
        raise ValueError("covariance must be positive semidefinite")
    if eigenvalues.min() < 0:
        values = values + np.eye(values.shape[0]) * (-eigenvalues.min())
    return values, labels


def _return_weights(weights, labels):
    if labels is None:
        return weights
    return pd.Series(weights, index=labels, name="weight")


def _returns_values(returns):
    labels = None
    if isinstance(returns, pd.DataFrame):
        labels = returns.columns
        values = returns.to_numpy(dtype=float)
    else:
        values = np.asarray(returns, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError("returns must contain at least two rows")
    if not np.isfinite(values).all():
        raise ValueError("returns must be finite")
    return values, labels


def sample_covariance(returns):
    values, labels = _returns_values(returns)
    covariance = np.cov(values, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    if labels is None:
        return covariance
    return pd.DataFrame(covariance, index=labels, columns=labels)


def shrinkage_covariance(returns, shrinkage=None):
    values, labels = _returns_values(returns)
    centered = values - values.mean(axis=0)
    covariance = centered.T @ centered / len(centered)
    n_assets = covariance.shape[0]
    mean_variance = np.trace(covariance) / n_assets
    if shrinkage is None:
        alpha = np.mean(covariance**2)
        denominator = (len(centered) + 1) * (
            alpha - mean_variance**2 / n_assets
        )
        if denominator <= 0:
            shrinkage = 1.0
        else:
            shrinkage = min((alpha + mean_variance**2) / denominator, 1.0)
    shrinkage = float(shrinkage)
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be in [0, 1]")
    target = np.eye(n_assets) * mean_variance
    covariance = (1 - shrinkage) * covariance + shrinkage * target
    if labels is None:
        return covariance
    return pd.DataFrame(covariance, index=labels, columns=labels)


def equal_weight(n_assets, max_weight=1.0, labels=None):
    n_assets = int(n_assets)
    if n_assets <= 0:
        raise ValueError("n_assets must be positive")
    cap = _validate_cap(n_assets, max_weight)
    weights = project_weights(np.full(n_assets, 1 / n_assets), cap)
    return _return_weights(weights, labels)


def inverse_volatility_weights(covariance, max_weight=1.0):
    values, labels = _covariance_values(covariance)
    variances = np.diag(values)
    if (variances <= 0).any():
        raise ValueError("variances must be positive")
    weights = _capped_proportions(1 / np.sqrt(variances), max_weight)
    return _return_weights(weights, labels)


def gmv_weights(covariance, max_weight=1.0, tolerance=1e-10):
    values, labels = _covariance_values(covariance)
    n_assets = values.shape[0]
    cap = _validate_cap(n_assets, max_weight)
    best_weights = None
    best_variance = np.inf
    for status in product((0, 1, 2), repeat=n_assets):
        status = np.asarray(status)
        free = np.flatnonzero(status == 1)
        capped = np.flatnonzero(status == 2)
        remaining = 1 - len(capped) * cap
        if remaining < -tolerance or remaining > len(free) * cap + tolerance:
            continue
        candidate = np.zeros(n_assets)
        candidate[capped] = cap
        if len(free) == 0:
            if abs(remaining) > tolerance:
                continue
        else:
            free_covariance = values[np.ix_(free, free)]
            fixed_effect = values[np.ix_(free, capped)] @ candidate[capped]
            system = np.block(
                [
                    [free_covariance, np.ones((len(free), 1))],
                    [np.ones((1, len(free))), np.zeros((1, 1))],
                ]
            )
            right_hand_side = np.r_[-fixed_effect, remaining]
            solution = np.linalg.lstsq(system, right_hand_side, rcond=None)[0]
            candidate[free] = solution[:-1]
        if (
            candidate.min() < -tolerance
            or candidate.max() > cap + tolerance
            or not np.isclose(candidate.sum(), 1, atol=tolerance)
        ):
            continue
        candidate = np.clip(candidate, 0, cap)
        candidate = candidate / candidate.sum()
        variance = float(candidate @ values @ candidate)
        if variance < best_variance:
            best_variance = variance
            best_weights = candidate
    if best_weights is None:
        raise RuntimeError("unable to solve constrained GMV portfolio")
    return _return_weights(best_weights, labels)


def sample_gmv_weights(returns, max_weight=1.0):
    return gmv_weights(sample_covariance(returns), max_weight)


def shrinkage_gmv_weights(returns, max_weight=1.0, shrinkage=None):
    return gmv_weights(shrinkage_covariance(returns, shrinkage), max_weight)


def risk_contributions(weights, covariance, normalized=True):
    values, labels = _covariance_values(covariance)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (values.shape[0],) or not np.isfinite(weights).all():
        raise ValueError("weights do not match covariance")
    contributions = weights * (values @ weights)
    if normalized:
        total = contributions.sum()
        if total <= 0:
            raise ValueError("portfolio variance must be positive")
        contributions = contributions / total
    return _return_weights(contributions, labels)


def erc_weights(covariance, max_weight=1.0, tolerance=1e-11, max_iterations=10000):
    values, labels = _covariance_values(covariance)
    n_assets = values.shape[0]
    cap = _validate_cap(n_assets, max_weight)
    scale = max(float(np.trace(values) / n_assets), 1.0)
    minimum_eigenvalue = float(np.linalg.eigvalsh(values).min())
    if minimum_eigenvalue <= np.finfo(float).eps * scale:
        values = values + np.eye(n_assets) * scale * 1e-12
    budgets = np.full(n_assets, 1 / n_assets)
    solution = np.ones(n_assets)
    for _ in range(max_iterations):
        previous = solution.copy()
        for index in range(n_assets):
            variance = values[index, index]
            cross_term = values[index] @ solution - variance * solution[index]
            discriminant = cross_term**2 + 4 * variance * budgets[index]
            solution[index] = (
                -cross_term + np.sqrt(max(discriminant, 0))
            ) / (2 * variance)
        if np.linalg.norm(solution - previous, ord=1) <= tolerance:
            break
    weights = solution / solution.sum()
    weights = project_weights(weights, cap)
    return _return_weights(weights, labels)


def estimate_weights(returns, method, max_weight=1.0, shrinkage=None):
    values, labels = _returns_values(returns)
    frame = pd.DataFrame(values, columns=labels) if labels is not None else values
    method = str(method).lower()
    if method in {"equal_weight", "1/n"}:
        return equal_weight(values.shape[1], max_weight, labels)
    covariance = sample_covariance(frame)
    if method in {"inverse_volatility", "inverse_vol"}:
        return inverse_volatility_weights(covariance, max_weight)
    if method == "sample_gmv":
        return gmv_weights(covariance, max_weight)
    if method == "shrinkage_gmv":
        return shrinkage_gmv_weights(frame, max_weight, shrinkage)
    if method in {"erc", "risk_parity"}:
        return erc_weights(covariance, max_weight)
    raise ValueError(f"unknown method: {method}")
