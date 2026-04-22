"""Lightweight statistical tools for LLM-stats agents."""

from __future__ import annotations

import math

import numpy as np


def correlation(data: np.ndarray, i: int, j: int) -> float:
    _validate_pair(data, i, j)
    value = float(np.corrcoef(data[:, i], data[:, j])[0, 1])
    return float(np.clip(value, -1.0, 1.0))


def partial_correlation(
    data: np.ndarray, i: int, j: int, conditioning_on: tuple[int, ...]
) -> float:
    _validate_pair(data, i, j)
    z = _normalize_conditioning_set(data, i, j, conditioning_on)
    if not z:
        return correlation(data, i, j)

    design = _design_matrix(data, z)
    xi = _residualize(data[:, i], design)
    xj = _residualize(data[:, j], design)
    value = float(np.corrcoef(xi, xj)[0, 1])
    return float(np.clip(value, -1.0, 1.0))


def independence_test(
    data: np.ndarray,
    i: int,
    j: int,
    conditioning_on: tuple[int, ...],
    alpha: float = 0.05,
) -> tuple[bool, float]:
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    z = _normalize_conditioning_set(data, i, j, conditioning_on)
    n = int(data.shape[0])
    dof = n - len(z) - 3
    if dof <= 0:
        raise ValueError(
            "Not enough samples for Fisher-Z test with conditioning set size "
            f"{len(z)} and n={n}"
        )

    rho = partial_correlation(data, i, j, z)
    rho = float(np.clip(rho, -0.999999, 0.999999))
    fisher_z = 0.5 * math.log((1.0 + rho) / (1.0 - rho)) * math.sqrt(dof)
    p_value = math.erfc(abs(fisher_z) / math.sqrt(2.0))
    return bool(p_value > alpha), float(p_value)


def _validate_pair(data: np.ndarray, i: int, j: int) -> None:
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data.shape}")
    d = int(data.shape[1])
    if i < 0 or i >= d or j < 0 or j >= d:
        raise ValueError(f"indices out of range for d={d}: {(i, j)}")
    if i == j:
        raise ValueError("indices i and j must differ")


def _normalize_conditioning_set(
    data: np.ndarray, i: int, j: int, conditioning_on: tuple[int, ...]
) -> tuple[int, ...]:
    d = int(data.shape[1])
    unique = sorted(set(int(x) for x in conditioning_on))
    if i in unique or j in unique:
        raise ValueError("conditioning_on must not include target variables")
    for idx in unique:
        if idx < 0 or idx >= d:
            raise ValueError(f"conditioning index out of range for d={d}: {idx}")
    return tuple(unique)


def _design_matrix(data: np.ndarray, conditioning_on: tuple[int, ...]) -> np.ndarray:
    z = data[:, list(conditioning_on)]
    ones = np.ones((z.shape[0], 1), dtype=float)
    return np.hstack([ones, z])


def _residualize(target: np.ndarray, design: np.ndarray) -> np.ndarray:
    coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return target - (design @ coeffs)
