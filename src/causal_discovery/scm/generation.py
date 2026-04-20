"""Random parameterization helpers for linear-Gaussian SCMs."""

from __future__ import annotations

import numpy as np

from causal_discovery.config import WeightRange
from causal_discovery.core import DAG, LinearGaussianSCM


def sample_edge_weights(
    dag: DAG,
    weight_range: WeightRange,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a full weight matrix with nonzero values exactly on DAG edges."""
    weights = np.zeros((dag.num_nodes, dag.num_nodes), dtype=float)
    neg_range, pos_range = weight_range
    for src, dst in sorted(dag.edges):
        if rng.random() < 0.5:
            weights[src, dst] = rng.uniform(*neg_range)
        else:
            weights[src, dst] = rng.uniform(*pos_range)
    return weights


def sample_noise_variances(d: int, noise_var: float | np.ndarray) -> np.ndarray:
    """Expand scalar noise variance config to a length-d vector."""
    if np.isscalar(noise_var):
        variances = np.full(d, float(noise_var), dtype=float)
    else:
        variances = np.array(noise_var, dtype=float, copy=True)
    if variances.shape != (d,):
        raise ValueError(f"Noise variance vector must have length {d}, got {variances.shape}")
    if np.any(variances <= 0.0):
        raise ValueError("All noise variances must be positive")
    return variances


def parameterize_linear_gaussian_scm(
    dag: DAG,
    weight_range: WeightRange,
    noise_var: float | np.ndarray,
    rng: np.random.Generator,
) -> LinearGaussianSCM:
    """Sample a linear-Gaussian SCM over a fixed DAG."""
    weights = sample_edge_weights(dag, weight_range, rng)
    noise_variances = sample_noise_variances(dag.num_nodes, noise_var)
    intercepts = np.zeros(dag.num_nodes, dtype=float)
    return LinearGaussianSCM(
        dag=dag,
        weights=weights,
        noise_variances=noise_variances,
        intercepts=intercepts,
    )
