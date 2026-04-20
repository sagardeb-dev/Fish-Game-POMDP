"""Observational and interventional sampling from linear-Gaussian SCMs."""

from __future__ import annotations

import numpy as np

from causal_discovery.core import LinearGaussianSCM
from causal_discovery.scm import topological_order_from_dag


def sample_observational_data(
    scm: LinearGaussianSCM,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample observational rows from an SCM."""
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    return _sample_scm_rows(scm, n_samples, rng, intervention=None)


def sample_interventional_data(
    scm: LinearGaussianSCM,
    var: int,
    value: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample interventional rows under do(V_var = value)."""
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if var < 0 or var >= scm.dag.num_nodes:
        raise ValueError(f"Intervention variable {var} out of range for d={scm.dag.num_nodes}")
    return _sample_scm_rows(scm, n_samples, rng, intervention=(var, float(value)))


def _sample_scm_rows(
    scm: LinearGaussianSCM,
    n_samples: int,
    rng: np.random.Generator,
    intervention: tuple[int, float] | None,
) -> np.ndarray:
    d = scm.dag.num_nodes
    samples = np.zeros((n_samples, d), dtype=float)
    order = topological_order_from_dag(scm.dag)
    intervention_var = -1 if intervention is None else intervention[0]
    intervention_value = 0.0 if intervention is None else intervention[1]

    for node in order:
        if node == intervention_var:
            samples[:, node] = intervention_value
            continue
        parents = scm.parents(node)
        if parents:
            parent_values = samples[:, parents]
            parent_weights = scm.weights[np.array(parents), node]
            structural = parent_values @ parent_weights
        else:
            structural = np.zeros(n_samples, dtype=float)
        noise = rng.normal(
            loc=0.0,
            scale=np.sqrt(float(scm.noise_variances[node])),
            size=n_samples,
        )
        samples[:, node] = float(scm.intercepts[node]) + structural + noise

    return samples

