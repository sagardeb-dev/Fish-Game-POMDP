"""Deterministic diagnostics for linear-Gaussian SCMs."""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

from causal_discovery.core import DAG, LinearGaussianSCM


def topological_order_from_dag(dag: DAG) -> tuple[int, ...]:
    """Return one valid topological order for a DAG."""
    indegree = [0] * dag.num_nodes
    children: list[list[int]] = [[] for _ in range(dag.num_nodes)]
    for src, dst in dag.edges:
        indegree[dst] += 1
        children[src].append(dst)
    queue = deque(sorted(node for node, degree in enumerate(indegree) if degree == 0))
    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in sorted(children[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != dag.num_nodes:
        raise ValueError("DAG must be acyclic")
    return tuple(order)


def scm_weight_matrix(scm: LinearGaussianSCM) -> np.ndarray:
    """Return the SEM coefficient matrix B in X = B X + eps form."""
    return scm.to_weight_matrix().T


def implied_covariance(scm: LinearGaussianSCM) -> np.ndarray:
    """Compute the exact covariance implied by the linear SEM."""
    b = scm_weight_matrix(scm)
    identity = np.eye(scm.dag.num_nodes, dtype=float)
    noise_cov = np.diag(scm.noise_variances)
    transform = np.linalg.inv(identity - b)
    cov = transform @ noise_cov @ transform.T
    return 0.5 * (cov + cov.T)


def is_near_singular(cov: np.ndarray, tol: float = 1e-8) -> bool:
    """Check whether a covariance matrix is numerically near-singular."""
    eigenvalues = np.linalg.eigvalsh(cov)
    return bool(np.min(eigenvalues) <= tol)


def partial_correlation(
    cov: np.ndarray,
    i: int,
    j: int,
    cond_set: Iterable[int],
) -> float:
    """Compute partial correlation rho(i, j | cond_set) from covariance."""
    cond = tuple(sorted(set(cond_set)))
    indices = (i, j, *cond)
    sub_cov = cov[np.ix_(indices, indices)]
    precision = np.linalg.inv(sub_cov)
    denom = precision[0, 0] * precision[1, 1]
    if denom <= 0.0:
        raise ValueError("Partial correlation is undefined for non-positive denominator")
    rho = -precision[0, 1] / np.sqrt(denom)
    return float(np.clip(rho, -1.0, 1.0))


def relevant_structural_partial_correlations(scm: LinearGaussianSCM) -> list[float]:
    """Compute one targeted partial correlation per edge for faithfulness checks."""
    cov = implied_covariance(scm)
    results: list[float] = []
    for src, dst in sorted(scm.dag.edges):
        cond_set = tuple(parent for parent in scm.parents(dst) if parent != src)
        results.append(partial_correlation(cov, src, dst, cond_set))
    return results
