"""Random DAG generation for benchmark instances."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from causal_discovery.core import DAG


def sample_random_topological_order(d: int, rng: np.random.Generator) -> tuple[int, ...]:
    """Sample a random topological order over node ids 0..d-1."""
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    return tuple(int(node) for node in rng.permutation(d))


def valid_forward_edges(order: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Enumerate all forward edges implied by a topological order."""
    return tuple((order[i], order[j]) for i, j in combinations(range(len(order)), 2))


def sample_random_dag(d: int, k: int, rng: np.random.Generator) -> DAG:
    """Sample a random DAG with exactly k forward edges."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    order = sample_random_topological_order(d, rng)
    edges = valid_forward_edges(order)
    if k > len(edges):
        raise ValueError(f"k must be <= {len(edges)} for d={d}, got {k}")
    chosen = rng.choice(len(edges), size=k, replace=False)
    sampled_edges = [edges[int(idx)] for idx in np.sort(chosen)]
    return DAG.from_edges(d, sampled_edges)
