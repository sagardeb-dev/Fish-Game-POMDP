"""Push-based runtime session for active causal discovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causal_discovery.benchmark import BenchmarkInstance
from causal_discovery.core import DAG
from causal_discovery.sampling import sample_interventional_data


@dataclass(frozen=True, slots=True)
class SessionOutput:
    """Terminal record produced when an agent submits its estimated graph."""

    estimated_graph: DAG
    interventions_used: int


class BenchmarkEnv:
    """Push-based environment wrapping a single benchmark instance."""

    def __init__(self, instance: BenchmarkInstance, rng: np.random.Generator) -> None:
        self._instance = instance
        self._rng = rng
        self._remaining_budget = int(instance.intervention_budget)
        self._observed = False
        self._sealed = False
        self._interventions_used = 0

    @property
    def remaining_budget(self) -> int:
        return self._remaining_budget

    @property
    def num_variables(self) -> int:
        return self._instance.config.d

    def observe(self) -> np.ndarray:
        if self._sealed:
            raise RuntimeError("Session is sealed")
        if self._observed:
            raise RuntimeError("Observational data already delivered")
        self._observed = True
        return self._instance.observational_data

    def intervene(self, var: int, value: float) -> np.ndarray:
        if self._sealed:
            raise RuntimeError("Session is sealed")
        if self._remaining_budget <= 0:
            raise RuntimeError("Intervention budget exhausted")
        samples = sample_interventional_data(
            self._instance.scm,
            var=var,
            value=float(value),
            n_samples=self._instance.config.n_int,
            rng=self._rng,
        )
        self._remaining_budget -= 1
        self._interventions_used += 1
        return samples

    def submit_graph(self, matrix: np.ndarray) -> SessionOutput:
        if self._sealed:
            raise RuntimeError("Session is sealed")
        dag = _dag_from_submission_matrix(matrix, self._instance.config.d)
        self._sealed = True
        return SessionOutput(estimated_graph=dag, interventions_used=self._interventions_used)


def _dag_from_submission_matrix(matrix: np.ndarray, d: int) -> DAG:
    array = np.asarray(matrix)
    if array.shape != (d, d):
        raise ValueError(
            f"Submitted graph must have shape ({d}, {d}), got {array.shape}"
        )
    unique_values = set(np.unique(array).tolist())
    if not unique_values.issubset({0, 1}):
        raise ValueError(
            f"Submitted graph must be binary (entries in {{0, 1}}), got values {sorted(unique_values)}"
        )
    if np.any(np.diag(array) != 0):
        raise ValueError("Submitted graph must have zero diagonal (no self-loops)")
    edges = [(int(src), int(dst)) for src, dst in zip(*np.nonzero(array))]
    try:
        return DAG.from_edges(d, edges)
    except ValueError as exc:
        raise ValueError(f"Submitted graph is not a valid DAG: {exc}") from exc
