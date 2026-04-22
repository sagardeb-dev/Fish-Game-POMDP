"""Push-based runtime session for active causal discovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causal_discovery.benchmark import BenchmarkInstance
from causal_discovery.sampling import sample_interventional_data
from causal_discovery.scoring.submission import GraphSubmission


@dataclass(frozen=True, slots=True)
class SessionOutput:
    """Terminal record produced when an agent submits its estimated graph."""

    submission: GraphSubmission


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
        return self._instance.observational_data.copy()

    def intervene(self, var: int, value: float) -> np.ndarray:
        if self._sealed:
            raise RuntimeError("Session is sealed")
        if not self._observed:
            raise RuntimeError("Call observe() before requesting interventions")
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

    def submit_graph(self, submission: GraphSubmission) -> SessionOutput:
        self._ensure_can_submit()
        if not isinstance(submission, GraphSubmission):
            raise TypeError(
                "submit_graph expects GraphSubmission; use submit_adjacency_matrix for matrices"
            )
        if submission.num_nodes != self._instance.config.d:
            raise ValueError(
                "Submitted graph node count "
                f"{submission.num_nodes} does not match d={self._instance.config.d}"
            )

        self._sealed = True
        sealed_submission = GraphSubmission(
            num_nodes=submission.num_nodes,
            directed_edges=submission.directed_edges,
            undirected_edges=submission.undirected_edges,
            interventions_used=self._interventions_used,
        )
        return SessionOutput(submission=sealed_submission)

    def submit_adjacency_matrix(self, matrix: np.ndarray) -> SessionOutput:
        self._ensure_can_submit()
        if np.asarray(matrix).shape != (self._instance.config.d, self._instance.config.d):
            raise ValueError(
                "Submitted graph matrix must have shape "
                f"({self._instance.config.d}, {self._instance.config.d})"
            )
        return self.submit_graph(
            GraphSubmission.from_adjacency_matrix(
                matrix,
                interventions_used=self._interventions_used,
            )
        )

    def submit_cpdag(
        self,
        directed_edges: set[tuple[int, int]] | frozenset[tuple[int, int]],
        undirected_edges: set[tuple[int, int]] | frozenset[tuple[int, int]],
    ) -> SessionOutput:
        self._ensure_can_submit()
        return self.submit_graph(
            GraphSubmission(
                num_nodes=self._instance.config.d,
                directed_edges=frozenset(directed_edges),
                undirected_edges=frozenset(undirected_edges),
                interventions_used=self._interventions_used,
            )
        )

    def _ensure_can_submit(self) -> None:
        if self._sealed:
            raise RuntimeError("Session is sealed")
        if not self._observed:
            raise RuntimeError("Call observe() before submitting a graph")
