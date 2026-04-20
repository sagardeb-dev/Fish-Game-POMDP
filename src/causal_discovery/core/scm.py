"""Linear-Gaussian SCM representation for benchmark truth objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causal_discovery.core.dag import DAG
from causal_discovery.core.permutation import Permutation


@dataclass(frozen=True, slots=True)
class LinearGaussianSCM:
    """Linear-Gaussian SCM with independent exogenous noise."""

    dag: DAG
    weights: np.ndarray
    noise_variances: np.ndarray
    intercepts: np.ndarray

    def __post_init__(self) -> None:
        weights = np.array(self.weights, dtype=float, copy=True)
        noise_variances = np.array(self.noise_variances, dtype=float, copy=True)
        intercepts = np.array(self.intercepts, dtype=float, copy=True)
        weights.setflags(write=False)
        noise_variances.setflags(write=False)
        intercepts.setflags(write=False)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "noise_variances", noise_variances)
        object.__setattr__(self, "intercepts", intercepts)
        self.validate()

    @classmethod
    def from_dag(
        cls,
        dag: DAG,
        weights: np.ndarray,
        noise_var: float | np.ndarray,
        intercepts: np.ndarray | None = None,
    ) -> "LinearGaussianSCM":
        weight_matrix = np.array(weights, dtype=float, copy=True)
        if np.isscalar(noise_var):
            noise_variances = np.full(dag.num_nodes, float(noise_var), dtype=float)
        else:
            noise_variances = np.array(noise_var, dtype=float, copy=True)
        if intercepts is None:
            intercept_vector = np.zeros(dag.num_nodes, dtype=float)
        else:
            intercept_vector = np.array(intercepts, dtype=float, copy=True)
        return cls(
            dag=dag,
            weights=weight_matrix,
            noise_variances=noise_variances,
            intercepts=intercept_vector,
        )

    def parents(self, node: int) -> tuple[int, ...]:
        return self.dag.parents(node)

    def weight(self, src: int, dst: int) -> float:
        if not self.dag.has_edge(src, dst):
            raise ValueError(f"Edge ({src}, {dst}) is not present in the DAG")
        return float(self.weights[src, dst])

    def implied_adjacency(self) -> np.ndarray:
        return (self.weights != 0.0).astype(np.int8)

    def to_weight_matrix(self) -> np.ndarray:
        return np.array(self.weights, copy=True)

    def relabel(self, permutation: Permutation) -> "LinearGaussianSCM":
        if permutation.size != self.dag.num_nodes:
            raise ValueError(
                "Permutation size does not match SCM dimensionality: "
                f"{permutation.size} != {self.dag.num_nodes}"
            )
        return LinearGaussianSCM(
            dag=self.dag.relabel(permutation),
            weights=permutation.apply_square_matrix(self.weights),
            noise_variances=permutation.apply_vector(self.noise_variances),
            intercepts=permutation.apply_vector(self.intercepts),
        )

    def validate(self) -> None:
        d = self.dag.num_nodes
        if self.weights.shape != (d, d):
            raise ValueError(
                f"Weight matrix shape must be ({d}, {d}), got {self.weights.shape}"
            )
        if self.noise_variances.shape != (d,):
            raise ValueError(
                "Noise variance vector must have length "
                f"{d}, got shape {self.noise_variances.shape}"
            )
        if self.intercepts.shape != (d,):
            raise ValueError(
                f"Intercept vector must have length {d}, got shape {self.intercepts.shape}"
            )
        if np.any(np.diag(self.weights) != 0.0):
            raise ValueError("Weight matrix must have zeros on the diagonal")
        if np.any(self.noise_variances <= 0.0):
            raise ValueError("All noise variances must be positive")

        dag_adjacency = self.dag.adjacency_matrix().astype(bool)
        nonzero_weights = self.weights != 0.0

        if np.any(nonzero_weights & ~dag_adjacency):
            raise ValueError("Weight matrix must have nonzero entries only on DAG edges")
        if np.any(dag_adjacency & ~nonzero_weights):
            raise ValueError("Every DAG edge must have a corresponding nonzero weight")
