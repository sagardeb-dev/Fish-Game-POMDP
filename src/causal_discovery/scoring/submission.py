"""Agent graph submission representation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from causal_discovery.core import DAG
from causal_discovery.equivalence import CPDAG
from causal_discovery.equivalence.cpdag import canonical_undirected_edge


DirectedEdge = tuple[int, int]
UndirectedEdge = tuple[int, int]


@dataclass(frozen=True, slots=True)
class GraphSubmission:
    """Directed and unresolved-undirected graph submitted by an agent."""

    num_nodes: int
    directed_edges: frozenset[DirectedEdge]
    undirected_edges: frozenset[UndirectedEdge]
    interventions_used: int = 0

    def __post_init__(self) -> None:
        if self.num_nodes <= 0:
            raise ValueError(f"num_nodes must be positive, got {self.num_nodes}")
        if self.interventions_used < 0:
            raise ValueError(
                f"interventions_used must be non-negative, got {self.interventions_used}"
            )

        directed = frozenset((int(src), int(dst)) for src, dst in self.directed_edges)
        undirected = frozenset(
            canonical_undirected_edge(int(a), int(b))
            for a, b in self.undirected_edges
        )
        object.__setattr__(self, "directed_edges", directed)
        object.__setattr__(self, "undirected_edges", undirected)
        self.validate()

    @classmethod
    def from_dag(cls, dag: DAG, interventions_used: int = 0) -> "GraphSubmission":
        """Create a fully directed submission from a DAG."""
        return cls(
            num_nodes=dag.num_nodes,
            directed_edges=frozenset(dag.edges),
            undirected_edges=frozenset(),
            interventions_used=interventions_used,
        )

    @classmethod
    def from_cpdag(
        cls, cpdag: CPDAG, interventions_used: int = 0
    ) -> "GraphSubmission":
        """Create a partially directed submission from a CPDAG."""
        return cls(
            num_nodes=cpdag.num_nodes,
            directed_edges=frozenset(cpdag.directed_edges),
            undirected_edges=frozenset(cpdag.undirected_edges),
            interventions_used=interventions_used,
        )

    @classmethod
    def from_adjacency_matrix(
        cls, matrix: np.ndarray, interventions_used: int = 0
    ) -> "GraphSubmission":
        """Parse a binary adjacency matrix.

        One asymmetric 1 encodes a directed edge. Symmetric 1s encode an
        unresolved undirected edge.
        """
        array = np.asarray(matrix)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError(f"Submitted graph matrix must be square, got {array.shape}")
        unique_values = set(np.unique(array).tolist())
        if not unique_values.issubset({0, 1}):
            raise ValueError(
                "Submitted graph matrix must be binary "
                f"(entries in {{0, 1}}), got values {sorted(unique_values)}"
            )
        if np.any(np.diag(array) != 0):
            raise ValueError("Submitted graph matrix must have zero diagonal")

        directed: set[DirectedEdge] = set()
        undirected: set[UndirectedEdge] = set()
        d = int(array.shape[0])
        for i in range(d):
            for j in range(i + 1, d):
                ij = int(array[i, j])
                ji = int(array[j, i])
                if ij == 1 and ji == 1:
                    undirected.add((i, j))
                elif ij == 1:
                    directed.add((i, j))
                elif ji == 1:
                    directed.add((j, i))

        return cls(
            num_nodes=d,
            directed_edges=frozenset(directed),
            undirected_edges=frozenset(undirected),
            interventions_used=interventions_used,
        )

    @classmethod
    def from_edges(
        cls,
        num_nodes: int,
        directed_edges: Iterable[DirectedEdge],
        undirected_edges: Iterable[Iterable[int]],
        interventions_used: int = 0,
    ) -> "GraphSubmission":
        """Create a submission from explicit edge iterables."""
        return cls(
            num_nodes=num_nodes,
            directed_edges=frozenset(directed_edges),
            undirected_edges=frozenset(
                canonical_undirected_edge(*tuple(edge)) for edge in undirected_edges
            ),
            interventions_used=interventions_used,
        )

    @property
    def skeleton_edges(self) -> frozenset[UndirectedEdge]:
        directed_skeleton = {
            canonical_undirected_edge(src, dst) for src, dst in self.directed_edges
        }
        return frozenset(directed_skeleton | set(self.undirected_edges))

    def validate(self) -> None:
        for src, dst in self.directed_edges:
            if src == dst:
                raise ValueError(f"Directed self-loop detected at node {src}")
            self._validate_node(src)
            self._validate_node(dst)
            if (dst, src) in self.directed_edges:
                raise ValueError(f"Directed 2-cycle detected between {src} and {dst}")

        for a, b in self.undirected_edges:
            self._validate_node(a)
            self._validate_node(b)
            if a >= b:
                raise ValueError(f"Undirected edge must be canonical, got {(a, b)}")

        for src, dst in self.directed_edges:
            if canonical_undirected_edge(src, dst) in self.undirected_edges:
                raise ValueError(f"Pair {(src, dst)} cannot be both directed and undirected")

        DAG.from_edges(self.num_nodes, self.directed_edges)

    def _validate_node(self, node: int) -> None:
        if node < 0 or node >= self.num_nodes:
            raise ValueError(
                f"Node {node} out of range for graph with {self.num_nodes} nodes"
            )
