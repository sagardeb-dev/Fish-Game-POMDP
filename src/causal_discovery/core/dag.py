"""Directed acyclic graph representation for benchmark truth objects."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from causal_discovery.core.permutation import Permutation


Edge = tuple[int, int]


@dataclass(frozen=True, slots=True)
class DAG:
    """Minimal DAG object owned by the benchmark."""

    num_nodes: int
    edges: frozenset[Edge]

    def __post_init__(self) -> None:
        if self.num_nodes <= 0:
            raise ValueError(f"num_nodes must be positive, got {self.num_nodes}")
        normalized_edges = frozenset((int(src), int(dst)) for src, dst in self.edges)
        object.__setattr__(self, "edges", normalized_edges)
        self.validate()

    @classmethod
    def from_edges(cls, num_nodes: int, edges: Iterable[Edge]) -> "DAG":
        """Construct a DAG from an explicit edge iterable."""
        return cls(num_nodes=num_nodes, edges=frozenset(edges))

    @property
    def nodes(self) -> tuple[int, ...]:
        return tuple(range(self.num_nodes))

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def parents(self, node: int) -> tuple[int, ...]:
        self._validate_node(node)
        return tuple(sorted(src for src, dst in self.edges if dst == node))

    def children(self, node: int) -> tuple[int, ...]:
        self._validate_node(node)
        return tuple(sorted(dst for src, dst in self.edges if src == node))

    def has_edge(self, src: int, dst: int) -> bool:
        self._validate_node(src)
        self._validate_node(dst)
        return (src, dst) in self.edges

    def adjacency_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.int8)
        for src, dst in self.edges:
            matrix[src, dst] = 1
        return matrix

    def relabel(self, permutation: Permutation) -> "DAG":
        if permutation.size != self.num_nodes:
            raise ValueError(
                f"Permutation size {permutation.size} does not match DAG size {self.num_nodes}"
            )
        return DAG(num_nodes=self.num_nodes, edges=permutation.apply_edges(self.edges))

    def validate(self) -> None:
        for src, dst in self.edges:
            if src == dst:
                raise ValueError(f"Self-loop detected at node {src}")
            self._validate_node(src)
            self._validate_node(dst)
        if not self._is_acyclic():
            raise ValueError("Graph must be acyclic")

    def _validate_node(self, node: int) -> None:
        if node < 0 or node >= self.num_nodes:
            raise ValueError(
                f"Node {node} out of range for graph with {self.num_nodes} nodes"
            )

    def _is_acyclic(self) -> bool:
        indegree = [0] * self.num_nodes
        children: list[list[int]] = [[] for _ in range(self.num_nodes)]
        for src, dst in self.edges:
            indegree[dst] += 1
            children[src].append(dst)

        queue = deque(node for node, degree in enumerate(indegree) if degree == 0)
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for child in children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        return visited == self.num_nodes
