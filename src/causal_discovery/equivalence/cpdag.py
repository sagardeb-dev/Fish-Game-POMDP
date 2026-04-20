"""Benchmark-owned CPDAG representation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from causal_discovery.core.permutation import Permutation


DirectedEdge = tuple[int, int]
UndirectedEdge = tuple[int, int]


def canonical_undirected_edge(a: int, b: int) -> UndirectedEdge:
    """Return a canonical undirected edge representation."""
    if a == b:
        raise ValueError("Undirected self-loops are not allowed")
    return (a, b) if a < b else (b, a)


@dataclass(frozen=True, slots=True)
class CPDAG:
    """Immutable CPDAG with directed and undirected edge sets."""

    num_nodes: int
    directed_edges: frozenset[DirectedEdge]
    undirected_edges: frozenset[UndirectedEdge]

    def __post_init__(self) -> None:
        if self.num_nodes <= 0:
            raise ValueError(f"num_nodes must be positive, got {self.num_nodes}")
        normalized_directed = frozenset((int(src), int(dst)) for src, dst in self.directed_edges)
        normalized_undirected = frozenset(
            canonical_undirected_edge(int(a), int(b)) for a, b in self.undirected_edges
        )
        object.__setattr__(self, "directed_edges", normalized_directed)
        object.__setattr__(self, "undirected_edges", normalized_undirected)
        self.validate()

    @property
    def nodes(self) -> tuple[int, ...]:
        return tuple(range(self.num_nodes))

    @property
    def num_directed_edges(self) -> int:
        return len(self.directed_edges)

    @property
    def num_undirected_edges(self) -> int:
        return len(self.undirected_edges)

    def has_directed_edge(self, src: int, dst: int) -> bool:
        self._validate_node(src)
        self._validate_node(dst)
        return (src, dst) in self.directed_edges

    def has_undirected_edge(self, a: int, b: int) -> bool:
        self._validate_node(a)
        self._validate_node(b)
        if a == b:
            return False
        return canonical_undirected_edge(a, b) in self.undirected_edges

    def parents(self, node: int) -> tuple[int, ...]:
        self._validate_node(node)
        return tuple(sorted(src for src, dst in self.directed_edges if dst == node))

    def children(self, node: int) -> tuple[int, ...]:
        self._validate_node(node)
        return tuple(sorted(dst for src, dst in self.directed_edges if src == node))

    def undirected_neighbors(self, node: int) -> tuple[int, ...]:
        self._validate_node(node)
        neighbors = set()
        for a, b in self.undirected_edges:
            if a == node:
                neighbors.add(b)
            elif b == node:
                neighbors.add(a)
        return tuple(sorted(neighbors))

    def skeleton_neighbors(self, node: int) -> tuple[int, ...]:
        self._validate_node(node)
        neighbors = set(self.undirected_neighbors(node))
        for src, dst in self.directed_edges:
            if src == node:
                neighbors.add(dst)
            elif dst == node:
                neighbors.add(src)
        return tuple(sorted(neighbors))

    def relabel(self, permutation: Permutation) -> "CPDAG":
        if permutation.size != self.num_nodes:
            raise ValueError(
                "Permutation size "
                f"{permutation.size} does not match CPDAG size {self.num_nodes}"
            )
        relabeled_directed = permutation.apply_edges(self.directed_edges)
        relabeled_undirected = frozenset(
            canonical_undirected_edge(permutation.apply_node(a), permutation.apply_node(b))
            for a, b in self.undirected_edges
        )
        return CPDAG(
            num_nodes=self.num_nodes,
            directed_edges=relabeled_directed,
            undirected_edges=relabeled_undirected,
        )

    def validate(self) -> None:
        for src, dst in self.directed_edges:
            if src == dst:
                raise ValueError(f"Directed self-loop detected at node {src}")
            self._validate_node(src)
            self._validate_node(dst)

        for a, b in self.undirected_edges:
            self._validate_node(a)
            self._validate_node(b)
            if a >= b:
                raise ValueError(f"Undirected edge must be canonical, got {(a, b)}")

        for src, dst in self.directed_edges:
            if canonical_undirected_edge(src, dst) in self.undirected_edges:
                raise ValueError(f"Pair {(src, dst)} cannot be both directed and undirected")
            if (dst, src) in self.directed_edges:
                raise ValueError(f"Directed 2-cycle detected between {src} and {dst}")

        if not self._directed_subgraph_is_acyclic():
            raise ValueError("Directed portion of CPDAG must be acyclic")

    def _validate_node(self, node: int) -> None:
        if node < 0 or node >= self.num_nodes:
            raise ValueError(
                f"Node {node} out of range for CPDAG with {self.num_nodes} nodes"
            )

    def _directed_subgraph_is_acyclic(self) -> bool:
        indegree = [0] * self.num_nodes
        children: list[list[int]] = [[] for _ in range(self.num_nodes)]
        for src, dst in self.directed_edges:
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
