"""Theory-layer graph derivations and rejection policies."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from causal_discovery.core import DAG, LinearGaussianSCM
from causal_discovery.equivalence.cpdag import (
    CPDAG,
    DirectedEdge,
    canonical_undirected_edge,
)
from causal_discovery.scm import (
    implied_covariance,
    is_near_singular,
    relevant_structural_partial_correlations,
)


@dataclass(slots=True)
class _MutablePDAG:
    num_nodes: int
    directed_edges: set[DirectedEdge]
    undirected_edges: set[tuple[int, int]]

    @classmethod
    def from_cpdag(cls, cpdag: CPDAG) -> "_MutablePDAG":
        return cls(
            num_nodes=cpdag.num_nodes,
            directed_edges=set(cpdag.directed_edges),
            undirected_edges=set(cpdag.undirected_edges),
        )

    def to_cpdag(self) -> CPDAG:
        return CPDAG(
            num_nodes=self.num_nodes,
            directed_edges=frozenset(self.directed_edges),
            undirected_edges=frozenset(self.undirected_edges),
        )

    def has_directed_edge(self, src: int, dst: int) -> bool:
        return (src, dst) in self.directed_edges

    def has_undirected_edge(self, a: int, b: int) -> bool:
        if a == b:
            return False
        return canonical_undirected_edge(a, b) in self.undirected_edges

    def has_any_edge(self, a: int, b: int) -> bool:
        return (
            self.has_directed_edge(a, b)
            or self.has_directed_edge(b, a)
            or self.has_undirected_edge(a, b)
        )

    def parents(self, node: int) -> tuple[int, ...]:
        return tuple(sorted(src for src, dst in self.directed_edges if dst == node))

    def children(self, node: int) -> tuple[int, ...]:
        return tuple(sorted(dst for src, dst in self.directed_edges if src == node))

    def skeleton_neighbors(self, node: int) -> tuple[int, ...]:
        neighbors = set()
        for src, dst in self.directed_edges:
            if src == node:
                neighbors.add(dst)
            elif dst == node:
                neighbors.add(src)
        for a, b in self.undirected_edges:
            if a == node:
                neighbors.add(b)
            elif b == node:
                neighbors.add(a)
        return tuple(sorted(neighbors))

    def orient(self, src: int, dst: int) -> bool:
        edge = canonical_undirected_edge(src, dst)
        if edge not in self.undirected_edges:
            return False
        self.undirected_edges.remove(edge)
        self.directed_edges.add((src, dst))
        return True


def dag_to_cpdag(dag: DAG) -> CPDAG:
    """Derive the observational CPDAG of a DAG."""
    pdag = _MutablePDAG(
        num_nodes=dag.num_nodes,
        directed_edges=set(),
        undirected_edges={canonical_undirected_edge(src, dst) for src, dst in dag.edges},
    )
    _orient_unshielded_colliders(dag, pdag)
    _apply_meek_closure(pdag, include_rule4=False)
    return pdag.to_cpdag()


def reject_graph(dag: DAG, cpdag: CPDAG) -> bool:
    """Return whether a generated graph structure is too trivial for the benchmark."""
    if cpdag.num_undirected_edges == 0:
        return True
    if cpdag.num_undirected_edges < 2:
        return True
    if _all_undirected_edges_share_one_node(cpdag):
        return True
    if _dag_skeleton_is_disconnected(dag):
        return True
    return False


def reject_scm(scm: LinearGaussianSCM, faithfulness_eps: float) -> bool:
    """Return whether an SCM should be rejected by the benchmark policy."""
    cov = implied_covariance(scm)
    if is_near_singular(cov):
        return True
    return any(
        abs(value) < faithfulness_eps
        for value in relevant_structural_partial_correlations(scm)
    )


def compute_minimum_intervention_set(dag: DAG, cpdag: CPDAG) -> tuple[int, ...]:
    """Return the lexicographically first minimum sufficient intervention set."""
    if dag.num_nodes != cpdag.num_nodes:
        raise ValueError("DAG and CPDAG must have the same number of nodes")

    for size in range(dag.num_nodes + 1):
        for targets in combinations(range(dag.num_nodes), size):
            completed = _orient_with_intervention_targets(cpdag, dag, targets)
            if _matches_true_dag(completed, dag):
                return targets
    raise ValueError("No sufficient intervention set found")


def reject_intervention_profile(intervention_set: tuple[int, ...], cpdag: CPDAG) -> bool:
    """Return whether an intervention profile is too trivial for the benchmark."""
    return len(intervention_set) == 0


def _orient_unshielded_colliders(dag: DAG, pdag: _MutablePDAG) -> None:
    for child in dag.nodes:
        parents = dag.parents(child)
        for left, right in combinations(parents, 2):
            if _dag_has_skeleton_edge(dag, left, right):
                continue
            pdag.orient(left, child)
            pdag.orient(right, child)


def _apply_meek_closure(pdag: _MutablePDAG, include_rule4: bool) -> None:
    while True:
        changed = (
            _apply_rule1(pdag)
            or _apply_rule2(pdag)
            or _apply_rule3(pdag)
            or (include_rule4 and _apply_rule4(pdag))
        )
        if not changed:
            return


def _apply_rule1(pdag: _MutablePDAG) -> bool:
    for a, b in sorted(pdag.undirected_edges):
        for src, dst in ((a, b), (b, a)):
            for parent in pdag.parents(src):
                if not pdag.has_any_edge(parent, dst):
                    return pdag.orient(src, dst)
    return False


def _apply_rule2(pdag: _MutablePDAG) -> bool:
    for a, b in sorted(pdag.undirected_edges):
        for src, dst in ((a, b), (b, a)):
            for middle in pdag.children(src):
                if pdag.has_directed_edge(middle, dst):
                    return pdag.orient(src, dst)
    return False


def _apply_rule3(pdag: _MutablePDAG) -> bool:
    for a, b in sorted(pdag.undirected_edges):
        for src, dst in ((a, b), (b, a)):
            neighbors = [node for node in pdag.skeleton_neighbors(src) if node != dst]
            for left, right in combinations(neighbors, 2):
                if pdag.has_directed_edge(left, dst) and pdag.has_directed_edge(right, dst):
                    if not pdag.has_any_edge(left, right):
                        return pdag.orient(src, dst)
    return False


def _apply_rule4(pdag: _MutablePDAG) -> bool:
    for a, b in sorted(pdag.undirected_edges):
        for src, dst in ((a, b), (b, a)):
            neighbors = [node for node in pdag.skeleton_neighbors(src) if node != dst]
            for c in neighbors:
                if not pdag.has_directed_edge(c, dst):
                    continue
                for d in neighbors:
                    if d == c:
                        continue
                    if pdag.has_directed_edge(d, c) and not pdag.has_any_edge(d, dst):
                        return pdag.orient(src, dst)
    return False


def _all_undirected_edges_share_one_node(cpdag: CPDAG) -> bool:
    if not cpdag.undirected_edges:
        return False
    iterator = iter(cpdag.undirected_edges)
    common = set(next(iterator))
    for edge in iterator:
        common &= set(edge)
        if not common:
            return False
    return True


def _dag_skeleton_is_disconnected(dag: DAG) -> bool:
    if dag.num_nodes == 1:
        return False
    adjacency = {node: set() for node in dag.nodes}
    for src, dst in dag.edges:
        adjacency[src].add(dst)
        adjacency[dst].add(src)
    frontier = [0]
    seen = {0}
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return len(seen) != dag.num_nodes


def _dag_has_skeleton_edge(dag: DAG, a: int, b: int) -> bool:
    return dag.has_edge(a, b) or dag.has_edge(b, a)


def _orient_with_intervention_targets(
    cpdag: CPDAG,
    dag: DAG,
    targets: tuple[int, ...],
) -> CPDAG:
    pdag = _MutablePDAG.from_cpdag(cpdag)
    target_set = set(targets)
    incident_edges = sorted(
        edge
        for edge in pdag.undirected_edges
        if edge[0] in target_set or edge[1] in target_set
    )
    for a, b in incident_edges:
        if dag.has_edge(a, b):
            pdag.orient(a, b)
        elif dag.has_edge(b, a):
            pdag.orient(b, a)
        else:
            raise ValueError(f"True DAG does not contain skeleton edge {(a, b)}")
    _apply_meek_closure(pdag, include_rule4=True)
    return pdag.to_cpdag()


def _matches_true_dag(cpdag: CPDAG, dag: DAG) -> bool:
    return cpdag.num_undirected_edges == 0 and cpdag.directed_edges == dag.edges
