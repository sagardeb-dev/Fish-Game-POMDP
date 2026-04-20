"""Scoring functions for submitted benchmark sessions."""

from __future__ import annotations

from dataclasses import dataclass

from causal_discovery.benchmark import BenchmarkInstance
from causal_discovery.core import DAG
from causal_discovery.equivalence import CPDAG
from causal_discovery.runtime import SessionOutput


@dataclass(frozen=True, slots=True)
class SessionScore:
    """Three independent scores for a single session submission."""

    observational: float
    interventional: float
    efficiency: float


def observational_score(estimated_dag: DAG, cpdag: CPDAG) -> float:
    """Combined skeleton + v-structure score against the observational ceiling."""
    cpdag_skeleton = {
        tuple(sorted((src, dst))) for src, dst in cpdag.directed_edges
    } | set(cpdag.undirected_edges)
    agent_skeleton = {tuple(sorted((src, dst))) for src, dst in estimated_dag.edges}

    denominator = len(cpdag_skeleton) + len(cpdag.directed_edges)
    if denominator == 0:
        return 1.0 if estimated_dag.num_edges == 0 else 0.0

    skeleton_hits = len(agent_skeleton & cpdag_skeleton)
    directed_hits = sum(
        1 for src, dst in cpdag.directed_edges if estimated_dag.has_edge(src, dst)
    )
    return (skeleton_hits + directed_hits) / denominator


def interventional_score(estimated_dag: DAG, true_dag: DAG) -> float:
    """F1 over directed edge sets against the true DAG."""
    estimated_edges = set(estimated_dag.edges)
    true_edges = set(true_dag.edges)

    tp = len(estimated_edges & true_edges)
    fp = len(estimated_edges - true_edges)
    fn = len(true_edges - estimated_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0

    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def efficiency_score(interventions_used: int, optimal_set_size: int) -> float:
    """Efficiency of intervention usage relative to the optimal set size."""
    if optimal_set_size <= 0:
        return 1.0
    denominator = max(interventions_used, optimal_set_size)
    return optimal_set_size / denominator


def score_session(instance: BenchmarkInstance, output: SessionOutput) -> SessionScore:
    """Score a completed session against hidden benchmark truth."""
    return SessionScore(
        observational=observational_score(
            output.estimated_graph, instance.observational_ceiling
        ),
        interventional=interventional_score(
            output.estimated_graph, instance.true_dag
        ),
        efficiency=efficiency_score(
            output.interventions_used, len(instance.optimal_intervention_set)
        ),
    )
