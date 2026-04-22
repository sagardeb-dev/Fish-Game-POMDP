"""Scoring functions for submitted benchmark graphs."""

from __future__ import annotations

from dataclasses import dataclass

from causal_discovery.benchmark import BenchmarkInstance
from causal_discovery.core import DAG
from causal_discovery.equivalence import CPDAG
from causal_discovery.equivalence.cpdag import canonical_undirected_edge
from causal_discovery.scoring.submission import GraphSubmission, UndirectedEdge


@dataclass(frozen=True, slots=True)
class ScoreReport:
    """Flat score report for one graph submission."""

    skeleton_precision: float
    skeleton_recall: float
    skeleton_f1: float
    compelled_precision: float
    compelled_recall: float
    compelled_f1: float
    directed_precision: float
    directed_recall: float
    directed_f1: float
    dag_shd: int
    interventions_used: int
    optimal_interventions: int
    efficiency: float


@dataclass(frozen=True, slots=True)
class _PRF:
    precision: float
    recall: float
    f1: float


def score_submission(
    instance: BenchmarkInstance, submission: GraphSubmission
) -> ScoreReport:
    """Score one submitted graph against the benchmark truth."""
    if submission.num_nodes != instance.config.d:
        raise ValueError(
            "Submission node count "
            f"{submission.num_nodes} does not match instance d={instance.config.d}"
        )

    skeleton = _skeleton_prf(submission, instance.observational_ceiling)
    compelled = _compelled_orientation_prf(
        submission, instance.observational_ceiling
    )
    directed = _directed_prf(submission, instance.true_dag)
    optimal = len(instance.optimal_intervention_set)

    return ScoreReport(
        skeleton_precision=skeleton.precision,
        skeleton_recall=skeleton.recall,
        skeleton_f1=skeleton.f1,
        compelled_precision=compelled.precision,
        compelled_recall=compelled.recall,
        compelled_f1=compelled.f1,
        directed_precision=directed.precision,
        directed_recall=directed.recall,
        directed_f1=directed.f1,
        dag_shd=dag_shd(submission, instance.true_dag),
        interventions_used=submission.interventions_used,
        optimal_interventions=optimal,
        efficiency=efficiency_score(submission.interventions_used, optimal),
    )


def efficiency_score(interventions_used: int, optimal_interventions: int) -> float:
    """Score intervention usage relative to the minimum resolving set size."""
    if interventions_used < 0:
        raise ValueError(
            f"interventions_used must be non-negative, got {interventions_used}"
        )
    if optimal_interventions < 0:
        raise ValueError(
            f"optimal_interventions must be non-negative, got {optimal_interventions}"
        )
    if optimal_interventions == 0:
        return 1.0 if interventions_used == 0 else 0.0
    return optimal_interventions / max(interventions_used, optimal_interventions)


def dag_shd(submission: GraphSubmission, true_dag: DAG) -> int:
    """Edge edit distance to the true DAG.

    Missing, extra, reversed, and unresolved true edges each count as one error.
    """
    if submission.num_nodes != true_dag.num_nodes:
        raise ValueError(
            "Submission node count "
            f"{submission.num_nodes} does not match true DAG size {true_dag.num_nodes}"
        )

    true_pairs = {
        canonical_undirected_edge(src, dst) for src, dst in true_dag.edges
    }
    submitted_pairs = submission.skeleton_edges
    all_pairs = true_pairs | submitted_pairs

    errors = 0
    for pair in all_pairs:
        true_edge = _true_directed_edge_for_pair(true_dag, pair)
        submitted_edge = _submitted_directed_edge_for_pair(submission, pair)
        submitted_undirected = pair in submission.undirected_edges

        if true_edge is None:
            errors += 1
        elif submitted_undirected:
            errors += 1
        elif submitted_edge is None:
            errors += 1
        elif submitted_edge != true_edge:
            errors += 1

    return errors


def _skeleton_prf(submission: GraphSubmission, cpdag: CPDAG) -> _PRF:
    truth = _cpdag_skeleton(cpdag)
    prediction = submission.skeleton_edges
    return _prf_sets(prediction, truth)


def _compelled_orientation_prf(
    submission: GraphSubmission, cpdag: CPDAG
) -> _PRF:
    tp = 0
    fp = 0
    fn = 0
    for src, dst in cpdag.directed_edges:
        if (src, dst) in submission.directed_edges:
            tp += 1
        elif (dst, src) in submission.directed_edges:
            fp += 1
            fn += 1
        else:
            fn += 1
    return _prf_counts(tp, fp, fn)


def _directed_prf(submission: GraphSubmission, true_dag: DAG) -> _PRF:
    return _prf_sets(submission.directed_edges, true_dag.edges)


def _prf_sets(prediction: set | frozenset, truth: set | frozenset) -> _PRF:
    tp = len(prediction & truth)
    fp = len(prediction - truth)
    fn = len(truth - prediction)
    return _prf_counts(tp, fp, fn)


def _prf_counts(tp: int, fp: int, fn: int) -> _PRF:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return _PRF(precision=precision, recall=recall, f1=f1)


def _cpdag_skeleton(cpdag: CPDAG) -> frozenset[UndirectedEdge]:
    return frozenset(
        {canonical_undirected_edge(src, dst) for src, dst in cpdag.directed_edges}
        | set(cpdag.undirected_edges)
    )


def _true_directed_edge_for_pair(
    true_dag: DAG, pair: UndirectedEdge
) -> tuple[int, int] | None:
    a, b = pair
    if (a, b) in true_dag.edges:
        return (a, b)
    if (b, a) in true_dag.edges:
        return (b, a)
    return None


def _submitted_directed_edge_for_pair(
    submission: GraphSubmission, pair: UndirectedEdge
) -> tuple[int, int] | None:
    a, b = pair
    if (a, b) in submission.directed_edges:
        return (a, b)
    if (b, a) in submission.directed_edges:
        return (b, a)
    return None
