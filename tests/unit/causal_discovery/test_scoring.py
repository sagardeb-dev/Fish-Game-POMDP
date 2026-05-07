import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from causal_discovery import (
    BenchmarkEnv,
    BenchmarkInstance,
    CPDAG,
    DAG,
    GraphSubmission,
    build_benchmark_instance,
    dag_shd,
    efficiency_score,
    make_v1_config,
    score_submission,
)


def _cpdag(directed, undirected, d):
    return CPDAG(
        num_nodes=d,
        directed_edges=frozenset(directed),
        undirected_edges=frozenset(undirected),
    )


def _instance(true_dag, cpdag, optimal=(0,)):
    cfg = make_v1_config(d=true_dag.num_nodes)
    return BenchmarkInstance(
        config=cfg,
        true_dag=true_dag,
        observational_ceiling=cpdag,
        scm=None,
        optimal_intervention_set=tuple(optimal),
        observational_data=np.zeros((cfg.n_obs, cfg.d)),
        intervention_budget=len(optimal),
        label_permutation=None,
    )


def test_graph_submission_from_matrix_parses_directed_and_undirected_edges():
    matrix = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
        ]
    )
    submission = GraphSubmission.from_adjacency_matrix(matrix, interventions_used=2)
    assert submission.directed_edges == frozenset({(0, 1)})
    assert submission.undirected_edges == frozenset({(1, 2)})
    assert submission.interventions_used == 2


def test_graph_submission_rejects_conflicting_directed_and_undirected_pair():
    with pytest.raises(ValueError, match="both directed and undirected"):
        GraphSubmission(
            num_nodes=3,
            directed_edges=frozenset({(0, 1)}),
            undirected_edges=frozenset({(0, 1)}),
        )


def test_graph_submission_rejects_directed_2_cycle():
    with pytest.raises(ValueError, match="Directed 2-cycle"):
        GraphSubmission(
            num_nodes=3,
            directed_edges=frozenset({(0, 1), (1, 0)}),
            undirected_edges=frozenset(),
        )


def test_graph_submission_rejects_cyclic_directed_part():
    with pytest.raises(ValueError, match="acyclic"):
        GraphSubmission(
            num_nodes=3,
            directed_edges=frozenset({(0, 1), (1, 2), (2, 0)}),
            undirected_edges=frozenset(),
        )


def test_skeleton_score_penalizes_false_positive_edges():
    true_dag = DAG.from_edges(4, [(0, 1), (1, 2)])
    cpdag = _cpdag(directed={(0, 1)}, undirected={(1, 2)}, d=4)
    submission = GraphSubmission(
        num_nodes=4,
        directed_edges=frozenset({(0, 1), (1, 2), (0, 2)}),
        undirected_edges=frozenset(),
    )

    scores = score_submission(_instance(true_dag, cpdag), submission)

    assert scores.skeleton_precision == pytest.approx(2 / 3)
    assert scores.skeleton_recall == 1.0
    assert scores.skeleton_f1 == pytest.approx(0.8)


def test_compelled_orientation_score_ignores_ambiguous_cpdag_edge_direction():
    true_dag = DAG.from_edges(4, [(0, 1), (1, 2)])
    cpdag = _cpdag(directed={(0, 1)}, undirected={(1, 2)}, d=4)
    submission = GraphSubmission(
        num_nodes=4,
        directed_edges=frozenset({(0, 1), (2, 1)}),
        undirected_edges=frozenset(),
    )

    scores = score_submission(_instance(true_dag, cpdag), submission)

    assert scores.compelled_precision == 1.0
    assert scores.compelled_recall == 1.0
    assert scores.compelled_f1 == 1.0


def test_compelled_orientation_score_penalizes_reversed_compelled_edge():
    true_dag = DAG.from_edges(4, [(0, 1), (1, 2)])
    cpdag = _cpdag(directed={(0, 1)}, undirected={(1, 2)}, d=4)
    submission = GraphSubmission(
        num_nodes=4,
        directed_edges=frozenset({(1, 0)}),
        undirected_edges=frozenset({(1, 2)}),
    )

    scores = score_submission(_instance(true_dag, cpdag), submission)

    assert scores.compelled_precision == 0.0
    assert scores.compelled_recall == 0.0
    assert scores.compelled_f1 == 0.0


def test_directed_score_treats_undirected_true_edge_as_unresolved():
    true_dag = DAG.from_edges(4, [(0, 1), (1, 2)])
    cpdag = _cpdag(directed={(0, 1)}, undirected={(1, 2)}, d=4)
    submission = GraphSubmission(
        num_nodes=4,
        directed_edges=frozenset({(0, 1)}),
        undirected_edges=frozenset({(1, 2)}),
    )

    scores = score_submission(_instance(true_dag, cpdag), submission)

    assert scores.directed_precision == 1.0
    assert scores.directed_recall == 0.5
    assert scores.directed_f1 == pytest.approx(2 / 3)
    assert scores.dag_shd == 1


def test_dag_shd_counts_reversed_edge_as_one_error_not_two():
    true_dag = DAG.from_edges(2, [(0, 1)])
    submission = GraphSubmission(
        num_nodes=2,
        directed_edges=frozenset({(1, 0)}),
        undirected_edges=frozenset(),
    )
    assert dag_shd(submission, true_dag) == 1


def test_dag_shd_counts_missing_extra_and_unresolved_edges():
    true_dag = DAG.from_edges(4, [(0, 1), (1, 2)])
    submission = GraphSubmission(
        num_nodes=4,
        directed_edges=frozenset({(2, 3)}),
        undirected_edges=frozenset({(1, 2)}),
    )
    assert dag_shd(submission, true_dag) == 3


def test_efficiency_score_zero_optimal_penalizes_unneeded_interventions():
    assert efficiency_score(interventions_used=0, optimal_interventions=0) == 1.0
    assert efficiency_score(interventions_used=2, optimal_interventions=0) == 0.0


def test_score_submission_with_oracle_submission():
    cfg = make_v1_config(d=5, n_obs=150, n_int=40, budget_slack=1)
    instance = build_benchmark_instance(cfg, np.random.default_rng(17))
    env = BenchmarkEnv(instance, np.random.default_rng(23))
    env.observe()
    output = env.submit_adjacency_matrix(instance.true_dag.adjacency_matrix())
    scores = score_submission(instance, output.submission)

    assert scores.skeleton_f1 == 1.0
    assert scores.compelled_f1 == 1.0
    assert scores.directed_f1 == 1.0
    assert scores.dag_shd == 0
    assert scores.efficiency == 1.0


def test_score_submission_fields_are_in_expected_ranges():
    cfg = make_v1_config(d=5, n_obs=120, n_int=40)
    instance = build_benchmark_instance(cfg, np.random.default_rng(31))
    submission = GraphSubmission(
        num_nodes=cfg.d,
        directed_edges=frozenset(),
        undirected_edges=frozenset(),
    )
    scores = score_submission(instance, submission)

    for value in (
        scores.skeleton_precision,
        scores.skeleton_recall,
        scores.skeleton_f1,
        scores.compelled_precision,
        scores.compelled_recall,
        scores.compelled_f1,
        scores.directed_precision,
        scores.directed_recall,
        scores.directed_f1,
        scores.efficiency,
    ):
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0
