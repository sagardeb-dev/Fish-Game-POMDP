import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from causal_discovery import DAG, LinearGaussianSCM, Permutation, make_v1_config


def test_config_applies_v1_defaults():
    cfg = make_v1_config(d=4)
    assert cfg.k == 5
    assert cfg.n_obs == 1000
    assert cfg.n_int == 1000
    assert cfg.noise_var == 1.0
    assert cfg.faithfulness_eps == 0.1
    assert cfg.budget_slack == 0


def test_config_requires_minimum_v1_dimension():
    with pytest.raises(ValueError, match="d must be >= 4"):
        make_v1_config(d=3)


def test_config_rejects_impossible_edge_count():
    with pytest.raises(ValueError, match="k must be <="):
        make_v1_config(d=4, k=7)


def test_dag_exposes_basic_structure_queries():
    dag = DAG.from_edges(4, [(0, 1), (0, 2), (2, 3)])
    assert dag.num_nodes == 4
    assert dag.num_edges == 3
    assert dag.parents(3) == (2,)
    assert dag.children(0) == (1, 2)
    assert dag.has_edge(0, 2) is True
    assert dag.has_edge(1, 2) is False
    assert np.array_equal(
        dag.adjacency_matrix(),
        np.array(
            [
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ],
            dtype=np.int8,
        ),
    )


def test_dag_rejects_cycles_and_self_loops():
    with pytest.raises(ValueError, match="acyclic"):
        DAG.from_edges(3, [(0, 1), (1, 2), (2, 0)])

    with pytest.raises(ValueError, match="Self-loop"):
        DAG.from_edges(2, [(0, 0)])


def test_permutation_relabels_nodes_edges_matrices_and_vectors():
    perm = Permutation.from_mapping([2, 0, 3, 1])
    assert perm.apply_node(1) == 0
    assert perm.apply_edges([(0, 1), (2, 3)]) == frozenset({(2, 0), (3, 1)})

    matrix = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
            [12.0, 13.0, 14.0, 15.0],
        ]
    )
    vector = np.array([10.0, 20.0, 30.0, 40.0])

    relabeled_matrix = perm.apply_square_matrix(matrix)
    relabeled_vector = perm.apply_vector(vector)

    assert relabeled_matrix[2, 0] == matrix[0, 1]
    assert np.array_equal(relabeled_vector, np.array([20.0, 40.0, 10.0, 30.0]))


def test_scm_validates_against_the_dag():
    dag = DAG.from_edges(4, [(0, 1), (0, 2), (2, 3)])
    scm = LinearGaussianSCM.from_dag(
        dag,
        weights=[
            [0.0, 1.0, -1.5, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.75],
            [0.0, 0.0, 0.0, 0.0],
        ],
        noise_var=1.0,
    )

    assert scm.parents(3) == (2,)
    assert scm.weight(0, 2) == -1.5
    assert np.array_equal(
        scm.implied_adjacency(),
        np.array(
            [
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ],
            dtype=np.int8,
        ),
    )
    assert scm.to_weight_matrix().shape == (4, 4)


def test_scm_rejects_non_edge_weights_missing_edge_weights_and_bad_noise():
    dag = DAG.from_edges(3, [(0, 1)])

    with pytest.raises(ValueError, match="nonzero entries only on DAG edges"):
        LinearGaussianSCM.from_dag(
            dag,
            weights=[
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
            ],
            noise_var=1.0,
        )

    with pytest.raises(ValueError, match="corresponding nonzero weight"):
        LinearGaussianSCM.from_dag(
            dag,
            weights=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            noise_var=1.0,
        )

    with pytest.raises(ValueError, match="positive"):
        LinearGaussianSCM.from_dag(
            dag,
            weights=[
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            noise_var=[1.0, 0.0, 1.0],
        )


def test_dag_and_scm_relabel_preserve_structure():
    dag = DAG.from_edges(4, [(0, 1), (0, 2), (2, 3)])
    scm = LinearGaussianSCM.from_dag(
        dag,
        weights=[
            [0.0, 1.0, -1.5, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.75],
            [0.0, 0.0, 0.0, 0.0],
        ],
        noise_var=1.0,
        intercepts=[0.0, 1.0, 2.0, 3.0],
    )
    perm = Permutation.from_mapping([2, 0, 3, 1])

    dag_public = dag.relabel(perm)
    scm_public = scm.relabel(perm)

    assert dag_public.num_edges == dag.num_edges
    assert dag_public.edges == frozenset({(2, 0), (2, 3), (3, 1)})
    assert scm_public.to_weight_matrix()[2, 0] == 1.0
    assert scm_public.to_weight_matrix()[2, 3] == -1.5
    assert np.array_equal(scm_public.intercepts, np.array([1.0, 3.0, 0.0, 2.0]))
