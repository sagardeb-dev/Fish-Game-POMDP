import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from causal_discovery import (
    implied_covariance,
    is_near_singular,
    parameterize_linear_gaussian_scm,
    partial_correlation,
    relevant_structural_partial_correlations,
    sample_edge_weights,
    sample_noise_variances,
    sample_random_dag,
    sample_random_topological_order,
    topological_order_from_dag,
    valid_forward_edges,
)
from causal_discovery.core import DAG, LinearGaussianSCM


def test_sample_random_topological_order_is_a_permutation():
    order = sample_random_topological_order(5, np.random.default_rng(7))
    assert tuple(sorted(order)) == (0, 1, 2, 3, 4)


def test_valid_forward_edges_match_the_order():
    order = (2, 0, 3, 1)
    assert valid_forward_edges(order) == (
        (2, 0),
        (2, 3),
        (2, 1),
        (0, 3),
        (0, 1),
        (3, 1),
    )


def test_sample_random_dag_returns_reproducible_acyclic_graphs():
    dag1 = sample_random_dag(5, 6, np.random.default_rng(123))
    dag2 = sample_random_dag(5, 6, np.random.default_rng(123))
    assert dag1.edges == dag2.edges
    assert dag1.num_edges == 6
    order = topological_order_from_dag(dag1)
    positions = {node: idx for idx, node in enumerate(order)}
    assert all(positions[src] < positions[dst] for src, dst in dag1.edges)


def test_sample_edge_weights_populates_exactly_the_dag_edges():
    dag = DAG.from_edges(4, [(0, 1), (0, 2), (2, 3)])
    weights = sample_edge_weights(dag, ((-2.0, -0.5), (0.5, 2.0)), np.random.default_rng(9))
    assert weights.shape == (4, 4)
    assert weights[0, 1] != 0.0
    assert weights[0, 2] != 0.0
    assert weights[2, 3] != 0.0
    assert weights[1, 3] == 0.0
    assert np.all(np.diag(weights) == 0.0)


def test_sample_noise_variances_expands_scalar_and_accepts_vector():
    scalar = sample_noise_variances(3, 1.0)
    vector = sample_noise_variances(3, np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(scalar, np.array([1.0, 1.0, 1.0]))
    assert np.array_equal(vector, np.array([1.0, 2.0, 3.0]))


def test_parameterize_linear_gaussian_scm_is_reproducible_and_zero_intercept():
    dag = DAG.from_edges(4, [(0, 1), (0, 2), (2, 3)])
    scm1 = parameterize_linear_gaussian_scm(
        dag, ((-2.0, -0.5), (0.5, 2.0)), 1.0, np.random.default_rng(21)
    )
    scm2 = parameterize_linear_gaussian_scm(
        dag, ((-2.0, -0.5), (0.5, 2.0)), 1.0, np.random.default_rng(21)
    )
    assert np.array_equal(scm1.to_weight_matrix(), scm2.to_weight_matrix())
    assert np.array_equal(scm1.intercepts, np.zeros(4))


def test_implied_covariance_matches_known_two_node_sem():
    dag = DAG.from_edges(2, [(0, 1)])
    scm = LinearGaussianSCM.from_dag(
        dag,
        weights=np.array([[0.0, 0.5], [0.0, 0.0]]),
        noise_var=np.array([1.0, 1.0]),
    )
    cov = implied_covariance(scm)
    expected = np.array([[1.0, 0.5], [0.5, 1.25]])
    assert np.allclose(cov, expected)


def test_implied_covariance_is_symmetric_positive_definite_for_valid_scm():
    dag = DAG.from_edges(4, [(0, 1), (0, 2), (2, 3)])
    scm = parameterize_linear_gaussian_scm(
        dag, ((-2.0, -0.5), (0.5, 2.0)), 1.0, np.random.default_rng(5)
    )
    cov = implied_covariance(scm)
    assert np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvalsh(cov) > 0.0)


def test_is_near_singular_distinguishes_well_conditioned_and_degenerate_cases():
    assert is_near_singular(np.eye(3), tol=1e-8) is False
    assert is_near_singular(np.array([[1.0, 1.0], [1.0, 1.0]]), tol=1e-8) is True


def test_partial_correlation_for_chain_vanishes_when_conditioned_on_middle_node():
    dag = DAG.from_edges(3, [(0, 1), (1, 2)])
    scm = LinearGaussianSCM.from_dag(
        dag,
        weights=np.array(
            [
                [0.0, 0.8, 0.0],
                [0.0, 0.0, 0.6],
                [0.0, 0.0, 0.0],
            ]
        ),
        noise_var=1.0,
    )
    cov = implied_covariance(scm)
    rho = partial_correlation(cov, 0, 2, [1])
    assert -1.0 <= rho <= 1.0
    assert abs(rho) < 1e-10


def test_relevant_structural_partial_correlations_returns_one_value_per_edge():
    dag = DAG.from_edges(4, [(0, 1), (0, 2), (2, 3)])
    scm = LinearGaussianSCM.from_dag(
        dag,
        weights=np.array(
            [
                [0.0, 1.0, -1.5, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.75],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
        noise_var=1.0,
    )
    values = relevant_structural_partial_correlations(scm)
    assert len(values) == dag.num_edges
    assert all(-1.0 <= value <= 1.0 for value in values)
