import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from causal_discovery import (
    CPDAG,
    DAG,
    LinearGaussianSCM,
    compute_minimum_intervention_set,
    dag_to_cpdag,
    reject_graph,
    reject_intervention_profile,
    reject_scm,
)
from causal_discovery.core import Permutation
from causal_discovery.equivalence.theory import (
    _MutablePDAG,
    _apply_meek_closure,
    _orient_with_intervention_targets,
)


def test_dag_to_cpdag_for_chain_keeps_edges_undirected():
    dag = DAG.from_edges(3, [(0, 1), (1, 2)])
    cpdag = dag_to_cpdag(dag)
    assert cpdag.directed_edges == frozenset()
    assert cpdag.undirected_edges == frozenset({(0, 1), (1, 2)})


def test_dag_to_cpdag_for_unshielded_collider_orients_v_structure():
    dag = DAG.from_edges(3, [(0, 2), (1, 2)])
    cpdag = dag_to_cpdag(dag)
    assert cpdag.directed_edges == frozenset({(0, 2), (1, 2)})
    assert cpdag.undirected_edges == frozenset()


def test_dag_to_cpdag_does_not_orient_shielded_collider():
    dag = DAG.from_edges(3, [(0, 1), (0, 2), (1, 2)])
    cpdag = dag_to_cpdag(dag)
    assert cpdag.directed_edges == frozenset()
    assert cpdag.undirected_edges == frozenset({(0, 1), (0, 2), (1, 2)})


def test_meek_rule1_orients_unshielded_triple_completion():
    pdag = _MutablePDAG(
        num_nodes=3,
        directed_edges={(2, 0)},
        undirected_edges={(0, 1)},
    )
    _apply_meek_closure(pdag, include_rule4=False)
    assert pdag.directed_edges == {(2, 0), (0, 1)}
    assert pdag.undirected_edges == set()


def test_meek_rule2_orients_along_directed_chain():
    pdag = _MutablePDAG(
        num_nodes=3,
        directed_edges={(0, 2), (2, 1)},
        undirected_edges={(0, 1)},
    )
    _apply_meek_closure(pdag, include_rule4=False)
    assert pdag.directed_edges == {(0, 2), (2, 1), (0, 1)}
    assert pdag.undirected_edges == set()


def test_meek_rule3_orients_from_two_nonadjacent_parents():
    pdag = _MutablePDAG(
        num_nodes=4,
        directed_edges={(2, 1), (3, 1)},
        undirected_edges={(0, 1), (0, 2), (0, 3)},
    )
    _apply_meek_closure(pdag, include_rule4=False)
    assert (0, 1) in pdag.directed_edges
    assert (0, 1) not in pdag.undirected_edges


def test_meek_rule4_orients_with_background_knowledge_pattern():
    pdag = _MutablePDAG(
        num_nodes=4,
        directed_edges={(3, 2), (2, 1)},
        undirected_edges={(0, 1), (0, 2), (0, 3)},
    )
    _apply_meek_closure(pdag, include_rule4=True)
    assert (0, 1) in pdag.directed_edges
    assert (0, 1) not in pdag.undirected_edges


def test_cpdag_relabel_preserves_directed_and_undirected_semantics():
    cpdag = CPDAG(
        num_nodes=4,
        directed_edges=frozenset({(0, 1)}),
        undirected_edges=frozenset({(1, 2), (2, 3)}),
    )
    relabeled = cpdag.relabel(Permutation.from_mapping((2, 0, 3, 1)))
    assert relabeled.directed_edges == frozenset({(2, 0)})
    assert relabeled.undirected_edges == frozenset({(0, 3), (1, 3)})


def test_reject_graph_rejects_fully_oriented_cpdag():
    dag = DAG.from_edges(3, [(0, 2), (1, 2)])
    cpdag = dag_to_cpdag(dag)
    assert reject_graph(dag, cpdag) is True


def test_reject_graph_rejects_when_too_few_undirected_edges():
    dag = DAG.from_edges(3, [(0, 1), (1, 2)])
    cpdag = CPDAG(
        num_nodes=3,
        directed_edges=frozenset({(0, 1)}),
        undirected_edges=frozenset({(1, 2)}),
    )
    assert reject_graph(dag, cpdag) is True


def test_reject_graph_rejects_when_all_ambiguity_shares_one_node():
    dag = DAG.from_edges(3, [(0, 1), (0, 2)])
    cpdag = CPDAG(
        num_nodes=3,
        directed_edges=frozenset(),
        undirected_edges=frozenset({(0, 1), (0, 2)}),
    )
    assert reject_graph(dag, cpdag) is True


def test_reject_graph_rejects_disconnected_skeleton():
    dag = DAG.from_edges(4, [(0, 1), (2, 3)])
    cpdag = CPDAG(
        num_nodes=4,
        directed_edges=frozenset(),
        undirected_edges=frozenset({(0, 1), (2, 3)}),
    )
    assert reject_graph(dag, cpdag) is True


def test_reject_graph_accepts_nontrivial_connected_ambiguous_graph():
    dag = DAG.from_edges(3, [(0, 1), (0, 2), (1, 2)])
    cpdag = dag_to_cpdag(dag)
    assert reject_graph(dag, cpdag) is False


def test_reject_scm_accepts_healthy_instance():
    dag = DAG.from_edges(3, [(0, 1), (1, 2)])
    scm = LinearGaussianSCM.from_dag(
        dag,
        weights=np.array(
            [
                [0.0, 0.8, 0.0],
                [0.0, 0.0, 0.7],
                [0.0, 0.0, 0.0],
            ]
        ),
        noise_var=1.0,
    )
    assert reject_scm(scm, faithfulness_eps=0.1) is False


def test_reject_scm_rejects_nearly_unfaithful_edge():
    dag = DAG.from_edges(2, [(0, 1)])
    scm = LinearGaussianSCM.from_dag(
        dag,
        weights=np.array([[0.0, 0.01], [0.0, 0.0]]),
        noise_var=1.0,
    )
    assert reject_scm(scm, faithfulness_eps=0.1) is True


def test_reject_scm_rejects_near_singular_covariance():
    dag = DAG.from_edges(2, [(0, 1)])
    scm = LinearGaussianSCM.from_dag(
        dag,
        weights=np.array([[0.0, 0.5], [0.0, 0.0]]),
        noise_var=np.array([1e-12, 1e-12]),
    )
    assert reject_scm(scm, faithfulness_eps=0.0) is True


def test_compute_minimum_intervention_set_returns_lexicographically_first_minimum():
    dag = DAG.from_edges(3, [(0, 1), (1, 2)])
    cpdag = dag_to_cpdag(dag)
    assert compute_minimum_intervention_set(dag, cpdag) == (0,)


def test_reject_intervention_profile_only_rejects_empty_set():
    cpdag = CPDAG(
        num_nodes=3,
        directed_edges=frozenset(),
        undirected_edges=frozenset({(0, 1), (1, 2)}),
    )
    assert reject_intervention_profile((), cpdag) is True
    assert reject_intervention_profile((1,), cpdag) is False


def test_cpdag_plus_true_orientations_recovers_true_dag():
    dag = DAG.from_edges(3, [(0, 1), (0, 2), (1, 2)])
    cpdag = dag_to_cpdag(dag)
    completed = _orient_with_intervention_targets(cpdag, dag, (0, 1, 2))
    assert completed.directed_edges == dag.edges
    assert completed.undirected_edges == frozenset()
