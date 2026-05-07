import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from causal_discovery import build_benchmark_instance, make_v1_config


def _skeleton_from_directed_edges(edges: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> set[tuple[int, int]]:
    return {tuple(sorted((src, dst))) for src, dst in edges}


def test_builder_returns_complete_instance_with_consistent_shapes():
    cfg = make_v1_config(d=5, n_obs=120, budget_slack=2)
    instance = build_benchmark_instance(cfg, np.random.default_rng(21))
    assert instance.true_dag.num_nodes == cfg.d
    assert instance.observational_ceiling.num_nodes == cfg.d
    assert instance.scm.dag.num_nodes == cfg.d
    assert instance.observational_data.shape == (cfg.n_obs, cfg.d)
    assert instance.intervention_budget == len(instance.optimal_intervention_set) + cfg.budget_slack
    assert all(0 <= node < cfg.d for node in instance.optimal_intervention_set)


def test_builder_relabels_graph_objects_consistently():
    cfg = make_v1_config(d=5)
    instance = build_benchmark_instance(cfg, np.random.default_rng(31))
    dag_skeleton = _skeleton_from_directed_edges(instance.true_dag.edges)
    cpdag_skeleton = set(instance.observational_ceiling.undirected_edges) | _skeleton_from_directed_edges(
        instance.observational_ceiling.directed_edges
    )
    assert dag_skeleton == cpdag_skeleton


def test_builder_is_reproducible_with_fixed_seed():
    cfg = make_v1_config(d=5, n_obs=60)
    inst1 = build_benchmark_instance(cfg, np.random.default_rng(77))
    inst2 = build_benchmark_instance(cfg, np.random.default_rng(77))
    assert inst1.true_dag.edges == inst2.true_dag.edges
    assert inst1.observational_ceiling.directed_edges == inst2.observational_ceiling.directed_edges
    assert inst1.observational_ceiling.undirected_edges == inst2.observational_ceiling.undirected_edges
    assert inst1.optimal_intervention_set == inst2.optimal_intervention_set
    assert np.array_equal(inst1.scm.weights, inst2.scm.weights)
    assert np.array_equal(inst1.observational_data, inst2.observational_data)


def test_builder_raises_when_max_attempts_exhausted():
    cfg = make_v1_config(d=5, faithfulness_eps=1.1)
    with pytest.raises(RuntimeError, match="Unable to build"):
        build_benchmark_instance(cfg, np.random.default_rng(9), max_attempts=3)

