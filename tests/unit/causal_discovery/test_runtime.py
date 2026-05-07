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
    GraphSubmission,
    SessionOutput,
    build_benchmark_instance,
    make_v1_config,
)


def _build_instance(seed: int = 0, d: int = 5, n_obs: int = 80, budget_slack: int = 2):
    cfg = make_v1_config(d=d, n_obs=n_obs, n_int=40, budget_slack=budget_slack)
    instance = build_benchmark_instance(cfg, np.random.default_rng(seed))
    return cfg, instance


def test_observe_returns_observational_snapshot_copy():
    cfg, instance = _build_instance()
    env = BenchmarkEnv(instance, np.random.default_rng(1))
    data = env.observe()
    data[0, 0] = 999.0
    assert data.shape == (cfg.n_obs, cfg.d)
    assert not np.array_equal(data, instance.observational_data)


def test_observe_rejects_second_call():
    _, instance = _build_instance()
    env = BenchmarkEnv(instance, np.random.default_rng(1))
    env.observe()
    with pytest.raises(RuntimeError, match="already delivered"):
        env.observe()


def test_intervene_requires_observe_first():
    _, instance = _build_instance()
    env = BenchmarkEnv(instance, np.random.default_rng(2))
    with pytest.raises(RuntimeError, match="observe"):
        env.intervene(var=0, value=0.0)


def test_intervene_returns_samples_and_decrements_budget():
    cfg, instance = _build_instance()
    env = BenchmarkEnv(instance, np.random.default_rng(2))
    env.observe()
    initial_budget = env.remaining_budget
    samples = env.intervene(var=0, value=0.0)
    assert samples.shape == (cfg.n_int, cfg.d)
    assert env.remaining_budget == initial_budget - 1


def test_intervene_rejects_out_of_range_var():
    _, instance = _build_instance()
    env = BenchmarkEnv(instance, np.random.default_rng(3))
    env.observe()
    with pytest.raises(ValueError, match="out of range"):
        env.intervene(var=999, value=0.0)


def test_intervene_raises_when_budget_exhausted():
    _, instance = _build_instance(budget_slack=0)
    env = BenchmarkEnv(instance, np.random.default_rng(4))
    env.observe()
    while env.remaining_budget > 0:
        env.intervene(var=0, value=0.0)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        env.intervene(var=0, value=0.0)


def test_submit_graph_happy_path_returns_output_with_interventions_used():
    _, instance = _build_instance()
    env = BenchmarkEnv(instance, np.random.default_rng(5))
    env.observe()
    env.intervene(var=0, value=0.0)
    env.intervene(var=1, value=0.0)
    submission = GraphSubmission.from_dag(instance.true_dag)
    output = env.submit_graph(submission)
    assert isinstance(output, SessionOutput)
    assert output.submission.interventions_used == 2
    assert output.submission.directed_edges == instance.true_dag.edges


def test_submit_graph_requires_observe_first():
    _, instance = _build_instance()
    env = BenchmarkEnv(instance, np.random.default_rng(5))
    submission = GraphSubmission.from_dag(instance.true_dag)
    with pytest.raises(RuntimeError, match="observe"):
        env.submit_graph(submission)


def test_submit_adjacency_matrix_requires_observe_first():
    _, instance = _build_instance()
    env = BenchmarkEnv(instance, np.random.default_rng(5))
    with pytest.raises(RuntimeError, match="observe"):
        env.submit_adjacency_matrix(instance.true_dag.adjacency_matrix())


def test_submit_adjacency_matrix_rejects_cyclic_directed_part():
    _, instance = _build_instance()
    env = BenchmarkEnv(instance, np.random.default_rng(6))
    env.observe()
    d = instance.config.d
    cyclic = np.zeros((d, d), dtype=int)
    cyclic[0, 1] = 1
    cyclic[1, 2] = 1
    cyclic[2, 0] = 1
    with pytest.raises(ValueError, match="acyclic"):
        env.submit_adjacency_matrix(cyclic)


def test_submit_adjacency_matrix_rejects_non_binary_matrix():
    _, instance = _build_instance()
    env = BenchmarkEnv(instance, np.random.default_rng(7))
    env.observe()
    d = instance.config.d
    bad = np.zeros((d, d), dtype=int)
    bad[0, 1] = 2
    with pytest.raises(ValueError, match="binary"):
        env.submit_adjacency_matrix(bad)


def test_submit_adjacency_matrix_rejects_wrong_shape():
    _, instance = _build_instance()
    env = BenchmarkEnv(instance, np.random.default_rng(8))
    env.observe()
    d = instance.config.d
    with pytest.raises(ValueError, match="shape"):
        env.submit_adjacency_matrix(np.zeros((d, d + 1), dtype=int))


def test_submit_adjacency_matrix_rejects_self_loops():
    _, instance = _build_instance()
    env = BenchmarkEnv(instance, np.random.default_rng(9))
    env.observe()
    d = instance.config.d
    matrix = np.zeros((d, d), dtype=int)
    matrix[0, 0] = 1
    with pytest.raises(ValueError, match="zero diagonal"):
        env.submit_adjacency_matrix(matrix)


def test_submit_cpdag_preserves_unresolved_edges():
    _, instance = _build_instance()
    env = BenchmarkEnv(instance, np.random.default_rng(10))
    env.observe()
    output = env.submit_cpdag(directed_edges={(0, 1)}, undirected_edges={(1, 2)})
    assert output.submission.directed_edges == frozenset({(0, 1)})
    assert output.submission.undirected_edges == frozenset({(1, 2)})


def test_session_is_sealed_after_submission():
    _, instance = _build_instance()
    env = BenchmarkEnv(instance, np.random.default_rng(11))
    env.observe()
    env.submit_adjacency_matrix(instance.true_dag.adjacency_matrix())
    with pytest.raises(RuntimeError, match="sealed"):
        env.observe()
    with pytest.raises(RuntimeError, match="sealed"):
        env.intervene(var=0, value=0.0)
    with pytest.raises(RuntimeError, match="sealed"):
        env.submit_adjacency_matrix(instance.true_dag.adjacency_matrix())


def test_remaining_budget_tracks_intervention_calls():
    _, instance = _build_instance(budget_slack=3)
    env = BenchmarkEnv(instance, np.random.default_rng(12))
    env.observe()
    start = env.remaining_budget
    env.intervene(var=0, value=0.0)
    env.intervene(var=1, value=0.5)
    assert env.remaining_budget == start - 2


def test_intervention_sampling_is_reproducible_for_fixed_seed():
    _, instance = _build_instance()
    env1 = BenchmarkEnv(instance, np.random.default_rng(42))
    env2 = BenchmarkEnv(instance, np.random.default_rng(42))
    env1.observe()
    env2.observe()
    samples1 = env1.intervene(var=0, value=0.0)
    samples2 = env2.intervene(var=0, value=0.0)
    assert np.array_equal(samples1, samples2)


def test_num_variables_property_matches_config():
    cfg, instance = _build_instance()
    env = BenchmarkEnv(instance, np.random.default_rng(13))
    assert env.num_variables == cfg.d
