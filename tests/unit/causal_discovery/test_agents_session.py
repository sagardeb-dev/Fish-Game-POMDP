import json
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
    InterveneAction,
    LLMRawAgent,
    MockAgent,
    SubmitGraphAction,
    build_benchmark_instance,
    make_v1_config,
    run_agent_session,
)
from causal_discovery.agents.actions import CorrelationAction
from causal_discovery.agents.tool_schema import allowed_actions_for_method


def _build_env(seed: int = 0, d: int = 5, n_obs: int = 25, n_int: int = 15):
    cfg = make_v1_config(d=d, n_obs=n_obs, n_int=n_int, budget_slack=1)
    instance = build_benchmark_instance(cfg, np.random.default_rng(seed))
    env = BenchmarkEnv(instance, np.random.default_rng(seed + 1000))
    return instance, env


def test_run_agent_session_with_mock_agent_and_intervention():
    instance, env = _build_env()
    actions = (
        InterveneAction(var=0, value=0.0),
        SubmitGraphAction(
            directed_edges=tuple(sorted(instance.true_dag.edges)),
            undirected_edges=tuple(),
            reasoning_summary="mock",
        ),
    )
    agent = MockAgent(planned_actions=actions)
    output = run_agent_session(env, agent)
    assert output.submission.directed_edges == instance.true_dag.edges
    assert output.submission.interventions_used == 1
    assert len(agent.intervention_results) == 1


def test_run_agent_session_supports_stats_tool_actions():
    instance, env = _build_env()
    actions = (
        CorrelationAction(i=0, j=1),
        SubmitGraphAction(
            directed_edges=tuple(sorted(instance.true_dag.edges)),
            undirected_edges=tuple(),
            reasoning_summary="stats",
        ),
    )
    agent = MockAgent(planned_actions=actions)
    output = run_agent_session(env, agent)
    assert output.submission.interventions_used == 0
    assert len(agent.tool_results) == 1
    assert agent.tool_results[0].tool == "correlation"
    assert agent.tool_results[0].payload["data_source"] == "observational"


def test_stats_tool_after_intervention_uses_latest_intervention_data():
    _, env = _build_env(seed=7)
    actions = (
        InterveneAction(var=0, value=3.0),
        CorrelationAction(i=1, j=2),
        SubmitGraphAction(
            directed_edges=tuple(),
            undirected_edges=tuple(),
            reasoning_summary="stats",
        ),
    )
    agent = MockAgent(planned_actions=actions)
    _ = run_agent_session(env, agent)
    assert len(agent.tool_results) == 1
    payload = agent.tool_results[0].payload
    assert payload["data_source"] == "latest_intervention"
    assert payload["rows"] == 15


def test_stats_tool_reports_undefined_correlation_without_crashing():
    _, env = _build_env(seed=7)
    actions = (
        InterveneAction(var=0, value=3.0),
        CorrelationAction(i=0, j=1),
        SubmitGraphAction(
            directed_edges=tuple(),
            undirected_edges=tuple(),
            reasoning_summary="stats",
        ),
    )
    agent = MockAgent(planned_actions=actions)
    _ = run_agent_session(env, agent)
    payload = agent.tool_results[0].payload
    assert payload["data_source"] == "latest_intervention"
    assert payload["value"] is None
    assert "constant" in payload["error"]


def test_run_agent_session_raises_on_over_budget_intervention_request():
    instance, env = _build_env(seed=5)
    actions = tuple(
        InterveneAction(var=0, value=0.0) for _ in range(instance.intervention_budget + 1)
    )
    agent = MockAgent(planned_actions=actions)
    with pytest.raises(RuntimeError, match="budget is exhausted"):
        run_agent_session(env, agent)


class _QueueModel:
    def __init__(self, outputs: list[dict]):
        self._outputs = [json.dumps(x) for x in outputs]

    def complete(self, **kwargs) -> str:
        _ = kwargs
        if not self._outputs:
            raise RuntimeError("No queued model outputs")
        return self._outputs.pop(0)


def test_llm_raw_agent_parses_json_actions_end_to_end():
    instance, env = _build_env(seed=11)
    model = _QueueModel(
        [
            {"action": "intervene", "var": 0, "value": 0.0},
            {
                "action": "submit_graph",
                "directed_edges": [list(edge) for edge in sorted(instance.true_dag.edges)],
                "undirected_edges": [],
                "reasoning_summary": "done",
            },
        ]
    )
    names = tuple(f"X{i}" for i in range(instance.config.d))
    agent = LLMRawAgent(
        model=model,
        variable_names=names,
        allowed_actions=allowed_actions_for_method("llm_raw"),
    )
    output = run_agent_session(env, agent)
    assert output.submission.directed_edges == instance.true_dag.edges
    assert output.submission.interventions_used == 1


def test_llm_raw_agent_accepts_x_prefixed_variable_indices():
    instance, env = _build_env(seed=13)
    model = _QueueModel(
        [
            {"action": "intervene", "var": "X0", "value": 0.0},
            {
                "action": "submit_graph",
                "directed_edges": [list(edge) for edge in sorted(instance.true_dag.edges)],
                "undirected_edges": [],
                "reasoning_summary": "done",
            },
        ]
    )
    names = tuple(f"X{i}" for i in range(instance.config.d))
    agent = LLMRawAgent(
        model=model,
        variable_names=names,
        allowed_actions=allowed_actions_for_method("llm_raw"),
    )
    output = run_agent_session(env, agent)
    assert output.submission.interventions_used == 1
