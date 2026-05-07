import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from causal_discovery.agents.llm import LLMRawAgent, LLMStatsAgent
from causal_discovery.agents.prompts import (
    build_session_prompt,
    build_system_prompt_raw,
    build_system_prompt_stats,
)
from causal_discovery.agents.tool_schema import allowed_actions_for_method


class _DummyModel:
    def complete(self, **kwargs):
        _ = kwargs
        return '{"action":"submit_graph","directed_edges":[],"undirected_edges":[]}'


class _RecordingModel:
    def __init__(self):
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return '{"action":"submit_graph","directed_edges":[],"undirected_edges":[]}'


def test_build_system_prompt_raw_observational_mentions_no_interventions():
    prompt = build_system_prompt_raw(allowed_actions_for_method("llm_raw_obs"))
    assert "No experiments/interventions are available" in prompt
    assert "- intervene(var, value)" not in prompt
    assert "- submit_graph(directed_edges, undirected_edges, reasoning_summary)" in prompt
    assert "Do not add an edge merely because two variables are correlated." in prompt


def test_build_system_prompt_stats_observational_mentions_no_interventions():
    prompt = build_system_prompt_stats(allowed_actions_for_method("llm_stats_obs"))
    assert "No experiments/interventions are available" in prompt
    assert "- intervene(var, value)" not in prompt
    assert "- correlation(i, j)" in prompt
    assert "- partial_correlation(i, j, conditioning_on)" in prompt
    assert "- independence_test(i, j, conditioning_on, alpha)" in prompt
    assert "- submit_graph(directed_edges, undirected_edges, reasoning_summary)" in prompt
    assert "Tool purpose:" in prompt
    assert "Tool data source:" in prompt
    assert "Decision policy:" in prompt
    assert "Perform at least 3 statistical tool actions before your first submit_graph action." in prompt
    assert "Orient edges when evidence supports direction" in prompt
    assert "Do not add an edge merely because two variables are correlated." in prompt


def test_build_system_prompt_stats_active_includes_tool_semantics():
    prompt = build_system_prompt_stats(allowed_actions_for_method("llm_stats"))
    assert "correlation(i,j): coarse association check" in prompt
    assert "partial_correlation(i,j,S): test direct dependence" in prompt
    assert "independence_test(i,j,S,alpha): decide whether evidence supports conditional independence." in prompt
    assert "After an intervention, statistical tools use the latest intervention dataset." in prompt
    assert "- intervene(var, value)" in prompt
    assert "Use statistical tools when helpful, and use interventions when they can resolve causal direction." in prompt
    assert "Perform at least 3 statistical tool actions before your first submit_graph action." not in prompt
    assert "Choose intervention values far enough from the observational mean" in prompt


def test_build_system_prompt_raw_active_includes_intervention_value_guidance():
    prompt = build_system_prompt_raw(allowed_actions_for_method("llm_raw"))
    assert "Choose intervention values far enough from the observational mean" in prompt


def test_session_prompt_action_schema_uses_allowed_actions():
    data = np.zeros((2, 3))
    payload = json.loads(
        build_session_prompt(
            ("X0", "X1", "X2"),
            data,
            budget=0,
            allowed_actions=allowed_actions_for_method("llm_raw_obs"),
        )
    )
    assert payload["output_schema"]["action"] == "submit_graph"
    assert payload["json_contract"].startswith("Return all fields shown in output_schema")
    assert "Use null for unused scalar fields" in payload["json_contract"]
    assert set(payload["output_schema"]) == {
        "action",
        "directed_edges",
        "undirected_edges",
        "reasoning_summary",
    }


def test_session_prompt_output_schema_matches_stats_obs_policy():
    data = np.zeros((2, 3))
    payload = json.loads(
        build_session_prompt(
            ("X0", "X1", "X2"),
            data,
            budget=0,
            allowed_actions=allowed_actions_for_method("llm_stats_obs"),
        )
    )
    assert payload["output_schema"]["action"] == (
        "correlation|partial_correlation|independence_test|submit_graph"
    )
    assert "var" not in payload["output_schema"]
    assert "value" not in payload["output_schema"]
    assert {"i", "j", "conditioning_on", "alpha"}.issubset(
        set(payload["output_schema"])
    )
    assert "reasoning_summary" in payload["output_schema"]


def test_session_prompt_output_schema_matches_raw_active_policy():
    data = np.zeros((2, 3))
    payload = json.loads(
        build_session_prompt(
            ("X0", "X1", "X2"),
            data,
            budget=2,
            allowed_actions=allowed_actions_for_method("llm_raw"),
        )
    )
    assert payload["output_schema"]["action"] == "intervene|submit_graph"
    assert {"var", "value"}.issubset(set(payload["output_schema"]))
    assert "i" not in payload["output_schema"]
    assert "j" not in payload["output_schema"]
    assert "conditioning_on" not in payload["output_schema"]
    assert "alpha" not in payload["output_schema"]


def test_llm_raw_agent_observational_mode_disallows_intervene_action():
    agent = LLMRawAgent(
        model=_DummyModel(),
        variable_names=("X0", "X1", "X2", "X3"),
        allowed_actions=allowed_actions_for_method("llm_raw_obs"),
    )
    assert agent.allowed_actions == frozenset({"submit_graph"})


def test_llm_stats_agent_observational_mode_disallows_intervene_action():
    agent = LLMStatsAgent(
        model=_DummyModel(),
        variable_names=("X0", "X1", "X2", "X3"),
        allowed_actions=allowed_actions_for_method("llm_stats_obs"),
    )
    assert "intervene" not in agent.allowed_actions
    assert "submit_graph" in agent.allowed_actions


def test_llm_agent_final_step_session_prompt_is_submit_only():
    model = _RecordingModel()
    agent = LLMStatsAgent(
        model=model,
        variable_names=("X0", "X1", "X2"),
        allowed_actions=allowed_actions_for_method("llm_stats"),
    )
    agent.on_observation(np.zeros((2, 3)), budget=2)

    action = agent.next_action(
        remaining_budget=2,
        allowed_actions=frozenset({"submit_graph"}),
        current_step=40,
        max_steps=40,
        final_step=True,
    )

    assert action.__class__.__name__ == "SubmitGraphAction"
    assert model.calls[0]["allowed_actions"] == frozenset({"submit_graph"})
    assert model.calls[0]["final_step"] is True
    payload = json.loads(model.calls[0]["session_prompt"])
    assert payload["output_schema"]["action"] == "submit_graph"
    assert set(payload["output_schema"]) == {
        "action",
        "directed_edges",
        "undirected_edges",
        "reasoning_summary",
    }


def test_llm_agent_feeds_reasoning_summary_back_as_working_memory():
    class MemoryModel:
        def __init__(self):
            self.calls = []

        def complete(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return (
                    '{"action":"correlation","i":0,"j":1,'
                    '"reasoning_summary":"checking X0-X1 before deciding"}'
                )
            return '{"action":"submit_graph","directed_edges":[],"undirected_edges":[],"reasoning_summary":"done"}'

    model = MemoryModel()
    agent = LLMStatsAgent(
        model=model,
        variable_names=("X0", "X1"),
        allowed_actions=allowed_actions_for_method("llm_stats_obs"),
    )
    agent.on_observation(np.zeros((2, 2)), budget=0)

    first = agent.next_action(remaining_budget=0, current_step=1, max_steps=3)
    assert first.__class__.__name__ == "CorrelationAction"
    second = agent.next_action(
        remaining_budget=0,
        allowed_actions=frozenset({"submit_graph"}),
        current_step=2,
        max_steps=3,
    )

    assert second.__class__.__name__ == "SubmitGraphAction"
    assert model.calls[1]["working_memory"] == (
        {
            "step": 1,
            "action": "correlation",
            "reasoning_summary": "checking X0-X1 before deciding",
        },
    )


def test_llm_agent_bounds_working_memory_entries_and_summary_length():
    class LongMemoryModel:
        def __init__(self):
            self.calls = []

        def complete(self, **kwargs):
            self.calls.append(kwargs)
            return (
                '{"action":"correlation","i":0,"j":1,"reasoning_summary":"'
                + ("x" * 500)
                + '"}'
            )

    model = LongMemoryModel()
    agent = LLMStatsAgent(
        model=model,
        variable_names=("X0", "X1"),
        allowed_actions=allowed_actions_for_method("llm_stats_obs"),
    )
    agent.on_observation(np.zeros((2, 2)), budget=0)

    for step in range(1, 15):
        agent.next_action(remaining_budget=0, current_step=step, max_steps=20)

    memory = model.calls[-1]["working_memory"]
    assert len(memory) == 12
    assert all(len(item["reasoning_summary"]) == 300 for item in memory)
