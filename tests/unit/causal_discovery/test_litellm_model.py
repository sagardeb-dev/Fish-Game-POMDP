import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from causal_discovery.agents.litellm_model import (  # noqa: E402
    LiteLLMJSONPolicyModel,
    _is_anthropic_model,
    normalize_litellm_model,
    provider_for_model,
)
from causal_discovery.agents.tool_schema import allowed_actions_for_method  # noqa: E402


class _Function:
    def __init__(self, name="causal_discovery_action", arguments=None):
        self.name = name
        self.arguments = arguments or (
            '{"action":"submit_graph","var":null,"value":null,"i":null,"j":null,'
            '"conditioning_on":[],"alpha":null,"directed_edges":[],"undirected_edges":[],'
            '"reasoning_summary":"done"}'
        )


class _ToolCall:
    def __init__(self, name="causal_discovery_action", arguments=None):
        self.function = _Function(name=name, arguments=arguments)


class _Response:
    def __init__(self, tool_calls, cost=None):
        self.choices = [SimpleNamespace(message=SimpleNamespace(tool_calls=tool_calls, content=""))]
        self.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=3, total_tokens=13)
        if cost is not None:
            self.usage.cost = cost

    def model_dump(self):
        return {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}}


def test_is_anthropic_model_detects_openrouter_anthropic():
    assert _is_anthropic_model("openrouter/anthropic/claude-haiku-4-5-20251001") is True
    assert _is_anthropic_model("anthropic/claude-sonnet-4-6") is True
    assert _is_anthropic_model("claude-haiku-4-5-20251001") is True
    assert _is_anthropic_model("openai/gpt-5.4") is False
    assert _is_anthropic_model("openrouter/deepseek/deepseek-v4-pro") is False


def test_anthropic_model_request_includes_cache_control(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _Response([_ToolCall()])

    monkeypatch.setattr("causal_discovery.agents.litellm_model._litellm_completion", fake_completion)
    monkeypatch.setattr("causal_discovery.agents.litellm_model._litellm_completion_cost", lambda r: 0.0)

    model = LiteLLMJSONPolicyModel(
        model="openrouter/anthropic/claude-haiku-4-5-20251001",
        allowed_actions=frozenset({"submit_graph"}),
    )
    model.complete(system_prompt="sys", session_prompt="{}", tool_history=tuple(), remaining_budget=0)

    sys_msg = captured["messages"][0]
    assert isinstance(sys_msg["content"], list)
    assert sys_msg["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert sys_msg["content"][0]["text"] == "sys"


def test_non_anthropic_model_request_has_no_cache_control(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _Response([_ToolCall()])

    monkeypatch.setattr("causal_discovery.agents.litellm_model._litellm_completion", fake_completion)
    monkeypatch.setattr("causal_discovery.agents.litellm_model._litellm_completion_cost", lambda r: 0.0)

    model = LiteLLMJSONPolicyModel(
        model="openai/gpt-5.4",
        allowed_actions=frozenset({"submit_graph"}),
    )
    model.complete(system_prompt="sys", session_prompt="{}", tool_history=tuple(), remaining_budget=0)

    sys_msg = captured["messages"][0]
    assert isinstance(sys_msg["content"], str)


def test_normalize_litellm_model_keeps_explicit_provider_prefix():
    assert normalize_litellm_model("openrouter/openai/gpt-5.4") == "openrouter/openai/gpt-5.4"
    assert provider_for_model("openrouter/openai/gpt-5.4") == "openrouter"


def test_normalize_litellm_model_accepts_legacy_names():
    assert normalize_litellm_model("gpt-5.4") == "openai/gpt-5.4"
    assert normalize_litellm_model("claude-sonnet-4-6") == "anthropic/claude-sonnet-4-6"


def test_litellm_model_parses_one_tool_call(monkeypatch):
    def fake_completion(**kwargs):
        assert kwargs["model"] == "openai/gpt-5.4"
        assert kwargs["tool_choice"]["function"]["name"] == "causal_discovery_action"
        return _Response([_ToolCall()])

    monkeypatch.setattr(
        "causal_discovery.agents.litellm_model._litellm_completion",
        fake_completion,
    )
    monkeypatch.setattr(
        "causal_discovery.agents.litellm_model._litellm_completion_cost",
        lambda response: 0.001,
    )
    model = LiteLLMJSONPolicyModel(
        model="gpt-5.4",
        allowed_actions=frozenset({"submit_graph"}),
    )

    raw = model.complete(
        system_prompt="system",
        session_prompt="{}",
        tool_history=tuple(),
        remaining_budget=0,
    )

    assert '"action":"submit_graph"' in raw
    assert model.prompt_tokens == 10
    assert model.completion_tokens == 3
    assert model.total_tokens == 13
    assert model.total_cost_usd == 0.001
    assert model.last_call is not None
    assert model.last_call["status"] == "success"
    assert model.last_call["cost_usd"] == 0.001


def test_litellm_model_rejects_missing_tool_call(monkeypatch):
    monkeypatch.setattr(
        "causal_discovery.agents.litellm_model._litellm_completion",
        lambda **kwargs: _Response([]),
    )
    model = LiteLLMJSONPolicyModel(
        model="openai/gpt-5.4",
        allowed_actions=frozenset({"submit_graph"}),
    )

    with pytest.raises(ValueError, match="exactly one"):
        model.complete(
            system_prompt="system",
            session_prompt="{}",
            tool_history=tuple(),
            remaining_budget=0,
        )
    assert model.last_call is not None
    assert model.last_call["status"] == "failed"


def test_litellm_model_rejects_wrong_tool_name(monkeypatch):
    monkeypatch.setattr(
        "causal_discovery.agents.litellm_model._litellm_completion",
        lambda **kwargs: _Response([_ToolCall(name="wrong_tool")]),
    )
    model = LiteLLMJSONPolicyModel(
        model="openai/gpt-5.4",
        allowed_actions=frozenset({"submit_graph"}),
    )

    with pytest.raises(ValueError, match="Unexpected tool call"):
        model.complete(
            system_prompt="system",
            session_prompt="{}",
            tool_history=tuple(),
            remaining_budget=0,
        )


def test_litellm_model_prefers_provider_reported_cost(monkeypatch):
    monkeypatch.setattr(
        "causal_discovery.agents.litellm_model._litellm_completion",
        lambda **kwargs: _Response([_ToolCall()], cost=0.0042),
    )
    model = LiteLLMJSONPolicyModel(
        model="openrouter/deepseek/deepseek-v4-flash",
        allowed_actions=frozenset({"submit_graph"}),
    )

    model.complete(
        system_prompt="system",
        session_prompt="{}",
        tool_history=tuple(),
        remaining_budget=0,
    )

    assert model.total_cost_usd == 0.0042
    assert model.last_call["cost_usd"] == 0.0042


def test_litellm_model_final_step_overrides_tool_schema(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _Response(
            [
                _ToolCall(
                    arguments=(
                        '{"action":"submit_graph","directed_edges":[],'
                        '"undirected_edges":[],"reasoning_summary":"final"}'
                    )
                )
            ]
        )

    monkeypatch.setattr(
        "causal_discovery.agents.litellm_model._litellm_completion",
        fake_completion,
    )
    model = LiteLLMJSONPolicyModel(
        model="openai/gpt-5.4",
        allowed_actions=allowed_actions_for_method("llm_stats"),
    )

    model.complete(
        system_prompt="system",
        session_prompt="{}",
        tool_history=tuple(),
        remaining_budget=1,
        working_memory=(
            {"step": 1, "action": "correlation", "reasoning_summary": "checked X0-X1"},
        ),
        allowed_actions=frozenset({"submit_graph"}),
        current_step=40,
        max_steps=40,
        final_step=True,
    )

    schema = captured["tools"][0]["function"]["parameters"]
    assert schema["properties"]["action"]["enum"] == ["submit_graph"]
    assert set(schema["properties"]) == {
        "action",
        "directed_edges",
        "undirected_edges",
        "reasoning_summary",
    }
    assert "This is the final allowed action. You must submit_graph now" in captured[
        "messages"
    ][1]["content"]
    assert "Step: 40/40" in captured["messages"][1]["content"]
    assert '"reasoning_summary":"checked X0-X1"' in captured["messages"][1]["content"]
