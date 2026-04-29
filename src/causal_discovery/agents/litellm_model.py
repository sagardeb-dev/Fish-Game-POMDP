"""LiteLLM-backed strict tool-call model adapter."""

from __future__ import annotations

import json
import time
from typing import Any

from causal_discovery.agents.actions import ToolResult
from causal_discovery.agents.tool_schema import make_action_response_schema, make_action_tool


def normalize_litellm_model(model: str) -> str:
    model = model.strip()
    if model.startswith(("openai/", "anthropic/", "openrouter/")):
        return model
    if model.startswith(("gpt-", "o")):
        return f"openai/{model}"
    if model.startswith("claude-"):
        return f"anthropic/{model}"
    raise ValueError(
        f"Cannot infer provider for model {model!r}. "
        "Use openai/..., anthropic/..., openrouter/... or a legacy gpt-/claude- name."
    )


def provider_for_model(model: str) -> str:
    normalized = normalize_litellm_model(model)
    return normalized.split("/", 1)[0]


def _litellm_completion(**kwargs):
    from litellm import completion

    return completion(**kwargs)


def _litellm_completion_cost(response) -> float | None:
    provider_cost = _get(_get(response, "usage", {}), "cost", None)
    if provider_cost is not None:
        return float(provider_cost)
    try:
        from litellm import completion_cost

        return float(completion_cost(completion_response=response))
    except Exception:  # noqa: BLE001
        return None


class LiteLLMJSONPolicyModel:
    """Strict single-tool-call adapter using LiteLLM completion transport."""

    def __init__(self, model: str, allowed_actions: frozenset[str]) -> None:
        self._model = normalize_litellm_model(model)
        self._provider = provider_for_model(model)
        self._action_schema = make_action_response_schema(allowed_actions)
        self._action_tool = make_action_tool(allowed_actions)
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0
        self.total_cost_usd = 0.0
        self.calls = 0
        self.last_call: dict[str, Any] | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return self._provider

    def complete(
        self,
        *,
        system_prompt: str,
        session_prompt: str,
        tool_history: tuple[ToolResult, ...],
        remaining_budget: int,
    ) -> str:
        self.calls += 1
        history_payload = [{"tool": item.tool, "payload": item.payload} for item in tool_history]
        user_msg = (
            f"Session data JSON: {session_prompt}\n"
            f"Remaining budget: {remaining_budget}\n"
            "Tool history JSON: "
            f"{json.dumps(history_payload, separators=(',', ':'), ensure_ascii=True, allow_nan=False)}\n"
            "Call the causal_discovery_action tool exactly once for your next action."
        )
        request = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "tools": [self._action_tool],
            "tool_choice": {
                "type": "function",
                "function": {"name": self._action_schema["name"]},
            },
            "temperature": 0,
        }
        started = time.perf_counter()
        response = None
        try:
            response = _litellm_completion(**request)
            parsed = self._parse_response(response)
            usage = _usage_payload(response)
            self._add_usage(usage)
            cost = _litellm_completion_cost(response)
            if cost is not None:
                self.total_cost_usd += cost
            self.last_call = {
                "provider": self._provider,
                "model": self._model,
                "request": request,
                "raw_response": _jsonable_response(response),
                "parsed_action": parsed,
                "usage": usage,
                "cost_usd": cost,
                "latency_sec": round(time.perf_counter() - started, 6),
                "status": "success",
                "error": None,
            }
            return json.dumps(parsed, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        except Exception as exc:
            self.last_call = {
                "provider": self._provider,
                "model": self._model,
                "request": request,
                "raw_response": _jsonable_response(response) if response is not None else None,
                "parsed_action": None,
                "usage": _usage_payload(response) if response is not None else {},
                "cost_usd": None,
                "latency_sec": round(time.perf_counter() - started, 6),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            raise

    def _parse_response(self, response) -> dict[str, Any]:
        message = _get(_get(response, "choices", [])[0], "message")
        tool_calls = _get(message, "tool_calls", None) or []
        if len(tool_calls) != 1:
            content = _get(message, "content", "") or ""
            snippet = str(content)[:240].replace("\n", "\\n")
            raise ValueError(
                "Model must return exactly one causal_discovery_action tool call; "
                f"got {len(tool_calls)}. content_prefix={snippet!r}"
            )
        tool_call = tool_calls[0]
        function = _get(tool_call, "function")
        name = _get(function, "name")
        if name != self._action_schema["name"]:
            raise ValueError(
                f"Unexpected tool call {name!r}; expected {self._action_schema['name']!r}"
            )
        content = _get(function, "arguments")
        try:
            parsed = json.loads(content) if isinstance(content, str) else content
        except json.JSONDecodeError as exc:
            snippet = str(content)[:240].replace("\n", "\\n")
            raise ValueError(
                "Model output JSON parse failed "
                f"({type(exc).__name__}: {exc}). content_prefix={snippet!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError("Model output must be a JSON object")
        return parsed

    def _add_usage(self, usage: dict[str, int]) -> None:
        self.prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.completion_tokens += int(usage.get("completion_tokens", 0))
        self.total_tokens += int(usage.get("total_tokens", 0))
        self.cache_creation_input_tokens += int(usage.get("cache_creation_input_tokens", 0))
        self.cache_read_input_tokens += int(usage.get("cache_read_input_tokens", 0))


def _get(value, name: str, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _usage_payload(response) -> dict[str, int]:
    usage = _get(response, "usage", None) or {}
    prompt = _int_usage(usage, "prompt_tokens") or _int_usage(usage, "input_tokens")
    completion = _int_usage(usage, "completion_tokens") or _int_usage(usage, "output_tokens")
    cache_creation = _int_usage(usage, "cache_creation_input_tokens")
    cache_read = _int_usage(usage, "cache_read_input_tokens")
    total = _int_usage(usage, "total_tokens")
    if total == 0:
        total = prompt + completion + cache_creation + cache_read
    return {
        "prompt_tokens": prompt + cache_creation + cache_read,
        "completion_tokens": completion,
        "total_tokens": total,
        "cache_creation_input_tokens": cache_creation,
        "cache_read_input_tokens": cache_read,
    }


def _int_usage(usage, name: str) -> int:
    value = _get(usage, name, 0)
    return int(value or 0)


def _jsonable_response(response):
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    return repr(response)
