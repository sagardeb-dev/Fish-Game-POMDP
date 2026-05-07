"""LiteLLM-backed strict tool-call model adapter."""

from __future__ import annotations

import json
import time
from typing import Any

from causal_discovery.agents.actions import ToolResult
from causal_discovery.agents.tool_schema import make_action_response_schema, make_action_tool


_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _is_reasoning_model(model: str) -> bool:
    bare = model.rsplit("/", 1)[-1]
    return bare.startswith(_REASONING_MODEL_PREFIXES)


def _is_anthropic_model(model: str) -> bool:
    return "anthropic/" in normalize_litellm_model(model)


def normalize_litellm_model(model: str) -> str:
    model = model.strip()
    if model.startswith(("openai/", "anthropic/", "openrouter/")):
        return model
    if model.startswith("google/"):
        return f"openrouter/{model}"
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
    import litellm
    from litellm import completion
    from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception

    litellm.suppress_debug_info = True

    def _is_retryable(exc: BaseException) -> bool:
        name = type(exc).__name__
        msg = str(exc).lower()
        if "ratelimit" in name.lower() or "429" in msg:
            return True
        if "serviceunavailable" in name.lower() or "500" in msg or "502" in msg or "503" in msg:
            return True
        if "apierror" in name.lower() and ("internal server" in msg or "temporarily" in msg):
            return True
        return False

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=2, max=30, jitter=2),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    def _call():
        return completion(**kwargs)

    return _call()


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
        working_memory: tuple[dict, ...] = tuple(),
        allowed_actions: frozenset[str] | None = None,
        current_step: int | None = None,
        max_steps: int | None = None,
        final_step: bool = False,
    ) -> str:
        self.calls += 1
        effective_allowed_actions = allowed_actions or frozenset(
            self._action_schema["schema"]["properties"]["action"]["enum"]
        )
        action_schema = make_action_response_schema(effective_allowed_actions)
        action_tool = make_action_tool(effective_allowed_actions)
        history_payload = [{"tool": item.tool, "payload": item.payload} for item in tool_history]
        step_text = ""
        if current_step is not None and max_steps is not None:
            steps_remaining = max(max_steps - current_step, 0)
            step_text = (
                f"Step: {current_step}/{max_steps}\n"
                f"Steps remaining after this action: {steps_remaining}\n"
            )
        final_text = ""
        if final_step:
            final_text = (
                "This is the final allowed action. You must submit_graph now using "
                "all evidence collected so far. Do not request more tools or interventions.\n"
            )
        user_msg = (
            f"Session data JSON: {session_prompt}\n"
            f"Remaining budget: {remaining_budget}\n"
            f"{step_text}"
            "Tool history JSON: "
            f"{json.dumps(history_payload, separators=(',', ':'), ensure_ascii=True, allow_nan=False)}\n"
            "Working memory JSON: "
            f"{json.dumps(working_memory, separators=(',', ':'), ensure_ascii=True, allow_nan=False)}\n"
            f"{final_text}"
            "Call the causal_discovery_action tool exactly once for your next action."
        )
        if _is_anthropic_model(self._model):
            system_msg: dict = {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
            }
        else:
            system_msg = {"role": "system", "content": system_prompt}
        request = {
            "model": self._model,
            "messages": [system_msg, {"role": "user", "content": user_msg}],
            "tools": [action_tool],
            "tool_choice": {
                "type": "function",
                "function": {"name": action_schema["name"]},
            },
        }
        if not _is_reasoning_model(self._model):
            request["temperature"] = 0
        started = time.perf_counter()
        response = None
        try:
            response = _litellm_completion(**request)
            parsed = self._parse_response(response, action_schema)
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

    def _parse_response(self, response, action_schema: dict[str, Any]) -> dict[str, Any]:
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
        if name != action_schema["name"]:
            raise ValueError(
                f"Unexpected tool call {name!r}; expected {action_schema['name']!r}"
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
