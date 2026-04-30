"""LLM-backed agent implementations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from causal_discovery.agents.actions import (
    AgentAction,
    CorrelationAction,
    IndependenceTestAction,
    InterveneAction,
    PartialCorrelationAction,
    SubmitGraphAction,
    ToolResult,
)
from causal_discovery.agents.prompts import (
    build_session_prompt,
    build_system_prompt_raw,
    build_system_prompt_stats,
)
from causal_discovery.equivalence.cpdag import canonical_undirected_edge


class LLMDecisionModel(Protocol):
    def complete(
        self,
        *,
        system_prompt: str,
        session_prompt: str,
        tool_history: tuple[ToolResult, ...],
        remaining_budget: int,
    ) -> str: ...


class SessionAgent(Protocol):
    def on_observation(self, data: np.ndarray, budget: int) -> None: ...

    def next_action(self, remaining_budget: int) -> AgentAction: ...

    def on_intervention_result(
        self, var: int, value: float, data: np.ndarray, remaining_budget: int
    ) -> None: ...

    def on_tool_result(self, result: ToolResult) -> None: ...


@dataclass(slots=True)
class _BaseLLMAgent:
    model: LLMDecisionModel
    variable_names: tuple[str, ...]
    allowed_actions: frozenset[str]
    _system_prompt: str

    def __post_init__(self) -> None:
        self._observational_data: np.ndarray | None = None
        self._session_prompt: str = ""
        self._tool_history: list[ToolResult] = []

    def on_observation(self, data: np.ndarray, budget: int) -> None:
        self._observational_data = np.array(data, copy=True)
        self._session_prompt = build_session_prompt(
            self.variable_names, self._observational_data, budget, self.allowed_actions
        )

    def next_action(self, remaining_budget: int) -> AgentAction:
        if self._observational_data is None:
            raise RuntimeError("Observation must be provided before requesting actions")

        raw = self.model.complete(
            system_prompt=self._system_prompt,
            session_prompt=self._session_prompt,
            tool_history=tuple(self._tool_history),
            remaining_budget=remaining_budget,
        )
        payload = _parse_json_object(raw)
        action_name = str(payload.get("action", "")).strip()
        if action_name not in self.allowed_actions:
            raise ValueError(
                f"Action '{action_name}' is not allowed for this agent: "
                f"{sorted(self.allowed_actions)}"
            )
        return _parse_action_payload(payload)

    def on_intervention_result(
        self, var: int, value: float, data: np.ndarray, remaining_budget: int
    ) -> None:
        self._tool_history.append(
            ToolResult(
                tool="intervene",
                payload={
                    "var": int(var),
                    "value": float(value),
                    "rows": np.asarray(data).tolist(),
                    "remaining_budget": int(remaining_budget),
                },
            )
        )

    def on_tool_result(self, result: ToolResult) -> None:
        self._tool_history.append(result)


class LLMRawAgent(_BaseLLMAgent):
    def __init__(
        self,
        model: LLMDecisionModel,
        variable_names: tuple[str, ...],
        allowed_actions: frozenset[str],
    ):
        super().__init__(
            model=model,
            variable_names=variable_names,
            allowed_actions=allowed_actions,
            _system_prompt=build_system_prompt_raw(allowed_actions),
        )


class LLMStatsAgent(_BaseLLMAgent):
    def __init__(
        self,
        model: LLMDecisionModel,
        variable_names: tuple[str, ...],
        allowed_actions: frozenset[str],
    ):
        super().__init__(
            model=model,
            variable_names=variable_names,
            allowed_actions=allowed_actions,
            _system_prompt=build_system_prompt_stats(allowed_actions),
        )


def _parse_json_object(raw: str) -> dict:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        decoder = json.JSONDecoder()
        try:
            payload, _ = decoder.raw_decode(raw.lstrip())
        except json.JSONDecodeError:
            raise ValueError(f"Model output must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Model output must be a JSON object, got {type(payload).__name__}")
    return payload


def _parse_action_payload(payload: dict) -> AgentAction:
    action = str(payload["action"]).strip()
    if action == "intervene":
        return InterveneAction(
            var=_parse_index(payload["var"], field_name="var"),
            value=float(payload["value"]),
        )
    if action == "submit_graph":
        directed = tuple(_parse_directed_edges(payload.get("directed_edges", [])))
        undirected = tuple(_parse_undirected_edges(payload.get("undirected_edges", [])))
        return SubmitGraphAction(
            directed_edges=directed,
            undirected_edges=undirected,
            reasoning_summary=str(payload.get("reasoning_summary", "")),
        )
    if action == "correlation":
        return CorrelationAction(
            i=_parse_index(payload["i"], field_name="i"),
            j=_parse_index(payload["j"], field_name="j"),
        )
    if action == "partial_correlation":
        cond = tuple(
            _parse_index(x, field_name="conditioning_on")
            for x in (payload.get("conditioning_on") or [])
        )
        return PartialCorrelationAction(
            i=_parse_index(payload["i"], field_name="i"),
            j=_parse_index(payload["j"], field_name="j"),
            conditioning_on=cond,
        )
    if action == "independence_test":
        cond = tuple(
            _parse_index(x, field_name="conditioning_on")
            for x in (payload.get("conditioning_on") or [])
        )
        return IndependenceTestAction(
            i=_parse_index(payload["i"], field_name="i"),
            j=_parse_index(payload["j"], field_name="j"),
            conditioning_on=cond,
            alpha=float(payload.get("alpha", 0.05)),
        )
    raise ValueError(f"Unsupported action '{action}'")


def _parse_index(value: object, *, field_name: str) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
            return int(raw)
        if len(raw) >= 2 and raw[0] in {"X", "x"} and raw[1:].isdigit():
            return int(raw[1:])
    raise ValueError(f"{field_name} must be an integer index (or Xk), got {value!r}")


def _parse_directed_edges(raw_edges: object) -> list[tuple[int, int]]:
    if raw_edges is None:
        raw_edges = []
    if not isinstance(raw_edges, list):
        raise ValueError("directed_edges must be a list")
    out: list[tuple[int, int]] = []
    for edge in raw_edges:
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError("Each directed edge must be [src, dst]")
        out.append((int(edge[0]), int(edge[1])))
    return out


def _parse_undirected_edges(raw_edges: object) -> list[tuple[int, int]]:
    if raw_edges is None:
        raw_edges = []
    if not isinstance(raw_edges, list):
        raise ValueError("undirected_edges must be a list")
    out: list[tuple[int, int]] = []
    for edge in raw_edges:
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError("Each undirected edge must be [a, b]")
        out.append(canonical_undirected_edge(int(edge[0]), int(edge[1])))
    return out
