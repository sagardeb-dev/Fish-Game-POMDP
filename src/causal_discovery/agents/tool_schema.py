"""Strict action-tool schema for LLM causal-discovery agents."""

from __future__ import annotations

from typing import Any


ACTION_NAMES = (
    "intervene",
    "correlation",
    "partial_correlation",
    "independence_test",
    "submit_graph",
)


FIELD_SCHEMAS: dict[str, dict[str, Any]] = {
    "action": {"type": "string"},
    "var": {"type": ["integer", "null"]},
    "value": {"type": ["number", "null"]},
    "i": {"type": ["integer", "null"]},
    "j": {"type": ["integer", "null"]},
    "conditioning_on": {
        "type": "array",
        "items": {"type": "integer"},
    },
    "alpha": {"type": ["number", "null"]},
    "directed_edges": {
        "type": "array",
        "items": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
        },
    },
    "undirected_edges": {
        "type": "array",
        "items": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
        },
    },
    "reasoning_summary": {"type": "string"},
}


ACTION_FIELDS = {
    "intervene": ("var", "value", "reasoning_summary"),
    "correlation": ("i", "j", "reasoning_summary"),
    "partial_correlation": ("i", "j", "conditioning_on", "reasoning_summary"),
    "independence_test": ("i", "j", "conditioning_on", "alpha", "reasoning_summary"),
    "submit_graph": ("directed_edges", "undirected_edges", "reasoning_summary"),
}


def action_fields_for_allowed_actions(allowed_actions: frozenset[str]) -> tuple[str, ...]:
    fields = ["action"]
    for action in ACTION_NAMES:
        if action not in allowed_actions:
            continue
        for field in ACTION_FIELDS[action]:
            if field not in fields:
                fields.append(field)
    return tuple(fields)


def make_action_response_schema(allowed_actions: frozenset[str]) -> dict[str, Any]:
    unknown = allowed_actions.difference(ACTION_NAMES)
    if unknown:
        raise ValueError(f"Unknown action names in tool schema: {sorted(unknown)}")
    fields = action_fields_for_allowed_actions(allowed_actions)
    properties = {name: dict(FIELD_SCHEMAS[name]) for name in fields}
    properties["action"] = {"type": "string", "enum": sorted(allowed_actions)}
    return {
        "name": "causal_discovery_action",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": properties,
            "required": list(fields),
            "additionalProperties": False,
        },
    }


def make_action_tool(allowed_actions: frozenset[str]) -> dict[str, Any]:
    schema = make_action_response_schema(allowed_actions)
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": "Submit exactly one causal-discovery benchmark action.",
            "strict": schema["strict"],
            "parameters": schema["schema"],
        },
    }


def allowed_actions_for_method(method: str) -> frozenset[str]:
    if method == "llm_raw":
        return frozenset({"intervene", "submit_graph"})
    if method == "pc_cpdag_llm":
        return frozenset({"intervene", "submit_graph"})
    if method == "llm_raw_obs":
        return frozenset({"submit_graph"})
    if method == "llm_stats":
        return frozenset(
            {
                "correlation",
                "partial_correlation",
                "independence_test",
                "intervene",
                "submit_graph",
            }
        )
    if method in {"llm_stats_obs", "llm_stats_cpdag_greedy"}:
        return frozenset(
            {
                "correlation",
                "partial_correlation",
                "independence_test",
                "submit_graph",
            }
        )
    raise ValueError(f"Method does not use an LLM action schema: {method}")

