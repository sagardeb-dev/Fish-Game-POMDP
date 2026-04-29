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


def make_action_response_schema(allowed_actions: frozenset[str]) -> dict[str, Any]:
    unknown = allowed_actions.difference(ACTION_NAMES)
    if unknown:
        raise ValueError(f"Unknown action names in tool schema: {sorted(unknown)}")
    return {
        "name": "causal_discovery_action",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": sorted(allowed_actions),
                },
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
            },
            "required": [
                "action",
                "var",
                "value",
                "i",
                "j",
                "conditioning_on",
                "alpha",
                "directed_edges",
                "undirected_edges",
                "reasoning_summary",
            ],
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
    if method == "llm_stats_obs":
        return frozenset(
            {
                "correlation",
                "partial_correlation",
                "independence_test",
                "submit_graph",
            }
        )
    raise ValueError(f"Method does not use an LLM action schema: {method}")

