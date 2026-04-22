"""Prompt builders for causal discovery LLM agents."""

from __future__ import annotations

import json

import numpy as np


INTERVENTION_RULE_TEXT = (
    "An intervention forces one variable to a fixed value, breaking its normal "
    "incoming dependencies. All other variables still follow their normal "
    "relationships. By comparing data before and after intervention, you can "
    "determine causal direction."
)


def build_system_prompt_raw(allow_interventions: bool = True) -> str:
    base = (
        "You are performing active causal discovery over an unknown linear-Gaussian "
        "DAG with full observability and no hidden confounders.\n"
        f"{INTERVENTION_RULE_TEXT}\n"
    )
    if allow_interventions:
        actions = (
            "Allowed actions:\n"
            "- intervene(var, value)\n"
            "- submit_graph(directed_edges, undirected_edges, reasoning_summary)\n"
        )
    else:
        actions = (
            "No experiments/interventions are available for this run.\n"
            "Allowed actions:\n"
            "- submit_graph(directed_edges, undirected_edges, reasoning_summary)\n"
        )
    return (
        base
        + actions
        + "Return strict JSON with one action.\n"
        + "Do not request unavailable metadata."
    )


def build_system_prompt_stats(allow_interventions: bool = True) -> str:
    base = (
        "You are performing active causal discovery over an unknown linear-Gaussian "
        "DAG with full observability and no hidden confounders.\n"
        f"{INTERVENTION_RULE_TEXT}\n"
    )
    if allow_interventions:
        actions = (
            "Allowed actions:\n"
            "- correlation(i, j)\n"
            "- partial_correlation(i, j, conditioning_on)\n"
            "- independence_test(i, j, conditioning_on, alpha)\n"
            "- intervene(var, value)\n"
            "- submit_graph(directed_edges, undirected_edges, reasoning_summary)\n"
        )
    else:
        actions = (
            "No experiments/interventions are available for this run.\n"
            "Allowed actions:\n"
            "- correlation(i, j)\n"
            "- partial_correlation(i, j, conditioning_on)\n"
            "- independence_test(i, j, conditioning_on, alpha)\n"
            "- submit_graph(directed_edges, undirected_edges, reasoning_summary)\n"
        )
    return (
        base
        + actions
        + "Return strict JSON with one action.\n"
        + "Do not request unavailable metadata."
    )


def build_session_prompt(
    variable_names: tuple[str, ...], observational_data: np.ndarray, budget: int
) -> str:
    payload = {
        "variables": list(variable_names),
        "budget": int(budget),
        "observational_data": observational_data.tolist(),
        "output_schema": {
            "action": "intervene|correlation|partial_correlation|independence_test|submit_graph",
            "var": "int",
            "value": "float",
            "i": "int",
            "j": "int",
            "conditioning_on": "list[int]",
            "alpha": "float",
            "directed_edges": "list[list[int, int]]",
            "undirected_edges": "list[list[int, int]]",
            "reasoning_summary": "string",
        },
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
