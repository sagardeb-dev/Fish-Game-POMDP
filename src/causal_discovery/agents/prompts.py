"""Prompt builders for causal discovery LLM agents."""

from __future__ import annotations

import json

import numpy as np

from causal_discovery.agents.tool_schema import ACTION_NAMES, action_fields_for_allowed_actions


INTERVENTION_RULE_TEXT = (
    "An intervention forces one variable to a fixed value, breaking its normal "
    "incoming dependencies. All other variables still follow their normal "
    "relationships. By comparing data before and after intervention, you can "
    "determine causal direction."
)

ACTION_SIGNATURES = {
    "intervene": "intervene(var, value)",
    "correlation": "correlation(i, j)",
    "partial_correlation": "partial_correlation(i, j, conditioning_on)",
    "independence_test": "independence_test(i, j, conditioning_on, alpha)",
    "submit_graph": "submit_graph(directed_edges, undirected_edges, reasoning_summary)",
}

OUTPUT_FIELD_DESCRIPTIONS = {
    "action": "",
    "var": "int|null; required for intervene",
    "value": "float|null; required for intervene",
    "i": "int|null; required for statistical tools",
    "j": "int|null; required for statistical tools",
    "conditioning_on": "list[int]",
    "alpha": "float|null; required for independence_test",
    "directed_edges": "list[list[int, int]]",
    "undirected_edges": "list[list[int, int]]",
    "reasoning_summary": "string",
}


def format_allowed_actions(allowed_actions: frozenset[str]) -> str:
    ordered = [action for action in ACTION_NAMES if action in allowed_actions]
    lines = ["Allowed actions:"]
    lines.extend(f"- {ACTION_SIGNATURES[action]}" for action in ordered)
    return "\n".join(lines) + "\n"


def build_system_prompt_raw(allowed_actions: frozenset[str]) -> str:
    base = (
        "You are performing active causal discovery over an unknown linear-Gaussian "
        "DAG with full observability and no hidden confounders.\n"
        f"{INTERVENTION_RULE_TEXT}\n"
    )
    actions = ""
    if "intervene" not in allowed_actions:
        actions += "No experiments/interventions are available for this run.\n"
    actions += format_allowed_actions(allowed_actions)
    return (
        base
        + actions
        + "Use the provided action tool exactly once for each turn.\n"
        + "Do not request unavailable metadata."
    )


def build_system_prompt_stats(allowed_actions: frozenset[str]) -> str:
    base = (
        "You are performing active causal discovery over an unknown linear-Gaussian "
        "DAG with full observability and no hidden confounders.\n"
        f"{INTERVENTION_RULE_TEXT}\n"
        "Tool purpose:\n"
        "- correlation(i,j): coarse association check, not causal direction by itself.\n"
        "- partial_correlation(i,j,S): test direct dependence after conditioning on S.\n"
        "- independence_test(i,j,S,alpha): decide whether evidence supports conditional independence.\n"
        "Tool data source:\n"
        "- Before any intervention, statistical tools use the observational dataset.\n"
        "- After an intervention, statistical tools use the latest intervention dataset.\n"
        "- A tool can return value=null with an error when a statistic is undefined.\n"
        "Decision policy:\n"
        "- Use statistical tools to gather evidence before submitting.\n"
        "- Perform at least 3 statistical tool actions before your first submit_graph action.\n"
        "- Orient edges when evidence supports direction; leave undirected only when evidence is ambiguous.\n"
        "- Avoid submitting all-undirected graphs unless evidence truly supports ambiguity.\n"
    )
    actions = ""
    if "intervene" not in allowed_actions:
        actions += "No experiments/interventions are available for this run.\n"
    actions += format_allowed_actions(allowed_actions)
    return (
        base
        + actions
        + "Use the provided action tool exactly once for each turn.\n"
        + "Do not request unavailable metadata."
    )


def build_session_prompt(
    variable_names: tuple[str, ...],
    observational_data: np.ndarray,
    budget: int,
    allowed_actions: frozenset[str],
) -> str:
    ordered_actions = [action for action in ACTION_NAMES if action in allowed_actions]
    output_schema = {
        field: OUTPUT_FIELD_DESCRIPTIONS[field]
        for field in action_fields_for_allowed_actions(allowed_actions)
    }
    output_schema["action"] = "|".join(ordered_actions)
    payload = {
        "variables": list(variable_names),
        "budget": int(budget),
        "observational_data": observational_data.tolist(),
        "output_schema": output_schema,
        "json_contract": "Return all output_schema fields on every action.",
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
