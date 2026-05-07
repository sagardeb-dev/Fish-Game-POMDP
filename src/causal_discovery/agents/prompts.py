"""Prompt builders for causal discovery LLM agents."""

from __future__ import annotations

import json

import numpy as np

from causal_discovery.agents.tool_schema import ACTION_NAMES, action_fields_for_allowed_actions


INTERVENTION_RULE_TEXT = (
    "An intervention forces one variable to a fixed value, breaking its normal "
    "incoming dependencies. All other variables still follow their normal "
    "relationships. Comparing data before and after intervention can provide "
    "evidence about causal direction."
)


DEPENDENCE_CAUTION_TEXT = (
    "Dependence and correlation are not direct causation by themselves. "
    "Chains and shared causes can induce statistical dependence between variables "
    "that are not directly connected. Do not add an edge merely because two "
    "variables are correlated.\n"
)


INTERVENTION_VALUE_TEXT = (
    "Choose intervention values far enough from the observational mean to make "
    "downstream shifts detectable, while staying on the observed scale.\n"
)


OUTPUT_FIELD_DESCRIPTIONS = {
    "action": "",
    "var": "int; required for intervene",
    "value": "float; required for intervene",
    "i": "int; required for statistical tools",
    "j": "int; required for statistical tools",
    "conditioning_on": "list[int]; required for partial_correlation and independence_test",
    "alpha": "float; required for independence_test",
    "directed_edges": "list[list[int, int]]; required for submit_graph",
    "undirected_edges": "list[list[int, int]]; required for submit_graph",
    "reasoning_summary": "string; brief public state summary required for every action",
}


def build_system_prompt_raw(allowed_actions: frozenset[str]) -> str:
    allow_interventions = "intervene" in allowed_actions
    base = (
        "You are performing causal discovery over an unknown linear-Gaussian "
        "DAG with full observability and no hidden confounders.\n"
        f"{DEPENDENCE_CAUTION_TEXT}"
    )
    if allow_interventions:
        actions = (
            f"{INTERVENTION_RULE_TEXT}\n"
            f"{INTERVENTION_VALUE_TEXT}"
            "The final target is a directed graph. Use undirected_edges only for "
            "genuinely unresolved uncertainty.\n"
            "Allowed actions:\n"
            "- intervene(var, value)\n"
            "- submit_graph(directed_edges, undirected_edges, reasoning_summary)\n"
        )
    else:
        actions = (
            "No experiments/interventions are available for this run.\n"
            "Submit a partial graph. Use directed_edges only for directions supported "
            "by the observational evidence, and use undirected_edges for adjacencies "
            "whose direction is not supported.\n"
            "Allowed actions:\n"
            "- submit_graph(directed_edges, undirected_edges, reasoning_summary)\n"
        )
    return (
        base
        + actions
        + "Use the provided action tool exactly once for each turn.\n"
        + "Do not request hidden or unavailable metadata."
    )


def build_system_prompt_stats(allowed_actions: frozenset[str]) -> str:
    allow_interventions = "intervene" in allowed_actions
    decision_policy = (
        "Decision policy:\n"
        "- Use statistical tools to gather evidence before submitting.\n"
        "- Orient edges when evidence supports direction; leave undirected only when evidence is ambiguous.\n"
    )
    if allow_interventions:
        decision_policy = (
            "Decision policy:\n"
            "- Use statistical tools when helpful, and use interventions when they can resolve causal direction.\n"
            "- Orient edges when evidence supports direction; leave undirected only when evidence is ambiguous.\n"
        )
    else:
        decision_policy = (
            decision_policy
            + "- Perform at least 3 statistical tool actions before your first submit_graph action.\n"
        )
    base = (
        "You are performing causal discovery over an unknown linear-Gaussian "
        "DAG with full observability and no hidden confounders.\n"
        f"{DEPENDENCE_CAUTION_TEXT}"
        "Tool purpose:\n"
        "- correlation(i,j): coarse association check, not causal direction by itself.\n"
        "- partial_correlation(i,j,S): test direct dependence after conditioning on S.\n"
        "- independence_test(i,j,S,alpha): decide whether evidence supports conditional independence.\n"
        "Tool data source:\n"
        "- Before any intervention, statistical tools use the observational dataset.\n"
        "- A tool can return value=null with an error when a statistic is undefined.\n"
        f"{decision_policy}"
    )
    if allow_interventions:
        actions = (
            f"{INTERVENTION_RULE_TEXT}\n"
            f"{INTERVENTION_VALUE_TEXT}"
            "- After an intervention, statistical tools use the latest intervention dataset.\n"
            "The final target is a directed graph. Use undirected_edges only for "
            "genuinely unresolved uncertainty.\n"
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
            "Submit a partial graph using undirected_edges when direction is not supported.\n"
            "Allowed actions:\n"
            "- correlation(i, j)\n"
            "- partial_correlation(i, j, conditioning_on)\n"
            "- independence_test(i, j, conditioning_on, alpha)\n"
            "- submit_graph(directed_edges, undirected_edges, reasoning_summary)\n"
        )
    return (
        base
        + actions
        + "Use the provided action tool exactly once for each turn.\n"
        + "Do not request hidden or unavailable metadata."
    )


def build_session_prompt(
    variable_names: tuple[str, ...],
    observational_data: np.ndarray,
    budget: int,
    allowed_actions: frozenset[str],
    *,
    include_observational_data: bool = True,
    session_context: dict | None = None,
) -> str:
    action_text = "|".join(name for name in ACTION_NAMES if name in allowed_actions)
    output_schema = {"action": action_text}
    for field in action_fields_for_allowed_actions(allowed_actions):
        if field == "action":
            continue
        output_schema[field] = OUTPUT_FIELD_DESCRIPTIONS[field]
    payload = {
        "variables": list(variable_names),
        "budget": int(budget),
        "output_schema": output_schema,
        "json_contract": "Return all fields shown in output_schema on every action. Use null for unused scalar fields, [] for unused list fields, and a brief reasoning_summary.",
    }
    if include_observational_data:
        payload["observational_data"] = observational_data.tolist()
    if session_context:
        payload.update(session_context)
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
