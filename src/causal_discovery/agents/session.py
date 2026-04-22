"""Session driver for benchmark-facing agents."""

from __future__ import annotations

import numpy as np

from causal_discovery.agents.actions import (
    CorrelationAction,
    IndependenceTestAction,
    InterveneAction,
    PartialCorrelationAction,
    SubmitGraphAction,
    ToolResult,
)
from causal_discovery.agents.llm import SessionAgent
from causal_discovery.agents.stats_tools import (
    correlation,
    independence_test,
    partial_correlation,
)
from causal_discovery.runtime import BenchmarkEnv, SessionOutput
from causal_discovery.scoring.submission import GraphSubmission


def run_agent_session(env: BenchmarkEnv, agent: SessionAgent) -> SessionOutput:
    """Run one agent session through the benchmark runtime contract."""
    observational_data = env.observe()
    latest_intervention_data: np.ndarray | None = None
    agent.on_observation(observational_data, env.remaining_budget)

    while True:
        action = agent.next_action(env.remaining_budget)

        if isinstance(action, InterveneAction):
            if env.remaining_budget <= 0:
                raise RuntimeError("Agent requested intervention but budget is exhausted")
            samples = env.intervene(var=action.var, value=action.value)
            latest_intervention_data = samples
            agent.on_intervention_result(
                var=action.var,
                value=action.value,
                data=samples,
                remaining_budget=env.remaining_budget,
            )
            continue

        if isinstance(action, CorrelationAction):
            data_source = "latest_intervention" if latest_intervention_data is not None else "observational"
            data = latest_intervention_data if latest_intervention_data is not None else observational_data
            payload = {
                "i": action.i,
                "j": action.j,
                "data_source": data_source,
                "rows": int(data.shape[0]),
            }
            try:
                payload["value"] = correlation(data, action.i, action.j)
            except ValueError as exc:
                payload["value"] = None
                payload["error"] = str(exc)
            agent.on_tool_result(
                ToolResult(
                    tool="correlation",
                    payload=payload,
                )
            )
            continue

        if isinstance(action, PartialCorrelationAction):
            data_source = "latest_intervention" if latest_intervention_data is not None else "observational"
            data = latest_intervention_data if latest_intervention_data is not None else observational_data
            payload = {
                "i": action.i,
                "j": action.j,
                "conditioning_on": list(action.conditioning_on),
                "data_source": data_source,
                "rows": int(data.shape[0]),
            }
            try:
                payload["value"] = partial_correlation(
                    data, action.i, action.j, action.conditioning_on
                )
            except ValueError as exc:
                payload["value"] = None
                payload["error"] = str(exc)
            agent.on_tool_result(
                ToolResult(
                    tool="partial_correlation",
                    payload=payload,
                )
            )
            continue

        if isinstance(action, IndependenceTestAction):
            data_source = "latest_intervention" if latest_intervention_data is not None else "observational"
            data = latest_intervention_data if latest_intervention_data is not None else observational_data
            payload = {
                "i": action.i,
                "j": action.j,
                "conditioning_on": list(action.conditioning_on),
                "alpha": action.alpha,
                "data_source": data_source,
                "rows": int(data.shape[0]),
            }
            try:
                independent, p_value = independence_test(
                    data,
                    action.i,
                    action.j,
                    action.conditioning_on,
                    alpha=action.alpha,
                )
                payload["independent"] = independent
                payload["p_value"] = p_value
            except ValueError as exc:
                payload["independent"] = None
                payload["p_value"] = None
                payload["error"] = str(exc)
            agent.on_tool_result(
                ToolResult(
                    tool="independence_test",
                    payload=payload,
                )
            )
            continue

        if isinstance(action, SubmitGraphAction):
            submission = GraphSubmission(
                num_nodes=env.num_variables,
                directed_edges=frozenset(action.directed_edges),
                undirected_edges=frozenset(action.undirected_edges),
            )
            return env.submit_graph(submission)

        raise TypeError(f"Unsupported action type: {type(action).__name__}")
