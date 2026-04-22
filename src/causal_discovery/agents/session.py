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
    agent.on_observation(observational_data, env.remaining_budget)

    while True:
        action = agent.next_action(env.remaining_budget)

        if isinstance(action, InterveneAction):
            if env.remaining_budget <= 0:
                raise RuntimeError("Agent requested intervention but budget is exhausted")
            samples = env.intervene(var=action.var, value=action.value)
            agent.on_intervention_result(
                var=action.var,
                value=action.value,
                data=samples,
                remaining_budget=env.remaining_budget,
            )
            continue

        if isinstance(action, CorrelationAction):
            value = correlation(observational_data, action.i, action.j)
            agent.on_tool_result(
                ToolResult(
                    tool="correlation",
                    payload={"i": action.i, "j": action.j, "value": value},
                )
            )
            continue

        if isinstance(action, PartialCorrelationAction):
            value = partial_correlation(
                observational_data, action.i, action.j, action.conditioning_on
            )
            agent.on_tool_result(
                ToolResult(
                    tool="partial_correlation",
                    payload={
                        "i": action.i,
                        "j": action.j,
                        "conditioning_on": list(action.conditioning_on),
                        "value": value,
                    },
                )
            )
            continue

        if isinstance(action, IndependenceTestAction):
            independent, p_value = independence_test(
                observational_data,
                action.i,
                action.j,
                action.conditioning_on,
                alpha=action.alpha,
            )
            agent.on_tool_result(
                ToolResult(
                    tool="independence_test",
                    payload={
                        "i": action.i,
                        "j": action.j,
                        "conditioning_on": list(action.conditioning_on),
                        "alpha": action.alpha,
                        "independent": independent,
                        "p_value": p_value,
                    },
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
