"""Deterministic mock agent for infrastructure tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causal_discovery.agents.actions import AgentAction, ToolResult
from causal_discovery.agents.llm import SessionAgent


@dataclass(slots=True)
class MockAgent(SessionAgent):
    planned_actions: tuple[AgentAction, ...]

    def __post_init__(self) -> None:
        self._index = 0
        self._observed = False
        self.intervention_results: list[dict] = []
        self.tool_results: list[ToolResult] = []

    def on_observation(self, data: np.ndarray, budget: int) -> None:
        _ = data
        _ = budget
        self._observed = True

    def next_action(self, remaining_budget: int) -> AgentAction:
        _ = remaining_budget
        if not self._observed:
            raise RuntimeError("MockAgent requires observation before next_action")
        if self._index >= len(self.planned_actions):
            raise RuntimeError("MockAgent has no more planned actions")
        action = self.planned_actions[self._index]
        self._index += 1
        return action

    def on_intervention_result(
        self, var: int, value: float, data: np.ndarray, remaining_budget: int
    ) -> None:
        self.intervention_results.append(
            {
                "var": int(var),
                "value": float(value),
                "shape": tuple(int(x) for x in data.shape),
                "remaining_budget": int(remaining_budget),
            }
        )

    def on_tool_result(self, result: ToolResult) -> None:
        self.tool_results.append(result)
