"""Agent action and tool result types."""

from __future__ import annotations

from dataclasses import dataclass

from causal_discovery.scoring.submission import DirectedEdge, UndirectedEdge


@dataclass(frozen=True, slots=True)
class InterveneAction:
    var: int
    value: float


@dataclass(frozen=True, slots=True)
class SubmitGraphAction:
    directed_edges: tuple[DirectedEdge, ...]
    undirected_edges: tuple[UndirectedEdge, ...]
    reasoning_summary: str = ""


@dataclass(frozen=True, slots=True)
class CorrelationAction:
    i: int
    j: int


@dataclass(frozen=True, slots=True)
class PartialCorrelationAction:
    i: int
    j: int
    conditioning_on: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class IndependenceTestAction:
    i: int
    j: int
    conditioning_on: tuple[int, ...]
    alpha: float = 0.05


AgentAction = (
    InterveneAction
    | SubmitGraphAction
    | CorrelationAction
    | PartialCorrelationAction
    | IndependenceTestAction
)


@dataclass(frozen=True, slots=True)
class ToolResult:
    tool: str
    payload: dict
