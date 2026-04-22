"""Single-entry ladder runner for causal discovery benchmark evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_discovery import (  # noqa: E402
    BenchmarkEnv,
    CorrelationAction,
    GraphSubmission,
    IndependenceTestAction,
    InterveneAction,
    LLMRawAgent,
    LLMStatsAgent,
    PartialCorrelationAction,
    SubmitGraphAction,
    ToolResult,
    build_benchmark_instance,
    make_v1_config,
    parse_causallearn_endpoint_matrix,
    score_submission,
)
from causal_discovery.agents.stats_tools import (  # noqa: E402
    correlation,
    independence_test,
    partial_correlation,
)


SHIFT_THRESHOLD = 0.5
INTERVENTION_OFFSET = 3.0

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


@dataclass(frozen=True, slots=True)
class LevelSpec:
    level_id: int
    d: int
    k: int
    n_obs: int
    n_int: int
    noise_var: float
    budget_slack: int


@dataclass(frozen=True, slots=True)
class WorkItem:
    panel: str
    method: str
    level_id: int
    seed: int
    model: str

    @property
    def key(self) -> str:
        return (
            f"panel={self.panel}|method={self.method}|level={self.level_id}|"
            f"seed={self.seed}|model={self.model}"
        )


class OpenAIJSONPolicyModel:
    """Strict single-object JSON model adapter for ladder runs."""

    def __init__(self, client: OpenAI, model: str, allowed_actions: frozenset[str]) -> None:
        self._client = client
        self._model = model
        self._action_schema = make_action_response_schema(allowed_actions)
        self._action_tool = make_action_tool(allowed_actions)
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.calls = 0

    def complete(
        self,
        *,
        system_prompt: str,
        session_prompt: str,
        tool_history: tuple[ToolResult, ...],
        remaining_budget: int,
    ) -> str:
        self.calls += 1
        history_payload = [{"tool": item.tool, "payload": item.payload} for item in tool_history]
        user_msg = (
            f"Session data JSON: {session_prompt}\n"
            f"Remaining budget: {remaining_budget}\n"
            "Tool history JSON: "
            f"{json.dumps(history_payload, separators=(',', ':'), ensure_ascii=True, allow_nan=False)}\n"
            "Call the causal_discovery_action tool exactly once for your next action."
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            tools=[self._action_tool],
            tool_choice={
                "type": "function",
                "function": {"name": self._action_schema["name"]},
            },
            parallel_tool_calls=False,
            temperature=0,
        )
        if response.usage is not None:
            self.prompt_tokens += int(response.usage.prompt_tokens or 0)
            self.completion_tokens += int(response.usage.completion_tokens or 0)
            self.total_tokens += int(response.usage.total_tokens or 0)
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        if len(tool_calls) != 1:
            content = message.content or ""
            snippet = content[:240].replace("\n", "\\n")
            raise ValueError(
                "Model must return exactly one causal_discovery_action tool call; "
                f"got {len(tool_calls)}. content_prefix={snippet!r}"
            )
        tool_call = tool_calls[0]
        if tool_call.function.name != self._action_schema["name"]:
            raise ValueError(
                "Unexpected tool call "
                f"{tool_call.function.name!r}; expected {self._action_schema['name']!r}"
            )
        content = tool_call.function.arguments
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            snippet = content[:240].replace("\n", "\\n")
            raise ValueError(
                "Model output JSON parse failed "
                f"({type(exc).__name__}: {exc}). content_prefix={snippet!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError("Model output must be a JSON object")
        return json.dumps(parsed, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


class TraceWriter:
    def __init__(self, path: Path) -> None:
        self._fh = path.open("a", encoding="utf-8")

    def log(self, event_type: str, key: str, payload: dict[str, Any]) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "key": key,
            "payload": payload,
        }
        self._fh.write(
            json.dumps(record, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"
        )
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def ladder_levels() -> dict[int, LevelSpec]:
    return {
        0: LevelSpec(0, d=4, k=5, n_obs=50, n_int=25, noise_var=0.5, budget_slack=2),
        1: LevelSpec(1, d=5, k=6, n_obs=25, n_int=15, noise_var=1.0, budget_slack=1),
        2: LevelSpec(2, d=5, k=6, n_obs=15, n_int=10, noise_var=1.5, budget_slack=1),
        3: LevelSpec(3, d=7, k=9, n_obs=25, n_int=15, noise_var=1.0, budget_slack=1),
        4: LevelSpec(4, d=5, k=6, n_obs=25, n_int=15, noise_var=1.0, budget_slack=0),
        5: LevelSpec(5, d=7, k=9, n_obs=15, n_int=10, noise_var=1.5, budget_slack=0),
    }


def parse_levels(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("No levels selected")
    return sorted(set(values))


def parse_models(text: str) -> list[str]:
    models = [x.strip() for x in text.split(",") if x.strip()]
    if not models:
        raise ValueError("No models provided")
    return models


def build_seed_map_from_file(path: Path, levels: list[int], seeds_per_level: int) -> dict[int, list[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, list[int]] = {}
    for level_id in levels:
        raw = data.get(str(level_id), data.get(level_id))
        if not isinstance(raw, list):
            raise ValueError(f"Seed map missing list for level {level_id}")
        seeds = [int(x) for x in raw]
        if len(seeds) < seeds_per_level:
            raise ValueError(
                f"Level {level_id} seed list too short: required {seeds_per_level}, got {len(seeds)}"
            )
        out[level_id] = seeds[:seeds_per_level]
    return out


def config_from_level(level: LevelSpec):
    return make_v1_config(
        d=level.d,
        k=level.k,
        n_obs=level.n_obs,
        n_int=level.n_int,
        noise_var=level.noise_var,
        budget_slack=level.budget_slack,
    )


def _seed_accepts(level: LevelSpec, seed: int) -> bool:
    cfg = config_from_level(level)
    try:
        build_benchmark_instance(cfg, np.random.default_rng(seed))
    except RuntimeError:
        return False
    return True


def build_seed_map_preflight(
    level_specs: dict[int, LevelSpec],
    levels: list[int],
    seeds_per_level: int,
    preflight_seed: int,
    max_candidates_per_level: int,
) -> dict[int, list[int]]:
    rng = np.random.default_rng(preflight_seed)
    out: dict[int, list[int]] = {}
    for level_id in levels:
        level = level_specs[level_id]
        accepted: list[int] = []
        seen: set[int] = set()
        attempts = 0
        while len(accepted) < seeds_per_level:
            if attempts >= max_candidates_per_level:
                raise RuntimeError(
                    f"Unable to find {seeds_per_level} accepted seeds for level {level_id} "
                    f"within {max_candidates_per_level} candidates"
                )
            attempts += 1
            candidate = int(rng.integers(1, 2_147_483_647))
            if candidate in seen:
                continue
            seen.add(candidate)
            if _seed_accepts(level, candidate):
                accepted.append(candidate)
        out[level_id] = accepted
    return out


def make_work_items(levels: list[int], seed_map: dict[int, list[int]], models: list[str]) -> list[WorkItem]:
    items: list[WorkItem] = []
    for level_id in levels:
        for seed in seed_map[level_id]:
            items.append(WorkItem("observational", "pc", level_id, seed, "baseline"))
            for model in models:
                items.append(WorkItem("observational", "llm_raw_obs", level_id, seed, model))
                items.append(WorkItem("observational", "llm_stats_obs", level_id, seed, model))
            items.append(WorkItem("active", "pc_greedy", level_id, seed, "baseline"))
            for model in models:
                items.append(WorkItem("active", "llm_raw", level_id, seed, model))
                items.append(WorkItem("active", "llm_stats", level_id, seed, model))
            items.append(WorkItem("active", "oracle", level_id, seed, "baseline"))
    return items


def would_create_cycle(directed: set[tuple[int, int]], src: int, dst: int) -> bool:
    adj: dict[int, list[int]] = {}
    for a, b in directed:
        adj.setdefault(a, []).append(b)
    stack = [dst]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if node == src:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adj.get(node, []))
    return False


def _sanitize_submission_edges(
    directed_edges: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    undirected_edges: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]], int]:
    directed = frozenset((int(src), int(dst)) for src, dst in directed_edges)
    directed_pairs = {
        (min(int(src), int(dst)), max(int(src), int(dst))) for src, dst in directed
    }
    undirected: set[tuple[int, int]] = set()
    dropped = 0
    for a, b in undirected_edges:
        edge = (min(int(a), int(b)), max(int(a), int(b)))
        if edge in directed_pairs:
            dropped += 1
            continue
        undirected.add(edge)
    return directed, frozenset(undirected), dropped


def resolve_with_interventions(
    env: BenchmarkEnv,
    directed: set[tuple[int, int]],
    undirected: set[tuple[int, int]],
    mu_obs: np.ndarray,
) -> int:
    used = 0
    while undirected and env.remaining_budget > 0:
        degree: dict[int, int] = {}
        for edge in undirected:
            for node in edge:
                degree[node] = degree.get(node, 0) + 1
        target = min(degree.keys(), key=lambda node: (-degree[node], node))
        value = float(mu_obs[target] + INTERVENTION_OFFSET)
        samples = env.intervene(var=target, value=value)
        used += 1
        mu_int = samples.mean(axis=0)
        for edge in [edge for edge in undirected if target in edge]:
            other = edge[1] if edge[0] == target else edge[0]
            shifted = abs(mu_int[other] - mu_obs[other]) > SHIFT_THRESHOLD
            candidate = (target, other) if shifted else (other, target)
            if would_create_cycle(directed, *candidate):
                continue
            directed.add(candidate)
            undirected.discard(edge)
    return used


def default_row(
    run_id: str,
    item: WorkItem,
    level: LevelSpec,
    seed: int,
    runtime_seed: int,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "panel": item.panel,
        "method": item.method,
        "model": item.model,
        "level": level.level_id,
        "seed": seed,
        "runtime_seed": runtime_seed,
        "d": level.d,
        "k": level.k,
        "n_obs": level.n_obs,
        "n_int": level.n_int,
        "noise_var": level.noise_var,
        "budget_slack": level.budget_slack,
        "status": "",
        "error": "",
        "latency_sec": "",
        "prompt_tokens": "",
        "completion_tokens": "",
        "total_tokens": "",
        "true_edges": "",
        "cpdag_directed": "",
        "cpdag_undirected": "",
        "opt_set_size": "",
        "budget": "",
        "interventions_used": "",
        "total_actions": "",
        "stats_actions": "",
        "intervene_actions": "",
        "submit_actions": "",
        "sanitized_overlap_count": "",
        "submit_directed": "",
        "submit_undirected": "",
        "skeleton_precision": "",
        "skeleton_recall": "",
        "skeleton_f1": "",
        "compelled_precision": "",
        "compelled_recall": "",
        "compelled_f1": "",
        "directed_precision": "",
        "directed_recall": "",
        "directed_f1": "",
        "dag_shd": "",
        "efficiency": "",
    }


def enrich_instance_fields(row: dict[str, Any], instance) -> None:
    row["true_edges"] = len(instance.true_dag.edges)
    row["cpdag_directed"] = len(instance.observational_ceiling.directed_edges)
    row["cpdag_undirected"] = len(instance.observational_ceiling.undirected_edges)
    row["opt_set_size"] = len(instance.optimal_intervention_set)
    row["budget"] = instance.intervention_budget


def enrich_score_fields(row: dict[str, Any], scores) -> None:
    row["skeleton_precision"] = scores.skeleton_precision
    row["skeleton_recall"] = scores.skeleton_recall
    row["skeleton_f1"] = scores.skeleton_f1
    row["compelled_precision"] = scores.compelled_precision
    row["compelled_recall"] = scores.compelled_recall
    row["compelled_f1"] = scores.compelled_f1
    row["directed_precision"] = scores.directed_precision
    row["directed_recall"] = scores.directed_recall
    row["directed_f1"] = scores.directed_f1
    row["dag_shd"] = scores.dag_shd
    row["efficiency"] = scores.efficiency


def run_pc_observational(instance, runtime_seed: int, alpha: float):
    from causallearn.search.ConstraintBased.PC import pc

    _ = runtime_seed
    env = BenchmarkEnv(instance, np.random.default_rng(runtime_seed))
    data = env.observe()
    cg = pc(data, alpha=alpha, indep_test="fisherz", show_progress=False, verbose=False)
    directed, undirected = parse_causallearn_endpoint_matrix(cg.G.graph)
    submission = GraphSubmission(
        num_nodes=instance.config.d,
        directed_edges=directed,
        undirected_edges=undirected,
    )
    output = env.submit_graph(submission)
    scores = score_submission(instance, output.submission)
    return output.submission, scores


def run_pc_greedy_active(instance, runtime_seed: int, alpha: float):
    from causallearn.search.ConstraintBased.PC import pc

    env = BenchmarkEnv(instance, np.random.default_rng(runtime_seed))
    data = env.observe()
    mu_obs = data.mean(axis=0)
    cg = pc(data, alpha=alpha, indep_test="fisherz", show_progress=False, verbose=False)
    directed_frozen, undirected_frozen = parse_causallearn_endpoint_matrix(cg.G.graph)
    directed = set(directed_frozen)
    undirected = set(undirected_frozen)
    _ = resolve_with_interventions(env, directed, undirected, mu_obs)
    submission = GraphSubmission(
        num_nodes=instance.config.d,
        directed_edges=frozenset(directed),
        undirected_edges=frozenset(undirected),
    )
    output = env.submit_graph(submission)
    scores = score_submission(instance, output.submission)
    return output.submission, scores


def run_oracle(instance, runtime_seed: int):
    env = BenchmarkEnv(instance, np.random.default_rng(runtime_seed))
    _ = env.observe()
    submission = GraphSubmission.from_dag(instance.true_dag)
    output = env.submit_graph(submission)
    scores = score_submission(instance, output.submission)
    return output.submission, scores


def run_llm(
    *,
    instance,
    runtime_seed: int,
    model_id: str,
    client: OpenAI,
    method: str,
    max_steps_raw: int,
    max_steps_stats: int,
    trace: TraceWriter,
    work_key: str,
) -> tuple[GraphSubmission, Any, OpenAIJSONPolicyModel, dict[str, int]]:
    env = BenchmarkEnv(instance, np.random.default_rng(runtime_seed))
    obs = env.observe()
    latest_intervention_data: np.ndarray | None = None

    allow_interventions = method in {"llm_raw", "llm_stats"}
    is_stats = method in {"llm_stats", "llm_stats_obs"}
    max_steps = max_steps_stats if is_stats else max_steps_raw
    logical_budget = env.remaining_budget if allow_interventions else 0
    total_actions = 0
    stats_actions = 0
    intervene_actions = 0
    submit_actions = 0
    sanitized_overlap_count = 0

    allowed_actions = allowed_actions_for_method(method)
    model = OpenAIJSONPolicyModel(
        client=client,
        model=model_id,
        allowed_actions=allowed_actions,
    )
    names = tuple(f"X{i}" for i in range(instance.config.d))
    if is_stats:
        agent = LLMStatsAgent(
            model=model,
            variable_names=names,
            allow_interventions=allow_interventions,
        )
    else:
        agent = LLMRawAgent(
            model=model,
            variable_names=names,
            allow_interventions=allow_interventions,
        )

    if agent.allowed_actions != allowed_actions:
        raise RuntimeError(
            f"Agent/tool action mismatch for {method}: "
            f"agent={sorted(agent.allowed_actions)} tool={sorted(allowed_actions)}"
        )

    agent.on_observation(obs, logical_budget)
    trace.log(
        "llm_observation",
        work_key,
        {"obs_shape": [int(obs.shape[0]), int(obs.shape[1])], "logical_budget": logical_budget},
    )

    for step in range(1, max_steps + 1):
        action = agent.next_action(logical_budget if not allow_interventions else env.remaining_budget)
        total_actions += 1
        trace.log("llm_action", work_key, {"step": step, "action_type": type(action).__name__})

        if isinstance(action, InterveneAction):
            intervene_actions += 1
            if not allow_interventions:
                raise RuntimeError("Protocol violation: intervention requested in observational mode")
            if env.remaining_budget <= 0:
                raise RuntimeError("Protocol violation: intervention requested with exhausted budget")
            samples = env.intervene(var=action.var, value=action.value)
            latest_intervention_data = samples
            logical_budget = env.remaining_budget
            agent.on_intervention_result(
                var=action.var,
                value=action.value,
                data=samples,
                remaining_budget=logical_budget,
            )
            trace.log(
                "llm_intervention_result",
                work_key,
                {
                    "step": step,
                    "var": action.var,
                    "value": action.value,
                    "rows": int(samples.shape[0]),
                    "remaining_budget": logical_budget,
                },
            )
            continue

        if isinstance(action, CorrelationAction):
            stats_actions += 1
            data_source = "latest_intervention" if latest_intervention_data is not None else "observational"
            data = latest_intervention_data if latest_intervention_data is not None else obs
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
            trace.log("llm_tool_result", work_key, {"step": step, "tool": "correlation", **payload})
            continue

        if isinstance(action, PartialCorrelationAction):
            stats_actions += 1
            data_source = "latest_intervention" if latest_intervention_data is not None else "observational"
            data = latest_intervention_data if latest_intervention_data is not None else obs
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
            trace.log("llm_tool_result", work_key, {"step": step, "tool": "partial_correlation", **payload})
            continue

        if isinstance(action, IndependenceTestAction):
            stats_actions += 1
            data_source = "latest_intervention" if latest_intervention_data is not None else "observational"
            data = latest_intervention_data if latest_intervention_data is not None else obs
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
            trace.log("llm_tool_result", work_key, {"step": step, "tool": "independence_test", **payload})
            continue

        if isinstance(action, SubmitGraphAction):
            submit_actions += 1
            directed_edges, undirected_edges, dropped = _sanitize_submission_edges(
                action.directed_edges,
                action.undirected_edges,
            )
            sanitized_overlap_count += dropped
            if dropped > 0:
                trace.log(
                    "llm_submit_sanitized",
                    work_key,
                    {"step": step, "dropped_undirected_edges": dropped},
                )
            submission = GraphSubmission(
                num_nodes=instance.config.d,
                directed_edges=directed_edges,
                undirected_edges=undirected_edges,
            )
            output = env.submit_graph(submission)
            scores = score_submission(instance, output.submission)
            diagnostics = {
                "total_actions": total_actions,
                "stats_actions": stats_actions,
                "intervene_actions": intervene_actions,
                "submit_actions": submit_actions,
                "sanitized_overlap_count": sanitized_overlap_count,
            }
            return output.submission, scores, model, diagnostics

        raise RuntimeError(f"Unsupported action type: {type(action).__name__}")

    raise RuntimeError(f"LLM session exceeded max_steps={max_steps} without submit_graph")


def now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


def append_row(path: Path, row: dict[str, Any], header: list[str]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_checkpoint(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    completed = payload.get("completed", {})
    out: dict[str, dict[str, str]] = {}
    for key, value in completed.items():
        if isinstance(value, dict):
            out[key] = {"status": str(value.get("status", ""))}
    return out


def load_success_rows(long_csv: Path) -> list[dict[str, Any]]:
    if not long_csv.exists():
        return []
    rows: list[dict[str, Any]] = []
    with long_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "success":
                rows.append(row)
    return rows


def load_rows(long_csv: Path) -> list[dict[str, Any]]:
    if not long_csv.exists():
        return []
    rows: list[dict[str, Any]] = []
    with long_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _float_field(row: dict[str, Any], name: str) -> float | None:
    raw = row.get(name, "")
    if raw in ("", None):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def summarize_results(rows: list[dict[str, Any]], out_csv: Path) -> None:
    metrics = ("skeleton_f1", "directed_f1", "dag_shd", "efficiency")
    behavior_metrics = (
        "submit_directed",
        "submit_undirected",
        "interventions_used",
        "total_actions",
        "stats_actions",
        "intervene_actions",
        "submit_actions",
        "sanitized_overlap_count",
    )
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["panel"], row["level"], row["method"], row["model"])
        grouped.setdefault(key, []).append(row)

    header = [
        "panel",
        "level",
        "method",
        "model",
        "n_total",
        "n_success",
        "n_failed",
    ]
    for metric in metrics:
        header.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_ci95_low", f"{metric}_ci95_high"])
    for metric in behavior_metrics:
        header.append(f"{metric}_mean")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for (panel, level, method, model), group in sorted(grouped.items()):
            success_group = [r for r in group if r.get("status") == "success"]
            out: dict[str, Any] = {
                "panel": panel,
                "level": level,
                "method": method,
                "model": model,
                "n_total": len(group),
                "n_success": len(success_group),
                "n_failed": len(group) - len(success_group),
            }
            for metric in metrics:
                values = [v for v in (_float_field(r, metric) for r in success_group) if v is not None]
                if not values:
                    out[f"{metric}_mean"] = ""
                    out[f"{metric}_std"] = ""
                    out[f"{metric}_ci95_low"] = ""
                    out[f"{metric}_ci95_high"] = ""
                    continue
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                ci = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
                out[f"{metric}_mean"] = mean
                out[f"{metric}_std"] = std
                out[f"{metric}_ci95_low"] = mean - ci
                out[f"{metric}_ci95_high"] = mean + ci
            for metric in behavior_metrics:
                values = [v for v in (_float_field(r, metric) for r in success_group) if v is not None]
                out[f"{metric}_mean"] = float(np.mean(values)) if values else ""
            writer.writerow(out)


def print_summary_table(summary_csv: Path, panel: str) -> None:
    if not summary_csv.exists():
        return
    rows: list[dict[str, str]] = []
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("panel") == panel:
                rows.append(row)
    if not rows:
        return
    print(f"\n[{panel.upper()} RESULTS]")
    print(
        f"{'level':>5} {'method':>18} {'model':>12} {'n':>4} "
        f"{'skel_f1':>9} {'dir_f1':>9} {'shd':>9} {'eff':>9}"
    )
    print("-" * 80)
    for row in rows:
        print(
            f"{row['level']:>5} {row['method']:>18} {row['model']:>12} {row['n_success']:>4} "
            f"{float(row['skeleton_f1_mean'] or 0.0):>9.3f} "
            f"{float(row['directed_f1_mean'] or 0.0):>9.3f} "
            f"{float(row['dag_shd_mean'] or 0.0):>9.3f} "
            f"{float(row['efficiency_mean'] or 0.0):>9.3f}"
        )


def parse_args() -> argparse.Namespace:
    default_env = Path(__file__).resolve().parent.parent / ".env"
    parser = argparse.ArgumentParser(description="Run full causal-discovery ladder.")
    parser.add_argument("--levels", default="0,1,2,3,4,5")
    parser.add_argument("--seeds-per-level", type=int, default=8)
    parser.add_argument("--models", default="gpt-5.4")
    parser.add_argument("--env-file", default=str(default_env))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-steps-raw", type=int, default=16)
    parser.add_argument("--max-steps-stats", type=int, default=32)
    parser.add_argument("--seed-map-file", default="")
    parser.add_argument("--preflight-seed", type=int, default=20260422)
    parser.add_argument("--max-candidates-per-level", type=int, default=50000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    levels_catalog = ladder_levels()
    levels = parse_levels(args.levels)
    for level in levels:
        if level not in levels_catalog:
            raise ValueError(f"Unknown level id: {level}")
    models = parse_models(args.models)

    run_id = now_run_id()
    out_dir = Path(args.out_dir) if args.out_dir else (Path("traces") / "ladder" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "run_manifest.json"
    checkpoint_path = out_dir / "checkpoint.json"
    events_path = out_dir / "events.jsonl"
    long_csv = out_dir / "results_long.csv"
    summary_csv = out_dir / "results_summary.csv"

    if args.resume and manifest_path.exists():
        manifest_existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = str(manifest_existing["run_id"])
        seed_map_raw = manifest_existing["seed_map"]
        seed_map = {int(k): [int(x) for x in v] for k, v in seed_map_raw.items()}
        models = [str(x) for x in manifest_existing.get("models", models)]
        levels = [int(x) for x in manifest_existing.get("levels", levels)]
    else:
        run_id = now_run_id()
        if args.seed_map_file:
            seed_map = build_seed_map_from_file(
                Path(args.seed_map_file), levels, args.seeds_per_level
            )
        else:
            seed_map = build_seed_map_preflight(
                levels_catalog,
                levels,
                args.seeds_per_level,
                args.preflight_seed,
                args.max_candidates_per_level,
            )

        manifest = {
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "levels": levels,
            "models": models,
            "seeds_per_level": args.seeds_per_level,
            "seed_map": {str(k): v for k, v in seed_map.items()},
            "alpha": args.alpha,
            "max_steps_raw": args.max_steps_raw,
            "max_steps_stats": args.max_steps_stats,
        }
        write_json_atomic(manifest_path, manifest)

    completed = load_checkpoint(checkpoint_path) if args.resume else {}
    trace = TraceWriter(events_path)
    header = list(
        default_row(
            run_id=run_id,
            item=WorkItem("observational", "pc", 0, 0, "baseline"),
            level=levels_catalog[0],
            seed=0,
            runtime_seed=0,
        ).keys()
    )

    env_file = Path(args.env_file)
    load_dotenv(env_file)
    api_key = os.getenv("OPENAI_API_KEY", "")
    client = OpenAI(api_key=api_key) if api_key else None

    work_items = make_work_items(levels, seed_map, models)
    if client is None and any(item.method.startswith("llm_") for item in work_items):
        raise RuntimeError(
            "OPENAI_API_KEY missing; required for ladder run because LLM methods are enabled"
        )
    total = len(work_items)
    done = 0
    instance_cache: dict[tuple[int, int], Any] = {}

    for item in work_items:
        done += 1
        print(f"[{done}/{total}] {item.key}")
        if item.key in completed:
            prev_status = completed[item.key].get("status", "")
            if prev_status == "success":
                trace.log("skip_completed", item.key, {"status": prev_status})
                continue
            if prev_status == "failed" and not args.retry_failed:
                trace.log("skip_failed", item.key, {"status": prev_status})
                continue

        level = levels_catalog[item.level_id]
        runtime_seed = int(item.seed * 10_000 + item.level_id * 101 + 7)
        row = default_row(run_id, item, level, item.seed, runtime_seed)
        started = time.perf_counter()

        try:
            cache_key = (item.level_id, item.seed)
            if cache_key not in instance_cache:
                cfg = config_from_level(level)
                instance_cache[cache_key] = build_benchmark_instance(
                    cfg, np.random.default_rng(item.seed)
                )
            instance = instance_cache[cache_key]
            enrich_instance_fields(row, instance)

            if item.method == "pc":
                submission, scores = run_pc_observational(instance, runtime_seed, args.alpha)
                row["prompt_tokens"] = 0
                row["completion_tokens"] = 0
                row["total_tokens"] = 0
                row["total_actions"] = 0
                row["stats_actions"] = 0
                row["intervene_actions"] = 0
                row["submit_actions"] = 0
                row["sanitized_overlap_count"] = 0
            elif item.method == "pc_greedy":
                submission, scores = run_pc_greedy_active(instance, runtime_seed, args.alpha)
                row["prompt_tokens"] = 0
                row["completion_tokens"] = 0
                row["total_tokens"] = 0
                row["total_actions"] = 0
                row["stats_actions"] = 0
                row["intervene_actions"] = 0
                row["submit_actions"] = 0
                row["sanitized_overlap_count"] = 0
            elif item.method == "oracle":
                submission, scores = run_oracle(instance, runtime_seed)
                row["prompt_tokens"] = 0
                row["completion_tokens"] = 0
                row["total_tokens"] = 0
                row["total_actions"] = 0
                row["stats_actions"] = 0
                row["intervene_actions"] = 0
                row["submit_actions"] = 0
                row["sanitized_overlap_count"] = 0
            elif item.method in {"llm_raw", "llm_stats", "llm_raw_obs", "llm_stats_obs"}:
                if client is None:
                    raise RuntimeError(
                        "OPENAI_API_KEY missing; required for LLM methods"
                    )
                submission, scores, model_usage, diagnostics = run_llm(
                    instance=instance,
                    runtime_seed=runtime_seed,
                    model_id=item.model,
                    client=client,
                    method=item.method,
                    max_steps_raw=args.max_steps_raw,
                    max_steps_stats=args.max_steps_stats,
                    trace=trace,
                    work_key=item.key,
                )
                row["prompt_tokens"] = model_usage.prompt_tokens
                row["completion_tokens"] = model_usage.completion_tokens
                row["total_tokens"] = model_usage.total_tokens
                row["total_actions"] = diagnostics["total_actions"]
                row["stats_actions"] = diagnostics["stats_actions"]
                row["intervene_actions"] = diagnostics["intervene_actions"]
                row["submit_actions"] = diagnostics["submit_actions"]
                row["sanitized_overlap_count"] = diagnostics["sanitized_overlap_count"]
            else:
                raise RuntimeError(f"Unknown method: {item.method}")

            row["interventions_used"] = submission.interventions_used
            row["submit_directed"] = len(submission.directed_edges)
            row["submit_undirected"] = len(submission.undirected_edges)
            enrich_score_fields(row, scores)
            row["status"] = "success"
            row["error"] = ""

            trace.log(
                "work_success",
                item.key,
                {
                    "interventions_used": submission.interventions_used,
                    "directed_f1": row["directed_f1"],
                    "dag_shd": row["dag_shd"],
                },
            )
            completed[item.key] = {"status": "success"}
        except Exception as exc:  # noqa: BLE001
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            trace.log("work_failed", item.key, {"error": row["error"]})
            completed[item.key] = {"status": "failed"}
        finally:
            row["latency_sec"] = round(time.perf_counter() - started, 6)
            append_row(long_csv, row, header)
            write_json_atomic(
                checkpoint_path,
                {
                    "run_id": run_id,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "completed": completed,
                },
            )

    trace.close()

    all_rows = load_rows(long_csv)
    summarize_results(all_rows, summary_csv)
    print_summary_table(summary_csv, panel="observational")
    print_summary_table(summary_csv, panel="active")
    print(f"\n[done] long={long_csv}")
    print(f"[done] summary={summary_csv}")
    print(f"[done] manifest={manifest_path}")
    print(f"[done] checkpoint={checkpoint_path}")
    print(f"[done] events={events_path}")


if __name__ == "__main__":
    main()
