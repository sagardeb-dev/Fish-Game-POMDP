"""Stats agent + PC-algorithm guidebook: methodology-knowledge probe.

This ablation isolates the role of *procedural knowledge* in LLM causal
discovery. The LLM gets the same data and the same statistical tools as
`llm_stats` (raw observational rows, plus `correlation`,
`partial_correlation`, `independence_test`, `intervene`, `submit_graph`).

The only difference from `llm_stats` is the system prompt: it contains an
explicit walkthrough of the PC algorithm followed by the greedy active-phase
strategy that `pc_greedy` uses.

If `llm_stats_guided` >> `llm_stats` then the LLM gap is largely *procedural
knowledge*: current LLMs have the statistics chops but don't know which
algorithm to run with the tools. If they're indistinguishable then the gap
is execution, not knowledge.

Non-goals / anti-leakage:
- Does not pre-compute PC results for the LLM (that's `llm_pc_handoff`).
- Does not give the LLM the true CPDAG.
- Does not modify the underlying tool implementations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_discovery import (  # noqa: E402
    BenchmarkEnv,
    GraphSubmission,
    LiteLLMJSONPolicyModel,
    ToolResult,
    build_benchmark_instance,
    provider_for_model,
    score_submission,
)
from causal_discovery.agents.stats_tools import (  # noqa: E402
    correlation,
    independence_test,
    partial_correlation,
)
from run_ladder import (  # noqa: E402
    config_from_level,
    enrich_instance_fields,
    enrich_score_fields,
    ladder_levels,
    parse_levels,
    parse_models,
    would_create_cycle,
)


METHOD = "llm_stats_guided"
PANEL = "active"
ALLOWED_ACTIONS = frozenset(
    {
        "correlation",
        "partial_correlation",
        "independence_test",
        "intervene",
        "submit_graph",
    }
)

LONG_FIELDS = (
    "run_id",
    "timestamp_utc",
    "panel",
    "method",
    "model",
    "level",
    "seed",
    "runtime_seed",
    "d",
    "k",
    "n_obs",
    "n_int",
    "noise_var",
    "budget_slack",
    "status",
    "error",
    "latency_sec",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "true_edges",
    "cpdag_directed",
    "cpdag_undirected",
    "opt_set_size",
    "budget",
    "interventions_used",
    "total_actions",
    "stats_actions",
    "intervene_actions",
    "submit_actions",
    "sanitized_overlap_count",
    "sanitized_invalid_count",
    "sanitized_cycle_count",
    "submit_directed",
    "submit_undirected",
    "skeleton_precision",
    "skeleton_recall",
    "skeleton_f1",
    "compelled_precision",
    "compelled_recall",
    "compelled_f1",
    "directed_precision",
    "directed_recall",
    "directed_f1",
    "dag_shd",
    "efficiency",
)

SUMMARY_METRICS = (
    "skeleton_precision",
    "skeleton_recall",
    "skeleton_f1",
    "compelled_precision",
    "compelled_recall",
    "compelled_f1",
    "directed_precision",
    "directed_recall",
    "directed_f1",
    "dag_shd",
    "efficiency",
)

BEHAVIOR_METRICS = (
    "submit_directed",
    "submit_undirected",
    "interventions_used",
    "total_actions",
    "stats_actions",
    "intervene_actions",
    "submit_actions",
    "sanitized_overlap_count",
    "sanitized_invalid_count",
    "sanitized_cycle_count",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)


class EventWriter:
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
            json.dumps(record, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
            + "\n"
        )
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def append_row(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LONG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_manifest_seed_map(
    manifest_path: Path, levels: list[int], seeds_per_level: int
) -> dict[int, list[int]]:
    if seeds_per_level <= 0:
        raise ValueError("--seeds-per-level must be positive")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_seed_map = data.get("seed_map")
    if not isinstance(raw_seed_map, dict):
        raise ValueError(f"Manifest missing seed_map: {manifest_path}")

    out: dict[int, list[int]] = {}
    for level_id in levels:
        raw = raw_seed_map.get(str(level_id), raw_seed_map.get(level_id))
        if not isinstance(raw, list):
            raise ValueError(f"Manifest seed_map missing list for level {level_id}")
        seeds = [int(seed) for seed in raw]
        if len(seeds) < seeds_per_level:
            raise ValueError(
                f"Manifest level {level_id} has {len(seeds)} seeds; "
                f"required {seeds_per_level}"
            )
        out[level_id] = seeds[:seeds_per_level]
    return out


def build_stats_guided_system_prompt() -> str:
    return (
        "You are performing ACTIVE causal discovery over an unknown linear-Gaussian "
        "DAG with full observability and no hidden confounders.\n"
        "An intervention forces one variable to a fixed value, breaking its "
        "incoming dependencies. By comparing pre- and post-intervention means you "
        "can determine causal direction. Observations alone bound you to the CPDAG; "
        "only interventions can resolve undirected edges in the equivalence class.\n"
        "\n"
        "REQUIRED WORKFLOW. Both phases must happen, in order, before you submit. "
        "Sessions that submit without independence_test calls are protocol "
        "failures. Sessions that submit without intervene calls are protocol "
        "failures. Do not collapse the workflow.\n"
        "  Step A: build a skeleton from observational data via independence_test "
        "calls (Phase 1). At a minimum, test each variable pair (i, j) for "
        "independence with empty conditioning set, then test conditioning on each "
        "single neighbor. Without this, you cannot know which adjacencies exist.\n"
        "  Step B: while remaining_budget > 0 AND undirected edges remain, intervene "
        "(Phase 4). Without this, undirected edges in the CPDAG stay undirected.\n"
        "Only submit after both Step A and Step B have been performed.\n"
        "\n"
        "OPTIMAL CLASSICAL METHOD (PC + greedy interventions). Follow this "
        "procedure; it is the algorithm that solves this benchmark. All four "
        "phases are required; do not stop early.\n"
        "\n"
        "PHASE 1 — SKELETON CONSTRUCTION (use independence_test).\n"
        "  Initialize: every pair {i, j} is potentially adjacent.\n"
        "  For conditioning-set size k = 0, 1, 2, ... up to d - 2:\n"
        "    For each remaining adjacent pair (i, j):\n"
        "      For each subset S of size k drawn from variables currently adjacent to "
        "i or j (excluding j and i respectively):\n"
        "        Call independence_test(i, j, conditioning_on=S, alpha=0.05).\n"
        "        If independent=true, REMOVE the edge from the skeleton and "
        "RECORD S as the separating set Sepset(i, j).\n"
        "  The remaining adjacencies form the skeleton. THIS SKELETON IS YOUR "
        "ADJACENCY COMMITMENT — do not add edges in later phases that aren't in it.\n"
        "\n"
        "PHASE 2 — V-STRUCTURE ORIENTATION.\n"
        "  For each unshielded triple (a — c — b) where a and b are not adjacent in "
        "the skeleton:\n"
        "    If c is NOT in Sepset(a, b), orient a -> c <- b (this is an unshielded "
        "collider / v-structure).\n"
        "  Most edges WILL REMAIN UNDIRECTED after Phase 2. That is correct and "
        "expected. Do not orient edges from any other observational signal here.\n"
        "\n"
        "PHASE 3 — MEEK CLOSURE (apply repeatedly until no edges change).\n"
        "  R1: If a -> b and b — c with a not adjacent to c, orient b -> c.\n"
        "  R2: If a -> b -> c and a — c, orient a -> c.\n"
        "  R3: If a — b, a — c, a — d, c -> b, d -> b, and c not adjacent to d, "
        "orient a -> b.\n"
        "  After Phases 1-3 you should have a CPDAG: some compelled directed edges "
        "and a non-empty set of undirected edges. Do NOT orient remaining "
        "undirected edges from observational data alone — observational data "
        "cannot identify directions within a Markov-equivalence class.\n"
        "\n"
        "PHASE 4 — ACTIVE PHASE (use intervene). REQUIRED.\n"
        "  This phase is mandatory. The benchmark expects at least one intervene "
        "call.\n"
        "  While budget > 0 AND undirected edges remain in your current CPDAG:\n"
        "    Pick the undirected node with the MOST undirected edges (break ties by "
        "lowest variable index).\n"
        "    Set value = (observational mean of that node) + 3.0; call intervene.\n"
        "    For each undirected edge (target, other) incident to the intervened "
        "node, compare per-variable post-intervention means to observational means:\n"
        "      If |mu_int[other] - mu_obs[other]| > 0.5, orient target -> other.\n"
        "      Otherwise orient other -> target.\n"
        "    Skip orientations that would create a cycle.\n"
        "    Re-apply Meek closure (Phase 3) mentally after each intervention.\n"
        "\n"
        "Available tools:\n"
        "- correlation(i, j): Pearson correlation; coarse association check.\n"
        "- partial_correlation(i, j, S): direct dependence after conditioning.\n"
        "- independence_test(i, j, S, alpha): conditional independence decision; "
        "primary tool for skeleton construction in Phase 1.\n"
        "- intervene(var, value): perform an intervention; consumes one budget unit.\n"
        "- submit_graph(directed_edges, undirected_edges, reasoning_summary): "
        "terminal submission.\n"
        "\n"
        "PRE-SUBMIT CHECKLIST (verify all six before calling submit_graph):\n"
        "  1. I have called independence_test enough times to determine which "
        "pairs are adjacent (at minimum, every pair tested with empty "
        "conditioning set, then with single conditioning neighbors).\n"
        "  2. I have called intervene at least once this session (Phase 4 ran).\n"
        "  3. My submitted skeleton (directed ∪ undirected pairs) matches the "
        "Phase 1 skeleton — I have not invented adjacencies.\n"
        "  4. Every directed edge has a justification: it was a v-structure "
        "orientation, a Meek-rule consequence, or a post-intervention shift.\n"
        "  5. For each undirected edge in my CPDAG that I could not resolve via "
        "intervention, it remains undirected in my submission.\n"
        "  6. remaining_budget == 0 OR no undirected edges remain.\n"
        "If any item fails, do NOT submit yet — perform the missing step instead.\n"
        "\n"
        "ERROR MODEL (these errors are scored explicitly, no partial credit):\n"
        "- Wrong directed edge: 1 SHD error (counts as both wrong-direction FP and "
        "missing-direction FN). Submitting unsupported orientations is costly.\n"
        "- Missing skeleton edge: 1 SHD error.\n"
        "- Extra skeleton edge: 1 SHD error.\n"
        "- Unresolved (undirected when truth is directed): 1 SHD error — but this is "
        "ONE error, the same as a reversal. When uncertain, leave undirected.\n"
        "\n"
        "Use the provided action tool exactly once per turn. Do not request "
        "unavailable metadata."
    )


def build_stats_guided_session_prompt(
    *,
    variable_names: tuple[str, ...],
    observational_data: np.ndarray,
    budget: int,
) -> str:
    payload = {
        "variables": list(variable_names),
        "d": len(variable_names),
        "n_obs": int(observational_data.shape[0]),
        "budget": int(budget),
        "observational_data": observational_data.tolist(),
        "output_schema": {
            "action": (
                "correlation|partial_correlation|independence_test|intervene|submit_graph"
            ),
            "var": "int|null; required for intervene, otherwise null",
            "value": "float|null; required for intervene, otherwise null",
            "i": "int|null; required for statistical tools, otherwise null",
            "j": "int|null; required for statistical tools, otherwise null",
            "conditioning_on": "list[int]",
            "alpha": "float|null; required for independence_test, otherwise null",
            "directed_edges": "list[list[int, int]]",
            "undirected_edges": "list[list[int, int]]",
            "reasoning_summary": "string",
        },
        "json_contract": (
            "Return all output_schema fields. Use null for unused scalar fields and "
            "[] for unused list fields."
        ),
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _edge_pair(raw: object) -> tuple[int, int] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        return int(raw[0]), int(raw[1])
    except (TypeError, ValueError):
        return None


def submission_from_model_payload(
    payload: dict[str, Any], d: int, interventions_used: int
) -> tuple[GraphSubmission, dict[str, int]]:
    if str(payload.get("action", "")).strip() != "submit_graph":
        raise ValueError(f"Expected submit_graph action, got {payload.get('action')!r}")

    directed: set[tuple[int, int]] = set()
    directed_pairs: set[tuple[int, int]] = set()
    invalid_count = 0
    cycle_count = 0
    overlap_count = 0

    for raw_edge in payload.get("directed_edges") or []:
        edge = _edge_pair(raw_edge)
        if edge is None:
            invalid_count += 1
            continue
        src, dst = edge
        if src == dst or not (0 <= src < d) or not (0 <= dst < d):
            invalid_count += 1
            continue
        pair = (min(src, dst), max(src, dst))
        if (dst, src) in directed:
            overlap_count += 1
            continue
        if would_create_cycle(directed, src, dst):
            cycle_count += 1
            continue
        directed.add((src, dst))
        directed_pairs.add(pair)

    undirected: set[tuple[int, int]] = set()
    for raw_edge in payload.get("undirected_edges") or []:
        edge = _edge_pair(raw_edge)
        if edge is None:
            invalid_count += 1
            continue
        a, b = edge
        if a == b or not (0 <= a < d) or not (0 <= b < d):
            invalid_count += 1
            continue
        pair = (min(a, b), max(a, b))
        if pair in directed_pairs:
            overlap_count += 1
            continue
        undirected.add(pair)

    submission = GraphSubmission(
        num_nodes=d,
        directed_edges=frozenset(directed),
        undirected_edges=frozenset(undirected),
        interventions_used=int(interventions_used),
    )
    diagnostics = {
        "sanitized_overlap_count": overlap_count,
        "sanitized_invalid_count": invalid_count,
        "sanitized_cycle_count": cycle_count,
    }
    return submission, diagnostics


def make_model(model_id: str) -> LiteLLMJSONPolicyModel:
    return LiteLLMJSONPolicyModel(model=model_id, allowed_actions=ALLOWED_ACTIONS)


def _normalize_conditioning(payload: dict[str, Any]) -> tuple[int, ...]:
    raw = payload.get("conditioning_on") or []
    if not isinstance(raw, list):
        raise ValueError("conditioning_on must be a list of integers")
    return tuple(int(x) for x in raw)


def run_stats_guided_llm(
    *,
    instance: Any,
    runtime_seed: int,
    model_id: str,
    max_steps: int,
    trace: EventWriter | None,
    work_key: str,
) -> tuple[GraphSubmission, Any, Any, dict[str, int]]:
    env = BenchmarkEnv(instance, np.random.default_rng(runtime_seed))
    obs_data = env.observe()

    variable_names = tuple(f"X{i}" for i in range(instance.config.d))
    model = make_model(model_id)
    system_prompt = build_stats_guided_system_prompt()
    session_prompt = build_stats_guided_session_prompt(
        variable_names=variable_names,
        observational_data=obs_data,
        budget=env.remaining_budget,
    )

    if trace is not None:
        trace.log(
            "stats_guided_input",
            work_key,
            {
                "variables": list(variable_names),
                "n_obs": int(obs_data.shape[0]),
                "initial_budget": int(env.remaining_budget),
            },
        )

    tool_history: list[ToolResult] = []
    latest_intervention_data: np.ndarray | None = None
    total_actions = 0
    stats_actions = 0
    intervene_actions = 0
    submit_actions = 0
    sanitized_overlap = 0
    sanitized_invalid = 0
    sanitized_cycle = 0

    for step in range(1, max_steps + 1):
        raw = model.complete(
            system_prompt=system_prompt,
            session_prompt=session_prompt,
            tool_history=tuple(tool_history),
            remaining_budget=env.remaining_budget,
        )
        payload = json.loads(raw)
        action = str(payload.get("action", "")).strip()
        total_actions += 1

        if trace is not None:
            trace.log(
                "stats_guided_action",
                work_key,
                {
                    "step": step,
                    "action": action,
                    "remaining_budget": int(env.remaining_budget),
                },
            )

        data_source = (
            "latest_intervention" if latest_intervention_data is not None else "observational"
        )
        active_data = (
            latest_intervention_data
            if latest_intervention_data is not None
            else obs_data
        )

        if action == "correlation":
            stats_actions += 1
            i = int(payload["i"])
            j = int(payload["j"])
            tool_payload: dict[str, Any] = {
                "i": i,
                "j": j,
                "data_source": data_source,
                "rows": int(active_data.shape[0]),
            }
            try:
                tool_payload["value"] = correlation(active_data, i, j)
            except ValueError as exc:
                tool_payload["value"] = None
                tool_payload["error"] = str(exc)
            tool_history.append(ToolResult(tool="correlation", payload=tool_payload))
            continue

        if action == "partial_correlation":
            stats_actions += 1
            i = int(payload["i"])
            j = int(payload["j"])
            cond = _normalize_conditioning(payload)
            tool_payload = {
                "i": i,
                "j": j,
                "conditioning_on": list(cond),
                "data_source": data_source,
                "rows": int(active_data.shape[0]),
            }
            try:
                tool_payload["value"] = partial_correlation(active_data, i, j, cond)
            except ValueError as exc:
                tool_payload["value"] = None
                tool_payload["error"] = str(exc)
            tool_history.append(
                ToolResult(tool="partial_correlation", payload=tool_payload)
            )
            continue

        if action == "independence_test":
            stats_actions += 1
            i = int(payload["i"])
            j = int(payload["j"])
            cond = _normalize_conditioning(payload)
            alpha = float(payload.get("alpha") or 0.05)
            tool_payload = {
                "i": i,
                "j": j,
                "conditioning_on": list(cond),
                "alpha": alpha,
                "data_source": data_source,
                "rows": int(active_data.shape[0]),
            }
            try:
                independent, p_value = independence_test(
                    active_data, i, j, cond, alpha=alpha
                )
                tool_payload["independent"] = independent
                tool_payload["p_value"] = p_value
            except ValueError as exc:
                tool_payload["independent"] = None
                tool_payload["p_value"] = None
                tool_payload["error"] = str(exc)
            tool_history.append(
                ToolResult(tool="independence_test", payload=tool_payload)
            )
            continue

        if action == "intervene":
            if env.remaining_budget <= 0:
                raise RuntimeError(
                    "Protocol violation: intervention requested with exhausted budget"
                )
            try:
                var = int(payload["var"])
                value = float(payload["value"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid intervene payload: {exc}") from exc
            samples = env.intervene(var=var, value=value)
            latest_intervention_data = samples
            intervene_actions += 1
            tool_history.append(
                ToolResult(
                    tool="intervene",
                    payload={
                        "var": int(var),
                        "value": float(value),
                        "rows": samples.tolist(),
                        "remaining_budget": int(env.remaining_budget),
                    },
                )
            )
            if trace is not None:
                trace.log(
                    "stats_guided_intervention",
                    work_key,
                    {
                        "step": step,
                        "var": int(var),
                        "value": float(value),
                        "rows": int(samples.shape[0]),
                        "remaining_budget": int(env.remaining_budget),
                    },
                )
            continue

        if action == "submit_graph":
            submit_actions += 1
            submission, sanitize_diag = submission_from_model_payload(
                payload, instance.config.d, intervene_actions
            )
            sanitized_overlap += sanitize_diag["sanitized_overlap_count"]
            sanitized_invalid += sanitize_diag["sanitized_invalid_count"]
            sanitized_cycle += sanitize_diag["sanitized_cycle_count"]
            output = env.submit_graph(submission)
            scores = score_submission(instance, output.submission)
            diagnostics = {
                "total_actions": total_actions,
                "stats_actions": stats_actions,
                "intervene_actions": intervene_actions,
                "submit_actions": submit_actions,
                "sanitized_overlap_count": sanitized_overlap,
                "sanitized_invalid_count": sanitized_invalid,
                "sanitized_cycle_count": sanitized_cycle,
            }
            if trace is not None:
                trace.log(
                    "stats_guided_submit",
                    work_key,
                    {
                        "step": step,
                        "directed_edges": sorted(output.submission.directed_edges),
                        "undirected_edges": sorted(output.submission.undirected_edges),
                        "interventions_used": int(output.submission.interventions_used),
                        "diagnostics": diagnostics,
                    },
                )
            return output.submission, scores, model, diagnostics

        raise ValueError(f"Unsupported action for {METHOD}: {action!r}")

    raise RuntimeError(f"{METHOD} session exceeded max_steps={max_steps} without submit_graph")


def default_row(
    *,
    run_id: str,
    model: str,
    level: Any,
    seed: int,
    runtime_seed: int,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "panel": PANEL,
        "method": METHOD,
        "model": model,
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
        "cache_creation_input_tokens": "",
        "cache_read_input_tokens": "",
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
        "sanitized_invalid_count": "",
        "sanitized_cycle_count": "",
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


def _float_field(row: dict[str, Any], field: str) -> float | None:
    raw = row.get(field, "")
    if raw in ("", None):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _summary_order_key(key: tuple[str, str, str]) -> tuple[int, int | str, str]:
    level, method, model = key
    if level == "overall":
        return (1, 0, model)
    return (0, int(level), model)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["level"]), str(row["method"]), str(row["model"]))
        grouped[key].append(row)
        grouped[("overall", str(row["method"]), str(row["model"]))].append(row)

    summary: list[dict[str, Any]] = []
    for (level, method, model), group in sorted(
        grouped.items(), key=lambda item: _summary_order_key(item[0])
    ):
        success_group = [row for row in group if row.get("status") == "success"]
        out: dict[str, Any] = {
            "level": level,
            "method": method,
            "model": model,
            "n_total": len(group),
            "n_success": len(success_group),
            "n_failed": len(group) - len(success_group),
        }
        for metric in SUMMARY_METRICS:
            values = [
                v for v in (_float_field(row, metric) for row in success_group) if v is not None
            ]
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
        for metric in BEHAVIOR_METRICS:
            values = [
                v for v in (_float_field(row, metric) for row in success_group) if v is not None
            ]
            out[f"{metric}_mean"] = float(np.mean(values)) if values else ""
        summary.append(out)
    return summary


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["level", "method", "model", "n_total", "n_success", "n_failed"]
    for metric in SUMMARY_METRICS:
        fieldnames.extend(
            [f"{metric}_mean", f"{metric}_std", f"{metric}_ci95_low", f"{metric}_ci95_high"]
        )
    for metric in BEHAVIOR_METRICS:
        fieldnames.append(f"{metric}_mean")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def print_summary(summary: list[dict[str, Any]]) -> None:
    print("level          model   n  skel_f1   dir_f1      shd      eff")
    print("----------------------------------------------------------------")
    for row in summary:
        print(
            f"{str(row['level']):>7} {str(row['model']):>14} {int(row['n_success']):>3} "
            f"{float(row['skeleton_f1_mean'] or 0.0):>8.3f} "
            f"{float(row['directed_f1_mean'] or 0.0):>8.3f} "
            f"{float(row['dag_shd_mean'] or 0.0):>8.3f} "
            f"{float(row['efficiency_mean'] or 0.0):>8.3f}"
        )


def parse_args() -> argparse.Namespace:
    default_env = Path(__file__).resolve().parent.parent / ".env"
    parser = argparse.ArgumentParser(
        description=(
            "Run llm_stats_guided active-phase probe: same tools as llm_stats, "
            "but the system prompt contains an explicit walkthrough of the PC + "
            "greedy active algorithm."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--levels", default="0,1,2,3,4,5")
    parser.add_argument("--seeds-per-level", type=int, default=8)
    parser.add_argument("--models", default="gpt-5.4,claude-sonnet-4-6")
    parser.add_argument("--env-file", default=str(default_env))
    parser.add_argument(
        "--max-steps",
        type=int,
        default=32,
        help="Maximum LLM turns per session (matches the ladder's max_steps_stats).",
    )
    parser.add_argument(
        "--no-events",
        action="store_true",
        help="Skip writing events_stats_guided.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    levels_catalog = ladder_levels()
    levels = parse_levels(args.levels)
    for level_id in levels:
        if level_id not in levels_catalog:
            raise ValueError(f"Unknown ladder level: {level_id}")
    models = parse_models(args.models)
    seed_map = load_manifest_seed_map(args.manifest, levels, args.seeds_per_level)

    run_id = now_run_id()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    long_csv = args.out_dir / "results_stats_guided_long.csv"
    summary_csv = args.out_dir / "results_stats_guided_summary.csv"
    manifest_path = args.out_dir / "stats_guided_manifest.json"
    events_path = args.out_dir / "events_stats_guided.jsonl"

    env_loaded = load_dotenv(Path(args.env_file), override=True)

    required_providers = {provider_for_model(model) for model in models}
    provider_to_env = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    for provider in sorted(required_providers):
        env_name = provider_to_env.get(provider)
        if env_name is None:
            raise RuntimeError(f"Unsupported provider: {provider}")
        if not os.getenv(env_name):
            raise RuntimeError(
                f"{env_name} missing; required for provider={provider!r}. "
                f"env_file={args.env_file} loaded={env_loaded}"
            )

    write_json(
        manifest_path,
        {
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": METHOD,
            "panel": PANEL,
            "input_manifest": str(args.manifest),
            "levels": levels,
            "models": models,
            "seeds_per_level": args.seeds_per_level,
            "max_steps": args.max_steps,
            "seed_map": {str(k): v for k, v in seed_map.items()},
            "prompt_summary": (
                "raw observational rows + stats tools, with explicit PC + greedy "
                "active algorithm walkthrough in system prompt"
            ),
        },
    )

    trace = None if args.no_events else EventWriter(events_path)
    work_items = [
        (level_id, seed, model)
        for level_id in levels
        for seed in seed_map[level_id]
        for model in models
    ]
    total = len(work_items)
    instance_cache: dict[tuple[int, int], Any] = {}

    try:
        for index, (level_id, seed, model_id) in enumerate(work_items, start=1):
            level = levels_catalog[level_id]
            runtime_seed = int(seed * 10_000 + level_id * 101 + 7)
            work_key = (
                f"panel={PANEL}|method={METHOD}|level={level_id}|"
                f"seed={seed}|model={model_id}"
            )
            print(f"[{index}/{total}] {work_key}")
            row = default_row(
                run_id=run_id,
                model=model_id,
                level=level,
                seed=seed,
                runtime_seed=runtime_seed,
            )
            started = time.perf_counter()
            try:
                cache_key = (level_id, seed)
                if cache_key not in instance_cache:
                    cfg = config_from_level(level)
                    instance_cache[cache_key] = build_benchmark_instance(
                        cfg, np.random.default_rng(seed)
                    )
                instance = instance_cache[cache_key]
                enrich_instance_fields(row, instance)
                submission, scores, model_usage, diagnostics = run_stats_guided_llm(
                    instance=instance,
                    runtime_seed=runtime_seed,
                    model_id=model_id,
                    max_steps=args.max_steps,
                    trace=trace,
                    work_key=work_key,
                )
                row["prompt_tokens"] = model_usage.prompt_tokens
                row["completion_tokens"] = model_usage.completion_tokens
                row["total_tokens"] = model_usage.total_tokens
                row["cache_creation_input_tokens"] = model_usage.cache_creation_input_tokens
                row["cache_read_input_tokens"] = model_usage.cache_read_input_tokens
                row["interventions_used"] = submission.interventions_used
                row["submit_directed"] = len(submission.directed_edges)
                row["submit_undirected"] = len(submission.undirected_edges)
                for key, value in diagnostics.items():
                    row[key] = value
                enrich_score_fields(row, scores)
                row["status"] = "success"
                if trace is not None:
                    trace.log(
                        "work_success",
                        work_key,
                        {
                            "directed_f1": row["directed_f1"],
                            "dag_shd": row["dag_shd"],
                            "interventions_used": row["interventions_used"],
                        },
                    )
            except Exception as exc:  # noqa: BLE001
                row["status"] = "failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
                if trace is not None:
                    trace.log("work_failed", work_key, {"error": row["error"]})
            finally:
                row["latency_sec"] = round(time.perf_counter() - started, 6)
                append_row(long_csv, row)
    finally:
        if trace is not None:
            trace.close()

    rows = load_rows(long_csv)
    summary = summarize_rows(rows)
    write_summary_csv(summary_csv, summary)
    print_summary(summary)
    print(f"\n[done] long={long_csv}")
    print(f"[done] summary={summary_csv}")
    print(f"[done] manifest={manifest_path}")
    if not args.no_events:
        print(f"[done] events={events_path}")


if __name__ == "__main__":
    main()
