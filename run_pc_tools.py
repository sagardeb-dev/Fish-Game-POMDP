"""LLM with PC-pipeline phases as composable tools.

This ablation tests whether the LLM can drive the classical pipeline when
each algorithm phase is exposed as a callable tool. The LLM does not run
independence tests itself; instead it has higher-level primitives:

- pc_observational(alpha) — runs PC on the observational data and returns
  the resulting CPDAG (compelled directed + undirected edges).
- meek_closure(directed_edges, undirected_edges) — applies Meek rules 1-3
  to a partial DAG and returns the closed graph.
- intervene(var, value) — performs an intervention (consumes one budget
  unit) and returns rows + per-variable mean shift vs the observational
  baseline.
- submit_graph(directed_edges, undirected_edges, reasoning_summary) —
  terminal submission.

Compared to `llm_stats_guided`, this method does not require the LLM to
implement the algorithm in its head; it only has to *orchestrate* the
existing primitives. Compared to `llm_pc_handoff`, the LLM now actively
drives `pc_observational` (one extra step) and can re-run `meek_closure`
between interventions to propagate orientations — which is more than
`pc_greedy` does in the active phase.
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
    parse_causallearn_endpoint_matrix,
    provider_for_model,
    score_submission,
)
from causal_discovery.equivalence.cpdag import canonical_undirected_edge  # noqa: E402
from causal_discovery.equivalence.theory import (  # noqa: E402
    _MutablePDAG,
    _apply_meek_closure,
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


METHOD = "llm_pc_tools"
PANEL = "active"
ALLOWED_ACTIONS = frozenset(
    {"pc_observational", "meek_closure", "intervene", "submit_graph"}
)
PC_TOOLS_ACTION_NAMES = (
    "pc_observational",
    "meek_closure",
    "intervene",
    "submit_graph",
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
    "alpha",
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
    "pc_observational_actions",
    "meek_closure_actions",
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
    "pc_observational_actions",
    "meek_closure_actions",
    "sanitized_overlap_count",
    "sanitized_invalid_count",
    "sanitized_cycle_count",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)


def make_pc_tools_action_response_schema(
    allowed_actions: frozenset[str],
) -> dict[str, Any]:
    unknown = allowed_actions.difference(PC_TOOLS_ACTION_NAMES)
    if unknown:
        raise ValueError(f"Unknown action names in PC-tools schema: {sorted(unknown)}")
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
                "alpha",
                "directed_edges",
                "undirected_edges",
                "reasoning_summary",
            ],
            "additionalProperties": False,
        },
    }


def make_pc_tools_action_tool(allowed_actions: frozenset[str]) -> dict[str, Any]:
    schema = make_pc_tools_action_response_schema(allowed_actions)
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": "Submit exactly one causal-discovery benchmark action.",
            "strict": schema["strict"],
            "parameters": schema["schema"],
        },
    }


class PCToolsLiteLLMModel(LiteLLMJSONPolicyModel):
    """LiteLLM adapter with the pc_tools custom schema (extra action enum)."""

    def __init__(self, model: str, allowed_actions: frozenset[str]) -> None:
        # Initialize the parent with a placeholder allowed_actions that the upstream
        # schema accepts, then overwrite the schema/tool fields with our custom set.
        super().__init__(model=model, allowed_actions=frozenset({"intervene", "submit_graph"}))
        self._action_schema = make_pc_tools_action_response_schema(allowed_actions)
        self._action_tool = make_pc_tools_action_tool(allowed_actions)


def make_model(model_id: str) -> PCToolsLiteLLMModel:
    return PCToolsLiteLLMModel(model=model_id, allowed_actions=ALLOWED_ACTIONS)


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


def build_pc_tools_system_prompt() -> str:
    return (
        "You are performing ACTIVE causal discovery over an unknown linear-Gaussian "
        "DAG with full observability and no hidden confounders.\n"
        "An intervention forces one variable to a fixed value, breaking its "
        "incoming dependencies. By comparing pre- and post-intervention means you "
        "can determine causal direction.\n"
        "\n"
        "You have the PC + greedy classical pipeline available as composable tools. "
        "You orchestrate them; you do not need to implement the algorithm yourself.\n"
        "\n"
        "TOOLS:\n"
        "- pc_observational(alpha): runs the PC algorithm on the observational data. "
        "Returns the resulting CPDAG: compelled directed edges + undirected edges. "
        "Call this exactly once at the start to obtain the observational ceiling.\n"
        "- meek_closure(directed_edges, undirected_edges): applies Meek rules 1-3 to "
        "your current partial DAG and returns the closed graph. REQUIRED after "
        "every intervention — propagating one new orientation through Meek rules "
        "frequently resolves several other undirected edges at no budget cost.\n"
        "- intervene(var, value): performs an intervention; consumes one budget "
        "unit. Returns post-intervention rows AND per-variable mean shift vs the "
        "observational baseline.\n"
        "- submit_graph(directed_edges, undirected_edges, reasoning_summary): "
        "terminal submission.\n"
        "\n"
        "MANDATORY WORKFLOW (follow exactly; do not skip steps):\n"
        "Step 1. Call pc_observational(alpha=0.05) once. This gives you the CPDAG.\n"
        "  Note the count u of undirected edges. You will need at most u "
        "interventions, but typically fewer because each intervention resolves "
        "all undirected edges incident to the intervened node.\n"
        "Step 2. ITERATIVE ACTIVE LOOP. Continue this loop while "
        "(remaining_budget > 0 AND your current CPDAG has any undirected edge):\n"
        "  a. Pick the undirected node with the MOST undirected edges incident to "
        "it (break ties by lowest variable index) as the intervention target.\n"
        "  b. Call intervene(var=target, value=mu_obs[target] + 3.0). Read the "
        "mu_shift_vs_obs vector in the result.\n"
        "  c. For each undirected edge (target, other) incident to the intervened "
        "node: if |mu_shift[other]| > 0.5, orient target -> other (target is "
        "the ancestor); otherwise orient other -> target. Skip orientations "
        "that would create a cycle.\n"
        "  d. REQUIRED: Call meek_closure(updated_directed, remaining_undirected). "
        "Use the returned graph as your new CPDAG before deciding the next "
        "iteration. Skipping meek_closure leaves resolvable orientations on the "
        "table.\n"
        "  Then re-evaluate the loop condition. If undirected edges remain AND "
        "budget remains, do another iteration.\n"
        "Step 3. Submit the final DAG via submit_graph. Edges still undirected "
        "after the loop must be submitted as undirected.\n"
        "\n"
        "EXPECTED ACTION SEQUENCE (typical):\n"
        "  pc_observational -> intervene -> meek_closure "
        "[-> intervene -> meek_closure ...] -> submit_graph.\n"
        "Sequences ending pc_observational -> intervene -> submit_graph are usually "
        "premature: meek_closure is missing, and any remaining undirected edges "
        "haven't been considered for further interventions.\n"
        "\n"
        "CONSTRAINTS:\n"
        "- Skeleton constraint: every edge you submit (directed or undirected) "
        "must correspond to a pair that appeared in pc_observational's output. Do "
        "not invent adjacencies.\n"
        "- Do not call pc_observational more than once per session.\n"
        "- Use the provided action tool exactly once per turn.\n"
        "- Do not request raw observational rows or unavailable metadata.\n"
        "\n"
        "PRE-SUBMIT CHECKLIST (verify all four before calling submit_graph):\n"
        "  1. I have called pc_observational exactly once.\n"
        "  2. After every intervene I have called meek_closure (or remaining_budget "
        "had run out and submission was forced).\n"
        "  3. My submitted skeleton matches pc_observational's output skeleton.\n"
        "  4. remaining_budget == 0 OR no undirected edges remain.\n"
        "If any item fails, do NOT submit yet — perform the missing step instead."
    )


def build_pc_tools_session_prompt(
    *,
    variable_names: tuple[str, ...],
    d: int,
    n_obs: int,
    mu_obs: np.ndarray,
    budget: int,
) -> str:
    payload = {
        "variables": list(variable_names),
        "d": int(d),
        "n_obs": int(n_obs),
        "observational_means_rounded_3dp": [round(float(value), 3) for value in mu_obs],
        "remaining_budget": int(budget),
        "tools_summary": [
            "pc_observational(alpha) -> {compelled_directed_edges, undirected_edges}",
            "meek_closure(directed_edges, undirected_edges) -> {directed_edges, undirected_edges}",
            "intervene(var, value) -> {rows, mu_intervention, mu_shift_vs_obs, remaining_budget}",
            "submit_graph(directed_edges, undirected_edges, reasoning_summary)",
        ],
        "output_schema": {
            "action": "pc_observational|meek_closure|intervene|submit_graph",
            "var": "int|null; required for intervene, otherwise null",
            "value": "float|null; required for intervene, otherwise null",
            "alpha": "float|null; required for pc_observational, otherwise null",
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


def _sanitize_edge_lists(
    directed_raw: object, undirected_raw: object, d: int
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    directed: set[tuple[int, int]] = set()
    directed_pairs: set[tuple[int, int]] = set()
    if isinstance(directed_raw, list):
        for edge in directed_raw:
            pair_int = _edge_pair(edge)
            if pair_int is None:
                continue
            s, t = pair_int
            if s == t or not (0 <= s < d) or not (0 <= t < d):
                continue
            if (t, s) in directed:
                continue
            directed.add((s, t))
            directed_pairs.add((min(s, t), max(s, t)))

    undirected: set[tuple[int, int]] = set()
    if isinstance(undirected_raw, list):
        for edge in undirected_raw:
            pair_int = _edge_pair(edge)
            if pair_int is None:
                continue
            a, b = pair_int
            if a == b or not (0 <= a < d) or not (0 <= b < d):
                continue
            canon = canonical_undirected_edge(a, b)
            if canon in directed_pairs:
                continue
            undirected.add(canon)

    return directed, undirected


def run_pc_observational_tool(
    obs_data: np.ndarray, alpha: float
) -> dict[str, Any]:
    from causallearn.search.ConstraintBased.PC import pc

    cg = pc(obs_data, alpha=alpha, indep_test="fisherz", show_progress=False, verbose=False)
    directed, undirected = parse_causallearn_endpoint_matrix(cg.G.graph)
    return {
        "compelled_directed_edges": [list(edge) for edge in sorted(directed)],
        "undirected_edges": [list(edge) for edge in sorted(undirected)],
        "alpha": float(alpha),
    }


def run_meek_closure_tool(
    directed_raw: object, undirected_raw: object, d: int
) -> dict[str, Any]:
    directed, undirected = _sanitize_edge_lists(directed_raw, undirected_raw, d)
    pdag = _MutablePDAG(
        num_nodes=d,
        directed_edges=set(directed),
        undirected_edges=set(undirected),
    )
    try:
        _apply_meek_closure(pdag, include_rule4=False)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"meek_closure failed: {exc}"}
    return {
        "directed_edges": [list(edge) for edge in sorted(pdag.directed_edges)],
        "undirected_edges": [list(edge) for edge in sorted(pdag.undirected_edges)],
    }


def run_pc_tools_llm(
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
    mu_obs = obs_data.mean(axis=0)

    variable_names = tuple(f"X{i}" for i in range(instance.config.d))
    model = make_model(model_id)
    system_prompt = build_pc_tools_system_prompt()
    session_prompt = build_pc_tools_session_prompt(
        variable_names=variable_names,
        d=instance.config.d,
        n_obs=instance.config.n_obs,
        mu_obs=mu_obs,
        budget=env.remaining_budget,
    )

    if trace is not None:
        trace.log(
            "pc_tools_input",
            work_key,
            {
                "variables": list(variable_names),
                "n_obs": int(instance.config.n_obs),
                "initial_budget": int(env.remaining_budget),
                "observational_means": [float(x) for x in mu_obs],
            },
        )

    tool_history: list[ToolResult] = []
    total_actions = 0
    intervene_actions = 0
    submit_actions = 0
    pc_observational_actions = 0
    meek_closure_actions = 0
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
                "pc_tools_action",
                work_key,
                {
                    "step": step,
                    "action": action,
                    "remaining_budget": int(env.remaining_budget),
                },
            )

        if action == "pc_observational":
            pc_observational_actions += 1
            alpha = float(payload.get("alpha") or 0.05)
            result = run_pc_observational_tool(obs_data, alpha=alpha)
            tool_history.append(ToolResult(tool="pc_observational", payload=result))
            continue

        if action == "meek_closure":
            meek_closure_actions += 1
            result = run_meek_closure_tool(
                payload.get("directed_edges"),
                payload.get("undirected_edges"),
                instance.config.d,
            )
            tool_history.append(ToolResult(tool="meek_closure", payload=result))
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
            mu_int = samples.mean(axis=0)
            intervene_actions += 1
            tool_payload = {
                "var": int(var),
                "value": float(value),
                "rows": samples.tolist(),
                "mu_intervention_rounded_3dp": [round(float(x), 3) for x in mu_int],
                "mu_shift_vs_obs_rounded_3dp": [
                    round(float(mu_int[k] - mu_obs[k]), 3)
                    for k in range(instance.config.d)
                ],
                "remaining_budget": int(env.remaining_budget),
            }
            tool_history.append(ToolResult(tool="intervene", payload=tool_payload))
            if trace is not None:
                trace.log(
                    "pc_tools_intervention",
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
                "stats_actions": 0,
                "intervene_actions": intervene_actions,
                "submit_actions": submit_actions,
                "pc_observational_actions": pc_observational_actions,
                "meek_closure_actions": meek_closure_actions,
                "sanitized_overlap_count": sanitized_overlap,
                "sanitized_invalid_count": sanitized_invalid,
                "sanitized_cycle_count": sanitized_cycle,
            }
            if trace is not None:
                trace.log(
                    "pc_tools_submit",
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
    alpha: float,
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
        "alpha": alpha,
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
        "pc_observational_actions": "",
        "meek_closure_actions": "",
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
            "Run llm_pc_tools: LLM with PC pipeline phases (pc_observational, "
            "meek_closure) plus intervene + submit_graph as composable tools."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--levels", default="0,1,2,3,4,5")
    parser.add_argument("--seeds-per-level", type=int, default=8)
    parser.add_argument("--models", default="gpt-5.4,claude-sonnet-4-6")
    parser.add_argument("--env-file", default=str(default_env))
    parser.add_argument(
        "--default-alpha",
        type=float,
        default=0.05,
        help="Recorded in the row (LLM picks its own alpha when calling pc_observational).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=24,
        help=(
            "Maximum LLM turns per session. PC-tools sessions need a few extra turns "
            "vs llm_raw because of pc_observational + meek_closure calls."
        ),
    )
    parser.add_argument(
        "--no-events",
        action="store_true",
        help="Skip writing events_pc_tools.jsonl.",
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
    long_csv = args.out_dir / "results_pc_tools_long.csv"
    summary_csv = args.out_dir / "results_pc_tools_summary.csv"
    manifest_path = args.out_dir / "pc_tools_manifest.json"
    events_path = args.out_dir / "events_pc_tools.jsonl"

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
            "default_alpha": args.default_alpha,
            "max_steps": args.max_steps,
            "seed_map": {str(k): v for k, v in seed_map.items()},
            "prompt_summary": (
                "PC pipeline phases (pc_observational, meek_closure) exposed as "
                "composable tools alongside intervene + submit_graph"
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
                alpha=args.default_alpha,
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
                submission, scores, model_usage, diagnostics = run_pc_tools_llm(
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
                            "pc_observational_actions": row["pc_observational_actions"],
                            "meek_closure_actions": row["meek_closure_actions"],
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
