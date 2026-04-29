"""PC-handoff active-phase LLM probe.

This ablation isolates the active phase of causal discovery. It runs the
classical PC algorithm on the observational data (the same call that
`pc_greedy` uses), hands the resulting CPDAG to the LLM, and lets the LLM
choose interventions and orient the remaining undirected edges within the
benchmark budget.

Compared to `run_corr_obs_probe.py`, which probes the observational phase by
giving the LLM PC's input (correlation matrix), this script probes the
active phase by giving the LLM PC's output (CPDAG). Together the two
probes decompose where LLMs lose to `pc_greedy`.

Non-goals / anti-leakage:
- Does not call the true DAG before submission.
- Does not give the LLM raw observational rows or stat tools.
- Does not let the LLM modify the skeleton; only orient and intervene.
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
from run_ladder import (  # noqa: E402
    config_from_level,
    enrich_instance_fields,
    enrich_score_fields,
    ladder_levels,
    parse_levels,
    parse_models,
    would_create_cycle,
)


METHOD = "llm_pc_handoff"
PANEL = "active"
ALLOWED_ACTIONS = frozenset({"intervene", "submit_graph"})

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
    "pc_input_directed",
    "pc_input_undirected",
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
    "pc_input_directed",
    "pc_input_undirected",
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


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def build_pc_handoff_system_prompt() -> str:
    return (
        "You are performing the active phase of causal discovery over an unknown "
        "linear-Gaussian DAG with full observability and no hidden confounders.\n"
        "The observational phase has already been completed for you by the PC "
        "algorithm. You are given the resulting CPDAG (Completed Partially "
        "Directed Acyclic Graph):\n"
        "- compelled_directed_edges: PC has confidently oriented these from the "
        "observational data. Treat them as fixed and include them in your final "
        "submission as-is.\n"
        "- undirected_edges: PC could not orient these from observational data "
        "alone (Markov-equivalent ambiguity). Your job is to orient them using "
        "interventions.\n"
        "- observational_means: per-variable sample means under the observational "
        "distribution. Use these as the baseline for detecting post-intervention "
        "shifts.\n"
        "An intervention forces one variable to a fixed value, breaking its "
        "incoming dependencies; the other variables still follow their normal "
        "relationships. If intervening on X causes Y's mean to shift relative to "
        "the observational baseline, then X is an ancestor of Y in the SCM. "
        "Picking interventions on high-degree undirected nodes resolves the most "
        "edges per call.\n"
        "Skeleton constraint: do not invent skeleton edges. Every edge you submit "
        "(directed or undirected) must correspond to a pair that appeared in the "
        "input CPDAG (either as a compelled edge or as an undirected edge).\n"
        "Allowed actions:\n"
        "- intervene(var, value)\n"
        "- submit_graph(directed_edges, undirected_edges, reasoning_summary)\n"
        "Use the causal_discovery_action tool exactly once per turn. Do not "
        "request raw observational rows, statistical tools, or unavailable "
        "metadata."
    )


def build_pc_handoff_session_prompt(
    *,
    variable_names: tuple[str, ...],
    d: int,
    n_obs: int,
    cpdag_directed: list[tuple[int, int]],
    cpdag_undirected: list[tuple[int, int]],
    mu_obs: np.ndarray,
    budget: int,
) -> str:
    payload = {
        "variables": list(variable_names),
        "d": int(d),
        "n_obs": int(n_obs),
        "compelled_directed_edges": [[int(s), int(t)] for s, t in cpdag_directed],
        "undirected_edges": [[int(a), int(b)] for a, b in cpdag_undirected],
        "observational_means_rounded_3dp": [
            round(float(value), 3) for value in mu_obs
        ],
        "remaining_budget": int(budget),
        "instructions_summary": [
            "Compelled edges are PC's oriented edges; keep them in the final submission.",
            "Undirected edges are unresolved; orient them using interventions.",
            "Mean shift after intervening on X reveals X is an ancestor of the shifted variable.",
            "High-degree undirected nodes are usually the most informative intervention targets.",
            "Do not introduce edges that are not present in the input CPDAG skeleton.",
        ],
        "output_schema": {
            "action": "intervene|submit_graph",
            "var": "int|null; required for intervene, otherwise null",
            "value": "float|null; required for intervene, otherwise null",
            "i": None,
            "j": None,
            "conditioning_on": [],
            "alpha": None,
            "directed_edges": "list[list[int, int]]",
            "undirected_edges": "list[list[int, int]]",
            "reasoning_summary": "string",
        },
        "json_contract": (
            "Return all output_schema fields. Use null for unused scalar fields "
            "and [] for unused list fields."
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


def run_pc_offline(obs_data: np.ndarray, alpha: float):
    """Run PC on observational data and return (directed, undirected) edge sets."""
    from causallearn.search.ConstraintBased.PC import pc

    cg = pc(obs_data, alpha=alpha, indep_test="fisherz", show_progress=False, verbose=False)
    return parse_causallearn_endpoint_matrix(cg.G.graph)


def run_pc_handoff_llm(
    *,
    instance: Any,
    runtime_seed: int,
    model_id: str,
    alpha: float,
    max_steps: int,
    trace: EventWriter | None,
    work_key: str,
) -> tuple[GraphSubmission, Any, Any, dict[str, int], dict[str, int]]:
    env = BenchmarkEnv(instance, np.random.default_rng(runtime_seed))
    obs_data = env.observe()
    mu_obs = obs_data.mean(axis=0)

    pc_directed_frozen, pc_undirected_frozen = run_pc_offline(obs_data, alpha)
    pc_directed = sorted(pc_directed_frozen)
    pc_undirected = sorted(pc_undirected_frozen)

    variable_names = tuple(f"X{i}" for i in range(instance.config.d))
    model = make_model(model_id)
    system_prompt = build_pc_handoff_system_prompt()
    session_prompt = build_pc_handoff_session_prompt(
        variable_names=variable_names,
        d=instance.config.d,
        n_obs=instance.config.n_obs,
        cpdag_directed=pc_directed,
        cpdag_undirected=pc_undirected,
        mu_obs=mu_obs,
        budget=env.remaining_budget,
    )

    if trace is not None:
        trace.log(
            "pc_handoff_input",
            work_key,
            {
                "variables": list(variable_names),
                "n_obs": int(instance.config.n_obs),
                "compelled_directed_edges": [list(edge) for edge in pc_directed],
                "undirected_edges": [list(edge) for edge in pc_undirected],
                "observational_means": [float(x) for x in mu_obs],
                "initial_budget": int(env.remaining_budget),
                "alpha": float(alpha),
            },
        )

    tool_history: list[ToolResult] = []
    total_actions = 0
    intervene_actions = 0
    submit_actions = 0
    sanitized_overlap = 0
    sanitized_invalid = 0
    sanitized_cycle = 0

    pc_input_counts = {
        "pc_input_directed": len(pc_directed),
        "pc_input_undirected": len(pc_undirected),
    }

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
                "pc_handoff_action",
                work_key,
                {
                    "step": step,
                    "action": action,
                    "remaining_budget": int(env.remaining_budget),
                },
            )

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
                    round(float(mu_int[k] - mu_obs[k]), 3) for k in range(instance.config.d)
                ],
                "remaining_budget": int(env.remaining_budget),
            }
            tool_history.append(ToolResult(tool="intervene", payload=tool_payload))
            if trace is not None:
                trace.log(
                    "pc_handoff_intervention_result",
                    work_key,
                    {
                        "step": step,
                        "var": int(var),
                        "value": float(value),
                        "rows": int(samples.shape[0]),
                        "mu_intervention": [float(x) for x in mu_int],
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
                "sanitized_overlap_count": sanitized_overlap,
                "sanitized_invalid_count": sanitized_invalid,
                "sanitized_cycle_count": sanitized_cycle,
            }
            if trace is not None:
                trace.log(
                    "pc_handoff_submit",
                    work_key,
                    {
                        "step": step,
                        "directed_edges": sorted(output.submission.directed_edges),
                        "undirected_edges": sorted(output.submission.undirected_edges),
                        "interventions_used": int(output.submission.interventions_used),
                        "diagnostics": diagnostics,
                    },
                )
            return output.submission, scores, model, diagnostics, pc_input_counts

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
        "pc_input_directed": "",
        "pc_input_undirected": "",
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
            "Run PC-handoff active-phase LLM probe: PC computes the CPDAG, the "
            "LLM only chooses interventions and orients the remaining edges."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--levels", default="0,1,2,3,4,5")
    parser.add_argument("--seeds-per-level", type=int, default=8)
    parser.add_argument("--models", default="gpt-5.4,claude-sonnet-4-6")
    parser.add_argument("--env-file", default=str(default_env))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=16,
        help="Maximum LLM turns per session (interventions + final submit_graph).",
    )
    parser.add_argument(
        "--no-events",
        action="store_true",
        help="Skip writing events_pc_handoff.jsonl.",
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
    long_csv = args.out_dir / "results_pc_handoff_long.csv"
    summary_csv = args.out_dir / "results_pc_handoff_summary.csv"
    manifest_path = args.out_dir / "pc_handoff_manifest.json"
    events_path = args.out_dir / "events_pc_handoff.jsonl"

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
            "alpha": args.alpha,
            "max_steps": args.max_steps,
            "seed_map": {str(k): v for k, v in seed_map.items()},
            "prompt_summary": (
                "PC-computed CPDAG (compelled + undirected edges) + "
                "observational means + budget; LLM chooses interventions "
                "and orients remaining edges."
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
                alpha=args.alpha,
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
                submission, scores, model_usage, diagnostics, pc_counts = run_pc_handoff_llm(
                    instance=instance,
                    runtime_seed=runtime_seed,
                    model_id=model_id,
                    alpha=args.alpha,
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
                for key, value in pc_counts.items():
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
                            "pc_input_undirected": row["pc_input_undirected"],
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
