"""Root-only LLM correlation/standard-deviation observational probe.

This diagnostic checks whether a compact observational summary helps LLM causal
graph recovery. It intentionally does not modify or depend on ladder internals
beyond reusing the existing model adapters, level configs, and scoring helpers.
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
from openai import OpenAI

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_discovery import GraphSubmission, build_benchmark_instance, score_submission  # noqa: E402
from run_ladder import (  # noqa: E402
    AnthropicJSONPolicyModel,
    OpenAIJSONPolicyModel,
    config_from_level,
    enrich_instance_fields,
    enrich_score_fields,
    ladder_levels,
    parse_levels,
    parse_models,
    provider_for_model,
    would_create_cycle,
)


METHOD = "llm_corr_obs"
ALLOWED_ACTIONS = frozenset({"submit_graph"})

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


def load_manifest_seed_map(manifest_path: Path, levels: list[int], seeds_per_level: int) -> dict[int, list[int]]:
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


def corr_std_summary(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D observational data, got shape {array.shape}")
    if array.shape[0] < 2:
        raise ValueError("At least two observational rows are required")
    corr = np.round(np.corrcoef(array, rowvar=False), 3)
    std = np.round(np.std(array, axis=0, ddof=1), 3)
    return corr, std


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def format_std_row(variable_names: tuple[str, ...], std: np.ndarray) -> str:
    return ", ".join(
        f"{name}={_format_float(float(value))}"
        for name, value in zip(variable_names, std, strict=True)
    )


def format_corr_matrix(variable_names: tuple[str, ...], corr: np.ndarray) -> str:
    array = np.asarray(corr, dtype=float)
    expected_shape = (len(variable_names), len(variable_names))
    if array.shape != expected_shape:
        raise ValueError(f"Correlation matrix shape {array.shape} != {expected_shape}")
    header = "      " + " ".join(f"{name:>7}" for name in variable_names)
    rows = [header]
    for name, row in zip(variable_names, array, strict=True):
        rows.append(f"{name:>5} " + " ".join(f"{_format_float(value):>7}" for value in row))
    return "\n".join(rows)


def build_corr_obs_system_prompt() -> str:
    return (
        "You are performing observational causal discovery over an unknown "
        "linear-Gaussian DAG with full observability and no hidden confounders.\n"
        "No experiments or interventions are available.\n"
        "You are given only variable names, the number of observational samples, "
        "sample standard deviations, and a rounded sample correlation matrix.\n"
        "Correlation does not imply direct causation. Shared causes and causal "
        "chains can induce correlation between variables that do not share a "
        "direct edge.\n"
        "Larger variance can be consistent with downstream accumulation in a "
        "linear system, but it is not proof of causal direction.\n"
        "Prefer undirected edges over unsupported directions. Orient an edge only "
        "when the summary evidence gives a defensible directional reason.\n"
        "Allowed action: submit_graph(directed_edges, undirected_edges, reasoning_summary).\n"
        "Use the causal_discovery_action tool exactly once. Do not request raw rows, "
        "interventions, statistical tools, or unavailable metadata."
    )


def build_corr_obs_session_prompt(
    variable_names: tuple[str, ...], n_obs: int, std: np.ndarray, corr: np.ndarray
) -> str:
    payload = {
        "variables": list(variable_names),
        "d": len(variable_names),
        "n_obs": int(n_obs),
        "sample_std_ddof1_rounded_3dp": [float(x) for x in std],
        "sample_std_row": format_std_row(variable_names, std),
        "correlation_matrix_rounded_3dp": [
            [float(value) for value in row] for row in corr
        ],
        "correlation_matrix_text": format_corr_matrix(variable_names, corr),
        "warnings": [
            "Correlation does not imply direct causation.",
            "Shared causes and chains can induce correlation.",
            "Larger variance can be consistent with downstream accumulation but is not proof of direction.",
            "Prefer undirected edges over unsupported directions.",
            "No experiments are available.",
        ],
        "output_schema": {
            "action": "submit_graph",
            "var": None,
            "value": None,
            "i": None,
            "j": None,
            "conditioning_on": [],
            "alpha": None,
            "directed_edges": "list[list[int, int]]",
            "undirected_edges": "list[list[int, int]]",
            "reasoning_summary": "string",
        },
        "json_contract": "Return all output_schema fields. Use null for unused scalar fields and [] for unused list fields.",
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _edge_pair(raw: object) -> tuple[int, int] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        return int(raw[0]), int(raw[1])
    except (TypeError, ValueError):
        return None


def submission_from_model_payload(payload: dict[str, Any], d: int) -> tuple[GraphSubmission, dict[str, int]]:
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
        interventions_used=0,
    )
    diagnostics = {
        "sanitized_overlap_count": overlap_count,
        "sanitized_invalid_count": invalid_count,
        "sanitized_cycle_count": cycle_count,
    }
    return submission, diagnostics


def make_model(
    *,
    model_id: str,
    openai_client: OpenAI | None,
    anthropic_api_key: str,
):
    provider = provider_for_model(model_id)
    if provider == "openai":
        if openai_client is None:
            raise RuntimeError("OPENAI_API_KEY missing; required for OpenAI model(s)")
        return OpenAIJSONPolicyModel(
            client=openai_client,
            model=model_id,
            allowed_actions=ALLOWED_ACTIONS,
        )
    if provider == "anthropic":
        if not anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY missing; required for Anthropic model(s)")
        return AnthropicJSONPolicyModel(
            api_key=anthropic_api_key,
            model=model_id,
            allowed_actions=ALLOWED_ACTIONS,
        )
    raise RuntimeError(f"Unsupported provider: {provider}")


def run_corr_obs_llm(
    *,
    instance: Any,
    model_id: str,
    openai_client: OpenAI | None,
    anthropic_api_key: str,
    trace: EventWriter | None,
    work_key: str,
) -> tuple[GraphSubmission, Any, Any, dict[str, int]]:
    variable_names = tuple(f"X{i}" for i in range(instance.config.d))
    corr, std = corr_std_summary(instance.observational_data)
    model = make_model(
        model_id=model_id,
        openai_client=openai_client,
        anthropic_api_key=anthropic_api_key,
    )
    system_prompt = build_corr_obs_system_prompt()
    session_prompt = build_corr_obs_session_prompt(
        variable_names=variable_names,
        n_obs=instance.config.n_obs,
        std=std,
        corr=corr,
    )
    if trace is not None:
        trace.log(
            "corr_obs_summary",
            work_key,
            {
                "variables": list(variable_names),
                "n_obs": int(instance.config.n_obs),
                "std": [float(x) for x in std],
                "corr": [[float(value) for value in row] for row in corr],
            },
        )
    raw = model.complete(
        system_prompt=system_prompt,
        session_prompt=session_prompt,
        tool_history=tuple(),
        remaining_budget=0,
    )
    payload = json.loads(raw)
    submission, diagnostics = submission_from_model_payload(payload, instance.config.d)
    scores = score_submission(instance, submission)
    diagnostics.update(
        {
            "total_actions": 1,
            "stats_actions": 0,
            "intervene_actions": 0,
            "submit_actions": 1,
        }
    )
    if trace is not None:
        trace.log(
            "corr_obs_submit",
            work_key,
            {
                "directed_edges": sorted(submission.directed_edges),
                "undirected_edges": sorted(submission.undirected_edges),
                "diagnostics": diagnostics,
            },
        )
    return submission, scores, model, diagnostics


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
        "panel": "observational",
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
    for (level, method, model), group in sorted(grouped.items(), key=lambda item: _summary_order_key(item[0])):
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
            values = [v for v in (_float_field(row, metric) for row in success_group) if v is not None]
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
            values = [v for v in (_float_field(row, metric) for row in success_group) if v is not None]
            out[f"{metric}_mean"] = float(np.mean(values)) if values else ""
        summary.append(out)
    return summary


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["level", "method", "model", "n_total", "n_success", "n_failed"]
    for metric in SUMMARY_METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_ci95_low", f"{metric}_ci95_high"])
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
        description="Run root-only LLM correlation/std observational probe."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--levels", default="1,3,5")
    parser.add_argument("--seeds-per-level", type=int, default=4)
    parser.add_argument("--models", default="gpt-5.4,claude-sonnet-4-6")
    parser.add_argument("--env-file", default=str(default_env))
    parser.add_argument(
        "--no-events",
        action="store_true",
        help="Skip writing events_corr_obs.jsonl.",
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
    long_csv = args.out_dir / "results_corr_obs_long.csv"
    summary_csv = args.out_dir / "results_corr_obs_summary.csv"
    manifest_path = args.out_dir / "corr_obs_manifest.json"
    events_path = args.out_dir / "events_corr_obs.jsonl"

    env_loaded = load_dotenv(Path(args.env_file), override=True)
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
    openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None

    required_providers = {provider_for_model(model) for model in models}
    if "openai" in required_providers and openai_client is None:
        raise RuntimeError(
            f"OPENAI_API_KEY missing; required for OpenAI model(s). "
            f"env_file={args.env_file} loaded={env_loaded}"
        )
    if "anthropic" in required_providers and not anthropic_api_key:
        raise RuntimeError(
            f"ANTHROPIC_API_KEY missing; required for Anthropic model(s). "
            f"env_file={args.env_file} loaded={env_loaded}"
        )

    write_json(
        manifest_path,
        {
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": METHOD,
            "input_manifest": str(args.manifest),
            "levels": levels,
            "models": models,
            "seeds_per_level": args.seeds_per_level,
            "seed_map": {str(k): v for k, v in seed_map.items()},
            "prompt_summary": "variables, n_obs, sample std ddof=1 rounded to 3 decimals, sample correlation matrix rounded to 3 decimals",
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
            work_key = f"panel=observational|method={METHOD}|level={level_id}|seed={seed}|model={model_id}"
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
                submission, scores, model_usage, diagnostics = run_corr_obs_llm(
                    instance=instance,
                    model_id=model_id,
                    openai_client=openai_client,
                    anthropic_api_key=anthropic_api_key,
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
