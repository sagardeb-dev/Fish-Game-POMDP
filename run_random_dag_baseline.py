"""Blind random-uniform DAG sanity baseline for ladder runs.

Purpose:
- Estimate whether LLM graph submissions beat blind random graph guessing.
- Reuse an existing ladder run_manifest.json so the benchmark instances match
  the GPT/Sonnet runs exactly by (level, seed).
- Generate random DAG submissions using only public metadata: d.
- Sample edge count uniformly from 0..d(d-1)/2; this baseline does not know k.

Non-goals / anti-leakage:
- Does not call LLMs.
- Does not call PC.
- Does not call observe().
- Does not call intervene().
- Does not inspect the CPDAG or true DAG before submission.
- Does not use the true edge count k for generation.
- Uses the true graph only indirectly through the existing score_submission()
  function after a random graph has already been submitted.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_discovery import (  # noqa: E402
    GraphSubmission,
    build_benchmark_instance,
    make_v1_config,
    score_submission,
)
from run_ladder import config_from_level, ladder_levels  # noqa: E402


@dataclass(frozen=True, slots=True)
class RandomBaselineLevelSpec:
    level_id: int
    purpose: str
    d: int
    k: int
    n_obs: int
    n_int: int
    noise_var: float
    budget_slack: int


LONG_FIELDS = (
    "run_id",
    "timestamp_utc",
    "method",
    "model",
    "level",
    "seed",
    "sample_index",
    "baseline_seed",
    "d",
    "k",
    "max_edges",
    "density",
    "true_edges",
    "cpdag_directed",
    "cpdag_undirected",
    "opt_set_size",
    "budget",
    "interventions_used",
    "sampled_edges",
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

SUMMARY_FIELDS = (
    "level",
    "method",
    "model",
    "n",
    "d",
    "k",
    "max_edges",
    "density",
    "skeleton_precision",
    "skeleton_recall",
    "skeleton_f1",
    "directed_precision",
    "directed_recall",
    "directed_f1",
    "dag_shd",
    "efficiency",
)


def max_edges_for_d(d: int) -> int:
    return d * (d - 1) // 2


def density_for_d_k(d: int, k: int) -> float:
    return float(k / max_edges_for_d(d))


def density_probe_levels() -> dict[int, RandomBaselineLevelSpec]:
    return {
        0: RandomBaselineLevelSpec(
            0,
            purpose="tutorial only",
            d=4,
            k=3,
            n_obs=50,
            n_int=25,
            noise_var=0.5,
            budget_slack=2,
        ),
        1: RandomBaselineLevelSpec(
            1,
            purpose="small core",
            d=6,
            k=6,
            n_obs=25,
            n_int=15,
            noise_var=1.0,
            budget_slack=1,
        ),
        2: RandomBaselineLevelSpec(
            2,
            purpose="medium core",
            d=8,
            k=9,
            n_obs=25,
            n_int=15,
            noise_var=1.0,
            budget_slack=1,
        ),
        3: RandomBaselineLevelSpec(
            3,
            purpose="real scale",
            d=10,
            k=12,
            n_obs=25,
            n_int=15,
            noise_var=1.0,
            budget_slack=1,
        ),
        4: RandomBaselineLevelSpec(
            4,
            purpose="structural hard",
            d=12,
            k=14,
            n_obs=25,
            n_int=15,
            noise_var=1.0,
            budget_slack=1,
        ),
        5: RandomBaselineLevelSpec(
            5,
            purpose="noisy hard",
            d=10,
            k=12,
            n_obs=15,
            n_int=10,
            noise_var=1.5,
            budget_slack=0,
        ),
    }


def config_from_random_level(level: RandomBaselineLevelSpec):
    return make_v1_config(
        d=level.d,
        k=level.k,
        n_obs=level.n_obs,
        n_int=level.n_int,
        noise_var=level.noise_var,
        budget_slack=level.budget_slack,
    )


def make_random_uniform_submission(d: int, rng: np.random.Generator) -> GraphSubmission:
    """Create one blind random DAG submission without using the true edge count."""
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    order = tuple(int(node) for node in rng.permutation(d))
    possible_edges = tuple((order[i], order[j]) for i, j in combinations(range(d), 2))
    m = int(rng.integers(0, len(possible_edges) + 1))
    if m == 0:
        chosen_edges: tuple[tuple[int, int], ...] = ()
    else:
        chosen = rng.choice(len(possible_edges), size=m, replace=False)
        chosen_edges = tuple(possible_edges[int(idx)] for idx in np.sort(chosen))
    return GraphSubmission(
        num_nodes=d,
        directed_edges=frozenset(chosen_edges),
        undirected_edges=frozenset(),
        interventions_used=0,
    )


def _load_seed_map(manifest_path: Path) -> dict[int, list[int]]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_seed_map = data.get("seed_map")
    if not isinstance(raw_seed_map, dict):
        raise ValueError(f"Manifest missing seed_map: {manifest_path}")
    out: dict[int, list[int]] = {}
    for raw_level, raw_seeds in raw_seed_map.items():
        if not isinstance(raw_seeds, list):
            raise ValueError(f"Manifest seed_map[{raw_level!r}] is not a list")
        out[int(raw_level)] = [int(seed) for seed in raw_seeds]
    return out


def _seed_accepts(level: RandomBaselineLevelSpec, seed: int) -> bool:
    cfg = config_from_random_level(level)
    try:
        build_benchmark_instance(cfg, np.random.default_rng(seed))
    except RuntimeError:
        return False
    return True


def build_density_probe_seed_map(
    levels: dict[int, RandomBaselineLevelSpec],
    seeds_per_level: int,
    preflight_seed: int,
    max_candidates_per_level: int,
) -> dict[int, list[int]]:
    if seeds_per_level <= 0:
        raise ValueError("--seeds-per-level must be positive")
    if max_candidates_per_level <= 0:
        raise ValueError("--max-candidates-per-level must be positive")

    rng = np.random.default_rng(preflight_seed)
    out: dict[int, list[int]] = {}
    for level_id in sorted(levels):
        level = levels[level_id]
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


def _score_row(
    *,
    run_id: str,
    level_id: int,
    seed: int,
    sample_index: int,
    baseline_seed: int,
    instance: Any,
    submission: GraphSubmission,
) -> dict[str, Any]:
    scores = score_submission(instance, submission)
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": "random_uniform",
        "model": "baseline",
        "level": level_id,
        "seed": seed,
        "sample_index": sample_index,
        "baseline_seed": baseline_seed,
        "d": instance.config.d,
        "k": instance.config.k,
        "max_edges": max_edges_for_d(instance.config.d),
        "density": density_for_d_k(instance.config.d, instance.config.k),
        "true_edges": len(instance.true_dag.edges),
        "cpdag_directed": len(instance.observational_ceiling.directed_edges),
        "cpdag_undirected": len(instance.observational_ceiling.undirected_edges),
        "opt_set_size": len(instance.optimal_intervention_set),
        "budget": instance.intervention_budget,
        "interventions_used": submission.interventions_used,
        "sampled_edges": len(submission.directed_edges),
        "submit_directed": len(submission.directed_edges),
        "submit_undirected": len(submission.undirected_edges),
        "skeleton_precision": scores.skeleton_precision,
        "skeleton_recall": scores.skeleton_recall,
        "skeleton_f1": scores.skeleton_f1,
        "compelled_precision": scores.compelled_precision,
        "compelled_recall": scores.compelled_recall,
        "compelled_f1": scores.compelled_f1,
        "directed_precision": scores.directed_precision,
        "directed_recall": scores.directed_recall,
        "directed_f1": scores.directed_f1,
        "dag_shd": scores.dag_shd,
        "efficiency": scores.efficiency,
    }


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return float(sum(float(row[field]) for row in rows) / len(rows))


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["level"])].append(row)
        grouped["overall"].append(row)

    summary: list[dict[str, Any]] = []
    for level in sorted(grouped.keys(), key=lambda x: (x == "overall", int(x) if x.isdigit() else 0)):
        group = grouped[level]
        d_values = {int(row["d"]) for row in group}
        k_values = {int(row["k"]) for row in group}
        max_edge_values = {int(row["max_edges"]) for row in group}
        density_values = {float(row["density"]) for row in group}
        summary.append(
            {
                "level": level,
                "method": "random_uniform",
                "model": "baseline",
                "n": len(group),
                "d": next(iter(d_values)) if len(d_values) == 1 else "",
                "k": next(iter(k_values)) if len(k_values) == 1 else "",
                "max_edges": next(iter(max_edge_values)) if len(max_edge_values) == 1 else "",
                "density": next(iter(density_values)) if len(density_values) == 1 else "",
                "skeleton_precision": _mean(group, "skeleton_precision"),
                "skeleton_recall": _mean(group, "skeleton_recall"),
                "skeleton_f1": _mean(group, "skeleton_f1"),
                "directed_precision": _mean(group, "directed_precision"),
                "directed_recall": _mean(group, "directed_recall"),
                "directed_f1": _mean(group, "directed_f1"),
                "dag_shd": _mean(group, "dag_shd"),
                "efficiency": _mean(group, "efficiency"),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: list[dict[str, Any]]) -> None:
    print("level          n  skel_p  skel_r  skel_f1   dir_p   dir_r  dir_f1     shd")
    print("----------------------------------------------------------------------------")
    for row in summary:
        print(
            f"{str(row['level']):>7} {int(row['n']):>10} "
            f"{float(row['skeleton_precision']):>7.3f} "
            f"{float(row['skeleton_recall']):>7.3f} "
            f"{float(row['skeleton_f1']):>8.3f} "
            f"{float(row['directed_precision']):>7.3f} "
            f"{float(row['directed_recall']):>7.3f} "
            f"{float(row['directed_f1']):>7.3f} "
            f"{float(row['dag_shd']):>7.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a blind random-uniform DAG baseline on a manifest or density probe catalog."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to an existing ladder run_manifest.json.",
    )
    parser.add_argument(
        "--level-set",
        choices=("density_probe",),
        help="Built-in level set to generate and run without a ladder manifest.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for random baseline CSVs.",
    )
    parser.add_argument(
        "--samples-per-instance",
        type=int,
        default=100,
        help="Random-uniform DAG submissions per benchmark instance.",
    )
    parser.add_argument(
        "--baseline-seed",
        type=int,
        default=12345,
        help="Seed namespace for random baseline submissions.",
    )
    parser.add_argument(
        "--seeds-per-level",
        type=int,
        default=8,
        help="Accepted benchmark seeds per built-in level-set level.",
    )
    parser.add_argument(
        "--preflight-seed",
        type=int,
        default=20260422,
        help="Seed used to search for accepted benchmark seeds in built-in level-set mode.",
    )
    parser.add_argument(
        "--max-candidates-per-level",
        type=int,
        default=50000,
        help="Maximum accepted-seed candidates to try per built-in level-set level.",
    )
    return parser.parse_args()


def _run_random_baseline(
    *,
    seed_map: dict[int, list[int]],
    level_configs: dict[int, Any],
    samples_per_instance: int,
    baseline_seed: int,
    run_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level_id in sorted(seed_map):
        if level_id not in level_configs:
            raise ValueError(f"Unknown level {level_id}")
        cfg = level_configs[level_id]
        for seed in seed_map[level_id]:
            instance = build_benchmark_instance(cfg, np.random.default_rng(seed))
            for sample_index in range(samples_per_instance):
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [int(baseline_seed), int(level_id), int(seed), int(sample_index)]
                    )
                )
                submission = make_random_uniform_submission(instance.config.d, rng)
                rows.append(
                    _score_row(
                        run_id=run_id,
                        level_id=level_id,
                        seed=seed,
                        sample_index=sample_index,
                        baseline_seed=baseline_seed,
                        instance=instance,
                        submission=submission,
                    )
                )
    return rows


def _level_metadata(levels: dict[int, RandomBaselineLevelSpec]) -> dict[str, dict[str, Any]]:
    return {
        str(level_id): {
            "purpose": level.purpose,
            "d": level.d,
            "k": level.k,
            "max_edges": max_edges_for_d(level.d),
            "density": density_for_d_k(level.d, level.k),
            "n_obs": level.n_obs,
            "n_int": level.n_int,
            "noise_var": level.noise_var,
            "budget_slack": level.budget_slack,
        }
        for level_id, level in sorted(levels.items())
    }


def main() -> None:
    args = parse_args()
    if args.samples_per_instance <= 0:
        raise ValueError("--samples-per-instance must be positive")
    if (args.manifest is None) == (args.level_set is None):
        raise ValueError("Provide exactly one of --manifest or --level-set")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_payload: dict[str, Any]

    if args.manifest is not None:
        seed_map = _load_seed_map(args.manifest)
        levels = ladder_levels()
        level_configs = {}
        for level_id in sorted(seed_map):
            if level_id not in levels:
                raise ValueError(f"Manifest contains unknown ladder level {level_id}")
            level_configs[level_id] = config_from_level(levels[level_id])
        manifest_payload = {
            "input_manifest": str(args.manifest),
        }
    else:
        levels = density_probe_levels()
        seed_map = build_density_probe_seed_map(
            levels,
            args.seeds_per_level,
            args.preflight_seed,
            args.max_candidates_per_level,
        )
        level_configs = {
            level_id: config_from_random_level(level) for level_id, level in levels.items()
        }
        manifest_payload = {
            "level_set": args.level_set,
            "levels": sorted(levels),
            "level_config": _level_metadata(levels),
            "seeds_per_level": args.seeds_per_level,
            "preflight_seed": args.preflight_seed,
            "max_candidates_per_level": args.max_candidates_per_level,
            "seed_map": {str(k): v for k, v in seed_map.items()},
        }

    rows = _run_random_baseline(
        seed_map=seed_map,
        level_configs=level_configs,
        samples_per_instance=args.samples_per_instance,
        baseline_seed=args.baseline_seed,
        run_id=run_id,
    )

    summary = summarize_rows(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "results_random_dag_long.csv", rows, LONG_FIELDS)
    write_csv(args.out_dir / "results_random_dag_summary.csv", summary, SUMMARY_FIELDS)
    (args.out_dir / "random_dag_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "baseline_seed": args.baseline_seed,
                "samples_per_instance": args.samples_per_instance,
                "num_rows": len(rows),
                **manifest_payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print_summary(summary)
    print(f"\n[done] long={args.out_dir / 'results_random_dag_long.csv'}")
    print(f"[done] summary={args.out_dir / 'results_random_dag_summary.csv'}")
    print(f"[done] manifest={args.out_dir / 'random_dag_manifest.json'}")


if __name__ == "__main__":
    main()
