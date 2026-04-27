"""Run only PC and PC+greedy on an existing ladder manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_ladder as rl  # noqa: E402


LONG_FIELDS = (
    "level",
    "seed",
    "panel",
    "method",
    "d",
    "k",
    "skeleton_precision",
    "skeleton_recall",
    "skeleton_f1",
    "directed_precision",
    "directed_recall",
    "directed_f1",
    "dag_shd",
    "efficiency",
    "submit_directed",
    "submit_undirected",
    "interventions_used",
)

SUMMARY_FIELDS = (
    "level",
    "panel",
    "method",
    "n",
    "skeleton_f1",
    "directed_precision",
    "directed_recall",
    "directed_f1",
    "dag_shd",
    "efficiency",
    "submit_directed",
    "submit_undirected",
    "interventions_used",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PC-only ladder from a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(row["level"], row["panel"], row["method"]) for row in rows})
    keys.extend(
        ("overall", panel, method)
        for panel, method in sorted({(row["panel"], row["method"]) for row in rows})
    )
    for level, panel, method in keys:
        subset = [
            row
            for row in rows
            if row["panel"] == panel
            and row["method"] == method
            and (level == "overall" or row["level"] == level)
        ]
        out.append(
            {
                "level": level,
                "panel": panel,
                "method": method,
                "n": len(subset),
                "skeleton_f1": mean([row["skeleton_f1"] for row in subset]),
                "directed_precision": mean([row["directed_precision"] for row in subset]),
                "directed_recall": mean([row["directed_recall"] for row in subset]),
                "directed_f1": mean([row["directed_f1"] for row in subset]),
                "dag_shd": mean([row["dag_shd"] for row in subset]),
                "efficiency": mean([row["efficiency"] for row in subset]),
                "submit_directed": mean([row["submit_directed"] for row in subset]),
                "submit_undirected": mean([row["submit_undirected"] for row in subset]),
                "interventions_used": mean([row["interventions_used"] for row in subset]),
            }
        )
    return out


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    print(
        f"{'level':>7} {'panel':>13} {'method':>10} {'n':>3} "
        f"{'skel_f1':>8} {'dir_p':>8} {'dir_r':>8} {'dir_f1':>8} "
        f"{'shd':>8} {'eff':>8} {'dir_edges':>9} {'undir':>7} {'ints':>6}"
    )
    print("-" * 125)
    for row in rows:
        print(
            f"{str(row['level']):>7} {row['panel']:>13} {row['method']:>10} {row['n']:>3} "
            f"{100 * row['skeleton_f1']:>8.1f} "
            f"{100 * row['directed_precision']:>8.1f} "
            f"{100 * row['directed_recall']:>8.1f} "
            f"{100 * row['directed_f1']:>8.1f} "
            f"{row['dag_shd']:>8.3f} "
            f"{100 * row['efficiency']:>8.1f} "
            f"{row['submit_directed']:>9.2f} "
            f"{row['submit_undirected']:>7.2f} "
            f"{row['interventions_used']:>6.2f}"
        )


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    levels_catalog = rl.ladder_levels()
    levels = [int(x) for x in manifest["levels"]]
    seed_map = {int(k): [int(x) for x in v] for k, v in manifest["seed_map"].items()}
    alpha = float(manifest.get("alpha", 0.05))

    rows: list[dict[str, Any]] = []
    for level_id in levels:
        level = levels_catalog[level_id]
        cfg = rl.config_from_level(level)
        for seed in seed_map[level_id]:
            instance = rl.build_benchmark_instance(cfg, np.random.default_rng(seed))
            runtime_seed = int(seed * 10_000 + level_id * 101 + 7)
            for panel, method, runner in (
                ("observational", "pc", rl.run_pc_observational),
                ("active", "pc_greedy", rl.run_pc_greedy_active),
            ):
                submission, scores = runner(instance, runtime_seed, alpha)
                rows.append(
                    {
                        "level": level_id,
                        "seed": seed,
                        "panel": panel,
                        "method": method,
                        "d": level.d,
                        "k": level.k,
                        "skeleton_precision": scores.skeleton_precision,
                        "skeleton_recall": scores.skeleton_recall,
                        "skeleton_f1": scores.skeleton_f1,
                        "directed_precision": scores.directed_precision,
                        "directed_recall": scores.directed_recall,
                        "directed_f1": scores.directed_f1,
                        "dag_shd": scores.dag_shd,
                        "efficiency": scores.efficiency,
                        "submit_directed": len(submission.directed_edges),
                        "submit_undirected": len(submission.undirected_edges),
                        "interventions_used": submission.interventions_used,
                    }
                )

    summary = summarize(rows)
    long_csv = out_dir / "results_pc_only_long.csv"
    summary_csv = out_dir / "results_pc_only_summary.csv"
    write_csv(long_csv, LONG_FIELDS, rows)
    write_csv(summary_csv, SUMMARY_FIELDS, summary)

    print(f"manifest={manifest_path}")
    print(f"alpha={alpha} rows={len(rows)}")
    print_summary(summary)
    print(f"[done] long={long_csv}")
    print(f"[done] summary={summary_csv}")


if __name__ == "__main__":
    main()
