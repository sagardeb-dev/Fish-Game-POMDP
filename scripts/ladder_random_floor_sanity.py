"""Compare candidate ladders against the blind-random directed-F1 floor.

This is a diagnostic only. It does not modify the benchmark ladder, the SCM,
or the paper. Random submissions use only d, then canonical scoring evaluates
them against generated benchmark instances.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from causal_discovery import build_benchmark_instance, score_submission  # noqa: E402
from run_ladder import LevelSpec, config_from_level  # noqa: E402
from run_random_dag_baseline import make_random_uniform_submission  # noqa: E402


V0_LIKE_DENSITY_PROBE = {
    0: LevelSpec(0, d=4, k=3, n_obs=50, n_int=25, noise_var=0.5, budget_slack=2),
    1: LevelSpec(1, d=6, k=6, n_obs=25, n_int=15, noise_var=1.0, budget_slack=1),
    2: LevelSpec(2, d=8, k=9, n_obs=25, n_int=15, noise_var=1.0, budget_slack=1),
    3: LevelSpec(3, d=10, k=12, n_obs=25, n_int=15, noise_var=1.0, budget_slack=1),
    4: LevelSpec(4, d=12, k=14, n_obs=25, n_int=15, noise_var=1.0, budget_slack=1),
    5: LevelSpec(5, d=10, k=12, n_obs=15, n_int=10, noise_var=1.5, budget_slack=0),
}


D7_GRID = {
    0: LevelSpec(0, d=7, k=7, n_obs=50, n_int=15, noise_var=1.0, budget_slack=1),
    1: LevelSpec(1, d=7, k=7, n_obs=25, n_int=15, noise_var=1.0, budget_slack=1),
    2: LevelSpec(2, d=7, k=7, n_obs=15, n_int=15, noise_var=1.0, budget_slack=1),
    3: LevelSpec(3, d=7, k=6, n_obs=50, n_int=15, noise_var=1.0, budget_slack=1),
    4: LevelSpec(4, d=7, k=6, n_obs=25, n_int=15, noise_var=1.0, budget_slack=1),
    5: LevelSpec(5, d=7, k=6, n_obs=15, n_int=15, noise_var=1.0, budget_slack=1),
}


LEVEL_SETS = {
    "v0_like_density_probe": V0_LIKE_DENSITY_PROBE,
    "d7_grid": D7_GRID,
}

LEVEL_SET_SEED_NAMESPACE = {
    "v0_like_density_probe": 101,
    "d7_grid": 202,
}


def max_edges(d: int) -> int:
    return d * (d - 1) // 2


def exact_expected_directed_f1(d: int, k: int) -> float:
    """Exact discrete marginal under the uniform-m random DAG policy."""
    m_max = max_edges(d)
    total = 0.0
    for m in range(m_max + 1):
        if m + k > 0:
            total += k * m / (m_max * (m + k))
    return total / (m_max + 1)


def midpoint_envelope(d: int, k: int) -> float:
    rho = k / max_edges(d)
    return rho / (1.0 + 2.0 * rho)


def selected_level_sets(name: str) -> dict[str, dict[int, LevelSpec]]:
    if name == "both":
        return LEVEL_SETS
    return {name: LEVEL_SETS[name]}


def accepted_seeds(
    levels: dict[int, LevelSpec],
    *,
    instances_per_level: int,
    seed_offset: int,
    max_candidates_per_level: int,
) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for level_id, spec in sorted(levels.items()):
        cfg = config_from_level(spec)
        seeds: list[int] = []
        for candidate_idx in range(max_candidates_per_level):
            seed = seed_offset + level_id * 100_000 + candidate_idx
            try:
                build_benchmark_instance(cfg, np.random.default_rng(seed))
            except RuntimeError:
                continue
            seeds.append(seed)
            if len(seeds) >= instances_per_level:
                break
        if len(seeds) < instances_per_level:
            raise RuntimeError(
                f"Only found {len(seeds)} accepted seeds for level {level_id}; "
                f"needed {instances_per_level}"
            )
        out[level_id] = seeds
    return out


def run_diagnostic(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "level_set": args.level_set,
        "instances_per_level": args.instances_per_level,
        "samples_per_instance": args.samples_per_instance,
        "instance_seed_offset": args.instance_seed_offset,
        "baseline_seed": args.baseline_seed,
        "max_candidates_per_level": args.max_candidates_per_level,
        "levels": {},
        "seed_map": {},
    }

    for set_name, levels in selected_level_sets(args.level_set).items():
        seed_map = accepted_seeds(
            levels,
            instances_per_level=args.instances_per_level,
            seed_offset=args.instance_seed_offset,
            max_candidates_per_level=args.max_candidates_per_level,
        )
        manifest["levels"][set_name] = {
            str(level_id): asdict(spec) for level_id, spec in sorted(levels.items())
        }
        manifest["seed_map"][set_name] = {str(level_id): seeds for level_id, seeds in seed_map.items()}

        for level_id, spec in sorted(levels.items()):
            cfg = config_from_level(spec)
            m_max = max_edges(spec.d)
            rho = spec.k / m_max
            expected_f1 = exact_expected_directed_f1(spec.d, spec.k)
            midpoint_f1 = midpoint_envelope(spec.d, spec.k)

            for instance_index, seed in enumerate(seed_map[level_id]):
                instance = build_benchmark_instance(cfg, np.random.default_rng(seed))
                for sample_index in range(args.samples_per_instance):
                    rng = np.random.default_rng(
                        np.random.SeedSequence(
                            [
                                int(args.baseline_seed),
                                LEVEL_SET_SEED_NAMESPACE[set_name],
                                int(level_id),
                                int(seed),
                                int(sample_index),
                            ]
                        )
                    )
                    submission = make_random_uniform_submission(spec.d, rng)
                    score = score_submission(instance, submission)
                    rows.append(
                        {
                            "level_set": set_name,
                            "level": level_id,
                            "instance_index": instance_index,
                            "seed": seed,
                            "sample_index": sample_index,
                            "d": spec.d,
                            "k": spec.k,
                            "M": m_max,
                            "rho": rho,
                            "n_obs": spec.n_obs,
                            "n_int": spec.n_int,
                            "noise_var": spec.noise_var,
                            "budget_slack": spec.budget_slack,
                            "sampled_edges": len(submission.directed_edges),
                            "expected_f1": expected_f1,
                            "midpoint_envelope": midpoint_f1,
                            "directed_f1": float(score.directed_f1),
                            "dag_shd": float(score.dag_shd),
                        }
                    )

    long = pd.DataFrame(rows)
    summary = summarize(long, monotonic_tolerance=args.monotonic_tolerance)
    return long, summary, manifest


def summarize(long: pd.DataFrame, *, monotonic_tolerance: float) -> pd.DataFrame:
    summary = (
        long.groupby(
            [
                "level_set",
                "level",
                "d",
                "k",
                "M",
                "rho",
                "n_obs",
                "n_int",
                "noise_var",
                "budget_slack",
            ],
            as_index=False,
        )
        .agg(
            expected_f1=("expected_f1", "first"),
            midpoint_envelope=("midpoint_envelope", "first"),
            measured_f1_mean=("directed_f1", "mean"),
            measured_f1_std=("directed_f1", "std"),
            measured_shd_mean=("dag_shd", "mean"),
            sampled_edges_mean=("sampled_edges", "mean"),
            n=("directed_f1", "count"),
        )
    )
    summary["measured_f1_se"] = summary["measured_f1_std"] / np.sqrt(summary["n"])
    summary["gap_measured_minus_expected"] = summary["measured_f1_mean"] - summary["expected_f1"]
    summary["monotone_violation"] = False

    for set_name, group in summary.groupby("level_set", sort=False):
        prev_f1: float | None = None
        for idx, row in group.sort_values("level").iterrows():
            current = float(row["measured_f1_mean"])
            if prev_f1 is not None and current > prev_f1 + monotonic_tolerance:
                summary.loc[idx, "monotone_violation"] = True
            prev_f1 = current
    return summary


def print_report(summary: pd.DataFrame) -> None:
    for set_name, group in summary.groupby("level_set", sort=False):
        print()
        print(f"{set_name}")
        print(
            "level  d  k   rho n_obs n_int  expected  midpoint  measured      gap      shd  jump"
        )
        print(
            "--------------------------------------------------------------------------------"
        )
        for _, row in group.sort_values("level").iterrows():
            jump = "yes" if bool(row["monotone_violation"]) else ""
            print(
                f"{int(row['level']):>5} "
                f"{int(row['d']):>2} "
                f"{int(row['k']):>2} "
                f"{float(row['rho']):>5.3f} "
                f"{int(row['n_obs']):>5} "
                f"{int(row['n_int']):>5} "
                f"{float(row['expected_f1']):>9.4f} "
                f"{float(row['midpoint_envelope']):>9.4f} "
                f"{float(row['measured_f1_mean']):>9.4f} "
                f"{float(row['gap_measured_minus_expected']):>8.4f} "
                f"{float(row['measured_shd_mean']):>8.3f} "
                f"{jump:>5}"
            )
        if bool(group["monotone_violation"].any()):
            print("Random-floor curve has upward jumps beyond tolerance.")
        else:
            print("Random-floor curve is monotone non-increasing within tolerance.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare candidate ladders against the exact blind-random directed-F1 floor."
    )
    parser.add_argument(
        "--level-set",
        choices=("v0_like_density_probe", "d7_grid", "both"),
        default="both",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--instances-per-level", type=int, default=16)
    parser.add_argument("--samples-per-instance", type=int, default=100)
    parser.add_argument("--instance-seed-offset", type=int, default=20260427)
    parser.add_argument("--baseline-seed", type=int, default=12345)
    parser.add_argument("--max-candidates-per-level", type=int, default=50_000)
    parser.add_argument("--monotonic-tolerance", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.instances_per_level <= 0:
        raise ValueError("--instances-per-level must be positive")
    if args.samples_per_instance <= 0:
        raise ValueError("--samples-per-instance must be positive")

    long, summary, manifest = run_diagnostic(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.out_dir / "results_long.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    (args.out_dir / "sanity_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print_report(summary)
    print()
    print(f"[done] long={args.out_dir / 'results_long.csv'}")
    print(f"[done] summary={args.out_dir / 'summary.csv'}")
    print(f"[done] manifest={args.out_dir / 'sanity_manifest.json'}")


if __name__ == "__main__":
    main()
