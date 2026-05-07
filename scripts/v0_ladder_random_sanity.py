"""Sanity check: does the v0 ladder produce monotone difficulty for the
uniform-m random baseline?

For each v0 level (run_ladder.ladder_levels()), build N benchmark instances,
draw R blind random DAG submissions per instance (m drawn uniform on
{0, ..., M}, M = d(d-1)/2), and score each submission with the canonical
score_submission(). Compare the mean measured directed F1 to the analytic

  E[F1] = (1 / (M + 1)) * sum_{m=0}^{M}  k * m / [M * (m + k)].

If measured F1 is monotone non-increasing in level (or matches analytic
within a few pp at every level), v0's difficulty ordering for the random
baseline is real. If not, that flags a confound in the level definitions.

This script is read-only: it imports existing utilities, does not modify
run_ladder.py, src/, or the paper. It writes one CSV under traces/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from causal_discovery import build_benchmark_instance, score_submission  # noqa: E402
from run_ladder import config_from_level, ladder_levels  # noqa: E402
from run_random_dag_baseline import make_random_uniform_submission  # noqa: E402


OUTPUT_DIR = ROOT / "traces" / "v0_random_sanity"
N_INSTANCES_PER_LEVEL = 16
N_RANDOM_DRAWS_PER_INSTANCE = 32
INSTANCE_SEED_OFFSET = 0
DRAW_SEED_OFFSET = 1_000_000


def analytic_expected_f1(d: int, k: int) -> float:
    """Exact discrete marginal under uniform-m random policy."""
    m_max = d * (d - 1) // 2
    total = 0.0
    for m in range(m_max + 1):
        if m + k == 0:
            continue
        total += k * m / (m_max * (m + k))
    return total / (m_max + 1)


def run() -> pd.DataFrame:
    rows: list[dict] = []
    for level_id, spec in sorted(ladder_levels().items()):
        cfg = config_from_level(spec)
        m_max = spec.d * (spec.d - 1) // 2
        e_f1 = analytic_expected_f1(spec.d, spec.k)
        print(f"Level {level_id}: d={spec.d} k={spec.k} M={m_max} "
              f"rho={spec.k/m_max:.3f} E[F1]={e_f1:.4f}")

        for inst_idx in range(N_INSTANCES_PER_LEVEL):
            inst_seed = INSTANCE_SEED_OFFSET + inst_idx + level_id * 10_000
            try:
                instance = build_benchmark_instance(
                    cfg, np.random.default_rng(inst_seed)
                )
            except RuntimeError as exc:
                print(f"  inst {inst_idx} seed={inst_seed} REJECTED: {exc}")
                continue

            for draw_idx in range(N_RANDOM_DRAWS_PER_INSTANCE):
                draw_seed = DRAW_SEED_OFFSET + draw_idx + inst_idx * 1000 + level_id * 10_000_000
                draw_rng = np.random.default_rng(draw_seed)
                submission = make_random_uniform_submission(spec.d, draw_rng)
                score = score_submission(instance, submission)
                rows.append({
                    "level": level_id,
                    "d": spec.d,
                    "k": spec.k,
                    "m_max": m_max,
                    "rho": spec.k / m_max,
                    "instance_seed": inst_seed,
                    "draw_seed": draw_seed,
                    "submitted_edges": len(submission.directed_edges),
                    "directed_f1": float(score.directed_f1),
                    "shd": float(score.dag_shd),
                    "expected_f1_analytic": e_f1,
                })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["level", "d", "k", "m_max", "rho"], as_index=False)
        .agg(
            measured_f1_mean=("directed_f1", "mean"),
            measured_f1_std=("directed_f1", "std"),
            measured_shd_mean=("shd", "mean"),
            expected_f1=("expected_f1_analytic", "first"),
            n=("directed_f1", "count"),
        )
    )
    summary["gap"] = summary["measured_f1_mean"] - summary["expected_f1"]
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = run()
    df.to_csv(OUTPUT_DIR / "results_long.csv", index=False)
    summary = summarize(df)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    print()
    print("Per-level summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    f1s = summary["measured_f1_mean"].tolist()
    monotone = all(f1s[i] >= f1s[i + 1] for i in range(len(f1s) - 1))
    print(f"Measured F1 monotone non-increasing across levels: {monotone}")
    if not monotone:
        breaks = [(i, i + 1, f1s[i], f1s[i + 1])
                  for i in range(len(f1s) - 1) if f1s[i] < f1s[i + 1]]
        for a, b, fa, fb in breaks:
            print(f"  L{a}->L{b} jumps UP: {fa:.4f} -> {fb:.4f}")


if __name__ == "__main__":
    main()
