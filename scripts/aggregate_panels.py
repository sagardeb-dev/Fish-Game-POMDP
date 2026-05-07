"""Aggregate per-panel results_long.csv into a single per-(panel, level, method, model) CSV for Appendix E.

Reads the five canonical model panels, dedupes by (level, seed, panel, method, model)
keeping latest by timestamp (matches make_paper_figures.py:load_long), restricts to
status==success and panel==active, then computes mean and SE per cell. Adds the
random-floor closed form and gap-above-floor for directed_f1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "traces" / "aggregated" / "per_model_per_level_per_method.csv"

PANELS = {
    "GPT-5.5": "full_gpt55",
    "GPT-5.4-mini": "full_gpt54mini",
    "Sonnet-4.6": "full_sonnet46_or",
    "Haiku-4.5": "full_haiku45_or",
    "Gemini-3-Flash": "full_gemini3flash",
}

LADDER_DK = {0: (4, 3), 1: (6, 6), 2: (8, 9), 3: (10, 12), 4: (12, 14), 5: (14, 16)}

DEDUP_KEYS = ["level", "seed", "panel", "method", "model"]

METRICS = [
    "skeleton_f1",
    "compelled_f1",
    "directed_f1",
    "dag_shd",
    "cost_usd",
    "total_actions",
    "intervene_actions",
    "stats_actions",
]


def expected_f1(d: int, k: int) -> float:
    """Closed-form E[directed F1] under the uniform-m random DAG sampler (Appendix C)."""
    M = d * (d - 1) // 2
    total = 0.0
    for m in range(M + 1):
        if m + k > 0:
            total += k * m / (M * (m + k))
    return total / (M + 1)


FLOOR = {lvl: expected_f1(d, k) for lvl, (d, k) in LADDER_DK.items()}


def load_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["status"] == "success"].copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    return df.sort_values("timestamp_utc").drop_duplicates(DEDUP_KEYS, keep="last")


def aggregate_panel(panel_label: str, panel_dir: str) -> pd.DataFrame:
    df = load_long(ROOT / "traces" / "ladder" / panel_dir / "results_long.csv")
    df = df[df["panel"] == "active"].copy()

    rows = []
    for (level, method), grp in df.groupby(["level", "method"]):
        n = len(grp)
        d, k = LADDER_DK[level]
        row = {
            "panel_label": panel_label,
            "panel_dir": panel_dir,
            "level": level,
            "d": d,
            "k": k,
            "method": method,
            "n": n,
            "floor_dir_f1": FLOOR[level],
        }
        for m in METRICS:
            vals = grp[m].dropna().values
            if len(vals) == 0:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_se"] = np.nan
            else:
                row[f"{m}_mean"] = float(np.mean(vals))
                row[f"{m}_se"] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        row["gap_above_floor"] = row["directed_f1_mean"] - FLOOR[level]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    parts = [aggregate_panel(label, d) for label, d in PANELS.items()]
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["panel_label", "level", "method"]).reset_index(drop=True)
    out.to_csv(OUT, index=False, float_format="%.6f")
    print(f"[done] {OUT}  ({len(out)} rows)")
    print()
    print("Per-level random-floor (closed form):")
    for lvl, (d, k) in LADDER_DK.items():
        print(f"  L{lvl}: d={d}, k={k}, floor={FLOOR[lvl]:.4f}")


if __name__ == "__main__":
    main()
