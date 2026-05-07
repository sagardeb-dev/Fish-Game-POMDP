"""Print paired significance checks used in the ACDB v0 paper.

The script is intentionally read-only: it loads the two ladder long CSVs,
deduplicates resumed GPT rows by keeping the latest attempt, and prints the
small set of comparisons cited in the draft.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GPT_LONG = ROOT / "traces" / "ladder" / "full_ladder_toolcall_run1" / "results_long.csv"
SONNET_LONG = ROOT / "traces" / "ladder" / "sonnet46_full_ladder_run1" / "results_long.csv"
DEDUP_KEYS = ["level", "seed", "panel", "method", "model"]


def load_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["status"] == "success"].copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    return df.sort_values("timestamp_utc").drop_duplicates(DEDUP_KEYS, keep="last")


def signflip_pvalue(diffs: np.ndarray, *, iterations: int = 200_000) -> float:
    """Two-sided paired sign-flip p-value for mean difference."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    observed = abs(float(np.mean(diffs)))
    n = len(diffs)
    if n == 0:
        return float("nan")
    if n <= 20:
        count = 0
        total = 1 << n
        for mask in range(total):
            signs = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(n)])
            count += abs(float(np.mean(signs * diffs))) >= observed - 1e-15
        return count / total

    rng = np.random.default_rng(12345)
    count = 0
    done = 0
    batch = 10_000
    while done < iterations:
        size = min(batch, iterations - done)
        signs = rng.choice([-1.0, 1.0], size=(size, n))
        values = np.abs(np.mean(signs * diffs, axis=1))
        count += int(np.sum(values >= observed - 1e-15))
        done += size
    return count / iterations


def subset(
    df: pd.DataFrame,
    *,
    panel: str,
    method: str,
    model: str,
    levels: set[int] | None = None,
) -> pd.DataFrame:
    out = df[(df["panel"] == panel) & (df["method"] == method) & (df["model"] == model)].copy()
    if levels is not None:
        out = out[out["level"].isin(levels)]
    return out


def paired_report(name: str, left: pd.DataFrame, right: pd.DataFrame, metric: str) -> str:
    merged = left[["level", "seed", metric]].merge(
        right[["level", "seed", metric]],
        on=["level", "seed"],
        suffixes=("_left", "_right"),
    )
    diffs = merged[f"{metric}_left"] - merged[f"{metric}_right"]
    scale = 100.0 if metric.startswith(("skeleton_", "directed_", "compelled_")) else 1.0
    return (
        f"{name},{metric},n={len(merged)},"
        f"left={merged[f'{metric}_left'].mean() * scale:.3f},"
        f"right={merged[f'{metric}_right'].mean() * scale:.3f},"
        f"diff={diffs.mean() * scale:.3f},"
        f"p_two_sided={signflip_pvalue(diffs.to_numpy()):.5f}"
    )


def main() -> None:
    gpt = load_long(GPT_LONG)
    sonnet = load_long(SONNET_LONG)
    df = (
        pd.concat([gpt, sonnet], ignore_index=True)
        .sort_values("timestamp_utc")
        .drop_duplicates(DEDUP_KEYS, keep="last")
    )

    pc_active = subset(df, panel="active", method="pc_greedy", model="baseline")
    sonnet_raw_active = subset(df, panel="active", method="llm_raw", model="claude-sonnet-4-6")
    gpt_raw_active = subset(df, panel="active", method="llm_raw", model="gpt-5.4")
    sonnet_raw_obs = subset(
        df, panel="observational", method="llm_raw_obs", model="claude-sonnet-4-6"
    )
    gpt_raw_obs = subset(df, panel="observational", method="llm_raw_obs", model="gpt-5.4")

    reports = [
        paired_report("pc_greedy_minus_sonnet_raw_active", pc_active, sonnet_raw_active, "directed_f1"),
        paired_report("pc_greedy_minus_sonnet_raw_active", pc_active, sonnet_raw_active, "dag_shd"),
        paired_report("sonnet_raw_minus_gpt_raw_active", sonnet_raw_active, gpt_raw_active, "directed_f1"),
        paired_report("sonnet_raw_minus_gpt_raw_obs", sonnet_raw_obs, gpt_raw_obs, "directed_f1"),
        paired_report(
            "sonnet_raw_obs_minus_active_L2",
            subset(df, panel="observational", method="llm_raw_obs", model="claude-sonnet-4-6", levels={2}),
            subset(df, panel="active", method="llm_raw", model="claude-sonnet-4-6", levels={2}),
            "directed_f1",
        ),
        paired_report(
            "sonnet_raw_obs_minus_active_L5",
            subset(df, panel="observational", method="llm_raw_obs", model="claude-sonnet-4-6", levels={5}),
            subset(df, panel="active", method="llm_raw", model="claude-sonnet-4-6", levels={5}),
            "directed_f1",
        ),
    ]
    print("\n".join(reports))


if __name__ == "__main__":
    main()
