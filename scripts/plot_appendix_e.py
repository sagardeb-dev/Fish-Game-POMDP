"""Generate Appendix E diagnostic figures from the aggregated per-cell CSV.

Two figures:
- fig_e_score_axis_disagreement: dir_f1 vs comp_f1 per (level, method) per model
- fig_e_gap_above_floor_by_level: gap-above-random-floor bars per level per method, per model
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "traces" / "aggregated" / "per_model_per_level_per_method.csv"
OUT_DIR = ROOT / "research" / "figures" / "generated"

COLORS = {
    "ink": "#222222",
    "muted": "#707070",
    "pc_greedy": "#2f4f4f",
    "llm_raw": "#2b6cb0",
    "llm_stats": "#b45309",
    "pc_cpdag_llm": "#2e7d32",
    "llm_stats_cpdag_greedy": "#8b5fbf",
}

METHOD_LABEL = {
    "pc_greedy": "pc_greedy",
    "llm_raw": "llm_raw",
    "llm_stats": "llm_stats",
    "pc_cpdag_llm": "pc_cpdag_llm",
    "llm_stats_cpdag_greedy": "llm_stats_cpdag_greedy",
}

ACTIVE_METHODS = ["pc_greedy", "llm_raw", "llm_stats", "pc_cpdag_llm", "llm_stats_cpdag_greedy"]
MODELS_ORDER = ["GPT-5.5", "GPT-5.4-mini", "Sonnet-4.6", "Haiku-4.5", "Gemini-3-Flash"]
LEVEL_MARKERS = {0: "o", 1: "s", 2: "D", 3: "^", 4: "v", 5: "P"}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": COLORS["muted"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def fig_score_axis_disagreement(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(11.0, 2.6), sharex=True, sharey=True)
    for ax, model in zip(axes, MODELS_ORDER):
        sub = df[(df["panel_label"] == model) & (df["method"].isin(ACTIVE_METHODS)) & (df["level"] >= 1)]
        for method in ACTIVE_METHODS:
            ms = sub[sub["method"] == method]
            for _, r in ms.iterrows():
                ax.scatter(
                    r["directed_f1_mean"],
                    r["compelled_f1_mean"],
                    color=COLORS[method],
                    marker=LEVEL_MARKERS[int(r["level"])],
                    s=28,
                    edgecolor="white",
                    linewidth=0.4,
                    alpha=0.9,
                )
        ax.plot([0, 1], [0, 1], color=COLORS["muted"], lw=0.6, linestyle=":", zorder=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(model, fontsize=8.5)
        ax.set_xlabel(r"directed $F_1$")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"compelled-edge $F_1$")

    method_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[m], markersize=6, label=METHOD_LABEL[m])
        for m in ACTIVE_METHODS
    ]
    level_handles = [
        plt.Line2D([0], [0], marker=LEVEL_MARKERS[lvl], color=COLORS["muted"], markersize=5, linestyle="none", label=f"L{lvl}")
        for lvl in range(1, 6)
    ]
    fig.legend(
        handles=method_handles + level_handles,
        loc="lower center",
        ncol=10,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = OUT_DIR / "fig_e_score_axis_disagreement.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    print(f"[done] {out}")


def fig_gap_above_floor_by_level(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(11.0, 2.8), sharey=True)
    levels = [1, 2, 3, 4, 5]
    n_methods = len(ACTIVE_METHODS)
    bar_w = 0.15

    for ax, model in zip(axes, MODELS_ORDER):
        sub = df[(df["panel_label"] == model) & (df["method"].isin(ACTIVE_METHODS)) & (df["level"].isin(levels))]
        x = np.arange(len(levels))
        for i, method in enumerate(ACTIVE_METHODS):
            vals = []
            for lvl in levels:
                row = sub[(sub["method"] == method) & (sub["level"] == lvl)]
                vals.append(float(row["gap_above_floor"].iloc[0]) if not row.empty else np.nan)
            offset = (i - (n_methods - 1) / 2) * bar_w
            ax.bar(x + offset, vals, width=bar_w, color=COLORS[method], label=METHOD_LABEL[method], edgecolor="white", linewidth=0.3)
        ax.axhline(0, color=COLORS["ink"], lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in levels])
        ax.set_title(model, fontsize=8.5)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel(r"$F_1 - \mathrm{floor}$")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS[m], label=METHOD_LABEL[m])
        for m in ACTIVE_METHODS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = OUT_DIR / "fig_e_gap_above_floor_by_level.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    print(f"[done] {out}")


def main() -> None:
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC)
    fig_score_axis_disagreement(df)
    fig_gap_above_floor_by_level(df)


if __name__ == "__main__":
    main()
