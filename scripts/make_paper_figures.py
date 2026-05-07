"""Generate ACDB paper figures from existing result artifacts.

This script is deterministic and read-only with respect to experiment outputs.
It writes PDF figures under research/figures/generated/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "research" / "figures" / "generated"
RESULTS_NOTE = ROOT / "docs" / "notes" / "RESULTS.md"

GPT_LONG = ROOT / "traces" / "ladder" / "full_ladder_toolcall_run1" / "results_long.csv"
SONNET_LONG = ROOT / "traces" / "ladder" / "sonnet46_full_ladder_run1" / "results_long.csv"
RANDOM_SUMMARY = (
    ROOT / "traces" / "ladder" / "random_uniform_baseline" / "results_random_dag_summary.csv"
)
DENSITY_SUMMARY = (
    ROOT / "traces" / "ladder" / "random_density_probe" / "results_random_dag_summary.csv"
)

DEDUP_KEYS = ["level", "seed", "panel", "method", "model"]


COLORS = {
    "ink": "#222222",
    "muted": "#707070",
    "light": "#f2f2f2",
    "mid": "#d9d9d9",
    "accent": "#2b6cb0",
    "accent2": "#b45309",
    "pc": "#2f4f4f",
    "llm": "#2b6cb0",
    "random": "#8a8a8a",
    "gpt": "#2b6cb0",
    "sonnet": "#c05a00",
    "gemini": "#4a7f32",
    "haiku": "#8b5fbf",
    "cpdag": "#2e7d32",
    "stats_greedy": "#8b5fbf",
}


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


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["status"] == "success"].copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    return df.sort_values("timestamp_utc").drop_duplicates(DEDUP_KEYS, keep="last")


def load_all_ladder() -> pd.DataFrame:
    return pd.concat([load_long(GPT_LONG), load_long(SONNET_LONG)], ignore_index=True)


def load_results_note() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    current_model: str | None = None
    current_level: str | None = None
    try:
        text = RESULTS_NOTE.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = RESULTS_NOTE.read_text(encoding="cp1252")
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("## "):
            current_model = line[3:].strip()
        elif line.startswith("### "):
            current_level = line[4:].strip().replace("–", "-")
        elif line.startswith("| Method |"):
            i += 2
            while i < len(lines):
                row = lines[i].strip()
                if not row.startswith("|"):
                    break
                parts = [part.strip() for part in row.strip("|").split("|")]
                if len(parts) == 5 and parts[0] != "oracle":
                    rows.append(
                        {
                            "model": current_model,
                            "level": current_level,
                            "method": parts[0],
                            "dir_f1": float(parts[1]),
                            "SHD": float(parts[2]),
                            "skel_f1": float(parts[3]),
                            "comp_f1": float(parts[4]),
                        }
                    )
                i += 1
            continue
        i += 1
    return pd.DataFrame(rows)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT_DIR / name, bbox_inches="tight")
    fig.savefig(OUT_DIR / name.replace(".pdf", ".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)


def rounded_box(ax, xy, width, height, text, *, facecolor, edgecolor=None, fontsize=8):
    edgecolor = edgecolor or COLORS["muted"]
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=0.8,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
    )
    return patch


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            color=COLORS["muted"],
        )
    )


def flow_arrow(ax, start, end, *, color=None, connectionstyle="arc3", mutation_scale=9):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=0.85,
            color=color or COLORS["muted"],
            connectionstyle=connectionstyle,
            shrinkA=0,
            shrinkB=0,
        )
    )


def flow_box(ax, center, width, height, text, *, facecolor=None, fontsize=7.3):
    x, y = center
    return rounded_box(
        ax,
        (x - width / 2, y - height / 2),
        width,
        height,
        text,
        facecolor=facecolor or COLORS["light"],
        fontsize=fontsize,
    )


def flow_diamond(ax, center, width, height, text, *, fontsize=7.1):
    x, y = center
    points = [
        (x, y + height / 2),
        (x + width / 2, y),
        (x, y - height / 2),
        (x - width / 2, y),
    ]
    patch = Polygon(points, closed=True, facecolor="#fff4e6", edgecolor=COLORS["muted"], linewidth=0.85)
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=COLORS["ink"])
    return patch


def flow_elbow_return(ax, points, *, color=None):
    color = color or COLORS["accent2"]
    xs, ys = zip(*points[:-1], strict=True)
    ax.plot(xs, ys, color=color, linewidth=0.85)
    flow_arrow(ax, points[-2], points[-1], color=color, mutation_scale=10)


def flow_reject_return(ax, gate_left, elbow_x, reentry, *, color=None):
    color = color or COLORS["accent2"]
    y = gate_left[1]
    flow_arrow(ax, gate_left, (elbow_x, y), color=color, mutation_scale=10)
    ax.plot([elbow_x, elbow_x], [y, reentry[1]], color=color, linewidth=0.85)
    flow_arrow(ax, (elbow_x, reentry[1]), reentry, color=color, mutation_scale=10)


def flow_label(ax, x, y, text, *, color=None, ha="center"):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va="center",
        fontsize=7.2,
        color=color or COLORS["muted"],
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.25},
        zorder=5,
    )


def draw_graph_panel(
    ax,
    title: str,
    *,
    directed_edges: list[tuple[str, str]],
    undirected_edges: list[tuple[str, str]] | None = None,
    highlighted_nodes: set[str] | None = None,
) -> None:
    positions = {
        "X1": (0.12, 0.50),
        "X2": (0.43, 0.78),
        "X3": (0.43, 0.22),
        "X4": (0.78, 0.50),
    }
    undirected_edges = undirected_edges or []
    highlighted_nodes = highlighted_nodes or set()

    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5)

    def edge_endpoint(source: str, target: str, shrink: float = 0.055):
        sx, sy = positions[source]
        tx, ty = positions[target]
        vec = np.array([tx - sx, ty - sy])
        dist = float(np.linalg.norm(vec))
        unit = vec / dist
        return (sx + unit[0] * shrink, sy + unit[1] * shrink), (
            tx - unit[0] * shrink,
            ty - unit[1] * shrink,
        )

    for u, v in undirected_edges:
        start, end = edge_endpoint(u, v)
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=COLORS["muted"],
            linewidth=1.1,
            linestyle=(0, (3, 2)),
            zorder=1,
        )

    for u, v in directed_edges:
        start, end = edge_endpoint(u, v)
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.1,
                color=COLORS["ink"],
                zorder=1,
            )
        )

    for node, (x, y) in positions.items():
        fill = "#e8f1fb" if node in highlighted_nodes else "white"
        edge = COLORS["accent"] if node in highlighted_nodes else COLORS["ink"]
        ax.add_patch(Circle((x, y), 0.055, facecolor=fill, edgecolor=edge, linewidth=1.1, zorder=3))
        ax.text(x, y, node, ha="center", va="center", fontsize=8, color=COLORS["ink"], zorder=4)


def fig_dag_cpdag_intervention() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.0))
    directed_true = [("X1", "X2"), ("X1", "X3"), ("X2", "X4"), ("X3", "X4")]
    directed_compelled = [("X2", "X4"), ("X3", "X4")]
    undirected_cpdag = [("X1", "X2"), ("X1", "X3")]

    draw_graph_panel(axes[0], "True DAG", directed_edges=directed_true)
    draw_graph_panel(
        axes[1],
        "Observational CPDAG",
        directed_edges=directed_compelled,
        undirected_edges=undirected_cpdag,
    )
    draw_graph_panel(
        axes[2],
        "After $do(X_1)$",
        directed_edges=directed_true,
        highlighted_nodes={"X1"},
    )

    fig.text(
        0.335,
        0.18,
        "observations\nonly",
        ha="center",
        va="center",
        fontsize=7.2,
        color=COLORS["muted"],
    )
    fig.text(
        0.665,
        0.18,
        "intervene\non $X_1$",
        ha="center",
        va="center",
        fontsize=7.2,
        color=COLORS["muted"],
    )
    fig.tight_layout(w_pad=1.7)
    save(fig, "fig_dag_cpdag_intervention.pdf")


def fig_instance_generation() -> None:
    fig, ax = plt.subplots(figsize=(4.8, 5.25))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    steps = [
        ("Sample DAG\n$d,k,seed$", 0.93, "box"),
        ("Compute\nCPDAG", 0.815, "box"),
        ("reject\ngraph?", 0.685, "diamond"),
        ("Parameterize\nSCM", 0.545, "box"),
        ("reject\nSCM?", 0.410, "diamond"),
        ("Compute optimal\nintervention set", 0.280, "box"),
        ("Permute labels\n$X_1,\\ldots,X_d$", 0.170, "box"),
        ("Sample data\n$D^{obs},D^{int}$", 0.060, "box"),
    ]
    for label, y, kind in steps:
        if kind == "diamond":
            flow_diamond(ax, (0.56, y), 0.34, 0.112, label, fontsize=7.4)
        else:
            fill = "#e8f1fb" if "Sample data" in label else COLORS["light"]
            flow_box(ax, (0.56, y), 0.50, 0.080, label, facecolor=fill, fontsize=7.5)

    for (_, y0, _), (_, y1, _) in zip(steps[:-1], steps[1:], strict=True):
        flow_arrow(ax, (0.56, y0 - 0.040), (0.56, y1 + 0.046), mutation_scale=10)

    graph_gate_y = 0.685
    scm_gate_y = 0.410
    flow_reject_return(ax, (0.39, graph_gate_y), 0.20, (0.31, 0.93))
    flow_label(ax, 0.345, graph_gate_y + 0.045, "yes", color=COLORS["accent2"])
    flow_label(ax, 0.595, graph_gate_y - 0.073, "no", ha="left")

    flow_reject_return(ax, (0.39, scm_gate_y), 0.105, (0.31, 0.93))
    flow_label(ax, 0.315, scm_gate_y + 0.045, "yes", color=COLORS["accent2"])
    flow_label(ax, 0.595, scm_gate_y - 0.073, "no", ha="left")

    ax.text(
        0.86,
        0.685,
        "faithfulness\nand validity\nfilters",
        ha="center",
        va="center",
        fontsize=7.3,
        color=COLORS["muted"],
    )
    fig.tight_layout()
    save(fig, "fig_instance_generation.pdf")


def fig_protocol_scoring() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.15))
    for ax in axes:
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax = axes[0]
    ax.set_title("ACDB episode", loc="left", fontsize=9, fontweight="bold", pad=2)
    boxes = [
        ((0.02, 0.50), "SCM\n$G, M$"),
        ((0.27, 0.50), "Observe\n$D^{obs}$"),
        ((0.52, 0.50), "Intervene\n$do(X_i=v)$"),
        ((0.77, 0.50), "Submit\n$\\hat{G}$"),
    ]
    for xy, label in boxes:
        rounded_box(ax, xy, 0.20, 0.30, label, facecolor=COLORS["light"], fontsize=8)
    for x0, x1 in [(0.22, 0.27), (0.47, 0.52), (0.72, 0.77)]:
        arrow(ax, (x0, 0.65), (x1, 0.65))
    ax.text(
        0.5,
        0.25,
        "same manifest, same API, same scoring",
        ha="center",
        va="center",
        fontsize=7.2,
        color=COLORS["muted"],
        wrap=True,
    )

    ax = axes[1]
    ax.set_title("Scoring layers", loc="left", fontsize=9, fontweight="bold", pad=2)
    layers = [
        ((0.08, 0.68), "Observational\nskeleton + compelled F1", COLORS["light"]),
        ((0.08, 0.32), "Final DAG\nDir F1 + SHD", "#e8f1fb"),
    ]
    for xy, label, color in layers:
        rounded_box(ax, xy, 0.84, 0.19, label, facecolor=color)
    for y0, y1 in [(0.68, 0.51)]:
        arrow(ax, (0.5, y0), (0.5, y1))
    ax.text(
        0.5,
        0.07,
        "Separates observational structure from final directed recovery",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=COLORS["muted"],
    )

    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_protocol_scoring.pdf")


def fig_density_leakage() -> None:
    original = pd.read_csv(RANDOM_SUMMARY)
    density = pd.read_csv(DENSITY_SUMMARY)
    original = original[original["level"] != "overall"].copy()
    density = density[density["level"] != "overall"].copy()
    original["density"] = original["level"].map(
        {"0": 5 / 6, "1": 6 / 10, "2": 6 / 10, "3": 9 / 21, "4": 6 / 10, "5": 9 / 21}
    )
    for df in (original, density):
        df["directed_f1_pct"] = 100 * df["directed_f1"].astype(float)
        df["density"] = df["density"].astype(float)
        df["level_num"] = df["level"].astype(int)

    original = original.sort_values("level_num")
    density = density.sort_values("level_num")
    x = original["level_num"].to_numpy()

    fig, ax = plt.subplots(figsize=(5.65, 2.45))
    ax.plot(
        x,
        original["directed_f1_pct"],
        marker="o",
        markersize=4.2,
        linewidth=1.2,
        color=COLORS["accent2"],
        label="v0 ladder",
        zorder=3,
    )
    ax.plot(
        x,
        density["directed_f1_pct"],
        marker="s",
        markersize=4.0,
        linewidth=1.2,
        color=COLORS["accent"],
        label="density probe",
        zorder=3,
    )

    v0_mean = 100 * pd.read_csv(RANDOM_SUMMARY).query("level == 'overall'")["directed_f1"].iloc[0]
    probe_mean = 100 * pd.read_csv(DENSITY_SUMMARY).query("level == 'overall'")["directed_f1"].iloc[0]
    ax.axhline(v0_mean, color=COLORS["accent2"], linewidth=0.8, linestyle=":")
    ax.axhline(probe_mean, color=COLORS["accent"], linewidth=0.8, linestyle=":")
    ax.text(5.12, v0_mean + 0.25, f"v0 mean {v0_mean:.1f}%", color=COLORS["accent2"], fontsize=7)
    ax.text(
        5.12,
        probe_mean + 0.25,
        f"probe mean {probe_mean:.1f}%",
        color=COLORS["accent"],
        fontsize=7,
    )

    for row in original.itertuples(index=False):
        ax.text(
            row.level_num,
            row.directed_f1_pct + 0.72,
            f"{row.directed_f1_pct:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.7,
            color=COLORS["accent2"],
        )
    for row in density.itertuples(index=False):
        ax.text(
            row.level_num,
            row.directed_f1_pct - 0.82,
            f"{row.directed_f1_pct:.1f}",
            ha="center",
            va="top",
            fontsize=6.7,
            color=COLORS["accent"],
        )

    tick_labels = [
        f"L{int(row.level_num)}\n$\\rho$ {row.density:.2f}/{density.iloc[i]['density']:.2f}"
        for i, row in enumerate(original.itertuples(index=False))
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=6.8)
    ax.set_xlabel(r"level (density $\rho$: v0 / probe)")
    ax.set_ylabel("random directed F1 (%)")
    ax.set_xlim(-0.35, 5.85)
    ax.set_ylim(11.3, 30.5)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    fig.tight_layout()
    save(fig, "fig_density_leakage.pdf")


def fig_results_model_ablation() -> None:
    df = load_results_note()
    df = df[df["level"] == "Avg L1-L5"].copy()

    method_order = ["llm_raw", "llm_stats", "pc_cpdag_llm", "llm_stats_cpdag_greedy"]
    method_labels = {
        "llm_raw": "LLM raw",
        "llm_stats": "LLM stats",
        "pc_cpdag_llm": "PC CPDAG + LLM",
        "llm_stats_cpdag_greedy": "LLM stats + greedy",
    }
    method_colors = {
        "llm_raw": COLORS["accent"],
        "llm_stats": COLORS["accent2"],
        "pc_cpdag_llm": "#3d6b35",
        "llm_stats_cpdag_greedy": "#7c5ca7",
    }

    raw_order = (
        df[df["method"] == "llm_raw"][["model", "dir_f1"]]
        .sort_values("dir_f1", ascending=False)["model"]
        .tolist()
    )
    model_order = list(reversed(raw_order))
    y = np.arange(len(model_order))
    offsets = np.array([-0.24, -0.08, 0.08, 0.24])

    fig, axes = plt.subplots(1, 2, figsize=(5.35, 2.2), sharey=True)
    metrics = [("dir_f1", "directed F1 (higher is better)"), ("SHD", "SHD (lower is better)")]

    baseline = (
        df[df["method"] == "pc_greedy"]
        .groupby("method", as_index=False)[["dir_f1", "SHD"]]
        .mean()
        .iloc[0]
    )

    for ax, (metric, xlabel) in zip(axes, metrics, strict=True):
        for offset, method in zip(offsets, method_order, strict=True):
            subset = (
                df[df["method"] == method]
                .set_index("model")
                .reindex(model_order)
                .reset_index()
            )
            values = subset[metric].to_numpy()
            if metric == "dir_f1":
                ax.barh(
                    y + offset,
                    values,
                    height=0.13,
                    color=method_colors[method],
                    label=method_labels[method],
                    zorder=3,
                )
            else:
                ax.barh(
                    y + offset,
                    values,
                    left=0.0,
                    height=0.13,
                    color=method_colors[method],
                    label=method_labels[method],
                    zorder=3,
                )
                for yi, value in zip(y + offset, values, strict=True):
                    ax.text(
                        value + 0.16,
                        yi,
                        f"{value:.1f}",
                        va="center",
                        ha="left",
                        fontsize=5.0,
                        color=COLORS["ink"],
                    )
        ax.axvline(
            float(baseline[metric]),
            color=COLORS["pc"],
            linestyle="--",
            linewidth=1.0,
            label="PC + greedy" if metric == "dir_f1" else None,
            zorder=2,
        )
        ax.set_xlabel(xlabel, fontsize=7.6)
        ax.grid(axis="x", color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=7)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(model_order, fontsize=7.4)
    axes[1].tick_params(axis="y", labelleft=False)
    for ax in axes:
        ax.invert_yaxis()
    axes[0].set_xlim(0.0, 0.8)
    axes[1].set_xlim(0.0, 24.0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=5.9,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.06),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.0)
    save(fig, "fig_results_model_ablation_v2.pdf")


def fig_headline_bars() -> None:
    df = load_all_ladder()
    rows = [
        ("Oracle", "oracle", "baseline", "active", COLORS["muted"]),
        ("PC+greedy", "pc_greedy", "baseline", "active", COLORS["pc"]),
        ("Sonnet raw", "llm_raw", "claude-sonnet-4-6", "active", COLORS["accent"]),
        ("Sonnet stats", "llm_stats", "claude-sonnet-4-6", "active", COLORS["accent2"]),
        ("Random uniform", None, None, None, COLORS["random"]),
        ("GPT raw", "llm_raw", "gpt-5.4", "active", COLORS["llm"]),
        ("GPT stats", "llm_stats", "gpt-5.4", "active", "#6b7280"),
    ]
    random_floor = 100 * pd.read_csv(RANDOM_SUMMARY).query("level == 'overall'")["directed_f1"].iloc[0]
    labels, values, colors = [], [], []
    for label, method, model, panel, color in rows:
        labels.append(label)
        colors.append(color)
        if method is None:
            values.append(random_floor)
        else:
            subset = df[(df["method"] == method) & (df["model"] == model) & (df["panel"] == panel)]
            values.append(100 * subset["directed_f1"].mean())

    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(5.75, 2.65))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.68)
    ax.axvline(random_floor, color=COLORS["random"], linestyle="--", linewidth=1.0)
    ax.text(
        random_floor + 1.0,
        len(labels) - 0.35,
        f"random floor {random_floor:.1f}%",
        ha="left",
        va="center",
        fontsize=7,
        color=COLORS["random"],
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("directed F1 (%)")
    ax.set_xlim(0, 105)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    for yi, value in zip(y, values, strict=True):
        ax.text(value + 1.2, yi, f"{value:.1f}", ha="left", va="center", fontsize=7.2, color=COLORS["ink"])
    fig.tight_layout()
    save(fig, "fig_headline_directed_f1.pdf")


def fig_precision_recall() -> None:
    df = load_all_ladder()
    groups = [
        ("PC+greedy", "active", "pc_greedy", "baseline", COLORS["pc"]),
        ("GPT raw", "active", "llm_raw", "gpt-5.4", COLORS["llm"]),
        ("Sonnet raw", "active", "llm_raw", "claude-sonnet-4-6", COLORS["accent"]),
        ("GPT stats", "active", "llm_stats", "gpt-5.4", COLORS["muted"]),
        ("Sonnet stats", "active", "llm_stats", "claude-sonnet-4-6", COLORS["accent2"]),
    ]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for label, panel, method, model, color in groups:
        subset = df[(df["panel"] == panel) & (df["method"] == method) & (df["model"] == model)]
        ax.scatter(
            100 * subset["directed_recall"].mean(),
            100 * subset["directed_precision"].mean(),
            s=40,
            color=color,
            label=label,
        )
    ax.set_xlabel("directed recall (%)")
    ax.set_ylabel("directed precision (%)")
    ax.set_xlim(0, 40)
    ax.set_ylim(20, 75)
    ax.grid(color="#e6e6e6", linewidth=0.6)
    ax.legend(frameon=False, fontsize=6.8, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_precision_recall.pdf")


def fig_efficiency_recall() -> None:
    df = load_all_ladder()
    groups = [
        ("PC+greedy", "pc_greedy", "baseline", COLORS["pc"]),
        ("GPT raw", "llm_raw", "gpt-5.4", COLORS["llm"]),
        ("Sonnet raw", "llm_raw", "claude-sonnet-4-6", COLORS["accent"]),
        ("GPT stats", "llm_stats", "gpt-5.4", COLORS["muted"]),
        ("Sonnet stats", "llm_stats", "claude-sonnet-4-6", COLORS["accent2"]),
    ]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for label, method, model, color in groups:
        subset = df[(df["panel"] == "active") & (df["method"] == method) & (df["model"] == model)]
        ax.scatter(
            100 * subset["efficiency"].mean(),
            100 * subset["directed_recall"].mean(),
            s=40,
            color=color,
            label=label,
        )
    ax.set_xlabel("efficiency (%)")
    ax.set_ylabel("directed recall (%)")
    ax.set_xlim(85, 101)
    ax.set_ylim(0, 40)
    ax.grid(color="#e6e6e6", linewidth=0.6)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_efficiency_recall.pdf")


def main() -> None:
    ensure_out_dir()
    set_style()
    fig_dag_cpdag_intervention()
    fig_instance_generation()
    fig_protocol_scoring()
    fig_density_leakage()
    fig_results_model_ablation()
    for path in sorted(OUT_DIR.glob("*.pdf")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
