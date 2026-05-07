"""Generate prototype ACDB figures without touching paper assets.

These figures are for narrative inspection only. They intentionally write under
reports/figure_prototypes/ instead of research/figures/generated/.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRACE_ROOT = ROOT / "traces" / "ladder"
REPORT_ROOT = ROOT / "reports" / "figure_prototypes"

COLORS = {
    "ink": "#222222",
    "muted": "#6b7280",
    "grid": "#e5e7eb",
    "light": "#f3f4f6",
    "blue": "#2b6cb0",
    "blue_light": "#e8f1fb",
    "green": "#2f855a",
    "green_light": "#e8f5ee",
    "orange": "#b45309",
    "orange_light": "#fff4e6",
    "red": "#b91c1c",
    "red_light": "#fee2e2",
    "purple": "#6b46c1",
    "purple_light": "#f0e7ff",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
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


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> list[str]:
    paths = []
    for suffix, dpi in [("pdf", None), ("png", 180)]:
        path = out_dir / f"{name}.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str | None = None,
    fontsize: float = 8,
    weight: str = "normal",
) -> None:
    edgecolor = edgecolor or COLORS["muted"]
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=0.85,
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
        fontweight=weight,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str | None = None,
    lw: float = 0.9,
    style: str = "-|>",
    dashed: bool = False,
    curve: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=10,
            linewidth=lw,
            linestyle=(0, (3, 2)) if dashed else "solid",
            color=color or COLORS["muted"],
            connectionstyle=curve,
            shrinkA=0,
            shrinkB=0,
        )
    )


def parse_key(key: str) -> dict[str, str]:
    values = {}
    for part in key.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            values[k] = v
    return values


def load_success_results() -> tuple[pd.DataFrame, list[str]]:
    frames = []
    sources = []
    for path in sorted(TRACE_ROOT.glob("*/results_long.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "status" not in df.columns:
            continue
        df = df[df["status"] == "success"].copy()
        if df.empty:
            continue
        df["source_trace"] = path.parent.name
        frames.append(df)
        sources.append(str(path.relative_to(ROOT)))
    if not frames:
        return pd.DataFrame(), sources
    df_all = pd.concat(frames, ignore_index=True)
    keys = [c for c in ["level", "seed", "panel", "method", "model", "source_trace"] if c in df_all.columns]
    if "timestamp_utc" in df_all.columns:
        df_all["timestamp_utc"] = pd.to_datetime(df_all["timestamp_utc"], errors="coerce")
        df_all = df_all.sort_values("timestamp_utc")
    return df_all.drop_duplicates(keys, keep="last"), sources


def load_events() -> tuple[list[dict[str, Any]], list[str]]:
    events = []
    sources = []
    for path in sorted(TRACE_ROOT.glob("*/events.jsonl")):
        sources.append(str(path.relative_to(ROOT)))
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event["_source_trace"] = path.parent.name
                events.append(event)
    return events, sources


def normalize_edges(edges: Any) -> set[tuple[int, int]]:
    out = set()
    for edge in edges or []:
        if len(edge) == 2:
            out.add((int(edge[0]), int(edge[1])))
    return out


def normalize_undirected(edges: Any) -> set[tuple[int, int]]:
    return {tuple(sorted(edge)) for edge in normalize_edges(edges)}


def edge_count_text(edges: set[tuple[int, int]]) -> str:
    return ", ".join(f"X{u}->X{v}" for u, v in sorted(edges)) or "none"


def canonical_results(results: pd.DataFrame) -> pd.DataFrame:
    """Keep only the full GPT/Sonnet ladder traces for aggregate prototype figures."""
    if results.empty or "source_trace" not in results.columns:
        return results
    source = results["source_trace"].astype(str)
    model = results["model"].astype(str)
    keep = (
        ((source == "full_ladder_toolcall_run1") & (model.str.contains("gpt", case=False, na=False) | (model == "baseline")))
        | ((source == "sonnet46_full_ladder_run1") & model.str.contains("sonnet", case=False, na=False))
    )
    return results[keep].copy()


def find_representative(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    metadata_by_case: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    success_by_case: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}

    for event in events:
        fields = parse_key(event.get("key", ""))
        if "level" not in fields or "seed" not in fields or "panel" not in fields:
            continue
        source = event.get("_source_trace", "")
        case = (source, fields["panel"], fields["level"], fields["seed"])
        method_case = (source, fields["panel"], fields["level"], fields["seed"], fields.get("method", ""))
        if event.get("event_type") == "instance_metadata":
            metadata_by_case.setdefault(case, event)
        elif event.get("event_type") == "work_success":
            success_by_case[method_case] = event

    candidates = []
    for case, meta_event in metadata_by_case.items():
        source, panel, level, seed = case
        if panel != "active":
            continue
        pc = success_by_case.get((source, panel, level, seed, "pc_greedy"))
        llm = success_by_case.get((source, panel, level, seed, "llm_stats")) or success_by_case.get(
            (source, panel, level, seed, "llm_raw")
        )
        if not pc or not llm:
            continue
        if "directed_edges" not in pc.get("payload", {}) or "directed_edges" not in llm.get("payload", {}):
            continue
        meta = meta_event["payload"]
        undirected = meta.get("cpdag_undirected_edges", [])
        true_edges = normalize_edges(meta.get("true_dag_edges", []))
        llm_edges = normalize_edges(llm["payload"].get("directed_edges", []))
        extra = len(llm_edges - true_edges)
        missing = len(true_edges - llm_edges)
        candidates.append(
            (
                len(undirected),
                extra + missing,
                1 if source == "deepseek_v4_flash_ladder_2seed" else 0,
                int(level),
                int(seed),
                {
                    "case": case,
                    "metadata": meta_event,
                    "pc": pc,
                    "llm": llm,
                },
            )
        )

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][-1]


def circular_positions(d: int) -> dict[int, tuple[float, float]]:
    angles = np.linspace(math.pi / 2, math.pi / 2 - 2 * math.pi, d, endpoint=False)
    return {i: (0.5 + 0.34 * math.cos(a), 0.52 + 0.34 * math.sin(a)) for i, a in enumerate(angles)}


def draw_graph(
    ax: plt.Axes,
    title: str,
    d: int,
    *,
    directed: set[tuple[int, int]],
    undirected: set[tuple[int, int]] | None = None,
    reference_directed: set[tuple[int, int]] | None = None,
    highlight_nodes: set[int] | None = None,
) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
    positions = circular_positions(d)
    undirected = undirected or set()
    reference_directed = reference_directed or set()
    highlight_nodes = highlight_nodes or set()

    def endpoints(u: int, v: int, shrink: float = 0.055):
        sx, sy = positions[u]
        tx, ty = positions[v]
        vec = np.array([tx - sx, ty - sy])
        dist = float(np.linalg.norm(vec))
        unit = vec / dist
        return (sx + unit[0] * shrink, sy + unit[1] * shrink), (
            tx - unit[0] * shrink,
            ty - unit[1] * shrink,
        )

    for u, v in sorted(undirected):
        start, end = endpoints(u, v)
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=COLORS["muted"],
            linewidth=1.0,
            linestyle=(0, (3, 2)),
            zorder=1,
        )

    for u, v in sorted(directed):
        start, end = endpoints(u, v)
        if reference_directed:
            color = COLORS["green"] if (u, v) in reference_directed else COLORS["red"]
        else:
            color = COLORS["ink"]
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.15,
                color=color,
                zorder=1,
            )
        )

    for node, (x, y) in positions.items():
        fill = COLORS["blue_light"] if node in highlight_nodes else "white"
        edge = COLORS["blue"] if node in highlight_nodes else COLORS["ink"]
        ax.add_patch(Circle((x, y), 0.055, facecolor=fill, edgecolor=edge, linewidth=1.0, zorder=3))
        ax.text(x, y, f"X{node}", ha="center", va="center", fontsize=7.5, color=COLORS["ink"], zorder=4)


def fig_hidden_state(out_dir: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.21, 0.93, "Evaluator state", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(0.79, 0.93, "Agent-visible state", ha="center", va="center", fontsize=10, fontweight="bold")

    left_items = [
        ("true DAG", 0.78),
        ("CPDAG / MEC", 0.63),
        ("minimum identifying set", 0.48),
        ("SCM parameters", 0.33),
        ("layered scores", 0.18),
    ]
    right_items = [
        ("observational samples", 0.78),
        ("optional stats tools", 0.63),
        ("optional interventions", 0.48),
        ("budget", 0.33),
        ("graph submission", 0.18),
    ]
    for text, y in left_items:
        rounded_box(ax, (0.05, y - 0.055), 0.32, 0.10, text, facecolor=COLORS["green_light"])
    for text, y in right_items:
        rounded_box(ax, (0.63, y - 0.055), 0.32, 0.10, text, facecolor=COLORS["blue_light"])

    arrow(ax, (0.37, 0.48), (0.63, 0.48), color=COLORS["muted"])
    ax.text(
        0.50,
        0.57,
        "hidden state is used only\nfor scoring and attribution",
        ha="center",
        va="center",
        fontsize=8,
        color=COLORS["muted"],
    )
    ax.text(
        0.50,
        0.08,
        "ACDB measures where an agent fails, not just whether the final graph is wrong.",
        ha="center",
        va="center",
        fontsize=8.5,
        color=COLORS["ink"],
    )
    fig.tight_layout()
    return save_figure(fig, out_dir, "01_hidden_agent_asymmetry")


def fig_scoring_contract(out_dir: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    rows = [
        ("Skeleton F1", "adjacency recovery", COLORS["blue_light"]),
        ("Compelled F1", "observationally identifiable direction", COLORS["purple_light"]),
        ("Directed F1 / SHD", "full DAG recovery", COLORS["green_light"]),
        ("Efficiency", "budget use vs. identifying set", COLORS["orange_light"]),
    ]
    for idx, (metric, referent, color) in enumerate(rows):
        y = 0.78 - idx * 0.19
        rounded_box(ax, (0.08, y - 0.055), 0.27, 0.10, metric, facecolor=color, weight="bold")
        arrow(ax, (0.36, y), (0.49, y))
        rounded_box(ax, (0.50, y - 0.055), 0.39, 0.10, referent, facecolor="white")
    ax.text(0.5, 0.94, "Each score maps to a theoretical object", ha="center", fontsize=10, fontweight="bold")
    ax.text(
        0.5,
        0.06,
        "The apparatus decomposes adjacency, orientation, full-DAG, and budget-use failures.",
        ha="center",
        fontsize=8.4,
        color=COLORS["muted"],
    )
    fig.tight_layout()
    return save_figure(fig, out_dir, "02_layered_scoring_contract")


def fig_dag_cpdag_intervention(out_dir: Path) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.1))
    d = 4
    true_edges = {(0, 1), (0, 2), (1, 3), (2, 3)}
    cpdag_dir = {(1, 3), (2, 3)}
    cpdag_undir = {(0, 1), (0, 2)}
    draw_graph(axes[0], "True DAG", d, directed=true_edges)
    draw_graph(axes[1], "Observational CPDAG", d, directed=cpdag_dir, undirected=cpdag_undir)
    draw_graph(axes[2], "After do(X0)", d, directed=true_edges, highlight_nodes={0})
    fig.text(
        0.5,
        0.02,
        "Observation identifies a Markov-equivalence class; interventions can resolve remaining directions.",
        ha="center",
        fontsize=8.2,
        color=COLORS["muted"],
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return save_figure(fig, out_dir, "03_dag_cpdag_intervention")


def fig_representative_graph(out_dir: Path, events: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    rep = find_representative(events)
    if rep is None:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No complete representative graph trace found", ha="center", va="center")
        return save_figure(fig, out_dir, "04_representative_graph_output"), {"status": "placeholder"}

    meta = rep["metadata"]["payload"]
    pc_payload = rep["pc"]["payload"]
    llm_payload = rep["llm"]["payload"]
    source, _, level, seed = rep["case"]
    d = int(meta["config"]["d"])
    true_edges = normalize_edges(meta.get("true_dag_edges", []))
    cpdag_dir = normalize_edges(meta.get("cpdag_directed_edges", []))
    cpdag_undir = normalize_undirected(meta.get("cpdag_undirected_edges", []))
    opt = {int(x) for x in meta.get("optimal_intervention_set", [])}
    pc_edges = normalize_edges(pc_payload.get("directed_edges", []))
    pc_undir = normalize_undirected(pc_payload.get("undirected_edges", []))
    llm_edges = normalize_edges(llm_payload.get("directed_edges", []))
    llm_undir = normalize_undirected(llm_payload.get("undirected_edges", []))

    fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.35))
    draw_graph(axes[0], "True DAG", d, directed=true_edges)
    draw_graph(axes[1], "CPDAG", d, directed=cpdag_dir, undirected=cpdag_undir)
    draw_graph(axes[2], "PC+greedy", d, directed=pc_edges, undirected=pc_undir, reference_directed=true_edges)
    draw_graph(axes[3], "LLM output", d, directed=llm_edges, undirected=llm_undir, reference_directed=true_edges)
    fig.text(
        0.5,
        0.03,
        f"Prototype instance: level {level}, seed {seed}. Green directed edges match truth; red directed edges are extra/wrong.",
        ha="center",
        fontsize=8.0,
        color=COLORS["muted"],
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    info = {
        "status": "data-derived",
        "level": int(level),
        "seed": int(seed),
        "source_trace": source,
        "true_edges": edge_count_text(true_edges),
        "optimal_intervention_set": sorted(opt),
        "pc_directed_edges": edge_count_text(pc_edges),
        "llm_directed_edges": edge_count_text(llm_edges),
    }
    return save_figure(fig, out_dir, "04_representative_graph_output"), info


def fig_random_floor(out_dir: Path) -> tuple[list[str], dict[str, Any]]:
    summary_paths = [
        TRACE_ROOT / "random_density_probe" / "results_random_dag_summary.csv",
        TRACE_ROOT / "random_uniform_baseline" / "results_random_dag_summary.csv",
    ]
    density_df = None
    source = None
    for path in summary_paths:
        if path.exists():
            df = pd.read_csv(path)
            has_density = "rho" in df.columns or "density" in df.columns
            has_directed = "directed_f1_mean" in df.columns or "directed_f1" in df.columns
            if has_density and has_directed:
                density_df = df.copy()
                source = str(path.relative_to(ROOT))
                break
    if density_df is None:
        fig, ax = plt.subplots(figsize=(5.4, 2.7))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No random density summary with rho found", ha="center", va="center")
        return save_figure(fig, out_dir, "05_random_floor_density"), {"status": "placeholder"}

    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    rho_col = "rho" if "rho" in density_df.columns else "density"
    f1_col = "directed_f1_mean" if "directed_f1_mean" in density_df.columns else "directed_f1"
    density_df = density_df.sort_values(rho_col)
    y = 100 * density_df[f1_col]
    ax.plot(density_df[rho_col], y, color=COLORS["muted"], linewidth=1.0)
    ax.scatter(density_df[rho_col], y, s=36, color=COLORS["blue"], label="measured random floor")
    if "expected_directed_f1" in density_df.columns:
        ax.plot(
            density_df[rho_col],
            100 * density_df["expected_directed_f1"],
            color=COLORS["orange"],
            linestyle=(0, (3, 2)),
            linewidth=1.1,
            label="expected floor",
        )
    else:
        rho = np.linspace(max(0.01, float(density_df[rho_col].min()) * 0.85), float(density_df[rho_col].max()) * 1.05, 100)
        ax.plot(
            rho,
            100 * (rho / (1 + 2 * rho)),
            color=COLORS["orange"],
            linestyle=(0, (3, 2)),
            linewidth=1.1,
            label="rho / (1 + 2rho)",
        )
    ax.set_xlabel("graph density rho")
    ax.set_ylabel("random directed F1 (%)")
    ax.grid(color=COLORS["grid"], linewidth=0.6)
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.set_title("Density calibrates the random floor", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, out_dir, "05_random_floor_density"), {"status": "data-derived", "source": source}


def exact_random_floor(d: int, k: int) -> float:
    m_max = d * (d - 1) // 2
    terms = [0.0]
    terms.extend(k * m / (m_max * (m + k)) for m in range(1, m_max + 1))
    return float(sum(terms) / (m_max + 1))


def fig_v0_v1_ladder_regions(out_dir: Path) -> tuple[list[str], dict[str, Any]]:
    v0 = [
        {"level": 0, "d": 4, "k": 5},
        {"level": 1, "d": 5, "k": 6},
        {"level": 2, "d": 5, "k": 6},
        {"level": 3, "d": 7, "k": 9},
        {"level": 4, "d": 5, "k": 6},
        {"level": 5, "d": 7, "k": 9},
    ]
    v1 = [
        {"level": 0, "d": 4, "k": 3},
        {"level": 1, "d": 6, "k": 6},
        {"level": 2, "d": 8, "k": 9},
        {"level": 3, "d": 10, "k": 12},
        {"level": 4, "d": 12, "k": 14},
        {"level": 5, "d": 14, "k": 16},
    ]

    def frame(rows: list[dict[str, int]], label: str) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        df["M"] = df["d"] * (df["d"] - 1) / 2
        df["rho"] = df["k"] / df["M"]
        df["floor"] = [exact_random_floor(int(row.d), int(row.k)) for row in df.itertuples()]
        df["ladder"] = label
        return df

    df = pd.concat([frame(v0, "v0"), frame(v1, "v1")], ignore_index=True)
    rho_grid = np.linspace(0.05, 0.9, 240)
    floor_grid = np.array([rho / (1 + 2 * rho) for rho in rho_grid])

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(
        rho_grid,
        100 * floor_grid,
        color=COLORS["muted"],
        linewidth=1.4,
        label=r"midpoint envelope $\rho/(1+2\rho)$",
    )
    for label, color, marker, yoff in [
        ("v0", COLORS["orange"], "o", 1.3),
        ("v1", COLORS["blue"], "s", -2.0),
    ]:
        subset = df[df["ladder"] == label]
        ax.scatter(
            subset["rho"],
            100 * subset["floor"],
            s=44,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=f"{label} ladder levels",
            zorder=5,
        )
        grouped_points = (
            subset.groupby(["rho", "floor"], as_index=False)
            .agg(level=("level", lambda xs: "/".join(f"L{int(x)}" for x in sorted(xs))))
        )
        for _, row in grouped_points.iterrows():
            ax.annotate(
                row["level"],
                (row["rho"], 100 * row["floor"]),
                xytext=(4, yoff),
                textcoords="offset points",
                fontsize=7.0,
                color=color,
            )

    ax.axvspan(df[df["ladder"] == "v0"]["rho"].min(), df[df["ladder"] == "v0"]["rho"].max(), color=COLORS["orange"], alpha=0.08)
    ax.axvspan(df[df["ladder"] == "v1"]["rho"].min(), df[df["ladder"] == "v1"]["rho"].max(), color=COLORS["blue"], alpha=0.08)
    ax.set_xlabel(r"graph density $\rho = k / [d(d-1)/2]$")
    ax.set_ylabel("expected random directed F1 (%)")
    ax.set_title("Where the v0 and v1 ladders sit on the random floor", fontsize=10, fontweight="bold")
    ax.set_xlim(0.05, 0.9)
    ax.set_ylim(8, 34)
    ax.grid(color=COLORS["grid"], linewidth=0.6)
    ax.legend(frameon=False, fontsize=7.2, loc="upper left")
    ax.text(
        0.50,
        9.4,
        "v1 shifts levels toward lower-density regimes where random guessing is less rewarded.",
        ha="center",
        fontsize=7.8,
        color=COLORS["ink"],
    )
    fig.tight_layout()
    info = {
        "status": "derived-from-ladder-params",
        "v0": df[df["ladder"] == "v0"][["level", "d", "k", "rho", "floor"]].to_dict(orient="records"),
        "v1": df[df["ladder"] == "v1"][["level", "d", "k", "rho", "floor"]].to_dict(orient="records"),
    }
    return save_figure(fig, out_dir, "08_v0_v1_ladder_regions"), info


def short_label(row: pd.Series) -> str:
    method = row["method"]
    model = str(row.get("model", ""))
    if method in {"pc", "pc_greedy", "oracle"}:
        return method
    if "sonnet" in model:
        return method.replace("llm_", "") + " / Sonnet"
    if "gpt" in model:
        return method.replace("llm_", "") + " / GPT"
    if "deepseek" in model:
        return method.replace("llm_", "") + " / DeepSeek"
    return method


def fig_precision_recall_shd(out_dir: Path, results: pd.DataFrame) -> tuple[list[str], dict[str, Any]]:
    needed = {"panel", "method", "model", "directed_precision", "directed_recall", "dag_shd"}
    if results.empty or not needed.issubset(results.columns):
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No complete precision/recall result table found", ha="center", va="center")
        return save_figure(fig, out_dir, "06_precision_recall_shd"), {"status": "placeholder"}

    results = canonical_results(results)
    df = results[results["panel"] == "active"].copy()
    df = df[df["method"].isin(["pc_greedy", "llm_raw", "llm_stats", "oracle"])]
    grouped = (
        df.groupby(["method", "model"], dropna=False)
        .agg(
            directed_precision=("directed_precision", "mean"),
            directed_recall=("directed_recall", "mean"),
            dag_shd=("dag_shd", "mean"),
            n=("dag_shd", "size"),
        )
        .reset_index()
    )
    if grouped.empty:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No active method rows found", ha="center", va="center")
        return save_figure(fig, out_dir, "06_precision_recall_shd"), {"status": "placeholder"}

    grouped["label"] = grouped.apply(short_label, axis=1)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.5, 5.0), gridspec_kw={"height_ratios": [3, 1.5]})
    colors = [COLORS["green"] if "pc" in x else COLORS["orange"] if "oracle" in x else COLORS["blue"] for x in grouped["label"]]
    ax.scatter(100 * grouped["directed_precision"], 100 * grouped["directed_recall"], s=48, color=colors)
    for _, row in grouped.iterrows():
        ax.annotate(row["label"], (100 * row["directed_precision"], 100 * row["directed_recall"]), xytext=(4, 3), textcoords="offset points", fontsize=6.5)
    ax.set_xlabel("directed precision (%)")
    ax.set_ylabel("directed recall (%)")
    ax.grid(color=COLORS["grid"], linewidth=0.6)
    ax.set_title("Precision and recall expose different failure modes", fontsize=10, fontweight="bold")

    order = grouped.sort_values("dag_shd")
    ax2.barh(order["label"], order["dag_shd"], color=COLORS["light"], edgecolor=COLORS["muted"], linewidth=0.7)
    ax2.set_xlabel("mean SHD")
    ax2.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    fig.tight_layout()
    return save_figure(fig, out_dir, "06_precision_recall_shd"), {"status": "data-derived", "rows": int(len(df))}


def fig_active_gain(out_dir: Path, results: pd.DataFrame) -> tuple[list[str], dict[str, Any]]:
    needed = {"panel", "method", "model", "level", "seed", "directed_f1"}
    if results.empty or not needed.issubset(results.columns):
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No paired active/observational rows found", ha="center", va="center")
        return save_figure(fig, out_dir, "07_active_gain"), {"status": "placeholder"}

    results = canonical_results(results)
    pairs = [
        ("PC", "pc", "pc_greedy", "baseline", "full_ladder_toolcall_run1"),
        ("raw / GPT", "llm_raw_obs", "llm_raw", "gpt", "full_ladder_toolcall_run1"),
        ("stats / GPT", "llm_stats_obs", "llm_stats", "gpt", "full_ladder_toolcall_run1"),
        ("raw / Sonnet", "llm_raw_obs", "llm_raw", "sonnet", "sonnet46_full_ladder_run1"),
        ("stats / Sonnet", "llm_stats_obs", "llm_stats", "sonnet", "sonnet46_full_ladder_run1"),
    ]
    rows = []
    for label, obs_method, active_method, model_hint, source_hint in pairs:
        source_mask = results["source_trace"].astype(str) == source_hint
        obs = results[source_mask & (results["panel"] == "observational") & (results["method"] == obs_method)].copy()
        active = results[source_mask & (results["panel"] == "active") & (results["method"] == active_method)].copy()
        if model_hint != "baseline":
            obs = obs[obs["model"].astype(str).str.contains(model_hint, case=False, na=False)]
            active = active[active["model"].astype(str).str.contains(model_hint, case=False, na=False)]
        else:
            obs = obs[obs["model"].astype(str) == "baseline"]
            active = active[active["model"].astype(str) == "baseline"]
        merged = obs.merge(active, on=["level", "seed", "model", "source_trace"], suffixes=("_obs", "_active"))
        if merged.empty:
            continue
        rows.append(
            {
                "label": label,
                "obs": float(merged["directed_f1_obs"].mean()),
                "active": float(merged["directed_f1_active"].mean()),
                "n": int(len(merged)),
            }
        )

    if not rows:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No matched active-gain pairs found", ha="center", va="center")
        return save_figure(fig, out_dir, "07_active_gain"), {"status": "placeholder"}

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    max_bound = max(50, int(math.ceil(max(df["obs"].max(), df["active"].max()) * 100 / 10 + 1) * 10))
    ax.plot([0, max_bound], [0, max_bound], color=COLORS["muted"], linestyle=(0, (3, 2)), linewidth=0.9, label="no active gain")
    offsets = {
        "PC": (5, 5),
        "raw / GPT": (10, -11),
        "stats / GPT": (5, -13),
        "raw / Sonnet": (5, 5),
        "stats / Sonnet": (5, 7),
        "raw / DeepSeek": (5, 6),
        "stats / DeepSeek": (5, -8),
    }
    alignments = {}
    for _, row in df.iterrows():
        color = COLORS["green"] if row["active"] >= row["obs"] else COLORS["red"]
        ax.scatter(100 * row["obs"], 100 * row["active"], color=color, s=44)
        ax.annotate(
            f"{row['label']} (n={row['n']})",
            (100 * row["obs"], 100 * row["active"]),
            xytext=offsets.get(row["label"], (4, 3)),
            textcoords="offset points",
            fontsize=6.7,
            ha=alignments.get(row["label"], "left"),
        )
    ax.set_xlabel("observational directed F1 (%)")
    ax.set_ylabel("active directed F1 (%)")
    ax.set_xlim(-5, max_bound)
    ax.set_ylim(-5, max_bound)
    ax.grid(color=COLORS["grid"], linewidth=0.6)
    ax.set_title("Paired active gain", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, out_dir, "07_active_gain"), {"status": "data-derived", "pairs": rows}


def write_readme(out_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# ACDB Figure Prototypes",
        "",
        "These artifacts are for narrative inspection only. They are not final paper figures.",
        "",
        "They were generated from current traces and may include partial or v0 runs.",
        "",
        "## Generated Figures",
        "",
    ]
    for name, paths in manifest["figures"].items():
        lines.append(f"- `{name}`")
        for path in paths:
            lines.append(f"  - `{path}`")
    lines.extend(
        [
            "",
            "## Representative Graph",
            "",
            "```json",
            json.dumps(manifest.get("representative_graph", {}), indent=2),
            "```",
            "",
            "## Source Files",
            "",
        ]
    )
    for source in manifest["sources"]:
        lines.append(f"- `{source}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    set_style()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPORT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=False)

    results, result_sources = load_success_results()
    events, event_sources = load_events()

    manifest: dict[str, Any] = {
        "generated_at_utc": stamp,
        "output_dir": str(out_dir.relative_to(ROOT)),
        "sources": sorted(set(result_sources + event_sources)),
        "figures": {},
        "notes": [
            "Prototype artifacts only; do not cite these as final paper numbers.",
            "The final figures should be regenerated after prompts, ladder, and seed map are locked.",
            "Aggregate result figures use gpt-5.4 and claude-sonnet-4-6 traces; no gpt-5.5 ladder trace was found.",
            "The representative graph uses the newer edge-complete DeepSeek trace because older GPT/Sonnet events do not store submitted edge lists.",
        ],
    }

    manifest["figures"]["hidden_agent_asymmetry"] = fig_hidden_state(out_dir)
    manifest["figures"]["layered_scoring_contract"] = fig_scoring_contract(out_dir)
    manifest["figures"]["dag_cpdag_intervention"] = fig_dag_cpdag_intervention(out_dir)
    paths, rep_info = fig_representative_graph(out_dir, events)
    manifest["figures"]["representative_graph_output"] = paths
    manifest["representative_graph"] = rep_info
    paths, random_info = fig_random_floor(out_dir)
    manifest["figures"]["random_floor_density"] = paths
    manifest["random_floor"] = random_info
    paths, ladder_region_info = fig_v0_v1_ladder_regions(out_dir)
    manifest["figures"]["v0_v1_ladder_regions"] = paths
    manifest["v0_v1_ladder_regions"] = ladder_region_info
    paths, pr_info = fig_precision_recall_shd(out_dir, results)
    manifest["figures"]["precision_recall_shd"] = paths
    manifest["precision_recall_shd"] = pr_info
    paths, gain_info = fig_active_gain(out_dir, results)
    manifest["figures"]["active_gain"] = paths
    manifest["active_gain"] = gain_info

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_readme(out_dir, manifest)

    print(out_dir.relative_to(ROOT))
    for name, paths in manifest["figures"].items():
        print(f"{name}:")
        for path in paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
