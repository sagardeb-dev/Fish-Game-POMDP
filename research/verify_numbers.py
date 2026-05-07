"""Verify body prose numbers against the aggregated CSV.

Source of truth: traces/aggregated/per_model_per_level_per_method.csv
Outputs differences between body prose and authoritative aggregates.

Usage: uv run python research/verify_numbers.py
"""
import csv
from pathlib import Path

CSV = Path("traces/aggregated/per_model_per_level_per_method.csv")
LEVELS = ["1", "2", "3", "4", "5"]


def load():
    rows = []
    with CSV.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def filt(rows, *, model=None, method=None, level=None):
    out = rows
    if model is not None:
        out = [r for r in out if r.get("panel_label") == model or r.get("panel_dir", "").endswith(model)]
    if method is not None:
        out = [r for r in out if r["method"] == method]
    if level is not None:
        out = [r for r in out if r["level"] == level]
    return out


def by_panel(rows, panel):
    return [r for r in rows if r["panel_label"] == panel]


def f(x):
    return float(x) if x not in ("", None) else float("nan")


def avg_level_mean(rows, model, method, metric, levels=LEVELS):
    """Mean of per-level means."""
    vals = []
    for L in levels:
        cells = [r for r in rows if r["panel_label"] == model and r["method"] == method and r["level"] == L]
        if not cells:
            continue
        vals.append(f(cells[0][metric]))
    return sum(vals) / len(vals) if vals else float("nan")


def avg_sample_weighted(rows, model, method, metric, levels=LEVELS):
    """n-weighted mean across levels."""
    num = 0.0
    den = 0
    for L in levels:
        cells = [r for r in rows if r["panel_label"] == model and r["method"] == method and r["level"] == L]
        if not cells:
            continue
        n = int(cells[0]["n"])
        num += n * f(cells[0][metric])
        den += n
    return num / den if den else float("nan")


def main():
    rows = load()
    panels = sorted({r["panel_label"] for r in rows})
    print(f"Panels in CSV: {panels}\n")

    print("=" * 70)
    print("BODY §5.1 CLAIM: llm_raw average L1-L5 dir_f1 per model")
    print("=" * 70)
    print(f"{'Model':<22} {'lvl-mean':>10} {'n-wgt':>10} {'body-prose':>12}")
    body = {
        "GPT-5.5": 0.697,
        "GPT-5.4-mini": 0.145,
        "Sonnet-4.6": 0.314,
        "Haiku-4.5": 0.113,
        "Gemini-3-Flash": 0.281,
    }
    for m, prose in body.items():
        lm = avg_level_mean(rows, m, "llm_raw", "directed_f1_mean")
        sw = avg_sample_weighted(rows, m, "llm_raw", "directed_f1_mean")
        flag = ""
        if abs(prose - lm) < 0.005:
            flag = "  matches level-mean"
        elif abs(prose - sw) < 0.005:
            flag = "  matches sample-weighted"
        else:
            flag = "  *** DISAGREES BOTH ***"
        print(f"{m:<22} {lm:>10.3f} {sw:>10.3f} {prose:>12.3f}{flag}")

    print()
    print("=" * 70)
    print("BODY §5.1 CLAIM: llm_raw average L1-L5 SHD per model (ordering check)")
    print("=" * 70)
    print(f"{'Model':<22} {'lvl-mean':>10} {'n-wgt':>10}")
    shds = {}
    for m in body:
        lm = avg_level_mean(rows, m, "llm_raw", "dag_shd_mean")
        sw = avg_sample_weighted(rows, m, "llm_raw", "dag_shd_mean")
        shds[m] = sw
        print(f"{m:<22} {lm:>10.3f} {sw:>10.3f}")
    f1_order = sorted(body.keys(), key=lambda m: -body[m])
    shd_order = sorted(shds.keys(), key=lambda m: shds[m])
    print(f"\n  dir_f1 best->worst: {f1_order}")
    print(f"  SHD best->worst:    {shd_order}")
    print(f"  ORDERINGS MATCH: {f1_order == shd_order}")

    print()
    print("=" * 70)
    print("BODY §5.1 CLAIM: ablation gain = pc_cpdag_llm minus llm_raw (avg L1-L5)")
    print("=" * 70)
    body_gains = {
        "GPT-5.5": +0.004,
        "GPT-5.4-mini": +0.284,
        "Sonnet-4.6": +0.312,
        "Haiku-4.5": +0.366,
        "Gemini-3-Flash": +0.304,
    }
    print(f"{'Model':<22} {'lvl-gain':>10} {'sw-gain':>10} {'body':>10}")
    for m, prose in body_gains.items():
        cp = avg_level_mean(rows, m, "pc_cpdag_llm", "directed_f1_mean")
        lr = avg_level_mean(rows, m, "llm_raw", "directed_f1_mean")
        sw_cp = avg_sample_weighted(rows, m, "pc_cpdag_llm", "directed_f1_mean")
        sw_lr = avg_sample_weighted(rows, m, "llm_raw", "directed_f1_mean")
        flag = ""
        if abs(prose - (cp - lr)) < 0.005:
            flag = "  matches lvl"
        elif abs(prose - (sw_cp - sw_lr)) < 0.005:
            flag = "  matches sw"
        else:
            flag = "  *** DISAGREES ***"
        print(f"{m:<22} {cp - lr:>+10.3f} {sw_cp - sw_lr:>+10.3f} {prose:>+10.3f}{flag}")

    print()
    print("=" * 70)
    print("BODY §5.3 CLAIM: pc_greedy and llm_raw L1 vs L5 per model")
    print("=" * 70)
    body_l1l5 = {
        ("pc_greedy", "1"): 0.802,
        ("pc_greedy", "5"): 0.642,
        ("GPT-5.5 llm_raw", "1"): 0.824,
        ("GPT-5.5 llm_raw", "5"): 0.537,
        ("Sonnet-4.6 llm_raw", "1"): 0.426,
        ("Sonnet-4.6 llm_raw", "5"): 0.252,
        ("Gemini-3-Flash llm_raw", "1"): 0.414,
        ("Gemini-3-Flash llm_raw", "5"): 0.185,
        ("Haiku-4.5 llm_raw", "1"): 0.192,
        ("Haiku-4.5 llm_raw", "5"): 0.047,
    }
    for (key, lvl), prose in body_l1l5.items():
        if key == "pc_greedy":
            cells = [r for r in rows if r["method"] == "pc_greedy" and r["level"] == lvl]
        else:
            model, method = key.split(" ", 1)
            cells = [r for r in rows if r["panel_label"] == model and r["method"] == method and r["level"] == lvl]
        if not cells:
            print(f"{key} {lvl}: NO ROW")
            continue
        v = f(cells[0]["directed_f1_mean"])
        flag = "  OK" if abs(v - prose) < 0.005 else "  *** DISAGREES ***"
        print(f"{key:<26} {lvl}  csv={v:.3f}  body={prose:.3f}{flag}")

    print()
    print("=" * 70)
    print("TABLE 2 (GPT-5.5 average L1-L5)")
    print("=" * 70)
    body_t2 = {
        "pc_greedy":     (0.780, 0.612, 0.689, 5.3),
        "pc_cpdag_llm":  (0.789, 0.615, 0.701, 5.2),
        "llm_raw":       (0.753, 0.732, 0.697, 6.3),
        "llm_stats":     (0.704, 0.673, 0.640, 7.3),
    }
    for method, (skel, comp, dir_, shd) in body_t2.items():
        for metric, prose, label in [
            ("skeleton_f1_mean", skel, "skel"),
            ("compelled_f1_mean", comp, "comp"),
            ("directed_f1_mean", dir_, "dir"),
            ("dag_shd_mean", shd, "SHD"),
        ]:
            if method == "pc_greedy":
                # pc_greedy is one row per level (no panel_label dep)
                lm = avg_level_mean(rows, "GPT-5.5", "pc_greedy", metric)
            else:
                lm = avg_level_mean(rows, "GPT-5.5", method, metric)
            digits = 1 if label == "SHD" else 3
            v = f"{lm:.{digits}f}"
            flag = "OK" if abs(lm - prose) < 0.05 else "*** DISAGREE ***"
            print(f"  {method:<18} {label:<5} csv={v:>7}  body={prose:<7}  {flag}")


if __name__ == "__main__":
    main()
