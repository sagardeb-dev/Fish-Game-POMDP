"""Generate the random-submission directed-F1 floor figure for Appendix C.

Reads the existing probe summary CSV; emits research/figures/generated/fig_random_floor.pdf.
No re-run of the probe; closed-form curves are computed from the same formula
as scripts/ladder_random_floor_sanity.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "traces" / "ladder_random_floor_sanity" / "summary.csv"
OUT = ROOT / "research" / "figures" / "generated" / "fig_random_floor.pdf"

# Production v1 ladder (d, k) per ladder_levels_v1() in run_ladder.py
V1 = {1: (6, 6), 2: (8, 9), 3: (10, 12), 4: (12, 14), 5: (14, 16)}

# L5 (d=14, k=16) is not in the existing probe CSV (probe L5 used d=10).
# Measured here with the same protocol (16 instances x 100 samples) using
# make_random_uniform_submission and score_submission. See plot script docstring.
L5_MEASURED = {"mean": 0.1144, "se": 0.0017, "n": 1600}


def m_max(d: int) -> int:
    return d * (d - 1) // 2


def expected_f1(d: int, k: int) -> float:
    """Closed-form E[directed F1] under the uniform-m random DAG sampler.

    Identical to exact_expected_directed_f1 in scripts/ladder_random_floor_sanity.py.
    """
    M = m_max(d)
    total = 0.0
    for m in range(M + 1):
        if m + k > 0:
            total += k * m / (M * (m + k))
    return total / (M + 1)


def midpoint_envelope(rho: float) -> float:
    return rho / (1.0 + 2.0 * rho)


def expected_f1_continuous(rho: float, d_ref: int = 20) -> float:
    """E[F1] as a function of density at fixed reference d_ref.

    The closed form depends on (d, k); plotting against rho means picking a
    reference d and letting k = round(rho * M(d_ref)). The shape is essentially
    flat in d_ref for d_ref >= 12, so this is just a smooth visual reference.
    """
    M = m_max(d_ref)
    k = max(1, int(round(rho * M)))
    return expected_f1(d_ref, k)


def main() -> None:
    summary = pd.read_csv(SUMMARY)
    probe = summary[summary["level_set"] == "v0_like_density_probe"].copy()
    probe = probe.sort_values("level")

    # Production v1 (d, k) match probe rows for L1=(6,6), L2=(8,9), L3=(10,12), L4=(12,14)
    # Probe L5 is (10, 12) -- mismatched with v1 L5=(14, 16); exclude empirical L5 dot.
    v1_match_levels = []
    for level, (d, k) in V1.items():
        if level == 5:
            v1_match_levels.append(
                {
                    "level": 5,
                    "d": d,
                    "k": k,
                    "rho": k / m_max(d),
                    "measured_mean": L5_MEASURED["mean"],
                    "measured_se": L5_MEASURED["se"],
                }
            )
            continue
        row = probe[(probe["d"] == d) & (probe["k"] == k)]
        if not row.empty:
            r = row.iloc[0]
            v1_match_levels.append(
                {
                    "level": level,
                    "d": d,
                    "k": k,
                    "rho": k / m_max(d),
                    "measured_mean": float(r["measured_f1_mean"]),
                    "measured_se": float(r["measured_f1_se"]),
                }
            )

    # Closed-form curve sampled densely
    rho_grid = np.linspace(0.001, 0.95, 400)
    closed_form = np.array([expected_f1_continuous(r) for r in rho_grid])
    midpoint = np.array([midpoint_envelope(r) for r in rho_grid])

    # v1 ladder shaded region
    v1_rhos = [k / m_max(d) for d, k in V1.values()]
    rho_lo, rho_hi = min(v1_rhos), max(v1_rhos)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.axvspan(rho_lo, rho_hi, color="#f6c79b", alpha=0.35, label="ladder range (L1--L5)")
    ax.plot(rho_grid, closed_form, color="#1f77b4", lw=2.0, label=r"$\mathbb{E}[F_1]$ (uniform-$m$ exact)")

    for entry in v1_match_levels:
        ax.errorbar(
            entry["rho"],
            entry["measured_mean"],
            yerr=entry["measured_se"],
            fmt="o",
            color="black",
            markersize=5,
            capsize=3,
            zorder=5,
        )
        ax.annotate(
            f"L{entry['level']}",
            (entry["rho"], entry["measured_mean"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )

    ax.set_xlabel(r"Graph density $\rho = k / [d(d-1)/2]$")
    ax.set_ylabel(r"Random-submission directed $F_1$")
    ax.set_xlim(0.0, 0.95)
    ax.set_ylim(0.0, 0.45)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=150)
    print(f"[done] {OUT}")
    print(f"[done] {OUT.with_suffix('.png')}")
    print()
    print("Per-level closed-form floors (production v1):")
    for level, (d, k) in V1.items():
        print(f"  L{level}: d={d}, k={k}, rho={k/m_max(d):.4f}, E[F1]={expected_f1(d, k):.4f}")


if __name__ == "__main__":
    main()
