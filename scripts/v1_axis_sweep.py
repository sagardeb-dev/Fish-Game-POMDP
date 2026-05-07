"""Sweep candidate (d, k, n_obs) tuples for the v1 ladder redesign.

Computes two diagnostic axes per cell:

  Identifiability axis: undirected-edge fraction in the true CPDAG, averaged
  over random DAGs at fixed (d, k). Maps to the observational-layer ceiling.

  Statistical axis: Fisher-z power at the smallest |partial correlation|
  along the relevant structural set, evaluated at sample size n_obs and
  conditioning-set size taken from the actual edge. Maps to PC's
  finite-sample resolution.

Output: traces/v1_axis_sweep/results.csv with one row per
(d, k, n_obs, dag_seed). Aggregation across seeds is done by the consumer.

This script is read-only relative to the codebase: it imports existing
utilities and writes a CSV. It does not modify run_ladder.py, the SCM
parameterization, or the paper.
"""

from __future__ import annotations

import sys
from math import atanh, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from causal_discovery.config import DEFAULT_WEIGHT_RANGE
from causal_discovery.equivalence import dag_to_cpdag, reject_graph
from causal_discovery.graph_gen import sample_random_dag
from causal_discovery.scm import (
    implied_covariance,
    parameterize_linear_gaussian_scm,
    relevant_structural_partial_correlations,
)


OUTPUT_DIR = ROOT / "traces" / "v1_axis_sweep"
ALPHA = 0.05
N_DAGS_PER_CELL = 8
NOISE_VAR = 1.0
SCM_RNG_OFFSET = 10_000


def fisher_z_power(r: float, n: int, cond_size: int, alpha: float = ALPHA) -> float:
    """Two-sided Fisher-z power at partial correlation r, sample n, |S|=cond_size."""
    if abs(r) >= 1.0 or n - cond_size - 3 <= 0:
        return float("nan")
    z_crit = norm.ppf(1 - alpha / 2)
    nc = abs(atanh(r)) * sqrt(n - cond_size - 3)
    return float(norm.cdf(nc - z_crit))


def candidate_grid() -> list[tuple[int, int]]:
    """Enumerate (d, k) cells with rho in the feasible band [0.25, 0.40].

    The originally-targeted [0.20, 0.30] band is largely unreachable after
    reject_graph filters trivial CPDAG structure; see V1_CHANGE_LOG.md
    "Proposed v1 ladder redesign" entry for the empirical narrowing.
    """
    cells: list[tuple[int, int]] = []
    for d in (5, 6, 7, 8):
        m_max = d * (d - 1) // 2
        for k in range(2, m_max + 1):
            rho = k / m_max
            if 0.25 <= rho <= 0.40:
                cells.append((d, k))
    return cells


def n_obs_grid() -> list[int]:
    """n_obs ladder targeted at the empirical |partial r| range observed at this
    benchmark's weight bound (|r| typically 0.45-0.65 because DEFAULT_WEIGHT_RANGE
    excludes near-zero edges). Spans Fisher-z power roughly 0.4-0.99."""
    return [15, 25, 50, 150, 600]


def sweep() -> pd.DataFrame:
    rows: list[dict] = []
    cells = candidate_grid()
    n_obs_values = n_obs_grid()

    for d, k in cells:
        m_max = d * (d - 1) // 2
        rho = k / m_max
        accepted_for_cell = 0
        for dag_seed in range(N_DAGS_PER_CELL * 20):
            if accepted_for_cell >= N_DAGS_PER_CELL:
                break
            dag_rng = np.random.default_rng(seed=dag_seed + 1_000 * (d * 100 + k))
            attempts = 0
            dag = None
            cpdag = None
            while attempts < 500:
                attempts += 1
                candidate = sample_random_dag(d, k, dag_rng)
                cand_cpdag = dag_to_cpdag(candidate)
                if not reject_graph(candidate, cand_cpdag):
                    dag = candidate
                    cpdag = cand_cpdag
                    break
            if dag is None:
                break  # cell too restrictive, skip remaining seeds
            accepted_for_cell += 1

            undirected_fraction = cpdag.num_undirected_edges / k

            scm_rng = np.random.default_rng(seed=dag_seed + SCM_RNG_OFFSET)
            scm = parameterize_linear_gaussian_scm(
                dag,
                DEFAULT_WEIGHT_RANGE,
                NOISE_VAR,
                scm_rng,
            )
            partials = relevant_structural_partial_correlations(scm)
            min_abs_r = min(abs(r) for r in partials)
            cond_sizes = [len(scm.parents(dst)) - 1 for _, dst in sorted(scm.dag.edges)]
            min_idx = int(np.argmin([abs(r) for r in partials]))
            cond_size_at_min = max(0, cond_sizes[min_idx])

            for n_obs in n_obs_values:
                power = fisher_z_power(min_abs_r, n_obs, cond_size_at_min)
                rows.append({
                    "d": d,
                    "k": k,
                    "rho": rho,
                    "dag_seed": dag_seed,
                    "n_obs": n_obs,
                    "undirected_fraction": undirected_fraction,
                    "num_undirected_edges": cpdag.num_undirected_edges,
                    "min_abs_partial_correlation": min_abs_r,
                    "cond_size_at_min": cond_size_at_min,
                    "fisher_z_power": power,
                })
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = sweep()
    out = OUTPUT_DIR / "results.csv"
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows to {out}")

    summary = (
        df.groupby(["d", "k", "rho", "n_obs"], as_index=False)
        .agg(
            mean_undirected_fraction=("undirected_fraction", "mean"),
            std_undirected_fraction=("undirected_fraction", "std"),
            mean_min_partial=("min_abs_partial_correlation", "mean"),
            mean_power=("fisher_z_power", "mean"),
        )
    )
    summary_path = OUTPUT_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {len(summary)} aggregated rows to {summary_path}")
    print()
    print("Cell summary (sorted by power):")
    print(summary.sort_values(["d", "k", "n_obs"]).to_string(index=False))


if __name__ == "__main__":
    main()
