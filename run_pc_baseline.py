"""Observational PC baseline against the causal discovery benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_discovery import (
    BenchmarkEnv,
    GraphSubmission,
    build_benchmark_instance,
    make_v1_config,
    score_submission,
)
from causallearn.search.ConstraintBased.PC import pc


def cpdag_graph_to_submission(
    endpoint_matrix: np.ndarray, interventions_used: int = 0
) -> GraphSubmission:
    """Convert causal-learn endpoint matrix to a benchmark graph submission."""
    d = endpoint_matrix.shape[0]
    directed: set[tuple[int, int]] = set()
    undirected: set[tuple[int, int]] = set()
    for i in range(d):
        for j in range(i + 1, d):
            a = endpoint_matrix[i, j]
            b = endpoint_matrix[j, i]
            if a == -1 and b == 1:
                directed.add((i, j))
            elif a == 1 and b == -1:
                directed.add((j, i))
            elif a == -1 and b == -1:
                undirected.add((i, j))
    return GraphSubmission(
        num_nodes=d,
        directed_edges=frozenset(directed),
        undirected_edges=frozenset(undirected),
        interventions_used=interventions_used,
    )


def run_one(seed: int, d: int = 5, n_obs: int = 500, alpha: float = 0.05) -> dict:
    cfg = make_v1_config(d=d, n_obs=n_obs, n_int=40, budget_slack=1)
    instance = build_benchmark_instance(cfg, np.random.default_rng(seed))
    env = BenchmarkEnv(instance, np.random.default_rng(seed + 1))

    data = env.observe()
    cg = pc(data, alpha=alpha, indep_test="fisherz", show_progress=False, verbose=False)
    submission = cpdag_graph_to_submission(cg.G.graph)
    output = env.submit_graph(submission)
    scores = score_submission(instance, output.submission)

    return {
        "seed": seed,
        "true_edges": len(instance.true_dag.edges),
        "cpdag_directed": len(instance.observational_ceiling.directed_edges),
        "cpdag_undirected": len(instance.observational_ceiling.undirected_edges),
        "submitted_directed": len(output.submission.directed_edges),
        "submitted_undirected": len(output.submission.undirected_edges),
        "obs": scores.skeleton_f1,
        "comp": scores.compelled_f1,
        "dag": scores.directed_f1,
        "shd": scores.dag_shd,
        "eff": scores.efficiency,
    }


def main() -> None:
    seeds = [0, 1, 2, 3, 4, 5, 6, 7]
    header = (
        f"{'seed':>4} {'true':>4} {'cpdag_d':>8} {'cpdag_u':>8} "
        f"{'sub_d':>6} {'sub_u':>6} {'skel':>6} {'comp':>6} "
        f"{'dag':>6} {'shd':>4} {'eff':>6}"
    )
    print(header)
    print("-" * len(header))
    results = [run_one(s) for s in seeds]
    for r in results:
        print(
            f"{r['seed']:>4} {r['true_edges']:>4} {r['cpdag_directed']:>8} "
            f"{r['cpdag_undirected']:>8} {r['submitted_directed']:>6} "
            f"{r['submitted_undirected']:>6} {r['obs']:>6.3f} "
            f"{r['comp']:>6.3f} {r['dag']:>6.3f} {r['shd']:>4} {r['eff']:>6.3f}"
        )
    obs_mean = np.mean([r["obs"] for r in results])
    comp_mean = np.mean([r["comp"] for r in results])
    dag_mean = np.mean([r["dag"] for r in results])
    eff_mean = np.mean([r["eff"] for r in results])
    print("-" * len(header))
    print(
        f"mean                                      {obs_mean:>6.3f} "
        f"{comp_mean:>6.3f} {dag_mean:>6.3f}      {eff_mean:>6.3f}"
    )


if __name__ == "__main__":
    main()
