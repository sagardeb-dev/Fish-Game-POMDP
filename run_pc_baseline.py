"""PC baseline against the causal discovery benchmark.

Runs observational-only PC on env.observe() data, commits undirected CPDAG
edges by acyclic completion (forward orientation under the current topo order),
and submits the resulting DAG. Reports scores across a few seeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_discovery import (
    BenchmarkEnv,
    build_benchmark_instance,
    make_v1_config,
    score_session,
)
from causallearn.search.ConstraintBased.PC import pc


def cpdag_graph_to_adjacency(endpoint_matrix: np.ndarray) -> np.ndarray:
    """Convert causal-learn endpoint matrix to a binary DAG adjacency.

    Convention: for i != j, endpoint_matrix[j, i] == 1 and [i, j] == -1 means i -> j.
    Undirected edges have both entries -1; orient them i -> j when i < j (forward
    in node-index order). Since the underlying skeleton admits that orientation
    without creating cycles (all arrows point from lower to higher index), the
    result is acyclic by construction.
    """
    d = endpoint_matrix.shape[0]
    adjacency = np.zeros((d, d), dtype=int)
    for i in range(d):
        for j in range(i + 1, d):
            a = endpoint_matrix[i, j]
            b = endpoint_matrix[j, i]
            if a == -1 and b == 1:
                adjacency[i, j] = 1
            elif a == 1 and b == -1:
                adjacency[j, i] = 1
            elif a == -1 and b == -1:
                adjacency[i, j] = 1
    return adjacency


def run_one(seed: int, d: int = 5, n_obs: int = 500, alpha: float = 0.05) -> dict:
    cfg = make_v1_config(d=d, n_obs=n_obs, n_int=40, budget_slack=1)
    instance = build_benchmark_instance(cfg, np.random.default_rng(seed))
    env = BenchmarkEnv(instance, np.random.default_rng(seed + 1))

    data = env.observe()
    cg = pc(data, alpha=alpha, indep_test="fisherz", show_progress=False, verbose=False)
    adjacency = cpdag_graph_to_adjacency(cg.G.graph)
    output = env.submit_graph(adjacency)
    scores = score_session(instance, output)

    return {
        "seed": seed,
        "true_edges": len(instance.true_dag.edges),
        "cpdag_directed": len(instance.observational_ceiling.directed_edges),
        "cpdag_undirected": len(instance.observational_ceiling.undirected_edges),
        "submitted_edges": int(adjacency.sum()),
        "obs": scores.observational,
        "int": scores.interventional,
        "eff": scores.efficiency,
    }


def main() -> None:
    seeds = [0, 1, 2, 3, 4, 5, 6, 7]
    header = (
        f"{'seed':>4} {'true':>4} {'cpdag_d':>8} {'cpdag_u':>8} "
        f"{'sub':>4} {'obs':>6} {'int':>6} {'eff':>6}"
    )
    print(header)
    print("-" * len(header))
    results = [run_one(s) for s in seeds]
    for r in results:
        print(
            f"{r['seed']:>4} {r['true_edges']:>4} {r['cpdag_directed']:>8} "
            f"{r['cpdag_undirected']:>8} {r['submitted_edges']:>4} "
            f"{r['obs']:>6.3f} {r['int']:>6.3f} {r['eff']:>6.3f}"
        )
    obs_mean = np.mean([r["obs"] for r in results])
    int_mean = np.mean([r["int"] for r in results])
    eff_mean = np.mean([r["eff"] for r in results])
    print("-" * len(header))
    print(f"mean                           {obs_mean:>6.3f} {int_mean:>6.3f} {eff_mean:>6.3f}")


if __name__ == "__main__":
    main()
