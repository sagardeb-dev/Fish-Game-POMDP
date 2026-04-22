"""PC + greedy interventional baseline.

Stage 1: run observational PC to get a CPDAG.
Stage 2: greedily intervene on the variable incident to the most undirected
         edges; orient each incident undirected edge by whether the neighbor's
         mean shifted under do(target=0).
Unresolved edges remain undirected in the submitted graph.
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
    GraphSubmission,
    build_benchmark_instance,
    make_v1_config,
    score_submission,
)
from causallearn.search.ConstraintBased.PC import pc


SHIFT_THRESHOLD = 0.5
INTERVENTION_OFFSET = 3.0


def parse_cpdag(endpoint_matrix: np.ndarray):
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
    return directed, undirected


def would_create_cycle(directed: set[tuple[int, int]], src: int, dst: int) -> bool:
    adj: dict[int, list[int]] = {}
    for a, b in directed:
        adj.setdefault(a, []).append(b)
    stack = [dst]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if node == src:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adj.get(node, []))
    return False


def resolve_with_interventions(env, directed, undirected, mu_obs) -> int:
    used = 0
    while undirected and env.remaining_budget > 0:
        deg: dict[int, int] = {}
        for edge in undirected:
            for v in edge:
                deg[v] = deg.get(v, 0) + 1
        target = min(deg.keys(), key=lambda v: (-deg[v], v))

        intervention_value = float(mu_obs[target] + INTERVENTION_OFFSET)
        samples = env.intervene(var=target, value=intervention_value)
        used += 1
        mu_int = samples.mean(axis=0)

        for edge in [e for e in undirected if target in e]:
            other = edge[1] if edge[0] == target else edge[0]
            shifted = abs(mu_int[other] - mu_obs[other]) > SHIFT_THRESHOLD
            candidate = (target, other) if shifted else (other, target)
            if would_create_cycle(directed, *candidate):
                continue
            directed.add(candidate)
            undirected.discard(edge)
    return used


def run_one(seed: int, d: int = 5, n_obs: int = 500, alpha: float = 0.05) -> dict:
    cfg = make_v1_config(d=d, n_obs=n_obs, n_int=40, budget_slack=1)
    instance = build_benchmark_instance(cfg, np.random.default_rng(seed))
    env = BenchmarkEnv(instance, np.random.default_rng(seed + 1))

    data = env.observe()
    mu_obs = data.mean(axis=0)

    cg = pc(data, alpha=alpha, indep_test="fisherz", show_progress=False, verbose=False)
    directed, undirected = parse_cpdag(cg.G.graph)

    used = resolve_with_interventions(env, directed, undirected, mu_obs)
    submission = GraphSubmission(
        num_nodes=d,
        directed_edges=frozenset(directed),
        undirected_edges=frozenset(undirected),
    )
    output = env.submit_graph(submission)
    scores = score_submission(instance, output.submission)

    return {
        "seed": seed,
        "true_edges": len(instance.true_dag.edges),
        "cpdag_d": len(instance.observational_ceiling.directed_edges),
        "cpdag_u": len(instance.observational_ceiling.undirected_edges),
        "budget": instance.intervention_budget,
        "used": used,
        "sub_d": len(output.submission.directed_edges),
        "sub_u": len(output.submission.undirected_edges),
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
        f"{'budget':>7} {'used':>5} {'sub_d':>6} {'sub_u':>6} "
        f"{'skel':>6} {'comp':>6} {'dag':>6} {'shd':>4} {'eff':>6}"
    )
    print(header)
    print("-" * len(header))
    results = [run_one(s) for s in seeds]
    for r in results:
        print(
            f"{r['seed']:>4} {r['true_edges']:>4} {r['cpdag_d']:>8} "
            f"{r['cpdag_u']:>8} {r['budget']:>7} {r['used']:>5} "
            f"{r['sub_d']:>6} {r['sub_u']:>6} {r['obs']:>6.3f} "
            f"{r['comp']:>6.3f} {r['dag']:>6.3f} {r['shd']:>4} {r['eff']:>6.3f}"
        )
    obs_mean = np.mean([r["obs"] for r in results])
    comp_mean = np.mean([r["comp"] for r in results])
    dag_mean = np.mean([r["dag"] for r in results])
    eff_mean = np.mean([r["eff"] for r in results])
    used_mean = np.mean([r["used"] for r in results])
    print("-" * len(header))
    print(
        f"mean                               {used_mean:>5.1f}              "
        f"{obs_mean:>6.3f} {comp_mean:>6.3f} {dag_mean:>6.3f}      {eff_mean:>6.3f}"
    )


if __name__ == "__main__":
    main()
