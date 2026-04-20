# causal_discovery

Core package for the active causal discovery benchmark. Built in strict layers — each module depends only on those above it in this table.

| Module | Role |
|---|---|
| `core/` | `DAG`, `LinearGaussianSCM`, `Permutation` — immutable data objects |
| `graph_gen/` | Random DAG sampling under edge-count constraint `k` |
| `scm/` | Linear-Gaussian SCM parameterization + structural partial-correlation diagnostics |
| `equivalence/` | `CPDAG`, `dag_to_cpdag` (Meek R1–R3), `compute_minimum_intervention_set`, rejection rules |
| `sampling/` | Observational + interventional samplers (perfect hard interventions) |
| `config/` | `BenchmarkConfig`, `make_v1_config` — frozen knob bundle |
| `benchmark/` | `BenchmarkInstance`, `build_benchmark_instance` (rejection loop, label permutation) |
| `runtime/` | `BenchmarkEnv`, `SessionOutput` — push-based agent interface |
| `scoring/` | `observational_score`, `interventional_score`, `efficiency_score`, `score_session` |

## Agent-facing API

```python
from causal_discovery import (
    make_v1_config,
    build_benchmark_instance,
    BenchmarkEnv,
    score_session,
)

cfg = make_v1_config(d=5, n_obs=500, n_int=40, budget_slack=1)
instance = build_benchmark_instance(cfg, rng)
env = BenchmarkEnv(instance, rng)

data = env.observe()                                  # (n_obs, d) numpy array; callable once
samples = env.intervene(var=0, value=0.0)             # (n_int, d); decrements budget
output = env.submit_graph(adjacency_matrix)           # seals the session
scores = score_session(instance, output)
```

### Session rules

- `observe()` is callable **once** per session.
- `intervene(var, value)` decrements `env.remaining_budget`; raises on exhaustion.
- `submit_graph(matrix)` requires shape `(d, d)`, binary `{0, 1}`, zero diagonal, acyclic.
- After `submit_graph`, the session is sealed — all methods raise.

### Hidden truth

`BenchmarkInstance` also carries `true_dag`, `observational_ceiling` (CPDAG), `scm`, `optimal_intervention_set`, and `label_permutation`. These are read only by `score_session` — an agent that imports the instance directly to inspect them is on the honor system.

## Scoring

Three scores in `[0, 1]`, no composite:

- **`observational_score(estimated_dag, cpdag)`** — `(skeleton_hits + directed_hits) / (|skeleton| + |cpdag.directed_edges|)`. Skeleton is direction-insensitive; directed component only credits CPDAG-directed edges oriented correctly. Undirected CPDAG edges contribute only to skeleton. False positives beyond the CPDAG skeleton are not penalized here (interventional F1 catches them).
- **`interventional_score(estimated_dag, true_dag)`** — F1 over directed edge sets. Penalizes false positives and reversed edges.
- **`efficiency_score(interventions_used, optimal_set_size)`** — `optimal / max(used, optimal)`. Clamps to 1.0 when `used < optimal` (lucky guesses should not be double-penalized; `interventional_score` catches wrong commitments).

## Design commitments

- **Push-based session.** Agent calls env methods directly. No event loop, no generator protocol.
- **Full-DAG submissions only.** Agents cannot submit a CPDAG; they must commit orientations for undirected edges. This rewards commitment over calibrated uncertainty.
- **Fresh RNG per session.** `BenchmarkEnv` takes an explicit `np.random.Generator`. No module-level randomness.
- **Single session per env.** Re-running a trial requires a new `BenchmarkEnv` and a new RNG.

## Tests

```bash
uv run pytest tests/unit/causal_discovery -q
```
