# causal_discovery

Core package for the active causal discovery benchmark. The package is layered
so graph generation, SCM semantics, equivalence theory, sampling, runtime, and
scoring can be verified independently.

| Module | Role |
|---|---|
| `core/` | `DAG`, `LinearGaussianSCM`, `Permutation` immutable data objects |
| `graph_gen/` | random DAG sampling under edge-count constraint `k` |
| `scm/` | linear-Gaussian SCM parameterization and diagnostics |
| `equivalence/` | `CPDAG`, `dag_to_cpdag`, minimum intervention set, rejection rules |
| `sampling/` | observational and perfect-hard-intervention samplers |
| `config/` | `BenchmarkConfig`, `make_v1_config` |
| `benchmark/` | `BenchmarkInstance`, rejection loop, label permutation |
| `runtime/` | `BenchmarkEnv`, `SessionOutput` |
| `scoring/` | `GraphSubmission`, `ScoreReport`, `score_submission` |

## Agent-Facing API

```python
from causal_discovery import (
    BenchmarkEnv,
    GraphSubmission,
    build_benchmark_instance,
    make_v1_config,
    score_submission,
)

cfg = make_v1_config(d=5, n_obs=500, n_int=40, budget_slack=1)
instance = build_benchmark_instance(cfg, rng)
env = BenchmarkEnv(instance, rng)

data = env.observe()
samples = env.intervene(var=0, value=0.0)
submission = GraphSubmission.from_adjacency_matrix(adjacency_matrix)
output = env.submit_graph(submission)
scores = score_submission(instance, output.submission)
```

Convenience submission methods:

```python
env.submit_adjacency_matrix(matrix)
env.submit_cpdag(directed_edges, undirected_edges)
```

## Session Rules

- `observe()` is callable once and returns a copy of the observational data.
- `intervene(var, value)` requires `observe()` first and decrements budget.
- `submit_graph(submission)` requires `observe()` first and seals the session.
- After submission, all session methods raise.

## Scoring

The public scoring path is:

```text
GraphSubmission -> score_submission(instance, submission) -> ScoreReport
```

`GraphSubmission` supports directed and unresolved undirected edges. This allows
observational algorithms like PC to submit a CPDAG-style graph without fake DAG
completion.

`ScoreReport` contains:

- skeleton precision/recall/F1 against the true CPDAG skeleton
- compelled-orientation precision/recall/F1 against true CPDAG directed edges
- directed precision/recall/F1 against the true DAG
- DAG SHD against the true DAG
- intervention efficiency against the minimum intervention set size

See `docs/specs/scoring.md` for exact metric definitions.
