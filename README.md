# Causal Discovery Benchmark

A benchmark for evaluating LLM agents on **active causal discovery**: given observational data from a hidden linear-Gaussian SCM and a limited intervention budget, recover the true DAG.

The agent interacts with a push-based session (`env.observe()`, `env.intervene(var, val)`, `env.submit_graph(matrix)`) and is scored on three dimensions — structure recovery under observational equivalence, full DAG recovery against ground truth, and intervention efficiency.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd RL-environment
uv sync
```

## Run the PC baseline

```bash
uv run python run_pc_baseline.py
```

PC (Peter-Clark) is a purely observational constraint-based algorithm. It calls `env.observe()` once, returns a CPDAG, and we acyclically complete the undirected edges before submitting.

## Results — PC baseline (observational only)

Mean across 8 seeds, `d=5`, `n_obs=500`, `alpha=0.05`:

| Metric | Mean | Notes |
|---|---:|---|
| observational | 0.825 | skeleton + v-structure match vs true CPDAG |
| interventional | 0.520 | F1 over directed edges vs true DAG |
| efficiency | 1.000 | clamps to 1.0 because no interventions are used |

Per-seed detail:

| seed | true edges | cpdag directed | cpdag undirected | submitted | obs | int | eff |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6 | 3 | 3 | 5 | 0.778 | 0.727 | 1.000 |
| 1 | 6 | 0 | 6 | 5 | 0.833 | 0.182 | 1.000 |
| 2 | 6 | 3 | 3 | 5 | 0.778 | 0.545 | 1.000 |
| 3 | 6 | 3 | 3 | 6 | 1.000 | 0.833 | 1.000 |
| 4 | 6 | 2 | 4 | 5 | 0.875 | 0.364 | 1.000 |
| 5 | 6 | 0 | 6 | 5 | 0.833 | 0.182 | 1.000 |
| 6 | 6 | 0 | 6 | 5 | 0.833 | 0.727 | 1.000 |
| 7 | 6 | 0 | 6 | 5 | 0.667 | 0.600 | 1.000 |

Interventional F1 swings widely (0.18–0.83) on instances where the CPDAG has many undirected edges — PC cannot orient them from observational data alone, and our acyclic-completion heuristic orients them arbitrarily by forward index order. This is the exact signal we expect interventional baselines to close.

## Project structure

```
RL-environment/
  src/
    causal_discovery/
      benchmark/      BenchmarkInstance assembly
      config/         v1 config builder
      core/           DAG, LinearGaussianSCM, Permutation
      equivalence/    CPDAG, Meek rules, min intervention set
      graph_gen/      random DAG sampler
      runtime/        push-based BenchmarkEnv + SessionOutput
      sampling/       observational + interventional samplers
      scm/            linear-Gaussian parameterization + diagnostics
      scoring/        observational / interventional / efficiency scores
  tests/
    unit/causal_discovery/
  docs/
    specs/            v1 pseudocode + module specs
  run_pc_baseline.py  PC baseline entrypoint (uses causal-learn)
```

See `src/causal_discovery/README.md` for module-level details.

## Tests

```bash
uv run pytest tests/unit/causal_discovery -q
```
