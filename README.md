# Causal Discovery Benchmark

A benchmark for evaluating agents on active causal discovery: given observational
data from a hidden linear-Gaussian SCM and a limited intervention budget, recover
the true DAG.

The runtime exposes `env.observe()`, `env.intervene(var, val)`, and graph
submission methods. Agents submit a `GraphSubmission`, which can contain both
directed edges and unresolved undirected edges. Scoring is shared across PC,
LLM, oracle, and active baselines.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd RL-environment
uv sync
```

## Run Baselines

```bash
uv run python run_pc_baseline.py
uv run python run_pc_interventional_baseline.py
```

`run_pc_baseline.py` submits the observational PC CPDAG directly. It does not
force unresolved edges into arbitrary directions.

`run_pc_interventional_baseline.py` runs PC first, uses interventions to orient
some unresolved edges, and leaves the rest undirected.

## Project Structure

```text
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
      scoring/        GraphSubmission + ScoreReport
  tests/
    unit/causal_discovery/
  docs/
    specs/
```

See `docs/specs/scoring.md` for the scoring contract.

## Tests

```bash
uv run pytest tests/unit/causal_discovery -q
```
