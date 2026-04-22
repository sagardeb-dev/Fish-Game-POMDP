# Causal Discovery Benchmark (V1)

This repository implements an **active causal discovery benchmark** where an agent must recover a hidden DAG from observational data plus budget-limited interventions.

## Problem Statement

Given samples from an unknown linear-Gaussian SCM over `d` observed variables, the agent must infer the causal graph:

- Observational phase: infer what is identifiable up to Markov equivalence (CPDAG ceiling).
- Active phase: use interventions to resolve ambiguous orientations and recover the true DAG.

### Core Research Question

Can an LLM policy discover causal structure under constrained interventions, and how does it compare to classical structure-learning baselines under the same environment and scoring contract?

## Benchmark Contract

### V1 assumptions

- Linear-Gaussian SCM
- Causal sufficiency (no latent confounders)
- Causal Markov + faithfulness filtering
- Perfect single-node hard interventions `do(X_i = v)`
- Full observability of node values, no graph leakage

### Agent interface

- `observe() -> observational_data` (one-time only)
- `intervene(var, value) -> interventional_rows` (budgeted)
- `submit_graph(...) -> terminal submission`

### Submission format

All methods submit `GraphSubmission`:

- `directed_edges`
- `undirected_edges` (allowed, unresolved)
- `interventions_used`

This is shared across LLM, PC baselines, and oracle.

### Scoring layers

- **Observational layer** (against true CPDAG):
  - `skeleton_f1`
  - `compelled_f1`
- **DAG layer** (against true DAG):
  - `directed_f1`
  - `dag_shd`
- **Efficiency layer**:
  - intervention efficiency vs. precomputed minimum intervention set

Full scoring spec: [`docs/specs/scoring.md`](docs/specs/scoring.md)

## High-Level Pseudocode

Full spec: [`docs/specs/causal-discovery-v1-pseudocode.md`](docs/specs/causal-discovery-v1-pseudocode.md)

### 1) Build benchmark instance

```text
BUILD_BENCHMARK_INSTANCE(config):
  repeat:
    dag = SAMPLE_RANDOM_DAG(d, k)
    cpdag = DAG_TO_CPDAG(dag)
    if REJECT_GRAPH(dag, cpdag): continue

    scm = PARAMETERIZE_LINEAR_GAUSSIAN_SCM(dag, weight_range, noise_var)
    if REJECT_SCM(scm, faithfulness_eps): continue

    optimal_set = COMPUTE_MIN_INTERVENTION_SET(dag, cpdag)
    if REJECT_INTERVENTION_PROFILE(optimal_set, cpdag): continue

    permute labels
    obs_data = SAMPLE_OBSERVATIONAL_DATA(scm_public, n_obs)
    budget = |optimal_set| + budget_slack

    return instance(
      true_dag, cpdag, scm, obs_data, optimal_set, budget, metadata
    )
```

### 2) Runtime session

```text
RUNTIME_SESSION(instance, agent):
  obs = observe()   # one-time
  while budget > 0:
    action = agent.next_action()
    if action is intervene:
      return n_int rows sampled under do(var=value)
      budget -= 1
    elif action is submit_graph:
      break
  score_submission(instance, submission)
```

### 3) Score submission

```text
SCORE(instance, submission):
  observational metrics against CPDAG
  directed metrics + SHD against true DAG
  efficiency against |optimal_intervention_set|
```

## Ladder Configuration

The ladder has 6 levels:

- L0 tutorial: `d=4, k=5, n_obs=50, n_int=25, noise=0.5, slack=2`
- L1 standard: `d=5, k=6, n_obs=25, n_int=15, noise=1.0, slack=1`
- L2 statistical: `d=5, k=6, n_obs=15, n_int=10, noise=1.5, slack=1`
- L3 structural: `d=7, k=9, n_obs=25, n_int=15, noise=1.0, slack=1`
- L4 pressure: `d=5, k=6, n_obs=25, n_int=15, noise=1.0, slack=0`
- L5 hard: `d=7, k=9, n_obs=15, n_int=10, noise=1.5, slack=0`

Methods:

- Observational panel: `pc`, `llm_raw_obs`, `llm_stats_obs`
- Active panel: `pc_greedy`, `llm_raw`, `llm_stats`, `oracle`

## Results (Full Ladder Run)

Run artifacts:

- Run ID: `20260422T081508Z`
- Manifest: [`traces/ladder/full_ladder_run1/run_manifest.json`](traces/ladder/full_ladder_run1/run_manifest.json)
- Long table: [`traces/ladder/full_ladder_run1/results_long.csv`](traces/ladder/full_ladder_run1/results_long.csv)
- Summary table: [`traces/ladder/full_ladder_run1/results_summary.csv`](traces/ladder/full_ladder_run1/results_summary.csv)
- Aggregated by level: [`traces/ladder/full_ladder_run1/aggregated_by_level.csv`](traces/ladder/full_ladder_run1/aggregated_by_level.csv)
- Aggregated by seed: [`traces/ladder/full_ladder_run1/aggregated_by_seed.csv`](traces/ladder/full_ladder_run1/aggregated_by_seed.csv)

Coverage:

- Planned jobs: `336`
- Successful jobs: `334`
- Failed jobs: `2`

Failures:

- `active / llm_stats / level=2 / seed=1426457537`: `JSONDecodeError: Extra data`
- `observational / llm_raw_obs / level=3 / seed=976218778`: invalid submission (same pair directed and undirected)

### Observational panel (mean by level)

`directed_f1 / dag_shd`

| Level | llm_raw_obs (gpt-5.4) | llm_stats_obs (gpt-5.4) | pc (baseline) |
|---|---:|---:|---:|
| 0 | 0.377 / 4.000 | 0.000 / 6.000 | 0.139 / 4.375 |
| 1 | 0.418 / 6.125 | 0.000 / 9.875 | 0.238 / 5.125 |
| 2 | 0.303 / 7.125 | 0.000 / 9.875 | 0.031 / 5.875 |
| 3 | 0.137 / 17.714 | 0.000 / 19.125 | 0.172 / 8.000 |
| 4 | 0.345 / 6.125 | 0.000 / 10.000 | 0.200 / 5.250 |
| 5 | 0.137 / 17.625 | 0.000 / 15.250 | 0.038 / 8.875 |

### Active panel (mean by level)

`directed_f1 / dag_shd`

| Level | llm_raw (gpt-5.4) | llm_stats (gpt-5.4) | pc_greedy (baseline) | oracle |
|---|---:|---:|---:|---:|
| 0 | 0.348 / 4.250 | 0.318 / 4.375 | 0.651 / 2.250 | 1.000 / 0.000 |
| 1 | 0.319 / 6.500 | 0.278 / 6.125 | 0.497 / 3.750 | 1.000 / 0.000 |
| 2 | 0.294 / 6.500 | 0.279 / 7.143 | 0.307 / 4.750 | 1.000 / 0.000 |
| 3 | 0.314 / 13.250 | 0.167 / 11.250 | 0.489 / 5.750 | 1.000 / 0.000 |
| 4 | 0.320 / 7.125 | 0.146 / 8.000 | 0.465 / 4.000 | 1.000 / 0.000 |
| 5 | 0.199 / 13.875 | 0.086 / 12.000 | 0.152 / 8.250 | 1.000 / 0.000 |

### Weighted overall averages across levels

Observational:

- `llm_raw_obs`: `skeleton_f1=0.684`, `directed_f1=0.290`, `dag_shd=9.617`
- `llm_stats_obs`: `skeleton_f1=0.690`, `directed_f1=0.000`, `dag_shd=11.688`
- `pc`: `skeleton_f1=0.621`, `directed_f1=0.136`, `dag_shd=6.250`

Active:

- `llm_raw`: `skeleton_f1=0.598`, `directed_f1=0.299`, `dag_shd=8.583`, `efficiency=0.917`
- `llm_stats`: `skeleton_f1=0.531`, `directed_f1=0.211`, `dag_shd=8.170`, `efficiency=0.968`
- `pc_greedy`: `skeleton_f1=0.621`, `directed_f1=0.427`, `dag_shd=4.792`, `efficiency=0.896`
- `oracle`: perfect on all metrics

### Cost (gpt-5.4, this run)

- Prompt tokens: `564,750`
- Completion tokens: `37,440`
- Total tokens: `602,190`
- Estimated cost: `~$1.97` (input `$2.50/M`, output `$15.00/M`)

## How to Run

Setup:

```bash
uv sync
```

Unit tests:

```bash
uv run pytest tests/unit/causal_discovery -q
```

PC baselines:

```bash
uv run python run_pc_baseline.py
uv run python run_pc_interventional_baseline.py
```

Full ladder:

```bash
uv run python run_ladder.py --models gpt-5.4 --env-file "C:\projects\Random Research\Internet of Agents Benchmark\.env" --out-dir "traces/ladder/full_ladder_run1"
```

Resume + retry failed:

```bash
uv run python run_ladder.py --models gpt-5.4 --env-file "C:\projects\Random Research\Internet of Agents Benchmark\.env" --out-dir "traces/ladder/full_ladder_run1" --resume --retry-failed
```

## Repository Structure

```text
RL-environment/
  src/causal_discovery/
    agents/        LLM policies + action parsing + stats tools
    baselines/     PC parser/shared baseline utilities
    benchmark/     instance assembly
    config/        v1 benchmark config
    core/          DAG, SCM, permutation primitives
    equivalence/   CPDAG + Meek + minimum intervention set
    graph_gen/     random DAG generation
    runtime/       benchmark environment/session API
    sampling/      observational/interventional samplers
    scoring/       submission schema + score functions
    scm/           SEM parameterization + covariance diagnostics
  docs/specs/
    causal-discovery-v1-pseudocode.md
    causal-discovery-v1-modules.md
    scoring.md
  run_ladder.py
  run_pc_baseline.py
  run_pc_interventional_baseline.py
```

## Known Gaps / Next Fixes

- Harden model output parsing against multi-object JSON responses.
- Add sanitizer for edge-pair overlap in LLM submissions before final validation.
- Add explicit cost/token budget guardrails to `run_ladder.py`.
