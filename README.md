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

## Results (Full Ladder Tool-Call Run)

Run artifacts:

- Run ID: `20260422T105637Z`
- Manifest: [`traces/ladder/full_ladder_toolcall_run1/run_manifest.json`](traces/ladder/full_ladder_toolcall_run1/run_manifest.json)
- Long table: [`traces/ladder/full_ladder_toolcall_run1/results_long.csv`](traces/ladder/full_ladder_toolcall_run1/results_long.csv)
- Summary table: [`traces/ladder/full_ladder_toolcall_run1/results_summary.csv`](traces/ladder/full_ladder_toolcall_run1/results_summary.csv)

Coverage:

- Planned jobs: `336`
- Unique completed jobs: `336`
- Unique successful jobs: `336`
- Unique failed jobs: `0`

Note: this run used `--resume --retry-failed`. The raw CSV contains appended retry attempts. The tables below deduplicate by `(level, seed, panel, method, model)` and keep the latest attempt. The current generated `results_summary.csv` should not be used directly for resumed runs until summary deduplication is fixed.

### Observational panel (mean by level)

`skeleton_f1% / directed_f1% / dag_shd`

| Level | llm_raw_obs (gpt-5.4) | llm_stats_obs (gpt-5.4) | pc (baseline) |
|---|---:|---:|---:|
| 0 | 75.1% / 39.8% / 3.875 | 66.9% / 12.3% / 5.000 | 79.7% / 13.9% / 4.375 |
| 1 | 60.9% / 27.7% / 7.000 | 35.5% / 13.4% / 6.375 | 68.8% / 23.8% / 5.125 |
| 2 | 76.2% / 7.5% / 9.000 | 28.8% / 9.4% / 6.250 | 46.7% / 3.1% / 5.875 |
| 3 | 54.6% / 18.8% / 14.250 | 23.0% / 11.4% / 8.625 | 65.0% / 17.2% / 8.000 |
| 4 | 71.2% / 24.4% / 7.500 | 38.4% / 21.9% / 5.375 | 65.4% / 20.0% / 5.250 |
| 5 | 56.3% / 17.7% / 14.750 | 23.0% / 11.8% / 8.875 | 46.9% / 3.8% / 8.875 |

### Active panel (mean by level)

`skeleton_f1% / directed_f1% / dag_shd`

| Level | llm_raw (gpt-5.4) | llm_stats (gpt-5.4) | pc_greedy (baseline) | oracle |
|---|---:|---:|---:|---:|
| 0 | 67.3% / 28.5% / 4.625 | 72.2% / 31.2% / 4.250 | 79.7% / 65.1% / 2.250 | 100.0% / 100.0% / 0.000 |
| 1 | 57.9% / 24.1% / 6.875 | 42.3% / 13.4% / 6.375 | 68.8% / 49.7% / 3.750 | 100.0% / 100.0% / 0.000 |
| 2 | 60.0% / 27.6% / 6.625 | 33.4% / 12.2% / 6.375 | 46.7% / 30.7% / 4.750 | 100.0% / 100.0% / 0.000 |
| 3 | 52.1% / 20.6% / 14.000 | 24.3% / 18.3% / 8.625 | 65.0% / 48.9% / 5.750 | 100.0% / 100.0% / 0.000 |
| 4 | 65.2% / 15.0% / 7.625 | 40.5% / 21.6% / 5.375 | 65.4% / 46.5% / 4.000 | 100.0% / 100.0% / 0.000 |
| 5 | 50.4% / 21.6% / 15.875 | 27.5% / 16.4% / 8.500 | 46.9% / 15.2% / 8.250 | 100.0% / 100.0% / 0.000 |

### Weighted overall averages across levels

Observational:

- `llm_raw_obs`: `skeleton_f1=65.7%`, `directed_f1=22.6%`, `dag_shd=9.396`
- `llm_stats_obs`: `skeleton_f1=35.9%`, `directed_f1=13.3%`, `dag_shd=6.750`
- `pc`: `skeleton_f1=62.1%`, `directed_f1=13.6%`, `dag_shd=6.250`

Active:

- `llm_raw`: `skeleton_f1=58.8%`, `directed_f1=22.9%`, `dag_shd=9.271`, `efficiency=92.7%`
- `llm_stats`: `skeleton_f1=40.0%`, `directed_f1=18.8%`, `dag_shd=6.583`, `efficiency=100.0%`
- `pc_greedy`: `skeleton_f1=62.1%`, `directed_f1=42.7%`, `dag_shd=4.792`, `efficiency=89.6%`
- `oracle`: perfect on all metrics

### Cost (gpt-5.4, this run)

- Prompt tokens: `1,397,089`
- Completion tokens: `94,871`
- Total tokens: `1,491,960`
- Estimated cost: `~$4.92` (input `$2.50/M`, output `$15.00/M`)

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
uv run python run_ladder.py --models gpt-5.4 --env-file "C:\projects\Random Research\Internet of Agents Benchmark\.env" --out-dir "traces/ladder/full_ladder_toolcall_run1"
```

Resume + retry failed:

```bash
uv run python run_ladder.py --models gpt-5.4 --env-file "C:\projects\Random Research\Internet of Agents Benchmark\.env" --out-dir "traces/ladder/full_ladder_toolcall_run1" --resume --retry-failed
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

- Deduplicate retry attempts inside `results_summary.csv` so resumed runs cannot double-count old attempts.
- Add explicit cost/token budget guardrails to `run_ladder.py`.
