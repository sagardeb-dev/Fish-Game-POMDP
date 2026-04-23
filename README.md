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

## Results

Two full ladder runs are available:

- GPT-5.4: [`traces/ladder/full_ladder_toolcall_run1/results_long.csv`](traces/ladder/full_ladder_toolcall_run1/results_long.csv)
- Claude Sonnet 4.6: [`traces/ladder/sonnet46_full_ladder_run1/results_long.csv`](traces/ladder/sonnet46_full_ladder_run1/results_long.csv)

Each run covers the same ladder structure:

- Planned jobs: `336`
- Unique completed jobs: `336`
- GPT-5.4 successful jobs: `336`
- Sonnet 4.6 successful jobs: `335` (`1` failed `llm_stats` active row)

Note: resumed runs can append retry attempts to `results_long.csv`. The tables below deduplicate by `(level, seed, panel, method, model)` and keep the latest attempt. The generated `results_summary.csv` should not be treated as authoritative for resumed runs until summary deduplication is fixed.

### GPT-5.4: observational panel

All precision/recall/F1 values are percentages.

| Level | Method | Skel P | Skel R | Skel F1 | Dir P | Dir R | Dir F1 | DAG SHD |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | `llm_raw_obs` | 82.1 | 70.0 | 75.1 | 42.9 | 37.5 | 39.8 | 3.875 |
| 0 | `llm_stats_obs` | 88.1 | 60.0 | 66.9 | 41.7 | 10.0 | 12.3 | 5.000 |
| 0 | `pc` | 100.0 | 67.5 | 79.7 | 78.1 | 12.5 | 13.9 | 4.375 |
| 1 | `llm_raw_obs` | 65.1 | 62.5 | 60.9 | 60.6 | 22.9 | 27.7 | 7.000 |
| 1 | `llm_stats_obs` | 65.6 | 27.1 | 35.5 | 81.2 | 10.4 | 13.4 | 6.375 |
| 1 | `pc` | 96.9 | 54.2 | 68.8 | 68.8 | 16.7 | 23.8 | 5.125 |
| 2 | `llm_raw_obs` | 65.8 | 95.8 | 76.2 | 96.9 | 6.2 | 7.5 | 9.000 |
| 2 | `llm_stats_obs` | 70.8 | 18.8 | 28.8 | 68.8 | 6.2 | 9.4 | 6.250 |
| 2 | `pc` | 100.0 | 31.2 | 46.7 | 93.8 | 2.1 | 3.1 | 5.875 |
| 3 | `llm_raw_obs` | 55.2 | 66.7 | 54.6 | 82.3 | 13.9 | 18.8 | 14.250 |
| 3 | `llm_stats_obs` | 81.2 | 13.9 | 23.0 | 81.2 | 6.9 | 11.4 | 8.625 |
| 3 | `pc` | 100.0 | 48.6 | 65.0 | 89.6 | 11.1 | 17.2 | 8.000 |
| 4 | `llm_raw_obs` | 68.1 | 81.2 | 71.2 | 79.8 | 20.8 | 24.4 | 7.500 |
| 4 | `llm_stats_obs` | 87.5 | 25.0 | 38.4 | 68.8 | 14.6 | 21.9 | 5.375 |
| 4 | `pc` | 96.9 | 50.0 | 65.4 | 71.9 | 14.6 | 20.0 | 5.250 |
| 5 | `llm_raw_obs` | 52.5 | 72.2 | 56.3 | 82.1 | 13.9 | 17.7 | 14.750 |
| 5 | `llm_stats_obs` | 68.8 | 13.9 | 23.0 | 68.8 | 6.9 | 11.8 | 8.875 |
| 5 | `pc` | 97.5 | 31.9 | 46.9 | 93.8 | 2.8 | 3.8 | 8.875 |

### GPT-5.4: active panel

All precision/recall/F1/efficiency values are percentages.

| Level | Method | Skel P | Skel R | Skel F1 | Dir P | Dir R | Dir F1 | DAG SHD | Eff |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `llm_raw` | 77.7 | 60.0 | 67.3 | 33.8 | 25.0 | 28.5 | 4.625 | 56.2 |
| 0 | `llm_stats` | 82.9 | 65.0 | 72.2 | 42.3 | 27.5 | 31.2 | 4.250 | 100.0 |
| 0 | `pc_greedy` | 100.0 | 67.5 | 79.7 | 82.3 | 55.0 | 65.1 | 2.250 | 75.0 |
| 0 | `oracle` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.000 | 100.0 |
| 1 | `llm_raw` | 62.5 | 56.2 | 57.9 | 29.2 | 20.8 | 24.1 | 6.875 | 100.0 |
| 1 | `llm_stats` | 71.9 | 31.2 | 42.3 | 37.5 | 8.3 | 13.4 | 6.375 | 100.0 |
| 1 | `pc_greedy` | 96.9 | 54.2 | 68.8 | 67.7 | 39.6 | 49.7 | 3.750 | 87.5 |
| 1 | `oracle` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.000 | 100.0 |
| 2 | `llm_raw` | 64.8 | 58.3 | 60.0 | 35.2 | 22.9 | 27.6 | 6.625 | 100.0 |
| 2 | `llm_stats` | 66.7 | 22.9 | 33.4 | 35.4 | 8.3 | 12.2 | 6.375 | 100.0 |
| 2 | `pc_greedy` | 100.0 | 31.2 | 46.7 | 62.5 | 20.8 | 30.7 | 4.750 | 93.8 |
| 2 | `oracle` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.000 | 100.0 |
| 3 | `llm_raw` | 47.8 | 63.9 | 52.1 | 38.3 | 16.7 | 20.6 | 14.000 | 100.0 |
| 3 | `llm_stats` | 68.8 | 15.3 | 24.3 | 58.3 | 11.1 | 18.3 | 8.625 | 100.0 |
| 3 | `pc_greedy` | 100.0 | 48.6 | 65.0 | 77.3 | 36.1 | 48.9 | 5.750 | 81.2 |
| 3 | `oracle` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.000 | 100.0 |
| 4 | `llm_raw` | 65.8 | 70.8 | 65.2 | 43.8 | 12.5 | 15.0 | 7.625 | 100.0 |
| 4 | `llm_stats` | 87.5 | 27.1 | 40.5 | 58.3 | 14.6 | 21.6 | 5.375 | 100.0 |
| 4 | `pc_greedy` | 96.9 | 50.0 | 65.4 | 69.8 | 35.4 | 46.5 | 4.000 | 100.0 |
| 4 | `oracle` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.000 | 100.0 |
| 5 | `llm_raw` | 49.9 | 72.2 | 50.4 | 37.5 | 18.1 | 21.6 | 15.875 | 100.0 |
| 5 | `llm_stats` | 85.4 | 16.7 | 27.5 | 56.2 | 9.7 | 16.4 | 8.500 | 100.0 |
| 5 | `pc_greedy` | 97.5 | 31.9 | 46.9 | 41.7 | 9.7 | 15.2 | 8.250 | 100.0 |
| 5 | `oracle` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.000 | 100.0 |

### Claude Sonnet 4.6: observational panel

All precision/recall/F1 values are percentages.

| Level | Method | Skel P | Skel R | Skel F1 | Dir P | Dir R | Dir F1 | DAG SHD |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | `llm_raw_obs` | 78.3 | 65.0 | 70.7 | 29.4 | 25.0 | 26.9 | 4.625 |
| 0 | `llm_stats_obs` | 83.3 | 60.0 | 68.7 | 44.8 | 20.0 | 24.7 | 4.625 |
| 0 | `pc` | 100.0 | 67.5 | 79.7 | 78.1 | 12.5 | 13.9 | 4.375 |
| 1 | `llm_raw_obs` | 65.5 | 54.2 | 58.8 | 40.0 | 31.2 | 34.9 | 5.875 |
| 1 | `llm_stats_obs` | 61.3 | 64.6 | 58.9 | 85.4 | 10.4 | 14.0 | 8.125 |
| 1 | `pc` | 96.9 | 54.2 | 68.8 | 68.8 | 16.7 | 23.8 | 5.125 |
| 2 | `llm_raw_obs` | 66.2 | 62.5 | 63.6 | 39.8 | 39.6 | 39.3 | 5.500 |
| 2 | `llm_stats_obs` | 60.7 | 75.0 | 63.1 | 93.8 | 6.2 | 6.2 | 8.375 |
| 2 | `pc` | 100.0 | 31.2 | 46.7 | 93.8 | 2.1 | 3.1 | 5.875 |
| 3 | `llm_raw_obs` | 39.7 | 54.2 | 44.6 | 34.3 | 34.7 | 33.8 | 13.000 |
| 3 | `llm_stats_obs` | 50.9 | 72.2 | 53.8 | 92.9 | 8.3 | 10.9 | 15.875 |
| 3 | `pc` | 100.0 | 48.6 | 65.0 | 89.6 | 11.1 | 17.2 | 8.000 |
| 4 | `llm_raw_obs` | 67.7 | 54.2 | 59.9 | 47.5 | 37.5 | 41.7 | 5.375 |
| 4 | `llm_stats_obs` | 66.7 | 70.8 | 64.9 | 56.2 | 18.8 | 22.4 | 7.375 |
| 4 | `pc` | 96.9 | 50.0 | 65.4 | 71.9 | 14.6 | 20.0 | 5.250 |
| 5 | `llm_raw_obs` | 49.2 | 50.0 | 48.8 | 42.3 | 38.9 | 40.1 | 10.375 |
| 5 | `llm_stats_obs` | 49.8 | 69.4 | 55.1 | 88.2 | 11.1 | 12.0 | 15.125 |
| 5 | `pc` | 97.5 | 31.9 | 46.9 | 93.8 | 2.8 | 3.8 | 8.875 |

### Claude Sonnet 4.6: active panel

All precision/recall/F1/efficiency values are percentages.

| Level | Method | Skel P | Skel R | Skel F1 | Dir P | Dir R | Dir F1 | DAG SHD | Eff |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `llm_raw` | 73.1 | 57.5 | 63.7 | 31.9 | 27.5 | 29.4 | 4.625 | 87.5 |
| 0 | `llm_stats` | 89.6 | 67.5 | 76.4 | 32.3 | 22.5 | 26.4 | 4.250 | 100.0 |
| 0 | `pc_greedy` | 100.0 | 67.5 | 79.7 | 82.3 | 55.0 | 65.1 | 2.250 | 75.0 |
| 0 | `oracle` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.000 | 100.0 |
| 1 | `llm_raw` | 66.3 | 54.2 | 58.6 | 34.4 | 25.0 | 28.4 | 6.250 | 100.0 |
| 1 | `llm_stats` | 77.3 | 62.5 | 68.6 | 51.0 | 25.0 | 33.2 | 5.500 | 100.0 |
| 1 | `pc_greedy` | 96.9 | 54.2 | 68.8 | 67.7 | 39.6 | 49.7 | 3.750 | 87.5 |
| 1 | `oracle` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.000 | 100.0 |
| 2 | `llm_raw` | 60.3 | 54.2 | 56.7 | 38.8 | 35.4 | 36.9 | 5.875 | 100.0 |
| 2 | `llm_stats` | 65.1 | 64.3 | 62.8 | 35.0 | 26.2 | 29.7 | 6.571 | 100.0 |
| 2 | `pc_greedy` | 100.0 | 31.2 | 46.7 | 62.5 | 20.8 | 30.7 | 4.750 | 93.8 |
| 2 | `oracle` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.000 | 100.0 |
| 3 | `llm_raw` | 51.0 | 52.8 | 51.4 | 30.7 | 29.2 | 29.7 | 10.875 | 100.0 |
| 3 | `llm_stats` | 47.7 | 50.0 | 46.9 | 64.2 | 20.8 | 28.7 | 12.625 | 100.0 |
| 3 | `pc_greedy` | 100.0 | 48.6 | 65.0 | 77.3 | 36.1 | 48.9 | 5.750 | 81.2 |
| 3 | `oracle` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.000 | 100.0 |
| 4 | `llm_raw` | 72.1 | 58.3 | 63.9 | 37.1 | 31.2 | 33.6 | 5.500 | 100.0 |
| 4 | `llm_stats` | 69.9 | 75.0 | 70.7 | 33.4 | 22.9 | 26.5 | 6.750 | 100.0 |
| 4 | `pc_greedy` | 96.9 | 50.0 | 65.4 | 69.8 | 35.4 | 46.5 | 4.000 | 100.0 |
| 4 | `oracle` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.000 | 100.0 |
| 5 | `llm_raw` | 49.8 | 40.3 | 43.5 | 36.9 | 29.2 | 31.9 | 10.375 | 100.0 |
| 5 | `llm_stats` | 49.3 | 55.6 | 50.7 | 58.4 | 19.4 | 23.5 | 12.500 | 100.0 |
| 5 | `pc_greedy` | 97.5 | 31.9 | 46.9 | 41.7 | 9.7 | 15.2 | 8.250 | 100.0 |
| 5 | `oracle` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.000 | 100.0 |

### Weighted overall averages across levels

Observational:

- `llm_raw_obs`: `skeleton P/R/F1=64.8/74.7/65.7%`, `directed P/R/F1=74.1/19.2/22.6%`, `dag_shd=9.396`
- `llm_stats_obs`: `skeleton P/R/F1=77.0/26.4/35.9%`, `directed P/R/F1=68.4/9.2/13.3%`, `dag_shd=6.750`
- `pc`: `skeleton P/R/F1=98.5/47.2/62.1%`, `directed P/R/F1=82.6/10.0/13.6%`, `dag_shd=6.250`

Active:

- `llm_raw`: `skeleton P/R/F1=61.4/63.6/58.8%`, `directed P/R/F1=36.3/19.3/22.9%`, `dag_shd=9.271`, `efficiency=92.7%`
- `llm_stats`: `skeleton P/R/F1=77.2/29.7/40.0%`, `directed P/R/F1=48.0/13.3/18.8%`, `dag_shd=6.583`, `efficiency=100.0%`
- `pc_greedy`: `skeleton P/R/F1=98.5/47.2/62.1%`, `directed P/R/F1=66.9/32.8/42.7%`, `dag_shd=4.792`, `efficiency=89.6%`
- `oracle`: perfect on all metrics

Sonnet 4.6 observational:

- `llm_raw_obs`: `skeleton P/R/F1=61.1/56.7/57.7%`, `directed P/R/F1=38.9/34.5/36.1%`, `dag_shd=7.458`
- `llm_stats_obs`: `skeleton P/R/F1=62.1/68.7/60.7%`, `directed P/R/F1=76.9/12.5/15.0%`, `dag_shd=9.917`
- `pc`: `skeleton P/R/F1=98.5/47.2/62.1%`, `directed P/R/F1=82.6/10.0/13.6%`, `dag_shd=6.250`

Sonnet 4.6 active:

- `llm_raw`: `skeleton P/R/F1=62.1/52.9/56.3%`, `directed P/R/F1=35.0/29.6/31.7%`, `dag_shd=7.250`, `efficiency=97.9%`
- `llm_stats`: `skeleton P/R/F1=66.5/62.4/62.7%`, `directed P/R/F1=46.0/22.7/28.0%`, `dag_shd=8.064`, `efficiency=100.0%`, `n=47`
- `pc_greedy`: `skeleton P/R/F1=98.5/47.2/62.1%`, `directed P/R/F1=66.9/32.8/42.7%`, `dag_shd=4.792`, `efficiency=89.6%`
- `oracle`: perfect on all metrics

### Model comparison

| Panel | Method | Model/Baseline | Dir F1 | DAG SHD |
|---|---|---|---:|---:|
| Obs | `llm_raw_obs` | GPT-5.4 | 22.6 | 9.396 |
| Obs | `llm_raw_obs` | Sonnet 4.6 | 36.1 | 7.458 |
| Obs | `llm_stats_obs` | GPT-5.4 | 13.3 | 6.750 |
| Obs | `llm_stats_obs` | Sonnet 4.6 | 15.0 | 9.917 |
| Obs | `pc` | Baseline | 13.6 | 6.250 |
| Active | `llm_raw` | GPT-5.4 | 22.9 | 9.271 |
| Active | `llm_raw` | Sonnet 4.6 | 31.7 | 7.250 |
| Active | `llm_stats` | GPT-5.4 | 18.8 | 6.583 |
| Active | `llm_stats` | Sonnet 4.6 | 28.0 | 8.064 |
| Active | `pc_greedy` | Baseline | 42.7 | 4.792 |

### Interpretation

The headline result is not that LLMs beat classical causal discovery. They do not under the most error-sensitive metric. `pc_greedy` is the strongest non-oracle active method overall: it has the best active directed F1 (`42.7%`) and the lowest active DAG SHD (`4.792`), meaning it makes fewer total graph mistakes.

Sonnet 4.6 is the stronger LLM on directed recovery. It improves directed F1 over GPT-5.4 in every LLM setting, especially raw observational (`36.1%` vs `22.6%`) and raw active (`31.7%` vs `22.9%`). This means the weak GPT-5.4 result is partly a model-level limitation, not only a benchmark artifact.

The deeper bottleneck still appears general across LLMs. Even Sonnet 4.6 remains below `pc_greedy` on active directed F1 (`31.7%`/`28.0%` vs `42.7%`) and SHD (`7.250`/`8.064` vs `4.792`). The LLMs recover some causal structure, but they do not consistently turn observational and interventional evidence into low-error DAGs.

The stats-tool agents do not dominate raw agents. GPT-5.4 stats reduces SHD relative to raw but loses directed recall; Sonnet stats improves skeleton recovery but has worse SHD than Sonnet raw. This suggests that tool access alone is not enough: current LLMs still struggle with deciding which statistical evidence matters and when to submit a clean graph.

The main research takeaway is precision/recall asymmetry. LLM agents can propose plausible causal structure, but they either overcommit edges or underuse statistical evidence. The benchmark should report precision, recall, F1, and SHD together; F1 alone can make aggressive guessing look better than it is.

### Cost (gpt-5.4, this run)

- Prompt tokens: `1,397,089`
- Completion tokens: `94,871`
- Total tokens: `1,491,960`
- Estimated cost: `~$4.92` (input `$2.50/M`, output `$15.00/M`)

### Cost (Sonnet 4.6, this run)

- Estimated completed run cost: `~$10-$13`
- Partial-run verified cost before resume: `$0.6255` for `11` completed LLM rows
- Anthropic prompt caching was not effective in this run (`cache_read_input_tokens=0` during the checked partial run)

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
