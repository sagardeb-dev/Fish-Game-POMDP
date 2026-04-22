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

### Active panel (mean by level)

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

### Interpretation

The headline result is not that LLMs beat classical causal discovery. They do not under the most error-sensitive metric. `pc_greedy` is the strongest non-oracle active method overall: it has the best directed F1 (`42.7%`) and the lowest DAG SHD (`4.792`), meaning it makes fewer total graph mistakes.

`llm_raw` and `llm_raw_obs` often look competitive on F1 because they recover more edges than conservative methods. The precision/recall split shows the cost: active `llm_raw` has directed recall `19.3%` but directed precision only `36.3%`, while `pc_greedy` has directed precision `66.9%` and recall `32.8%`. The LLM is more willing to assert causal edges, which can lift recall but also increases false positives. SHD exposes this: active `llm_raw` has `dag_shd=9.271`, almost double `pc_greedy`.

`llm_stats` is more conservative than `llm_raw`. It has better SHD than raw (`6.583` vs `9.271`) but lower skeleton and directed recall. This suggests the statistical tools reduce some hallucinated structure, but the model still does not use them well enough to match the PC-based active baseline.

The main research takeaway is therefore precision/recall asymmetry: LLM agents can propose plausible causal structure, but they overcommit or underuse statistical evidence. The benchmark should report F1 together with precision, recall, and SHD; F1 alone can make aggressive guessing look better than it is.

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
