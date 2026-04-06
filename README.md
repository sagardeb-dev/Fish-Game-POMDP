# Fishing Game POMDP Benchmark

A POMDP-based benchmark for evaluating LLM agent capabilities in **causal discovery**, **tool use**, **Bayesian inference**, and **decision-making under uncertainty**. The agent manages a fishing fleet across 4 zones over a 20-day season, facing hidden risks (storms, equipment failures, tide) that must be discovered through database analysis — not told in the prompt.

Inspired by [NewtonBench](https://arxiv.org/abs/2503.02453) (ICLR 2026) and [DiscoveryBench](https://arxiv.org/abs/2407.01725) — agents must discover hidden causal structure through data, not instruction-following.

## Architecture

```
                          ┌─────────────────────────────────────┐
                          │         Hidden Generative Model      │
                          │  80 states = 2(storm) × 4(wind)     │
                          │       × 5(equip_failure) × 2(tide)  │
                          │                                      │
                          │  config.py: BENCHMARK_CONFIG         │
                          │  pomdp.py:  FishingPOMDP             │
                          └──────────┬────────────┬─────────────┘
                                     │            │
                          transitions│            │observations
                          T(s'|s)    │            │O(o|s)
                                     ▼            ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     FishingGameEnv  (simulator.py)                       │
│                                                                          │
│  reset(seed) ──► _init_db() ──► _generate_historical_data()             │
│                        │                                                 │
│                        ▼                                                 │
│           ┌─────────────────────────────────┐                            │
│           │     SQLite Episode Database      │                            │
│           │                                  │                            │
│           │  catch_history (30 days)         │  Historical data for       │
│           │  sensor_log (buoy, equip, temp)  │  causal discovery.         │
│           │  maintenance_log (alerts)        │  All 4 zones, 30 days.     │
│           │  weather_signals                 │                            │
│           │  daily_log (current season)      │                            │
│           │  daily_conditions (HIDDEN)       │  Blocked from agent SQL.   │
│           └─────────────────────────────────┘                            │
│                                                                          │
│  Sensor Zone Subsampling (BENCHMARK_CONFIG):                             │
│  Only 2 of 4 zones report sensors each day (randomly selected).          │
│  Forces agents to rely on transition model between steps.                │
│                                                                          │
│  Observation Tiers:                                                      │
│  ┌─────────────────────────────────┬────────────────────────────────┐    │
│  │ TIER 1 — Free                   │ TIER 2 — SQL-Discoverable      │    │
│  │                                 │                                │    │
│  │  sea_color    (storm signal)    │  buoy_readings   (storm)       │    │
│  │  equip_indicator (equip signal) │  equip_readings  (equip)       │    │
│  │  barometer    (storm signal)    │                                │    │
│  │  maintenance_alerts (equip)     │  Promoted to agent's trace     │    │
│  │  water_temp   (tide + confound) │  only if SQL tools were used   │    │
│  │                                 │  this step                     │    │
│  │  (only for 2 sensor zones/day)  │  (only for 2 sensor zones/day) │    │
│  └─────────────────────────────────┴────────────────────────────────┘    │
│                                                                          │
│  Budget-Gated Tools (1/day each in BENCHMARK_CONFIG):                    │
│  check_weather_reports, check_equipment_reports,                         │
│  query_fishing_log (SQL), query_maintenance_log (SQL),                   │
│  analyze_data (Python sandbox), evaluate_options, forecast_scenario      │
│                                                                          │
│  submit_decisions(allocation, beliefs, reasoning)  ──►  advances day     │
└──────────────────────────────────────────────────────────────────────────┘
        │                                               ▲
        │ observation bundle                            │ tool calls +
        │ + reward                                      │ submit_decisions
        ▼                                               │
┌──────────────────────────────────────────────────────────────────────────┐
│                           Agent Layer                                    │
│                                                                          │
│  Baselines (baselines.py) — no API key needed:                           │
│  ├── RandomAgent          — random zone, no tools                        │
│  ├── NaivePatternMatcher  — falls for all causal traps                   │
│  ├── CausalLearner        — learns params from DB via SQL + statistics   │
│  ├── CausalReasoner       — true params + exact Bayesian filtering       │
│  └── OracleAgent          — reads hidden state (upper bound)             │
│                                                                          │
│  LLM Agents (require OpenAI API key):                                    │
│  ├── LLMAgent (llm_agent.py)      — free-form tool-calling LLM          │
│  │   └── GPTAgent (gpt_agent.py)  — OpenAI GPT integration              │
│  └── LLMSolverAgent (llm_solver_agent.py)                                │
│      — LLM estimates world model on day 1, solver does exact Bayes       │
│      — Isolates model discovery from inference/planning                  │
└──────────────────────────────────────────────────────────────────────────┘
        │                                               │
        │ episode trace                                 │ episode trace
        ▼                                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    Evaluator  (evaluator.py)                             │
│                                                                          │
│  3-Way Cost Decomposition (algebraic identity, holds every step):       │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  oracle_reward - actual_reward =                                   │  │
│  │      tool_use_gap    (oracle - retrieved: did agent gather info?)  │  │
│  │    + inference_gap   (retrieved - belief:  did agent reason well?) │  │
│  │    + planning_gap    (belief - actual:     did agent act on it?)   │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  Episode Metrics:                                                        │
│  total_reward, mean_brier_{storm,equip}, detection_lag,                 │
│  total_{tool_use,inference,planning}_gap, tool_usage_counts             │
└──────────────────────────────────────────────────────────────────────────┘
```

### Turn Sequence (each of 20 days)

```
1. Env transitions hidden state:  s_t → s_{t+1}  via T(s'|s)
2. Env selects 2 random sensor zones for the day
3. Env emits observations + weather/equipment signals into DB
4. Agent receives observation bundle (Tier 1 sensors for 2 zones + metadata)
5. Agent calls tools (SQL queries, reports, analysis) — day does NOT advance
6. Agent calls submit_decisions(allocation, beliefs, reasoning) — day advances
7. Env computes reward, updates DB, returns next observation
```

### Causal Traps (discoverable, not told)

```
Trap 1: Wave Propagation               Trap 2: Age-Confounded Equipment

  Zones form a ring: A─B─C─D─A           Zone ages: A=25yr B=15yr C=5yr D=2yr

  Storm in zone X causes elevated         Old zones always show high equip
  buoy readings in ADJACENT zones,        readings and maintenance alerts
  not just the source zone.               regardless of actual failure.

Trap 3: Fish Abundance Bonus            Trap 4: Water Temperature Confound

  Zones adjacent to storm get +3/boat     Zone age offsets water temp readings.
  bonus (storm currents bring fish).      Zone A always reads warm, not because
  Simpson's Paradox: storm-adjacent       of tide. Agent must subtract zone
  zones look profitable overall, but      offset to correctly infer tide state.
  only if the storm isn't IN that zone.
```

## Agent Descriptions

### CausalLearner
Discovers POMDP parameters from the 30-day historical database via 2 SQL queries on day 1. Classifies historical days by reward values to identify storm/equipment/tide states, then estimates buoy distributions, equipment age offsets, maintenance Poisson rates, water temperature parameters, and transition matrices. Runs exact Bayesian filtering with estimated (imperfect) parameters for days 1-20.

### CausalReasoner
Has the true POMDP parameters hardcoded. Runs exact Bayesian filtering (predict → belief_update → optimal_action) with correct causal likelihoods. Uses SQL to unlock Tier 2 observations. Near-oracle performance.

### OracleAgent
Reads the hidden state directly (cheats). Picks the optimal allocation considering fish abundance bonus and tide bonus. Submits exact beliefs (Brier = 0). Upper bound on performance.

### LLMAgent
Free-form tool-calling agent. The LLM receives observations, calls tools (SQL, analysis, reports), and submits decisions. Must discover the causal structure, maintain beliefs, do inference, and plan — all in-context. No separation of concerns.

### LLM+Solver (LLMSolverAgent)
LLM estimates world model parameters on day 1 from raw historical data. The LLM receives the data + a blank parameter schema (field names only, no causal explanations). It must discover what the parameters mean from data patterns. A deterministic solver then runs exact Bayesian filtering with the LLM's estimated model for days 1-20. Isolates model discovery (LLM) from inference/planning (solver).

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd RL-environment
uv sync
```

For LLM agents, create a `.env` file in the parent directory:

```
OPENAI_API_KEY=sk-...
```

## Usage

### Run baseline ablation suite (no API key needed)

```bash
uv run python main.py
```

Runs 5 baselines x 3 ablation configs x 10 seeds = 150 episodes (parallelized across CPU cores). Verifies decomposition identity, baseline ordering, and tool use gaps.

### Run LLM benchmark (requires OpenAI API key)

```bash
uv run python run_llm_benchmark.py
```

Runs LLMAgent and LLM+Solver (GPT 5.4) on 5 seeds. Saves traces and updates `benchmark_results.md`.

### Run individual LLM+Solver episode

```bash
uv run python run_llm_solver.py 42       # seed 42
```

### Run tests

```bash
uv run pytest tests/test_fishing_game.py -v
```

118 tests covering config, POMDP, simulator, evaluator, baselines, ablation suite, and LLM+Solver.

## Project Structure

```
RL-environment/
├── fishing_game/
│   ├── config.py              # EASY_CONFIG, HARD_CONFIG, BENCHMARK_CONFIG (80 states)
│   ├── simulator.py           # FishingGameEnv — SQLite DB, tools, sensor zone subsampling
│   ├── pomdp.py               # FishingPOMDP — exact Bayesian belief updates, optimal actions
│   ├── evaluator.py           # Evaluator — 3-way cost decomposition, Brier scores
│   ├── baselines.py           # 5 baselines: Random, NaivePattern, CausalLearner, CausalReasoner, Oracle
│   ├── llm_agent.py           # LLMAgent base class, tool schemas, system prompt
│   ├── llm_solver_agent.py    # LLMSolverAgent — LLM discovers model, solver does Bayes
│   ├── gpt_agent.py           # GPTAgent — OpenAI API integration
│   ├── runner.py              # Parallelized ablation suite runner with verification
│   └── traced_runner.py       # TracedLLMAgent + LLM+Solver trace support
├── tests/
│   └── test_fishing_game.py   # 118 tests
├── main.py                    # Entry point for baseline ablation suite
├── run_llm_benchmark.py       # Run LLMAgent + LLM+Solver on 5 seeds
├── run_llm_solver.py          # Run single LLM+Solver episode
└── benchmark_results.md       # Latest benchmark results
```

## Configs

| Parameter | EASY_CONFIG | HARD_CONFIG | BENCHMARK_CONFIG |
|---|---|---|---|
| States | 80 | 80 | 80 |
| Sensor zones/day | 4 (all) | 4 (all) | **2 (random)** |
| Sensor noise | Low | High | High |
| Age confound | 0.10 | 0.15 | 0.15 |
| Tool budgets | 2/day | 1/day | 1/day |
| Tide bonus | 2/boat | 1/boat | 1/boat |
| Episode length | 20 days | 20 days | 20 days |
| Max boats | 10 | 10 | 10 |

`CONFIG = BENCHMARK_CONFIG` is the default used everywhere.

## Evaluation Metrics

| Metric | What it measures |
|---|---|
| `total_reward` | Cumulative fishing profit over 20 days |
| `mean_brier_storm` | Belief calibration on storm presence |
| `mean_brier_equip` | Belief calibration on equipment failure |
| `detection_lag` | Days between risk onset and agent detecting it |
| `tool_use_gap` | Cost of not gathering available information |
| `inference_gap` | Cost of misinterpreting gathered information |
| `planning_gap` | Cost of not acting optimally on stated beliefs |

The three gaps sum exactly to `oracle_reward - actual_reward` at every step (algebraic invariant, verified in tests).

## Results (BENCHMARK_CONFIG, 5 seeds)

| Agent | Reward (mean) | Brier(S) | Brier(E) | Tool Gap | Inf Gap | Plan Gap |
|---|---:|---:|---:|---:|---:|---:|
| Random | 472.6 | 0.2500 | 0.3460 | 558.0 | 516.8 | 126.6 |
| NaivePattern | 472.6 | 0.2068 | 0.2173 | 558.0 | 282.2 | 361.2 |
| **LLMAgent (GPT 5.4)** | **720.0** | — | — | — | — | — |
| **CausalLearner** | **1480.0** | **0.0685** | **0.1855** | **0.0** | **102.0** | **0.0** |
| **LLM+Solver (GPT 5.4)** | **1524.0** | — | — | **0.0** | — | **0.0** |
| CausalReasoner | 1582.0 | 0.0484 | 0.1817 | 0.0 | 0.0 | 0.0 |
| Oracle | 1716.0 | 0.0000 | 0.0000 | 0.0 | -134.0 | 0.0 |

**Key observations:**

- **LLM+Solver (1524) > CausalLearner (1480)**: GPT 5.4's parameter estimation is competitive with the hardcoded statistical pipeline, particularly for buoy/storm parameters (2-4% error).
- **LLM+Solver (1524) < CausalReasoner (1582)**: The LLM's estimated model is imperfect — equipment age confound and maintenance rates have 25-75% error. The inference gap captures this.
- **LLMAgent (720) << LLM+Solver (1524)**: Separating model discovery from inference/planning doubles performance. The free-form LLM struggles to maintain consistent beliefs and do in-context Bayes.
- **Planning gap = 0** for all Bayesian agents (CausalLearner, LLM+Solver, CausalReasoner): exact solver always acts optimally on its beliefs.
- **Random ≈ NaivePattern (473)**: With only 2 sensor zones visible per day, NaivePattern's heuristics become unreliable. Both are far below agents that use SQL for causal discovery.

### Benchmark Ladder

```
Random (473) ≈ NaivePattern (473) << LLMAgent (720) << CausalLearner (1480) ≈ LLM+Solver (1524) < CausalReasoner (1582) < Oracle (1716)
```
