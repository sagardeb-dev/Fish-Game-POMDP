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
│  ├── LLMSolverAgent (llm_solver_agent.py)                                │
│  │   — LLM estimates world model on day 1, solver does exact Bayes       │
│  │   — Isolates model discovery from inference/planning                  │
│  └── CodingAgent (coding_agent.py) [BUGGED]                              │
│      — Agno framework + PythonTools for persistent code execution        │
│      — Designed to write analysis code, but GPT-5.4 ignores Python REPL  │
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

### CodingAgent [BUGGED]
Uses Agno framework with PythonTools (persistent Python REPL) + FishingGameTools. Designed to write statistical analysis code on day 1, store thresholds in Python variables, and compute risk scores numerically on days 2-20. Currently bugged: GPT-5.4 ignores the Python REPL entirely despite explicit prompting, falling back to SQL queries + natural language reasoning. Scores ~1069 (between NaivePattern and CausalLearner) but should score higher if the coding loop worked.

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

Runs LLMAgent, LLM+Solver, and CodingAgent (GPT 5.4) on 5 seeds. Saves traces and updates `benchmark_results.md`.

### Run individual episodes

```bash
uv run python run_llm_solver.py 42       # LLM+Solver, seed 42
uv run python run_coding_agent.py 42     # CodingAgent, seed 42
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
│   ├── coding_agent.py        # CodingAgent — Agno + PythonTools [BUGGED]
│   ├── runner.py              # Parallelized ablation suite runner with verification
│   └── traced_runner.py       # TracedLLMAgent + LLM+Solver trace support
├── tests/
│   └── test_fishing_game.py   # 118 tests
├── main.py                    # Entry point for baseline ablation suite
├── run_llm_benchmark.py       # Run LLMAgent + LLM+Solver + CodingAgent on 5 seeds
├── run_llm_solver.py          # Run single LLM+Solver episode
├── run_coding_agent.py        # Run single CodingAgent episode
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
| Random | 472.6 | 0.2500 | 0.3460 | 382.8 | 470.0 | 126.6 |
| NaivePattern | 435.4 | 0.2219 | 0.2485 | 382.8 | 507.4 | 126.4 |
| **LLMAgent (GPT 5.4)** | **663** | **0.0870** | **0.2280** | **0.0** | — | — |
| **CodingAgent (GPT 5.4)** | **1069** | — | — | — | — | — |
| **LLM+Solver (GPT 5.4)** | **1124.0** | **0.1870** | **0.3093** | **0.0** | **392.0** | **0.0** |
| **CausalLearner** | **1324.0** | **0.1331** | **0.2152** | **0.0** | **192.0** | **0.0** |
| CausalReasoner | 1516.0 | 0.1236 | 0.2104 | 0.0 | 0.0 | 0.0 |
| Oracle | 1716.0 | 0.0000 | 0.0000 | 0.0 | -200.0 | 0.0 |

*LLMAgent and CodingAgent results are partial (3/5 and 1/5 seeds respectively).*

**Key observations:**

- **CausalLearner (1324) > LLM+Solver (1124)**: With tightened distributions, the LLM's parameter estimation errors matter more. Equipment params are frequently swapped (broken/ok means confused), causing large inference gaps.
- **LLM+Solver (1124) < CausalReasoner (1516)**: The inference gap (392.0) is entirely from imperfect LLM parameter estimates. Planning gap = 0 confirms the solver acts optimally on its beliefs.
- **CodingAgent (1069) [BUGGED]**: Uses Agno framework with PythonTools, but GPT-5.4 ignores the Python REPL entirely — never calls `run_python_code` despite explicit prompting. Falls back to SQL + natural language reasoning. Should score higher if the coding loop worked.
- **LLMAgent (663) << LLM+Solver (1124)**: Free-form LLM struggles with consistent beliefs and in-context Bayes. Separating model discovery from inference nearly doubles performance.
- **Planning gap = 0** for all Bayesian agents (CausalLearner, LLM+Solver, CausalReasoner): exact solver always acts optimally on its beliefs.
- **Random (473) ≈ NaivePattern (435)**: With only 2 sensor zones visible per day, NaivePattern's heuristics become unreliable.

### Benchmark Ladder

```
Random (473) ≈ NaivePattern (435) << LLMAgent (663) < CodingAgent* (1069) < LLM+Solver (1124) < CausalLearner (1324) < CausalReasoner (1516) < Oracle (1716)

* CodingAgent is bugged — does not use Python REPL
```
