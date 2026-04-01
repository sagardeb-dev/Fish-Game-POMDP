# Fishing Game POMDP Benchmark

A POMDP-based benchmark for evaluating LLM agent capabilities in **causal discovery**, **tool use**, **Bayesian inference**, and **decision-making under uncertainty**. The agent manages a fishing fleet across 4 zones over a 20-day season, facing two hidden risks (storms and equipment failures) that must be discovered through database analysis — not told in the prompt.

Inspired by [NewtonBench](https://arxiv.org/abs/2503.02453) (ICLR 2026) and [DiscoveryBench](https://arxiv.org/abs/2407.01725) — agents must discover hidden causal structure through data, not instruction-following.

## Architecture

```
                          ┌─────────────────────────────────────┐
                          │         Hidden Generative Model      │
                          │  40 states = 2(storm) × 4(wind)     │
                          │              × 5(equip_failure)      │
                          │                                      │
                          │  config.py: CONFIG / HARD_CONFIG     │
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
│           │  daily_conditions                │  1 row/day: storm_active,  │
│           │  sensor_log                      │  storm_zone, equip_zone,   │
│           │  catch_history                   │  buoy & equip readings     │
│           │  weather_signals                 │  per zone (30 historical   │
│           │  maintenance_log                 │  + current season)         │
│           │  daily_log                       │                            │
│           └─────────────────────────────────┘                            │
│                                                                          │
│  Observation Tiers:                                                      │
│  ┌─────────────────────────────────┬────────────────────────────────┐    │
│  │ TIER 1 — Free (7 observations)  │ TIER 2 — SQL-Discoverable (8) │    │
│  │ _build_free_observations()      │ _build_sql_discoverable_obs() │    │
│  │                                 │                                │    │
│  │  sea_color    (storm signal)    │  4× buoy_readings   (storm)   │    │
│  │  equip_indicator (equip signal) │  4× equip_readings  (equip)   │    │
│  │  barometer    (storm signal)    │                                │    │
│  │  4× maintenance_alerts (equip)  │  Promoted to agent's trace    │    │
│  │                                 │  only if SQL tools were used   │    │
│  │  Always in evaluator trace      │  this step                    │    │
│  └─────────────────────────────────┴────────────────────────────────┘    │
│                                                                          │
│  Budget-Gated Tools:                                                     │
│  ┌───────────────────────────┬────────┬───────────────────────────────┐  │
│  │ Tool                      │Budget  │ Method                        │  │
│  ├───────────────────────────┼────────┼───────────────────────────────┤  │
│  │ check_weather_reports     │ 2/day  │ env.check_weather_reports()   │  │
│  │ check_equipment_reports   │ 2/day  │ env.check_equipment_reports() │  │
│  │ query_fishing_log  (SQL)  │ 2/day  │ env.query_fishing_log()      │  │
│  │ query_maintenance_log     │ 2/day  │ env.query_maintenance_log()  │  │
│  │ analyze_data (Python)     │ 1/day  │ env.analyze_data()           │  │
│  │ evaluate_options          │ 1/day  │ env.evaluate_options()       │  │
│  │ forecast_scenario         │ 1/day  │ env.forecast_scenario()      │  │
│  └───────────────────────────┴────────┴───────────────────────────────┘  │
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
│  LLMAgent (llm_agent.py)         GPTAgent (gpt_agent.py)                │
│  ├── act(env, obs)               ├── _call_llm() → OpenAI API          │
│  ├── _call_llm()  [abstract]     └── extends LLMAgent                  │
│  └── conversation_history                                               │
│                                                                          │
│  TracedLLMAgent (traced_runner.py)                                      │
│  └── wraps any LLMAgent, captures full I/O trace per step              │
│                                                                          │
│  Baselines (baselines.py):                                               │
│  ├── RandomAgent          — random zone, no tools                       │
│  ├── NaivePatternMatcher  — uses weather/equip reports, no SQL          │
│  ├── CausalReasoner       — full Bayesian + SQL discovery               │
│  └── OracleAgent          — reads hidden state (upper bound)            │
└──────────────────────────────────────────────────────────────────────────┘
        │                                               │
        │ episode trace                                 │ episode trace
        ▼                                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    Evaluator  (evaluator.py)                             │
│                                                                          │
│  evaluate_episode(trace) → per-step and episode-level metrics           │
│                                                                          │
│  Uses FishingPOMDP (pomdp.py) for exact Bayesian belief updates:        │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  For each step:                                                    │  │
│  │    1. Oracle posterior  = POMDP.belief_update(prior, ALL 15 obs)   │  │
│  │    2. Agent posterior   = POMDP.belief_update(prior, agent's obs)  │  │
│  │    3. oracle_reward     = best action under oracle posterior        │  │
│  │    4. retrieved_reward  = best action under agent posterior         │  │
│  │    5. belief_reward     = best action under agent's stated beliefs │  │
│  │    6. actual_reward     = agent's actual action                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  3-Way Cost Decomposition (algebraic identity, holds every step):       │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                                                                    │  │
│  │  oracle_reward - actual_reward =                                   │  │
│  │      tool_use_gap    (oracle - retrieved: did agent gather info?)  │  │
│  │    + inference_gap   (retrieved - belief:  did agent reason well?) │  │
│  │    + planning_gap    (belief - actual:     did agent act on it?)   │  │
│  │                                                                    │  │
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
2. Env emits observations + weather/equipment signals into DB
3. Agent receives observation bundle (Tier 1 sensors + metadata)
4. Agent calls tools (SQL queries, reports, analysis) — day does NOT advance
5. Agent calls submit_decisions(allocation, beliefs, reasoning) — day advances
6. Env computes reward, updates DB, returns next observation
```

### Causal Traps (discoverable, not told)

```
Trap 1: Wave Propagation               Trap 2: Age-Confounded Equipment

  Zones form a ring: A─B─C─D─A           Zone ages: A=25yr B=15yr C=5yr D=2yr

  Storm in zone X causes elevated         Old zones always show high equip
  buoy readings in ADJACENT zones,        readings regardless of failure.
  not just the source zone.               Agent must use historical baselines
                                          to calibrate, not raw readings.
  Discoverable via:
  SELECT dc.storm_zone, sl.zone,          Discoverable via:
    AVG(sl.buoy_reading)                  SELECT sl.zone,
  FROM sensor_log sl                        AVG(sl.equipment_reading)
  JOIN daily_conditions dc                FROM sensor_log sl
    ON sl.day = dc.day                    JOIN daily_conditions dc
  WHERE dc.storm_active = 1                ON sl.day = dc.day
  GROUP BY dc.storm_zone, sl.zone         WHERE dc.equip_zone != sl.zone
  → source ~4.5, adjacent ~2.8,          GROUP BY sl.zone
    far ~1.6                              → A ~4.5, B ~3.5, C ~2.5, D ~2.2

Trap 3: Fish Abundance Bonus (Simpson's Paradox)
  Zones adjacent to an active storm get +3 reward/boat
  (more fish driven by storm currents). Naive analysis
  shows storm-adjacent zones are profitable — but ONLY
  if the storm isn't IN that zone (-18/boat).
```

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd RL-environment
uv sync
```

For running LLM agents, create a `.env` file in the parent directory with your API key:

```
OPENAI_API_KEY=sk-...
```

## Usage

### Run baseline ablation suite (no API key needed)

```bash
uv run python main.py
```

Runs 4 baselines x 3 ablation configs x 5 seeds = 60 episodes. Prints comparison table and verifies:
- Decomposition identity at every step
- Baseline ordering: Random < NaivePattern < CausalReasoner <= Oracle
- Tool use gaps: positive for non-SQL agents, ~0 for SQL agents

### Run LLM agent (requires OpenAI API key)

```bash
# Easy mode
uv run python run_gpt_v3.py easy 42 gpt-5-mini

# Hard mode (noisier sensors, tighter budgets)
uv run python run_gpt_v3.py hard 42 gpt-5-mini
```

Arguments: `<mode> <seed> <model>`

Traces are saved to `traces/<model>_v4_<difficulty>_seed<seed>.json`.

### Run tests

```bash
uv run pytest tests/test_fishing_game.py -v
```

89 tests covering simulator, evaluator, baselines, decomposition identity, and ablation suite.

## Project Structure

```
RL-environment/
├── fishing_game/
│   ├── config.py          # CONFIG and HARD_CONFIG dicts (40 states, rewards, noise params)
│   ├── simulator.py       # FishingGameEnv — SQLite DB, tools, observation tiers
│   ├── pomdp.py           # FishingPOMDP — exact Bayesian belief updates, optimal actions
│   ├── evaluator.py       # Evaluator — 3-way cost decomposition, Brier scores
│   ├── baselines.py       # 4 baselines: Random, NaivePattern, CausalReasoner, Oracle
│   ├── llm_agent.py       # LLMAgent base class, tool schemas, system prompt
│   ├── gpt_agent.py       # GPTAgent — OpenAI API integration
│   ├── runner.py          # Ablation suite runner with verification
│   └── traced_runner.py   # TracedLLMAgent — full I/O capture for LLM runs
├── tests/
│   └── test_fishing_game.py  # 89 tests
├── main.py                # Entry point for baseline ablation suite
├── run_gpt_v3.py          # Entry point for LLM agent runs
├── fish-game.md           # Original design specification
└── pyproject.toml         # Dependencies: numpy, openai, python-dotenv
```

## Difficulty Modes

| Parameter | Easy (`CONFIG`) | Hard (`HARD_CONFIG`) |
|---|---|---|
| Sensor noise | Low | High |
| Age confound strength | 0.10 | 0.15 |
| Tool budgets | 2/day each | 1/day each |
| Storm persistence | 0.80 | 0.80 |
| Episode length | 20 days | 20 days |

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

## Results (hard mode, seed 42)

| Agent | Reward | Brier(S) | Brier(E) | Tool Gap | Inf Gap | Plan Gap |
|---|---:|---:|---:|---:|---:|---:|
| Random | 109.0 | 0.2500 | 0.1300 | 93.0 | 32.0 | 231.0 |
| NaivePattern | 233.0 | 0.2198 | 0.1684 | 93.0 | 9.0 | 130.0 |
| **GPT-5.4** | **354.0** | **0.0586** | **0.1598** | **93.0** | **18.0** | **0.0** |
| CausalReasoner | 465.0 | 0.0075 | 0.0620 | 0.0 | 0.0 | 0.0 |
| Oracle | 465.0 | 0.0000 | 0.0000 | 0.0 | 0.0 | 0.0 |

**Key observations:**
- GPT-5.4 scores 354/465 (76% of oracle) — the gap is entirely `tool_use_gap` (93.0) and `inference_gap` (18.0)
- `planning_gap = 0` means GPT-5.4 acts optimally on its own beliefs, but those beliefs are poorly calibrated
- `tool_use_gap = 93.0` (same as Random/NaivePattern) — GPT-5.4 used SQL only 4 times across 20 days, failing to discover the causal structure in the historical database
- CausalReasoner matches Oracle perfectly on this seed — Bayesian inference + SQL discovery closes the full gap
