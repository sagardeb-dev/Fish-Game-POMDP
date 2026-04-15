# Fishing Game POMDP Benchmark

A POMDP-based benchmark for evaluating LLM agent capabilities in **causal discovery**, **tool use**, **Bayesian inference**, and **decision-making under uncertainty**. The agent manages a fishing fleet across 4 zones over a 20-day season, facing hidden risks (storms, equipment failures, tide) that must be discovered through database analysis — not told in the prompt.

Inspired by [NewtonBench](https://arxiv.org/abs/2503.02453) (ICLR 2026) and [DiscoveryBench](https://arxiv.org/abs/2407.01725).

## Architecture

```
                          +-------------------------------------+
                          |         Hidden Generative Model      |
                          |  80 states = 2(storm) x 4(wind)     |
                          |       x 5(equip_failure) x 2(tide)  |
                          +------------------+------------------+
                                             |
                          transitions T(s'|s) + observations O(o|s)
                                             |
+--------------------------------------------------------------------------+
|                     FishingGameEnv  (simulator.py)                        |
|                                                                          |
|  SQLite Episode Database:                                                |
|    catch_history, sensor_log, maintenance_log (30 days historical)       |
|    daily_conditions (HIDDEN - blocked from agent SQL)                    |
|                                                                          |
|  Sensor Zone Subsampling:                                                |
|    Only 2 of 4 zones report sensors each day (randomly selected).        |
|                                                                          |
|  Observation Tiers:                                                      |
|    Tier 1 (free): sea_color, equip_indicator, barometer,                 |
|                   maintenance_alerts, water_temp                         |
|    Tier 2 (SQL):  buoy_readings, equip_readings                         |
|                   (promoted only if SQL tools were used this step)       |
|                                                                          |
|  Budget-Gated Tools (1/day each):                                        |
|    check_weather_reports, check_equipment_reports,                        |
|    query_fishing_log, query_maintenance_log,                             |
|    analyze_data, evaluate_options, forecast_scenario                     |
+--------------------------------------------------------------------------+
         |                                              ^
         | observation + reward                         | tool calls + submit
         v                                              |
+--------------------------------------------------------------------------+
|                           Agent Layer                                    |
|                                                                          |
|  Baselines (no API key):                                                 |
|    RandomAgent, NaivePatternMatcher, CausalLearner,                      |
|    CausalReasoner, OracleAgent                                           |
|                                                                          |
|  LLM Agents (require OpenAI API key):                                    |
|    LLMAgent, LLM+Solver, CodingAgent                                    |
+--------------------------------------------------------------------------+
         |
         v
+--------------------------------------------------------------------------+
|                    Evaluator                                             |
|                                                                          |
|  3-Way Cost Decomposition (algebraic identity):                          |
|    oracle - actual = tool_use_gap + inference_gap + planning_gap         |
+--------------------------------------------------------------------------+
```

### Turn Sequence (each of 20 days)

1. Env transitions hidden state via T(s'|s)
2. Env selects 2 random sensor zones
3. Env emits observations into DB
4. Agent receives observation bundle
5. Agent calls tools (SQL, reports, analysis) -- day does NOT advance
6. Agent calls submit_decisions -- day advances
7. Env computes reward, returns next observation

### Causal Traps (discoverable, not told)

| Trap | Mechanism |
|---|---|
| Wave Propagation | Storm in zone X elevates buoy readings in adjacent zones, not just the source |
| Age-Confounded Equipment | Old zones (A=25yr) always show high equip readings regardless of failure |
| Fish Abundance Bonus | Zones adjacent to storm get +3/boat (Simpson's Paradox) |
| Water Temp Confound | Zone age offsets water temp; must subtract offset to infer tide |

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
python -m scripts.run_baselines
```

### Run LLM benchmark (requires OpenAI API key)

```bash
python -m scripts.run_llm_benchmark
```

### Run individual episodes

```bash
python -m scripts.run_llm_solver 42
python -m scripts.run_coding_agent 42
```

### Run world generator demos

```bash
python -m world_gen.demo_pipeline --level 0.5 --seed 42 --episodes 2 --agents random reasoner oracle
python -m world_gen.demo_generate --level 0.8 --seed 42 --d-prime 1.8
```

### Run tests

```bash
python -m scripts.run_tests
python -m scripts.run_tests tests/unit/ -q
python -m scripts.run_tests tests/integration/ -q
```

## Project Structure

```
RL-environment/
  src/
    fishing_game/
      environment/    config, simulator, pomdp, evaluator, contracts
      agents/         RandomAgent, CausalLearner, LLMAgent, LLM+Solver, etc.
      runners/        ablation suite runner, traced LLM runner
      prompts/        extracted prompt text for LLM agents
    world_gen/
      core/           knobs, generator, validator
      demos/          demo_generate, demo_pipeline
  scripts/            canonical runnable entrypoints
  tests/
    unit/             fishing_game and world_gen unit tests
    integration/      end-to-end generator + demo smoke tests
```

## Configs

| Parameter | EASY | HARD | BENCHMARK |
|---|---|---|---|
| States | 80 | 80 | 80 |
| Sensor zones/day | 4 (all) | 4 (all) | 2 (random) |
| Sensor noise | Low | High | High |
| Age confound | 0.10 | 0.15 | 0.15 |
| Tool budgets | 2/day | 1/day | 1/day |
| Tide bonus | 2/boat | 1/boat | 1/boat |
| Episode length | 20 | 20 | 20 |

`CONFIG = BENCHMARK_CONFIG` is the default.

## Evaluation Metrics

| Metric | Measures |
|---|---|
| `total_reward` | Cumulative fishing profit over 20 days |
| `mean_brier_storm` | Belief calibration on storm presence |
| `mean_brier_equip` | Belief calibration on equipment failure |
| `tool_use_gap` | Cost of not gathering available information |
| `inference_gap` | Cost of misinterpreting gathered information |
| `planning_gap` | Cost of not acting optimally on stated beliefs |

The three gaps sum exactly to `oracle_reward - actual_reward` at every step (algebraic invariant, verified in tests).

## Results (BENCHMARK_CONFIG, 5 seeds)

| Agent | Reward (mean) | Brier(S) | Brier(E) | Tool Gap | Inf Gap | Plan Gap |
|---|---:|---:|---:|---:|---:|---:|
| Random | 472.6 | 0.2500 | 0.3460 | 382.8 | 470.0 | 126.6 |
| NaivePattern | 435.4 | 0.2219 | 0.2485 | 382.8 | 507.4 | 126.4 |
| **LLMAgent (GPT 5.4)** | **663** | **0.0870** | **0.2280** | **0.0** | -- | -- |
| **CodingAgent (GPT 5.4)** | **1069** | -- | -- | -- | -- | -- |
| **LLM+Solver (GPT 5.4)** | **1124.0** | **0.1870** | **0.3093** | **0.0** | **392.0** | **0.0** |
| **CausalLearner** | **1324.0** | **0.1331** | **0.2152** | **0.0** | **192.0** | **0.0** |
| CausalReasoner | 1516.0 | 0.1236 | 0.2104 | 0.0 | 0.0 | 0.0 |
| Oracle | 1716.0 | 0.0000 | 0.0000 | 0.0 | -200.0 | 0.0 |

*LLMAgent and CodingAgent results are partial (3/5 and 1/5 seeds respectively).*

### Benchmark Ladder

```
Random (473) ~ NaivePattern (435) << LLMAgent (663) < CodingAgent* (1069) < LLM+Solver (1124) < CausalLearner (1324) < CausalReasoner (1516) < Oracle (1716)

* CodingAgent is bugged -- does not use Python REPL
```

## Curriculum Learning Results (WorldGen, 5 levels)

Generated configs using `curriculum_knobs()`, testing agent robustness to varying observability and causal structure.

| Agent | L0.0 | L0.25 | L0.5 | L0.75 | L1.0 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Random | 391 | 356 | 317 | 379 | -185 | -576 |
| NaivePattern | 722 | 400 | 168 | 498 | 160 | -562 |
| CausalLearner | 1136 | 769 | 726 | 538 | 68 | -1068 |
| CausalReasoner | 1127 | 819 | 1222 | 686 | 279 | -848 |
| LLM+Solver | 819 | 204 | 125 | 201 | -319 | -1138 |
| Oracle | 1237 | 1238 | 1426 | 1321 | 1534 | +297 |

Key findings:
- Oracle improves with difficulty (+297): richer causal structure is beneficial with full-state observation
- CausalReasoner is most robust (-848): explicit causal reasoning adapts best across levels
- CausalLearner collapses at hard levels (-1068): learning-based estimation fails when data diversity is low
- LLM+Solver fails even on trivial (819 at L0.0): one-shot parameter discovery cannot reliably estimate 50+ parameters
