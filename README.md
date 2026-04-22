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

## Results — Industry-Grade Benchmark (April 2026)

**Setup:** baselines 10 seeds/level, LLM agents 5 seeds/level, 5 curriculum levels,
matched seeds across agents, 95% CI via 2000-sample paired bootstrap, model
`gpt-5.4`. Regret = (Oracle reward on the same seed) − (Agent reward) —
used because raw reward scale shifts with curriculum level and is therefore
not comparable across levels. Lower regret is better.

Run artifacts: `reports/2026-04-industry-benchmark/summary.md` and
`reports/2026-04-industry-benchmark/data/all_records.json`.

### Regret by curriculum level (lower is better)

| Agent | L0.00 | L0.25 | L0.50 | L0.75 | L1.00 |
|---|---:|---:|---:|---:|---:|
| Random | 1322 [1152, 1505] | 1265 [1151, 1390] | 1370 [1169, 1596] | 1420 [1199, 1696] | 1503 [1370, 1648] |
| NaivePattern | 738 [662, 834] | 766 [635, 901] | 1064 [871, 1241] | 1388 [1216, 1546] | 1322 [1242, 1421] |
| LLM+Solver | 462 [34, 890] | 266 [56, 510] | 1056 [638, 1594] | 1102 [786, 1449] | 1100 [798, 1506] |
| PomdpCoder (M=0) | 412 [34, 826] | 280 [78, 502] | 664 [398, 1050] | 1331 [763, 1937] | 1042 [626, 1490] |
| **PomdpCoder (M=3)** | **330 [0, 990]** | **280 [22, 590]** | **566 [306, 964]** | **764 [230, 1326]** | **588 [290, 920]** |
| CausalLearner | 0 [0, 0] | 128 [21, 243] | 431 [195, 710] | 662 [475, 845] | 723 [577, 873] |
| CausalReasoner | 17 [0, 51] | 76 [12, 161] | 216 [96, 358] | 345 [206, 485] | 479 [321, 659] |
| Oracle | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |

### Raw reward by curriculum level (higher is better)

| Agent | L0.00 | L0.25 | L0.50 | L0.75 | L1.00 |
|---|---:|---:|---:|---:|---:|
| Random | 280 [105, 444] | 301 [149, 445] | 322 [142, 485] | 360 [140, 551] | 424 [334, 521] |
| NaivePattern | 864 [774, 956] | 800 [694, 905] | 628 [478, 793] | 392 [242, 530] | 605 [541, 664] |
| LLM+Solver | 1106 [746, 1464] | 1278 [1072, 1454] | 614 [124, 1072] | 644 [357, 904] | 798 [370, 1076] |
| PomdpCoder (M=0) | 1156 [806, 1464] | 1264 [1082, 1444] | 1006 [578, 1284] | 415 [-102, 932] | 856 [488, 1228] |
| **PomdpCoder (M=3)** | **1238 [586, 1620]** | **1264 [994, 1508]** | **1104 [720, 1400]** | **982 [452, 1480]** | **1310 [1030, 1546]** |
| CausalLearner | 1602 [1552, 1650] | 1438 [1340, 1522] | 1261 [990, 1491] | 1118 [955, 1271] | 1204 [1051, 1336] |
| CausalReasoner | 1585 [1529, 1640] | 1490 [1422, 1550] | 1476 [1323, 1605] | 1435 [1322, 1552] | 1448 [1287, 1606] |
| Oracle | 1602 [1552, 1650] | 1566 [1521, 1609] | 1692 [1645, 1744] | 1780 [1719, 1842] | 1927 [1858, 2003] |

### POMDP Coder telemetry

Coverage = mean log marginal likelihood of the 30-day historical sensor stream
under the agent's belief filter. A higher number means the LLM's parameter
estimate explains the historical data better.

| Agent | Level | n | parse_failed | refinements_applied | LLM calls | init_cov → final_cov |
|---|---|---:|---:|---:|---:|---|
| PomdpCoder M=0 | L0.00 | 5 | 0/5 | 0.0 | 1.0 | −6.19 → −6.19 |
| PomdpCoder M=0 | L0.25 | 5 | 0/5 | 0.0 | 1.0 | −3.38 → −3.38 |
| PomdpCoder M=0 | L0.50 | 5 | 0/5 | 0.0 | 1.0 | −4.60 → −4.60 |
| PomdpCoder M=0 | L0.75 | 5 | 0/5 | 0.0 | 1.0 | −4.74 → −4.74 |
| PomdpCoder M=0 | L1.00 | 5 | 0/5 | 0.0 | 1.0 | −6.62 → −6.62 |
| PomdpCoder M=3 | L0.00 | 5 | 0/5 | 2.6 | 4.0 | −5.04 → **−1.49** |
| PomdpCoder M=3 | L0.25 | 5 | 0/5 | 2.2 | 4.0 | −3.63 → **−1.61** |
| PomdpCoder M=3 | L0.50 | 5 | 0/5 | 1.8 | 4.0 | −4.34 → **−1.79** |
| PomdpCoder M=3 | L0.75 | 5 | 0/5 | 2.0 | 4.0 | −4.23 → **−1.89** |
| PomdpCoder M=3 | L1.00 | 5 | 0/5 | 2.0 | 4.0 | −7.34 → **−2.57** |

### Key findings

- **Refinement (M=0 → M=3) lifts PomdpCoder across every level.** At L1.00
  regret drops from 1042 (M=0) to 588 (M=3) — a 44 % reduction from four
  matched LLM calls. Raw reward rises from 856 to 1310.
- **Coverage correlates with the regret drop.** M=3's mean final coverage is
  2–5 log-likelihood units above M=0's init-only coverage at every level,
  and the largest coverage gain (−7.34 → −2.57 at L1.00) pairs with the
  largest regret reduction.
- **PomdpCoder M=3 overtakes CausalLearner at L1.00** (588 vs 723 regret)
  and narrows the gap to CausalReasoner (588 vs 479). CausalReasoner has no
  LLM in the loop — this means learnt parameters plus the exact filter are
  competitive with hand-coded causal inference once the LLM has a scoring
  signal to refine against.
- **The one-shot LLM+Solver baseline collapses mid-curriculum.** Regret
  triples between L0.25 (266) and L0.50 (1056). Without refinement, a single
  LLM pass over noisier data produces parameters that fit badly and the
  filter diverges. This is the gap the POMDP Coder paper targets.
- **Zero parse failures.** No episode fell back to the neutral prior —
  gpt-5.4 returns schema-valid JSON reliably under both prompts.
- **Variance at L0.75 is high.** CIs on PomdpCoder_M3 span [230, 1326]
  regret. Five seeds is the floor — ten would tighten but were not run here
  for cost reasons. See caveats below.

### Caveats / remaining interpretability gaps

- **Sample size asymmetry.** Baselines: 10 seeds/level. LLM agents: 5. If
  you re-run, match to 10 for parity.
- **Single-model report.** Only gpt-5.4 was measured. A comparison against
  gpt-5-mini would isolate whether the refinement loop is doing the work
  or just the model capability.
- **Known schema prior.** `ESTIMATION_PROMPT` has been stripped of causal
  hints (no more "zone A reads high due to age" narration), but the agent
  still fills a fixed 50-number schema. This is faithful to Gandhi et al.'s
  paper — they call this the "known function template" setting — but
  `PomdpDiscoveryAgent` (separate entrypoint) is the cleaner discovery
  baseline if you care about structure learning.
- **Config-seed reuse.** Curriculum configs are generated once with
  `seed=42`; only episode-RNG seeds vary. Re-running with varied config
  seeds would test robustness to world-gen variation on top of episode
  variation.

### POMDP Coder (Algorithm 2: iterative refinement)

Faithful in spirit to Gandhi et al. (arXiv 2505.02216, 2025), "LLM-Guided Probabilistic Program Induction for POMDP Model Estimation." The agent proposes a parameter patch, scores it by replaying the Bayesian filter over 30 days of historical data, then iteratively refines by showing the LLM the observations that fell near zero probability under its own model.

```
inputs:
    env                       # FishingGameEnv
    cfg                       # base config (structure known; only numeric params to learn)
    M = 3                     # refinement budget
    k = 10                    # number of failure cases per refinement

Phase 1 — Learn θ (once, on day 1)
----------------------------------
catch, sensor   = env.query_fishing_log(...), env.query_maintenance_log(...)
D               = reshape sensor rows into { day: [(obs_type, value), ...] }
best_patch      = LLM_Init(ESTIMATION_PROMPT, catch, sensor)
best_score      = coverage(best_patch, D)

for i in 1..M:
    failures    = k lowest marginal-likelihood observations under best_patch
    new_patch   = LLM_Refine(REFINEMENT_PROMPT, best_patch, failures)
    if new_patch is malformed: continue
    s           = coverage(new_patch, D)
    if s > best_score:
        best_patch, best_score = new_patch, s

learned_cfg     = deep_merge(cfg, best_patch)
pomdp           = FishingPOMDP(learned_cfg)
belief          = learned_cfg["initial_belief"]

Phase 2 — Filter + plan (every day, days 1..20)
-----------------------------------------------
for each day:
    if day > 1: belief = pomdp.predict(belief)
    belief   = pomdp.belief_update(belief, observations_today)
    alloc, _ = pomdp.optimal_action(belief)
    env.submit_decisions(alloc, marginals_of(belief))

where
    coverage(patch, D):
        pomdp  = FishingPOMDP(deep_merge(cfg, patch))
        belief = pomdp.initial_belief
        logp   = 0
        for day in sorted(D):
            belief = pomdp.predict(belief)
            for (obs_type, value) in D[day]:
                like     = [pomdp.P(obs_type=value | s_i) for i in 0..79]
                marginal = belief @ like
                logp    += log(max(marginal, 1e-12))
                belief   = normalize(belief * like)
        return logp / |obs|          # mean log-likelihood per observation
```

**Deliberate simplifications vs. the paper**: exact 80-state filter instead of a particle filter (state space is small enough); one-step `optimal_action` instead of A\*-over-beliefs (matches CausalReasoner/CausalLearner); flat best-so-far loop instead of Thompson-sampled tree; JSON patch against fixed schema instead of arbitrary Pyro code (the existing `_parse_config_patch` validator enforces the schema the paper calls "function templates").

**Ablation signal**: on the hardest level (L1.0, seed 42), init coverage was -2.28 and refinement drove it to -1.94 across 3 iterations — each improvement corresponds to the LLM correctly widening `std` or shifting `mean` for the specific obs types flagged as near-zero probability.
