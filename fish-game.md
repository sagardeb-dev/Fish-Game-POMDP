# Fishing Game POMDP — Reference Implementation Spec

A toy environment that is architecturally identical to the main benchmark. Same tools, same database, same signal system, same evaluation, same decomposition. The hidden state is tiny (4 states) so everything is exactly solvable, making it the validation fixture for the full system.

Build the fishing game first. The main benchmark reuses every pattern — different domain, same code paths.

---

## 1. Domain

A fisher goes out each day for 20 days. A hidden storm may be active, hitting one of two fishing zones. The fisher receives noisy weather observations, can use tools to investigate, then decides where to fish and how many boats to send.

---

## 2. Hidden generative model

### 2.1 Latent state S_t

Two variables, 4 total states. Never visible to the agent.

- `storm ∈ {0, 1}`
- `wind ∈ {north, south}`
- derived: `affected_zone = "A" if wind == "north" else "B"`

### 2.2 Transition matrix T(s'|s)

Does not depend on agent actions.

```
From (storm=0, wind=N): → (0,N): 0.68  (0,S): 0.17  (1,N): 0.12  (1,S): 0.03
From (storm=1, wind=N): → (0,N): 0.16  (0,S): 0.04  (1,N): 0.64  (1,S): 0.16
From (storm=0, wind=S): → (0,N): 0.17  (0,S): 0.68  (1,N): 0.03  (1,S): 0.12
From (storm=1, wind=S): → (0,N): 0.04  (0,S): 0.16  (1,N): 0.16  (1,S): 0.64
```

Storm persistence: 80%. Storm onset: 15%. Wind flip: 20%.

### 2.3 Observation emission O(o|s)

**Free observation (every step, no tool cost):**

Sea color:
```
storm=0: {green: 0.70, murky: 0.25, dark: 0.05}
storm=1: {green: 0.05, murky: 0.35, dark: 0.60}
```

**Tool-gated observations (via tool calls only):**

Barometer:
```
storm=0: Normal(mean=1013, std=3)
storm=1: Normal(mean=998,  std=5)
```

Buoy:
```
zone=affected AND storm=1: Normal(mean=4.0, std=0.5)
otherwise:                 Normal(mean=1.2, std=0.3)
```

### 2.4 Signal emission (tiered)

Each step, the generative model emits 2-5 weather signals into a searchable corpus. Signals have hidden tiers the agent cannot see.

**Tier 1 — direct (emitted when disruption is active):**
```
emission_prob if storm=1: 0.80
emission_prob if storm=0: 0.03
```
Example headlines:
- "STORM WARNING: gale force winds reported offshore"
- "Coast guard issues severe weather advisory for fishing zones"
- "Harbor master suspends departures due to storm conditions"

**Tier 2 — inferential (emitted when risk factors are elevated):**
```
emission_prob if storm=1: 0.55
emission_prob if storm=0: 0.10
```
Example headlines:
- "Northern fleet reports unusual swell patterns this week"
- "Insurance premiums for fishing vessels trending upward"
- "Barometric pressure readings inconsistent across coastal stations"

**Tier 3 — noise (emitted regardless of state):**
```
emission_prob: 0.20 always
```
Example headlines:
- "Annual fishing quota review scheduled for next month"
- "New sonar technology tested by research vessel"
- "Local fishing tournament postponed for unrelated reasons"
- "Marine biologists report unusual jellyfish migration"

Each signal gets a unique ID, a day number, a source type ("coast_guard", "market_data", "industry_news", "social_media"), a headline, and a body. The tier and linked state are stored in the generative model for evaluation but never shown to the agent.

### 2.5 Reward function R(s, a)

```
zone != affected_zone OR storm == 0:  reward = +7  × boats
zone == affected_zone AND storm == 1: reward = -18 × boats
```

### 2.6 Impact parameters

The true impact of hidden state on the operational world:
- `safe_profit_per_boat = 7`
- `danger_loss_per_boat = -18`
- `affected_zone = "A" or "B"` (determined by wind)

---

## 3. Episode-local visible database

A fresh SQLite database created on `reset()`. Updated each step after `submit_decisions`. The agent queries this via `run_sql`. Hidden state never appears here.

### Tables

**daily_log** (appended after each day completes):
```sql
CREATE TABLE daily_log (
    day          INTEGER PRIMARY KEY,
    sea_color    TEXT,
    zone_fished  TEXT,
    boats_sent   INTEGER,
    reward       REAL,
    cumulative   REAL
);
```

**weather_signals** (appended each day from signal emission):
```sql
CREATE TABLE weather_signals (
    signal_id    TEXT PRIMARY KEY,
    day          INTEGER,
    source_type  TEXT,
    headline     TEXT,
    body         TEXT
);
```
Note: tier and linked_state are NOT in this table. Agent sees signals but not their ground truth relevance.

**catch_history** (identical data to daily_log, but queryable for analysis):
```sql
CREATE TABLE catch_history (
    day          INTEGER,
    zone         TEXT,
    boats        INTEGER,
    reward       REAL
);
```

### Schema description (provided to agent)

```
Available tables:

daily_log (day, sea_color, zone_fished, boats_sent, reward, cumulative)
  - Your complete fishing log

weather_signals (signal_id, day, source_type, headline, body)
  - Weather and maritime intelligence reports

catch_history (day, zone, boats, reward)
  - Historical catch outcomes by zone
```

---

## 4. Tool surface

Five tools. Budget per day, does not roll over.

### 4.1 search_signals(query: str, max_results: int = 3) → list[dict]
Budget: 2 per day.
Keyword search over the weather_signals table. Returns matching signals sorted by relevance (simple keyword matching + recency boost). Agent chooses what to search for.

### 4.2 run_sql(query: str) → list[dict]
Budget: 2 per day.
Read-only SQL against the visible database. Only SELECT and WITH statements allowed.

### 4.3 run_code(code: str) → str
Budget: 1 per day.
Sandboxed Python execution with math, statistics, random. Agent can analyze patterns, compute posteriors, run correlations on historical data. Returns stdout.

### 4.4 run_optimizer(params: dict) → dict
Budget: 1 per day.
Expected value calculator. Takes:
```json
{
    "storm_probability": 0.6,
    "zone_a_danger_probability": 0.7,
    "risk_tolerance": "neutral"
}
```
Returns optimal (zone, boats) under those beliefs by computing expected reward for all 6 action combinations (2 zones × 3 boat counts).

### 4.5 run_whatif(params: dict) → dict
Budget: 1 per day.
Projects forward N days under a specified scenario. Takes:
```json
{
    "horizon_days": 5,
    "assume_storm_persists": true,
    "assume_zone": "A",
    "strategy": {"zone": "B", "boats": 2}
}
```
Returns projected cumulative reward, projected daily outcomes. Uses the transition model to simulate forward but does NOT advance the live episode. Deterministic under scenario params.

### Tool schemas

All tools exposed as OpenEnv-compatible tool definitions (typed methods with docstrings) so they work with function-calling LLMs and TRL's GRPOTrainer.

---

## 5. Agent interface

### 5.1 Observation bundle (received each day)

```json
{
    "day": 5,
    "days_remaining": 15,
    "sea_color": "murky",
    "yesterday_reward": 14,
    "cumulative_reward": 42,
    "tools_available": ["search_signals", "run_sql", "run_code", "run_optimizer", "run_whatif"],
    "tool_budget": {"search_signals": 2, "run_sql": 2, "run_code": 1, "run_optimizer": 1, "run_whatif": 1},
    "db_schema": "..."
}
```

### 5.2 Belief schema (mandatory every day)

```json
{
    "storm_active": 0.65,
    "zone_a_is_dangerous": 0.70
}
```

Both floats, 0.0 to 1.0. All agents including baselines must submit these. If an agent fails to provide them, score as 0.5 (maximum uncertainty).

### 5.3 Decision submission (world action)

```json
{
    "zone": "B",
    "boats": 3,
    "beliefs": {
        "storm_active": 0.65,
        "zone_a_is_dangerous": 0.70
    },
    "reasoning": "Barometer low, buoy shows high waves in A, fishing B with full fleet."
}
```

This is the ONLY action that advances the day.

### 5.4 Step return

```json
{
    "observation": { ... next day's bundle ... },
    "reward": 21.0,
    "done": false,
    "info": {
        "belief_brier_storm": 0.04,
        "belief_brier_zone": 0.09,
        "bayesian_posterior": [0.02, 0.01, 0.68, 0.29],
        "optimal_action_under_posterior": {"zone": "B", "boats": 3},
        "optimal_reward_under_posterior": 21.0,
        "tools_used_this_step": ["search_signals", "run_sql"],
        "step_cost_decomposition": {
            "inference_gap": 0.0,
            "planning_gap": 0.0,
            "tool_use_gap": 3.5
        }
    }
}
```

The `info` dict is for the evaluator. In evaluation mode the agent can see reward but not info. In debug mode info is visible.

---

## 6. Turn sequence

1. **Generative model advances**: sample s_{t+1} from T(s_t). Deterministic under seed.
2. **Emit observations**: sample sea_color from O(s). Generate 2-5 weather signals from the signal templates. Append signals to visible DB.
3. **Agent receives observation bundle**.
4. **Frozen turn begins**: agent calls tools in any order, up to budget. World clock does not move. Each tool call is a `step()` in OpenEnv that returns a tool result but does not advance the day.
5. **Agent submits decision**: calls `submit_decisions(zone, boats, beliefs, reasoning)`. This is a `step()` in OpenEnv that DOES advance the day.
6. **Environment processes**: compute reward from R(s, a). Update visible DB (daily_log, catch_history). Prepare next observation.
7. **Return**: next observation bundle + reward + info + done.

---

## 7. POMDP model (separate from simulator)

A standalone mathematical object. The agent never touches it. The evaluator and belief-aware baseline use it.

### 7.1 Contents

- `states`: list of 4 states [(0,N), (0,S), (1,N), (1,S)]
- `transition_matrix`: 4×4 matrix as specified in §2.2
- `observation_distributions`: sea_color table, barometer params, buoy params as specified in §2.3
- `reward_function(state, zone, boats)`: as specified in §2.5
- `belief_update(prior, observations) -> posterior`: exact Bayesian filtering. Prior is a length-4 probability vector. observations is a list of (obs_type, obs_value) pairs. Returns posterior length-4 vector.
- `optimal_action(belief) -> (zone, boats)`: computes expected reward for all 6 action combos under the belief, returns the best.
- `solve() -> policy`: full POMDP solution via value iteration over the belief simplex. Returns a mapping from belief regions to actions. (4 states makes this tractable.)

### 7.2 Consistency requirement

The simulator MUST be derived from the same parameters as the POMDP model. Both must use the same T, O, R. A single config dict should parameterize both.

---

## 8. Evaluator

### 8.1 Per-step scoring

At each step, the evaluator:

1. Takes the observation sequence the agent received (free obs + tool results)
2. Runs `POMDP.belief_update(prior, observations)` to get the true Bayesian posterior
3. Computes belief accuracy:
   - `brier_storm = (agent_storm_belief - bayesian_P(storm=1))²`
   - `brier_zone = (agent_zone_a_belief - bayesian_P(affected=A))²`
4. Computes cost decomposition:
   - `oracle_reward`: reward of optimal action under full Bayesian posterior (given all available evidence including tool results)
   - `retrieved_oracle_reward`: reward of optimal action under Bayesian posterior given only the evidence the agent actually retrieved (not all available evidence — if the agent skipped a tool, the oracle doesn't get that info either)
   - `belief_optimal_reward`: reward of optimal action under the agent's stated beliefs
   - `actual_reward`: reward of the agent's actual action
   - `tool_use_gap = oracle_reward - retrieved_oracle_reward` (did the agent gather enough info?)
   - `inference_gap = retrieved_oracle_reward - belief_optimal_reward` (did the agent interpret the info correctly?)
   - `planning_gap = belief_optimal_reward - actual_reward` (did the agent act optimally on its beliefs?)
5. Tracks detection lag: for each storm onset, the step at which agent's `storm_active` first exceeds 0.5

### 8.2 Episode-level metrics

- `total_reward`: sum of daily rewards
- `mean_brier_storm`: average brier score on storm belief
- `mean_brier_zone`: average brier score on zone belief
- `mean_detection_lag`: average steps between storm onset and agent detection
- `total_tool_use_gap`: sum of per-step tool_use_gap
- `total_inference_gap`: sum of per-step inference_gap
- `total_planning_gap`: sum of per-step planning_gap
- `tool_usage_counts`: {search: N, sql: N, code: N, optimizer: N, whatif: N}
- `reward_per_quarter`: [days 1-5, 6-10, 11-15, 16-20] for coherence tracking

---

## 9. Baselines

All must submit beliefs. All use the same simulator interface.

### 9.1 RandomAgent
- Never calls tools
- Random zone, 1 boat
- Beliefs: `{storm_active: 0.5, zone_a_is_dangerous: 0.5}`

### 9.2 NoToolsHeuristic
- Never calls tools
- Uses only sea_color: dark → 1 boat random zone, murky → 2 boats random zone, green → 3 boats random zone
- Beliefs derived from sea_color likelihood ratio only (no history, no tools)

### 9.3 SearchOnlyHeuristic
- Calls `search_signals("storm warning weather")` each day
- If any result contains alert keywords ("storm", "gale", "severe", "warning"), reduces boats to 1 and randomizes zone
- Otherwise 3 boats, random zone
- Beliefs: binary based on whether alert keywords were found

### 9.4 BeliefAwareBaseline
- Runs `POMDP.belief_update` internally from all observations it gathers
- Uses tools strategically: always reads barometer if available, reads buoy if storm probability > 0.4
- Calls `POMDP.optimal_action` on its posterior
- Submits exact posterior as beliefs
- This is the Bayesian reference. Its Brier score should be near-optimal given its tool usage.

### 9.5 OracleAgent
- Reads true hidden state directly (cheats)
- Always fishes safe zone with 3 boats
- Submits exact true state as beliefs (Brier = 0.0)

---

## 10. Ablation configurations

Same environment, different tool subsets enabled. Run each baseline and each LLM agent under every configuration.

| Config | search | sql | code | optimizer | whatif |
|--------|--------|-----|------|-----------|--------|
| full | yes | yes | yes | yes | yes |
| no_search | no | yes | yes | yes | yes |
| search_only | yes | no | no | no | no |
| no_optimizer | yes | yes | yes | no | yes |
| no_whatif | yes | yes | yes | yes | no |
| no_tools | no | no | no | no | no |

Purpose: if ablation X degrades model A more than model B, that capability is a differentiator between models.

---

## 11. Experiment runner

- Takes: agent, ablation config, list of seeds
- For each seed: reset environment, run 20 days, collect episode trace
- For each episode: run evaluator, compute all metrics
- Aggregate across seeds: mean ± std for every metric
- Print comparison table:

```
Agent               | Reward | Brier(storm) | Brier(zone) | Detection Lag | Tool Gap | Inference Gap | Planning Gap
Random              |   70.2 |        0.250 |       0.250 |          inf  |    42.0  |          0.0  |        18.5
NoToolsHeuristic    |   92.4 |        0.180 |       0.250 |          3.1  |    28.0  |          5.2  |         8.3
SearchOnlyHeuristic |  108.3 |        0.150 |       0.230 |          2.4  |    18.0  |          8.1  |         5.6
BeliefAware         |  128.6 |        0.020 |       0.035 |          1.2  |     2.0  |          1.5  |         0.5
Oracle              |  138.0 |        0.000 |       0.000 |          0.0  |     0.0  |          0.0  |         0.0
```

(Numbers are illustrative, not pre-computed.)

---

## 12. Test assertions

- Same seed → identical hidden trajectory, identical observations, identical baseline scores
- Agent never sees `_storm`, `_wind`, `_affected_zone`, signal tier, or signal linked_state
- Visible database contains no hidden state columns
- Tool budget enforced: over-budget call returns error string
- `submit_decisions` is the only action that advances the day counter
- `run_sql` only accepts SELECT/WITH, rejects writes
- `run_whatif` never mutates live episode state
- Baseline ordering on every tested seed: Random < NoToolsHeuristic < SearchOnlyHeuristic < BeliefAwareBaseline < Oracle
- Oracle Brier scores = 0.0
- BeliefAwareBaseline Brier scores < 0.05
- Decomposition identity holds at every step: `tool_use_gap + inference_gap + planning_gap = oracle_reward - actual_reward`

---

## 13. What NOT to build

- No Docker, HTTP, or FastAPI wrapping (run in-process)
- No RL training loop or gradient computation
- No domain other than fishing
- No multi-echelon propagation (single step: storm → zone → reward)
- No web UI

---

## 14. Milestone

Build all components. Run all 5 baselines × all 6 ablation configs × 5 seeds = 150 episodes. Print the full comparison table. Verify baseline ordering holds under every config. Verify decomposition identity holds at every step. That's done.