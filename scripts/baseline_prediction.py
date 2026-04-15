"""
Baseline score prediction based on POMDP architecture analysis.

ARCHITECTURE ANALYSIS (BENCHMARK_CONFIG, Tightened Distributions):
===================================================================

1. STATE SPACE & SENSORS
   - 80 states (2 storm × 4 wind × 5 equip × 2 tide)
   - Only 2 of 4 zones visible per day (sensor_zones_per_step=2)
   - High observation noise (d_prime~0.9-1.0 gap at BENCHMARK level)
   - Forces model reliance (true POMDP, not MDP-like)

2. AGENT COMPETENCIES REQUIRED
   a) Inference: Maintain belief over 80 states, update from partial obs
   b) Planning: Multi-step horizon (discount gamma=0.95, episode_length=20)
   c) Causal reasoning: Distinguish P(Y|X) from P(Y|do(X)) via 4 confounds

3. BASELINE HIERARCHY
   Random:        No belief, random allocation
   NaivePattern:  Tier-1 sensor readings, heuristic rules → falls for all traps
   CausalLearner: Day-1 SQL discovery + Bayesian filtering → imperfect params
   CausalReasoner: True params + exact Bayesian → near-oracle on inference
   Oracle:        Reads true state → upper bound

PREDICTION FRAMEWORK
====================

Per-baseline maximum reward (safe_profit=7, danger_loss=-18, danger_loss_both=-25):
  Max per day: 7*10 boats = 70/day (safe, all zones profitable)
  Min per day: -25*10 = -250/day (both risks hit all boats)
  20-day max: 1400
  20-day min: -5000
  Reasonable range: 400-1400

RANDOM AGENT
-----------
  Allocation: Random zone, random 1-10 boats
  Beliefs: Uniform (50% storm, 20% equip, 25% per-zone)
  Expected behavior:
    - 25% chance boat is in storm zone → -18*allocation loss
    - 20% chance equip hits random zone → -10 per boat
    - 75% otherwise profit ~7*allocation

  Analytical estimate:
    E[allocation] ~ 5 boats
    E[profit/day] = 5 * (0.75*7 - 0.25*18 - 0.20*10)
                  = 5 * (5.25 - 4.5 - 2)
                  = 5 * (-1.25)
                  = -6.25/day
    20-day: -125 (bad, but with high variance due to randomness)

  Actual prediction: 450-550 (high variance, few 1000+ episodes)

NAIVE PATTERN MATCHER
---------------------
  Allocation: Uses Tier-1 free sensors (sea_color, equip_indicator, buoy, water_temp)
  Falls for ALL traps:
    - Wave propagation: Avoids zone with buoy>2.0 (but that includes adjacent+source)
    - Age confound: Mistrust high equip_readings/maintenance in old zone A
    - Fish bonus: Ignores (nets neutral)
    - Temp confound: Picks warmest zone = always zone A (old, confounded)

  Analytical estimate:
    Can see ~2/4 zones per day (sensor_zones=2)
    When both storm and equip hit: high losses to traps
    When only storm: buoy reading misleads (propagates to adjacent)
    When safe: chooses zone A (confounded by age+temp)

  Empirical observation (v2 results):
    - Often avoids actual storm zone (buoy trap) but hits adjacent zones
    - Mistrusts zone A equipment (age confound) but it's actually safest for equip
    - Picks zone A for temperature (confounded)

  Actual prediction: 400-550
    Similar to Random or slightly worse due to active confounding
    (the heuristics make confident WRONG choices)

CAUSAL LEARNER
--------------
  Day 1: Runs 2 SQL queries on 30-day historical data
    Query 1: Classify days by (catch, losses) → identify storm/equip presence
    Query 2: Zone-by-zone statistics → estimate equip_to_zone mapping

  Learns:
    - Storm transition: P(storm_t+1 | storm_t) ≈ true (0.85 in EASY → 0.75 HARD)
    - Wind → zone mapping: Correct (wind encodes zone)
    - Equipment → zone mapping: High error (tightened equip_inspection overlaps)
    - Barometer/buoy params: Estimated with error (tightened d_prime hurts)

  Expected learning error on TIGHTENED config:
    - Storm/wind/tide: ~5-10% error (deterministic in historical data)
    - Equipment: ~20-30% error (tightened readings overlap, hard to classify)
    - Buoy: ~15-20% error (wave propagation confuses source/propagated)
    - Overall: Parameter error reduces expected reward by ~150-250 points vs CausalReasoner

  Days 2-20: Runs exact Bayesian filtering with learned (imperfect) parameters
    - Inference gap from param error: Beliefs are less sharp, suboptimal actions

  Actual prediction: 1200-1350
    Previous v2 on HARD: 1324±219 (5 seeds)
    Tighter distributions should hurt equip discovery specifically
    Prediction: 1100-1250 (slightly lower due to tighter equip overlap)

CAUSAL REASONER
---------------
  Day 1: Hardcoded true POMDP params (storm, wind, equip, tide transitions)
  Days 1-20: Exact Bayesian filtering + optimal action (value iteration on belief space)

  Expected behavior:
    - Storm inference: Near-perfect (barometer is d_prime~1.0, distinct enough)
    - Equipment inference: Tightened equip_inspection (d_prime~0.8) → harder
    - Action selection: Optimal given beliefs (uses full value function)
    - Tool usage: Uses all Tier-2 tools (buoy, inspection) → zero tool_use_gap

  Expected loss vs Oracle:
    - Inference gap from partial observability: belief entropy still ~3-4 bits
    - Planning gap: None (exact solver on true params)
    - Expected: 100-200 point gap below Oracle

  Actual prediction: 1450-1550
    Previous v2 on HARD: 1516±143 (5 seeds)
    Tighter distributions hurt equip slightly → expect ~20-50 point drop
    Prediction: 1400-1480

ORACLE AGENT
------------
  Reads true hidden state directly (cheats)
  Always picks optimal allocation considering:
    - Avoid storm zone (or reduce boats there)
    - Avoid equipment failure zone
    - Exploit tide bonus (prefer high tide in safe zone)
    - Exploit fish abundance (adjacent to storm gets +3/boat)

  Expected behavior:
    - No inference error (knows true state)
    - No planning error (optimal actions)
    - Tool usage: Maximal but irrelevant (already knows state)

  Expected reward:
    Each day averages:
      - 3 safe zones, 1 at risk: E[profit] = 3 zones * 7 boats + risk zone * 0-5 boats
      - Expected: ~1400/20 = 70/day = 1400 total

  Previous v2 on HARD: 1716±53 (5 seeds)
  Tightened slightly, expect: 1650-1750

BASELINE ORDERING & DIFFERENCES
================================
Random (450-550)
  ↓ ~50-150 points
NaivePattern (400-550)
  ↓ ~550-800 points
CausalLearner (1100-1250)
  ↓ ~150-250 points
CausalReasoner (1400-1480)
  ↓ ~200-300 points
Oracle (1650-1750)

KEY INSIGHTS FOR TIGHTENED CONFIG
==================================
1. Observation noise hurts both learners + reasoner roughly equally (d_prime tighter)
2. Equipment discovery should be hardest (equip_inspection_params tightened most)
3. Storm inference should remain strong (barometer gap large enough)
4. Causal traps still active (confound strength same), benefit CausalReasoner/Oracle
5. Sensor availability constraint (2/4 zones) forces model reliance

PREDICTION SUMMARY (5 seeds, BENCHMARK_CONFIG with tightened distributions)
===========================================================================
Random:        mean ~480, std ~180    (wide variance)
NaivePattern:  mean ~430, std ~220    (confounding is active, heuristics hurt)
CausalLearner: mean ~1200, std ~200   (equipment param error cost ~120 points)
CausalReasoner: mean ~1450, std ~150  (tighter obs noise cost ~50 points)
Oracle:        mean ~1700, std ~50    (robust to noise, state-independent)

Expected ordering preserved: Random ≈ NaivePattern << CausalLearner < CausalReasoner < Oracle
"""

def predict_scores():
    return {
        "Random": (480, 180),
        "NaivePattern": (430, 220),
        "CausalLearner": (1200, 200),
        "CausalReasoner": (1450, 150),
        "Oracle": (1700, 50),
    }

if __name__ == "__main__":
    predictions = predict_scores()
    print("=" * 70)
    print("BASELINE SCORE PREDICTIONS (BENCHMARK_CONFIG, Tightened Distributions)")
    print("=" * 70)
    print("\nAgent                Mean    Std")
    print("-" * 40)
    for agent, (mean, std) in predictions.items():
        print(f"{agent:<16} {mean:>6.0f} {std:>6.0f}")
    print("\nExpected ordering: Random ≈ NaivePattern << CausalLearner < CausalReasoner < Oracle")
