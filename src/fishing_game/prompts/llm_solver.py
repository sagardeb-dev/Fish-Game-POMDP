"""Prompt assets for the fishing game LLM+Solver agent."""

ESTIMATION_PROMPT = """\
You are analyzing historical fishing data for 4 zones: A, B, C, D.
Zone infrastructure ages: A=25yr, B=15yr, C=5yr, D=2yr (affects some readings).

HISTORICAL DATA INTERPRETATION:
Each day, 10 boats sent to one zone. Reward = profit (7 per boat safe) - losses (storm: -18/boat, equip: -10/boat, both: -25/boat).

Key insight: Two hidden risk factors vary independently:
1. STORM: Binary state, affects buoy readings and catch losses
2. EQUIPMENT FAILURE: Can affect any zone, affects equipment readings and maintenance alerts

PARAMETER GUIDE (what to look for in the data):

1. BUOY READINGS: Wave-based signal for storm presence
   - "normal": buoy when no storm (baseline wave activity)
   - "source": buoy in the zone with storm (high waves)
   - "propagated": buoy in adjacent zones to storm (wave propagation)
   - "far_propagated": buoy in opposite zones (barely affected)
   Strategy: Look for days with HIGH catches (safe) vs LOW catches (storm hit).
   When catches are LOW in multiple adjacent zones on the same day, storm hit the SOURCE zone.

2. EQUIPMENT READINGS: Sensor for equipment failure (affected by zone age)
   - "broken": reading when equipment is actually broken in that zone
   - "ok": reading when equipment is healthy
   - "age_offset_factor": older zones (A=25yr) always show higher readings (confound!)
   Strategy: Compare readings across zones. Zone A always reads high due to age.
   Look for days where reward was low ONLY in one zone (equipment failure there).
   Subtract zone age offset to see true equipment signal.

3. MAINTENANCE ALERTS: Count of maintenance issues (Poisson)
   - "age_rate_factor": higher in old zones (age is a confound)
   - "failure_signal": additional alerts when equipment is broken
   Strategy: Old zones have more alerts regardless. Look for day+zone combos with extra alerts.

4. WATER TEMPERATURE: Tide indicator (but confounded by zone age)
   - "base": base temperature
   - "tide_effect": temperature boost during high tide
   - "zone_temp_offset": zone-specific offset (zone A warmer due to age)
   Strategy: Find pairs of days with same zone but different outcomes.
   Look for temperature pattern repeats (repeating high/low patterns = tide cycle).

5. STORM & EQUIP TRANSITIONS: Hidden state persistence
   - storm_transition[i][j]: P(storm_t+1 = j | storm_t = i)
   - equip_transition[i][j]: P(equip_t+1 = j | equip_t = i)
   - 5 equip states: 0=none, 1=zone_A, 2=zone_B, 3=zone_C, 4=zone_D
   Strategy: Look at consecutive days in the catch history.
   If reward is low multiple days in same/adjacent zones, risk persists (high persistence).

6. TIDE TRANSITION: Binary state persistence
   - Look at water_temp pattern. Does high temp cluster? (high tide persistence)

ANALYSIS WORKFLOW:
1. Identify LOW-reward days (when storm or equip hit, reward < 0)
2. Cluster by which zones were hit
3. Estimate when storm affects which zones (buoy readings should spike in source+adjacent)
4. Estimate when equipment fails (maintenance alerts spike, equipment readings spike)
5. Compute zone age offsets by comparing old vs new zone readings
6. Infer transitions from consecutive day patterns

OUTPUT: Single JSON object matching this schema exactly (replace every _ with a number):

{
  "buoy_params": {
    "normal": {"mean": _, "std": _},
    "source": {"mean": _, "std": _},
    "propagated": {"mean": _, "std": _},
    "far_propagated": {"mean": _, "std": _}
  },
  "equipment_inspection_params": {
    "broken": {"mean": _, "std": _},
    "ok": {"mean": _, "std": _}
  },
  "equipment_age_offset_factor": _,
  "maintenance_alert_params": {
    "age_rate_factor": _,
    "failure_signal": _
  },
  "water_temp_params": {
    "base": {"mean": _, "std": _},
    "tide_effect": _
  },
  "zone_temp_offset": {"A": _, "B": _, "C": _, "D": _},
  "storm_transition": [[_, _], [_, _]],
  "wind_transition": [[_, _, _, _], [_, _, _, _], [_, _, _, _], [_, _, _, _]],
  "equip_transition": [[_, _, _, _, _], [_, _, _, _, _], [_, _, _, _, _], [_, _, _, _, _], [_, _, _, _, _]],
  "tide_transition": [[_, _], [_, _]]
}

RULES (CRITICAL):
- All std values must be >= 0.3 (sensors have noise)
- All transition matrix rows must sum to 1.0
- For equip_transition: Row 0 = no-failure state, Rows 1-4 = zone-specific failures
- Transition rows should be "sticky" (diagonal > 0.3) if risk persists across consecutive days
- Buoy/equip/temp means should respect zone age: older zones have higher base readings
- Output valid JSON only (you may include reasoning/analysis text before the JSON block)
"""
