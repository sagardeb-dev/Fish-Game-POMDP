"""
LLM+Solver agent: LLM estimates world model parameters, solver does exact Bayesian inference.

The LLM gets raw historical data + a blank parameter schema (field names only,
no causal explanations). It must discover from the data what the parameters mean.
The solver then runs exact Bayesian filtering + optimal action planning.

This isolates model discovery (LLM) from inference/planning (solver).
"""

import copy
import json
import re
import random as stdlib_random
import numpy as np
from fishing_game.pomdp import FishingPOMDP
from fishing_game.config import CONFIG


# =============================================================================
# Estimation prompt — minimal structure, no causal explanations
# =============================================================================

_ESTIMATION_PROMPT = """\
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


# =============================================================================
# Helpers
# =============================================================================

def _deep_merge(base, patch):
    """Recursively merge patch into base. Only merges keys that exist in base."""
    for key in patch:
        if key not in base:
            continue
        if isinstance(base[key], dict) and isinstance(patch[key], dict):
            _deep_merge(base[key], patch[key])
        else:
            base[key] = patch[key]
    return base


def _parse_config_patch(text):
    """Extract and validate a JSON config patch from LLM response text."""
    # Extract JSON from text (handle ```json fences)
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find outermost { ... } by matching braces
        start = text.find("{")
        if start == -1:
            return {}
        depth = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        json_str = text[start:end]

    try:
        raw = json.loads(json_str)
    except json.JSONDecodeError:
        return {}

    if not isinstance(raw, dict):
        return {}

    validated = {}

    # Validate buoy_params
    if "buoy_params" in raw and isinstance(raw["buoy_params"], dict):
        bp = {}
        for key in ("normal", "source", "propagated", "far_propagated"):
            if key in raw["buoy_params"] and isinstance(raw["buoy_params"][key], dict):
                entry = raw["buoy_params"][key]
                if _is_numeric(entry.get("mean")) and _is_numeric(entry.get("std")):
                    bp[key] = {"mean": float(entry["mean"]),
                               "std": max(float(entry["std"]), 0.3)}
        if len(bp) == 4:
            validated["buoy_params"] = bp

    # Validate equipment_inspection_params
    if "equipment_inspection_params" in raw and isinstance(raw["equipment_inspection_params"], dict):
        ep = {}
        for key in ("broken", "ok"):
            if key in raw["equipment_inspection_params"] and isinstance(raw["equipment_inspection_params"][key], dict):
                entry = raw["equipment_inspection_params"][key]
                if _is_numeric(entry.get("mean")) and _is_numeric(entry.get("std")):
                    ep[key] = {"mean": float(entry["mean"]),
                               "std": max(float(entry["std"]), 0.3)}
        if len(ep) == 2:
            validated["equipment_inspection_params"] = ep

    # Validate equipment_age_offset_factor
    if _is_numeric(raw.get("equipment_age_offset_factor")):
        validated["equipment_age_offset_factor"] = float(raw["equipment_age_offset_factor"])

    # Validate maintenance_alert_params
    if "maintenance_alert_params" in raw and isinstance(raw["maintenance_alert_params"], dict):
        mp = raw["maintenance_alert_params"]
        if _is_numeric(mp.get("age_rate_factor")) and _is_numeric(mp.get("failure_signal")):
            validated["maintenance_alert_params"] = {
                "age_rate_factor": float(mp["age_rate_factor"]),
                "failure_signal": float(mp["failure_signal"]),
            }

    # Validate water_temp_params
    if "water_temp_params" in raw and isinstance(raw["water_temp_params"], dict):
        wt = raw["water_temp_params"]
        if (isinstance(wt.get("base"), dict)
                and _is_numeric(wt["base"].get("mean"))
                and _is_numeric(wt["base"].get("std"))
                and _is_numeric(wt.get("tide_effect"))):
            validated["water_temp_params"] = {
                "base": {"mean": float(wt["base"]["mean"]),
                         "std": max(float(wt["base"]["std"]), 0.3)},
                "tide_effect": float(wt["tide_effect"]),
            }

    # Validate zone_temp_offset
    if "zone_temp_offset" in raw and isinstance(raw["zone_temp_offset"], dict):
        zt = raw["zone_temp_offset"]
        if all(_is_numeric(zt.get(z)) for z in ("A", "B", "C", "D")):
            validated["zone_temp_offset"] = {z: float(zt[z]) for z in ("A", "B", "C", "D")}

    # Validate transition matrices
    for key, shape in [("storm_transition", (2, 2)), ("tide_transition", (2, 2)),
                       ("wind_transition", (4, 4)), ("equip_transition", (5, 5))]:
        if key in raw:
            matrix = _validate_transition_matrix(raw[key], shape)
            if matrix is not None:
                validated[key] = matrix

    return validated


def _is_numeric(val):
    """Check if a value is numeric (int or float)."""
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def _validate_transition_matrix(raw, shape):
    """Validate and normalize a transition matrix. Returns list-of-lists or None."""
    rows, cols = shape
    if not isinstance(raw, list) or len(raw) != rows:
        return None
    matrix = []
    for row in raw:
        if not isinstance(row, list) or len(row) != cols:
            return None
        if not all(_is_numeric(v) for v in row):
            return None
        row_sum = sum(row)
        if row_sum <= 0:
            return None
        if abs(row_sum - 1.0) > 0.1:
            return None  # Too far from normalized
        # Normalize
        matrix.append([v / row_sum for v in row])
    return matrix


def _format_data_tables(catch_data, sensor_data):
    """Format SQL results as text tables for the LLM prompt."""
    lines = []

    lines.append("=== CATCH HISTORY (30 days, 10 boats sent to one zone each day) ===")
    lines.append(f"{'day':>5} {'zone':>5} {'boats':>6} {'reward':>8}")
    for row in catch_data:
        lines.append(f"{row['day']:>5} {row['zone']:>5} {row['boats']:>6} {row['reward']:>8.0f}")

    lines.append("")
    lines.append("=== SENSOR LOG + MAINTENANCE ALERTS (all 4 zones per day) ===")
    lines.append(f"{'day':>5} {'zone':>5} {'buoy':>8} {'equip':>8} {'w_temp':>8} {'alerts':>7}")
    for row in sensor_data:
        lines.append(
            f"{row['day']:>5} {row['zone']:>5} {row['buoy_reading']:>8.2f} "
            f"{row['equipment_reading']:>8.2f} {row['water_temp']:>8.2f} {row['alerts']:>7}"
        )

    return "\n".join(lines)


def _print_estimated_params(patch):
    """Pretty-print the estimated config patch to CLI."""
    def _print_dict(d, prefix=""):
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _print_dict(v, full_key)
            elif isinstance(v, list):
                print(f"  {full_key} = {v}")
            else:
                print(f"  {full_key} = {v}")
    _print_dict(patch)


# =============================================================================
# LLMSolverAgent
# =============================================================================

class LLMSolverAgent:
    """
    LLM estimates world model on day 1, solver runs Bayesian filtering days 1-20.

    The LLM receives raw historical data and a blank parameter schema.
    It must discover what the parameters mean from the data patterns.
    The solver then does exact Bayesian inference with the estimated model.

    This is NOT an LLMAgent subclass — it's a standalone architecture.
    """

    def __init__(self, llm_fn, config=None):
        """
        Args:
            llm_fn: callable(messages: list[dict]) -> str
                Takes OpenAI-style messages, returns text response.
            config: optional config dict override.
        """
        self.llm_fn = llm_fn
        self.cfg = config or CONFIG
        self.pomdp = None
        self.belief = None
        self._learned = False
        # Stored for debugging/traces
        self._raw_model_response = None
        self._parsed_config_patch = None
        self._learned_config = None

    def reset(self):
        self.pomdp = None
        self.belief = None
        self._learned = False
        self._raw_model_response = None
        self._parsed_config_patch = None
        self._learned_config = None

    def _learn_from_history(self, env):
        """Query historical data, call LLM for parameter estimation, build POMDP."""
        print("[LLM+Solver] Querying historical data...", flush=True)

        catch_data = env.query_fishing_log(
            "SELECT day, zone, boats, reward FROM catch_history WHERE day < 0 ORDER BY day"
        )
        if isinstance(catch_data, dict) and "error" in catch_data:
            catch_data = []

        sensor_data = env.query_maintenance_log(
            "SELECT sl.day, sl.zone, sl.buoy_reading, sl.equipment_reading, "
            "sl.water_temp, ml.alerts "
            "FROM sensor_log sl JOIN maintenance_log ml "
            "ON sl.day=ml.day AND sl.zone=ml.zone "
            "WHERE sl.day < 0 ORDER BY sl.day, sl.zone"
        )
        if isinstance(sensor_data, dict) and "error" in sensor_data:
            sensor_data = []

        print(f"[LLM+Solver] Got {len(catch_data)} catch rows, {len(sensor_data)} sensor rows",
              flush=True)

        # Build prompt
        data_text = _format_data_tables(catch_data, sensor_data)
        messages = [
            {"role": "system", "content": _ESTIMATION_PROMPT},
            {"role": "user", "content": data_text},
        ]

        # Call LLM
        print("[LLM+Solver] Calling LLM for parameter estimation...", flush=True)
        response = self.llm_fn(messages)
        self._raw_model_response = response

        # Print LLM reasoning (text before JSON)
        json_start = response.find("{")
        if json_start > 0:
            reasoning = response[:json_start].strip()
            if reasoning:
                print("[LLM+Solver] LLM reasoning:", flush=True)
                for line in reasoning.split("\n"):
                    print(f"  {line}", flush=True)

        # Parse config patch
        patch = _parse_config_patch(response)
        self._parsed_config_patch = patch

        if patch:
            print("[LLM+Solver] Estimated params:", flush=True)
            _print_estimated_params(patch)
        else:
            print("[LLM+Solver] WARNING: No valid params extracted, using defaults", flush=True)

        # Deep merge with config
        learned = copy.deepcopy(self.cfg)
        _deep_merge(learned, patch)
        self._learned_config = learned

        # Build POMDP
        self.pomdp = FishingPOMDP(learned)
        self.belief = np.array(learned["initial_belief"], dtype=np.float64)
        self._learned = True

    def act(self, env, obs, rng=None):
        rng = rng or stdlib_random

        if not self._learned:
            self._learn_from_history(env)
        else:
            # Trigger SQL for Tier 2 observations
            env.query_fishing_log("SELECT 1")

        # Prediction step
        if obs["day"] > 1:
            self.belief = self.pomdp.predict(self.belief)

        # Gather observations (only available sensor zones)
        observations = [
            ("sea_color", obs["sea_color"]),
            ("equip_indicator", obs["equip_indicator"]),
            ("barometer", obs["barometer"]),
        ]
        for zone in obs.get("buoy_readings", {}):
            observations.append((("buoy", zone), obs["buoy_readings"][zone]))
        for zone in obs.get("equipment_readings", {}):
            observations.append((("equip_inspection", zone), obs["equipment_readings"][zone]))
        for zone in obs.get("maintenance_alerts", {}):
            observations.append((("maintenance_alerts", zone), obs["maintenance_alerts"][zone]))
        for zone in obs.get("water_temp_readings", {}):
            observations.append((("water_temp", zone), obs["water_temp_readings"][zone]))

        # Bayesian update
        self.belief = self.pomdp.belief_update(self.belief, observations)

        # Optimal action
        alloc, er = self.pomdp.optimal_action(self.belief)

        # Extract marginals
        p_storm = float(self.pomdp.p_storm(self.belief))
        p_equip = float(self.pomdp.p_equip_failure(self.belief))
        p_tide = float(self.pomdp.p_tide(self.belief))
        storm_zp = {z: float(self.pomdp.p_storm_zone(self.belief, z))
                     for z in self.cfg["zones"]}
        equip_zp = {z: float(self.pomdp.p_equip_zone(self.belief, z))
                     for z in self.cfg["zones"]}

        # Normalize zone probs
        szp_sum = sum(storm_zp.values())
        storm_zp_norm = {z: v / szp_sum for z, v in storm_zp.items()} if szp_sum > 0 else {z: 0.25 for z in self.cfg["zones"]}
        ezp_sum = sum(equip_zp.values())
        equip_zp_norm = {z: v / ezp_sum for z, v in equip_zp.items()} if ezp_sum > 0 else {z: 0.25 for z in self.cfg["zones"]}

        beliefs = {
            "storm_active": p_storm,
            "storm_zone_probs": storm_zp_norm,
            "equip_failure_active": p_equip,
            "equip_zone_probs": equip_zp_norm,
            "tide_high": p_tide,
        }

        # CLI output
        print(
            f"[LLM+Solver] Day {obs['day']}: "
            f"P(storm)={p_storm:.2f} P(equip)={p_equip:.2f} P(tide)={p_tide:.2f} "
            f"alloc={alloc} E[R]={er:.1f}",
            flush=True,
        )

        return env.submit_decisions(
            allocation=alloc,
            beliefs=beliefs,
            reasoning=f"LLM+Solver: P(storm)={p_storm:.3f}, P(equip)={p_equip:.3f}, "
                      f"P(tide)={p_tide:.3f}, alloc={alloc}, E[R]={er:.1f}.",
        )


# =============================================================================
# Mock agent for testing (no API key needed)
# =============================================================================

# Reasonable defaults close to HARD_CONFIG values
_MOCK_CONFIG_PATCH = {
    "buoy_params": {
        "normal": {"mean": 1.5, "std": 0.6},
        "source": {"mean": 3.5, "std": 0.8},
        "propagated": {"mean": 2.5, "std": 0.7},
        "far_propagated": {"mean": 1.8, "std": 0.6},
    },
    "equipment_inspection_params": {
        "broken": {"mean": 6.5, "std": 1.5},
        "ok": {"mean": 3.0, "std": 1.0},
    },
    "equipment_age_offset_factor": 0.15,
    "maintenance_alert_params": {
        "age_rate_factor": 0.3,
        "failure_signal": 5.0,
    },
    "water_temp_params": {
        "base": {"mean": 15.0, "std": 1.5},
        "tide_effect": 1.0,
    },
    "zone_temp_offset": {"A": 1.5, "B": 0.8, "C": 0.0, "D": -0.3},
    "storm_transition": [[0.75, 0.25], [0.40, 0.60]],
    "wind_transition": [
        [0.50, 0.15, 0.20, 0.15],
        [0.15, 0.50, 0.15, 0.20],
        [0.20, 0.15, 0.50, 0.15],
        [0.15, 0.20, 0.15, 0.50],
    ],
    "equip_transition": [
        [0.65, 0.10, 0.08, 0.09, 0.08],
        [0.30, 0.50, 0.07, 0.06, 0.07],
        [0.30, 0.07, 0.50, 0.07, 0.06],
        [0.30, 0.06, 0.07, 0.50, 0.07],
        [0.30, 0.07, 0.06, 0.07, 0.50],
    ],
    "tide_transition": [[0.70, 0.30], [0.35, 0.65]],
}


class MockLLMSolverAgent(LLMSolverAgent):
    """Mock LLM+Solver agent that returns hardcoded config patch. No API key needed."""

    def __init__(self, config=None, patch=None):
        mock_patch = patch or _MOCK_CONFIG_PATCH

        def mock_llm_fn(messages):
            return "Analysis complete.\n```json\n" + json.dumps(mock_patch) + "\n```"

        super().__init__(llm_fn=mock_llm_fn, config=config)
