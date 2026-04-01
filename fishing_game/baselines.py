"""
Four baseline agents for the Fishing Game v4 — Discoverable causal structure.

All submit allocation dicts + expanded beliefs (10 marginals).
All use the same simulator interface.

1. RandomAgent — random allocation, uniform beliefs, no tools
2. NaivePatternMatcher — falls for ALL causal traps, no SQL → Tier 1 only
3. CausalReasoner — correct Bayesian inference, uses SQL → Tier 1 + Tier 2
4. OracleAgent — reads true hidden state, uses SQL → Tier 1 + Tier 2

Expected ordering: Random < NaivePatternMatcher < CausalReasoner <= Oracle
"""

import random as stdlib_random
import numpy as np
from fishing_game.pomdp import FishingPOMDP
from fishing_game.config import CONFIG


def _uniform_zone_probs():
    return {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}


def _default_beliefs():
    return {
        "storm_active": 0.5,
        "storm_zone_probs": _uniform_zone_probs(),
        "equip_failure_active": 0.2,
        "equip_zone_probs": _uniform_zone_probs(),
    }


class RandomAgent:
    """Never calls tools. Random allocation of 1 boat. Uniform beliefs."""

    def __init__(self, config=None):
        self.cfg = config or CONFIG

    def act(self, env, obs, rng=None):
        rng = rng or stdlib_random
        zone = rng.choice(self.cfg["zones"])
        allocation = {z: 0 for z in self.cfg["zones"]}
        allocation[zone] = 1

        return env.submit_decisions(
            allocation=allocation,
            beliefs=_default_beliefs(),
            reasoning="Random agent: no tools, random zone, 1 boat.",
        )


class NaivePatternMatcher:
    """
    Falls for ALL causal traps despite having all free sensor data.

    Trap 1 (wave propagation): Avoids ALL zones with elevated buoy readings,
    not just the storm source. Treats buoy > 2.0 as "dangerous".

    Trap 2 (age confound): Picks the zone with highest raw maintenance alerts
    as the broken zone, ignoring that old zones always have high alerts.
    Also trusts raw equipment inspection readings without age correction.

    Result: Worse than v2's NoToolsHeuristic despite having MORE data,
    because the extra data misleads without causal reasoning.
    """

    def __init__(self, config=None):
        self.cfg = config or CONFIG

    def act(self, env, obs, rng=None):
        rng = rng or stdlib_random
        color = obs["sea_color"]
        equip_ind = obs["equip_indicator"]
        buoys = obs["buoy_readings"]
        equip_readings = obs["equipment_readings"]
        maint_alerts = obs["maintenance_alerts"]

        # Storm belief from sea_color (correct)
        p_color_s0 = self.cfg["sea_color_probs"][0][color]
        p_color_s1 = self.cfg["sea_color_probs"][1][color]
        p_storm = p_color_s1 / (p_color_s0 + p_color_s1)

        # TRAP 1: Naive buoy interpretation — mark ALL elevated zones as dangerous
        dangerous_zones = set()
        for z, reading in buoys.items():
            if reading > 2.0:  # Propagated waves also exceed this threshold!
                dangerous_zones.add(z)

        # Equip belief from equip_indicator (correct)
        p_ind_e0 = self.cfg["equip_indicator_probs"][0][equip_ind]
        p_ind_e1 = self.cfg["equip_indicator_probs"][1][equip_ind]
        p_equip = p_ind_e1 / (p_ind_e0 + p_ind_e1)

        # TRAP 2: Naive equipment — pick zone with highest raw alert count
        # This always picks old Zone A (~7.5 base alerts) even when it's fine
        worst_equip_zone = max(maint_alerts, key=maint_alerts.get)

        # TRAP 2b: Also trust raw inspection readings without age correction
        # Zone A (age=25) reads ~4.5 even when ok, Zone D (age=2) reads ~2.2 when ok
        worst_inspection_zone = max(equip_readings, key=equip_readings.get)

        # Build naive zone probs: put most weight on highest-alert zone
        equip_zone_probs = {z: 0.1 for z in self.cfg["zones"]}
        equip_zone_probs[worst_equip_zone] = 0.7

        # Storm zone probs: put most weight on zones with high buoy readings
        storm_zone_probs = {}
        total_buoy = sum(max(0, v - 1.2) for v in buoys.values())
        for z in self.cfg["zones"]:
            if total_buoy > 0:
                storm_zone_probs[z] = max(0, buoys[z] - 1.2) / total_buoy
            else:
                storm_zone_probs[z] = 0.25

        # Allocation: avoid dangerous zones naively
        # This often avoids 3 zones (source + 2 adjacent) leaving only 1 option
        equip_avoid = {worst_equip_zone, worst_inspection_zone}
        safe_zones = [z for z in self.cfg["zones"]
                      if z not in dangerous_zones and z not in equip_avoid]

        if not safe_zones:
            # Everything looks dangerous — pick least-buoy zone
            safe_zones = [min(buoys, key=buoys.get)]

        # Boat count based on perceived risk
        if p_storm > 0.6 or p_equip > 0.6:
            boats = 1
        elif p_storm > 0.3 or p_equip > 0.3:
            boats = 2
        else:
            boats = 3

        zone = rng.choice(safe_zones)
        allocation = {z: 0 for z in self.cfg["zones"]}
        allocation[zone] = boats

        beliefs = {
            "storm_active": p_storm,
            "storm_zone_probs": storm_zone_probs,
            "equip_failure_active": p_equip,
            "equip_zone_probs": equip_zone_probs,
        }

        return env.submit_decisions(
            allocation=allocation,
            beliefs=beliefs,
            reasoning=f"NaivePattern: buoy_dangerous={dangerous_zones}, "
                      f"worst_equip={worst_equip_zone}, boats={boats}, zone={zone}.",
        )


class CausalReasoner:
    """
    Full Bayesian inference with correct causal likelihoods.

    Uses ALL free sensor data with proper causal models:
    - Wave propagation: identifies storm SOURCE by comparing buoy pattern
    - Age-adjusted equipment: subtracts age baseline from maintenance alerts
    - Proper Bayesian belief update over 40-state space

    Near-oracle performance when sensors are informative.
    """

    def __init__(self, config=None):
        self.cfg = config or CONFIG
        self.pomdp = FishingPOMDP(self.cfg)
        self.belief = None

    def reset(self):
        self.belief = np.array(self.cfg["initial_belief"], dtype=np.float64)

    def act(self, env, obs, rng=None):
        rng = rng or stdlib_random

        if self.belief is None:
            self.reset()

        # Trigger SQL tool to promote Tier 2 observations (uses 1 of 2 daily budget slots)
        env.query_fishing_log("SELECT 1")

        # Prediction step (transition model)
        if obs["day"] > 1:
            self.belief = self.pomdp.predict(self.belief)

        # Gather ALL observations for Bayesian update (Tier 1 + Tier 2)
        observations = [
            ("sea_color", obs["sea_color"]),
            ("equip_indicator", obs["equip_indicator"]),
            ("barometer", obs["barometer"]),
        ]
        for zone in self.cfg["zones"]:
            observations.append((("buoy", zone), obs["buoy_readings"][zone]))
        for zone in self.cfg["zones"]:
            observations.append((("equip_inspection", zone), obs["equipment_readings"][zone]))
        for zone in self.cfg["zones"]:
            observations.append((("maintenance_alerts", zone), obs["maintenance_alerts"][zone]))

        # Full Bayesian update with causal likelihoods
        self.belief = self.pomdp.belief_update(self.belief, observations)

        # Optimal action under current belief
        alloc, er = self.pomdp.optimal_action(self.belief)

        # Extract marginals for reporting
        p_storm = float(self.pomdp.p_storm(self.belief))
        p_equip = float(self.pomdp.p_equip_failure(self.belief))
        storm_zp = {z: float(self.pomdp.p_storm_zone(self.belief, z))
                     for z in self.cfg["zones"]}
        equip_zp = {z: float(self.pomdp.p_equip_zone(self.belief, z))
                     for z in self.cfg["zones"]}

        # Normalize zone probs
        szp_sum = sum(storm_zp.values())
        if szp_sum > 0:
            storm_zp_norm = {z: v / szp_sum for z, v in storm_zp.items()}
        else:
            storm_zp_norm = _uniform_zone_probs()

        ezp_sum = sum(equip_zp.values())
        if ezp_sum > 0:
            equip_zp_norm = {z: v / ezp_sum for z, v in equip_zp.items()}
        else:
            equip_zp_norm = _uniform_zone_probs()

        beliefs = {
            "storm_active": p_storm,
            "storm_zone_probs": storm_zp_norm,
            "equip_failure_active": p_equip,
            "equip_zone_probs": equip_zp_norm,
        }

        return env.submit_decisions(
            allocation=alloc,
            beliefs=beliefs,
            reasoning=f"CausalReasoner: P(storm)={p_storm:.3f}, P(equip)={p_equip:.3f}, "
                      f"alloc={alloc}, E[R]={er:.1f}.",
        )


class OracleAgent:
    """
    Reads true hidden state directly (cheats).
    Picks optimal zone considering fish abundance bonus:
    - Adjacent to storm gets +3/boat (10 total vs 7)
    - Avoids storm zone and equip failure zone
    Submits exact true state as beliefs (Brier = 0.0).
    """

    def __init__(self, config=None):
        self.cfg = config or CONFIG
        self.pomdp = FishingPOMDP(self.cfg)

    def act(self, env, obs, rng=None):
        # Trigger SQL tool to promote Tier 2 observations (uses 1 of 2 daily budget slots)
        env.query_fishing_log("SELECT 1")

        storm, wind, equip = env._hidden_state
        storm_zone = self.cfg["wind_to_zone"][wind] if storm == 1 else None
        equip_zone = self.cfg["equip_to_zone"][equip]

        # Use POMDP reward to find optimal allocation
        # This correctly handles fish abundance bonus
        best_alloc = None
        best_reward = -float("inf")
        for alloc in self.cfg["valid_allocations"]:
            r = self.pomdp.reward(env._hidden_state_idx, alloc)
            if r > best_reward:
                best_reward = r
                best_alloc = alloc

        # Exact beliefs
        p_storm = 1.0 if storm == 1 else 0.0
        if storm == 1:
            storm_zp = {z: (1.0 if z == storm_zone else 0.0) for z in self.cfg["zones"]}
        else:
            storm_zp = _uniform_zone_probs()

        p_equip = 1.0 if equip > 0 else 0.0
        if equip > 0:
            equip_zp = {z: (1.0 if z == equip_zone else 0.0) for z in self.cfg["zones"]}
        else:
            equip_zp = _uniform_zone_probs()

        beliefs = {
            "storm_active": p_storm,
            "storm_zone_probs": storm_zp,
            "equip_failure_active": p_equip,
            "equip_zone_probs": equip_zp,
        }

        return env.submit_decisions(
            allocation=best_alloc,
            beliefs=beliefs,
            reasoning=f"Oracle: storm={storm}, wind={wind}, equip={equip}, "
                      f"optimal alloc={best_alloc}, reward={best_reward}.",
        )
