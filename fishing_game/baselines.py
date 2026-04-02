"""
Five baseline agents for the Fishing Game v5 — Labels removed + tide/water temp.

All submit allocation dicts + expanded beliefs (10+ marginals).
All use the same simulator interface.

1. RandomAgent — random allocation, uniform beliefs, no tools
2. NaivePatternMatcher — falls for ALL causal traps, no SQL -> Tier 1 only
3. CausalLearner — discovers POMDP params from historical DB via SQL, then Bayesian inference
4. CausalReasoner — correct Bayesian inference, uses SQL -> Tier 1 + Tier 2
5. OracleAgent — reads true hidden state, uses SQL -> Tier 1 + Tier 2

Expected ordering: Random < NaivePatternMatcher < CausalLearner < CausalReasoner <= Oracle
(CausalLearner may match CausalReasoner on easy configs)
"""

import math
import copy
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
        allocation[zone] = rng.randint(1, self.cfg["max_boats"])

        return env.submit_decisions(
            allocation=allocation,
            beliefs=_default_beliefs(),
            reasoning="Random agent: no tools, random zone, random boats.",
        )


class NaivePatternMatcher:
    """
    Falls for ALL causal traps despite having all free sensor data.

    Trap 1 (wave propagation): Avoids ALL zones with elevated buoy readings,
    not just the storm source. Treats buoy > 2.0 as "dangerous".

    Trap 2 (age confound): Picks the zone with highest raw maintenance alerts
    as the broken zone, ignoring that old zones always have high alerts.
    Also trusts raw equipment inspection readings without age correction.

    Trap 3 (fish abundance): Not exploited.

    Trap 4 (water temp confound): Picks warmest zone as "best fishing",
    always choosing zone A (old infrastructure = warm water offset).

    Result: Worse than v2's NoToolsHeuristic despite having MORE data,
    because the extra data misleads without causal reasoning.
    """

    def __init__(self, config=None):
        self.cfg = config or CONFIG

    def act(self, env, obs, rng=None):
        rng = rng or stdlib_random
        color = obs["sea_color"]
        equip_ind = obs["equip_indicator"]
        buoys = obs.get("buoy_readings", {})
        equip_readings = obs.get("equipment_readings", {})
        maint_alerts = obs.get("maintenance_alerts", {})
        water_temps = obs.get("water_temp_readings", {})

        # Storm belief from sea_color (correct)
        p_color_s0 = self.cfg["sea_color_probs"][0][color]
        p_color_s1 = self.cfg["sea_color_probs"][1][color]
        p_storm = p_color_s1 / (p_color_s0 + p_color_s1)

        # TRAP 1: Naive buoy interpretation — mark ALL elevated observed zones as dangerous
        dangerous_zones = set()
        for z, reading in buoys.items():
            if reading > 2.0:
                dangerous_zones.add(z)

        # Equip belief from equip_indicator (correct)
        p_ind_e0 = self.cfg["equip_indicator_probs"][0][equip_ind]
        p_ind_e1 = self.cfg["equip_indicator_probs"][1][equip_ind]
        p_equip = p_ind_e1 / (p_ind_e0 + p_ind_e1)

        # TRAP 2: Naive equipment — pick observed zone with highest raw alert count
        worst_equip_zone = max(maint_alerts, key=maint_alerts.get) if maint_alerts else None
        worst_inspection_zone = max(equip_readings, key=equip_readings.get) if equip_readings else None

        # TRAP 4: Naive water temp — pick warmest observed zone
        if water_temps:
            best_temp_zone = max(water_temps, key=water_temps.get)
        else:
            best_temp_zone = "A"

        # Build naive zone probs: put most weight on highest-alert zone (observed only)
        equip_zone_probs = {z: 0.25 for z in self.cfg["zones"]}
        if worst_equip_zone:
            equip_zone_probs = {z: 0.1 for z in self.cfg["zones"]}
            equip_zone_probs[worst_equip_zone] = 0.7

        # Storm zone probs: from observed buoy readings, uniform for unobserved
        storm_zone_probs = {}
        total_buoy = sum(max(0, v - 1.2) for v in buoys.values())
        for z in self.cfg["zones"]:
            if z in buoys:
                if total_buoy > 0:
                    storm_zone_probs[z] = max(0, buoys[z] - 1.2) / total_buoy
                else:
                    storm_zone_probs[z] = 0.25
            else:
                storm_zone_probs[z] = 0.25
        # Renormalize
        szp_total = sum(storm_zone_probs.values())
        if szp_total > 0:
            storm_zone_probs = {z: v / szp_total for z, v in storm_zone_probs.items()}

        # Allocation: avoid dangerous/broken observed zones, prefer warmest observed zone
        equip_avoid = {z for z in (worst_equip_zone, worst_inspection_zone) if z is not None}
        safe_zones = [z for z in self.cfg["zones"]
                      if z not in dangerous_zones and z not in equip_avoid]

        if not safe_zones:
            safe_zones = [best_temp_zone]

        # Boat count based on perceived risk
        if p_storm > 0.6 or p_equip > 0.6:
            boats = max(1, self.cfg["max_boats"] // 2)
        elif p_storm > 0.3 or p_equip > 0.3:
            boats = max(1, self.cfg["max_boats"] * 3 // 4)
        else:
            boats = self.cfg["max_boats"]

        # Prefer warmest zone if available in safe_zones
        if best_temp_zone in safe_zones:
            zone = best_temp_zone
        else:
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
                      f"worst_equip={worst_equip_zone}, boats={boats}, zone={zone}, "
                      f"warmest={best_temp_zone}.",
        )


class CausalReasoner:
    """
    Full Bayesian inference with correct causal likelihoods.

    Uses ALL free sensor data with proper causal models:
    - Wave propagation: identifies storm SOURCE by comparing buoy pattern
    - Age-adjusted equipment: subtracts age baseline from maintenance alerts
    - Water temperature: properly incorporates tide + zone offset
    - Proper Bayesian belief update over 80-state space

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

        # Gather observations for Bayesian update (only for available sensor zones)
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

        # Full Bayesian update with causal likelihoods
        self.belief = self.pomdp.belief_update(self.belief, observations)

        # Optimal action under current belief
        alloc, er = self.pomdp.optimal_action(self.belief)

        # Extract marginals for reporting
        p_storm = float(self.pomdp.p_storm(self.belief))
        p_equip = float(self.pomdp.p_equip_failure(self.belief))
        p_tide = float(self.pomdp.p_tide(self.belief))
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
            "tide_high": p_tide,
        }

        return env.submit_decisions(
            allocation=alloc,
            beliefs=beliefs,
            reasoning=f"CausalReasoner: P(storm)={p_storm:.3f}, P(equip)={p_equip:.3f}, "
                      f"P(tide)={p_tide:.3f}, alloc={alloc}, E[R]={er:.1f}.",
        )


class CausalLearner:
    """
    Discovers POMDP parameters from historical database via SQL, then runs
    the same Bayesian filtering as CausalReasoner with estimated (imperfect)
    parameters.

    Day 1: Queries catch_history and sensor_log+maintenance_log to estimate
    buoy params, equipment params, maintenance params, water temp params,
    transition matrices, and zone adjacency.

    Days 1-20: predict → update → optimal_action with estimated FishingPOMDP.

    Sources of error (why CausalLearner < CausalReasoner):
    1. Incomplete storm detection (~50% of storm days identifiable)
    2. Small sample (4-8 storm days → noisy estimates)
    3. Wind/equip transitions nearly unestimable
    4. Tide only classifiable on safe days

    Uses SQL tools → tool_use_gap ~0. Has inference_gap > 0 from estimation error.
    """

    # Reward values for 10 boats in 1 zone (used for day classification)
    _REWARD_STORM = -180      # storm only
    _REWARD_EQUIP = -100      # equip failure only
    _REWARD_BOTH = -250       # storm + equip
    _REWARD_SAFE_LO = 70      # safe, low tide, not adjacent
    _REWARD_SAFE_HI = 90      # safe, high tide, not adjacent
    _REWARD_ADJ_LO = 100      # safe, low tide, adjacent to storm
    _REWARD_ADJ_HI = 120      # safe, high tide, adjacent to storm

    # Minimum samples for estimation; below this, use defaults
    _MIN_SAMPLES = 3
    # Minimum std to prevent belief collapse
    _MIN_STD = 0.3

    def __init__(self, config=None):
        self.cfg = config or CONFIG
        self.pomdp = None  # built on day 1
        self.belief = None
        self._learned = False

    def reset(self):
        self.pomdp = None
        self.belief = None
        self._learned = False

    def _learn_from_history(self, env):
        """Run SQL queries and estimate POMDP parameters from historical data."""
        # Query 1: catch history
        catch_data = env.query_fishing_log(
            "SELECT day, zone, boats, reward FROM catch_history WHERE day < 0 ORDER BY day"
        )
        if isinstance(catch_data, dict) and "error" in catch_data:
            catch_data = []

        # Query 2: sensor + maintenance joined
        sensor_data = env.query_maintenance_log(
            "SELECT sl.day, sl.zone, sl.buoy_reading, sl.equipment_reading, "
            "sl.water_temp, ml.alerts "
            "FROM sensor_log sl JOIN maintenance_log ml "
            "ON sl.day=ml.day AND sl.zone=ml.zone "
            "WHERE sl.day < 0 ORDER BY sl.day, sl.zone"
        )
        if isinstance(sensor_data, dict) and "error" in sensor_data:
            sensor_data = []

        # Classify days from rewards
        day_labels = self._classify_days(catch_data)

        # Organize sensor data by day and zone
        sensors = {}
        for row in sensor_data:
            d = row["day"]
            z = row["zone"]
            if d not in sensors:
                sensors[d] = {}
            sensors[d][z] = {
                "buoy": row["buoy_reading"],
                "equip": row["equipment_reading"],
                "water_temp": row["water_temp"],
                "alerts": row["alerts"],
            }

        # Estimate parameters
        buoy_params = self._estimate_buoy_params(day_labels, sensors)
        equip_params, age_offset = self._estimate_equipment_params(day_labels, sensors)
        maint_params = self._estimate_maintenance_params(day_labels, sensors)
        wtemp_params, zone_offsets = self._estimate_water_temp_params(day_labels, sensors)
        transitions = self._estimate_transitions(day_labels)

        # Build learned config
        learned_config = self._build_learned_config(
            buoy_params, equip_params, age_offset, maint_params,
            wtemp_params, zone_offsets, transitions,
        )

        self.pomdp = FishingPOMDP(learned_config)
        self.belief = np.array(learned_config["initial_belief"], dtype=np.float64)
        self._learned = True

    def _classify_days(self, catch_data):
        """Classify historical days from reward values.

        Returns dict: day -> {storm_zone, equip_zone, tide, fished_zone, reward, type}
        """
        labels = {}
        zones = self.cfg["zones"]
        boats = self.cfg["max_boats"]

        # Precompute expected rewards per boat for classification
        storm_reward = self.cfg["danger_loss_per_boat"] * boats
        equip_reward = self.cfg["danger_loss_equip_per_boat"] * boats
        both_reward = self.cfg["danger_loss_both_per_boat"] * boats
        safe_lo = self.cfg["safe_profit_per_boat"] * boats
        safe_hi = (self.cfg["safe_profit_per_boat"] + self.cfg["tide_bonus"].get(1, 0)) * boats
        # Adjacent to storm bonuses
        adj_bonus = self.cfg["fish_abundance_bonus"].get(1, 0) * boats

        for row in catch_data:
            day = row["day"]
            zone = row["zone"]
            reward = row["reward"]
            label = {
                "fished_zone": zone,
                "reward": reward,
                "storm_zone": None,
                "equip_zone": None,
                "tide": None,
                "type": "unknown",
            }

            r = round(reward)

            if r == round(storm_reward):
                # Storm hit the fished zone
                label["storm_zone"] = zone
                label["type"] = "storm_hit"
            elif r == round(equip_reward):
                # Equip failure in fished zone
                label["equip_zone"] = zone
                label["type"] = "equip_hit"
            elif r == round(both_reward):
                # Both storm and equip in fished zone
                label["storm_zone"] = zone
                label["equip_zone"] = zone
                label["type"] = "both_hit"
            elif r == round(safe_lo):
                # Safe, low tide, not adjacent to storm
                label["tide"] = 0
                label["type"] = "safe"
            elif r == round(safe_hi):
                # Safe, high tide, not adjacent
                label["tide"] = 1
                label["type"] = "safe"
            elif r == round(safe_lo + adj_bonus):
                # Safe, low tide, adjacent to storm (fish abundance)
                label["tide"] = 0
                label["type"] = "safe_adjacent"
            elif r == round(safe_hi + adj_bonus):
                # Safe, high tide, adjacent to storm (fish abundance)
                label["tide"] = 1
                label["type"] = "safe_adjacent"

            labels[day] = label

        return labels

    def _estimate_buoy_params(self, day_labels, sensors):
        """Estimate buoy distribution parameters from known storm days."""
        zones = self.cfg["zones"]
        # Default fallback: ring topology A-B-C-D-A
        adjacency = self.cfg["zone_adjacency"]  # use known ring as fallback

        source_readings = []
        propagated_readings = []
        far_readings = []
        normal_readings = []

        for day, label in day_labels.items():
            if day not in sensors:
                continue
            storm_zone = label.get("storm_zone")

            if storm_zone and label["type"] in ("storm_hit", "both_hit"):
                # Known storm day with known source zone
                for z in zones:
                    if z not in sensors[day]:
                        continue
                    reading = sensors[day][z]["buoy"]
                    dist = adjacency[storm_zone][z]
                    if dist == 0:
                        source_readings.append(reading)
                    elif dist == 1:
                        propagated_readings.append(reading)
                    else:
                        far_readings.append(reading)
            elif label["type"] == "safe":
                # Definitely no storm hitting fished zone, but storm may exist elsewhere
                # We can only safely use this as "normal" if we know no storm
                # Conservative: skip — we can't be sure no storm exists
                pass

        # Also collect normal readings from days we're confident had no storm
        # Days with safe reward AND no adjacent bonus → no storm active
        for day, label in day_labels.items():
            if day not in sensors:
                continue
            if label["type"] == "safe":
                # safe + not adjacent → could still have storm elsewhere
                # But low/normal buoy readings on such days are likely no-storm
                for z in zones:
                    if z in sensors[day]:
                        normal_readings.append(sensors[day][z]["buoy"])

        def _safe_stats(readings, default_mean, default_std):
            if len(readings) >= self._MIN_SAMPLES:
                return {
                    "mean": float(np.mean(readings)),
                    "std": max(float(np.std(readings, ddof=1)) if len(readings) > 1 else default_std,
                              self._MIN_STD),
                }
            return {"mean": default_mean, "std": default_std}

        return {
            "normal": _safe_stats(normal_readings, 1.2, 0.5),
            "source": _safe_stats(source_readings, 4.5, 0.8),
            "propagated": _safe_stats(propagated_readings, 2.8, 0.8),
            "far_propagated": _safe_stats(far_readings, 1.6, 0.6),
        }

    def _estimate_equipment_params(self, day_labels, sensors):
        """Estimate equipment inspection parameters and age offset factor."""
        zones = self.cfg["zones"]
        zone_ages = self.cfg["zone_infrastructure_age"]

        ok_readings = {z: [] for z in zones}
        broken_readings = {z: [] for z in zones}

        for day, label in day_labels.items():
            if day not in sensors:
                continue
            equip_zone = label.get("equip_zone")
            if equip_zone and label["type"] in ("equip_hit", "both_hit"):
                for z in zones:
                    if z not in sensors[day]:
                        continue
                    if z == equip_zone:
                        broken_readings[z].append(sensors[day][z]["equip"])
                    else:
                        ok_readings[z].append(sensors[day][z]["equip"])
            elif label["type"] in ("safe", "safe_adjacent"):
                for z in zones:
                    if z in sensors[day]:
                        ok_readings[z].append(sensors[day][z]["equip"])

        # Estimate age offset from OK readings across zones
        # Linear regression: mean_reading_z = base_ok_mean + age_z * offset_factor
        zone_means = {}
        for z in zones:
            if len(ok_readings[z]) >= self._MIN_SAMPLES:
                zone_means[z] = float(np.mean(ok_readings[z]))

        age_offset = 0.1  # default
        base_ok_mean = 2.0  # default
        if len(zone_means) >= 2:
            ages = np.array([zone_ages[z] for z in zone_means])
            means = np.array([zone_means[z] for z in zone_means])
            if np.std(ages) > 0:
                slope, intercept = np.polyfit(ages, means, 1)
                age_offset = max(float(slope), 0.01)
                base_ok_mean = max(float(intercept), 0.5)

        # Broken mean: subtract age offset to get base broken mean
        all_broken = []
        for z in zones:
            for r in broken_readings[z]:
                all_broken.append(r - zone_ages[z] * age_offset)

        if len(all_broken) >= self._MIN_SAMPLES:
            broken_mean = float(np.mean(all_broken))
            broken_std = max(float(np.std(all_broken, ddof=1)), self._MIN_STD)
        else:
            broken_mean = 8.5
            broken_std = 1.5

        # OK std from residuals
        all_ok_residuals = []
        for z in zones:
            expected = base_ok_mean + zone_ages[z] * age_offset
            for r in ok_readings[z]:
                all_ok_residuals.append(r - expected)

        if len(all_ok_residuals) >= self._MIN_SAMPLES:
            ok_std = max(float(np.std(all_ok_residuals, ddof=1)), self._MIN_STD)
        else:
            ok_std = 0.8

        return {
            "broken": {"mean": broken_mean, "std": broken_std},
            "ok": {"mean": base_ok_mean, "std": ok_std},
        }, age_offset

    def _estimate_maintenance_params(self, day_labels, sensors):
        """Estimate maintenance alert Poisson parameters."""
        zones = self.cfg["zones"]
        zone_ages = self.cfg["zone_infrastructure_age"]

        ok_alerts = {z: [] for z in zones}
        broken_alerts = {z: [] for z in zones}

        for day, label in day_labels.items():
            if day not in sensors:
                continue
            equip_zone = label.get("equip_zone")
            if equip_zone and label["type"] in ("equip_hit", "both_hit"):
                for z in zones:
                    if z in sensors[day]:
                        if z == equip_zone:
                            broken_alerts[z].append(sensors[day][z]["alerts"])
                        else:
                            ok_alerts[z].append(sensors[day][z]["alerts"])
            elif label["type"] in ("safe", "safe_adjacent"):
                for z in zones:
                    if z in sensors[day]:
                        ok_alerts[z].append(sensors[day][z]["alerts"])

        # Estimate age_rate_factor from ok_alerts: mean_alerts_z / age_z
        rates = []
        for z in zones:
            if len(ok_alerts[z]) >= self._MIN_SAMPLES and zone_ages[z] > 0:
                rates.append(float(np.mean(ok_alerts[z])) / zone_ages[z])
        age_rate = float(np.mean(rates)) if rates else 0.3

        # Estimate failure_signal from broken alerts
        failure_signals = []
        for z in zones:
            if len(broken_alerts[z]) >= 1:
                expected_base = zone_ages[z] * age_rate
                excess = float(np.mean(broken_alerts[z])) - expected_base
                failure_signals.append(max(excess, 0.0))
        failure_signal = float(np.mean(failure_signals)) if failure_signals else 5.0

        return {"age_rate_factor": age_rate, "failure_signal": failure_signal}

    def _estimate_water_temp_params(self, day_labels, sensors):
        """Estimate water temperature parameters: base, tide_effect, zone offsets."""
        zones = self.cfg["zones"]
        zone_ages = self.cfg["zone_infrastructure_age"]

        lo_temps = {z: [] for z in zones}
        hi_temps = {z: [] for z in zones}
        all_temps = {z: [] for z in zones}

        for day, label in day_labels.items():
            if day not in sensors:
                continue
            tide = label.get("tide")
            if label["type"] in ("safe", "safe_adjacent"):
                for z in zones:
                    if z in sensors[day]:
                        t = sensors[day][z]["water_temp"]
                        all_temps[z].append(t)
                        if tide == 0:
                            lo_temps[z].append(t)
                        elif tide == 1:
                            hi_temps[z].append(t)

        # Estimate zone offsets relative to overall mean
        all_readings = []
        for z in zones:
            all_readings.extend(all_temps[z])
        global_mean = float(np.mean(all_readings)) if all_readings else 15.0

        zone_offsets = {}
        for z in zones:
            if len(all_temps[z]) >= self._MIN_SAMPLES:
                zone_offsets[z] = float(np.mean(all_temps[z])) - global_mean
            else:
                zone_offsets[z] = self.cfg["zone_temp_offset"].get(z, 0.0)

        # Estimate tide effect from difference between high and low tide days
        lo_all = []
        hi_all = []
        for z in zones:
            for t in lo_temps[z]:
                lo_all.append(t - zone_offsets[z])
            for t in hi_temps[z]:
                hi_all.append(t - zone_offsets[z])

        if len(lo_all) >= self._MIN_SAMPLES and len(hi_all) >= self._MIN_SAMPLES:
            tide_effect = float(np.mean(hi_all)) - float(np.mean(lo_all))
            tide_effect = max(tide_effect, 0.0)
            base_mean = float(np.mean(lo_all))
        else:
            tide_effect = 1.5
            base_mean = global_mean

        # Estimate std from residuals
        residuals = []
        for z in zones:
            for t in lo_temps[z]:
                residuals.append(t - (base_mean + zone_offsets[z]))
            for t in hi_temps[z]:
                residuals.append(t - (base_mean + tide_effect + zone_offsets[z]))
        base_std = max(float(np.std(residuals, ddof=1)), self._MIN_STD) if len(residuals) > 1 else 1.0

        wtemp_params = {
            "base": {"mean": base_mean, "std": base_std},
            "tide_effect": tide_effect,
        }
        return wtemp_params, zone_offsets

    def _estimate_transitions(self, day_labels):
        """Estimate factored transition matrices from consecutive-day labels."""
        # Storm transitions: count storm→storm, storm→no_storm, etc.
        storm_counts = [[1, 1], [1, 1]]  # Laplace smoothing
        tide_counts = [[1, 1], [1, 1]]

        sorted_days = sorted(day_labels.keys())
        for i in range(len(sorted_days) - 1):
            d1 = sorted_days[i]
            d2 = sorted_days[i + 1]
            if d2 != d1 + 1:
                continue  # skip non-consecutive

            l1 = day_labels[d1]
            l2 = day_labels[d2]

            # Storm state inference
            s1 = self._infer_storm_state(l1)
            s2 = self._infer_storm_state(l2)
            if s1 is not None and s2 is not None:
                storm_counts[s1][s2] += 1

            # Tide transitions
            t1 = l1.get("tide")
            t2 = l2.get("tide")
            if t1 is not None and t2 is not None:
                tide_counts[t1][t2] += 1

        def _normalize_rows(counts):
            result = []
            for row in counts:
                s = sum(row)
                result.append([x / s for x in row])
            return result

        storm_trans = _normalize_rows(storm_counts)
        tide_trans = _normalize_rows(tide_counts)

        # Wind transitions: very sparse data, use persistence prior
        wind_trans = [
            [0.60, 0.13, 0.14, 0.13],
            [0.13, 0.60, 0.13, 0.14],
            [0.14, 0.13, 0.60, 0.13],
            [0.13, 0.14, 0.13, 0.60],
        ]

        # Equipment transitions: use persistence prior
        equip_trans = [
            [0.78, 0.06, 0.06, 0.05, 0.05],
            [0.38, 0.47, 0.05, 0.05, 0.05],
            [0.38, 0.05, 0.47, 0.05, 0.05],
            [0.38, 0.05, 0.05, 0.47, 0.05],
            [0.38, 0.05, 0.05, 0.05, 0.47],
        ]

        return {
            "storm": storm_trans,
            "wind": wind_trans,
            "equip": equip_trans,
            "tide": tide_trans,
        }

    def _infer_storm_state(self, label):
        """Infer storm active (1) or not (0) from a day label, or None if unknown."""
        if label["type"] in ("storm_hit", "both_hit"):
            return 1
        if label["type"] == "safe_adjacent":
            return 1  # adjacent bonus means storm was active
        if label["type"] == "safe":
            # Safe with no adjacent bonus — could still have storm in non-adjacent zone
            # Conservative: treat as no storm (most likely)
            return 0
        return None  # equip_hit or unknown — can't determine storm state

    def _build_learned_config(self, buoy_params, equip_params, age_offset,
                              maint_params, wtemp_params, zone_offsets, transitions):
        """Assemble config dict from estimated parameters."""
        learned = copy.deepcopy(self.cfg)

        # Override estimated params
        learned["buoy_params"] = buoy_params
        learned["equipment_inspection_params"] = equip_params
        learned["equipment_age_offset_factor"] = age_offset
        learned["maintenance_alert_params"] = maint_params
        learned["water_temp_params"] = wtemp_params
        learned["zone_temp_offset"] = zone_offsets

        # Transition matrices
        learned["storm_transition"] = transitions["storm"]
        learned["wind_transition"] = transitions["wind"]
        learned["equip_transition"] = transitions["equip"]
        learned["tide_transition"] = transitions["tide"]

        # Keep true values for instrument calibration knowledge
        # (sea_color_probs, equip_indicator_probs, barometer_params are "known")
        # These stay as self.cfg originals (already in learned via deepcopy)

        return learned

    def act(self, env, obs, rng=None):
        rng = rng or stdlib_random

        if not self._learned:
            self._learn_from_history(env)
        else:
            # Trigger SQL tool to promote Tier 2 observations
            env.query_fishing_log("SELECT 1")

        # Prediction step
        if obs["day"] > 1:
            self.belief = self.pomdp.predict(self.belief)

        # Gather observations for Bayesian update (only for available sensor zones)
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

        # Bayesian update with learned (estimated) likelihoods
        self.belief = self.pomdp.belief_update(self.belief, observations)

        # Optimal action under current belief
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
            "tide_high": p_tide,
        }

        return env.submit_decisions(
            allocation=alloc,
            beliefs=beliefs,
            reasoning=f"CausalLearner: P(storm)={p_storm:.3f}, P(equip)={p_equip:.3f}, "
                      f"P(tide)={p_tide:.3f}, alloc={alloc}, E[R]={er:.1f}.",
        )


class OracleAgent:
    """
    Reads true hidden state directly (cheats).
    Picks optimal zone considering fish abundance bonus + tide bonus:
    - Adjacent to storm gets +3/boat (+ tide bonus if high tide)
    - Avoids storm zone and equip failure zone
    Submits exact true state as beliefs (Brier = 0.0).
    """

    def __init__(self, config=None):
        self.cfg = config or CONFIG
        self.pomdp = FishingPOMDP(self.cfg)

    def act(self, env, obs, rng=None):
        # Trigger SQL tool to promote Tier 2 observations (uses 1 of 2 daily budget slots)
        env.query_fishing_log("SELECT 1")

        storm, wind, equip, tide = env._hidden_state
        storm_zone = self.cfg["wind_to_zone"][wind] if storm == 1 else None
        equip_zone = self.cfg["equip_to_zone"][equip]

        # Use POMDP reward to find optimal allocation
        # This correctly handles fish abundance bonus + tide bonus
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

        p_tide = 1.0 if tide == 1 else 0.0

        beliefs = {
            "storm_active": p_storm,
            "storm_zone_probs": storm_zp,
            "equip_failure_active": p_equip,
            "equip_zone_probs": equip_zp,
            "tide_high": p_tide,
        }

        return env.submit_decisions(
            allocation=best_alloc,
            beliefs=beliefs,
            reasoning=f"Oracle: storm={storm}, wind={wind}, equip={equip}, tide={tide}, "
                      f"optimal alloc={best_alloc}, reward={best_reward}.",
        )
