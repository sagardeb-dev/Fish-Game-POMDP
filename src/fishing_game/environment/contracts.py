"""Shared contracts for observation and belief payloads."""

from __future__ import annotations

import numpy as np


def uniform_zone_probs(zones: list[str]) -> dict[str, float]:
    """Return a uniform categorical distribution over zones."""
    if not zones:
        return {}
    p = 1.0 / len(zones)
    return {zone: p for zone in zones}


def observation_bundle_to_pomdp_observations(obs: dict) -> list[tuple[object, object]]:
    """Convert an environment observation bundle into POMDP observations."""
    observations = [
        ("sea_color", obs["sea_color"]),
        ("equip_indicator", obs["equip_indicator"]),
        ("barometer", obs["barometer"]),
    ]
    for zone, value in obs.get("buoy_readings", {}).items():
        observations.append((("buoy", zone), value))
    for zone, value in obs.get("equipment_readings", {}).items():
        observations.append((("equip_inspection", zone), value))
    for zone, value in obs.get("maintenance_alerts", {}).items():
        observations.append((("maintenance_alerts", zone), value))
    for zone, value in obs.get("water_temp_readings", {}).items():
        observations.append((("water_temp", zone), value))
    return observations


def normalize_zone_probs(zone_probs: dict[str, float], zones: list[str]) -> dict[str, float]:
    """Normalize zone probabilities, falling back to uniform when empty."""
    total = sum(float(zone_probs.get(zone, 0.0)) for zone in zones)
    if total <= 0:
        return uniform_zone_probs(zones)
    return {zone: float(zone_probs.get(zone, 0.0)) / total for zone in zones}


def belief_vector_to_decision_beliefs(pomdp, belief) -> dict[str, object]:
    """Project an 80-state belief vector to the payload used by the simulator/evaluator."""
    storm_zone_probs = normalize_zone_probs(pomdp.storm_zone_probs(belief), pomdp.zones)
    equip_zone_probs = normalize_zone_probs(pomdp.equip_zone_probs(belief), pomdp.zones)
    return {
        "storm_active": float(pomdp.p_storm(belief)),
        "storm_zone_probs": storm_zone_probs,
        "equip_failure_active": float(pomdp.p_equip_failure(belief)),
        "equip_zone_probs": equip_zone_probs,
        "tide_high": float(pomdp.p_tide(belief)),
    }


def belief_dict_to_vector(cfg: dict, pomdp, beliefs: dict) -> np.ndarray:
    """Convert the public belief payload back into an 80-state vector."""
    zones = cfg["zones"]
    p_storm = max(0.0, min(1.0, beliefs.get("storm_active", 0.5)))
    p_equip = max(0.0, min(1.0, beliefs.get("equip_failure_active", 0.2)))
    p_tide_high = max(0.0, min(1.0, beliefs.get("tide_high", 0.5)))
    storm_zone_probs = normalize_zone_probs(
        beliefs.get("storm_zone_probs", uniform_zone_probs(zones)),
        zones,
    )
    equip_zone_probs = normalize_zone_probs(
        beliefs.get("equip_zone_probs", uniform_zone_probs(zones)),
        zones,
    )

    belief = np.zeros(pomdp.n_states, dtype=np.float64)
    for i, (storm, wind, equip, tide) in enumerate(cfg["states"]):
        p_s = p_storm if storm == 1 else (1.0 - p_storm)

        zone_for_wind = cfg["wind_to_zone"][wind]
        p_w = storm_zone_probs.get(zone_for_wind, 0.0)

        if equip == 0:
            p_e = 1.0 - p_equip
        else:
            equip_zone = cfg["equip_to_zone"][equip]
            p_e = p_equip * equip_zone_probs.get(equip_zone, 0.0)

        p_t = p_tide_high if tide == 1 else (1.0 - p_tide_high)
        belief[i] = p_s * p_w * p_e * p_t

    total = belief.sum()
    if total > 0:
        belief /= total
    else:
        belief = np.ones(pomdp.n_states, dtype=np.float64) / pomdp.n_states
    return belief
