"""Procedural generator for fishing game configs."""

from __future__ import annotations

import itertools
import random

from world_gen.knobs import WorldKnobs


FIXED_REWARD_SCHEDULE = {
    "safe_profit_per_boat": 7,
    "danger_loss_per_boat": -18,
    "danger_loss_equip_per_boat": -10,
    "danger_loss_both_per_boat": -25,
}


def generate_config(knobs: WorldKnobs, seed: int | None = None) -> dict:
    """Generate a complete fishgame config from knobs and seed."""
    if seed is None:
        seed = knobs.seed

    rng = random.Random(seed)
    zones = ["A", "B", "C", "D"]
    winds = ["N", "S", "E", "W"]
    wind_to_zone = {w: z for w, z in zip(winds, zones)}
    states = [(s, w, e, t) for s in [0, 1] for w in winds for e in [0, 1, 2, 3, 4] for t in [0, 1]]
    equip_to_zone = {0: None, 1: "A", 2: "B", 3: "C", 4: "D"}
    zone_to_equip = {"A": 1, "B": 2, "C": 3, "D": 4}
    zone_adjacency = {
        "A": {"A": 0, "B": 1, "C": 2, "D": 1},
        "B": {"A": 1, "B": 0, "C": 1, "D": 2},
        "C": {"A": 2, "B": 1, "C": 0, "D": 1},
        "D": {"A": 1, "B": 2, "C": 1, "D": 0},
    }

    storm_T = _make_2x2_transition(knobs.transition_alpha, rng)
    wind_T = _make_nxn_transition(4, knobs.transition_alpha, rng)
    equip_T = _make_equip_transition(5, knobs.transition_alpha, rng)
    tide_T = _make_2x2_transition(knobs.transition_alpha * 0.8, rng)

    config = {
        "states": states,
        "zones": zones,
        "storm_transition": storm_T,
        "wind_transition": wind_T,
        "equip_transition": equip_T,
        "tide_transition": tide_T,
        "tide_to_label": {0: "low", 1: "high"},
        "tide_bonus": {0: 0, 1: max(0, int(_lerp(1.0, 2.0, knobs.d_prime / 3.0)))},
        "wind_to_zone": wind_to_zone,
        "equip_to_zone": equip_to_zone,
        "zone_to_equip": zone_to_equip,
        "zone_adjacency": zone_adjacency,
        "sea_color_probs": _make_sea_color_from_d_prime(knobs.d_prime),
        "equip_indicator_probs": _make_equip_indicator_from_d_prime(knobs.d_prime),
        "barometer_params": _make_barometer_from_d_prime(knobs.d_prime),
        "buoy_params": _make_buoy_from_d_prime(knobs.d_prime, knobs.trap_strength),
        "equipment_inspection_params": _make_equip_inspection_from_d_prime(knobs.d_prime),
        "equipment_age_offset_factor": _lerp(0.05, 0.20, knobs.trap_strength),
        "zone_infrastructure_age": _make_zone_ages(zones, knobs.trap_strength, rng),
        "maintenance_alert_params": {
            "age_rate_factor": _lerp(0.1, 0.4, knobs.trap_strength),
            "failure_signal": _lerp(2.0, 7.0, knobs.trap_strength),
        },
        "water_temp_params": {
            "base": {"mean": 15.0, "std": _lerp(2.0, 1.0, knobs.d_prime / 3.0)},
            "tide_effect": _lerp(2.0, 0.5, knobs.trap_strength),
        },
        "zone_temp_offset": _make_temp_offsets(zones, knobs.trap_strength),
        "fish_abundance_bonus": _make_fish_bonus(knobs.trap_strength),
        "signal_tiers": _make_signal_tiers(),
        "equipment_signal_tiers": _make_equipment_signal_tiers(),
        "signal_sources": ["coast_guard", "market_data", "industry_news", "social_media"],
        "signals_per_step_range": (2, 5),
        **FIXED_REWARD_SCHEDULE,
        "episode_length": knobs.episode_length,
        "max_boats": knobs.max_boats,
        "valid_allocations": _generate_valid_allocations(knobs.max_boats, zones),
        "tool_budgets": _make_tool_budgets(knobs.sensor_zones),
        "initial_belief": [1.0 / len(states)] * len(states),
        "sensor_zones_per_step": knobs.sensor_zones,
        "_knobs": {
            "difficulty": knobs.difficulty,
            "d_prime": knobs.d_prime,
            "transition_alpha": knobs.transition_alpha,
            "sensor_zones": knobs.sensor_zones,
            "reward_asymmetry": knobs.reward_asymmetry,
            "trap_strength": knobs.trap_strength,
            "seed": seed,
            "name": knobs.name,
        },
    }
    return config


def _make_tool_budgets(sensor_zones: int) -> dict:
    if sensor_zones >= 3:
        return {
            "check_weather_reports": 2,
            "check_equipment_reports": 2,
            "query_fishing_log": 2,
            "query_maintenance_log": 2,
            "analyze_data": 1,
            "evaluate_options": 1,
            "forecast_scenario": 1,
        }
    return {
        "check_weather_reports": 1,
        "check_equipment_reports": 1,
        "query_fishing_log": 1,
        "query_maintenance_log": 1,
        "analyze_data": 1,
        "evaluate_options": 1,
        "forecast_scenario": 1,
    }


def _make_2x2_transition(alpha: float, rng: random.Random) -> list[list[float]]:
    p_stay_0 = _lerp(0.5, 0.95, alpha / 2.0)
    p_stay_1 = _lerp(0.5, 0.90, alpha / 2.0)
    p_stay_0 = max(0.1, min(0.99, p_stay_0 + rng.gauss(0, 0.02)))
    p_stay_1 = max(0.1, min(0.99, p_stay_1 + rng.gauss(0, 0.02)))
    return [[p_stay_0, 1.0 - p_stay_0], [1.0 - p_stay_1, p_stay_1]]


def _make_nxn_transition(n: int, alpha: float, rng: random.Random) -> list[list[float]]:
    p_stay = _lerp(1.0 / n, 0.85, alpha / 2.0)
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(p_stay if i == j else (1.0 - p_stay) / (n - 1))
        row = [max(0.01, value + rng.gauss(0, 0.01)) for value in row]
        total = sum(row)
        matrix.append([value / total for value in row])
    return matrix


def _make_equip_transition(n: int, alpha: float, rng: random.Random) -> list[list[float]]:
    p_onset = _lerp(0.15, 0.05, alpha / 2.0) * (n - 1)
    p_onset = min(0.6, p_onset)
    p_recover = _lerp(0.5, 0.15, alpha / 2.0)
    matrix = []
    row0 = [1.0 - p_onset]
    per_zone = p_onset / (n - 1)
    for _ in range(n - 1):
        row0.append(max(0.01, per_zone + rng.gauss(0, 0.005)))
    total = sum(row0)
    matrix.append([value / total for value in row0])
    for i in range(1, n):
        row = [p_recover]
        for j in range(1, n):
            row.append(1.0 - p_recover - 0.02 * (n - 2) if j == i else 0.02)
        row = [max(0.01, value + rng.gauss(0, 0.005)) for value in row]
        total = sum(row)
        matrix.append([value / total for value in row])
    return matrix


def _make_barometer_from_d_prime(d_prime: float) -> dict:
    std = 5.0
    gap = d_prime * std
    base = 1010.0
    return {0: {"mean": base, "std": std}, 1: {"mean": base - gap, "std": std}}


def _make_buoy_from_d_prime(d_prime: float, trap_strength: float) -> dict:
    std = 0.8
    normal_mean = 1.5
    source_mean = _lerp(2.2, 3.5, d_prime / 3.0)
    prop_mean = _lerp(normal_mean + 0.5, source_mean - 0.3, trap_strength)
    far_mean = _lerp(normal_mean + 0.2, prop_mean - 0.2, trap_strength)
    return {
        "normal": {"mean": round(normal_mean, 2), "std": std},
        "source": {"mean": round(source_mean, 2), "std": std},
        "propagated": {"mean": round(prop_mean, 2), "std": std},
        "far_propagated": {"mean": round(far_mean, 2), "std": std},
    }


def _make_equip_inspection_from_d_prime(d_prime: float) -> dict:
    std = _lerp(2.0, 0.5, d_prime / 3.0)
    gap = max(0.6, min(2.5, d_prime)) * std / 1.5
    ok_mean = 3.0
    return {
        "broken": {"mean": round(ok_mean + gap, 1), "std": round(std, 1)},
        "ok": {"mean": round(ok_mean, 1), "std": round(std, 1)},
    }


def _make_sea_color_from_d_prime(d_prime: float) -> dict:
    p_dark_safe = _lerp(0.25, 0.02, d_prime / 3.0)
    p_green_storm = _lerp(0.25, 0.02, d_prime / 3.0)
    return {
        0: _normalize_dict({"green": 1.0 - p_dark_safe - 0.2, "murky": 0.2, "dark": p_dark_safe}),
        1: _normalize_dict({"green": p_green_storm, "murky": 0.3, "dark": 1.0 - p_green_storm - 0.3}),
    }


def _make_equip_indicator_from_d_prime(d_prime: float) -> dict:
    p_crit_ok = _lerp(0.20, 0.02, d_prime / 3.0)
    p_norm_broken = _lerp(0.30, 0.05, d_prime / 3.0)
    return {
        0: _normalize_dict({"normal": 1.0 - p_crit_ok - 0.15, "warning": 0.15, "critical": p_crit_ok}),
        1: _normalize_dict({"normal": p_norm_broken, "warning": 0.35, "critical": 1.0 - p_norm_broken - 0.35}),
    }


def _make_zone_ages(zones: list[str], trap_strength: float, rng: random.Random) -> dict:
    if trap_strength < 0.05:
        return {zone: 10 for zone in zones}
    max_age = int(_lerp(10, 35, trap_strength))
    min_age = int(_lerp(5, 1, trap_strength))
    ages = sorted([rng.randint(min_age, max_age) for _ in zones], reverse=True)
    return {zone: age for zone, age in zip(zones, ages)}


def _make_temp_offsets(zones: list[str], trap_strength: float) -> dict:
    offset_range = _lerp(0.0, 2.5, trap_strength)
    n_zones = len(zones)
    return {
        zone: round(offset_range * (1.0 - 2.0 * i / max(1, n_zones - 1)), 1)
        for i, zone in enumerate(zones)
    }


def _make_fish_bonus(trap_strength: float) -> dict:
    return {0: 0, 1: int(_lerp(0, 5, trap_strength)), 2: 0}


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def _normalize_dict(values: dict) -> dict:
    safe_values = {key: max(0.01, value) for key, value in values.items()}
    total = sum(safe_values.values())
    normalized = {key: value / total for key, value in safe_values.items()}
    first_key = next(iter(normalized))
    normalized[first_key] += 1.0 - sum(normalized.values())
    return normalized


def _generate_valid_allocations(max_boats: int, zones: list[str]) -> list[dict]:
    allocations = []
    for total in range(1, max_boats + 1):
        for combo in itertools.combinations_with_replacement(range(len(zones)), total):
            counts = [0] * len(zones)
            for idx in combo:
                counts[idx] += 1
            allocation = {zones[i]: counts[i] for i in range(len(zones))}
            if allocation not in allocations:
                allocations.append(allocation)
    return allocations


def _make_signal_tiers() -> dict:
    return {
        1: {
            "emission_prob": {0: 0.05, 1: 0.75},
            "headlines": [
                "STORM WARNING: gale force winds reported offshore",
                "Coast guard issues severe weather advisory",
                "Harbor master suspends departures",
            ],
        },
        2: {
            "emission_prob": {0: 0.15, 1: 0.50},
            "headlines": [
                "Northern fleet reports unusual swell patterns",
                "Insurance premiums trending upward",
                "Barometric pressure readings inconsistent",
            ],
        },
        3: {
            "emission_prob_always": 0.20,
            "headlines": [
                "Annual fishing quota review scheduled",
                "New sonar technology tested",
                "Local tournament postponed",
                "Jellyfish migration reported",
            ],
        },
    }


def _make_equipment_signal_tiers() -> dict:
    return {
        1: {
            "emission_prob": {0: 0.05, 1: 0.75},
            "headlines": [
                "EQUIPMENT ALERT: Critical failure reported",
                "Maintenance crew reports severe malfunction",
                "Emergency inspection ordered",
            ],
        },
        2: {
            "emission_prob": {0: 0.15, 1: 0.50},
            "headlines": [
                "Fleet logs show unusual wear patterns",
                "Equipment insurance claims rising",
                "Inspection backlogs causing concern",
            ],
        },
        3: {
            "emission_prob_always": 0.20,
            "headlines": [
                "New fishing gear technology showcased",
                "Annual certification process begins",
                "Trade group publishes guidelines",
                "Supplier announces new product",
            ],
        },
    }
