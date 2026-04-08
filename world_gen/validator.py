"""Validation framework for generated POMDP configs."""

from __future__ import annotations

import numpy as np


class ValidationError(Exception):
    """Raised when a config fails validation."""


def validate_config(cfg: dict, strict: bool = True) -> dict:
    """Validate a generated config."""
    results = {"valid": True, "errors": [], "warnings": []}
    try:
        _validate_states(cfg)
        _validate_transition_matrices(cfg, results)
        _validate_observation_informativeness(cfg, results)
        _validate_reward_structure(cfg, results)
        _validate_sensor_budget(cfg, results)
    except ValidationError as exc:
        results["valid"] = False
        results["errors"].append(str(exc))
        raise

    _check_confound_structure(cfg, results)
    _check_allocation_diversity(cfg, results)
    if strict and results["warnings"]:
        raise ValidationError(f"Validation warnings (strict mode): {results['warnings']}")
    return results


def _validate_states(cfg: dict) -> None:
    states = cfg.get("states", [])
    if len(states) != 80:
        raise ValidationError(f"Expected 80 states, got {len(states)}")
    for i, state in enumerate(states):
        if not isinstance(state, tuple) or len(state) != 4:
            raise ValidationError(f"State {i} is not a 4-tuple: {state}")
        storm, wind, equip, tide = state
        if storm not in [0, 1]:
            raise ValidationError(f"State {i}: storm not in {{0,1}}")
        if wind not in ["N", "S", "E", "W"]:
            raise ValidationError(f"State {i}: wind not in {{N,S,E,W}}")
        if equip not in [0, 1, 2, 3, 4]:
            raise ValidationError(f"State {i}: equip not in {{0,1,2,3,4}}")
        if tide not in [0, 1]:
            raise ValidationError(f"State {i}: tide not in {{0,1}}")
    if set(cfg.get("zones", [])) != {"A", "B", "C", "D"}:
        raise ValidationError(f"Zones must be {{A, B, C, D}}, got {cfg.get('zones')}")


def _validate_transition_matrices(cfg: dict, results: dict) -> None:
    matrices = {"storm_transition": 2, "wind_transition": 4, "equip_transition": 5, "tide_transition": 2}
    for name, expected_size in matrices.items():
        matrix = np.array(cfg.get(name))
        if matrix.shape != (expected_size, expected_size):
            raise ValidationError(f"{name}: shape {matrix.shape}, expected ({expected_size}, {expected_size})")
        if not np.allclose(np.sum(matrix, axis=1), 1.0, atol=1e-5):
            raise ValidationError(f"{name}: rows don't sum to 1")
        if np.any(matrix < -1e-6):
            raise ValidationError(f"{name}: negative entries detected")
        diag = np.diag(matrix)
        if np.any(diag >= 0.99999):
            results["warnings"].append(f"{name}: potential absorbing states (diagonal >= 0.99999)")
        entropies = [_entropy(row) for row in matrix]
        mean_entropy = np.mean(entropies)
        max_entropy = np.log(expected_size)
        if mean_entropy < 0.05 * max_entropy:
            results["warnings"].append(f"{name}: very low row entropy {mean_entropy:.3f} (max {max_entropy:.3f})")
        if mean_entropy > 0.95 * max_entropy:
            results["warnings"].append(f"{name}: very high row entropy {mean_entropy:.3f} (nearly uniform)")


def _validate_observation_informativeness(cfg: dict, results: dict) -> None:
    bar_params = cfg.get("barometer_params", {})
    if bar_params:
        d_prime_bar = _compute_d_prime(
            bar_params[0]["mean"],
            bar_params[0]["std"],
            bar_params[1]["mean"],
            bar_params[1]["std"],
        )
        if not (0.5 <= d_prime_bar <= 3.0):
            raise ValidationError(f"Barometer d_prime out of range: {d_prime_bar:.2f} (expected [0.5, 3.0])")

    buoy_params = cfg.get("buoy_params", {})
    if buoy_params:
        d_prime_buoy = _compute_d_prime(
            buoy_params["source"]["mean"],
            buoy_params["source"]["std"],
            buoy_params["normal"]["mean"],
            buoy_params["normal"]["std"],
        )
        if not (0.5 <= d_prime_buoy <= 3.0):
            results["warnings"].append(f"Buoy d_prime {d_prime_buoy:.2f}: informative but may be low")

    equip_params = cfg.get("equipment_inspection_params", {})
    if equip_params:
        d_prime_equip = _compute_d_prime(
            equip_params["broken"]["mean"],
            equip_params["broken"]["std"],
            equip_params["ok"]["mean"],
            equip_params["ok"]["std"],
        )
        if d_prime_equip < 0.35 or d_prime_equip > 3.0:
            raise ValidationError(f"Equipment inspection d_prime out of range: {d_prime_equip:.2f}")
        if d_prime_equip < 0.5:
            results["warnings"].append(f"Equipment inspection d_prime very low {d_prime_equip:.2f}")

    for key in [0, 1]:
        _validate_prob_dict(cfg.get("sea_color_probs", {}).get(key), f"sea_color_probs[{key}]")
        _validate_prob_dict(cfg.get("equip_indicator_probs", {}).get(key), f"equip_indicator_probs[{key}]")


def _validate_reward_structure(cfg: dict, results: dict) -> None:
    safe = cfg.get("safe_profit_per_boat", 0)
    loss = abs(cfg.get("danger_loss_per_boat", 0))
    loss_equip = abs(cfg.get("danger_loss_equip_per_boat", 0))
    loss_both = abs(cfg.get("danger_loss_both_per_boat", 0))
    if safe <= 0:
        raise ValidationError(f"safe_profit_per_boat must be positive, got {safe}")
    if loss < safe * 0.5:
        results["warnings"].append(f"Reward asymmetry low: loss={loss}, profit={safe}")
    if loss > safe * 10:
        results["warnings"].append(f"Reward asymmetry extreme: loss={loss}, profit={safe}")
    if loss_both < max(loss, loss_equip):
        results["warnings"].append(f"danger_loss_both should be worst: both={loss_both}, loss={loss}, equip={loss_equip}")


def _validate_sensor_budget(cfg: dict, results: dict) -> None:
    sensor_zones = cfg.get("sensor_zones_per_step", 4)
    n_zones = len(cfg.get("zones", []))
    if not (1 <= sensor_zones <= n_zones):
        raise ValidationError(f"sensor_zones_per_step must be in [1, {n_zones}], got {sensor_zones}")
    if sensor_zones >= n_zones and cfg.get("episode_length", 20) > 5:
        results["warnings"].append(f"Full observability (sensor_zones={sensor_zones}) may be too easy")


def _check_confound_structure(cfg: dict, results: dict) -> None:
    knobs = cfg.get("_knobs", {})
    trap_strength = knobs.get("trap_strength", 0.0)
    age_offset = cfg.get("equipment_age_offset_factor", 0.0)
    if trap_strength > 0.3 and age_offset < 0.05:
        results["warnings"].append(f"High trap_strength {trap_strength} but low age_offset {age_offset}")
    fish_bonus = cfg.get("fish_abundance_bonus", {}).get(1, 0)
    if trap_strength > 0.3 and fish_bonus == 0:
        results["warnings"].append(f"High trap_strength {trap_strength} but zero fish_bonus")
    zone_temp = cfg.get("zone_temp_offset", {})
    if zone_temp:
        temp_spread = max(zone_temp.values()) - min(zone_temp.values())
        if trap_strength > 0.3 and temp_spread < 0.5:
            results["warnings"].append(f"High trap_strength {trap_strength} but low temp_offset spread {temp_spread}")


def _check_allocation_diversity(cfg: dict, results: dict) -> None:
    allocations = cfg.get("valid_allocations", [])
    if len(allocations) < 10:
        results["warnings"].append(f"Very few valid allocations: {len(allocations)}")
    multi_zone_allocs = [allocation for allocation in allocations if sum(1 for value in allocation.values() if value > 0) >= 2]
    if allocations and len(multi_zone_allocs) < len(allocations) * 0.3:
        results["warnings"].append(f"Few multi-zone allocations: {len(multi_zone_allocs)} / {len(allocations)}")


def _validate_prob_dict(probs: dict | None, label: str) -> None:
    if probs is None:
        raise ValidationError(f"Missing probability dict: {label}")
    if not all(0.0 <= value <= 1.0 for value in probs.values()):
        raise ValidationError(f"{label} has invalid probabilities")
    if not np.isclose(sum(probs.values()), 1.0, atol=1e-3):
        raise ValidationError(f"{label} doesn't sum to 1")


def _compute_d_prime(mean_0: float, std_0: float, mean_1: float, std_1: float) -> float:
    pooled_std = np.sqrt((std_0 ** 2 + std_1 ** 2) / 2.0)
    if pooled_std < 1e-6:
        return 0.0
    return abs(mean_0 - mean_1) / pooled_std


def _entropy(probs: list | np.ndarray) -> float:
    probs = np.array(probs)
    probs = probs[probs > 1e-10]
    return -np.sum(probs * np.log2(probs))
