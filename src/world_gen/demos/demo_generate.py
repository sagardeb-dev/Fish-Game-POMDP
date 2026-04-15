"""Demo script for inspecting generated worlds."""

from __future__ import annotations

import argparse
import json

from world_gen import curriculum_knobs, describe_difficulty, generate_config, validate_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and inspect a fishgame world.")
    parser.add_argument("--level", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--d-prime", dest="d_prime", type=float)
    parser.add_argument("--transition-alpha", dest="transition_alpha", type=float)
    parser.add_argument("--sensor-zones", dest="sensor_zones", type=int)
    parser.add_argument("--trap-strength", dest="trap_strength", type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    overrides = {
        key: value
        for key, value in {
            "d_prime": args.d_prime,
            "transition_alpha": args.transition_alpha,
            "sensor_zones": args.sensor_zones,
            "trap_strength": args.trap_strength,
        }.items()
        if value is not None
    }
    knobs = curriculum_knobs(args.level, **overrides)
    config = generate_config(knobs, seed=args.seed)
    validation = validate_config(config, strict=False)

    print("=" * 80)
    print("WORLD GENERATION DEMO")
    print("=" * 80)
    print(f"Level: {args.level:.2f}")
    print(f"Seed: {args.seed}")
    print()
    print(describe_difficulty(knobs))
    print()
    print("Transition Summary:")
    print(f"  storm_transition: {config['storm_transition']}")
    print(f"  wind_transition row0: {config['wind_transition'][0]}")
    print(f"  equip_transition row0: {config['equip_transition'][0]}")
    print()
    print("Observation Summary:")
    print(f"  barometer: {json.dumps(config['barometer_params'], sort_keys=True)}")
    print(f"  buoy: {json.dumps(config['buoy_params'], sort_keys=True)}")
    print(f"  equipment_inspection: {json.dumps(config['equipment_inspection_params'], sort_keys=True)}")
    print(f"  sea_color_probs: {json.dumps(config['sea_color_probs'], sort_keys=True)}")
    print(f"  equip_indicator_probs: {json.dumps(config['equip_indicator_probs'], sort_keys=True)}")
    print()
    print("Confounds:")
    print(f"  zone_infrastructure_age: {config['zone_infrastructure_age']}")
    print(f"  zone_temp_offset: {config['zone_temp_offset']}")
    print(f"  fish_abundance_bonus: {config['fish_abundance_bonus']}")
    print()
    print("Reward Schedule:")
    print(f"  safe_profit_per_boat: {config['safe_profit_per_boat']}")
    print(f"  danger_loss_per_boat: {config['danger_loss_per_boat']}")
    print(f"  danger_loss_equip_per_boat: {config['danger_loss_equip_per_boat']}")
    print(f"  danger_loss_both_per_boat: {config['danger_loss_both_per_boat']}")
    print()
    print("Validator:")
    print(f"  valid: {validation['valid']}")
    print(f"  warnings: {validation['warnings']}")


if __name__ == "__main__":
    main()
