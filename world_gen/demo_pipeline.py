"""Generate a world and run baseline agents with verbose terminal logging."""

from __future__ import annotations

import argparse
import statistics

from fishing_game.baselines import CausalLearner, CausalReasoner, NaivePatternMatcher, OracleAgent, RandomAgent
from fishing_game.runner import run_episode
from world_gen import curriculum_knobs, describe_difficulty, generate_config, validate_config


AGENTS = {
    "random": ("Random", RandomAgent),
    "naive": ("NaivePattern", NaivePatternMatcher),
    "learner": ("CausalLearner", CausalLearner),
    "reasoner": ("CausalReasoner", CausalReasoner),
    "oracle": ("Oracle", OracleAgent),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a world and run non-LLM baselines.")
    parser.add_argument("--level", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--d-prime", dest="d_prime", type=float)
    parser.add_argument("--transition-alpha", dest="transition_alpha", type=float)
    parser.add_argument("--sensor-zones", dest="sensor_zones", type=int)
    parser.add_argument("--trap-strength", dest="trap_strength", type=float)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=sorted(AGENTS),
        default=["random", "naive", "learner", "reasoner", "oracle"],
    )
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
    print("WORLD GENERATION PIPELINE")
    print("=" * 80)
    print(f"Level: {args.level:.2f}")
    print(f"Base seed: {args.seed}")
    print(f"Episodes per agent: {args.episodes}")
    print(f"Agents: {', '.join(args.agents)}")
    print(f"Overrides: {overrides if overrides else 'none'}")
    print()
    print(describe_difficulty(knobs))
    print()
    print("Generated World:")
    print(f"  states: {len(config['states'])}")
    print(f"  allocations: {len(config['valid_allocations'])}")
    print(f"  sensor_zones_per_step: {config['sensor_zones_per_step']}")
    print(f"  barometer: {config['barometer_params']}")
    print(f"  buoy: {config['buoy_params']}")
    print(f"  equipment_inspection: {config['equipment_inspection_params']}")
    print(f"  zone_infrastructure_age: {config['zone_infrastructure_age']}")
    print(f"  zone_temp_offset: {config['zone_temp_offset']}")
    print(f"  fish_abundance_bonus: {config['fish_abundance_bonus']}")
    print()
    print(f"Validation: valid={validation['valid']} warnings={validation['warnings']}")
    print()

    summary_rows = []
    for agent_key in args.agents:
        agent_name, agent_cls = AGENTS[agent_key]
        rewards = []
        storm_brier = []
        equip_brier = []
        inference_gap = []
        planning_gap = []
        tool_gap = []
        print(f"[Agent] {agent_name}")
        for offset in range(args.episodes):
            episode_seed = args.seed + offset
            reward, _trace, eval_result = run_episode(agent_cls, seed=episode_seed, config=config)
            rewards.append(reward)
            storm_brier.append(eval_result["mean_brier_storm"])
            equip_brier.append(eval_result["mean_brier_equip"])
            inference_gap.append(eval_result["total_inference_gap"])
            planning_gap.append(eval_result["total_planning_gap"])
            tool_gap.append(eval_result["total_tool_use_gap"])
            print(
                f"  seed={episode_seed:<5} reward={reward:>8.1f} "
                f"brier_storm={eval_result['mean_brier_storm']:.4f} "
                f"brier_equip={eval_result['mean_brier_equip']:.4f} "
                f"tool_gap={eval_result['total_tool_use_gap']:.1f} "
                f"inference_gap={eval_result['total_inference_gap']:.1f} "
                f"planning_gap={eval_result['total_planning_gap']:.1f}"
            )
        summary_rows.append(
            (
                agent_name,
                statistics.mean(rewards),
                statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
                statistics.mean(storm_brier),
                statistics.mean(equip_brier),
                statistics.mean(tool_gap),
                statistics.mean(inference_gap),
                statistics.mean(planning_gap),
            )
        )
        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"{'Agent':<16} {'Reward':>10} {'Std':>8} {'Brier(S)':>10} {'Brier(E)':>10} "
        f"{'ToolGap':>10} {'InfGap':>10} {'PlanGap':>10}"
    )
    for row in summary_rows:
        print(
            f"{row[0]:<16} {row[1]:>10.1f} {row[2]:>8.1f} {row[3]:>10.4f} {row[4]:>10.4f} "
            f"{row[5]:>10.1f} {row[6]:>10.1f} {row[7]:>10.1f}"
        )


if __name__ == "__main__":
    main()
