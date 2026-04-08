from fishing_game.simulator import FishingGameEnv
from world_gen import curriculum_knobs, generate_config


def test_generate_config_is_deterministic_for_seed():
    knobs = curriculum_knobs(0.5)
    first = generate_config(knobs, seed=42)
    second = generate_config(knobs, seed=42)
    assert first == second


def test_generate_config_changes_across_seeds():
    knobs = curriculum_knobs(0.5)
    first = generate_config(knobs, seed=42)
    second = generate_config(knobs, seed=43)
    assert first != second


def test_generated_config_has_required_environment_keys():
    cfg = generate_config(curriculum_knobs(0.5), seed=42)
    for key in [
        "states",
        "storm_transition",
        "wind_transition",
        "equip_transition",
        "tide_transition",
        "sea_color_probs",
        "equip_indicator_probs",
        "barometer_params",
        "buoy_params",
        "equipment_inspection_params",
        "valid_allocations",
    ]:
        assert key in cfg


def test_generated_config_can_boot_environment():
    cfg = generate_config(curriculum_knobs(0.5), seed=42)
    env = FishingGameEnv(config=cfg)
    obs = env.reset(seed=42)
    assert obs["day"] == 1


def test_generated_reward_schedule_is_fixed_across_levels():
    easy = generate_config(curriculum_knobs(0.0), seed=42)
    hard = generate_config(curriculum_knobs(1.0), seed=42)
    assert easy["safe_profit_per_boat"] == hard["safe_profit_per_boat"] == 7
    assert easy["danger_loss_per_boat"] == hard["danger_loss_per_boat"] == -18
    assert easy["danger_loss_equip_per_boat"] == hard["danger_loss_equip_per_boat"] == -10
    assert easy["danger_loss_both_per_boat"] == hard["danger_loss_both_per_boat"] == -25
