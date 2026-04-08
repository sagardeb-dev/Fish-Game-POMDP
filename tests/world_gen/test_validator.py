import pytest

from world_gen import ValidationError, curriculum_knobs, generate_config, validate_config


def test_validator_accepts_generated_configs():
    cfg = generate_config(curriculum_knobs(0.5), seed=42)
    result = validate_config(cfg, strict=False)
    assert result["valid"] is True


def test_validator_rejects_bad_transition_rows():
    cfg = generate_config(curriculum_knobs(0.5), seed=42)
    cfg["storm_transition"] = [[0.5, 0.6], [0.2, 0.8]]
    with pytest.raises(ValidationError):
        validate_config(cfg, strict=False)


def test_validator_rejects_invalid_sensor_budget():
    cfg = generate_config(curriculum_knobs(0.5), seed=42)
    cfg["sensor_zones_per_step"] = 0
    with pytest.raises(ValidationError):
        validate_config(cfg, strict=False)


def test_validator_rejects_broken_observation_params():
    cfg = generate_config(curriculum_knobs(0.5), seed=42)
    cfg["barometer_params"][1]["mean"] = cfg["barometer_params"][0]["mean"]
    with pytest.raises(ValidationError):
        validate_config(cfg, strict=False)
