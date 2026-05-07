from world_gen import WorldKnobs, curriculum_knobs


def test_curriculum_knobs_is_deterministic():
    first = curriculum_knobs(0.5)
    second = curriculum_knobs(0.5)
    assert first == second


def test_curriculum_levels_are_valid():
    for level in [0.0, 0.5, 1.0]:
        knobs = curriculum_knobs(level)
        assert isinstance(knobs, WorldKnobs)
        assert 0.5 <= knobs.d_prime <= 3.0
        assert 0.1 <= knobs.transition_alpha <= 2.0
        assert 2 <= knobs.sensor_zones <= 4


def test_standard_curriculum_keeps_rewards_fixed():
    easy = curriculum_knobs(0.0)
    hard = curriculum_knobs(1.0)
    assert easy.reward_asymmetry == hard.reward_asymmetry


def test_standard_curriculum_respects_known_constraints():
    hard = curriculum_knobs(1.0)
    assert hard.sensor_zones >= 2
    if hard.d_prime < 1.5:
        assert hard.transition_alpha >= 0.5


def test_overrides_take_precedence():
    knobs = curriculum_knobs(1.0, sensor_zones=1, transition_alpha=0.2, d_prime=1.0)
    assert knobs.sensor_zones == 1
    assert knobs.transition_alpha == 0.2
    assert knobs.d_prime == 1.0
