"""Difficulty knobs for procedural fishgame world generation."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_REWARD_ASYMMETRY = 18.0 / 7.0


@dataclass
class WorldKnobs:
    """Difficulty knobs for the fishing game POMDP."""

    difficulty: float = 0.5
    d_prime: float | None = None
    transition_alpha: float | None = None
    sensor_zones: int | None = None
    reward_asymmetry: float | None = None
    trap_strength: float | None = None
    episode_length: int = 20
    max_boats: int = 10
    seed: int = 0
    name: str = ""

    def __post_init__(self):
        d = max(0.0, min(1.0, self.difficulty))
        defaults = {
            "d_prime": _lerp(3.0, 1.5, d),
            "transition_alpha": _lerp(2.0, 0.5, d),
            "sensor_zones": int(round(_lerp(4, 2, d))),
            "reward_asymmetry": DEFAULT_REWARD_ASYMMETRY,
            "trap_strength": _lerp(0.0, 1.0, d),
        }
        for key, value in defaults.items():
            if getattr(self, key) is None:
                setattr(self, key, value)

        assert 0.5 <= self.d_prime <= 3.0, f"d_prime out of range: {self.d_prime}"
        assert 0.1 <= self.transition_alpha <= 2.0, f"transition_alpha out of range: {self.transition_alpha}"
        assert 1 <= self.sensor_zones <= 4, f"sensor_zones out of range: {self.sensor_zones}"
        assert 0.5 <= self.reward_asymmetry <= 10.0, f"reward_asymmetry out of range: {self.reward_asymmetry}"
        assert 0.0 <= self.trap_strength <= 1.0, f"trap_strength out of range: {self.trap_strength}"


def curriculum_knobs(level: float, **overrides) -> WorldKnobs:
    """Generate a stable standard curriculum level in [0.0, 1.0]."""
    level = max(0.0, min(1.0, level))
    knobs = WorldKnobs(
        difficulty=level,
        episode_length=20,
        max_boats=10,
        seed=0,
        **overrides,
    )

    if "sensor_zones" not in overrides:
        knobs.sensor_zones = max(2, knobs.sensor_zones)
    if knobs.d_prime < 1.5 and "transition_alpha" not in overrides:
        knobs.transition_alpha = max(0.5, knobs.transition_alpha)
    return knobs


def describe_difficulty(knobs: WorldKnobs) -> str:
    """Human-readable description of difficulty level."""
    lines = [
        "Difficulty Profile:",
        f"  d_prime (obs. informativeness): {knobs.d_prime:.2f}",
        f"    - {_describe_d_prime(knobs.d_prime)}",
        f"  transition_alpha (stochasticity): {knobs.transition_alpha:.2f}",
        f"    - {_describe_alpha(knobs.transition_alpha)}",
        f"  sensor_zones (info. budget): {knobs.sensor_zones}/4",
        f"    - {_describe_sensor_zones(knobs.sensor_zones)}",
        f"  reward_asymmetry: {knobs.reward_asymmetry:.2f}x",
        f"    - Fixed for v1 standard curriculum",
        f"  trap_strength (confounds): {knobs.trap_strength:.2f}",
        f"    - {_describe_traps(knobs.trap_strength)}",
    ]
    return "\n".join(lines)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def _describe_d_prime(d_prime: float) -> str:
    if d_prime >= 2.8:
        return "Near-deterministic (trivial)"
    if d_prime >= 2.0:
        return "Easy (clean sensors)"
    if d_prime >= 1.5:
        return "Moderate-hard (usable noisy sensors)"
    if d_prime >= 1.0:
        return "Hard (weak sensors)"
    return "Very hard (near-uninformative sensors)"


def _describe_alpha(alpha: float) -> str:
    if alpha >= 1.5:
        return "Highly random (belief wanders)"
    if alpha >= 1.0:
        return "Neutral (balanced stochasticity)"
    if alpha >= 0.5:
        return "Meaningful persistence"
    return "Sparse transitions (near-deterministic)"


def _describe_sensor_zones(sensor_zones: int) -> str:
    if sensor_zones == 4:
        return "Full observability"
    if sensor_zones == 3:
        return "High observability"
    if sensor_zones == 2:
        return "Moderate observability"
    return "Minimal observability"


def _describe_traps(strength: float) -> str:
    if strength < 0.1:
        return "No confounds"
    if strength < 0.3:
        return "Weak confounds"
    if strength < 0.6:
        return "Moderate confounds"
    if strength < 0.85:
        return "Strong confounds"
    return "Maximum confounds"
