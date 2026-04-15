# core

The real world generation logic.

## Modules

| Module | Purpose |
|---|---|
| `knobs.py` | `WorldKnobs` dataclass + `curriculum_knobs(level)` function. Defines 5 difficulty parameters and their interpolation across curriculum levels. |
| `generator.py` | `generate_config(knobs, seed)`. Converts knobs into a complete fishgame config dict using Dirichlet-based transition matrices, d-prime parameterized observation models, and causal confound generation. |
| `validator.py` | `validate_config(cfg, strict=False)`. Structural validation: checks transition stochasticity, observation parameter ranges, reward sanity, sensor budget. Returns errors/warnings dict. |

## Generator Internals

The generator builds each config component independently:
1. Transition matrices (storm, wind, equipment, tide) via Dirichlet concentration
2. Observation distributions (barometer, buoy, equipment inspection) via d-prime separation
3. Categorical sensors (sea_color, equip_indicator) via confusion matrix scaling
4. Causal confounds (zone ages, temp offsets, fish abundance bonus) via trap_strength
5. Tool budgets and sensor zone count from knobs directly
6. Valid allocations (1000 allocations for 4 zones, 1-10 boats)

All randomness is seeded through `random.Random(seed)` for deterministic generation.
