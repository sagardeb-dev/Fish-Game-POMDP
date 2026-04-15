# world_gen

Procedural world generator for the fishing game benchmark. Generates complete fishgame config dicts from difficulty knobs and a random seed.

This is a stable PoC. Known limitations are listed at the end.

## Package Layout

```
world_gen/
  core/       knobs, generator, validator (the real logic)
  demos/      demo_generate, demo_pipeline (CLI entrypoints)
```

## How It Works

```
curriculum_knobs(level)  -->  WorldKnobs  -->  generate_config(knobs, seed)  -->  config dict
                                                                                      |
                                                                              validate_config(cfg)
                                                                                      |
                                                                              FishingGameEnv(config=cfg)
```

The generator takes 5 difficulty knobs and a seed, and outputs a standard fishgame config dict. The config is consumed by `FishingGameEnv` unchanged.

## Difficulty Knobs

| Knob | What it controls | Easy end | Hard end |
|---|---|---|---|
| `d_prime` | How informative sensor readings are | 3.0 (clean signals) | 1.5 (noisy) |
| `transition_alpha` | How much hidden state changes between days | stable | more volatile |
| `sensor_zones` | How many zones you can observe each day | 4 (all) | 2 |
| `trap_strength` | How misleading confound patterns are | 0 (none) | 1.0 (maximum) |
| `reward_asymmetry` | Kept fixed in v1 standard curriculum | -- | -- |

## CLI Usage

Inspect a generated world:
```bash
python -m world_gen.demo_generate --level 0.5 --seed 42
python -m world_gen.demo_generate --level 0.8 --seed 42 --d-prime 1.8 --sensor-zones 2
```

Generate a world and run baseline agents:
```bash
python -m world_gen.demo_pipeline --level 0.5 --seed 42 --episodes 2 --agents random learner reasoner oracle
python -m world_gen.demo_pipeline --level 0.6 --seed 42 --d-prime 1.8 --sensor-zones 2 --episodes 3
```

## Python API

```python
from world_gen import curriculum_knobs, generate_config, validate_config

knobs = curriculum_knobs(0.5)                          # preset from difficulty level
knobs = curriculum_knobs(0.5, d_prime=1.8)             # preset + manual override
cfg = generate_config(knobs, seed=42)                  # deterministic config dict
result = validate_config(cfg, strict=False)            # structural sanity check

from fishing_game.simulator import FishingGameEnv
env = FishingGameEnv(config=cfg)                       # plug and play
obs = env.reset(seed=42)
```

## WorldGen Baseline Scores (2 episodes per agent)

| Agent | L0.0 | L0.5 | L1.0 |
|---|---:|---:|---:|
| Random | 25 | 355 | 305 |
| NaivePattern | 508 | 440 | 572 |
| CausalLearner | 1475 | 1360 | 1170 |
| CausalReasoner | 1475 | 1320 | 1305 |
| Oracle | 1610 | 1660 | 1885 |

## Known Limitations

| Area | Issue |
|---|---|
| Reward comparability | `tide_bonus` and `fish_abundance_bonus` still vary by difficulty; cross-level scores are not directly comparable |
| Info leak (current-day) | `maintenance_log` exposes all 4 zones via SQL before acting, bypassing `sensor_zones` |
| Info leak (historical) | Historical `sensor_log` is full coverage for all zones regardless of `sensor_zones` setting |
| CausalLearner privilege | Keeps true `sea_color_probs`, `equip_indicator_probs`, `barometer_params` instead of learning them |
| LLM prompt drift | Tool descriptions hardcode 2/day budgets; generated worlds may set some to 1/day |
| Knob semantics | `transition_alpha` description and implementation are partially inverted |
| Dead knob | `reward_asymmetry` is exposed but ignored in v1 |
| Incomplete config | Generated configs omit `historical_days`; simulator falls back to 30 |
| Validator math | Entropy warnings mix bits and nats thresholds |

Do not treat generated-world results as paper-grade without fixing these.
