# WorldGen

`world_gen` is the procedural world generator for fishgame. It produces a normal fishgame config dict, so the generated world can be used directly by:

- `FishingGameEnv`
- `FishingPOMDP`
- baseline agents
- the demo runners in this package

This is a stable PoC for generation, inspection, and baseline experiments. It is not yet fully research-faithful for paper claims. Known issues are listed at the end.

## What It Does

The goal is simple: generate many fishgame worlds from a small set of knobs instead of hand-writing every config.

Those knobs are meant to control difficulty along the main axes that matter in fishgame:

| Knob | Meaning | Easy end | Hard end |
|---|---|---|---|
| `d_prime` | Observation quality | sensors are clean and separable | sensors are noisy and overlap |
| `transition_alpha` | Hidden-state dynamics | state is easier to predict | state is harder to predict |
| `sensor_zones` | Same-day info budget | more zones visible each day | fewer zones visible each day |
| `trap_strength` | Causal confounds | observational patterns are straightforward | observational patterns are misleading |

The generator keeps the fishgame structure fixed:

- 80 hidden states
- same 4 zones
- same zone topology
- same overall simulator/POMDP contract

It changes the parameterized parts:

- transition matrices
- observation distributions
- confounds
- tool budgets
- action space metadata

## Why These Knobs

These knobs come from the research docs in `research/`, but the practical reading is:

| Knob | Why it exists in fishgame |
|---|---|
| `d_prime` | controls how much a single reading helps belief update |
| `transition_alpha` | controls how much the agent can rely on temporal persistence |
| `sensor_zones` | controls whether the task feels close to an MDP or like a real POMDP |
| `trap_strength` | controls whether naive pattern matching gets fooled by confounds |

The intended difficulty story is:

1. harder worlds give weaker direct evidence
2. harder worlds force more reliance on transitions and history
3. harder worlds make observational shortcuts less trustworthy

## Package Layout

| File | Role |
|---|---|
| `knobs.py` | Defines `WorldKnobs` and `curriculum_knobs()` |
| `generator.py` | Converts knobs into a full fishgame config |
| `validator.py` | Structural checks for generated configs |
| `demo_generate.py` | Generate one world and print its structure |
| `demo_pipeline.py` | Generate a world, validate it, and run baseline agents |

## Architecture

```text
CLI / Python API
       |
       v
curriculum_knobs(level, overrides)
       |
       v
WorldKnobs
       |
       v
generate_config(knobs, seed)
       |
       +--> transitions
       +--> observation params
       +--> confounds
       +--> tool budgets
       +--> valid allocations
       |
       v
validate_config(config)
       |
       v
FishingGameEnv / FishingPOMDP / baselines
```

The important point is that `world_gen` does not create a separate environment type. It creates a standard fishgame config.

## Standard Curriculum

`curriculum_knobs(level)` gives a default difficulty profile for `level` in `[0.0, 1.0]`.

Use the level as a preset, then override specific knobs only when you want to inspect a particular behavior.

| Level | Intended feel |
|---|---|
| `0.0` | easy, high visibility, weak confounds |
| `0.3` | medium-easy |
| `0.5` | standard middle setting |
| `0.7` | hard |
| `1.0` | hardest standard setting in v1 |

This is a generation preset, not a proven benchmark tier.

## CLI

### Generate and inspect one world

```powershell
python -m world_gen.demo_generate --level 0.5 --seed 42
```

Useful overrides:

```powershell
python -m world_gen.demo_generate --level 0.6 --seed 42 --d-prime 1.8 --sensor-zones 2 --trap-strength 0.7
```

### Run the full pipeline on baseline agents

```powershell
python -m world_gen.demo_pipeline --level 0.5 --seed 42 --episodes 2
```

With explicit overrides:

```powershell
python -m world_gen.demo_pipeline --level 0.6 --seed 42 --d-prime 1.8 --sensor-zones 2 --episodes 2 --agents random learner reasoner oracle
```

## Main CLI Arguments

| Argument | Purpose |
|---|---|
| `--level` | base curriculum preset |
| `--seed` | deterministic world generation seed |
| `--d-prime` | override observation quality |
| `--transition-alpha` | override transition behavior |
| `--sensor-zones` | override visible zones per step |
| `--trap-strength` | override confound strength |
| `--episodes` | number of episodes per selected agent in `demo_pipeline.py` |
| `--agents` | subset of baseline agents to run |

## Python API

```python
from world_gen import curriculum_knobs, generate_config, validate_config

knobs = curriculum_knobs(0.5, d_prime=1.8, sensor_zones=2)
cfg = generate_config(knobs, seed=42)
report = validate_config(cfg, strict=False)
```

Then use `cfg` directly with fishgame:

```python
from fishing_game.simulator import FishingGameEnv

env = FishingGameEnv(config=cfg)
obs = env.reset(seed=123)
```

## What Is Stable In V1

| Stable today | Notes |
|---|---|
| deterministic generation | same seed + same knobs gives same config |
| standard fishgame compatibility | generated configs work with existing env and POMDP |
| 80-state world structure | matches the current fishgame design |
| 1000 valid allocations | matches the 4-zone / 10-boat setup |
| dedicated pytest coverage | under `tests/world_gen` |

## What This README Assumes

This README is written for someone who already knows fishgame at a high level and wants to understand:

- what the generator is
- what the knobs mean
- how to run it
- what to trust and what not to trust yet

It is intentionally lighter than the theory docs under `research/`.

## Current Limitations

These do not stop the package from working as a PoC, but they matter for benchmark claims.

| Area | Current issue |
|---|---|
| rewards | some reward-related bonuses still vary across levels |
| information budget | current-day maintenance SQL can reveal more than intended |
| history | historical sensor coverage is richer than same-step sensor coverage |
| baselines | `CausalLearner` still keeps some true parameters |
| prompts | LLM tool descriptions can drift from generated budgets |
| knob semantics | `transition_alpha` description and implementation are not fully aligned |
| config ownership | some simulator defaults are still not fully owned by the generator |
| validation | some validator warnings are not yet numerically trustworthy |

## Current Position

Use `world_gen` today for:

- demos
- debugging
- baseline experimentation
- inspecting generated fishgame worlds

Do not treat it yet as the finalized benchmark generator for paper results without fixing the limitations above.
