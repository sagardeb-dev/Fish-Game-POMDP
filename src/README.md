# src

All source code lives here. Two packages: `fishing_game` (the benchmark) and `world_gen` (procedural world generator).

## Quick Start: Writing a Script

```python
# Environment
from fishing_game.config import CONFIG, BENCHMARK_CONFIG, EASY_CONFIG, HARD_CONFIG
from fishing_game.simulator import FishingGameEnv
from fishing_game.pomdp import FishingPOMDP
from fishing_game.evaluator import Evaluator
from fishing_game.contracts import (
    observation_bundle_to_pomdp_observations,
    belief_vector_to_decision_beliefs,
    belief_dict_to_vector,
)

# Agents
from fishing_game.baselines import (
    RandomAgent,
    NaivePatternMatcher,
    CausalLearner,
    CausalReasoner,
    OracleAgent,
)
from fishing_game.llm_agent import LLMAgent
from fishing_game.llm_solver_agent import LLMSolverAgent, MockLLMSolverAgent
from fishing_game.gpt_agent import GPTAgent
from fishing_game.coding_agent import CodingAgent

# Runners
from fishing_game.runner import run_episode, run_ablation_suite
from fishing_game.traced_runner import run_traced_episode, run_llm_solver_episode

# World Generation
from world_gen import curriculum_knobs, generate_config, validate_config, WorldKnobs
```

## Minimal Example: Run a Baseline on a Hand-Written Config

```python
from fishing_game.config import CONFIG
from fishing_game.simulator import FishingGameEnv
from fishing_game.baselines import CausalReasoner
from fishing_game.evaluator import Evaluator

env = FishingGameEnv(config=CONFIG)
obs = env.reset(seed=42)
agent = CausalReasoner(config=CONFIG)

for day in range(CONFIG["episode_length"]):
    result = agent.act(env, obs)
    if result["done"]:
        break
    obs = result["observation"]

trace = env.get_trace()
ev = Evaluator(config=CONFIG).evaluate_episode(trace)
print(f"Reward: {ev['total_reward']}, Brier(storm): {ev['mean_brier_storm']:.4f}")
```

## Minimal Example: Run a Baseline on a Generated World

```python
from world_gen import curriculum_knobs, generate_config, validate_config
from fishing_game.simulator import FishingGameEnv
from fishing_game.baselines import CausalReasoner
from fishing_game.evaluator import Evaluator

knobs = curriculum_knobs(0.5, d_prime=1.8, sensor_zones=2)
cfg = generate_config(knobs, seed=42)
validate_config(cfg, strict=False)

env = FishingGameEnv(config=cfg)
obs = env.reset(seed=42)
agent = CausalReasoner(config=cfg)

for day in range(cfg["episode_length"]):
    result = agent.act(env, obs)
    if result["done"]:
        break
    obs = result["observation"]

trace = env.get_trace()
ev = Evaluator(config=cfg).evaluate_episode(trace)
print(f"Reward: {ev['total_reward']}")
```

## Packages

```
src/
  fishing_game/
    environment/    config, simulator, pomdp, evaluator, contracts
    agents/         all agent implementations
    runners/        ablation suite, traced LLM runner
    prompts/        extracted prompt text for LLM agents
  world_gen/
    core/           knobs, generator, validator
    demos/          CLI demo entrypoints
```

Each subdirectory has its own README.
