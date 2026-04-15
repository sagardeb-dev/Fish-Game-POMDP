# fishing_game

Core package for the Fishing Game POMDP benchmark. Contains the environment, agents, runners, and prompt assets.

## Package Layout

```
fishing_game/
  environment/    simulator, POMDP model, evaluator, config, contracts
  agents/         all agent implementations (baselines + LLM)
  runners/        benchmark execution helpers
  prompts/        extracted prompt text for LLM agents
```

## Top-Level Imports

The package preserves flat import paths for convenience:

```python
from fishing_game.config import CONFIG, BENCHMARK_CONFIG
from fishing_game.simulator import FishingGameEnv
from fishing_game.pomdp import FishingPOMDP
from fishing_game.evaluator import Evaluator
from fishing_game.baselines import RandomAgent, CausalReasoner, OracleAgent
from fishing_game.llm_solver_agent import LLMSolverAgent
from fishing_game.contracts import observation_bundle_to_pomdp_observations
```

These are thin re-export shims that delegate to the real modules under `environment/`, `agents/`, and `runners/`.

## Data Flow

```
config dict --> FishingGameEnv --> observation bundle --> Agent --> submit_decisions
                    |                                                   |
                    +-- SQLite DB (historical + live)                    |
                    +-- POMDP model (belief updates)                    |
                                                                        v
                                                              Evaluator (trace)
```

The config dict fully specifies the world: transitions, observations, rewards, tool budgets, sensor zones. Both hand-written configs (EASY, HARD, BENCHMARK) and generated configs from `world_gen` use the same schema.
