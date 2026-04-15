# environment

Core environment modules. These define the world, not the agents.

## Modules

| Module | Purpose |
|---|---|
| `config.py` | EASY_CONFIG, HARD_CONFIG, BENCHMARK_CONFIG (80-state POMDP parameter dicts) |
| `simulator.py` | FishingGameEnv: SQLite DB, tool execution, sensor zone subsampling, reward computation |
| `pomdp.py` | FishingPOMDP: exact Bayesian belief updates, optimal action computation, likelihood functions |
| `evaluator.py` | 3-way cost decomposition (tool_use_gap + inference_gap + planning_gap), Brier scores |
| `contracts.py` | Shared observation and belief payload converters used by agents, evaluator, and simulator |

## Config Schema

A config dict must specify:
- `states`, `zones`, `wind_to_zone`, `equip_to_zone` (topology)
- `storm_transition`, `wind_transition`, `equip_transition`, `tide_transition` (dynamics)
- `barometer_params`, `buoy_params`, `equipment_inspection_params` (observation distributions)
- `sea_color_probs`, `equip_indicator_probs` (categorical observations)
- `safe_profit_per_boat`, `danger_loss_per_boat`, etc. (rewards)
- `sensor_zones_per_step`, `tool_budgets` (information budget)
- `episode_length`, `valid_allocations` (episode structure)

Both hand-written configs and `world_gen.generate_config()` output produce this same schema.

## Contracts

`contracts.py` centralizes the observation and belief conversion logic so agents, the evaluator, and the simulator all use the same code path:

- `observation_bundle_to_pomdp_observations(obs)` -- env observation dict to POMDP observation list
- `belief_vector_to_decision_beliefs(pomdp, belief)` -- 80-state vector to the evaluator's belief payload
- `belief_dict_to_vector(cfg, pomdp, beliefs)` -- reverse direction
