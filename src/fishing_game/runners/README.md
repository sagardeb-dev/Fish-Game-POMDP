# runners

Benchmark execution helpers. These wire agents to the environment and collect traces.

## Modules

| Module | Purpose |
|---|---|
| `runner.py` | Parallelized ablation suite: runs N agents x M configs x K seeds, collects evaluator metrics |
| `traced_runner.py` | LLM agent runner with full trace capture (tool calls, beliefs, rewards per step) |

## runner.py

Main entry points:
- `run_episode(agent_cls, seed, config)` -- single episode, returns (reward, trace, eval_result)
- `run_ablation_suite(seeds, max_workers)` -- full benchmark sweep with parallel execution

Falls back to serial execution if `ProcessPoolExecutor` is unavailable (e.g. Windows permission issues).

## traced_runner.py

- `run_traced_episode(agent_cls, seed, config, save_path)` -- runs an LLM agent with full tool-call tracing
- `run_llm_solver_episode(agent, seed, config, save_path)` -- runs LLMSolverAgent specifically
- Saves JSON traces to `traces/` for post-hoc analysis
