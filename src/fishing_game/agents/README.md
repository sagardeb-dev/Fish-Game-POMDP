# agents

All agent implementations for the benchmark.

## Baselines (no API key)

| Agent | Strategy | Performance |
|---|---|---|
| RandomAgent | Random allocation, no tools, uniform beliefs | ~473 |
| NaivePatternMatcher | Simple heuristics, falls for all causal traps | ~435 |
| CausalLearner | Discovers params from historical DB via SQL, then Bayesian filtering | ~1324 |
| CausalReasoner | True params hardcoded, exact Bayesian filtering | ~1516 |
| OracleAgent | Reads hidden state directly (upper bound) | ~1716 |

## LLM Agents (require OpenAI API key)

| Agent | Strategy | Performance |
|---|---|---|
| LLMAgent | Free-form tool-calling, must discover + infer + plan in-context | ~663 |
| GPTAgent | OpenAI GPT integration layer for LLMAgent | -- |
| LLMSolverAgent | LLM estimates world model day 1, solver does exact Bayes days 1-20 | ~1124 |
| MockLLMSolverAgent | Deterministic mock for testing the solver path without API calls | -- |
| CodingAgent | Agno framework + PythonTools (currently bugged, GPT ignores REPL) | ~1069 |

## Adding a New Agent

An agent must implement:
1. Receive observation bundles from `FishingGameEnv`
2. Optionally call tools (SQL, reports, analysis)
3. Call `env.submit_decisions(allocation, beliefs, reasoning)` each day
4. Beliefs must include: `storm_active`, `storm_zone_probs`, `equip_failure_active`, `equip_zone_probs`, `tide_high`
