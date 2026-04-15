# prompts

Extracted prompt text for LLM-backed agents. Kept separate from agent logic so prompts can be reviewed, versioned, and modified independently.

## Files

| File | Used by | Content |
|---|---|---|
| `llm_agent.py` | LLMAgent / GPTAgent | `SYSTEM_PROMPT`: full game description, sensor list, tool budgets, belief schema |
| `llm_solver.py` | LLMSolverAgent | `ESTIMATION_PROMPT`: parameter estimation guide, analysis workflow, JSON output schema |

## Design Notes

- LLMAgent gets the causal structure partially disclosed (hidden variables named, but observation model not explained)
- LLM+Solver gets a blank parameter schema (field names only, no causal explanations) -- must discover what parameters mean from data patterns
- Neither agent is told about the causal traps (wave propagation, age confounding, etc.)
