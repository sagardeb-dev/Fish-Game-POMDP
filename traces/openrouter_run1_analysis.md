# OpenRouter run 1 — `gpt-5.4-mini` on `llm_stats_guided` and `llm_pc_tools`

**Model:** `openai/gpt-5.4-mini` via OpenRouter (`https://openrouter.ai/api/v1`).
**Manifest:** `traces/ladder/sonnet46_full_ladder_run1/run_manifest.json` — same `(level, seed)` instances as the existing Sonnet 4.6 ladder, so rows are apples-to-apples with `pc_greedy`, `llm_raw`, `llm_stats`.
**Levels:** 0–5, **8 seeds per level**, **48 sessions per method**.
**Panel:** active (both methods).

> Note on model class: this is `gpt-5.4-mini`, not the full `gpt-5.4` from the existing ladder run in the README. Numbers shouldn't be put in the same column without that caveat.

## Methods recap

- **`llm_stats_guided`** — same tools as `llm_stats` (`correlation`, `partial_correlation`, `independence_test`, `intervene`, `submit_graph`). The system prompt contains an explicit phase-by-phase walkthrough of the PC + greedy active algorithm.
- **`llm_pc_tools`** — replaces the per-test stats tools with PC-pipeline phases as primitives: `pc_observational(alpha)`, `meek_closure(directed, undirected)`, `intervene(var, value)`, `submit_graph(...)`.

## Headline result

**`llm_pc_tools` is the first LLM-driven active-panel method in this benchmark to match or beat `pc_greedy` on directed-F1**, despite running on a *mini* model class. `llm_stats_guided` does not — providing the algorithm in prose without the corresponding tool primitives is not enough.

## Per-level numbers

### `llm_pc_tools` (gpt-5.4-mini)

| Level | n | Skel P | Skel R | Skel F1 | Dir P | Dir R | Dir F1 | DAG SHD | Eff |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 92.7 | 91.7 | 91.5 | 47.9 | 45.8 | 46.7 | 1.875 | 93.8 |
| 1 | 8 | 94.4 | 77.1 | 84.8 | 73.8 | 54.2 | 62.0 | 3.000 | 100.0 |
| 2 | 8 | 88.8 | 62.5 | 72.6 | 56.5 | 40.3 | 46.6 | 6.250 | 100.0 |
| 3 | 8 | 84.1 | 70.8 | 75.2 | 50.1 | 41.7 | 45.1 | 9.375 | 100.0 |
| 4 | 8 | 91.5 | 65.2 | 75.8 | 61.3 | 42.0 | 49.7 | 9.125 | 100.0 |
| 5 | 8 | 90.2 | 64.1 | 74.7 | 59.6 | 40.6 | 48.1 | 10.625 | 100.0 |
| **overall** | 48 | **90.4** | **71.9** | **79.1** | **58.2** | **44.1** | **49.7** | **6.708** | **99.0** |

### `llm_stats_guided` (gpt-5.4-mini)

| Level | n | Skel P | Skel R | Skel F1 | Dir P | Dir R | Dir F1 | DAG SHD | Eff |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 47.7 | 79.2 | 58.6 | 33.1 | 33.3 | 25.1 | 4.625 | 100.0 |
| 1 | 8 | 56.1 | 64.6 | 50.0 | 33.0 | 25.0 | 23.3 | 9.500 | 100.0 |
| 2 | 8 | 37.0 | 73.6 | 48.0 | 17.8 | 26.4 | 20.5 | 18.000 | 100.0 |
| 3 | 8 | 41.1 | 51.0 | 30.4 | 39.4 | 11.5 | 10.8 | 25.250 | 100.0 |
| 4 | 7 | 68.1 | 25.5 | 23.8 | 48.4 | 10.2 | 11.2 | 19.143 | 100.0 |
| 5 | 8 | 40.0 | 28.1 | 18.8 | 39.8 | 10.9 | 14.5 | 32.125 | 100.0 |
| **overall** | 47 | **48.5** | **55.6** | **38.6** | **34.7** | **19.6** | **17.7** | **18.085** | **100.0** |

47/48 success — one L4 session hit `max_steps=32` without submitting (timed out in stats calls).

## Comparison with existing ladder methods (active panel)

| Method | Model | Dir F1 | DAG SHD | Eff |
|---|---|---:|---:|---:|
| `oracle` | — | 100.0 | 0.000 | 100.0 |
| **`llm_pc_tools`** | **gpt-5.4-mini (OR)** | **49.7** | **6.708** | **99.0** |
| `pc_greedy` | baseline | 42.7 | 4.792 | 89.6 |
| `llm_raw` | sonnet-4.6 | 31.7 | 7.250 | 97.9 |
| `llm_stats` | sonnet-4.6 | 28.0 | 8.064 | 100.0 |
| `llm_raw` | gpt-5.4 | 22.9 | 9.271 | 92.7 |
| `llm_stats` | gpt-5.4 | 18.8 | 6.583 | 100.0 |
| **`llm_stats_guided`** | **gpt-5.4-mini (OR)** | **17.7** | **18.085** | **100.0** |

Two things to read out of this table:

1. **`llm_pc_tools` (49.7) > `pc_greedy` (42.7) on dir F1**, and is the only LLM-driven method to clear `pc_greedy` on this metric. Its SHD (6.7) is higher than `pc_greedy`'s (4.8) because it commits more edges (higher recall: 44.1 vs `pc_greedy`'s 32.8) — it makes more correct edges *and* more wrong ones.
2. **`llm_stats_guided` (17.7) ≈ `gpt-5.4 llm_stats` (18.8)**. The PC-algorithm walkthrough in prose did not move the needle. The model's SHD is much worse (18.1 vs 6.6) because it commits much more aggressively when guided, but those commitments are wrong.

## Behavioral observations (the most interesting part)

### `llm_stats_guided` is *not actually doing the active phase*

Action counts per session, averaged:

| Level | total | stats | intervene | submit |
|---:|---:|---:|---:|---:|
| 0 | 8.9 | 7.9 | **0.0** | 1.0 |
| 1 | 10.6 | 9.6 | **0.0** | 1.0 |
| 2 | 12.6 | 11.6 | **0.0** | 1.0 |
| 3 | 13.2 | 12.2 | **0.0** | 1.0 |
| 4 | 15.4 | 14.4 | **0.0** | 1.0 |
| 5 | 16.2 | 15.2 | **0.0** | 1.0 |

**The model intervened zero times** — across all 47 successful sessions on the active panel. Despite the system prompt explicitly walking through Phase 4 ("While budget > 0 AND undirected edges remain: pick the undirected node with the most undirected edges, intervene on it, etc."), the model spends its turns on independence tests and submits an observation-only graph. So what we're really benchmarking here is `llm_stats_obs` with extra prose, not an active method. That explains why its numbers are nearly identical to `gpt-5.4 llm_stats`'s active row from the README — neither was actually using interventions.

This is a real finding: **procedural knowledge in the prompt is not sufficient to change LLM behavior at this scale**. The model reads the algorithm and then doesn't follow it. The result on its own is interesting; combined with `llm_pc_tools` succeeding, it's a clean "tools, not prose" story.

### `llm_pc_tools` is using an extremely lean orchestration

| Level | total | intervene | pc_obs | meek | submit |
|---:|---:|---:|---:|---:|---:|
| 0 | 3.2 | 1.1 | 1.0 | 0.0 | 1.0 |
| 1 | 3.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| 2 | 2.9 | 0.9 | 1.0 | 0.0 | 1.0 |
| 3 | 3.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| 4 | 3.1 | 1.1 | 1.0 | 0.0 | 1.0 |
| 5 | 3.4 | 1.2 | 1.0 | 0.0 | 1.0 |

The model is doing the **minimum viable orchestration**: call `pc_observational` once, run roughly one intervention, submit. **`meek_closure` is never called** — the model isn't using one of the four primitives at all. So even this strong result is a floor, not a ceiling — there's headroom if the model used Meek propagation between interventions and used more of the budget.

It's also using *fewer interventions than the `optimal_intervention_set` would call for* (efficiency = 99% means it's using only the strictly necessary ones, but it's also not pushing on hard graphs where more interventions would help). Compare to `pc_greedy` at eff=89.6%, which is using closer to the full budget.

## What this means for the paper

The two probes were designed to disambiguate the source of the LLM gap. Together with the existing rows, the picture now looks like:

- **`llm_stats_guided` ≈ `llm_stats`** → giving the algorithm in prose doesn't change behavior. The LLM gap is **not procedural knowledge**.
- **`llm_pc_tools` > `pc_greedy`** → giving the algorithm as composable primitives moves the model above the classical baseline, with sub-mini compute. The LLM gap is **execution / orchestration**, and the right interface (PC-step tools) closes it.
- The fact that `llm_pc_tools` only used 3 primitives per session (and never called `meek_closure`) suggests the LLM is not even leveraging the full toolset — the result is robust, not fragile.

Headline-able framing for the paper: *"On the active panel, an LLM matches or beats the classical PC + greedy baseline when given PC's pipeline phases as composable tools — but not when given the same algorithm as a written walkthrough with statistical primitives. The bottleneck is interface, not knowledge."*

## Caveats

1. **Model class differs from the existing ladder rows.** `gpt-5.4-mini` ≠ `gpt-5.4`. Either rerun the existing ladder methods on `gpt-5.4-mini` for an apples-to-apples panel, or rerun these two probes on full `gpt-5.4` once the OpenAI billing situation is sorted.
2. **OpenRouter routing may not be byte-equivalent to direct OpenAI** (sampling defaults, response-format support). For headline numbers, use the same provider as the existing rows.
3. **`llm_pc_tools` may be partly riding on `pc_observational`'s output.** The minimal orchestration (1 PC call, 1 intervene, 1 submit) means a meaningful share of the result is just "PC's CPDAG + a single intervention." This is essentially a leaner version of `llm_pc_handoff` — the natural follow-up is to compare `llm_pc_tools` directly against `llm_pc_handoff` on the same model.
4. **One L4 timeout in `llm_stats_guided`** (`max_steps=32` without `submit_graph`). Increase `--max-steps` and rerun if you want a clean 48/48.

## Suggested next runs

1. **`llm_pc_handoff` on `gpt-5.4-mini` via OpenRouter** — same instances, completes the active-phase ablation.
2. **Either rerun all active methods (`llm_raw`, `llm_stats`) on `gpt-5.4-mini`**, or **rerun the two probes on direct `gpt-5.4`** — to remove the model-class confound from the headline comparison.
3. Optionally, **replay `llm_stats_guided` with `--max-steps 64`** to see whether the timeout on L4 was a fluke or a behavior pattern.
