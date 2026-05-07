# V1 Change Log

Minimal notes for implementation changes that affect paper claims, results, or methodology.

## PC+greedy threshold calibration

- Replaced fixed `tau = 0.5` with a z-calibrated mean-shift threshold:
  `tau_j = 1.95996 * sqrt(s_obs,j^2 / n_obs + s_int,j^2 / n_int)`.
- Preserved v0 orientation behavior: significant shift means `target -> neighbor`; otherwise
  default to `neighbor -> target`.
- Audit on v0 seeds: active PC+greedy directed F1 `42.7 -> 48.8`, SHD `4.792 -> 4.479`.
- Paper note: non-significant shift is a heuristic reverse decision, not proof of reverse
  causality.

## Random directed-F1 floor

- Random policy: sample a random topological order, then sample `m` uniformly from
  `{0, ..., M}`, where `M = d(d-1)/2`.
- Conditional expected directed F1:
  `E[F1 | m] = k*m / [M*(m+k)] = rho*s / (rho+s)`, where `rho = k/M` and `s = m/M`.
- Exact discrete floor:
  `E[F1] = (1/(M+1)) * sum_{m=0}^{M} k*m / [M*(m+k)]`.
- `rho/(1+2*rho)` is only the midpoint/Jensen envelope at `s=1/2`, not the exact floor.
- Prior art: Petersen 2025 for random-graph F1 expectation; Reisach et al. 2021 for
  structure-blind diagnostic baselines.

## V1 scale-calibrated ladder

- `ladder_levels()` now returns v1. The original v0 ladder is preserved as
  `ladder_levels_v0()`; the active ladder is also available as `ladder_levels_v1()`.
- V1 is a graph-scale/generalization ladder with calibrated random floor, not a full
  factorial identifiability x statistics design.
- Sample sizes and noise are fixed after L0 so graph scale is not confounded with decreasing
  `n_obs`/`n_int`. Statistical scarcity should be a separate ablation.

| L | d | k | rho | n_obs | n_int | noise | slack | E_random Dir-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4 | 3 | 0.500 | 50 | 25 | 0.5 | 2 | 0.215 |
| 1 | 6 | 6 | 0.400 | 50 | 25 | 1.0 | 2 | 0.196 |
| 2 | 8 | 9 | 0.321 | 50 | 25 | 1.0 | 2 | 0.173 |
| 3 | 10 | 12 | 0.267 | 50 | 25 | 1.0 | 2 | 0.155 |
| 4 | 12 | 14 | 0.212 | 50 | 25 | 1.0 | 2 | 0.133 |
| 5 | 14 | 16 | 0.176 | 50 | 25 | 1.0 | 2 | 0.117 |

## Uniform intervention headroom

- Set `budget_slack = 2` for every v1 level. Budget remains
  `len(optimal_intervention_set) + budget_slack`.
- Reason: slack `0` at high levels made one bad intervention target unrecoverable, so the
  metric partly measured perfect intervention-set selection rather than active reasoning.
- Random floor is unchanged because it depends on graph density, not intervention budget.

## LLM action schema and prompt alignment

- Replaced the broad one-size action schema with method-specific allowed actions:
  raw-observational gets only `submit_graph`; raw-active gets `intervene` and `submit_graph`;
  stats policies get statistical tools only when intended.
- Session `output_schema` now matches the provider tool schema for each method, so models are
  not shown unavailable actions.
- Prompts now explicitly warn that dependence/correlation is not direct causation and that
  chains/shared causes can induce association.
- Active prompts include scale-aware intervention guidance: choose values far enough from the
  observational mean to make downstream shifts detectable while staying on the observed scale.

## Step-aware LLM loop

- Added step metadata to each LLM call: current step, max steps, and remaining steps.
- On the final allowed step, the tool schema is forced to `submit_graph` only. This is
  schema-enforced, not just prompt text.
- Default LLM caps are now `max_steps_raw = 20` and `max_steps_stats = 40`.
- Purpose: prevent stats/tool policies from looping until failure without submitting a graph.

## LLM working memory

- Every LLM action now carries a required `reasoning_summary`, including tool and intervention
  actions.
- The loop feeds back bounded public working memory on every call:
  last `12` non-empty summaries, truncated to `300` chars each.
- Tool outputs remain separate as `Tool history JSON`; summaries are an audit/state channel, not
  hidden chain-of-thought.
- Full provider requests/responses are still saved in `events.jsonl`.

## Hybrid LLM ablations

- Added active method `pc_cpdag_llm`:
  PC builds the observational partial graph; the LLM receives PC directed/undirected edges plus
  observational means/stds, then uses interventions and submits the final graph. Raw rows are not
  included in this policy.
- Added active method `llm_stats_cpdag_greedy`:
  LLM-stats builds an observational partial graph; calibrated greedy orients only the submitted
  undirected edges using interventions. Missing skeleton edges are not repaired.
- These isolate two capabilities:
  intervention planning/integration given a PC graph, and observational graph quality when
  intervention planning is removed.

## Smoke runner

- Added `scripts/run_l3_v1_all_policy_smoke.py`.
- It shells into the original `run_ladder.py` with `--levels 3 --seeds-per-level 1`; it does not
  duplicate ladder logic.
- Verified current L3 config through `ladder_levels() -> ladder_levels_v1() -> config_from_level()`:
  `d=10, k=12, n_obs=50, n_int=25, noise_var=1.0, budget_slack=2`.

## Validation probes

- Acceptance probe: `50/50` accepted instances at every v1 level with `max_attempts=1000`.
- PC probe on 8 seeds: PC+greedy SHD rises `0.625 -> 9.375`; observational PC SHD rises
  `3.000 -> 13.000`.
- Interpretation: random floor is calibrated and PC error burden increases. Directed F1 need
  not be perfectly monotone because it is ratio-sensitive and seed-limited.
- Unit verification after LLM-loop and hybrid-policy changes: `135 passed` for
  `tests/unit/causal_discovery`.

## Paper implications

- V1 result tables are stale until random, PC, PC+greedy, and all LLM/hybrid panels are rerun on
  the current v1 ladder and policy set.
- Report the exact random floor alongside directed F1 to separate skill from metric leakage.
- Preserve v0 numbers as an audit/calibration story, not as final v1 benchmark evidence.
- Treat `pc_cpdag_llm` and `llm_stats_cpdag_greedy` as diagnostic ablations, not headline
  replacements for end-to-end LLM policies.

## Deprecate observational-only LLM policies

Changed:
- Removed `llm_raw_obs` and `llm_stats_obs` from default work items in `make_work_items()`.
  Code paths for dispatching/running these methods are preserved but will not fire.
- `allowed_actions_for_method()` still handles both methods (used by tests and by
  `llm_stats_cpdag_greedy` which shares the `llm_stats_obs` action set).
- Removed from README, LLM_POLICY_PROMPTS.md, LLM_TRACE_DEEPDIVE.md,
  LLM_ACTION_POLICY_TEST_VIEW.md.
- Deleted root-level trace dumps: llm_call_llm_raw_obs.md, llm_call_llm_stats_obs.md.

Why:
- Obs-only policies are diagnostically redundant. `llm_stats_cpdag_greedy` already
  isolates LLM skeleton quality (LLM builds observational graph, greedy orients).
- `llm_raw_obs` submitted empty graphs — model has no tools and can't extract structure
  from raw numeric matrices. Not a meaningful evaluation.
- `llm_stats_obs` used all steps on correlations but found very few edges — the same
  skeleton quality is captured by `llm_stats_cpdag_greedy`'s skel_f1.
- Removing from docs prevents AI agents from getting stale context about deprecated policies.

Remaining policy set:
- Baselines: pc, pc_greedy, oracle
- End-to-end LLM: llm_raw, llm_stats
- Hybrid ablations: pc_cpdag_llm, llm_stats_cpdag_greedy
