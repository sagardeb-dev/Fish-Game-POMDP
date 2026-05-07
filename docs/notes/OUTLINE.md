# ACDB paper outline

## Paper identity

- Working title: `Active Causal Discovery as a Diagnostic Benchmark for LLM Agents`
- One-line thesis: ACDB is a calibrated evaluation instrument for active causal discovery that decomposes agent performance into observational structure recovery, interventional orientation, and end-to-end graph recovery.
- What the paper is: a benchmark-and-evaluation paper with empirical demonstrations.
- What the paper is not:
  - not a pure model-ranking paper;
  - not a claim that LLMs can or cannot do causal discovery in general;
  - not a trace-analysis paper.

## Authoritative sources

- Benchmark code truth: `run_ladder.py`, especially the current ladder definition and the active-policy runners.
- Paper-side results snapshot: `docs/notes/RESULTS.md`.
- Claim source of truth: `docs/notes/CLAIMS.md`.
- Calibration source of truth: `docs/notes/random-baseline-ladder-findings.md` plus `traces/ladder_random_floor_sanity/summary.csv`.
- Benchmark pseudocode summary: `docs/specs/causal-discovery-v1-pseudocode.md`.

Note: `README.md` still shows older slack values in the ladder table. For paper notes, treat `run_ladder.py` as authoritative and mention README drift only as documentation cleanup, not as a scientific ambiguity.

## Locked empirical scope

- Main active ablations:
  - `pc_greedy`
  - `llm_raw`
  - `llm_stats`
  - `pc_cpdag_llm`
  - `llm_stats_cpdag_greedy`
- Excluded from the main ablation set:
  - `pc_obs`
  - `oracle`
- Exclusion rationale:
  - `pc_obs` is outside the main active-ablation panel, and its observational skeleton reference is already inherited by `pc_greedy`.
  - `oracle` is a ceiling artifact, not a meaningful agent comparison.
- Current model panel in scope:
  - `GPT-5.5`
  - `GPT-5.4-mini`
  - `Sonnet-4.6`
  - `Haiku-4.5`
  - `Gemini-3-Flash`
- Pending or future runs do not enter the paper notes until they are copied into `docs/notes/RESULTS.md`.
- The earlier high-density ladder is calibration and development context only. It can motivate the current ladder and random-floor work, but it is not a main empirical panel.

## Section skeleton

- Live main-paper order:
  1. Introduction
  2. Benchmark Setup
  3. Scoring and Calibration
  4. Experimental Setup
  5. Results
  6. Related Work
  7. Conclusion
- There is currently no standalone `Discussion` or `Limitations` section in `research/main.tex`.
- Scope boundaries are carried implicitly through Section 5 wording and Section 7 conclusion wording.

### 1. Introduction

- Purpose:
  - establish the benchmark problem, not a model race;
  - motivate why aggregate graph scores are diagnostically weak.
- Evidence inputs:
  - `CLAIMS.md`
  - current abstract
- Keep out:
  - long benchmark mechanics;
  - model-by-model score narration.

### 2. Benchmark Setup

- Purpose:
  - define the hidden SCM, observational ceiling, public interface, and accepted public instance;
  - explain the benchmark setup without spilling into scoring or calibration.
- Evidence inputs:
  - `docs/notes/BENCHMARK_SETUP_PRIORS.md`
  - `src/causal_discovery/benchmark/instance.py`
  - `src/causal_discovery/runtime/session.py`
  - `run_ladder.py`
- Keep out:
  - full score formulas;
  - random-floor math;
  - prompt wording.

### 3. Scoring and Calibration

- Purpose:
  - justify `skeleton F1`, `compelled-edge F1`, `directed F1`, and `SHD`;
  - explain why layered scoring is the core diagnostic contribution;
  - justify the ladder and random-floor calibration.
- Evidence inputs:
  - `docs/notes/SCORING_CALIBRATION_PRIORS.md`
  - `src/causal_discovery/scoring/scores.py`
  - `run_ladder.py`
- Keep out:
  - full results discussion;
  - model-by-model interpretation.

### 4. Experimental Setup

- Purpose:
  - define the active ablations and model roster;
  - explain paired seeds and the benchmark-side data snapshot.
- Evidence inputs:
  - `run_ladder.py`
  - `docs/notes/RESULTS.md`
- Keep out:
  - prompt deep-dives;
  - token-cost discussion unless the final paper needs it.

### 5. Results

- Purpose:
  - support Claims 3-7 with direct metric comparisons;
  - emphasize capability decomposition, not leaderboard framing.
- Evidence inputs:
  - `docs/notes/RESULTS.md`
  - `docs/notes/CLAIMS.md`
- Keep out:
  - unsupported behavioral explanations from traces.

### 6. Related work

- Purpose:
  - position ACDB against direct LLM-facing method and benchmark neighbors.
- Evidence inputs:
  - `docs/notes/RELATED_WORK.md`
  - `docs/notes/RELATED_WORK_PRIORS.md`
- Keep out:
  - theory-heavy causal-discovery history already covered by Sections 1--3;
  - broad agent-benchmark catalogues;
  - novel empirical claims.

## Scope handling

- Purpose:
  - not a live standalone section in the current draft.
  - limitations are instead folded into the bounded claims of Section 5 and the scope paragraph of Section 7.

### 7. Conclusion

- Purpose:
  - restate the benchmark artifact first, scoring/calibration second, and empirical demonstration third;
  - absorb scope boundaries without reopening a separate limitations section.
- Evidence inputs:
  - claims and results summary

## Main-paper vs appendix split

- Main paper:
  - benchmark setup
  - scoring and calibration
  - main active ablations and main results
- Appendix:
  - exact graph and SCM rejection rules
  - exact minimum-intervention-set procedure
  - exact query/session prompt construction
  - tool schema details
  - additional per-level tables
  - extra figure variants and representative trace artifacts

## Writing order

1. Keep `ABSTRACT.md`, `CLAIMS.md`, `OUTLINE.md`, and `EVIDENCE_MAP.md` aligned.
2. Draft introduction only after the claim/evidence mapping is stable.
3. Draft benchmark and scoring sections before results prose.
4. Draft calibration before final result interpretation.
5. Draft results only from `docs/notes/RESULTS.md`, not from ad hoc terminal summaries.
