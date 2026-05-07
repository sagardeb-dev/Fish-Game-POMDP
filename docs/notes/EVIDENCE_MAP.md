# ACDB paper evidence map

## Ground rules

- Source of truth for paper-side results: `docs/notes/RESULTS.md`.
- Source of truth for the current ladder: `run_ladder.py`.
- Source of truth for calibration math and ladder rationale: `docs/notes/random-baseline-ladder-findings.md` and `traces/ladder_random_floor_sanity/summary.csv`.
- This file maps claims to evidence and safe interpretations. It is not a prose draft.

## Calibration evidence

### Current ladder

- Source:
  - `run_ladder.py:124-131`
- Current code truth:
  - `L0: d=4, k=3, n_obs=50, n_int=25, noise_var=0.5, budget_slack=2`
  - `L1: d=6, k=6, n_obs=50, n_int=25, noise_var=1.0, budget_slack=2`
  - `L2: d=8, k=9, n_obs=50, n_int=25, noise_var=1.0, budget_slack=2`
  - `L3: d=10, k=12, n_obs=50, n_int=25, noise_var=1.0, budget_slack=2`
  - `L4: d=12, k=14, n_obs=50, n_int=25, noise_var=1.0, budget_slack=2`
  - `L5: d=14, k=16, n_obs=50, n_int=25, noise_var=1.0, budget_slack=2`
- Note:
  - `README.md` still shows older slack values; do not treat that table as authoritative.

### Expected / random directed-F1 floor

- Sources:
  - `docs/notes/random-baseline-ladder-findings.md:146-201`
  - `traces/ladder_random_floor_sanity/summary.csv`
  - `reports/figure_prototypes/20260429T063348Z/05_random_floor_density.png`
  - `reports/figure_prototypes/20260429T063348Z/08_v0_v1_ladder_regions.png`
- Exact current-ladder floors from the prototype manifest:
  - `L0: 0.215`
  - `L1: 0.196`
  - `L2: 0.173`
  - `L3: 0.155`
  - `L4: 0.133`
  - `L5: 0.117`
- Allowed claim:
  - benchmark difficulty must be interpreted relative to a non-trivial blind-random floor that depends on graph density.
- Do not claim:
  - expected F1 is itself the benchmark score;
  - calibration alone explains all performance differences.
- Reviewer risk:
  - if paper tables use stale ladder parameters, calibration section becomes internally inconsistent.

## Claim 1 - Layered scoring makes the benchmark diagnostic, not just evaluative

- Metrics:
  - `skel_f1`
  - `comp_f1`
  - `dir_f1`
  - `SHD`
- Evidence:
  - `docs/notes/RESULTS.md`, `GPT-5.5`, `Avg L1-L5`
  - `llm_raw`: `dir_f1=0.697`, `SHD=6.3`, `skel_f1=0.753`, `comp_f1=0.732`
  - `pc_greedy`: `dir_f1=0.689`, `SHD=5.3`, `skel_f1=0.780`, `comp_f1=0.612`
- Allowed interpretation:
  - layered scores can reveal an observational-orientation advantage (`comp_f1`) that is not preserved in full-DAG quality (`dir_f1`, `SHD`);
  - a single graph metric would hide that distinction.
- Do not claim:
  - higher `comp_f1` means better end-to-end causal discovery overall.
- Reviewer risk:
  - example could look cherry-picked if presented as a model win rather than a metric-design demonstration.

## Claim 2 - Benchmark calibration materially affects what conclusions are valid

- Metrics / artifacts:
  - density `rho`
  - exact expected random directed-F1 floor
  - ladder region placement (earlier ladder vs current ladder)
  - budget semantics
- Evidence:
  - `run_ladder.py:124-131`
  - `docs/notes/random-baseline-ladder-findings.md:146-201`
  - `traces/ladder_random_floor_sanity/summary.csv`
  - `reports/figure_prototypes/20260429T063348Z/05_random_floor_density.png`
  - `reports/figure_prototypes/20260429T063348Z/08_v0_v1_ladder_regions.png`
- Allowed interpretation:
  - naive ladders can overstate capability by placing evaluation points in structurally guessable regions;
  - the current ladder explicitly moves away from that region and reports the floor beside model scores.
- Do not claim:
  - calibration is a generic afterthought applicable to every benchmark in the same way;
  - the expected floor replaces empirical baselines.
- Reviewer risk:
  - code/documentation drift on the ladder definition. Resolve by citing code as authority.

## Claim 3 - Across this benchmark family, model-family variation is larger than policy variation within a fixed model

- Metrics:
  - `dir_f1`
  - `SHD`
- Evidence:
  - `docs/notes/RESULTS.md`, `Avg L1-L5` across all five models
  - strong-model active range:
    - `GPT-5.5`: `llm_stats=0.640`, `llm_raw=0.697`, `pc_cpdag_llm=0.701`
  - cross-model end-to-end spread:
    - `llm_raw`: `GPT-5.5=0.697`, `Gemini-3-Flash=0.281`, `Sonnet-4.6=0.314`, `GPT-5.4-mini=0.145`, `Haiku-4.5=0.113`
- Allowed interpretation:
  - model family changes produce larger swings than prompt/policy changes inside a strong model class.
- Do not claim:
  - method does not matter;
  - this is a universal law about all LLM agents.
- Reviewer risk:
  - sensitive to which models are in the panel. Keep phrasing tied to the current benchmark family and current panel only.

## Claim 4 - LLM-only policies degrade faster with graph complexity than the classical active baseline

- Metrics:
  - `dir_f1` by level
  - `SHD` by level
- Evidence:
  - `docs/notes/RESULTS.md`
  - `pc_greedy`: `L1 0.802 -> L5 0.642` (`-0.160`, about `20%`)
  - `GPT-5.5 llm_raw`: `L1 0.824 -> L5 0.537` (`-0.287`, about `35%`)
  - `Gemini-3-Flash llm_raw`: `L1 0.414 -> L5 0.185` (`-0.229`, about `55%`)
  - `Sonnet-4.6 llm_raw`: `L1 0.426 -> L5 0.252` (`-0.174`, about `41%`)
  - `Haiku-4.5 llm_raw`: `L1 0.192 -> L5 0.047` (`-0.145`, about `76%`)
- Allowed interpretation:
  - LLM-only policies are less robust to graph scaling on this ladder than the classical active baseline.
- Do not claim:
  - a theorem about general LLM scaling behavior;
  - certainty about the internal mechanism.
- Reviewer risk:
  - relative-drop claims can depend on low starting points. Pair the percentages with absolute level-wise metrics.

## Claim 5 - Structural priors help substantially, but they do not remove the orientation subproblem

- Metrics:
  - `dir_f1`
  - `SHD`
- Evidence:
  - `docs/notes/RESULTS.md`, `Avg L1-L5`
  - `pc_cpdag_llm - llm_raw` directed-F1 lift:
    - `GPT-5.4-mini: +0.284`
    - `Sonnet-4.6: +0.312`
    - `Haiku-4.5: +0.366`
    - `Gemini-3-Flash: +0.304`
    - `GPT-5.5: +0.004`
- Allowed interpretation:
  - giving the model a strong observational prior removes a major bottleneck for weaker models;
  - observational structure recovery is a substantial part of the full task.
- Do not claim:
  - once the skeleton is supplied, the task is solved;
  - the same prior effect size will hold on any other benchmark family.
- Reviewer risk:
  - strongest-model lift is small; that is acceptable because the claim is about the panel, especially weaker models.

## Claim 6 - Interpreting interventional evidence is a distinct capability, separable from skeleton recovery

- Metrics:
  - `dir_f1`
  - `SHD`
- Evidence:
  - `docs/notes/RESULTS.md`, `Avg L1-L5`
  - `pc_cpdag_llm` vs `pc_greedy`:
    - `GPT-5.5: 0.701 / 5.2` vs `0.689 / 5.3`
    - `Sonnet-4.6: 0.626 / 6.1` vs `0.689 / 5.3`
    - `Gemini-3-Flash: 0.585 / 6.2` vs `0.689 / 5.3`
    - `Haiku-4.5: 0.479 / 7.6` vs `0.689 / 5.3`
    - `GPT-5.4-mini: 0.429 / 8.2` vs `0.689 / 5.3`
- Allowed interpretation:
  - once the observational graph is controlled, models still vary substantially in their ability to use interventions to orient edges.
- Do not claim:
  - only one model can ever do this;
  - general reasoning strength automatically transfers.
- Reviewer risk:
  - benchmark-specific active heuristic may influence the gap. Keep the claim comparative and local to ACDB.

## Claim 7 - Statistical tool access alone does not reliably improve active causal discovery

- Metrics:
  - `dir_f1`
  - `SHD`
- Evidence:
  - `docs/notes/RESULTS.md`, `Avg L1-L5`
  - `llm_stats` vs `llm_raw`:
    - `GPT-5.5: 0.640 < 0.697`
    - `GPT-5.4-mini: 0.103 < 0.145`
    - `Sonnet-4.6: 0.183 < 0.314`
    - `Haiku-4.5: 0.111 < 0.113`
    - `Gemini-3-Flash: 0.254 < 0.281`
- Allowed interpretation:
  - simply exposing conditional-independence-style tools does not yield consistent directed-graph gains on the current benchmark family.
- Do not claim:
  - tool use is bad in general;
  - the reason is known from output metrics alone.
- Reviewer risk:
  - prompt quality, step budget, or action interface could be alternative explanations. Keep this as a controlled benchmark finding, not a universal tool-use claim.

### Deferred trace-based mechanism

- Not for the current main paper:
  - repeated testing loops
  - delayed commitment
  - other behavioral narratives from tool traces
- Reason:
  - those require trace analysis, and the current paper foundation is restricted to metric-supported claims.

## Open evidence dependencies

- If a new model is added later, it does not change any claim unless its results are copied into `docs/notes/RESULTS.md`.
- If the final paper wants a behavioral claim about why stats tools fail, add a dedicated trace-analysis note first.
- If README is updated later to match code, this file should keep citing code as the authoritative ladder source anyway.
