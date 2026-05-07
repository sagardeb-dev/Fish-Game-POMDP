# Results priors

## Section objective

Section 5 should demonstrate what ACDB reveals once Sections 2--4 have fixed the benchmark world, the scoring contract, and the compared ablations. It supports the empirical claims without turning the paper into a leaderboard narrative.

This section should make three things legible:

1. the main end-to-end task separates models and ablations clearly;
2. one selected layered comparison shows why the score layers should be read together;
3. performance degrades across the ladder in different ways for classical and LLM-driven policies.

It should not restate the benchmark contribution in meta-language. The structure of the section should make that contribution obvious.

## Source-of-truth hierarchy

Use sources in this order:

1. `docs/notes/CLAIMS.md`
2. `docs/notes/EVIDENCE_MAP.md`
3. `docs/notes/RESULTS.md`
4. `docs/notes/OUTLINE.md`
5. `docs/notes/FIGURES.md`

Use `research/sections/05_results.tex` only as a stale audit. Do not draft from it.

## Hard drafting constraints

- keep the section short;
- keep the tone benchmark-paper sober, not blog-post emphatic;
- use paired `L1-L5` comparisons only;
- no `L0`;
- no `oracle`, `pc_obs`, random baseline, or correlation-only probe in the main results narrative;
- no trace-mechanism language;
- no per-model travelogue;
- no multi-finding laundry list;
- no significance-test clutter unless one specific test becomes load-bearing later.

## Section-level priming prompt

Write Section 5 like the results section of a serious benchmark paper. Each paragraph should be organized around one benchmark-controlled comparison, not one model. Let the benchmark contribution show through the structure of the comparisons rather than through explicit self-praise. Use short paragraphs, concrete metric references, and open-ended interpretation. State only what the metrics and ablations isolate. Keep the strongest-model competitive point secondary. Stop before the section turns into a catalogue of cells.

## Section shape

### Opening paragraph

Required content:

- the paper reports paired `L1-L5` results for the five in-scope active ablations;
- the results are organized by the distinctions ACDB is designed to expose;
- the section will move from score layers, to ablations, to scaling/model spread.

Keep out:

- any model ranking in the opener;
- any random-floor recap;
- any long reminder of the benchmark setup.

### 5.1 Cross-model comparison on the average ladder

Purpose:

- support Claims 3, 5, 6, and 7 through the main cross-model comparison;
- make the end-to-end task and the structural-prior ablation visible before drilling into layered scores.

Claim support:

- `docs/notes/CLAIMS.md` Claim 1
- `docs/notes/EVIDENCE_MAP.md` Claim 1

Required comparisons:

- `llm_raw` across all five models, `Avg L1-L5`;
- `pc_cpdag_llm` vs `llm_raw` across the same panel;
- `llm_stats` vs `llm_raw`;
- `pc_greedy` as the shared reference.

Load-bearing numbers:

- `GPT-5.5 llm_raw`: `dir_f1=0.697`, `SHD=6.3`, `skel_f1=0.753`, `comp_f1=0.732`
- `pc_greedy`: `dir_f1=0.689`, `SHD=5.3`, `skel_f1=0.780`, `comp_f1=0.612`

Allowed interpretation:

- `comp_f1` can reveal an observational-orientation advantage that does not survive intact in full-DAG quality;
- `skel_f1`, `comp_f1`, `dir_f1`, and `SHD` are measuring distinct error surfaces;
- a single headline graph score would hide this distinction.

Do not claim:

- higher `comp_f1` means stronger end-to-end causal discovery overall;
- the example proves one model is "better" in a broad sense.

Preferred artifact:

- one full-width cross-model figure with `dir_f1` and `SHD`;
- `pc_greedy` should appear as a shared reference, not duplicated as five separate bars.

Drafting guidance:

- keep this subsection short;
- one concrete example is enough;
- the point is the metric contract, not the model.

### 5.2 Selected layered-score comparison

Purpose:

- support Claim 1 with one compact selected slice;
- show why `skel_f1`, `comp_f1`, `dir_f1`, and `SHD` should be read together rather than collapsed.

Claim support:

- `docs/notes/CLAIMS.md` Claim 1
- `docs/notes/EVIDENCE_MAP.md` Claim 1

Selected slice:

- `GPT-5.5`, `Avg L1-L5`
- rows: `pc_greedy`, `pc_cpdag_llm`, `llm_raw`, `llm_stats`

Load-bearing numbers:

- `pc_greedy`: `skel_f1=0.780`, `comp_f1=0.612`, `dir_f1=0.689`, `SHD=5.3`
- `pc_cpdag_llm`: `0.789`, `0.615`, `0.701`, `5.2`
- `llm_raw`: `0.753`, `0.732`, `0.697`, `6.3`
- `llm_stats`: `0.704`, `0.673`, `0.640`, `7.3`

Allowed interpretation:

- methods close on directed F1 can still differ on CPDAG-facing scores and on SHD;
- the selected slice should read as one empirical example, not as the main model story.

Allowed interpretation:

- once the observational graph is controlled, the remaining gap is about active orientation quality rather than skeleton recovery;
- models still vary materially on this narrower subtask;
- the strongest-model near-parity point is a secondary empirical observation, not the headline of the section.

Do not claim:

- only one model can ever do this;
- general reasoning strength transfers automatically to interventional reasoning.

#### 5.2c Statistical tools

Load-bearing numbers:

- `llm_stats` vs `llm_raw`, `Avg L1-L5`:
  - `GPT-5.5: 0.640 < 0.697`
  - `GPT-5.4-mini: 0.103 < 0.145`
  - `Sonnet-4.6: 0.183 < 0.314`
  - `Haiku-4.5: 0.111 < 0.113`
  - `Gemini-3-Flash: 0.254 < 0.281`

Allowed interpretation:

- simply adding CI-style tools does not produce consistent directed-graph gains on the current benchmark family;
- the raw/stats comparison is informative because the task is shared and only the evidence interface changes.

Do not claim:

- tool use is bad in general;
- the reason is known from metrics alone.

Preferred artifact:

- one compact selected-slice table, not a full wide panel table.

Drafting guidance:

- this should be the longest subsection in Section 5;
- still keep it to controlled comparisons, not one paragraph per model.

### 5.3 Behavior across the ladder

Purpose:

- support Claim 4 with the ladder as the organizing object;
- use the same end-to-end comparison from 5.1, but level by level.

Claim support:

- `docs/notes/CLAIMS.md` Claims 3 and 4
- `docs/notes/EVIDENCE_MAP.md` Claims 3 and 4

Required internal arc:

1. the end-to-end ordering persists level by level;
2. the gap to `pc_greedy` widens as accepted instances become harder.

#### 5.3a Model-family spread

Load-bearing numbers:

- `llm_raw`, `Avg L1-L5`:
  - `GPT-5.5=0.697`
  - `Gemini-3-Flash=0.281`
  - `Sonnet-4.6=0.314`
  - `GPT-5.4-mini=0.145`
  - `Haiku-4.5=0.113`
- `GPT-5.5` within-model active range:
  - `llm_stats=0.640`
  - `llm_raw=0.697`
  - `pc_cpdag_llm=0.701`

Allowed interpretation:

- cross-model spread is large relative to within-strong-model policy variation;
- ACDB is sensitive to capability differences across the panel.

Do not claim:

- method does not matter;
- this is a universal law about all LLM agents.

#### 5.3b Difficulty scaling

Load-bearing numbers:

- `pc_greedy`: `L1 0.802 -> L5 0.642`
- `GPT-5.5 llm_raw`: `L1 0.824 -> L5 0.537`
- `Gemini-3-Flash llm_raw`: `L1 0.414 -> L5 0.185`
- `Sonnet-4.6 llm_raw`: `L1 0.426 -> L5 0.252`
- `Haiku-4.5 llm_raw`: `L1 0.192 -> L5 0.047`

Allowed interpretation:

- on this ladder, LLM-only policies weaken faster with graph complexity than the classical active baseline;
- the trend should be stated with both absolute values and relative drop language.

Do not claim:

- a theorem about general LLM scaling behavior;
- certainty about the internal mechanism.

Preferred artifact:

- one full-width scaling figure with `dir_f1` and `SHD` across `L1-L5`;
- selected end-to-end series plus the shared `pc_greedy` reference.

Drafting guidance:

- keep the subsection comparative rather than exhaustive;
- if space is tight, prioritize the scaling comparison over full model narration.

## What must stay out of Section 5

- any reopening of calibration as its own subsection;
- any discussion of the random baseline as a main evaluated method;
- any mention of `L0`;
- any `oracle` row discussion;
- any `pc_obs` discussion;
- any correlation-only probe;
- any prompt or provider discussion;
- any token-cost or latency discussion;
- any trace-derived explanation such as analysis loops, repeated testing, or delayed commitment.

## Figure and table ownership

Main-paper results artifacts should be limited to:

1. one cross-model `Avg L1-L5` figure for `dir_f1` and `SHD`;
2. one compact selected-slice table for layered scores;
3. one per-level `L1-L5` scaling figure for selected methods/models

Do not use as drafting anchors:

- `research/figures/tab_ladder_results.tex`
- the old headline figure flow in `research/sections/05_results.tex`
- any artifact that still assumes `L0`, `oracle`, older models, or older ablation sets

## Stale audit of `05_results.tex`

The current file is not a valid drafting source because it still contains:

- older model panel language;
- `L0--L5` framing;
- `oracle`, random baseline, and correlation-only probe narration in the main section;
- old significance-test framing;
- old Sonnet-vs-GPT story as the organizing arc;
- an efficiency subsection that is not load-bearing for the current paper.

Treat that file as deletion context only.

## Acceptance criteria

The eventual Section 5 draft is correct only if:

- it has one short opening paragraph and exactly three substantive subsections;
- `5.1` supports Claim 1;
- `5.2` supports Claims 5, 6, and 7;
- `5.3` supports Claims 3 and 4;
- the strongest-model competitive point remains secondary;
- `L0`, `oracle`, random baseline, and correlation-only probe are absent from the main results narrative;
- the section reads as controlled benchmark interpretation rather than a leaderboard summary.
