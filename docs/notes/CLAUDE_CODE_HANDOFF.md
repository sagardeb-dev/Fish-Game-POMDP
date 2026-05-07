# ACDB paper handoff for Claude Code

This note is the operational handoff for editing the ACDB paper in Claude Code. It is not paper prose. It is the working prior for how to understand the project, how to edit it safely, and how not to destabilize the LaTeX.

## 1. What this paper is

The live paper is:

- a benchmark and evaluation paper for active causal discovery;
- a controlled SCM-instance-family paper;
- a paper about diagnostic separability through causal objects, score layers, and ablations.

The live paper is not:

- a leaderboard paper;
- a claim that LLMs can or cannot do causal discovery in general;
- a prompt-engineering paper;
- a trace-analysis paper;
- a generic synthetic-benchmark paper.

The central artifact is:

> a controlled family of active causal-discovery instances generated from linear-Gaussian SCMs, exposed through a fixed observe-intervene-submit runtime, and interpreted through a layered CPDAG/DAG scoring contract.

Everything in the paper should reinforce that artifact.

## 2. Live paper structure

Current `research/main.tex` order:

1. `Introduction`
2. `Benchmark Setup`
3. `Scoring and Calibration`
4. `Experimental Setup`
5. `Results`
6. `Related Work`
7. `Conclusion`

Important:

- `Discussion` has been removed from the live paper.
- There is currently no standalone `Limitations` section.
- Scope boundaries are carried implicitly in Section 5 language and in the conclusion.
- `research/sections/06_discussion.tex` is stale and unreferenced.
- `research/sections/07_related_work.tex` compiles as Section 6.
- `research/sections/08_conclusion.tex` compiles as Section 7.

## 3. Source-of-truth hierarchy

Use sources in this order:

1. code truth:
   - `run_ladder.py`
   - `src/causal_discovery/benchmark/instance.py`
   - `src/causal_discovery/runtime/session.py`
   - `src/causal_discovery/scoring/scores.py`
   - `src/causal_discovery/agents/tool_schema.py`
   - `src/causal_discovery/agents/llm.py`
2. paper-note truth:
   - `docs/notes/PAPER_PRIORS.md`
   - section-specific priors
   - `docs/notes/CLAIMS.md`
   - `docs/notes/EVIDENCE_MAP.md`
3. results truth:
   - `docs/notes/RESULTS.md`
4. live TeX:
   - use as current integration target, not as authority when it conflicts with the notes/code

Do not draft from:

- stale removed sections;
- ad hoc terminal summaries;
- old ladder/version language lingering in backups;
- remembered older numbers.

## 4. Note stack to read before editing

Minimum reading order:

1. `docs/notes/CLAUDE_CODE_HANDOFF.md`
2. `docs/notes/PAPER_PRIORS.md`
3. the relevant section prior
4. `docs/notes/CLAIMS.md`
5. `docs/notes/RESULTS.md` if the section mentions results

Use these section priors:

- abstract / intro:
  - `docs/notes/ABSTRACT.md`
  - `docs/notes/ABSTRACT_INTRO_PRIORS.md`
  - `docs/notes/INTRODUCTION.md`
- Section 2:
  - `docs/notes/BENCHMARK_SETUP_PRIORS.md`
- Section 3:
  - `docs/notes/SCORING_CALIBRATION_PRIORS.md`
- Section 4:
  - `docs/notes/EXPERIMENTAL_SETUP_PRIORS.md`
- Section 5:
  - `docs/notes/RESULTS_PRIORS.md`
- Section 6:
  - `docs/notes/RELATED_WORK.md`
  - `docs/notes/RELATED_WORK_PRIORS.md`

## 5. Writing philosophy

The paper should read like a serious causal benchmark paper, not like a benchmark manifesto and not like a model-results story.

The order of emphasis is:

1. benchmark artifact;
2. scoring and calibration;
3. empirical demonstration of what the instrument reveals.

The paper should make the benchmark contribution visible by construction:

- concrete instance family;
- clear public/hidden split;
- theory-aligned score layers;
- controlled ablations;
- bounded empirical interpretation.

Do not keep explaining that the benchmark is principled. Make it obvious from the section order and the precision of the writing.

## 6. Style constraints

Prefer:

- concrete benchmark nouns: instance, SCM, DAG, CPDAG, observational sample, interventional sample, submitted graph, paired seed;
- controlled-comparison verbs: isolates, holds fixed, removes, supplies, compares;
- bounded interpretation: on this benchmark family, across the current panel, under the reported ladder.

Avoid:

- benchmark-manifesto language;
- leaderboard language;
- self-praise about being diagnostic;
- trace psychology;
- broad claims about causal reasoning in general;
- strong claims about model internals.

## 7. Live paper facts that should not drift

- Title: `Active Causal Discovery as a Diagnostic Benchmark for LLM Agents`
- Ladder in paper prose: `L1-L5` only
- Five in-scope policies:
  - `pc_greedy`
  - `llm_raw`
  - `llm_stats`
  - `pc_cpdag_llm`
  - `llm_stats_cpdag_greedy`
- Model panel:
  - `GPT-5.5`
  - `GPT-5.4-mini`
  - `Sonnet-4.6`
  - `Haiku-4.5`
  - `Gemini-3-Flash`
- Excluded from paper prose:
  - `L0`
  - `oracle` as an evaluated method
  - `pc_obs`
  - random baseline as a main compared method
  - correlation-only probe
  - provider failure narratives

## 8. Related-work stance

Related Work is intentionally narrow and LLM-first.

It should stay focused on:

- LLMs for causal reasoning and graph construction;
- direct benchmark neighbors for LLM causal discovery and interactive graph discovery;
- at most one short broad pointer to scientific-discovery benchmarks.

Do not re-expand it into:

- theory-heavy causal-discovery history;
- broad agent-benchmark catalogues;
- benchmark-methodology survey material.

## 9. LaTeX stability rules

This is the most important operational constraint.

The paper has already been destabilized once by mixing content changes, float controls, and figure redesign in the same passes. Do not repeat that.

### Non-negotiable workflow

1. Freeze content.
2. Change one layout variable only.
3. Compile.
4. Verify rendered placement, not just labels or page numbers.
5. Only then move to the next variable.

### Hard rules

- Do not mix prose rewrites with layout changes in the same pass.
- Do not add global float hacks during content work.
- Avoid `[H]`, `\FloatBarrier`, `placeins`, or package-level float controls unless the task is explicitly a dedicated layout pass.
- Do not resize multiple figures/tables at once.
- Do not “fix” adjacent layout issues while changing one float.
- Use fresh asset names when a figure changes shape materially, so stale compiled assets do not masquerade as the new one.

### Verification rule

Do not infer correctness from:

- aux labels,
- page numbers alone,
- “the figure is on the same page.”

Correct verification means checking the rendered PDF or screenshot and confirming:

- the float is visually under the right subsection;
- there is no large incoherent whitespace;
- the figure typography matches the paper reasonably;
- the change did not pull another float into the wrong section.

## 10. Known LaTeX pitfalls in this repo

- Figure placement is nonlocal; source order alone does not guarantee subsection placement.
- Stale generated figure assets have caused confusion before.
- Section 4 tables and results figures can put pressure on page flow quickly.
- Broad figure geometry changes can create the illusion that “font changed,” when the real issue is scaling.
- Paper changes should be compiled only after the target prose is stable; otherwise every rewrite churns page flow.

## 11. Editing discipline

When editing existing paper files:

- touch only the target section unless the change truly requires a global sync;
- if a note becomes stale because of your edit, update that note in the same pass;
- do not rename section files just to match numbering unless explicitly asked;
- preserve current terminology and section names;
- remove only stale lines that your own change invalidates.

## 12. Suggested prompting pattern for Claude Code

Use something close to this when asking Claude to edit the paper:

> Read `docs/notes/CLAUDE_CODE_HANDOFF.md`, `docs/notes/PAPER_PRIORS.md`, and the relevant section prior before editing. Treat code truth and note truth as authoritative and the live TeX as the integration target only. Keep the paper benchmark-first and LLM-second. Do not reopen removed structure such as `Discussion`. If the task is content, do not touch layout. If the task is layout, freeze content and change one layout variable only, then compile and verify rendered placement. Do not use global float hacks unless explicitly asked.

## 13. Immediate open items

- Related-work cite keys are still partially placeholder-level until the matching BibTeX entries are added.
- A future dedicated layout pass is still needed for page-budget cleanup and float polish.
- If a standalone `Limitations` section is reintroduced later, it should be a conscious structural decision, not an accidental drift.
