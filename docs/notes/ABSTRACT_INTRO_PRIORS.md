# ACDB abstract and introduction priors

## Objective

This note is the drafting packet for the live NeurIPS paper abstract and introduction.

Live paper targets:

- `research/main.tex` abstract
- `research/sections/01_introduction.tex`

This file is not draft prose. It is the source-controlled prior set that the drafter should use to keep the live abstract and introduction aligned with the current paper identity.

## Source-of-truth hierarchy

Use sources in this order:

1. **Benchmark/task/calibration code truth**
   - `run_ladder.py`
2. **Paper-facing claims and evidence**
   - `docs/notes/CLAIMS.md`
   - `docs/notes/EVIDENCE_MAP.md`
   - `docs/notes/RESULTS.md`
3. **Paper framing and structure**
   - `docs/notes/ABSTRACT.md`
   - `docs/notes/INTRODUCTION.md`
   - `docs/notes/OUTLINE.md`

Do not draft from:

- `README.md`
- ad hoc terminal summaries
- stale TeX wording in the live paper
- trace-behavior narratives

## Hard drafting constraints

- Benchmark-first, not model-ranking-first.
- Task-first introduction, not benchmark-name-first.
- LLM agents are mentioned only after ACDB is introduced.
- Claim only what the current output metrics and benchmark artifacts support.
- Do not rely on trace psychology, hidden reasoning, or speculative model mechanisms.
- Do not reintroduce old benchmark numbers, old ladder/version labels, or removed section structure.

## Abstract priors

Use a 5-sentence structure.

### Sentence 1 - Problem

- Job:
  - state the evaluation problem in the setting of active causal discovery.
- Must say:
  - aggregate benchmark scores can collapse distinct reasoning sub-tasks.
  - active causal discovery requires observational analysis plus budgeted interventions.
- Supports:
  - `docs/notes/ABSTRACT.md`
  - `docs/notes/CLAIMS.md` Claim 1
- Must not say:
  - that the paper is primarily a model comparison
  - that the paper proves whether LLMs can do causal discovery in general

### Sentence 2 - Artifact

- Job:
  - introduce ACDB as the main contribution.
- Must say:
  - fixed hidden linear-Gaussian SCM generator
  - fixed observe-intervene-submit protocol
  - layered scoring contract over observational and active structure-recovery slices
- Supports:
  - `docs/notes/ABSTRACT.md`
  - `run_ladder.py`
  - `docs/notes/CLAIMS.md` Claim 1
- Must not say:
  - implementation details like exact ladder values
  - prompt or tool-schema details

### Sentence 3 - Calibration

- Job:
  - make calibration part of the contribution, not a side note.
- Must say:
  - ACDB calibrates difficulty with expected/random directed-F1 floors
  - ladder placement avoids structurally guessable high-density regions
- Supports:
  - `docs/notes/ABSTRACT.md`
  - `docs/notes/EVIDENCE_MAP.md` calibration section
- Must not say:
  - that expected F1 replaces empirical evaluation
  - that calibration alone explains model performance

### Sentence 4 - Empirical demonstration

- Job:
  - summarize what the benchmark reveals using the current model/ablation panel.
- Must say:
  - wide capability spread across the panel
  - strong benefit from structural priors
  - no consistent directed-graph gains from statistical tools
- Supports:
  - `docs/notes/ABSTRACT.md`
  - `docs/notes/CLAIMS.md` Claims 3, 5, 7
  - `docs/notes/RESULTS.md`
- Must not say:
  - exact one-model headlines unless later locked in final wording
  - trace-based mechanism claims

### Sentence 5 - Main takeaway

- Job:
  - state the paper's evaluative conclusion.
- Must say:
  - ACDB is a scientific instrument for agentic causal reasoning
  - such benchmarks should report layered failures separately rather than compress them into a single graph score
- Supports:
  - `docs/notes/ABSTRACT.md`
  - `docs/notes/CLAIMS.md`
- Must not say:
  - that current LLMs solve the task
  - that one policy is universally best

## Introduction priors

The structure is defined in `docs/notes/INTRODUCTION.md`. Use that note as the paragraph skeleton and this note as the sentence-level prior source.

### Paragraph 1 - Active causal discovery as a sequential problem

- Must establish:
  - the task is sequential
  - observational evidence alone does not generally identify a unique DAG
  - interventions are part of the task, not a benchmark gimmick
- Supports:
  - `docs/notes/INTRODUCTION.md`
  - `docs/notes/CLAIMS.md`
- Citation placeholders:
  - active causal discovery
  - observational non-identifiability / equivalence classes

### Paragraph 2 - Why SCMs make this domain diagnostic

- Must establish:
  - SCMs define a precise hidden causal state
  - CPDAG / MEC theory separates what observation can identify from what intervention must resolve
  - this is why layered scoring is principled here
- Supports:
  - `docs/notes/INTRODUCTION.md`
  - `docs/notes/CLAIMS.md` Claim 1
- Citation placeholders:
  - SCM foundations
  - Markov equivalence / CPDAG
  - intervention identifiability

### Paragraph 3 - Evaluation gap

- Must establish:
  - current agent evaluations often collapse competencies
  - even causal-discovery evaluations do not always foreground separability and calibration
- Supports:
  - `docs/notes/INTRODUCTION.md`
  - `docs/notes/OUTLINE.md`
  - `docs/notes/CLAIMS.md` Claim 2
- Citation placeholders:
  - benchmark / evaluation-design references
  - causal-discovery benchmark references

### Paragraph 4 - ACDB as the answer

- Must establish:
  - fixed SCM generator
  - fixed observe-intervene-submit protocol
  - paired seeds
  - layered scoring
  - calibration through expected/random floor and ladder placement
- Must introduce:
  - LLM agents and classical baselines under the same protocol
- Supports:
  - `docs/notes/INTRODUCTION.md`
  - `docs/notes/ABSTRACT.md`
  - `docs/notes/OUTLINE.md`

### Paragraph 5 - What ACDB reveals

- Must preview:
  - layered scores reveal distinctions hidden by a single metric
  - structural priors help but do not remove the orientation burden
  - statistical tools do not reliably improve directed recovery
  - performance varies strongly across model families and graph complexity
- Supports:
  - `docs/notes/CLAIMS.md`
  - `docs/notes/RESULTS.md`
- Must not say:
  - exact ranking stories unless later frozen in final results language
  - behavioral claims that depend on trace analysis

### Contribution bullets

Keep exactly 3 bullets and keep them benchmark-first:

1. ACDB as the benchmark artifact with a theory-grounded layered scoring contract
2. calibration methodology through expected/random directed-F1 floors and ladder design
3. empirical study showing that the benchmark localizes failure across observational recovery, interventional orientation, and end-to-end integration

## Live-paper synchronization notes

The abstract in `research/main.tex` is now substantially aligned with the notes. Future edits should preserve:

- the benchmark-first opening;
- the artifact -> calibration -> empirical reveal -> evaluative takeaway order;
- the current title and one-line thesis from `docs/notes/ABSTRACT.md` and `docs/notes/OUTLINE.md`.

The introduction has already been rewritten into the benchmark-first framing. Future edits should refine it locally, not re-open the older random-baseline-first or audit-style framing.

Drafting rule:

- replace whole paragraphs when the framing changes;
- otherwise make only local sentence-level edits.

## Citation priors

### Must cite in abstract/introduction

- SCM / causal graphical models
  - check `research/references.bib`
- CPDAG / MEC / Markov equivalence
  - check `research/references.bib`
- interventional identifiability / active causal discovery
  - partially present in `research/references.bib`
- PC baseline
  - classical causal-discovery references already present
- benchmark / evaluation-design comparators
  - some benchmark citations already present

### Likely missing or needing later addition

- calibration-specific reference if used in the intro gap paragraph
- benchmark papers closer to E&D framing if explicitly cited in prose

### Current bibliography status

`research/references.bib` already includes:

- `spirtes2000causation`
- `verma1990equivalence`
- `andersson1997characterization`
- `eberhardt2005interventions`
- `hauser2012gies`
- several LLM causal-reasoning papers
- several agent-benchmark papers

It does not yet look like a finished abstract/introduction bibliography set. Draft with placeholders where needed, then do a later citation pass.

## Drafting order

1. Write a fresh abstract from the 5-sentence prior structure above
2. Write a fresh introduction from `INTRODUCTION.md` plus the paragraph priors in this note
3. Add citation placeholders during drafting
4. Only after the prose exists, do a citation-tightening pass

## Acceptance criteria

The abstract is ready only if:

- every sentence maps to one of the prior slots above
- it is benchmark-first
- it contains no stale older-ladder numbers
- it makes no trace-based claims

The introduction is ready only if:

- it follows the 5-paragraph order from `INTRODUCTION.md`
- the first two paragraphs make sense before ACDB is named
- LLM agents are first mentioned only after ACDB is introduced
- contribution bullets match `docs/notes/CLAIMS.md`
- no paragraph depends on the stale current TeX framing
