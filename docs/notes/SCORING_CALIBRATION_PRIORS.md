# ACDB Section 3 priors: Scoring and Calibration

## Objective

This note is the drafting packet for the paper section that should appear as `Scoring and Calibration`.

Its job is narrow and benchmark-owned:

1. define how ACDB scores submitted graphs;
2. justify why those scores are interpretable across levels.

This section should support only the benchmark claims about:

- layered scoring as a diagnostic contract;
- calibration as a condition for valid interpretation.

It should not drift into agent behavior, prompt design, or model findings.

## Live drafting target

- Primary target: `research/sections/03_benchmark.tex`
- Section title in the paper: `Scoring and Calibration`

This section begins only after `Benchmark Setup` has already established:

- the hidden SCM world;
- the CPDAG observational ceiling;
- the public `observe -> intervene -> submit` protocol;
- the accepted benchmark instance and budget semantics.

## Source-of-truth hierarchy

Use sources in this order.

1. Score definitions and scoring semantics:
   - `src/causal_discovery/scoring/scores.py`
   - `src/causal_discovery/scoring/submission.py`
2. Benchmark truth objects consumed by scoring:
   - `src/causal_discovery/benchmark/instance.py`
3. Ladder and paired-run protocol:
   - `run_ladder.py`
4. Calibration findings:
   - `docs/notes/random-baseline-ladder-findings.md`
5. Paper-facing claims and evidence:
   - `docs/notes/CLAIMS.md`
   - `docs/notes/EVIDENCE_MAP.md`
   - `docs/notes/FIGURES.md`

Non-authoritative:

- `research/sections/03_benchmark.tex` as currently written
- any old paper wording that uses internal ladder version labels as paper concepts
- any old observational-only LLM panel language

## What the section must accomplish

By the end of Section 3, a reviewer should understand:

- what is scored against the true CPDAG and what is scored against the true DAG;
- why unresolved undirected edges are meaningful benchmark outputs rather than mere omissions;
- why directed F1 needs a structure-blind random floor for interpretation;
- how the current ladder is calibrated to avoid structurally guessable regions;
- why these choices make ACDB a diagnostic evaluation instrument rather than a single-score benchmark.

## What the section must not do

- do not restate the benchmark world or full interaction protocol;
- do not justify prompt or tool choices;
- do not narrate model results;
- do not lean on trace interpretation;
- do not use `v0` / `v1` as paper-facing concepts;
- do not turn the calibration note into a mini-report.

## Section shape

This section should follow the same sober structure as Sections 1 and 2:

1. one short opening paragraph;
2. three named subsections;
3. one compact ladder table;
4. only the load-bearing equations.

Do not write it as a miscellaneous benchmark dump.

## Recommended section structure

### Opening paragraph

Purpose:

- bridge from `Benchmark Setup` to benchmark interpretation.

Must say:

- Section 2 defined the benchmark world and public contract.
- Section 3 defines the scoring contract and calibration methodology used to interpret submissions.
- These choices are part of the benchmark contribution, not later analysis.

Keep out:

- any model or baseline examples;
- any long reminder of the protocol;
- any formula immediately in the opening paragraph.

### 3.1 Layered Scoring Contract

Purpose:

- define the benchmark outputs precisely and tie each one to a truth object.

Must say:

- skeleton metrics are computed against the true CPDAG skeleton;
- compelled-edge metrics are computed against the directed part of the true CPDAG;
- directed metrics and SHD are computed against the true DAG;
- undirected submitted edges are distinct from missing or reversed directed edges;
- DAG scoring is intentionally not MEC-aware because observational ambiguity is already captured by the CPDAG layer;

Code anchors:

- `src/causal_discovery/scoring/scores.py:40-71`
- `src/causal_discovery/scoring/scores.py:74-121`
- `src/causal_discovery/scoring/scores.py:124-165`
- `src/causal_discovery/benchmark/instance.py`

Keep out:

- calibration plots;
- ladder-region discussion beyond a one-line handoff;
- agent-specific stories.

#### Equation inventory for 3.1

Keep in prose:

- generic precision / recall / F1 definitions;
- SHD semantics.

Reason:

- these are standard enough not to deserve main-text equation slots here;
- the section should stay normal and readable by simple paper reading.

For SHD, define the four one-error cases in prose:

- extra;
- missing;
- reversed;
- unresolved.

Do not over-formalize SHD.

### 3.2 Difficulty Ladder

Purpose:

- define the operational benchmark catalog used in the paper.

Must say:

- the ladder varies graph size, edge count, sample counts, noise variance, and budget slack;
- the ladder table in the paper must come directly from `run_ladder.py`;
- the same seed manifest is reused across agents, so comparisons are paired at the instance level;
- the ladder is meant to increase structural difficulty while keeping the public sample interface interpretable.

Code anchors:

- `run_ladder.py:124-136`
- `run_ladder.py:189-217`
- `run_ladder.py:220-235`

Keep in the main paper:

- a compact ladder table.

The ladder table should remain in Section 3, not Section 2, because here it is serving an evaluation-function role:

- it tells the reader what score comparisons are being made across.

Keep out:

- full instance-generation details;
- rejection-loop details;
- old ladder history.

### 3.3 Random-Floor Calibration

Purpose:

- justify why directed-F1 interpretation requires calibration against structure-blind random guessing.

Must say:

- directed F1 has a non-trivial blind-random floor that depends strongly on density;
- the random baseline must be structure-blind and may use public `d` but not true `k`;
- density can inflate apparent performance even when no causal reasoning is happening;
- the current ladder is chosen to keep non-tutorial levels below the most guessable regime;
- the random-floor heuristic is a calibration device, not a theorem.

Note anchors:

- `docs/notes/random-baseline-ladder-findings.md`
- `docs/notes/EVIDENCE_MAP.md`
- `docs/notes/FIGURES.md`

Keep out:

- old GPT/Sonnet random comparisons unless reused later in appendix;
- any suggestion that random-floor calibration itself measures model quality.

#### Equation inventory for 3.3

Keep explicit:

1. Density

```tex
\rho = \frac{k}{\binom{d}{2}}.
```

2. Probe-fit random-floor heuristic

```tex
\mathbb{E}[\mathrm{DirF1}_{\mathrm{random}}] \approx \frac{\rho}{1 + 2\rho}.
```

Required phrasing:

- describe this as a probe-based approximation or heuristic fit;
- do not call it a bound;
- do not present it as a theorem.

Budget handoff:

- the budget formula stays defined in Section 2;
- Section 3 may refer back to it when discussing budget slack as a ladder axis;
- do not re-own the budget equation here unless the draft absolutely needs a reminder sentence.

## Figure ownership

This section is the natural home for:

- the scoring-layer diagram;
- the density/random-floor figure;
- the ladder-region figure, if it survives the page budget.

This section is not the home for:

- representative model outputs;
- trace snippets;
- prompt or tool schema diagrams.

## Claims supported here

Safe direct claims:

- layered scoring maps to distinct theoretical objects in the task;
- unresolved orientations should not be collapsed into the same category as missing or reversed directed edges;
- directed-F1 interpretation requires a structure-blind random floor;
- graph density can inflate apparent directed-F1 performance if not calibrated;
- the current ladder is explicitly calibrated against that hazard.

Claims to avoid here:

- any model ranking claim;
- any statement that the random-floor heuristic is a formal law;
- any claim about why a specific model succeeded or failed.

## Readability guardrails

This section must be understandable by simple reading.

That means:

- keep the opening short;
- keep the number of equations small;
- prefer clear prose over generic metric algebra;
- use the table to carry ladder detail;
- avoid benchmark-philosophy filler that is not tied to ACDB.

If a sentence sounds like generic benchmark commentary rather than a statement about ACDB's scoring or calibration, cut it.

## Acceptance criteria

This priors packet is satisfied only if the drafted Section 3:

1. uses exactly this three-part structure:
   - layered scoring contract
   - difficulty ladder
   - random-floor calibration
2. makes the metric-to-truth-object mapping explicit;
3. keeps only the load-bearing equations in main text:
   - density
   - random-floor heuristic
4. keeps the ladder table in the main section;
5. avoids model or prompt narration entirely;
6. reads like a normal NeurIPS benchmark-methodology section rather than a stale benchmark memo.
