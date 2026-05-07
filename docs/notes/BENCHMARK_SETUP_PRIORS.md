# ACDB benchmark setup priors

## Objective

This note is the drafting packet for the section that should appear in the paper as `Benchmark Setup`.

This section should be the first substantial section after the introduction. It must be broad enough to stand on its own:

- define the hidden causal world;
- explain the observational ceiling and why interventions are necessary;
- define the public interaction protocol;
- define what an accepted benchmark instance contains.

It should feel like a real benchmark section, not a thin background note.

## Live drafting target

- Primary target: `research/sections/02_background.tex`
- Conceptual section title in prose: `Benchmark Setup`
- Recommended section label: `sec:benchmark`

Keep the file path unchanged for now. Change the section title and the section label.

## Source-of-truth hierarchy

Use sources in this order.

1. Runtime contract:
   - `src/causal_discovery/runtime/session.py`
2. Public benchmark instance shape:
   - `src/causal_discovery/benchmark/instance.py`
3. Configuration and ladder truth:
   - `src/causal_discovery/config/v1.py`
   - `run_ladder.py`
4. Causal/scoring objects referenced by the section:
   - `src/causal_discovery/scoring/scores.py`
5. Paper framing:
   - `docs/notes/OUTLINE.md`
   - `docs/notes/INTRODUCTION.md`
   - `docs/notes/ABSTRACT_INTRO_PRIORS.md`

## What this section must accomplish

By the end of this section, a reviewer should understand:

- the hidden world is a linear-Gaussian SCM over a DAG;
- observations generally identify a CPDAG, not a unique DAG;
- interventions are part of the task itself because they resolve CPDAG ambiguity;
- the runtime exposes a strict `observe -> intervene -> submit` protocol;
- each accepted public instance packages fixed observational data, a budget, and benchmark-owned hidden truth that the evaluator sees but the agent does not.

## What this section must not do

- do not define the score formulas in detail;
- do not define random-floor calibration in detail;
- do not narrate policies or model behavior;
- do not dump generator internals line-by-line.

Those belong later.

## Why the earlier split was wrong

The old split between `Problem Setup` and `The ACDB Benchmark` made Section 2 too thin and Section 3 too overloaded.

- `Problem Setup` carried the causal world and session contract, but not enough benchmark substance to feel like a full section.
- `The ACDB Benchmark` then repeated the setup boundary and mixed construction, scoring, ladder design, and calibration.

The fix is to merge task/world/protocol/accepted-instance material into one stronger section: `Benchmark Setup`.

## Recommended structure

Use 3 subsections.

### 2.1 Task and causal world

- Must say:
  - each instance is generated from a hidden DAG-backed linear-Gaussian SCM;
  - the SCM induces an observational distribution and interventional distributions under hard `do(X_i = v)` actions;
  - observational data do not generally identify a unique DAG;
  - the observational target is the CPDAG, with compelled directions and unresolved undirected edges.
- Keep:
  - the DAG -> CPDAG -> intervention figure in this subsection.
- Code anchors:
  - `src/causal_discovery/benchmark/instance.py:24-35`
  - `src/causal_discovery/runtime/session.py:48-64`
- Equation inventory:
  - linear-Gaussian SCM equation
  - optional compact observational/interventional notation

### 2.2 Interaction protocol

- Must say:
  - `observe()` releases the fixed observational sample exactly once;
  - interventions are only legal after observation;
  - each intervention consumes one unit of budget and returns a fresh interventional sample;
  - `submit_graph(...)` terminates the session;
  - mixed directed/undirected submissions are allowed, so unresolved orientation is explicit rather than silently collapsed into generic graph error.
- Code anchors:
  - `src/causal_discovery/runtime/session.py:40-47`
  - `src/causal_discovery/runtime/session.py:48-64`
  - `src/causal_discovery/runtime/session.py:66-120`

### 2.3 Accepted benchmark instance

- Must say:
  - accepted instances are built by sampling a DAG, parameterizing a linear-Gaussian SCM, rejecting graph/SCM profiles that violate benchmark constraints, sampling observational data, and computing a benchmark-owned intervention reference set;
  - node identities are anonymized by permutation before the public instance is exposed;
  - the public intervention budget is benchmark-owned and defined by:
    `B = |\mathcal{I}^\star| + \texttt{budget\_slack}`;
  - the accepted public instance exposes only the observational dataset, variable identities, and budget through the runtime, while the evaluator retains the hidden DAG, SCM, CPDAG, and optimal intervention set.
- Code anchors:
  - `src/causal_discovery/benchmark/instance.py:38-88`
  - `run_ladder.py:124-136`
- Equation inventory:
  - budget formula appears here, once

## Safe claims for this section

- active causal discovery is not generally solved by observations alone;
- CPDAG is the correct observational target object for this benchmark family;
- interventions resolve ambiguity left by the observational equivalence class;
- ACDB exposes a strict public session protocol over fixed observational data and budgeted single-node interventions;
- accepted benchmark instances are benchmark-owned objects with hidden truth and public runtime state.

## Claims to avoid here

- do not claim the benchmark covers causal discovery in general;
- do not claim the current SCM family is universally realistic;
- do not claim the random-floor approximation or ladder design here;
- do not claim any model behavior.

## Citation checklist

This section should have placeholders or concrete citations for:

- SCM / causal graphical model foundations
- Markov equivalence
- CPDAG characterization
- intervention semantics / active causal discovery
- faithfulness / causal sufficiency if mentioned

Existing relevant entries already present in `research/references.bib`:

- `spirtes2000causation`
- `verma1990equivalence`
- `andersson1997characterization`
- `eberhardt2005interventions`
- `hauser2012gies`

## Acceptance criteria

This priors packet is satisfied only if the drafted section:

1. reads like a full `Benchmark Setup` section;
2. can stand directly after the introduction without feeling incomplete;
3. includes the world, protocol, and accepted-instance boundaries in one place;
4. leaves scoring formulas and calibration to the next section.
