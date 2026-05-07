# ACDB introduction skeleton

## Objective

The introduction must establish ACDB as a benchmark-and-evaluation paper whose core contribution is a diagnostic instrument for active causal discovery. It should explain why active causal discovery is a strong substrate for agent evaluation, why SCM theory makes separability principled, what existing evaluations miss, and what ACDB adds.

The introduction is not a model-ranking section. It should not read like "we tested several LLMs and here are the scores." The empirical preview should be brief and should only demonstrate what the instrument reveals.

## Opening stance

- Open with the task, not the benchmark name.
- Open with active causal discovery as a sequential problem.
- Do not mention LLM agents before ACDB is introduced.
- Do not open with raw numerical-interface discussion.
- Do not open with model results.

## Structure

Use 5 paragraphs followed by 3 contribution bullets.

### Paragraph 1 - Active causal discovery is a sequential agent problem

- Purpose:
  - establish the task before the benchmark.
  - make clear that this is not just static graph prediction.
- Must say:
  - active causal discovery requires an agent to form structure hypotheses from observational evidence, choose interventions under budget, integrate the outcomes, and submit a final graph.
  - the task is sequential because later actions depend on earlier evidence.
  - observational evidence alone is generally insufficient for full DAG recovery.
- Allowed preview:
  - a brief statement that interventions are part of the task itself, not an added benchmark gimmick.
- Keep out:
  - ACDB name and details.
  - LLM agent panel.
  - detailed metric language.
- Citation placeholders:
  - active causal discovery background
  - observational non-identifiability / equivalence-class background

### Paragraph 2 - Why SCMs make this domain a strong evaluation substrate

- Purpose:
  - justify the domain choice.
  - explain why this task supports principled diagnostic evaluation.
- Must say:
  - SCMs provide an exact hidden causal state with intervention semantics.
  - the distinction between the true DAG, the observational equivalence class, and the intervention-resolvable remainder is mathematically explicit.
  - CPDAGs / MECs make the observational ceiling explicit instead of heuristic.
  - this lets the benchmark separate adjacency recovery, observationally compelled orientation, and intervention-resolved orientation.
- Allowed preview:
  - this kind of exact separability is unusual in generic agent benchmarks.
- Keep out:
  - full formal definitions.
  - scoring equations.
  - implementation details of graph generation.
- Citation placeholders:
  - SCM foundations
  - CPDAG / MEC theory
  - intervention identifiability / active orientation references

### Paragraph 3 - Existing evaluations are not diagnostic enough

- Purpose:
  - state the benchmark gap.
  - connect the task to the E&D framing.
- Must say:
  - many agent evaluations collapse distinct competencies into a single success metric.
  - this makes failure hard to attribute: evidence interpretation, experiment choice, intervention integration, and final commitment can all be confounded.
  - even causal-discovery evaluations often emphasize final graph quality more than clean separation of observational and interventional sub-tasks, and they do not always make calibration assumptions explicit.
- Allowed preview:
  - benchmark calibration matters because non-trivial score floors can distort interpretation.
- Keep out:
  - detailed criticism of specific papers unless needed later in related work.
  - full random-floor story.
- Citation placeholders:
  - benchmark / evaluation-design references
  - causal-discovery benchmark references
  - calibration / random-baseline reference if cited here

### Paragraph 4 - ACDB as the answer

- Purpose:
  - introduce ACDB only after the task and gap are clear.
  - define the artifact in one paragraph.
- Must say:
  - ACDB fixes the SCM family, observe-intervene-submit protocol, accepted seed map, and scoring contract.
  - ACDB evaluates layered recovery rather than a single graph score.
  - ACDB includes calibration through expected/random directed-F1 floors and ladder placement.
  - ACDB supports paired comparisons across agents on identical instances.
- Must introduce LLM agents here, not earlier:
  - the benchmark is used to evaluate LLM agents and classical baselines under the same protocol.
- Keep out:
  - long list of policies.
  - prompt/tool schema details.
  - per-level ladder parameters.
- Citation placeholders:
  - only if needed for the classical baseline mention; otherwise benchmark description can stand without external citation

### Paragraph 5 - What the instrument reveals

- Purpose:
  - preview the empirical value of the benchmark.
  - show that the apparatus yields separable findings.
- Must say:
  - layered scores reveal distinctions hidden by a single graph metric.
  - structural priors help substantially but do not remove the interventional orientation burden.
  - statistical-tool access does not reliably improve directed recovery on the current benchmark family.
  - performance varies strongly across model families and degrades with graph complexity.
- Allowed preview:
  - the strongest model can be competitive with the classical active baseline on some slices.
- Keep out:
  - exact tables, exact rankings, or fragile one-model headlines.
  - trace-based behavioral interpretations.
- Citation placeholders:
  - none required if this paragraph is purely a paper-internal findings preview

## Contribution bullets

Use 3 bullets. Keep them benchmark-first.

### Contribution 1

- ACDB: a benchmark for active causal discovery with a fixed SCM generator, fixed observe-intervene-submit protocol, and a theory-grounded layered scoring contract.

### Contribution 2

- A calibration methodology for active causal-discovery evaluation, including expected/random directed-F1 floors and a graph-scale ladder designed to avoid structurally guessable regions.

### Contribution 3

- An empirical benchmark study across classical and LLM-based policies showing that the instrument can localize failure to observational structure recovery, interventional orientation, and end-to-end integration.

## Citation checklist

Before drafting prose, make sure the introduction has placeholders or concrete citations for:

- SCM / causal graphical model foundation
- CPDAG / MEC / Markov equivalence
- interventional identifiability / active causal discovery
- PC algorithm
- benchmark / evaluation-design references relevant to diagnostic evaluation
- any calibration-specific paper cited in the gap paragraph

Primary sources are preferred where available.

## Guardrails

- Do not turn the introduction into a generic benchmark pitch with causality swapped in.
- Do not describe the benchmark as a pure causality-theory paper; it is still an evaluation artifact.
- Do not explain score formulas or random-floor math here.
- Do not introduce prompt design or tool schemas here.
- Do not make claims that rely on trace interpretation.
- Do not overstate the empirical findings beyond what `RESULTS.md` supports.
- Do not let the contribution bullets become model-ranking bullets.

## Drafting check

The introduction is ready to draft only if:

- the first two paragraphs make sense without naming ACDB yet;
- ACDB appears only after the task, domain rationale, and benchmark gap are established;
- LLM agents are first mentioned in Paragraph 4 or later;
- the contribution bullets can be copied directly into the draft with only stylistic edits.
