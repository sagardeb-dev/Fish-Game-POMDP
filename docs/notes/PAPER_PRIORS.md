# ACDB paper global priors

## Purpose

This note is the global drafting prior for the ACDB paper. Use it together with the relevant section prior before rewriting any section:

1. read this global prior;
2. read the section-specific prior;
3. read the current TeX only as a stale audit;
4. draft from code truth, notes truth, and results truth.

The paper should not feel like a sequence of benchmark-design claims followed by model scores. It should feel like a controlled causal world, exposed through an active runtime, scored by theory-aligned metrics, and used to make a small set of interpretable comparisons.

## Structural reference point

Use synthetic-causal-world papers such as Albert's World as the closest structural reference. The useful lesson is not their exact task or notation. The useful lesson is that the world/instance family is treated as the central artifact:

- the reader understands what one world is;
- the reader understands what observations and actions mean inside that world;
- the reader understands what is public, what is hidden, and why shortcuts are controlled;
- experiments inherit coherence because they are run over a clearly specified world family.

For ACDB, the corresponding central artifact is:

> a controlled family of active causal-discovery instances generated from linear-Gaussian SCMs, exposed through a fixed observe-intervene-submit runtime, and interpreted through a layered CPDAG/DAG scoring contract.

Every major section should reinforce this artifact.

## Source-of-truth hierarchy

Use sources in this order:

1. code truth:
   - `run_ladder.py`
   - `src/causal_discovery/benchmark/instance.py`
   - `src/causal_discovery/runtime/session.py`
   - `src/causal_discovery/scoring/scores.py`
   - `src/causal_discovery/agents/tool_schema.py`
   - `src/causal_discovery/agents/llm.py`
2. paper notes truth:
   - `docs/notes/CLAIMS.md`
   - `docs/notes/EVIDENCE_MAP.md`
   - section-specific priors
3. results truth:
   - `docs/notes/RESULTS.md`
4. current TeX:
   - stale audit only unless a sentence still survives the above sources.

Do not draft from the old paper. The live TeX may contain usable prose, but it is not authoritative.

## Paper identity

The paper is:

- a benchmark and evaluation paper for active causal discovery;
- a controlled SCM-instance-family paper;
- a paper about diagnostic separability through causal objects, score layers, and ablations.

The paper is not:

- a leaderboard paper;
- a broad claim about whether LLMs can or cannot do causal discovery;
- a trace-analysis paper;
- a paper about prompt engineering or provider reliability;
- a paper about arbitrary synthetic data generation.

## Contribution hierarchy

Keep the contribution hierarchy stable:

1. benchmark artifact: a controlled active causal-discovery instance family and runtime;
2. scoring/calibration: CPDAG/DAG-aligned scoring plus random-floor-aware ladder design;
3. experimental demonstration: active ablations showing that the benchmark localizes failure across observational recovery, interventional orientation, and end-to-end integration.

The empirical results matter because they demonstrate interpretability. They should not become the paper's center of gravity.

## Live paper structure

The current `research/main.tex` order is:

1. `Introduction`
2. `Benchmark Setup`
3. `Scoring and Calibration`
4. `Experimental Setup`
5. `Results`
6. `Related Work`
7. `Conclusion`

Do not reintroduce a standalone `Discussion` section casually. Do not assume a separate `Limitations` section exists. In the current draft, scope boundaries are carried implicitly through bounded claims in the results and through the conclusion's closing paragraph.

## Section roles

### 1. Introduction

Reader question:

- Why is active causal discovery a principled domain for diagnostic agent evaluation?

Must establish:

- active causal discovery is sequential;
- observations generally identify a CPDAG/equivalence class, not a unique DAG;
- SCMs provide exact hidden state and intervention semantics;
- ACDB uses that structure to evaluate separable task layers.

Tone:

- domain-first, not benchmark-philosophy-first;
- mention LLM agents only after ACDB is introduced;
- empirical preview should be brief and secondary.

### 2. Benchmark Setup

Reader question:

- What exactly is one benchmark instance, what is public, and what remains hidden?

Must establish:

- hidden DAG-backed linear-Gaussian SCM;
- observational and interventional sample generation;
- CPDAG as the observational ceiling;
- hard single-node interventions;
- public observe-intervene-submit runtime;
- accepted instance construction, filtering, anonymization, and budget semantics.

This is the most important section for global coherence. It should make the data/instance family concrete, not merely define notation.

### 3. Scoring and Calibration

Reader question:

- How are submissions over those instances interpreted?

Must establish:

- skeleton F1 and compelled-edge F1 are CPDAG-facing;
- directed F1 and SHD are DAG-facing;
- undirected submitted edges carry explicit unresolved-orientation meaning;
- the difficulty ladder and random floor make directed-F1 comparisons interpretable.

Do not reopen the world/protocol details except by short reference to Section 2.

### 4. Experimental Setup

Reader question:

- What is varied across compared methods, and what is held fixed?

Must establish:

- same accepted `L1-L5` instances;
- paired seeds;
- five in-scope policies;
- five-model panel;
- policy interfaces and tools only insofar as they define the ablations.

The section should read as experimental design, not harness documentation.

### 5. Results

Reader question:

- What do the controlled comparisons reveal?

Must establish:

- layered metrics disagree in informative ways;
- structural priors isolate observational bottlenecks;
- `pc_cpdag_llm` vs `pc_greedy` isolates active-orientation quality;
- stats tools do not reliably improve directed recovery;
- model spread and graph-scaling behavior are visible on the ladder.

Results should be compact and open-ended. Interpret only what the ablations and metrics support.

### 6. Related Work

Reader question:

- What conversations does ACDB sit inside?

Use an LLM-first lens:

- LLMs for causal reasoning and graph construction;
- direct benchmark neighbors for LLM causal discovery and interactive graph discovery;
- at most one short broad pointer to scientific-discovery benchmarks.

Do not turn related work into a second introduction.

Do not spend scarce space here on theory-heavy causal-discovery history that is already doing work in Sections 1--3.

### 7. Conclusion

Reader question:

- What should the reader remember?

Restate the benchmark artifact first, scoring/calibration second, empirical demonstration third.

The conclusion should also carry the paper's scope boundaries implicitly:

- linear-Gaussian SCM family;
- causal sufficiency / full observability;
- hard single-node interventions;
- fixed ladder and seed manifest;
- compact model and policy panel;
- no trace-mechanism claims.

## Data and instance visibility checklist

Before accepting any section rewrite, ask whether the paper has made these visible by that point:

- What is sampled?
- What is observed?
- What is intervened on?
- What is hidden from the agent?
- What is retained by the evaluator?
- What changes across levels?
- What is paired across methods?
- What is scored against CPDAG vs DAG?

Section 2 should answer most of these directly. Later sections should rely on those answers instead of redefining them.

## Tone constraints

Prefer:

- concrete benchmark nouns: instance, SCM, DAG, CPDAG, observational sample, interventional sample, submitted graph, paired seed;
- controlled-comparison language: isolates, holds fixed, removes, supplies, compares;
- bounded interpretation: on this benchmark family, across the current panel, under the reported ladder.

Avoid:

- generic benchmark manifesto language;
- leaderboard language;
- self-praise about being diagnostic;
- claims about model internals from output metrics alone;
- broad claims about causal reasoning in general;
- phrases that make calibration sound like a post-hoc repair.

The benchmark contribution should be visible through construction and organization, not repeated as a slogan.

## Paper-facing vocabulary

Use:

- `ACDB`
- `Benchmark Setup`
- `Scoring and Calibration`
- `Experimental Setup`
- `L1-L5`
- `current ladder`
- `accepted instance`
- `paired accepted-seed manifest`
- `structure-blind random floor`
- `linear-Gaussian SCM`
- `observe-intervene-submit protocol`

Do not use in paper prose:

- internal version labels for ladders;
- `L0`;
- `oracle`;
- `pc_obs`;
- random baseline as an evaluated method;
- correlation-only probe;
- provider/tool-call failure details.

Internal filenames may still contain older labels. Do not copy those labels into paper prose.

## Global-local drafting rule

For each section rewrite, use this order:

1. `PAPER_PRIORS.md` for global coherence;
2. the section prior for local structure and claims;
3. code files for behavioral truth;
4. `RESULTS.md` only for numeric claims;
5. current TeX only to identify stale material to delete.

If the global prior and a section prior conflict, the global prior controls paper identity and section role; the section prior controls local facts only if they are still consistent with the global role.

## Common failure modes

### Section 2 becomes too thin

Symptom:

- it defines SCM/CPDAG/protocol but does not make the benchmark instance family concrete.

Fix:

- foreground accepted instances, public/hidden fields, observational/interventional samples, filtering, and level variation.

### Section 3 becomes a metrics appendix

Symptom:

- it lists formulas without explaining why CPDAG and DAG layers are different truth objects.

Fix:

- organize around interpretation of submitted graphs, not metric inventory.

### Section 4 becomes harness documentation

Symptom:

- it over-describes tool schemas, prompts, or provider loop details.

Fix:

- organize around what each ablation holds fixed and what capability remains exposed.

### Section 5 becomes a model story

Symptom:

- paragraphs follow models rather than controlled comparisons.

Fix:

- organize by layered metrics, ablation contrasts, and graph-scaling behavior.

### Layout work destabilizes content work

Symptom:

- section prose, float placement, and page geometry are all being changed in the same pass;
- fixes in one area create new whitespace, float drift, or section-order confusion elsewhere.

Fix:

- freeze content before layout work;
- change one layout variable at a time;
- verify rendered placement, not just labels or page numbers;
- avoid global float controls during content drafting.

## Drafting prompt

Write the paper as a serious causal benchmark paper. Keep the controlled causal world in view. Make the instance family concrete before asking the reader to care about metrics or results. Use the CPDAG/DAG split as the organizing logic for scoring. Use ablations as controlled comparisons, not as a catalogue of policies. Interpret results only to the level supported by output metrics. Let the benchmark contribution emerge from the order and precision of the paper, not from repeated claims that the benchmark is diagnostic.
