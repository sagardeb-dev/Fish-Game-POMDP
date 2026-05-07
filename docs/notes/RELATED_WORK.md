# ACDB related work curation note

This note is not the Related Work section and not the section prior yet. It is the curation pass that decides what belongs in the paper at all.

Use it together with:

- `docs/notes/PAPER_PRIORS.md`
- `docs/notes/OUTLINE.md`
- Sections 1--5 of the current paper
- the user-provided resource pack

Do not use stale `research/sections/07_related_work.tex` as authority.

## Section objective

The Related Work section only needs to answer two reviewer questions:

1. What research conversations does ACDB actually sit inside?
2. Why are ACDB's benchmark design choices and comparison set reasonable?

Everything else is expendable.

Given the 9-page limit, the section should stay around `0.5--0.75` pages and should read as `3--4` compact paragraphs, not as a literature survey.

## Re-grounding in the paper

ACDB is:

- a benchmark for **active causal discovery**
- over a **controlled linear-Gaussian SCM family**
- with a fixed **observe -> intervene -> submit** runtime
- a **layered CPDAG/DAG scoring contract**
- paired-seed comparisons across **classical** and **LLM-based** policies
- explicit **calibration** against a structure-blind random floor

That means the most relevant related work is not "all causal reasoning with LLMs."

The closest conversations are:

1. causal discovery and intervention design
2. LLM-based causal reasoning and graph construction
3. causal/scientific-discovery benchmarks for LLMs and agents
4. benchmark validity and calibration

## What from the user-passed papers actually fits

Three external papers were passed as context:

1. `T3 — Reducing Belief Deviation in Reinforcement Learning for Active Reasoning`
2. `CausalEvolve: Towards Open-Ended Discovery with Causal Scratchpad`
3. `Revisiting Causal Reasoning in Language Models through Controlled Synthetic Worlds` ("Albert's World")

Only one is a plausible Related Work citation for ACDB:

- `Revisiting Causal Reasoning in Language Models through Controlled Synthetic Worlds`

Why:

- it is a **controlled synthetic causal-world benchmark**
- it is explicitly about **causal reasoning in LLMs**
- it is methodologically close to ACDB's use of a controlled world to block uncontrolled retrieval shortcuts

Why the other two should **not** anchor the section:

- `T3` is about RL training stability for active reasoning agents, not causal discovery or benchmark design
- `CausalEvolve` is about open-ended scientific discovery via causal scratchpads, not graph discovery, intervention design, or benchmark calibration

They are useful style and framing references, but not load-bearing Related Work for this paper.

## Recommended section shape

Use four compact paragraphs:

### A. Causal discovery and active intervention design

Purpose:

- establish the classical CD lineage ACDB inherits
- justify why `PC` / `PC + greedy` are meaningful baselines
- acknowledge that the field extends beyond classical CD

### B. LLMs for causal reasoning and graph construction

Purpose:

- position ACDB against LLM work that tries to infer causal graphs or use causal priors
- establish that active/interventional reasoning in LLMs is a known question, not an invented one

### C. Causal-reasoning and scientific-discovery benchmarks

Purpose:

- position ACDB against the closest benchmark competitors
- this is the most important paragraph

### D. Benchmark validity and calibration

Purpose:

- justify ACDB's calibration and anonymization choices as benchmark-methodology decisions
- keep this short; it should not become a second Section 3

## Curated citation set

This is the recommended paper pool for drafting. It is intentionally smaller than the resource pack.

### A. Causal discovery and active intervention design

Keep:

- `spirtes2000causation`
- `verma1990equivalence`
- `andersson1997characterization`
- `eberhardt2005interventions`
- `hauser2012gies`

Add:

- `tong2001active` or `murphy2001active`
- `agrawal2019abcd`

Modern-context acknowledgment:

- `zheng2018notears`
- `brouillard2020dcdi` or `lorch2022avici`

Why this set:

- `Spirtes/Verma/Andersson/Eberhardt/Hauser` are the theory and intervention-design spine that matches Section 2 and Section 3 directly.
- `Tong & Koller` or `Murphy` establishes that active intervention selection is an existing line, not something ACDB invented.
- `ABCD-Strategy` is the strongest modern budgeted-design citation for ACDB's budgeted active setting.
- `NOTEARS` plus one differentiable/amortized citation is enough to show awareness that the field moved beyond PC/GIES.

Do not expand this paragraph into a method catalog.

### B. LLMs for causal reasoning and graph construction

Keep:

- `kiciman2023llmcausal`
- `zecevic2023causalparrots`

Add:

- `long2023llmgraphs`
- `lampinen2023passive`
- `jiralerspong2024bfs`
- `vashishtha2023causalorder`
- `abdulaal2024cma`

Optional:

- `ban2023causal`
- `wu2025llmcannot`

Why this set:

- `Kiciman` and `Zecevic` anchor the high-level "LLMs and causality" conversation.
- `Long`, `Jiralerspong`, `Vashishtha`, and `Abdulaal` are the direct method neighbors for graph discovery, causal order, or hybrid data+metadata reasoning.
- `Lampinen` is the key conceptual bridge for active/interventional strategies in LLMs and should carry more weight than a generic survey.
- `Wu 2025` is useful if the final paper wants one explicit skeptical citation, but it is not mandatory for a tight related-work section.

### C. Causal-reasoning and scientific-discovery benchmarks

Must include:

- `jin2023cladder`
- `jin2024corr2cause`
- `chen2025autobench`
- `havrilla2025igda`

Strongly recommended:

- `zhou2024causalbenchllm`
- `jansen2024discoveryworld`
- `majumder2024discoverybench`
- `chen2024scienceagentbench`

Optional:

- `huang2024mlagentbench`
- `yamin2024review`
- `albertsworld2026` (placeholder key; add real BibTeX key later if used)

Why this set:

- `Auto-Bench` is the closest benchmark competitor and is non-negotiable.
- `IGDA` is not the same setup, but it is the clearest methodological complement: semantic-metadata-driven graph discovery under experiments rather than anonymized data-driven discovery.
- `CLadder`, `Corr2Cause`, and `CausalBench-LLM` represent the static causal-reasoning benchmark line that ACDB moves beyond.
- `DiscoveryWorld`, `DiscoveryBench`, and `ScienceAgentBench` place ACDB inside the broader scientific-discovery-agent benchmark conversation without confusing it with general agent benchmarks like SWE-bench or WebArena.

What to say about `Auto-Bench`:

- iterative discovery benchmark
- intervention loop
- causal-graph framing
- but no layered scoring, no calibration, no paired-seed classical baselines, and no anonymization against semantic retrieval

What to say about `IGDA`:

- complementary problem: semantic metadata plus edge experiments, no observational data
- ACDB solves the opposite half: anonymized observational/interventional samples with blocked semantic priors

What to say about `Albert's World` if included:

- controlled synthetic world for causal reasoning
- closer to ACDB's anti-retrieval methodology than to its active graph-discovery task
- belongs in this paragraph only as a secondary recent benchmark reference, not as a core competitor

### D. Benchmark validity and calibration

Keep this short.

Add:

- `yauney2024stronger`
- `liang2023helm`

Optional:

- `zhou2023cheater`
- `balloccu2024leak`

Why this set:

- `Yauney & Mimno` is the cleanest precedent for stronger-than-naive random baselines.
- `HELM` is enough to justify multi-metric evaluation without opening a long benchmark-methodology digression.
- contamination citations are optional; ACDB's anonymization can be defended without spending much page budget here.

## Recommended includes vs optional vs exclude

### Must include

- `spirtes2000causation`
- `verma1990equivalence`
- `andersson1997characterization`
- `eberhardt2005interventions`
- `hauser2012gies`
- `tong2001active` or `murphy2001active`
- `agrawal2019abcd`
- `zheng2018notears`
- `kiciman2023llmcausal`
- `zecevic2023causalparrots`
- `long2023llmgraphs`
- `lampinen2023passive`
- `jiralerspong2024bfs`
- `abdulaal2024cma`
- `jin2023cladder`
- `jin2024corr2cause`
- `chen2025autobench`
- `havrilla2025igda`
- `jansen2024discoveryworld`
- `yauney2024stronger`

### Strongly recommended if space holds

- `zhou2024causalbenchllm`
- `majumder2024discoverybench`
- `chen2024scienceagentbench`
- `liang2023helm`
- `vashishtha2023causalorder`
- `brouillard2020dcdi` or `lorch2022avici`

### Optional

- `yamin2024review`
- `wu2025llmcannot`
- `huang2024mlagentbench`
- `albertsworld2026`
- contamination-focused benchmark papers

### Exclude from main related work

- `T3`
- `CausalEvolve`
- general agent benchmarks like `SWE-bench`, `WebArena`, `OSWorld`, `GAIA`

Reason:

- they are too far from ACDB's actual contribution once the paper is only `0.5--0.75` pages.
- `DiscoveryWorld`, `DiscoveryBench`, and `ScienceAgentBench` are much closer scientific-agent comparators.

## Draft positioning logic

The section should not read like "here is everything vaguely nearby."

It should read like:

1. active causal discovery already has a classical and budgeted-design lineage;
2. LLMs have already been used for causal reasoning, causal order, and graph proposal;
3. existing causal-reasoning benchmarks are mostly static, semantic, or binary-success oriented, while the closest interactive benchmark is `Auto-Bench`;
4. ACDB contributes a calibrated, SCM-grounded, anonymized active-discovery benchmark with layered graph scoring and paired-seed classical/LLM comparison.

That is the whole job.

## Proposed drafting emphasis by paragraph

### Paragraph 1: causal discovery and active design

Target references:

- `spirtes2000causation`
- `verma1990equivalence`
- `hauser2012gies`
- `tong2001active`
- `agrawal2019abcd`
- `zheng2018notears`

### Paragraph 2: LLMs for causal reasoning

Target references:

- `kiciman2023llmcausal`
- `zecevic2023causalparrots`
- `long2023llmgraphs`
- `lampinen2023passive`
- `jiralerspong2024bfs`
- `abdulaal2024cma`

### Paragraph 3: benchmarks

Target references:

- `jin2023cladder`
- `jin2024corr2cause`
- `zhou2024causalbenchllm`
- `chen2025autobench`
- `havrilla2025igda`
- `jansen2024discoveryworld`
- `majumder2024discoverybench`
- `chen2024scienceagentbench`

### Paragraph 4: methodology / calibration

Target references:

- `yauney2024stronger`
- `liang2023helm`

Optional add:

- `albertsworld2026`

only if the draft needs one extra sentence connecting ACDB's anonymized synthetic world to recent controlled-world causal benchmarks.

## Bibliography impact

Current `research/references.bib` contains only a small subset of what this section needs.

Already present:

- `spirtes2000causation`
- `chickering2002ges`
- `verma1990equivalence`
- `andersson1997characterization`
- `eberhardt2005interventions`
- `hauser2012gies`
- `kiciman2023llmcausal`
- `zecevic2023causalparrots`
- `jin2023cladder`

Need to add before drafting:

- the rest of the selected set above, especially `chen2025autobench`, `havrilla2025igda`, `jiralerspong2024bfs`, `abdulaal2024cma`, `lampinen2023passive`, `agrawal2019abcd`, `jansen2024discoveryworld`, `yauney2024stronger`, and the selected science-agent benchmarks.

## Writing constraints for the later prior

When the actual prior is written, enforce these:

- do not turn Related Work into a second introduction;
- do not spend half the section on general agent benchmarks;
- do not cite `T3` or `CausalEvolve` unless a concrete sentence truly needs them;
- cite `Auto-Bench` directly and explicitly;
- treat `IGDA` as a complementary neighbor, not a direct apples-to-apples baseline;
- use `Albert's World` only if the draft needs a recent controlled synthetic-world benchmark citation;
- keep the section argument-driven, not exhaustive.
