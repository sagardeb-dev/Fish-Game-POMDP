# ACDB Section 6 priors: Related Work

This note is the drafting prior for the Related Work section.

Use it together with:

- `docs/notes/PAPER_PRIORS.md`
- `docs/notes/OUTLINE.md`
- `docs/notes/RELATED_WORK.md`
- Sections 1--5 of the live paper

If this note conflicts with the broader curation note, this note wins.

## Section objective

Related Work should answer one question only:

- what **LLM-facing** literature makes ACDB legible as a benchmark paper?

This section is **not** where we explain classical causal discovery, intervention design, or benchmark-calibration theory. Those belong in Sections 1--3.

Given the 9-page limit, this section should be:

- `2` real paragraphs
- plus at most `1` short closing sentence or very short paragraph for broad benchmark context
- around `0.5` pages

## Hard scope decision

Only include **direct related work**.

That means:

- LLM causal reasoning / graph-discovery methods
- LLM causal-reasoning / graph-discovery benchmarks

And only a very light closing reference to broader scientific-discovery benchmarks if the draft needs it.

Do **not** spend a full paragraph on:

- classical causal-discovery theory
- active intervention-design theory
- benchmark methodology
- broad agent benchmarks

## Target citation budget

Target:

- `7--9` papers total
- strong bias toward `2025` and early `2026`
- allow only `1--2` older anchors if they are uniquely load-bearing

If a paper does not help position ACDB against a direct LLM neighbor, cut it.

## Section shape

### Paragraph 1: LLMs for causal reasoning and graph discovery

Purpose:

- position ACDB against the closest LLM method papers

This paragraph should establish only that:

- LLMs have already been used to propose causal graphs, learn active causal strategies, and combine metadata with data-driven reasoning;
- ACDB contributes the benchmark, not another method.

### Paragraph 2: LLM causal-reasoning and graph-discovery benchmarks

Purpose:

- position ACDB against the closest benchmark competitors

This is the most important paragraph in the section.

It should communicate:

- static causal-reasoning benchmarks already exist;
- iterative graph-discovery benchmarks already exist;
- ACDB's niche is the anonymized, SCM-grounded, active-discovery setting with layered scoring.

### Optional closing sentence / mini-paragraph

If the section needs one broader outward-facing sentence, use it only to place ACDB inside the current scientific-discovery benchmark movement.

Do not let this become a benchmark catalog.

## Exact papers to cite

Below, each paper is listed by **exact title** so BibTeX can be added later without ambiguity. Placeholder keys are suggestions only.

## Paragraph 1 papers: methods

Keep only these:

1. **Passive Learning of Active Causal Strategies in Agents and Language Models**
   - placeholder key: `lampinen2023passive`
   - role: one older conceptual anchor that directly matches ACDB's active-intervention question

2. **Efficient Causal Graph Discovery Using Large Language Models**
   - placeholder key: `jiralerspong2024bfs`
   - role: closest full-graph LLM method paper

3. **Causal Modelling Agents: Causal Graph Discovery through Synergising Metadata- and Data-driven Reasoning**
   - placeholder key: `abdulaal2024cma`
   - role: closest hybrid data + metadata approach

Optional if one more is needed:

4. **Causal Inference Using LLM-Guided Discovery**
   - placeholder key: `vashishtha2023causalorder`
   - role: causal-order alternative to full graph output

Papers explicitly not needed here:

- **Causal Reasoning and Large Language Models: Opening a New Frontier for Causality**
- **Causal Parrots: Large Language Models May Talk Causality But Are Not Causal**
- **Can Large Language Models Build Causal Graphs?**

Reason:

- they are useful background, but not sharp enough for this page budget once we have the more direct method papers above.

## Paragraph 2 papers: benchmarks

Must include:

1. **Can Large Language Models Infer Causation from Correlation?**
   - placeholder key: `jin2024corr2cause`
   - role: one static causal-reasoning benchmark anchor

2. **Auto-Bench: An Automated Benchmark for Scientific Discovery in LLMs**
   - placeholder key: `chen2025autobench`
   - role: closest benchmark competitor; non-negotiable

3. **IGDA: Interactive Graph Discovery through Large Language Model Agents**
   - placeholder key: `havrilla2025igda`
   - role: closest complementary benchmark/method neighbor

4. **Revisiting Causal Reasoning in Language Models through Controlled Synthetic Worlds**
   - placeholder key: `albertsworld2026`
   - role: recent controlled synthetic-world benchmark reference

5. **Realizing LLMs' Causal Potential Requires Science-Grounded, Novel Benchmarks**
   - placeholder key: `realizingllmcausal2025`
   - role: recent benchmark-position paper aligned with ACDB's anti-retrieval design

Optional if the paragraph needs one extra older benchmark anchor:

6. **CLadder: Assessing Causal Reasoning in Language Models**
   - placeholder key: `jin2023cladder`
   - role: older canonical benchmark anchor

What this paragraph should say about each:

- `Corr2Cause`: static correlational-to-causal benchmark
- `Auto-Bench`: iterative causal-graph-discovery benchmark; closest competitor
- `IGDA`: complementary graph discovery with semantic metadata and edge experiments rather than anonymized data matrices
- `Albert's World`: controlled synthetic benchmark for causal reasoning, closer to ACDB's anti-retrieval design than to its active graph-discovery task
- `Realizing LLMs' Causal Potential...`: explicit external statement that causal benchmarks need stronger contamination control and science-grounded design

## Optional closing context

If needed, use only one of these:

1. **DiscoveryWorld: A Virtual Environment for Developing and Evaluating Automated Scientific Discovery Agents**
   - placeholder key: `jansen2024discoveryworld`

2. **ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery**
   - placeholder key: `chen2024scienceagentbench`

Use at most one sentence. No list.

## Papers explicitly excluded from Related Work

1. **Reducing Belief Deviation in Reinforcement Learning for Active Reasoning**
   - placeholder key: `t3_2026`
   - why excluded: active-reasoning RL method paper, not a direct causal-discovery benchmark neighbor

2. **CausalEvolve: Towards Open-Ended Discovery with Causal Scratchpad**
   - placeholder key: `causalevolve2026`
   - why excluded: open-ended scientific-discovery system paper, not a direct graph-discovery benchmark neighbor

3. broad agent benchmarks:
   - `SWE-bench`
   - `WebArena`
   - `OSWorld`
   - `GAIA`
   - why excluded: too broad and too far from ACDB's task

4. theory-heavy causal-discovery papers:
   - `Spirtes`, `Verma & Pearl`, `Andersson`, `Eberhardt`, `Hauser`, `NOTEARS`, `ABCD`, etc.
   - why excluded here: these should already do their work in Sections 1--3

5. benchmark-methodology papers:
   - `HELM`
   - `Stronger Random Baselines for In-Context Learning`
   - contamination/leakage benchmark papers
   - why excluded here: good citations, wrong section for this page budget

## Writing logic

The section should read like:

1. recent LLM work has tried to infer causal structure or use interventions directly;
2. recent LLM benchmarks have tested causal reasoning and interactive graph discovery;
3. ACDB contributes a more controlled active-discovery benchmark with anonymized variables and layered graph scoring.

Not like:

- a survey of all causal-reasoning literature
- a theory recap
- a benchmark zoo

## Priming prompt for later drafting

When writing the section:

- write only two real paragraphs
- use the method paragraph to establish direct LLM method neighbors
- use the benchmark paragraph to do the real positioning work
- cite `Auto-Bench` and `IGDA` explicitly
- prefer `2025/2026` papers wherever possible
- use `Albert's World` only as a controlled synthetic-world benchmark reference, not as a central competitor
- use a broad scientific-benchmark citation only as a final outward-facing sentence if needed
- do not cite `T3` or `CausalEvolve`
- do not cite broad agent benchmarks just to make the section look more complete

## Verifiable success criteria

This prior is good enough for drafting only if:

1. the eventual section can be written in `2` paragraphs plus an optional closing sentence;
2. `Auto-Bench` and `IGDA` are explicitly positioned;
3. the section is clearly LLM-first;
4. the reader can tell what ACDB adds without the section repeating the introduction;
5. the total citation count can plausibly stay under `10`.
