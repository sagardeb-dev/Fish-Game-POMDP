# ACDB paper abstract

## Live metadata

- Track: NeurIPS 2026 Evaluations and Datasets
- Live title: `Active Causal Discovery as a Diagnostic Benchmark for LLM Agents`
- Live abstract target: `research/main.tex`

## Live abstract (current paper copy)

Active causal discovery is a sequential task in which observations typically identify only a Markov-equivalence class, while interventions are needed to orient the remaining structure. We introduce the Active Causal Discovery Benchmark (ACDB), a controlled family of linear-Gaussian SCM instances with anonymized variables, a fixed observe-intervene-submit runtime, and a layered scoring contract that separates adjacency recovery, observationally identifiable orientation, and full directed-graph recovery. ACDB further calibrates evaluation with a random-floor-aware difficulty ladder and paired accepted-seed comparisons, so differences in directed performance are interpretable across methods and graph scales. Across five language models and five active policies, the benchmark reveals large spread across models, substantial gains from supplying observational structure priors, and a persistent gap between end-to-end LLM policies and a classical active baseline as graph complexity increases. Statistical tool access alone does not reliably close that gap. Taken together, these results position ACDB as a benchmark for measuring observational, interventional, and end-to-end failures separately rather than compressing active causal discovery into a single graph score.

## Abstract contract

- one paragraph only;
- benchmark-first, not model-ranking-first;
- artifact first, calibration second, empirical reveal third;
- no stale ladder/version language;
- no numbers unless a future final pass decides they are load-bearing;
- no claims about model internals or general causal reasoning beyond this benchmark family.

## Synchronization rule

If the abstract changes, keep these files aligned:

1. `research/main.tex`
2. `docs/notes/ABSTRACT.md`
3. `docs/notes/ABSTRACT_INTRO_PRIORS.md`
4. `docs/notes/OUTLINE.md` if the one-line thesis or title changes
