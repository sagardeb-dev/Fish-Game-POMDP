# NeurIPS 2026 Evaluations & Datasets (E&D) Track — Call for Papers

*Source: https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets (saved 2026-04-24)*

Formerly the "Datasets & Benchmarks" track, renamed "Evaluations & Datasets" for 2026 with broadened scope: evaluation itself is now an object of scientific study. Datasets remain central, but submissions must articulate the **evaluative role** the contribution plays — what claims it supports, under what assumptions, with what limitations.

## Key dates

| Event | Date |
|---|---|
| Submission portal opens | April 15, 2026 |
| Abstract submission deadline | **May 4, 2026 (AoE)** |
| Full paper + supplementary deadline | **May 6, 2026 (AoE)** |
| Author notification | September 24, 2026 (AoE) |

Authors must have an OpenReview profile at abstract submission.

## Scope (what qualifies)

Contributions that advance the science of AI evaluation. Non-exhaustive:

- Analyze strengths/limitations/failure modes of existing benchmarks or evaluation practices
- Study benchmark saturation or overfitting and their impact on scientific conclusions
- Compare evaluation designs, showing how different assumptions yield different conclusions
- Rigorous reproduction, auditing, stress-testing of prior evaluations
- Documentation methodologies (Data Cards, Model Cards, evaluation cards)
- New evaluation protocols, practices, methodologies
- Human- or interaction-centered evaluations (user studies, red-teaming)
- Datasets with explicit scope/assumptions/limitations and evaluative role
- Tools/analyses/frameworks for constructing or interpreting evaluative claims
- Negative results, critical analyses, use-case-inspired evaluations

Historically welcomed, still in scope:

- New datasets and dataset collections
- **Data generators and reinforcement learning environments** ← fits this project
- Data-centric AI methods and tools
- Advanced data collection and curation practices
- Audits of existing datasets
- Benchmarks on new/existing datasets, benchmarking tools/methodologies
- Systematic analyses of systems on novel datasets
- In-depth analyses of ML challenges/competitions
- Competition papers from prior NeurIPS competitions

**Important:** submissions need not introduce a new model or outperform prior work. What matters is how the contribution meaningfully changes, strengthens, or enables evaluation of AI/ML systems.

## Formatting

- Use `\usepackage[eandd]{neurips_2026}` (E&D track option)
- Only `neurips_2026.sty` is supported; tweaking the style file is grounds for desk reject
- For formatting, code of conduct, ethics review: follow NeurIPS 2026 Main Track CfP

## Review policy

- **Default: double-blind**
- Dataset-centered submissions may opt into single-blind (use `\usepackage[eandd, nonanonymous]{neurips_2026}`); practical concession given anonymization difficulty
- Benchmarks that can't be fully anonymized (build on existing codebases) require best-effort anonymization

## Dataset and code submission (required for executable artifacts)

**Dataset hosting.** New datasets must be long-term hosted on a dedicated ML hosting site: Dataverse, Kaggle, Hugging Face, or OpenML (or bespoke if required). Datasets >4 GB require a small inspectable sample.

**Code.** Contribution-dependent policy:

- **Required at submission** when the primary contribution is a reusable executable artifact (benchmark suite, evaluation environment, data generator, software tool) whose functionality must be inspected to evaluate claims
- **Not mandatory** for analytical/empirical/conceptual/methodological contributions without such artifacts, provided the paper has enough detail for review

Code hosted on GitHub, Bitbucket, etc., anonymized per review-policy requirements, executable, documented. Non-compliance justifies desk rejection.

**Metadata (Croissant).** Authors of datasets must include a Croissant metadata file (core + RAI fields) in the OpenReview submission. Auto-generated on Kaggle/OpenML/Hugging Face/Dataverse (core only — add RAI fields manually). Self-hosted datasets: generate Croissant manually. Validate via NeurIPS-provided online tool.

**Camera-ready.** All code and datasets documented and publicly available by camera-ready.

## Implications for this project

This RL-environment repo introduces an active causal discovery benchmark with a session API, ladder, scoring contract — that is **a reinforcement learning environment + benchmark**, squarely in "data generators and RL environments" and "benchmarks on new datasets." Code release is therefore **required at submission** because the benchmark *is* the contribution. Review mode will default to double-blind; the code submitted will need to be anonymized.

## Contact

E&D Track chairs: evaluationsdatasets@neurips.cc
