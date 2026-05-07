# ACDB paper claims

This file is for paper-facing claims, not raw observations. For NeurIPS E&D, the claims need to be about the evaluation instrument first and about model behavior second. The paper should not read as "we tested some LLMs and here are the scores." It should read as "we built a benchmark whose scoring contract and calibration support specific evaluative claims, and these experiments demonstrate what that instrument reveals."

The current evidence base is `RESULTS.md`, which presently includes `GPT-5.5`, `GPT-5.4-mini`, `Sonnet-4.6`, `Haiku-4.5`, and `Gemini-3-Flash` on levels `L1-L5`, with methods `pc_greedy`, `pc_cpdag_llm`, `llm_raw`, `llm_stats`, and `llm_stats_cpdag_greedy`.

## Core paper claims

### Claim 1 - Layered scoring makes the benchmark diagnostic, not just evaluative

ACDB's multi-layer scoring contract separates adjacency recovery, observationally identifiable orientation, and full DAG recovery instead of collapsing them into one graph score. In particular:

- `skeleton F1` measures adjacency recovery.
- `compelled-edge F1` measures orientation recovery only where the CPDAG makes direction identifiable from observational structure.
- `directed F1` and `SHD` measure full submitted-DAG quality.

This matters because the same method can look strong on one layer and weak on another. The benchmark therefore supports failure-mode attribution rather than only aggregate ranking.

The clearest example is `compelled-edge F1`: it can show an observational-orientation advantage that is not visible from `directed F1` or `SHD` alone. That is exactly the kind of distinction a single headline metric would hide.

Why this is E&D-relevant:
- The contribution is an evaluation design with explicit theoretical referents for each metric.
- The benchmark supports more precise claims about where a method fails, which is the central evaluative role of the artifact.

### Claim 2 - Benchmark calibration materially affects what conclusions are valid

ACDB's calibration probes show that naive causal-discovery ladders can inflate apparent performance through structural artifacts rather than reasoning quality. In particular, random-floor analysis, density-aware ladder design, intervention-budget semantics, and paired-seed comparisons make these artifacts visible and correctable.

This is a benchmark claim, not only a methodology footnote:

- random-guess performance is non-trivial on some graph families;
- naive difficulty ladders can overstate capability when graph density and structure create accidental score inflation;
- difficulty scaling can be confounded if graph complexity is not controlled carefully;
- intervention budgets must measure reasoning quality rather than exact optimal-set recovery;
- paired seeds are necessary to make cross-method comparisons legible.

Why this is E&D-relevant:
- The paper is not only introducing a benchmark, but also showing how benchmark design assumptions shape scientific conclusions.
- This directly matches the E&D track emphasis on what is measured, under what assumptions, and how results should be interpreted.

### Claim 3 - Across this benchmark family, model-family variation is larger than policy variation within a fixed model

On the current `L1-L5` results, changing the model family produces much larger swings in `directed F1` and `SHD` than changing the prompt/policy variant within a strong model. This establishes a capability spectrum across models rather than a narrow story about one prompt template winning.

Paper-safe phrasing:
- across model families, active causal-discovery performance spans a wide range;
- within a fixed strong model, `llm_raw`, `llm_stats`, and `pc_cpdag_llm` are closer to one another than weak and strong models are overall.

What not to claim:
- do not claim that "method does not matter";
- do not claim a universal law about all LLMs.

What the claim supports:
- ACDB is sensitive enough to distinguish model capability levels, not only benchmark-policy variants.

### Claim 4 - LLM-only policies degrade faster with graph complexity than the classical active baseline

On the `v1` ladder, fully LLM-based policies degrade faster from `L1` to `L5` than `pc_greedy`. The benchmark therefore captures an asymmetric scaling effect:

- the classical baseline weakens gradually with graph complexity;
- LLM-only methods weaken more sharply;
- weaker models collapse much earlier than frontier models.

Paper-safe phrasing:
- "LLM-based active-discovery policies are less robust to graph scaling on this benchmark family than the classical `PC + greedy orientation` baseline."

What not to overclaim:
- do not claim this proves a general theorem about LLM scaling;
- do not claim the mechanism is known with certainty.

Interpretation:
- the asymmetric degradation is consistent with LLM policies relying more heavily on local pattern heuristics that weaken as graph size, ambiguity, and edge-orientation burden grow.

### Claim 5 - Structural priors help substantially, but they do not remove the orientation subproblem

Supplying a strong observational structure prior via `pc_cpdag_llm` substantially improves active causal discovery, especially for weaker models. This establishes the magnitude of the prior effect and shows that observational structure recovery is a major bottleneck in the full task.

However, the same ablation does not solve the whole problem. Even when the observational graph is supplied, the model must still:

- choose informative interventions;
- interpret interventional outcomes;
- orient edges correctly under the budget.

This makes `pc_cpdag_llm` a load-bearing ablation:
- if a model improves strongly under this condition, its bottleneck was largely observational;
- if it still trails `pc_greedy`, the residual gap is no longer about missing skeleton structure.

### Claim 6 - Interpreting interventional evidence is a distinct capability, separable from skeleton recovery

The comparison between `pc_cpdag_llm` and `pc_greedy` isolates a narrow subtask: convert a strong observational graph plus intervention outcomes into a final oriented graph. Because both methods start from the same PC-derived observational structure, the remaining gap is not a skeleton-recovery gap.

This is one of the cleanest claims the benchmark enables:

- `pc_cpdag_llm` tests model-driven intervention choice and interventional-evidence interpretation;
- `pc_greedy` provides a non-LLM active reference on the same starting structure.

Current results show that only the strongest model class is near parity with the heuristic on this subtask, while smaller or weaker models remain clearly below it.

Why this matters:
- it shows that interventional reasoning should be treated as a specific capability slice, not bundled into a generic "causal reasoning" label.

### Claim 7 - Statistical tool access alone does not reliably improve active causal discovery

Across the current model set, giving the agent access to `correlation`, `partial_correlation`, and `independence_test` tools does not produce consistent directed-graph gains over raw-data access. In many settings, the tool-using policy is worse.

The careful claim is not "tools never help." The supported claim is:

- tool availability by itself is insufficient;
- the bottleneck is the agent's ability to use the tools effectively inside a sequential decision loop;
- the tools often shift behavior toward repeated testing, analysis loops, and delayed commitment rather than decisive graph construction;
- ACDB exposes this distinction because the raw and stats policies share the same task but differ in evidence interface.

This is benchmark-relevant because it converts a vague tool-use question into a controlled evaluation comparison.

## Secondary claims

These are useful, but should stay below the core claims unless the final paper needs them.

### Secondary claim A - Frontier models can be competitive with the classical active baseline on this benchmark family

The strongest current model is competitive with `pc_greedy` on average `L1-L5` and near parity on the `pc_cpdag_llm` ablation. This is an important empirical finding, but it is not the benchmark's main contribution and should not become the title-level claim.

### Secondary claim B - Observational performance alone understates active capability

A method that looks weak when forced to stop at the observational stage can improve substantially once interventions are allowed. This is one reason ACDB should report observational and active conditions separately.

### Secondary claim C - The benchmark supports negative results that remain interpretable

When a model performs poorly, the layered scores and hybrid ablations make the failure diagnosable:
- poor skeleton recovery;
- weak compelled-edge orientation;
- weak interventional interpretation;
- poor use of statistical tools.

This is valuable for E&D because negative results are only useful when the benchmark makes them interpretable.

## Claims to avoid

- "ACDB proves that LLMs cannot do causal discovery."
- "Tool use does not work for LLMs."
- "General reasoning strength transfers directly to interventional reasoning."
- "The benchmark shows human-level causal reasoning."
- "One policy is universally best."
- "Directed F1 alone is the right headline metric."

These are either broader than the evidence supports or inconsistent with the benchmark-first framing.

## Paper framing notes

When drafting the paper, keep the contribution hierarchy explicit:

1. Primary contribution: a theory-grounded evaluation instrument for active causal discovery.
2. Secondary contribution: calibrated benchmark methodology and ablations that localize failure.
3. Empirical demonstration: current model results show what the instrument reveals.

The benchmark claim should survive even if future model rankings change.
