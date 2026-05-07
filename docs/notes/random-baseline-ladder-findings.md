# Random Baseline And Ladder Design Findings

Created: 2026-04-23

This note captures the benchmark-design problems found while adding and running the corrected random DAG baseline, plus the paper-relevant implications. It is meant as future context before changing `run_ladder.py` or launching more LLM runs.

## Why This Matters

The benchmark was intended to measure causal discovery, but the original ladder made it hard to tell whether a model was discovering structure or benefiting from a guessable graph distribution. A blind random baseline exposed that issue.

The key lesson is that a causal-discovery ladder should be defined by mathematical difficulty axes, not by hand-picked `(d, k, n_obs, n_int, noise)` values. The two main axes are:

1. Graph combinatorics: how sparse/dense the DAG is.
2. Statistical power: how much data is available to detect the weakest relevant edge.

Everything else should be derived from those axes.

## Problems We Faced

### 1. The original ladder was partly guessable

The original ladder had levels where the true graph was dense enough that a blind random DAG achieved non-trivial directed F1. This weakens claims that an LLM is doing causal discovery unless it beats random by a meaningful margin on the same manifest.

Original random-uniform baseline, using the original GPT ladder manifest:

| Level | Random directed F1 | Random SHD |
|---:|---:|---:|
| 0 | 0.282 | 4.22 |
| 1 | 0.238 | 6.44 |
| 2 | 0.241 | 6.47 |
| 3 | 0.205 | 12.64 |
| 4 | 0.240 | 6.54 |
| 5 | 0.207 | 12.73 |
| overall | 0.236 | 8.18 |

Rule of thumb from this probe:

- Random directed F1 below 0.20 is preferable for real benchmark levels.
- Random directed F1 above 0.25 is too guessable unless the level is explicitly tutorial-only.

### 2. Random baseline must not use `k`

The corrected `random_uniform` baseline samples using only public metadata `d`. It does not use the true edge count `k`.

Current random generation:

1. Sample a random node order.
2. Enumerate the `M = d(d-1)/2` forward edges under that order.
3. Sample edge count uniformly from `0..M`.
4. Select that many forward edges uniformly.

This makes the baseline blind to graph density. That is the right anti-leakage behavior. Any random baseline that forces the true `k` is too informed and should not be used as the monkey baseline.

### 3. Density probe is not directly comparable to old GPT/Sonnet runs

The density probe uses a different level catalog than the original GPT/Sonnet ladder. It is valid for calibrating a future ladder, but not for direct LLM comparison.

For GPT/Sonnet comparison, use:

- `traces/ladder/random_uniform_baseline/results_random_dag_summary.csv`
- `traces/ladder/full_ladder_toolcall_run1/results_summary.csv`
- `traces/ladder/sonnet46_full_ladder_run1/results_summary.csv`

Those runs share the original level definitions and seed map.

### 4. Context window is not the current bottleneck

With the current prompt format, input size is controlled mainly by:

```text
observational payload ~= n_obs * d
intervention history ~= budget * n_int * d
```

If intervention budget scales with `d`, active raw mode grows roughly as `O(d^2)`. If intervention budget is capped near 3 or 4, it grows roughly linearly in `d`.

At `d <= 12`, `n_obs <= 50`, `n_int <= 25`, and budget near 3 or 4, context overflow is not a real risk. The practical bottleneck is reasoning quality: the model must make decisions over `M = d(d-1)/2` possible adjacencies.

For example:

| d | Max possible unordered adjacencies `M` |
|---:|---:|
| 10 | 45 |
| 12 | 66 |
| 16 | 120 |
| 20 | 190 |

For v1 paper experiments, `d=10` is a reasonable largest real level and `d=12` is a stress level. `d >= 16` is more appropriate as future work unless the prompt and evaluation design change.

## Empirical Findings

### Corrected density probe

The density probe was run with:

```powershell
python run_random_dag_baseline.py --level-set density_probe --out-dir traces\ladder\random_density_probe --seeds-per-level 8 --samples-per-instance 100 --baseline-seed 12345
```

Results:

| Level | Purpose | d | k | Density | Random directed F1 | Random SHD |
|---:|---|---:|---:|---:|---:|---:|
| 0 | tutorial only | 4 | 3 | 0.500 | 0.211 | 3.76 |
| 1 | small core | 6 | 6 | 0.400 | 0.195 | 8.88 |
| 2 | medium core | 8 | 9 | 0.321 | 0.170 | 16.53 |
| 3 | real scale | 10 | 12 | 0.267 | 0.156 | 25.48 |
| 4 | structural hard | 12 | 14 | 0.212 | 0.131 | 36.27 |
| 5 | noisy hard | 10 | 12 | 0.267 | 0.151 | 25.41 |
| overall | - | - | - | - | 0.169 | 19.39 |

Interpretation:

- Levels 1 through 5 pass the preferred random directed-F1 target of below 0.20.
- Level 0 is slightly above 0.20, but it is tutorial-only and below 0.25.
- Lower density successfully reduces blind random directed F1.
- SHD rises strongly with graph size, as expected, because there are more possible mistakes.

### GPT and Sonnet compared with original random baseline

These comparisons are against the original random baseline, not the density probe.

| System | Panel / method | n success | directed F1 | delta vs random | SHD | delta SHD vs random | skeleton F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Random uniform | blind baseline | 4800 | 0.236 | - | 8.18 | - | 0.466 |
| GPT-5.4 | observational / llm_raw_obs | 55 | 0.204 | -0.032 | 10.62 | +2.44 | 0.646 |
| GPT-5.4 | observational / llm_stats_obs | 55 | 0.140 | -0.095 | 6.98 | -1.19 | 0.346 |
| GPT-5.4 | active / llm_raw | 54 | 0.226 | -0.010 | 9.78 | +1.60 | 0.571 |
| GPT-5.4 | active / llm_stats | 56 | 0.188 | -0.048 | 6.89 | -1.28 | 0.378 |
| Claude Sonnet 4.6 | observational / llm_raw_obs | 48 | 0.361 | +0.126 | 7.46 | -0.72 | 0.577 |
| Claude Sonnet 4.6 | observational / llm_stats_obs | 48 | 0.150 | -0.085 | 9.92 | +1.74 | 0.607 |
| Claude Sonnet 4.6 | active / llm_raw | 48 | 0.317 | +0.081 | 7.25 | -0.93 | 0.563 |
| Claude Sonnet 4.6 | active / llm_stats | 47 | 0.280 | +0.044 | 8.06 | -0.11 | 0.627 |

Interpretation:

- Sonnet raw modes are meaningfully above blind random on directed F1 and often better on SHD.
- Sonnet stats mode is mixed. Active stats modestly beats random on directed F1; observational stats does not.
- GPT is close to random on directed F1 and often below it. GPT stats sometimes improves SHD, which means it may avoid some gross structural errors, but it does not show strong directed causal-discovery signal.
- Small deltas should be treated carefully because each LLM method has about 8 successful instances per level, while random has 800 samples per level.

Paper-facing phrasing:

> The random DAG baseline revealed that some earlier ladder settings were too guessable. After correcting the random baseline to use only public graph size and not true edge count, we found that benchmark density must be controlled explicitly. On the original ladder, GPT-level directed F1 was near the blind-random baseline, while Sonnet raw modes showed a clearer margin over random. This supports reporting random-normalized performance, not absolute F1 alone.

## Mathematical Ladder Definition

A principled ladder can be defined by two independent knobs:

```text
d_l      graph size for level l
rho_l    target graph density for level l
power_l  target statistical power for detecting a weak relevant edge
```

Everything else is derived.

### Edge count from density

For a DAG over `d` nodes, the maximum possible number of adjacencies is:

```text
M(d) = d(d - 1) / 2
```

Choose target density `rho_l`, then set:

```text
k_l = round(rho_l * M(d_l))
```

This directly controls how easy random guessing is.

For the corrected random baseline, a rough expected directed-F1 approximation is:

```text
E[random directed F1] ~= rho / (1 + 2*rho)
```

Reason:

- A true edge has about 1/2 chance of matching the sampled random order.
- The sampled random graph includes about half of all possible forward edges in expectation.
- So expected directed recall is near 1/4.
- Expected directed precision is near `rho/2`.

This gives:

| Density `rho` | Approx random directed F1 |
|---:|---:|
| 0.50 | 0.250 |
| 0.40 | 0.222 |
| 0.33 | 0.199 |
| 0.27 | 0.175 |
| 0.21 | 0.148 |

Design implication:

- Real benchmark levels should use `rho <= 0.33`.
- Tutorial levels may use `rho ~= 0.50`.
- A level with `rho > 0.50` is likely too guessable for directed F1.

### Sample size from statistical power

For linear-Gaussian causal discovery, conditional independence tests are often based on Fisher's z transform of partial correlation.

Let:

```text
alpha       significance level, e.g. 0.05
power       target detection probability, e.g. 0.80 or 0.95
s           conditioning set size used for the weakest-edge test
r_min       minimum relevant absolute partial correlation to detect
z_q         q-th normal quantile
```

Approximate required observational sample size:

```text
n_obs >= s + 3 + ((z_(1-alpha/2) + z_power) / atanh(r_min))^2
```

Then:

```text
n_obs_l = ceil(required sample size)
n_int_l = ceil(n_obs_l / 2)
```

Important caveat:

`r_min` should not be guessed from `noise_var` alone. In this environment it should be estimated from the accepted SCM distribution, for example as a low quantile of relevant structural partial correlations after graph and SCM rejection:

```text
r_min_l = quantile_q(
    abs(relevant_structural_partial_correlations(instance)),
    q=0.10
)
```

This makes the power equation empirical but still principled: graph size and density define combinatorial difficulty; the accepted SCM distribution defines the weakest effect the test is expected to detect.

### Intervention budget

Current instance generation sets:

```text
budget = size(optimal_intervention_set) + budget_slack
```

For paper-scale runs, keep:

```text
budget_slack = 2 for tutorial
budget_slack = 1 for normal levels
budget_slack = 0 for hard/noisy levels
```

Also consider a hard cap:

```text
budget <= 4
```

This keeps active raw prompts bounded and avoids letting context cost scale quadratically with `d`.

## Recommended V1 Ladder Policy

Use an equation-defined ladder, then verify it with random and PC before spending LLM budget.

Recommended constraints:

```text
d <= 10 for main paper levels
d = 12 allowed as stress level
rho <= 0.33 for real levels
rho ~= 0.50 only for tutorial
n_obs from Fisher-z power target, or use the current calibrated small-n values if explicitly described as low-sample stress tests
n_int = ceil(n_obs / 2)
budget_slack decreases with difficulty
```

The current density probe candidate is a good empirical calibration ladder:

| Level | Role | d | k | Density | n_obs | n_int | noise_var | slack |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | tutorial only | 4 | 3 | 0.500 | 50 | 25 | 0.5 | 2 |
| 1 | small core | 6 | 6 | 0.400 | 25 | 15 | 1.0 | 1 |
| 2 | medium core | 8 | 9 | 0.321 | 25 | 15 | 1.0 | 1 |
| 3 | real scale | 10 | 12 | 0.267 | 25 | 15 | 1.0 | 1 |
| 4 | structural stress | 12 | 14 | 0.212 | 25 | 15 | 1.0 | 1 |
| 5 | noisy hard | 10 | 12 | 0.267 | 15 | 10 | 1.5 | 0 |

For a cleaner paper design, prefer comparisons that isolate one axis at a time:

```text
same d, same density, different n_obs/noise     -> statistical difficulty
same density/power, increasing d                -> graph-size difficulty
same d/power, decreasing density                -> random-guessing difficulty
```

Avoid interpreting a level difference when `d`, density, sample size, noise, and budget all change together.

## Plausible Research Claims

These are plausible, not final, until rerun on the final equation-defined ladder.

1. Blind random guessing can score surprisingly well on dense small DAGs.

   This is a benchmark-design artifact, not causal competence. Density must be reported and controlled.

2. Directed F1 is more revealing than skeleton F1 for LLM causal discovery.

   GPT often gets moderate skeleton F1 but remains near random on directed F1. That suggests it may identify associations or plausible adjacency patterns without reliable orientation.

3. Sonnet raw modes show stronger causal-structure signal than GPT on the original ladder.

   Sonnet raw observational and active modes beat random directed F1 by meaningful margins. GPT does not show the same margin.

4. Statistical tool access does not automatically improve LLM performance.

   Stats modes are mixed. This may indicate tool-use strategy failures, not just lack of information.

5. Context window is not the bottleneck for v1.

   At `d <= 12`, context size is safe. The main issue is reasoning over many possible edges with limited samples.

6. A random-normalized score should be considered.

   Absolute directed F1 can mislead across densities. A paper metric could report:

   ```text
   normalized_directed_f1 = (model_f1 - random_f1) / (oracle_f1 - random_f1)
                          = (model_f1 - random_f1) / (1 - random_f1)
   ```

   This makes levels with different densities more comparable.

## Next Steps Before More LLM Runs

1. Decide whether the final ladder is the current density-probe table or a stricter equation-derived table with paired comparisons.

2. If changing `run_ladder.py`, update only:

   - `ladder_levels()`
   - output metadata fields for `max_edges` and `density`
   - tests that pin the level catalog

3. Do not resume old runs after changing level definitions. Same level IDs would refer to different benchmark worlds.

4. Run in this order:

   ```powershell
   python run_random_dag_baseline.py --level-set density_probe --out-dir traces\ladder\random_density_probe --seeds-per-level 8 --samples-per-instance 100 --baseline-seed 12345
   ```

   Then, after ladder changes and fresh manifest generation:

   ```powershell
   python run_random_dag_baseline.py --manifest <new-run>\run_manifest.json --out-dir <new-random-baseline> --samples-per-instance 100 --baseline-seed 12345
   ```

5. Compare every model only against the random baseline generated from the same manifest.

6. Defer token preflight unless `d`, `n_obs`, `n_int`, or budget are increased materially. For `d <= 12`, it is not urgent.

## Source Artifacts

Random probe:

- `traces/ladder/random_density_probe/results_random_dag_summary.csv`
- `traces/ladder/random_density_probe/results_random_dag_long.csv`
- `traces/ladder/random_density_probe/random_dag_manifest.json`

Original apples-to-apples comparison:

- `traces/ladder/random_uniform_baseline/results_random_dag_summary.csv`
- `traces/ladder/full_ladder_toolcall_run1/results_summary.csv`
- `traces/ladder/sonnet46_full_ladder_run1/results_summary.csv`

Implementation:

- `run_random_dag_baseline.py`
- `tests/unit/causal_discovery/test_random_dag_baseline.py`
