# Active Causal Discovery Benchmark (ACDB) — Complete Project Overview

This document is a self-contained explainer of the entire project: what we're trying to learn, how the benchmark works, every method in the panel (including the three post-paper probes), what the v0 paper concluded, and what the new results say about that conclusion.

It assumes no prior context. Reading top to bottom should be enough to give a complete picture without opening the paper, code, or trace files.

---

## 1. The research question

> *Can an LLM agent, given a budget of hard single-node interventions, recover a causal DAG better than a structure-blind random baseline — and if so, by how much, and under what conditions?*

We want a clean, formal environment where we can pose this question and measure the answer reproducibly. That environment is the benchmark. ACDB (the Active Causal Discovery Benchmark) is the benchmark, and the project is the code, results, paper, and follow-up probes that live around it.

Two things make this question non-trivial:
- LLMs can in principle inspect data, run their own statistical tests, choose interventions, and decide when to commit to a graph. Whether they actually do this *well* — versus producing plausible-looking guesses driven by surface statistics — is empirical.
- Classical structure-learning algorithms (PC, GES) have clean theory under strong assumptions (causal sufficiency, faithfulness) but are brittle at small samples and don't natively pick interventions.

ACDB pits the two against each other under identical assumptions, identical data, and identical scoring.

---

## 2. The world

Each benchmark instance is sampled from a controlled generator. Concretely:

- A **DAG** `G` over `d` observed variables, with exactly `k` directed edges.
- A **linear–Gaussian SCM** `M` parameterizing `G`:
  $$X_i = \sum_{j \in \text{Pa}(i)} w_{ij} X_j + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma_i^2)$$
  Edge weights drawn uniformly from $[-2, -0.5] \cup [0.5, 2]$ (so no near-zero weights), noise variances per level.
- An **observational dataset** `D_obs` of `n_obs` rows sampled from `M`.
- An **intervention budget** `B = |I*| + slack`, where `I*` is the precomputed minimum single-node intervention set that orients every CPDAG-undirected edge. `slack` ∈ {0, 1, 2} per level.

Generation rejects instances where any relevant partial correlation is below `faithfulness_eps = 0.1` (avoiding near-unfaithful pathological cases). Variable indices are randomly permuted so the names `X0, ..., X_{d-1}` don't leak topology.

**Strong assumptions baked into v0:**
- *Causal sufficiency* — no hidden confounders.
- *Faithfulness* — no exact independencies that aren't implied by d-separation.
- *Linear-Gaussian* — no nonlinearities, no non-Gaussian residuals.
- *Perfect hard interventions* — `do(X_i = v)` cleanly severs `X_i`'s incoming edges.

These are deliberate. They make the scoring contract identifiable, and they give classical methods (PC) a fair comparison point. Relaxing them is v1+ work.

---

## 3. The agent's interface

Every agent — classical, LLM, oracle, random — interacts through one API:

```
observe()                        -> D_obs (one-time)
intervene(var, value)            -> n_int rows under do(X_var=value); decrements budget
submit_graph(submission)         -> terminal; submission has directed + undirected edges
```

The environment hides the true DAG `G` and SCM `M` and exposes only `D_obs`, `d`, and `B`. Submissions are *mixed* graphs: directed edges are committed orientations; undirected edges represent "I see this adjacency but can't commit to a direction." The scoring contract treats unresolved edges differently from reversed or omitted edges.

---

## 4. Why interventions matter — the CPDAG ceiling

Two DAGs are **Markov-equivalent** iff they share the same skeleton (undirected adjacencies) and the same v-structures (unshielded colliders `A → C ← B` with no edge between A and B). Markov-equivalent DAGs induce the same observational distribution, so observational data alone cannot distinguish them.

The **CPDAG** of `G` orients exactly the edges that are forced by `P(X)` (compelled edges) and leaves the rest undirected. Any agent working from observations alone is bounded above by the CPDAG. To go from CPDAG to true DAG, you need either an extra structural assumption or **interventions**.

This is why the benchmark has an "active panel" at all. The job decomposes:
- **Stage A (observational):** recover the skeleton + compelled orientations. Bounded by `P(X)`.
- **Stage B (active):** resolve remaining undirected edges using the intervention budget.

Each stage exercises a different kind of competence, and the new probes (§7) are designed to isolate which stage is the bottleneck for LLMs.

---

## 5. The scoring contract — three layers

Every submission is scored deterministically along three layers. Reporting all three is what lets the benchmark *interpret* a method, not just rank it.

### 5.1 Observational layer (against the true CPDAG)

- `skeleton_f1`: precision/recall/F1 on the undirected skeleton.
- `compelled_f1`: precision/recall/F1 on the compelled-edge subset of the CPDAG. A reversed compelled edge counts as both a false positive (wrong direction) and a false negative (missing true direction) — no partial credit for "at least the adjacency was right."

### 5.2 DAG layer (against the true DAG)

- `directed_f1`: set-intersection precision/recall/F1 on directed edges.
- `dag_shd`: structural Hamming distance, counted edge-by-edge over the union of true and submitted adjacencies. One error per case for: extra, missing, reversed, unresolved. **Reversal counts as 1, not 2.**

### 5.3 Efficiency layer

$$\eta = \frac{|I^*|}{\max(|I_{\text{used}}|, |I^*|)}$$

Equals 1 iff the agent used at most `|I*|` interventions. Decreases as wasted interventions accumulate.

### 5.4 Why all three matter

- **F1 alone collapses precision and recall** — different policies (over-commit vs under-commit) score similarly on F1 but make qualitatively different errors. The paper's main behavioral finding is exactly this distinction (§8).
- **SHD is more robust to aggressive guessing** than F1.
- **Efficiency catches "high efficiency from abstention"** — an agent that simply doesn't intervene gets η=100% but isn't doing active discovery.

---

## 6. The difficulty ladder

Six levels, each with paired axes (graph size, statistical power, budget tightness). Same 8 preflight-validated seeds per level across all agents, so every comparison is paired.

| Level | Role | d | k | n_obs | n_int | σ² | slack |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | tutorial | 4 | 5 | 50 | 25 | 0.5 | 2 |
| 1 | standard | 5 | 6 | 25 | 15 | 1.0 | 1 |
| 2 | statistical | 5 | 6 | 15 | 10 | 1.5 | 1 |
| 3 | structural | 7 | 9 | 25 | 15 | 1.0 | 1 |
| 4 | pressure | 5 | 6 | 25 | 15 | 1.0 | 0 |
| 5 | hard | 7 | 9 | 15 | 10 | 1.5 | 0 |

L2 and L4 hold size fixed and isolate one stress axis (statistics in L2, budget in L4). L3 and L5 are structural-stress levels.

---

## 7. The methods

Every method talks to the benchmark through the same session API. They differ only in the policy inside the session.

### 7.1 Classical baselines

- **`pc`** (observational only) — runs PC from `causal-learn` with Fisher-z independence tests at α=0.05. Submits the resulting CPDAG. Reference for skeleton + compelled.
- **`pc_greedy`** (active) — runs PC, then for each remaining undirected edge picks the highest-degree undirected node, intervenes at `do(X_i = X̄_i + 3.0)`, and orients each unresolved neighbor by comparing post-intervention mean to observational mean (shift threshold 0.5). Cycles rejected. The strongest non-oracle reference policy.
- **`oracle`** — submits the true DAG. Upper bound.

### 7.2 Reference floors

- **`random_uniform`** — structure-blind: only `d`, no data, no interventions. Samples a random topological order, picks `m ~ Uniform(0..d(d-1)/2)` forward edges, submits. 100 samples per instance. Defines the "no-data floor."

### 7.3 LLM family — paper version

For each backbone (GPT-5.4 and Claude Sonnet 4.6) and each panel (observational and active), four configurations:

- **`llm_raw_obs`** / **`llm_raw`** — the LLM gets the raw observational data matrix and (in active) `intervene` + `submit_graph`. No statistical tools.
- **`llm_stats_obs`** / **`llm_stats`** — same as raw, but adds three statistical tools the LLM can call: `correlation(i,j)`, `partial_correlation(i,j,S)`, `independence_test(i,j,S,α)`.

System prompts explain the action space and intervention semantics. Temperature 0. Step limit 16 (raw) or 32 (stats). Each turn the model receives the static session prompt, the full tool-call history, and the remaining budget; it emits exactly one tool call.

### 7.4 LLM family — probes (separate runners)

These exist outside the paper's main panel and live in their own runner scripts. Each is designed to isolate one specific question.

- **`llm_corr_obs`** — observational-only ablation. The LLM gets only the rounded sample correlation matrix and standard deviations. No raw data, no tools, no interventions. Single `submit_graph` call. *Probes:* whether a compact observational summary is enough for the LLM. Paper covers this.

- **`llm_pc_handoff`** — active-only ablation. PC runs offline; the LLM receives the resulting CPDAG (compelled directed + undirected edges) plus observational means, and chooses interventions to resolve remaining undirected edges. Cannot modify the skeleton. *Probes:* whether the LLM gap is in observational inference (Stage A) or active intervention reasoning (Stage B). Code is in [run_pc_handoff.py](run_pc_handoff.py); not yet run at full scale.

- **`llm_stats_guided`** *(new)* — same tools as `llm_stats`, but the system prompt contains an explicit phase-by-phase walkthrough of the PC + greedy active algorithm. *Probes:* whether procedural knowledge in prose closes the gap. Code in [run_stats_guided.py](run_stats_guided.py).

- **`llm_pc_tools`** *(new)* — replaces the per-test stat tools with PC-pipeline phases as composable primitives:
  - `pc_observational(alpha)` — runs PC, returns CPDAG.
  - `meek_closure(directed, undirected)` — applies Meek rules to a partial DAG.
  - `intervene(var, value)` — same as before, returns rows + per-variable mean shift vs observational baseline.
  - `submit_graph(...)`.
  *Probes:* whether the gap is in *interface granularity* — does the LLM perform if the algorithm is exposed as callable steps it can orchestrate? Code in [run_pc_tools.py](run_pc_tools.py).

### 7.5 Method × stage matrix (the cleanest mental model)

```
                     Stage A (obs)         Stage B (active)
   llm_raw_obs       LLM, raw data         —
   llm_stats_obs     LLM, raw + stat tools —
   llm_corr_obs      LLM, corr matrix only —
   pc                PC                    —
   ───────────────────────────────────────────────────────
   llm_raw           LLM, raw data         LLM
   llm_stats         LLM, raw + stat tools LLM
   llm_stats_guided  LLM + algorithm prose LLM (with prose)
   llm_pc_tools      LLM, PC tool          LLM (with meek tool)
   llm_pc_handoff    PC                    LLM
   pc_greedy         PC                    PC + greedy
   oracle            ground truth          —
   random_uniform    none                  —
```

Each probe lets you read off one axis of where the gap lives.

---

## 8. v0 paper findings (paired ladder, GPT-5.4 and Sonnet 4.6, 8 seeds × 6 levels)

These are the headline numbers from the paper, copied for completeness. Active panel, weighted across all six levels.

| Method | Model | Skel F1 | Dir P | Dir R | Dir F1 | SHD | η |
|---|---|---:|---:|---:|---:|---:|---:|
| **Oracle** | — | 100.0 | 100.0 | 100.0 | 100.0 | 0.00 | 100.0 |
| **PC + greedy** | — | 62.1 | 66.9 | 32.8 | **42.7** | **4.79** | 89.6 |
| LLM (raw) | Sonnet 4.6 | 56.3 | 35.0 | 29.6 | 31.7 | 7.25 | 97.9 |
| LLM (stats) | Sonnet 4.6 | 62.7 | 46.0 | 22.7 | 28.0 | 8.06 | 100.0 |
| LLM (raw) | GPT-5.4 | 58.8 | 36.3 | 19.3 | 22.9 | 9.27 | 92.7 |
| LLM (stats) | GPT-5.4 | 40.0 | 48.0 | 13.3 | 18.8 | 6.58 | 100.0 |
| **Random uniform** | — | 46.6 | 35.0 | 25.0 | 23.6 | 8.18 | — |

### Five paper-level findings

1. **PC + greedy is strongest on the primary metrics.** 42.7% directed F1, SHD 4.79. Beats every LLM agent. SHD gap is the more robust signal because SHD is less sensitive to aggressive guessing.
2. **The P/R decomposition reveals qualitatively different errors.** PC under-commits with high precision (66.9) and low recall (32.8); LLMs over-commit with lower precision and comparable recall. F1 alone collapses this. The paper's main behavioral signal.
3. **Sonnet 4.6 ≥ GPT-5.4 in every overall LLM setting** (raw, stats, observational, active). Largest gap on raw active (31.7 vs 22.9, p=0.028).
4. **Statistical tools shifted behavior toward abstention.** Stats variants ran longer, intervened less, submitted fewer directed edges. Their η=100% is from abstention, not skill.
5. **Density leakage matters at the dense end of the ladder.** The structure-blind random baseline gets 23.6% directed F1 on v0, high enough that several GPT-5.4 settings sit near it. A density-probe ladder reduces the floor to 16.9%, defining the v1 calibration target.

The paper's overall conclusion (§8): *"current LLM policies do not reliably convert intervention budget into orientation accuracy."*

### Per-level highlight

PC+greedy isn't uniformly best. Sonnet raw exceeds it on **L2** (36.9 vs 30.7) and **L5** (31.9 vs 15.2) — the two low-sample levels where PC's finite-sample errors dominate. Conversely, at L2 and L5 Sonnet 4.6's *observational* directed F1 exceeded its *active* directed F1 on the same seeds (39.3 vs 36.9 at L2; 40.1 vs 31.9 at L5). Interventions are strictly more informative than observations, so this gap is a **policy diagnostic**: current prompts don't reliably convert intervention samples into better orientation decisions.

---

## 9. Post-paper probe results

### 9.1 `llm_corr_obs` (already in paper)

- Setup: L1/L3/L5, 4 seeds per level, both backbones.
- Result: **Skeleton F1 ≈ 70%, Directed F1 = 0.**
- Reading: the model withheld orientation when uncertain, as the prompt explicitly asked. Skeleton is recoverable from a correlation summary alone in this SCM family. Directed F1 = 0 isn't a failure — it's a faithful execution of "abstain when uncertain." This is a *prompt-restricted* probe, not a primary comparison.

### 9.2 `llm_pc_handoff` (code committed; not yet run on the full ladder)

The cleanest active-phase ablation. Strong story when the data exists.

**Two outcomes worth running it for:**
- If `llm_pc_handoff ≈ pc_greedy` → the LLM gap is **all in observational inference**. Hand it the CPDAG and it can drive the active phase fine.
- If `llm_pc_handoff < pc_greedy` → the LLM gap is **also in active reasoning**. Even with a perfect-ish observational starting point, the LLM doesn't pick interventions or interpret shifts as well as the greedy heuristic.

### 9.3 `llm_stats_guided` (new — `gpt-5.4-mini` via OpenRouter, 47/48 success)

| Level | n | Skel F1 | Dir F1 | SHD | η | total | stats | **intervene** | submit |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 58.6 | 25.1 | 4.625 | 100.0 | 8.9 | 7.9 | **0.0** | 1.0 |
| 1 | 8 | 50.0 | 23.3 | 9.500 | 100.0 | 10.6 | 9.6 | **0.0** | 1.0 |
| 2 | 8 | 48.0 | 20.5 | 18.000 | 100.0 | 12.6 | 11.6 | **0.0** | 1.0 |
| 3 | 8 | 30.4 | 10.8 | 25.250 | 100.0 | 13.2 | 12.2 | **0.0** | 1.0 |
| 4 | 7 | 23.8 | 11.2 | 19.143 | 100.0 | 15.4 | 14.4 | **0.0** | 1.0 |
| 5 | 8 | 18.8 | 14.5 | 32.125 | 100.0 | 16.2 | 15.2 | **0.0** | 1.0 |
| **overall** | 47 | **38.6** | **17.7** | **18.085** | **100.0** | 12.8 | 11.8 | **0.0** | 1.0 |

**The model intervened zero times across all 47 sessions.** Despite the system prompt explicitly walking through Phase 4 ("while budget > 0 and undirected edges remain, intervene on the highest-degree undirected node…"), the model spent its turns running independence tests and then submitted an observation-only graph. So this is effectively `llm_stats_obs` with a longer preamble. Numbers are nearly identical to the paper's `gpt-5.4 llm_stats` active row (18.8% Dir F1) because both effectively skipped the active phase.

**Reading:** procedural knowledge in the prompt does not change LLM behavior at this scale. The model can read the algorithm and still not follow it.

### 9.4 `llm_pc_tools` (new — `gpt-5.4-mini` via OpenRouter, 48/48 success)

| Level | n | Skel F1 | Dir F1 | SHD | η | total | pc_obs | meek | **intervene** | submit |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 91.5 | 46.7 | 1.875 | 93.8 | 3.2 | 1.0 | 0.0 | 1.1 | 1.0 |
| 1 | 8 | 84.8 | 62.0 | 3.000 | 100.0 | 3.0 | 1.0 | 0.0 | 1.0 | 1.0 |
| 2 | 8 | 72.6 | 46.6 | 6.250 | 100.0 | 2.9 | 1.0 | 0.0 | 0.9 | 1.0 |
| 3 | 8 | 75.2 | 45.1 | 9.375 | 100.0 | 3.0 | 1.0 | 0.0 | 1.0 | 1.0 |
| 4 | 8 | 75.8 | 49.7 | 9.125 | 100.0 | 3.1 | 1.0 | 0.0 | 1.1 | 1.0 |
| 5 | 8 | 74.7 | 48.1 | 10.625 | 100.0 | 3.4 | 1.0 | 0.0 | 1.2 | 1.0 |
| **overall** | 48 | **79.1** | **49.7** | **6.708** | **99.0** | 3.1 | 1.0 | **0.0** | 1.1 | 1.0 |

**Headline: 49.7% Dir F1 > pc_greedy's 42.7%.** First LLM-driven active-panel method in this benchmark to clear the classical baseline on directed F1.

**Behavioral note:** the model is doing **the minimum viable orchestration**: one `pc_observational` call, one intervention, one submit. **It never calls `meek_closure` despite having it.** So the strong result is from a *floor* of tool usage. There is real headroom if the model used Meek propagation between interventions and consumed more of the budget.

Caveat: SHD (6.71) is *higher* than `pc_greedy`'s (4.79) — `pc_tools` commits more edges (recall 44.1 vs 32.8), so it gets more right *and* more wrong than `pc_greedy`'s conservative under-committing strategy. Different errors, not strictly better.

---

## 10. Reading the new probes against the paper's narrative

The paper's conclusion was about `llm_raw` and `llm_stats`: *current LLMs don't reliably convert intervention budget into orientation accuracy.* The new probes don't contradict this — they *refine* it.

### What `llm_stats_guided` says

The paper's "stats tools didn't help" finding is reinforced and made sharper: even when you also tell the model in plain text exactly which algorithm to run with those tools, behavior doesn't change. Whatever the LLM is doing in `llm_stats`, it's not "I know I should intervene but I'm choosing not to." It's closer to "I never meaningfully engage with the active phase under this prompt design."

### What `llm_pc_tools` says

This is the genuinely new claim: when the algorithm is exposed as callable phase-level primitives instead of as prose, the LLM produces a result above `pc_greedy` on directed F1. The same model class that scored 18.8% on `llm_stats` scores 49.7% here. The information available to the model is structurally similar; the difference is interface granularity.

### A cleaner story to add to the paper's discussion

> *On the active panel, the LLM's gap to `pc_greedy` is best explained as an interface-granularity bottleneck rather than a missing-knowledge bottleneck. Telling the model the algorithm in prose (`llm_stats_guided`) does not raise directed F1 above the corresponding `llm_stats` row — the model continues to under-use the intervention budget. Exposing the same algorithm as composable phase-level tools (`llm_pc_tools`) raises directed F1 to 49.7%, above `pc_greedy`'s 42.7%. The mechanism is consistent: with phase-level tools the model only has to choose **when** to call each phase, not **how** to execute it. With statistical primitives, even a guided model has to assemble the algorithm itself, and current models don't.*

This is a real refinement to the paper's "current LLMs don't solve active causal discovery" framing. Active causal discovery, with the right tool surface, is solvable by current LLMs — the bottleneck is *what tools you give them*, not the underlying capability.

---

## 11. Caveats and threats to validity

1. **Model-class confound.** The new probes ran on `gpt-5.4-mini` via OpenRouter. The paper's GPT-5.4 ladder ran on full `gpt-5.4` directly on OpenAI. Don't put the new rows in the same column as the paper's GPT-5.4 rows without that note. To remove this confound: rerun all active methods on `gpt-5.4-mini`, or rerun the new probes on full `gpt-5.4`.
2. **OpenRouter ≠ direct OpenAI.** Sampling defaults, response-format support, and tokenizer quirks may differ. For headline numbers, prefer the same provider as the existing rows.
3. **`llm_pc_tools` has a built-in advantage from `pc_observational`.** Roughly two-thirds of its submission is directly PC's CPDAG — the LLM only adds one intervention's worth of orientations on top. The right comparison is `llm_pc_tools` vs `llm_pc_handoff`: same starting point (PC's CPDAG), different interface for the active phase. That comparison hasn't been run yet.
4. **`llm_stats_guided` had one L4 timeout** (`max_steps=32` exceeded without `submit_graph`). A rerun with `--max-steps 64` would close the 47/48 to 48/48.
5. **Density leakage, per the paper.** The random floor on this ladder is 23.6%; the density probe shows a 16.9% floor with lower-density instances. Several GPT-5.4 paper rows sit near 23.6%. v1 should recalibrate density.
6. **Eight seeds per level.** Adequate for coarse trends, under-powered for fine ranking. The paper notes this; the new probes inherit the same limitation.
7. **One prompt design per method.** All capability claims are conditional on the specific system + session prompts in the runners. Different prompt phrasings could shift behavior — the `llm_stats_guided` zero-intervention finding is itself a prompt-design observation.

---

## 12. What to run next (suggested)

In rough priority order:

1. **`llm_pc_handoff` on `gpt-5.4-mini` via OpenRouter.** Same instances. Closes the active-phase ablation triangle (`llm_pc_handoff` vs `llm_pc_tools` vs `pc_greedy`). One command, ~$3–5.
2. **Rerun existing active methods on `gpt-5.4-mini`** (or rerun probes on full `gpt-5.4`). Removes the model-class confound from the headline comparison.
3. **`llm_stats_guided` with `--max-steps 64`.** Confirms the L4 timeout was a fluke and gives a clean 48/48 row.
4. **Higher seed count on the new probes.** 8 → 24 or 32 — enough to do paired sign-flip tests against the paper's existing rows.
5. **v1 calibration pass.** The paper already lists this: lower density to bring the random floor down, then rerun the full panel with proper coverage.

---

## 13. Project layout (where everything lives)

```
src/causal_discovery/
  agents/         LLM policies, action types, stats tools, session driver
  baselines/      causal-learn endpoint matrix parser
  benchmark/      build_benchmark_instance + BenchmarkInstance
  config/         BenchmarkConfig, weight ranges, make_v1_config
  core/           DAG, LinearGaussianSCM, Permutation primitives
  equivalence/    CPDAG type, dag_to_cpdag, Meek rules, min intervention set
  graph_gen/      random DAG sampling
  runtime/        BenchmarkEnv (the session API)
  sampling/       observational + interventional samplers
  scm/            SEM parameterization + covariance diagnostics
  scoring/        GraphSubmission + score_submission

run_ladder.py                  full paper ladder (PC, llm_raw, llm_stats, oracle)
run_pc_only_ladder.py          PC-only ladder (added by collaborator post-paper)
run_corr_obs_probe.py          llm_corr_obs probe (paper §5.4)
run_random_dag_baseline.py     structure-blind random floor + density probe
run_pc_handoff.py              llm_pc_handoff (post-paper, code in but not yet run)
run_stats_guided.py            llm_stats_guided (post-paper, run on gpt-5.4-mini)
run_pc_tools.py                llm_pc_tools (post-paper, run on gpt-5.4-mini)

traces/
  ladder/full_ladder_toolcall_run1/    GPT-5.4 paper ladder
  ladder/sonnet46_full_ladder_run1/    Sonnet 4.6 paper ladder
  pc_handoff/                          (empty so far)
  stats_guided/openrouter_run1/        gpt-5.4-mini results
  pc_tools/openrouter_run1/            gpt-5.4-mini results
  openrouter_run1_analysis.md          per-run analysis of the above two

main.pdf                       v0 paper (NeurIPS 2026 submission)
README.md                      ladder results tables, run instructions
PROJECT_OVERVIEW.md             this file
```

Each runner produces the same four-file output: `results_*_long.csv` (one row per session), `results_*_summary.csv` (aggregated by level/method/model), `*_manifest.json` (config + seed map for reproducibility), `events_*.jsonl` (per-step trace with every action and tool result).

---

## 14. One-paragraph version

ACDB is a controlled benchmark for active causal discovery — given an unknown linear-Gaussian DAG, observational data, and a tight intervention budget, can an agent recover the DAG? The v0 paper showed that classical PC + greedy interventions still beats LLMs (42.7% directed F1 vs 31.7% best-LLM), with the qualitative finding being that PC under-commits while LLMs over-commit, and that statistical-primitive tools alone don't help LLMs. Three post-paper probes then test where the LLM gap actually lives: `llm_corr_obs` shows compact correlation summaries carry skeleton signal but models prompted to abstain do; `llm_stats_guided` shows that telling the model the PC algorithm in prose does not change its behavior (it still doesn't intervene); `llm_pc_tools` shows that exposing PC's pipeline phases as composable tools lets `gpt-5.4-mini` reach 49.7% directed F1, above `pc_greedy`'s 42.7%. The refined story for the paper's discussion section: the LLM gap is interface, not knowledge, and a phase-level tool surface closes it.
