# ACDB paper figure inventory

## Figure rule

Every main-paper figure must answer a concrete paper question. If a figure does not clarify the apparatus, the calibration, or a load-bearing empirical claim, it stays out.

## Main-paper candidates

### Figure 1 - Hidden-state / agent-state asymmetry

- Question:
  - what does the evaluator know that the agent does not?
- Core object:
  - hidden SCM / DAG / CPDAG / identifying set vs public observations and interventions
- Inputs:
  - benchmark concept only
  - existing prototype: `reports/figure_prototypes/20260429T063348Z/01_hidden_agent_asymmetry.png`
- Placement:
  - main paper, in `Scoring and Calibration`
- Status:
  - prototype exists
  - concept stable
  - regeneration optional, not blocked on final runs

### Figure 2 - Layered scoring contract

- Question:
  - what exactly do the benchmark metrics measure?
- Core object:
  - `skeleton F1`, `compelled-edge F1`, `directed F1`, `SHD`
- Inputs:
  - scoring contract
  - existing prototype: `reports/figure_prototypes/20260429T063348Z/02_layered_scoring_contract.png`
- Placement:
  - main paper, in `Benchmark Setup`
- Status:
  - prototype exists
  - concept stable
  - regeneration optional

### Figure 3 - DAG -> CPDAG -> post-intervention graph example

- Question:
  - why must observational ceiling and full DAG recovery be separated?
- Core object:
  - true DAG, CPDAG, one intervention resolving undirected edges
- Inputs:
  - equivalence theory
  - existing prototype: `reports/figure_prototypes/20260429T063348Z/03_dag_cpdag_intervention.png`
- Placement:
  - main paper
- Status:
  - prototype exists
  - concept stable
  - regeneration optional

### Figure 4 - Instance generation / acceptance pipeline

- Question:
  - how is a valid benchmark instance constructed and filtered?
- Core object:
  - DAG generation, graph rejection, SCM parameterization, SCM rejection, minimum intervention set, label permutation, data sampling
- Inputs:
  - `docs/specs/causal-discovery-v1-pseudocode.md`
  - `src/causal_discovery/benchmark/instance.py`
  - `src/causal_discovery/equivalence/theory.py`
- Placement:
  - appendix by default; only return to the main paper if page pressure is resolved elsewhere
- Status:
  - no current prototype in the tracked prototype bundle
  - should be built from the benchmark pseudocode, not from paper prose

### Figure 5 - Random-floor / expected-F1 calibration

- Question:
  - how does blind-random directed F1 scale with graph density, and where does the current ladder sit?
- Core object:
  - expected random directed-F1 floor vs density
- Inputs:
  - `docs/notes/random-baseline-ladder-findings.md`
  - `traces/ladder_random_floor_sanity/summary.csv`
  - prototype: `reports/figure_prototypes/20260429T063348Z/05_random_floor_density.png`
- Placement:
  - main paper, in `Scoring and Calibration`
- Status:
  - prototype exists
  - should be regenerated using the final exact floor data that the paper cites

### Figure 6 - Earlier vs current ladder region plot

- Question:
  - how did the benchmark move from the old guessable region to the calibrated one?
- Core object:
  - level placement in density / expected-floor space
- Inputs:
  - prototype manifest and ladder parameters
  - prototype: `reports/figure_prototypes/20260429T063348Z/08_v0_v1_ladder_regions.png`
- Placement:
  - main paper, in `Scoring and Calibration`
- Status:
  - prototype exists
  - concept stable
  - should be regenerated if labels or ladder annotations change

### Figure 7 - Cross-model ablation comparison

- Question:
  - how do the five models compare on the main active task, and how do the main ablations change that comparison?
- Core object:
  - `Avg L1-L5` cross-model comparison on `dir_f1` and `SHD`
- Inputs:
  - `docs/notes/RESULTS.md`
- Placement:
  - main paper
- Status:
  - no final paper artifact yet
  - should be generated directly from `docs/notes/RESULTS.md`
  - `pc_greedy` should appear as a shared reference line, not repeated bars

### Figure 8 - Ladder scaling comparison

- Question:
  - how does the same end-to-end comparison behave from `L1` to `L5`?
- Core object:
  - level-wise `dir_f1` and `SHD` on selected end-to-end policies plus `pc_greedy`
- Inputs:
  - `docs/notes/RESULTS.md`
- Placement:
  - main paper
- Status:
  - no final paper artifact yet
  - should be generated directly from `docs/notes/RESULTS.md`

## Appendix candidates

### Representative graph output comparison

- Question:
  - what do benchmark outputs look like on one concrete instance?
- Inputs:
  - in-scope final traces with submitted edge lists
  - current prototype: `reports/figure_prototypes/20260429T063348Z/04_representative_graph_output.png`
- Placement:
  - appendix by default
- Status:
  - current prototype uses a DeepSeek trace outside the current paper model panel
  - regenerate only after choosing a representative in-scope seed

### Active-gain plot

- Question:
  - how much do interventions help relative to observational stopping points?
- Inputs:
  - observational and active traces on matched seeds
  - current prototype: `reports/figure_prototypes/20260429T063348Z/07_active_gain.png`
- Placement:
  - appendix or optional discussion figure
- Status:
  - current prototype uses older GPT/Sonnet traces and includes `pc_obs` logic that is outside the current main ablation roster
  - not part of the current paper core

### Query / session formation diagram

- Question:
  - what exactly is passed to the agent at each step?
- Inputs:
  - `src/causal_discovery/agents/prompts.py`
  - `src/causal_discovery/agents/tool_schema.py`
  - `src/causal_discovery/agents/llm.py`
- Placement:
  - appendix only
- Status:
  - no current prototype
  - useful only if the paper needs more implementation transparency

## Current prototype bundle

- Directory:
  - `reports/figure_prototypes/20260429T063348Z`
- Existing artifacts:
  - `01_hidden_agent_asymmetry`
  - `02_layered_scoring_contract`
  - `03_dag_cpdag_intervention`
  - `04_representative_graph_output`
  - `05_random_floor_density`
  - `06_precision_recall_shd`
  - `07_active_gain`
  - `08_v0_v1_ladder_regions`
- Bundle caveat:
  - the manifest explicitly says these are prototypes only and should not be cited as final paper figures.
