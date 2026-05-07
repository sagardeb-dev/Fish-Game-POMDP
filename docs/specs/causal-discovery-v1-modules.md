# Causal Discovery V1 Module Spec

This document defines the module boundaries for the `feat/causal-discovery` project.

The goal is to keep the benchmark scientifically correct and implementation-safe by separating:

- graph generation
- SCM parameterization
- equivalence/intervention theory
- data sampling
- runtime agent interaction
- scoring

Library choices should be made inside module boundaries, not across them.

---

## 1. Design Principles

### 1.1 Source of truth

The benchmark is specified by:

- a true DAG
- a linear-Gaussian SCM over that DAG
- a CPDAG derived from the DAG
- an optimal intervention set derived from the DAG/CPDAG pair

Everything else is derived from those objects.

### 1.2 Hidden information

The agent must not see:

- internal topological order
- true DAG
- CPDAG
- optimal intervention set
- SCM parameters

The runtime exposes only sampled data and intervention results.

### 1.3 Separation rule

No module should do two jobs if those jobs can be validated independently.

In particular:

- the runtime must not compute CPDAGs
- the scorer must not generate data
- the sampler must not decide intervention optimality
- graph generation must not know about agent budgets

---

## 2. Module Layout

Recommended top-level package split:

```text
causal_discovery/
  config/
  graph_gen/
  scm/
  equivalence/
  sampling/
  benchmark/
  scoring/
  agents/
  baselines/
  utils/
```

The names can change. The boundary intent should not.

---

## 3. `config`

### Responsibility

Own all configurable benchmark parameters and defaults.

### Owns

- number of variables `d`
- number of edges `k`
- observational sample count `n_obs`
- interventional sample count `n_int`
- edge weight range
- noise variance
- faithfulness threshold
- budget slack
- rejection filter thresholds

### Returns

- a validated config object or dict

### Must not know

- how DAGs are represented internally
- how CPDAGs are computed
- how sampling is implemented

### Invariants

- config values are explicit and serializable
- no benchmark-critical constant is hardcoded outside this module

---

## 4. `graph_gen`

### Responsibility

Generate candidate DAG structures only.

### Inputs

- `d`
- `k`
- graph rejection thresholds if needed for structural prechecks

### Outputs

- internal DAG object
- optional internal topological order

### Required behavior

- sample a random topological order
- sample exactly `k` valid forward edges
- generate an acyclic DAG

### Must not know

- SCM coefficients
- observational data
- CPDAG scoring rules
- intervention budgets

### Invariants

- returned graph is acyclic
- returned graph has exactly `d` nodes
- returned graph has exactly `k` edges

---

## 5. `scm`

### Responsibility

Attach linear-Gaussian structural equations and noise to a DAG.

### Inputs

- DAG
- edge weight range
- noise variance settings
- faithfulness / conditioning thresholds

### Outputs

- SCM object
- implied parameter matrices as needed
- covariance matrix or equivalent derived quantities if cached

### Required behavior

- one equation per node
- parent-only linear structural dependence
- independent Gaussian exogenous noise

### Must not know

- CPDAG
- intervention budget
- runtime agent interface

### Invariants

- SCM matches DAG exactly
- exogenous noises are independent
- no hidden confounders in v1

### Notes

This is the scientific core. Prefer a clean local implementation over a library abstraction that hides semantics.

---

## 6. `equivalence`

### Responsibility

Own all graph-theoretic benchmark targets derived from the true DAG.

### Inputs

- true DAG
- optionally SCM if some diagnostics need it

### Outputs

- CPDAG
- optimal intervention set under the allowed intervention class
- optional diagnostics

### Required behavior

- compute CPDAG directly from DAG
- compute minimum perfect single-node intervention set needed to fully orient the CPDAG into the true DAG

### Must not know

- sampled data
- agent behavior
- runtime budget usage

### Invariants

- CPDAG is the observational ceiling
- optimal intervention set is computed under benchmark intervention semantics, not generic theory assumptions that differ from runtime

### Notes

If a library is used, this is the most likely place for it. Keep library-dependent code here so the rest of the system does not depend on library graph objects.

---

## 7. `sampling`

### Responsibility

Generate observational and interventional datasets from the SCM.

### Inputs

- SCM
- sample count
- optional intervention specification

### Outputs

- tabular observational samples
- tabular interventional samples

### Required behavior

- observational sampling from the unmanipulated SCM
- interventional sampling under perfect hard interventions `do(X_i = value)`

### Must not know

- CPDAG
- true DAG score targets
- intervention optimality
- agent strategy

### Invariants

- same SCM semantics for observational and interventional data
- intervention output is sampled data, not oracle structural facts

---

## 8. `benchmark`

### Responsibility

Assemble a full benchmark instance and run the agent-facing interaction loop.

### Inputs

- config
- graph generator
- SCM generator
- equivalence module
- sampler

### Outputs

- benchmark instance object
- runtime session outputs

### Required behavior

#### Instance build phase

- generate candidate DAG
- derive CPDAG
- parameterize SCM
- reject trivial / degenerate instances
- compute optimal intervention set
- compute budget = `|optimal_set| + slack`
- randomly permute public variable labels
- sample initial observational dataset

#### Runtime phase

- expose `observe()`
- expose `intervene(var, val)`
- expose `submit_graph(GraphSubmission)`
- expose convenience submission methods for adjacency matrices and CPDAG edge sets
- provide observational data once at session start
- decrement budget on intervention calls
- return `n_int` sampled rows per intervention

### Must not know

- graph-theory internals beyond public outputs from `equivalence`
- implementation internals of SCM library if one is used

### Invariants

- public variable labels do not reveal generator order
- budget accounting is exact
- agent sees only public labels and sampled data
- observational data is provided once at session start, not re-sampleable

---

## 9. `scoring`

### Responsibility

Score an agent submission against benchmark targets.

### Inputs

- submitted graph
- true DAG
- CPDAG
- interventions used
- optimal intervention set size
- total budget

### Outputs

- skeleton score
- compelled-orientation score
- directed DAG score
- DAG SHD
- efficiency score

### Required behavior

- compare submitted skeleton against CPDAG skeleton
- compare submitted compelled orientations against CPDAG directed edges
- compare submitted directed edges against true DAG
- compute DAG SHD against true DAG
- compute intervention-efficiency metric from used vs optimal

### Must not know

- how data were sampled
- how DAG was generated
- runtime decision logic of the agent

### Invariants

- observational score does not require impossible full-DAG recovery from observations alone
- final score uses true DAG only after allowing intervention access
- unresolved undirected edges are representable in submissions

---

## 10. `agents`

### Responsibility

Define the benchmark-facing agent interface only.

### Required interface

- `observe(data)`
- `intervene(var, val)` via runtime request
- `submit_graph(GraphSubmission)`

### Notes

This package should not own oracle logic, CPDAG logic, or SCM generation.

---

## 11. `baselines`

### Responsibility

Hold benchmark baselines, not benchmark truth.

### Likely baseline types

- observational-only baseline
- active-intervention baseline
- random intervention baseline
- oracle-intervention baseline

### Must not know

- hidden SCM internals except where explicitly allowed for oracle baselines

### Invariants

- baseline privileges must be explicit
- oracles must be separated from realistic baselines

---

## 12. Cross-Module Contracts

### 12.1 DAG contract

The internal DAG representation must support:

- node enumeration
- parent queries
- edge enumeration
- relabeling/permutation

### 12.2 SCM contract

The SCM object must support:

- observational sampling
- interventional sampling
- covariance or equivalent diagnostics

### 12.3 CPDAG contract

The CPDAG representation must support:

- undirected edge queries
- orientation comparison
- relabeling/permutation

### 12.4 Runtime contract

The benchmark runtime must support:

- loading an instance
- returning observational data
- returning sampled interventional data
- accepting final graph submission

---

## 13. Recommended Validation Order

Implement and verify in this order:

1. `graph_gen`
2. `equivalence`
3. `scm`
4. `sampling`
5. `benchmark`
6. `scoring`
7. `baselines`

Reason:

- first verify graph objects and CPDAG generation
- then verify SCM semantics
- then verify sampled data
- only then expose runtime interaction

---

## 14. What To Keep Flexible

These should be configurable, not hardcoded:

- `d`
- `k`
- `n_obs`
- `n_int`
- intervention value
- edge weight bounds
- noise variance
- faithfulness threshold
- budget slack

---

## 15. What Not To Couple Early

Do not let:

- the benchmark runtime depend on a specific CPDAG library object
- the scorer depend on sampler internals
- the sampler depend on runtime budget policy
- baseline code call graph-theory internals directly unless that baseline is explicitly theory-aware

If a library is later replaced, only one module should move.
