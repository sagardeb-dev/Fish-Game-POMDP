# Experimental Setup priors

## Section objective

Section 4 should define the compared policies, the model panel, and the shared execution protocol. It should answer one question only:

> Given the same benchmark instances and the same scoring contract, what exactly differs across compared methods, and how were those comparisons run?

This section is not benchmark setup, not scoring, and not results interpretation.

## Pre-rewrite priming context

After the rewrites of Sections 2 and 3, Section 4 has to inherit a clearer center of gravity. The paper is no longer introducing a generic benchmark with some ablations attached. It is describing a controlled family of accepted causal-world instances, a fixed runtime over those instances, and a small set of policy cuts that expose different parts of the same active-discovery pipeline.

That changes how Section 4 should read:

- it should feel like controlled comparison over the same accepted instances;
- it should keep the benchmark instance family in view without re-explaining Section 2;
- it should use the Section 3 scoring contract implicitly by defining what each ablation leaves exposed;
- it should sound closer to experimental design in a synthetic-world paper than to an agent harness memo.

The main writing hazard is drift toward interface-first description. Tool access matters here, but only because it defines what evidence and actions each ablation is allowed to use. The section should therefore privilege:

- what structure is supplied;
- what stage of the pipeline is fixed externally;
- what stage remains owned by the compared policy;
- what causal subproblem the comparison isolates.

Do not let action names or loop mechanics become the conceptual center of the section.

## Source-of-truth hierarchy

1. `run_ladder.py`
2. `src/causal_discovery/agents/tool_schema.py`
3. `src/causal_discovery/agents/llm.py`
4. `src/causal_discovery/agents/litellm_model.py`
5. `docs/notes/OUTLINE.md`
6. `docs/notes/RESULTS.md`
7. `docs/notes/CLAIMS.md`

Use `research/sections/04_agents.tex` only as a stale audit. Do not draft from it.

## Section shape

### Opening paragraph

Required content:

- all methods are evaluated on the same accepted `L1-L5` instances;
- comparisons are paired by seed;
- Sections 2 and 3 already defined the benchmark world and the scoring contract;
- Section 4 defines the policies, the model panel, and the shared run protocol.

Keep out:

- benchmark mechanics already owned by Section 2;
- scoring or calibration already owned by Section 3;
- any results preview.

### 4.1 Ablation design

This subsection should lead with the ablation logic, not the tool inventory. It should first say that the five-policy set is chosen to separate three questions:

- how well a method constructs observational structure;
- how well it uses interventions once observational structure is fixed;
- how well it integrates the full pipeline end to end.

Only then should it present a compact table and short prose clarifying the purpose of each ablation.

After the Section 2/3 rewrites, the table should not be interface-dominant. It should read as a decomposition of the active-causal-discovery pipeline over the same accepted instances.

Required policy inventory:

| Paper label | Code id | Supplied structure | Remaining policy burden | Additional tools/actions | Isolated question |
|---|---|---|---|---|---|
| PC + greedy | `pc_greedy` | raw `D^{obs}` only | PC graph construction plus fixed active resolver | no LLM tools | classical active reference |
| LLM raw | `llm_raw` | raw `D^{obs}` only | full end-to-end active discovery | `intervene`, `submit_graph` | can one model do the full task end to end? |
| LLM stats | `llm_stats` | raw `D^{obs}` only | full end-to-end active discovery | stats tools + `intervene`, `submit_graph` | do CI-style tools improve active discovery? |
| PC CPDAG + LLM | `pc_cpdag_llm` | supplied PC CPDAG and summary stats | intervention use and final orientation | `intervene`, `submit_graph` | how well does the model use interventional evidence when observational burden is removed? |
| LLM stats + greedy | `llm_stats_cpdag_greedy` | raw `D^{obs}` only | observational graph construction only | stats tools + `submit_graph` | how good is the LLM's observational graph if active planning is removed? |

Additional drafting constraints:

- first mention of each policy should include the code identifier in parentheses;
- explain the ablations as a decomposition, not as a laundry list;
- keep policy prose balanced; do not give one method a long implementation-defense paragraph while compressing the others;
- no mention of observational-only LLM policies;
- no mention of `pc_obs`;
- no mention of `oracle`.

#### Exact policy facts from code

`pc_greedy`

- runs PC with Fisher-`z` conditional-independence tests at `alpha = 0.05`;
- starts from the PC CPDAG produced from `D^{obs}`;
- repeatedly selects the node with highest undirected-edge degree;
- intervenes at observational mean `+ 3.0`;
- orients neighbors with the benchmark's `z_calibrated_mean_shift` rule;
- this is **not** a fixed `0.5` threshold rule.

`llm_raw`

- allowed actions from `tool_schema.py`: `intervene`, `submit_graph`;
- receives raw observational data at session start;
- receives full intervention sample matrices after each intervention;
- owns the full active stage itself.

`llm_stats`

- allowed actions: `correlation`, `partial_correlation`, `independence_test`, `intervene`, `submit_graph`;
- same observational and intervention evidence as `llm_raw`;
- tests whether explicit CI-style statistical primitives help.

`pc_cpdag_llm`

- benchmark computes a PC graph before the LLM loop;
- session context includes:
  - PC directed edges,
  - PC undirected edges,
  - observational means,
  - observational standard deviations,
  - instruction to start from the supplied partial graph;
- raw observational matrix is withheld (`include_observational_data=False`);
- allowed actions: `intervene`, `submit_graph`.

`llm_stats_cpdag_greedy`

- allowed actions: `correlation`, `partial_correlation`, `independence_test`, `submit_graph`;
- the LLM does not intervene in this ablation;
- after LLM submission, the benchmark applies the same greedy intervention resolver used in `pc_greedy`.

### 4.2 Model panel

Required model list:

- `GPT-5.5`
- `GPT-5.4-mini`
- `Sonnet-4.6`
- `Haiku-4.5`
- `Gemini-3-Flash`

Required framing:

- compact cross-vendor, cross-capability panel;
- intended to demonstrate what ACDB can diagnose;
- not intended as an exhaustive leaderboard.

Keep out:

- provider pricing;
- latency discussion;
- pending or failed runs;
- strong claims about vendor families beyond what the results later support.

### 4.3 Run protocol

Required shared execution facts:

- paper uses `L1-L5` only;
- same accepted seed manifest is reused across all methods and models;
- `8` seeds per level in the current main run protocol;
- all comparisons are paired at the instance level;
- each LLM episode is a stateful sequential loop;
- exactly one action is emitted per turn;
- final step is submit-only;
- raw-style policies use `20` max steps;
- stats-style policies use `40` max steps.

Optional short implementation note if needed:

- each decision is conditioned on the accumulated interaction history;
- non-reasoning models run with temperature `0`.

Keep out:

- prompt wording;
- tool schema JSON;
- tool-history / working-memory internals unless a single sentence is needed for clarity;
- trace-behavior discussion;
- cost accounting.

## What must stay out of Section 4

- SCM equations and intervention semantics;
- CPDAG theory;
- score definitions;
- random-floor calibration;
- figure-specific discussion better owned by Sections 2 or 3;
- any result interpretation;
- any mention of `oracle`, `pc_obs`, random baseline, or deprecated observational-only LLM policies as part of the paper comparison set.

## Stale audit of `04_agents.tex`

The current file is not a valid drafting source because it still contains:

- the old title `Agents and Baselines`;
- an `oracle` paragraph;
- a random-baseline paragraph that belongs to calibration, not experimental setup;
- old two-backbone wording;
- observational/active panel language from the older paper state;
- a correlation-only probe that is out of current paper scope;
- an incorrect `pc_greedy` description using a fixed `0.5` threshold.

All of that should be treated as deletion context only.

## Citations likely needed

Minimal expected citations in this section:

- PC / causal discovery reference when first defining `pc_greedy`: `spirtes2000causation`;
- no extra citation is needed for the benchmark-owned hybrid ablations themselves.

This section should stay mostly benchmark-internal and procedural.

## Acceptance criteria

The eventual Section 4 draft is correct only if:

- it has exactly three substantive subsections: ablation design, model panel, run protocol;
- every policy description matches current code behavior;
- `L1-L5` is the only paper level range mentioned;
- `oracle`, `pc_obs`, random baseline, and correlation-only probe are absent;
- the section reads as controlled experimental design over the same accepted instances, not as benchmark setup, harness documentation, or results discussion.
