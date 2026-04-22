"""Prompt assets for the fishing game LLM+Solver agent.

Policy on information disclosure:
  ESTIMATION_PROMPT:  gives the LLM only the parameter schema. No causal story,
                      no "what to look for," no hint that zone age is a
                      confound or that storms propagate. The LLM must infer
                      those patterns from the data itself. This is the honest
                      "causal inference over a known parameterization" setting
                      for POMDP Coder / LLM+Solver.

  DISCOVERY_PROMPT:   gives only the action space, tool API, and an abstract
                      reward statement ("each boat sent to a zone returns some
                      reward depending on hidden conditions"). The hidden
                      state factorization (storm / wind / equip / tide),
                      reward magnitudes, and the existence of two risk types
                      are NOT revealed. The agent must discover all of it.
"""

ESTIMATION_PROMPT = """\
You are analyzing a fishing-game dataset. The underlying environment has some
hidden state that evolves day-to-day and emits noisy observations. You will be
given 30 days of historical data (a catch log and a sensor/maintenance log)
and must estimate the numeric parameters of a pre-specified parametric model.

The schema of the model — which parameters exist and their shapes — is given
below. The causal / semantic meaning of each parameter, which observations are
informative for which latent variable, and what conditional structure the
environment actually has are NOT given. You must recover them from the data.

Treat "normal/source/propagated/far_propagated", "broken/ok", etc. as opaque
component labels of a mixture model: do NOT assume what they mean physically.
Estimate them so that, under any reasonable indexing, the parametric family
explains the observed data well.

OUTPUT: a single JSON object matching this schema exactly (every `_` replaced
by a number):

{
  "buoy_params": {
    "normal":         {"mean": _, "std": _},
    "source":         {"mean": _, "std": _},
    "propagated":     {"mean": _, "std": _},
    "far_propagated": {"mean": _, "std": _}
  },
  "equipment_inspection_params": {
    "broken": {"mean": _, "std": _},
    "ok":     {"mean": _, "std": _}
  },
  "equipment_age_offset_factor": _,
  "maintenance_alert_params": {
    "age_rate_factor": _,
    "failure_signal":  _
  },
  "water_temp_params": {
    "base": {"mean": _, "std": _},
    "tide_effect": _
  },
  "zone_temp_offset": {"A": _, "B": _, "C": _, "D": _},
  "storm_transition": [[_, _], [_, _]],
  "wind_transition":  [[_, _, _, _], [_, _, _, _], [_, _, _, _], [_, _, _, _]],
  "equip_transition": [[_, _, _, _, _], [_, _, _, _, _], [_, _, _, _, _], [_, _, _, _, _], [_, _, _, _, _]],
  "tide_transition":  [[_, _], [_, _]]
}

HARD CONSTRAINTS (the validator will reject otherwise):
- All std values >= 0.3.
- Every transition-matrix row sums to 1.0.
- Output valid JSON in a fenced ```json ... ``` block. Reasoning text before
  the block is allowed; do not put text after the closing fence.
"""


REFINEMENT_PROMPT = """\
You previously proposed these parameters:

{previous_patch_json}

When these parameters are used to run a Bayesian filter across the 30 days of
historical data, the following real observations receive near-zero probability
under your model. That means the model cannot explain what actually happened.

FAILURE CASES (lowest marginal likelihood first):
{failure_cases}

Rewrite the JSON so these observations become plausible. Keep the exact same
schema and field names. Change only numeric values (means, stds, rates,
transition entries, offsets).

HARD CONSTRAINTS (unchanged):
- All std values >= 0.3.
- Every transition-matrix row sums to 1.0.
- Output valid JSON in a fenced ```json ... ``` block.
"""


DISCOVERY_PROMPT = """\
You are managing a fishing fleet over a 20-day season across 4 zones
(A, B, C, D). Each day you allocate 1-10 boats across the zones and receive
a reward. You have access to a Python REPL (variables persist between days)
and to a set of SQL/report tools.

ACTION SPACE
------------
Each day submit {"A": n_A, "B": n_B, "C": n_C, "D": n_D} with 1 <= sum <= 10.

REWARD
------
Each boat returns some reward that depends on unknown, possibly hidden
conditions in the zone it is sent to. The reward function, the set of hidden
conditions, and how observations relate to them are NOT disclosed. You must
discover them from data.

AVAILABLE TOOLS
---------------
* run_python_code / save_to_file_and_run — a persistent Python REPL. Store
  any helper functions, learned models, or cached statistics here; they
  persist across days.
* query_fishing_log(sql) — read-only SELECT against a catch log with columns
  (day, zone, boats, reward) covering 30 days of pre-season data.
* query_maintenance_log(sql) — read-only SELECT against two joinable tables:
    sensor_log(day, zone, buoy_reading, equipment_reading, water_temp)
    maintenance_log(day, zone, alerts)
* Some tools are budget-limited per-day; check the returned error field.

Each morning you also receive a free observation bundle whose keys you can
inspect on day 1. Not all keys are reported on every day.

SUBMISSION
----------
Call submit_decisions(allocation=..., beliefs=..., reasoning=...).
`beliefs` is a free-form dict; put whatever summary statistics you compute.
If you do not know what to put in it, submit an empty dict — you will still
be graded on total reward.

YOUR TASK
---------
1. Day 1: query the historical database, analyze it with Python, and build
   any latent-variable model you want. There is no required parameterization.
2. Days 2-20: update your model with each day's free observations, pick
   an allocation, submit_decisions.

PRINCIPLES
----------
- Compute numerically in Python. Never eyeball numbers.
- Your functions must persist in the REPL — don't recompute SQL every day.
- Confounds are possible. Don't assume a single reading implies a single cause.
- You have no information about the hidden state structure beyond what you
  can infer from the data. There may be one latent variable, or several.
"""
