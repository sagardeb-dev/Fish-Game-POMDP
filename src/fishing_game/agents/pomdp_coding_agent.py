"""
POMDP Coder agent for the Fishing Game.

Faithful-spirit reimplementation of Gandhi et al. (arXiv 2505.02216, 2025),
"LLM-Guided Probabilistic Program Induction for POMDP Model Estimation."

Algorithm 2 (LearnModels), simplified:
  1. LLM_Init       — propose an initial parameter patch from historical data.
  2. Coverage score — replay the exact 80-state Bayesian filter on the 30 days
                      of historical sensor/maintenance data and compute the
                      mean log marginal likelihood per observation.
  3. Failure report — the K observations with lowest marginal likelihood are
                      shown to the LLM along with the previous patch.
  4. LLM_Refine     — up to M refinement rounds; keep the best-scoring patch.

Planning + per-day filtering are inherited unchanged from LLMSolverAgent.act().

Behavior differences vs the previous version:
  * NO silent fallback to _MOCK_CONFIG_PATCH. If LLM_Init produces an
    unparseable or schema-incomplete patch, `self.init_parse_failed` is set
    to True and a flat uniform-ish prior is used so the filter still runs —
    but this is recorded in telemetry so benchmark reports can exclude or
    flag these cases instead of silently scoring mock numbers.
  * M, k_failures, and all telemetry (init_coverage, final_coverage,
    n_refinements_applied, init_parse_failed, n_llm_calls) are exposed
    as instance attributes for the benchmark runner.
"""

import copy
import json
import math

import numpy as np

from fishing_game.agents.llm_solver_agent import (
    LLMSolverAgent,
    _deep_merge,
    _format_data_tables,
    _parse_config_patch,
    _print_estimated_params,
    _REQUIRED_PATCH_KEYS,
)
from fishing_game.pomdp import FishingPOMDP
from fishing_game.prompts.llm_solver import ESTIMATION_PROMPT, REFINEMENT_PROMPT


_EPS = 1e-12


# A neutral prior used only when LLM_Init fails to produce a schema-valid
# patch. Means are set near the midpoint of observed sensor ranges; stds are
# wide so nothing receives zero likelihood. Transition matrices are uniform.
# This prior is DELIBERATELY NOT tuned to ground truth — it is a failure
# sentinel, not a cheat.
_NEUTRAL_PRIOR_PATCH = {
    "buoy_params": {
        "normal":         {"mean": 2.0, "std": 2.0},
        "source":         {"mean": 2.0, "std": 2.0},
        "propagated":     {"mean": 2.0, "std": 2.0},
        "far_propagated": {"mean": 2.0, "std": 2.0},
    },
    "equipment_inspection_params": {
        "broken": {"mean": 4.0, "std": 3.0},
        "ok":     {"mean": 4.0, "std": 3.0},
    },
    "equipment_age_offset_factor": 0.0,
    "maintenance_alert_params": {
        "age_rate_factor": 0.5,
        "failure_signal":  0.5,
    },
    "water_temp_params": {
        "base": {"mean": 15.0, "std": 3.0},
        "tide_effect": 0.0,
    },
    "zone_temp_offset": {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0},
    "storm_transition": [[0.5, 0.5], [0.5, 0.5]],
    "wind_transition": [[0.25] * 4 for _ in range(4)],
    "equip_transition": [[0.2] * 5 for _ in range(5)],
    "tide_transition": [[0.5, 0.5], [0.5, 0.5]],
}


class PomdpCodingAgent(LLMSolverAgent):
    """LLMSolverAgent + Algorithm 2 refinement loop (simplified)."""

    def __init__(self, llm_fn, config=None, M=3, k_failures=10):
        super().__init__(llm_fn=llm_fn, config=config)
        self.M = M
        self.k_failures = k_failures

        # Telemetry — consumed by the benchmark runner.
        self.coverage_history = []
        self.init_coverage = None
        self.final_coverage = None
        self.n_refinements_applied = 0  # times a refinement patch beat the best
        self.n_refinement_attempts = 0  # total LLM_Refine calls made
        self.init_parse_failed = False
        self.n_llm_calls = 0

    def reset(self):
        super().reset()
        self.coverage_history = []
        self.init_coverage = None
        self.final_coverage = None
        self.n_refinements_applied = 0
        self.n_refinement_attempts = 0
        self.init_parse_failed = False
        self.n_llm_calls = 0

    # Wrap llm_fn so every call is counted.
    def _llm_call(self, messages):
        self.n_llm_calls += 1
        return self.llm_fn(messages)

    def _learn_from_history(self, env):
        catch_rows, sensor_rows = self._fetch_history_rows(env)
        D = self._index_sensors_by_day(sensor_rows)
        print(f"[PomdpCoder] Got {len(catch_rows)} catch rows, "
              f"{len(sensor_rows)} sensor rows", flush=True)

        best_patch = self._llm_init(catch_rows, sensor_rows)
        best_score = self._coverage(best_patch, D)
        self.init_coverage = best_score
        self.coverage_history = [best_score]
        print(f"[PomdpCoder] Init coverage = {best_score:.4f} "
              f"(parse_failed={self.init_parse_failed})", flush=True)

        if D and self.M > 0:
            for iteration in range(1, self.M + 1):
                failures = self._top_failures(best_patch, D, self.k_failures)
                if not failures:
                    break
                self.n_refinement_attempts += 1
                new_patch = self._llm_refine(best_patch, failures)
                if new_patch is None:
                    print(f"[PomdpCoder] Refine {iteration}: malformed — "
                          f"keep best", flush=True)
                    continue
                score = self._coverage(new_patch, D)
                print(f"[PomdpCoder] Refine {iteration}: cov {score:.4f} "
                      f"(best {best_score:.4f})", flush=True)
                if score > best_score:
                    best_patch, best_score = new_patch, score
                    self.coverage_history.append(score)
                    self.n_refinements_applied += 1
        else:
            reason = "M=0 (ablation)" if self.M == 0 else "no historical data"
            print(f"[PomdpCoder] Skipping refinement ({reason})", flush=True)

        self.final_coverage = best_score
        learned = copy.deepcopy(self.cfg)
        _deep_merge(learned, best_patch)
        self._learned_config = learned
        self._parsed_config_patch = best_patch
        self.pomdp = FishingPOMDP(learned)
        self.belief = np.array(learned["initial_belief"], dtype=np.float64)
        self._learned = True

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def _fetch_history_rows(self, env):
        catch_rows = env.query_fishing_log(
            "SELECT day, zone, boats, reward FROM catch_history "
            "WHERE day < 0 ORDER BY day"
        )
        if isinstance(catch_rows, dict) and "error" in catch_rows:
            catch_rows = []
        sensor_rows = env.query_maintenance_log(
            "SELECT sl.day, sl.zone, sl.buoy_reading, sl.equipment_reading, "
            "sl.water_temp, ml.alerts "
            "FROM sensor_log sl JOIN maintenance_log ml "
            "ON sl.day=ml.day AND sl.zone=ml.zone "
            "WHERE sl.day < 0 ORDER BY sl.day, sl.zone"
        )
        if isinstance(sensor_rows, dict) and "error" in sensor_rows:
            sensor_rows = []
        return catch_rows, sensor_rows

    def _index_sensors_by_day(self, sensor_rows):
        by_day = {}
        for row in sensor_rows:
            day = row["day"]
            zone = row["zone"]
            by_day.setdefault(day, []).extend([
                (("buoy", zone),               float(row["buoy_reading"])),
                (("equip_inspection", zone),   float(row["equipment_reading"])),
                (("water_temp", zone),         float(row["water_temp"])),
                (("maintenance_alerts", zone), int(row["alerts"])),
            ])
        return by_day

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _llm_init(self, catch_rows, sensor_rows):
        data_text = _format_data_tables(catch_rows, sensor_rows)
        messages = [
            {"role": "system", "content": ESTIMATION_PROMPT},
            {"role": "user",   "content": data_text},
        ]
        print("[PomdpCoder] LLM_Init...", flush=True)
        response = self._llm_call(messages)
        self._raw_model_response = response

        patch = _parse_config_patch(response)
        missing = _REQUIRED_PATCH_KEYS - set(patch)
        if missing:
            print(f"[PomdpCoder] LLM_Init missing {sorted(missing)} — "
                  f"using neutral prior (flagged in telemetry)", flush=True)
            self.init_parse_failed = True
            return copy.deepcopy(_NEUTRAL_PRIOR_PATCH)
        _print_estimated_params(patch)
        return patch

    def _llm_refine(self, prev_patch, failures):
        failure_text = "\n".join(
            self._format_failure_line(marginal, day, obs_type, value)
            for (marginal, day, obs_type, value) in failures
        )
        prompt = REFINEMENT_PROMPT.format(
            previous_patch_json=json.dumps(prev_patch, indent=2, default=str),
            failure_cases=failure_text,
        )
        response = self._llm_call([{"role": "user", "content": prompt}])
        self._raw_model_response = response
        patch = _parse_config_patch(response)
        if not _REQUIRED_PATCH_KEYS.issubset(patch):
            return None
        return patch

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _likelihood_vector(self, pomdp, obs_type, value):
        like = np.zeros(pomdp.n_states, dtype=np.float64)
        if isinstance(obs_type, tuple):
            kind, zone = obs_type
            for i in range(pomdp.n_states):
                if kind == "buoy":
                    like[i] = pomdp._obs_likelihood_buoy(value, i, zone)
                elif kind == "equip_inspection":
                    like[i] = pomdp._obs_likelihood_equip_inspection(value, i, zone)
                elif kind == "water_temp":
                    like[i] = pomdp._obs_likelihood_water_temp(value, i, zone)
                elif kind == "maintenance_alerts":
                    like[i] = pomdp._obs_likelihood_maintenance(value, i, zone)
                else:
                    raise ValueError(f"Unknown obs kind: {kind}")
        else:
            for i in range(pomdp.n_states):
                if obs_type == "sea_color":
                    like[i] = pomdp._obs_likelihood_sea_color(value, i)
                elif obs_type == "equip_indicator":
                    like[i] = pomdp._obs_likelihood_equip_indicator(value, i)
                elif obs_type == "barometer":
                    like[i] = pomdp._obs_likelihood_barometer(value, i)
                else:
                    raise ValueError(f"Unknown obs type: {obs_type}")
        return like

    def _coverage(self, patch, D):
        try:
            merged = copy.deepcopy(self.cfg)
            _deep_merge(merged, patch)
            pomdp = FishingPOMDP(merged)
        except Exception:
            return -float("inf")

        belief = np.array(merged["initial_belief"], dtype=np.float64)
        total_logp = 0.0
        n_obs = 0
        for day in sorted(D.keys()):
            belief = pomdp.predict(belief)
            for obs_type, value in D[day]:
                like = self._likelihood_vector(pomdp, obs_type, value)
                marginal = float(belief @ like)
                total_logp += math.log(max(marginal, _EPS))
                n_obs += 1
                post = belief * like
                total = post.sum()
                belief = post / total if total > 0 else np.ones_like(belief) / len(belief)
        return total_logp / n_obs if n_obs > 0 else -float("inf")

    def _top_failures(self, patch, D, k):
        merged = copy.deepcopy(self.cfg)
        _deep_merge(merged, patch)
        pomdp = FishingPOMDP(merged)
        belief = np.array(merged["initial_belief"], dtype=np.float64)
        scored = []
        for day in sorted(D.keys()):
            belief = pomdp.predict(belief)
            for obs_type, value in D[day]:
                like = self._likelihood_vector(pomdp, obs_type, value)
                marginal = float(belief @ like)
                scored.append((marginal, day, obs_type, value))
                post = belief * like
                total = post.sum()
                belief = post / total if total > 0 else np.ones_like(belief) / len(belief)
        scored.sort(key=lambda t: t[0])
        return scored[:k]

    @staticmethod
    def _format_failure_line(marginal, day, obs_type, value):
        label = (f"{obs_type[0]}[{obs_type[1]}]"
                 if isinstance(obs_type, tuple) else str(obs_type))
        if isinstance(value, float):
            return f"Day {day}, {label} = {value:.3f}  (marginal ≈ {marginal:.2e})"
        return f"Day {day}, {label} = {value}  (marginal ≈ {marginal:.2e})"
