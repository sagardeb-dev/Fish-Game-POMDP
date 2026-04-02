"""
Evaluator for the Fishing Game POMDP v5.

Takes POMDP model + episode trace. Computes Bayesian posterior at each step,
Brier scores (storm, storm_zone, equip, equip_zone), detection lags for both
risks, 3-way cost decomposition.

Critical invariant: decomposition identity holds at every step:
  tool_use_gap + inference_gap + planning_gap = oracle_reward - actual_reward
"""

import numpy as np
from fishing_game.pomdp import FishingPOMDP
from fishing_game.config import CONFIG


class Evaluator:
    def __init__(self, config=None):
        self.cfg = config or CONFIG
        self.pomdp = FishingPOMDP(self.cfg)

    def evaluate_episode(self, trace):
        """
        Evaluate a full episode trace.

        Returns dict with per-step and episode-level metrics.
        """
        belief = np.array(self.cfg["initial_belief"], dtype=np.float64)
        step_results = []
        storm_onsets = []
        equip_onsets = []
        prev_storm = False
        prev_equip = False

        for step in trace:
            day = step["day"]
            hidden_state_idx = step["hidden_state_idx"]
            hidden_state = step["hidden_state"]
            storm, wind, equip, tide = hidden_state
            storm_active = storm == 1
            equip_active = equip > 0
            storm_zone = self.cfg["wind_to_zone"][wind] if storm_active else None
            equip_zone = self.cfg["equip_to_zone"][equip]

            # Track storm onsets
            if storm_active and not prev_storm:
                storm_onsets.append({"onset_day": day, "detection_day": None})
            prev_storm = storm_active

            # Track equipment failure onsets
            if equip_active and not prev_equip:
                equip_onsets.append({"onset_day": day, "detection_day": None})
            prev_equip = equip_active

            # --- Prediction step ---
            if day > 1:
                belief = self.pomdp.predict(belief)

            # --- Oracle posterior (all available observations) ---
            available_obs = step["available_observations"]
            oracle_posterior = self.pomdp.belief_update(belief, available_obs)

            # --- Retrieved posterior (agent's actual observations) ---
            retrieved_obs = step["observations"]
            retrieved_posterior = self.pomdp.belief_update(belief, retrieved_obs)

            # --- Oracle reward: best action under full oracle posterior ---
            oracle_alloc, oracle_er = self.pomdp.optimal_action(oracle_posterior)
            oracle_reward = self.pomdp.reward(hidden_state_idx, oracle_alloc)

            # --- Retrieved oracle reward: best action under retrieved posterior ---
            ret_alloc, ret_er = self.pomdp.optimal_action(retrieved_posterior)
            retrieved_oracle_reward = self.pomdp.reward(hidden_state_idx, ret_alloc)

            # --- Belief-optimal reward: best action under agent's stated beliefs ---
            agent_beliefs = step["beliefs"]
            agent_belief_vec = self._agent_beliefs_to_vector(agent_beliefs)
            belief_alloc, belief_er = self.pomdp.optimal_action(agent_belief_vec)
            belief_optimal_reward = self.pomdp.reward(hidden_state_idx, belief_alloc)

            # --- Actual reward ---
            actual_reward = step["reward"]

            # --- Cost decomposition ---
            tool_use_gap = oracle_reward - retrieved_oracle_reward
            inference_gap = retrieved_oracle_reward - belief_optimal_reward
            planning_gap = belief_optimal_reward - actual_reward

            # --- Brier scores ---
            agent_p_storm = agent_beliefs.get("storm_active", 0.5)
            agent_storm_zone_probs = agent_beliefs.get(
                "storm_zone_probs", {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
            )
            agent_p_equip = agent_beliefs.get("equip_failure_active", 0.2)
            agent_equip_zone_probs = agent_beliefs.get(
                "equip_zone_probs", {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
            )

            # Storm active Brier
            true_storm = 1.0 if storm_active else 0.0
            brier_storm = (agent_p_storm - true_storm) ** 2

            # Storm zone Brier (4-way, only meaningful if storm active)
            brier_storm_zone = 0.0
            for z in self.cfg["zones"]:
                true_val = 1.0 if (storm_active and storm_zone == z) else 0.0
                pred_val = agent_p_storm * agent_storm_zone_probs.get(z, 0.25)
                brier_storm_zone += (pred_val - true_val) ** 2

            # Equip active Brier
            true_equip = 1.0 if equip_active else 0.0
            brier_equip = (agent_p_equip - true_equip) ** 2

            # Equip zone Brier (4-way)
            brier_equip_zone = 0.0
            for z in self.cfg["zones"]:
                true_val = 1.0 if (equip_active and equip_zone == z) else 0.0
                pred_val = agent_p_equip * agent_equip_zone_probs.get(z, 0.25)
                brier_equip_zone += (pred_val - true_val) ** 2

            # --- Detection lag tracking ---
            if agent_p_storm > 0.5:
                for onset in storm_onsets:
                    if onset["detection_day"] is None:
                        onset["detection_day"] = day

            if agent_p_equip > 0.5:
                for onset in equip_onsets:
                    if onset["detection_day"] is None:
                        onset["detection_day"] = day

            # --- Advance belief ---
            belief = retrieved_posterior.copy()

            step_result = {
                "day": day,
                "oracle_reward": oracle_reward,
                "retrieved_oracle_reward": retrieved_oracle_reward,
                "belief_optimal_reward": belief_optimal_reward,
                "actual_reward": actual_reward,
                "tool_use_gap": tool_use_gap,
                "inference_gap": inference_gap,
                "planning_gap": planning_gap,
                "brier_storm": brier_storm,
                "brier_storm_zone": brier_storm_zone,
                "brier_equip": brier_equip,
                "brier_equip_zone": brier_equip_zone,
                "bayesian_posterior": oracle_posterior.tolist(),
                "retrieved_posterior": retrieved_posterior.tolist(),
                "optimal_action_under_posterior": oracle_alloc,
                "tools_used": step["tools_used"],
            }
            step_results.append(step_result)

        # --- Episode-level metrics ---
        total_reward = sum(s["actual_reward"] for s in step_results)
        mean_brier_storm = float(np.mean([s["brier_storm"] for s in step_results]))
        mean_brier_storm_zone = float(np.mean([s["brier_storm_zone"] for s in step_results]))
        mean_brier_equip = float(np.mean([s["brier_equip"] for s in step_results]))
        mean_brier_equip_zone = float(np.mean([s["brier_equip_zone"] for s in step_results]))

        # Storm detection lag
        storm_lags = []
        for onset in storm_onsets:
            if onset["detection_day"] is not None:
                storm_lags.append(onset["detection_day"] - onset["onset_day"])
            else:
                storm_lags.append(float("inf"))
        mean_storm_detection_lag = float(np.mean(storm_lags)) if storm_lags else float("inf")

        # Equip detection lag
        equip_lags = []
        for onset in equip_onsets:
            if onset["detection_day"] is not None:
                equip_lags.append(onset["detection_day"] - onset["onset_day"])
            else:
                equip_lags.append(float("inf"))
        mean_equip_detection_lag = float(np.mean(equip_lags)) if equip_lags else float("inf")

        total_tool_use_gap = sum(s["tool_use_gap"] for s in step_results)
        total_inference_gap = sum(s["inference_gap"] for s in step_results)
        total_planning_gap = sum(s["planning_gap"] for s in step_results)

        # Tool usage counts
        tool_counts = {t: 0 for t in self.cfg["tool_budgets"]}
        for s in step_results:
            for tool in s["tools_used"]:
                if tool in tool_counts:
                    tool_counts[tool] += 1

        # Reward per quarter
        quarters = [[], [], [], []]
        for s in step_results:
            q = min((s["day"] - 1) // 5, 3)
            quarters[q].append(s["actual_reward"])
        reward_per_quarter = [sum(q) for q in quarters]

        return {
            "step_results": step_results,
            "total_reward": total_reward,
            "mean_brier_storm": mean_brier_storm,
            "mean_brier_storm_zone": mean_brier_storm_zone,
            "mean_brier_equip": mean_brier_equip,
            "mean_brier_equip_zone": mean_brier_equip_zone,
            "mean_storm_detection_lag": mean_storm_detection_lag,
            "mean_equip_detection_lag": mean_equip_detection_lag,
            "mean_detection_lag": mean_storm_detection_lag,  # backward compat alias
            "total_tool_use_gap": total_tool_use_gap,
            "total_inference_gap": total_inference_gap,
            "total_planning_gap": total_planning_gap,
            "tool_usage_counts": tool_counts,
            "reward_per_quarter": reward_per_quarter,
            "storm_onsets": storm_onsets,
            "equip_onsets": equip_onsets,
        }

    def _agent_beliefs_to_vector(self, beliefs):
        """
        Convert agent's marginal belief dict to an 80-element belief vector
        under independence assumption.

        beliefs = {
            "storm_active": float,
            "storm_zone_probs": {"A": f, "B": f, "C": f, "D": f},
            "equip_failure_active": float,
            "equip_zone_probs": {"A": f, "B": f, "C": f, "D": f},
            "tide_high": float (optional, defaults to 0.5),
        }
        """
        p_storm = max(0.0, min(1.0, beliefs.get("storm_active", 0.5)))
        storm_zone_probs = beliefs.get(
            "storm_zone_probs", {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        )
        p_equip = max(0.0, min(1.0, beliefs.get("equip_failure_active", 0.2)))
        equip_zone_probs = beliefs.get(
            "equip_zone_probs", {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        )
        p_tide_high = max(0.0, min(1.0, beliefs.get("tide_high", 0.5)))

        # Normalize zone probs
        szp_sum = sum(storm_zone_probs.get(z, 0.25) for z in self.cfg["zones"])
        ezp_sum = sum(equip_zone_probs.get(z, 0.25) for z in self.cfg["zones"])

        belief = np.zeros(self.pomdp.n_states, dtype=np.float64)
        for i, (storm, wind, equip, tide) in enumerate(self.cfg["states"]):
            # Storm component
            p_s = p_storm if storm == 1 else (1.0 - p_storm)

            # Wind component (conditional on storm_zone_probs)
            zone_for_wind = self.cfg["wind_to_zone"][wind]
            if szp_sum > 0:
                p_w = storm_zone_probs.get(zone_for_wind, 0.25) / szp_sum
            else:
                p_w = 0.25

            # Equipment component
            if equip == 0:
                p_e = 1.0 - p_equip
            else:
                equip_zone = self.cfg["equip_to_zone"][equip]
                if ezp_sum > 0:
                    p_e = p_equip * (equip_zone_probs.get(equip_zone, 0.25) / ezp_sum)
                else:
                    p_e = p_equip * 0.25

            # Tide component
            p_t = p_tide_high if tide == 1 else (1.0 - p_tide_high)

            belief[i] = p_s * p_w * p_e * p_t

        total = belief.sum()
        if total > 0:
            belief /= total
        else:
            belief = np.ones(self.pomdp.n_states, dtype=np.float64) / self.pomdp.n_states
        return belief
