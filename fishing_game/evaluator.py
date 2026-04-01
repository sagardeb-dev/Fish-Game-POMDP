"""
Evaluator for the Fishing Game POMDP.

Takes POMDP model + episode trace. Computes Bayesian posterior at each step,
Brier scores, detection lag, 3-way cost decomposition
(tool_use_gap + inference_gap + planning_gap).

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

        Args:
            trace: list of step dicts from env.get_trace()

        Returns:
            dict with per-step and episode-level metrics
        """
        belief = np.array(self.cfg["initial_belief"], dtype=np.float64)
        step_results = []
        storm_onsets = []  # (day, detection_day) pairs
        prev_storm = False

        for step in trace:
            day = step["day"]
            hidden_state_idx = step["hidden_state_idx"]
            hidden_state = step["hidden_state"]
            storm_active = hidden_state[0] == 1
            wind = hidden_state[1]
            affected_zone = "A" if wind == "N" else "B"

            # Track storm onsets
            if storm_active and not prev_storm:
                storm_onsets.append({"onset_day": day, "detection_day": None})
            prev_storm = storm_active

            # --- Prediction step (transition) ---
            if day > 1:
                belief = self.pomdp.predict(belief)

            # --- Compute oracle posterior (all available observations) ---
            available_obs = step["available_observations"]
            oracle_posterior = self.pomdp.belief_update(belief, available_obs)

            # --- Compute retrieved posterior (only observations the agent got) ---
            retrieved_obs = step["observations"]
            retrieved_posterior = self.pomdp.belief_update(belief, retrieved_obs)

            # --- Oracle reward: best action under full oracle posterior ---
            oracle_zone, oracle_boats, oracle_er = self.pomdp.optimal_action(oracle_posterior)
            oracle_reward = self.pomdp.reward(hidden_state_idx, oracle_zone, oracle_boats)

            # --- Retrieved oracle reward: best action under retrieved posterior ---
            ret_zone, ret_boats, ret_er = self.pomdp.optimal_action(retrieved_posterior)
            retrieved_oracle_reward = self.pomdp.reward(hidden_state_idx, ret_zone, ret_boats)

            # --- Belief-optimal reward: best action under AGENT's stated beliefs ---
            agent_beliefs = step["beliefs"]
            agent_p_storm = agent_beliefs.get("storm_active", 0.5)
            agent_p_zone_a = agent_beliefs.get("zone_a_is_dangerous", 0.5)

            # Convert agent beliefs to a 4-state belief vector
            agent_belief_vec = self._agent_beliefs_to_vector(agent_p_storm, agent_p_zone_a)
            belief_zone, belief_boats, belief_er = self.pomdp.optimal_action(agent_belief_vec)
            belief_optimal_reward = self.pomdp.reward(hidden_state_idx, belief_zone, belief_boats)

            # --- Actual reward ---
            action = step["action"]
            actual_reward = step["reward"]

            # --- Cost decomposition ---
            tool_use_gap = oracle_reward - retrieved_oracle_reward
            inference_gap = retrieved_oracle_reward - belief_optimal_reward
            planning_gap = belief_optimal_reward - actual_reward

            # --- Brier scores (against ground truth) ---
            true_storm = 1.0 if storm_active else 0.0
            true_zone_a = 1.0 if (storm_active and affected_zone == "A") else 0.0

            brier_storm = (agent_p_storm - true_storm) ** 2
            brier_zone = (agent_p_zone_a - true_zone_a) ** 2

            # --- Detection lag tracking ---
            if agent_p_storm > 0.5:
                for onset in storm_onsets:
                    if onset["detection_day"] is None:
                        onset["detection_day"] = day

            # --- Advance belief to retrieved posterior for next step ---
            # The evaluator tracks belief based on what the agent actually observed
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
                "brier_zone": brier_zone,
                "bayesian_posterior": oracle_posterior.tolist(),
                "retrieved_posterior": retrieved_posterior.tolist(),
                "optimal_action_under_posterior": {
                    "zone": oracle_zone, "boats": oracle_boats
                },
                "tools_used": step["tools_used"],
            }
            step_results.append(step_result)

        # --- Episode-level metrics ---
        total_reward = sum(s["actual_reward"] for s in step_results)
        mean_brier_storm = np.mean([s["brier_storm"] for s in step_results])
        mean_brier_zone = np.mean([s["brier_zone"] for s in step_results])

        # Detection lag
        detection_lags = []
        for onset in storm_onsets:
            if onset["detection_day"] is not None:
                lag = onset["detection_day"] - onset["onset_day"]
                detection_lags.append(lag)
            else:
                detection_lags.append(float("inf"))
        mean_detection_lag = np.mean(detection_lags) if detection_lags else float("inf")

        total_tool_use_gap = sum(s["tool_use_gap"] for s in step_results)
        total_inference_gap = sum(s["inference_gap"] for s in step_results)
        total_planning_gap = sum(s["planning_gap"] for s in step_results)

        # Tool usage counts
        tool_counts = {"check_weather_reports": 0, "query_fishing_log": 0,
                       "analyze_data": 0, "evaluate_options": 0,
                       "forecast_scenario": 0, "read_barometer": 0,
                       "read_buoy": 0}
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
            "mean_brier_storm": float(mean_brier_storm),
            "mean_brier_zone": float(mean_brier_zone),
            "mean_detection_lag": float(mean_detection_lag),
            "total_tool_use_gap": total_tool_use_gap,
            "total_inference_gap": total_inference_gap,
            "total_planning_gap": total_planning_gap,
            "tool_usage_counts": tool_counts,
            "reward_per_quarter": reward_per_quarter,
            "storm_onsets": storm_onsets,
        }

    def _agent_beliefs_to_vector(self, p_storm, p_zone_a_dangerous):
        """
        Convert agent's 2-value belief to a 4-state belief vector.

        p_storm = P(storm=1) = belief[2] + belief[3]
        p_zone_a_dangerous = P(storm=1 AND wind=N) = belief[2]

        So:
          belief[2] = p_zone_a_dangerous
          belief[3] = p_storm - p_zone_a_dangerous
          Remaining (1 - p_storm) split: we need P(wind=N | no storm)

        Since we don't know the agent's wind prior, we split no-storm
        proportionally to maintain consistency.
        """
        p_storm = max(0.0, min(1.0, p_storm))
        p_zone_a_dangerous = max(0.0, min(p_storm, p_zone_a_dangerous))

        b2 = p_zone_a_dangerous                    # P(storm=1, wind=N)
        b3 = p_storm - p_zone_a_dangerous           # P(storm=1, wind=S)
        p_no_storm = 1.0 - p_storm

        # Split no-storm proportionally to wind ratio from storm beliefs
        if p_storm > 0:
            wind_n_ratio = p_zone_a_dangerous / p_storm
        else:
            wind_n_ratio = 0.5
        b0 = p_no_storm * wind_n_ratio              # P(storm=0, wind=N)
        b1 = p_no_storm * (1.0 - wind_n_ratio)      # P(storm=0, wind=S)

        belief = np.array([b0, b1, b2, b3], dtype=np.float64)
        total = belief.sum()
        if total > 0:
            belief /= total
        return belief
