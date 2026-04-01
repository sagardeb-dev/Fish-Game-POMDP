"""
Single test file for the entire fishing game.
Tests are added incrementally as components are built.
"""

import math
import numpy as np
import pytest
from fishing_game.config import CONFIG
from fishing_game.pomdp import FishingPOMDP, _normal_pdf
from fishing_game.simulator import FishingGameEnv
from fishing_game.evaluator import Evaluator
from fishing_game.baselines import (
    RandomAgent, NoToolsHeuristic, SearchOnlyHeuristic,
    BeliefAwareBaseline, OracleAgent,
)


# =============================================================================
# POMDP Model Tests
# =============================================================================

class TestConfig:
    """Verify config is well-formed."""

    def test_transition_matrix_rows_sum_to_one(self):
        T = np.array(CONFIG["transition_matrix"])
        for i, row in enumerate(T):
            assert abs(sum(row) - 1.0) < 1e-10, f"Row {i} sums to {sum(row)}"

    def test_sea_color_probs_sum_to_one(self):
        for storm, probs in CONFIG["sea_color_probs"].items():
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-10, f"storm={storm} sums to {total}"

    def test_four_states(self):
        assert len(CONFIG["states"]) == 4

    def test_initial_belief_sums_to_one(self):
        assert abs(sum(CONFIG["initial_belief"]) - 1.0) < 1e-10


class TestPOMDPModel:
    """Test the POMDP model: belief_update, reward, optimal_action."""

    @pytest.fixture
    def pomdp(self):
        return FishingPOMDP()

    def test_states(self, pomdp):
        assert pomdp.states == [(0, "N"), (0, "S"), (1, "N"), (1, "S")]

    def test_transition_matrix_shape(self, pomdp):
        assert pomdp.T.shape == (4, 4)

    def test_affected_zone(self, pomdp):
        assert pomdp._affected_zone((0, "N")) == "A"
        assert pomdp._affected_zone((0, "S")) == "B"
        assert pomdp._affected_zone((1, "N")) == "A"
        assert pomdp._affected_zone((1, "S")) == "B"

    # --- Reward function tests ---
    def test_reward_safe(self, pomdp):
        # (0,N) = no storm, fishing zone A → safe
        assert pomdp.reward(0, "A", 3) == 21  # 7*3
        assert pomdp.reward(0, "B", 2) == 14  # 7*2

    def test_reward_danger(self, pomdp):
        # (1,N) = storm, wind=N, affected=A, fishing zone A → danger
        assert pomdp.reward(2, "A", 3) == -54  # -18*3
        # (1,S) = storm, wind=S, affected=B, fishing zone B → danger
        assert pomdp.reward(3, "B", 2) == -36  # -18*2

    def test_reward_storm_but_safe_zone(self, pomdp):
        # (1,N) = storm, affected=A, but fishing zone B → safe
        assert pomdp.reward(2, "B", 3) == 21  # 7*3
        # (1,S) = storm, affected=B, but fishing zone A → safe
        assert pomdp.reward(3, "A", 2) == 14  # 7*2

    # --- Belief update: hand-calculated tests ---
    def test_belief_update_sea_color_dark(self, pomdp):
        """
        Hand calculation: uniform prior [0.25, 0.25, 0.25, 0.25]
        Observe sea_color = "dark"
        P(dark | storm=0) = 0.05, P(dark | storm=1) = 0.60

        Likelihoods: [0.05, 0.05, 0.60, 0.60]
        Unnormalized: [0.0125, 0.0125, 0.15, 0.15]
        Sum = 0.325
        Posterior: [0.0125/0.325, 0.0125/0.325, 0.15/0.325, 0.15/0.325]
                 = [0.03846, 0.03846, 0.46154, 0.46154]
        """
        prior = np.array([0.25, 0.25, 0.25, 0.25])
        posterior = pomdp.belief_update(prior, [("sea_color", "dark")])

        expected = np.array([0.0125, 0.0125, 0.15, 0.15])
        expected = expected / expected.sum()

        np.testing.assert_allclose(posterior, expected, atol=1e-10)
        assert abs(pomdp.p_storm(posterior) - (0.15 + 0.15) / 0.325) < 1e-10

    def test_belief_update_sea_color_green(self, pomdp):
        """
        Uniform prior, observe green.
        P(green | storm=0) = 0.70, P(green | storm=1) = 0.05

        Likelihoods: [0.70, 0.70, 0.05, 0.05]
        Unnormalized: [0.175, 0.175, 0.0125, 0.0125]
        Sum = 0.375
        """
        prior = np.array([0.25, 0.25, 0.25, 0.25])
        posterior = pomdp.belief_update(prior, [("sea_color", "green")])

        expected = np.array([0.175, 0.175, 0.0125, 0.0125])
        expected = expected / expected.sum()

        np.testing.assert_allclose(posterior, expected, atol=1e-10)
        # P(storm) should be low after seeing green
        assert pomdp.p_storm(posterior) < 0.1

    def test_belief_update_multiple_observations(self, pomdp):
        """
        Sequential update: uniform → observe dark → observe barometer=998.

        Step 1: dark → posterior_1 as computed above
        Step 2: barometer=998 given posterior_1

        For barometer=998:
          P(998 | storm=0) = N(998; 1013, 3) ≈ very small
          P(998 | storm=1) = N(998; 998, 5) ≈ 0.0798

        This should push belief strongly toward storm=1.
        """
        prior = np.array([0.25, 0.25, 0.25, 0.25])
        posterior = pomdp.belief_update(
            prior, [("sea_color", "dark"), ("barometer", 998.0)]
        )

        # After dark + barometer=998, storm probability should be very high
        p_storm = pomdp.p_storm(posterior)
        assert p_storm > 0.99, f"P(storm) should be >0.99 after dark+baro=998, got {p_storm}"
        assert abs(posterior.sum() - 1.0) < 1e-10

    def test_belief_update_barometer_only(self, pomdp):
        """
        Hand calculation: uniform prior, observe barometer=1013.
        P(1013 | storm=0) = N(1013; 1013, 3) = 1/(3*sqrt(2pi))
        P(1013 | storm=1) = N(1013; 998, 5) = very small (3 sigma away)

        Should strongly favor storm=0.
        """
        prior = np.array([0.25, 0.25, 0.25, 0.25])
        posterior = pomdp.belief_update(prior, [("barometer", 1013.0)])

        p_storm = pomdp.p_storm(posterior)
        assert p_storm < 0.01, f"P(storm) should be <0.01 after baro=1013, got {p_storm}"

    def test_belief_update_buoy_high_in_zone_a(self, pomdp):
        """
        Uniform prior, observe buoy reading 4.0 in zone A.

        For each state:
          (0,N): affected=A, storm=0 → normal params (1.2, 0.3) → P(4.0) ≈ 0
          (0,S): affected=B, storm=0 → normal params (1.2, 0.3) → P(4.0) ≈ 0
          (1,N): affected=A, storm=1 → danger params (4.0, 0.5) → P(4.0) = high
          (1,S): affected=B, storm=1 → buoy in A, affected=B → normal (1.2, 0.3) → P(4.0) ≈ 0

        Should strongly concentrate on state (1,N).
        """
        prior = np.array([0.25, 0.25, 0.25, 0.25])
        posterior = pomdp.belief_update(prior, [(("buoy", "A"), 4.0)])

        assert posterior[2] > 0.99, f"State (1,N) should dominate, got {posterior[2]}"
        assert abs(posterior.sum() - 1.0) < 1e-10

    def test_belief_update_preserves_normalization(self, pomdp):
        """Any sequence of observations should yield a normalized posterior."""
        prior = np.array([0.1, 0.3, 0.2, 0.4])
        posterior = pomdp.belief_update(
            prior,
            [("sea_color", "murky"), ("barometer", 1005.0), (("buoy", "B"), 2.0)],
        )
        assert abs(posterior.sum() - 1.0) < 1e-10
        assert all(p >= 0 for p in posterior)

    def test_belief_update_prediction_step(self, pomdp):
        """Test that predict() applies the transition matrix correctly."""
        # Start concentrated in state 0 = (0,N)
        belief = np.array([1.0, 0.0, 0.0, 0.0])
        predicted = pomdp.predict(belief)

        # Should equal first row of T
        np.testing.assert_allclose(predicted, pomdp.T[0], atol=1e-10)

    def test_belief_update_sequence_with_prediction(self, pomdp):
        """
        Full cycle: predict → update.
        Start in (0,N), predict, then observe dark.
        """
        belief = np.array([1.0, 0.0, 0.0, 0.0])
        predicted = pomdp.predict(belief)
        posterior = pomdp.belief_update(predicted, [("sea_color", "dark")])

        assert abs(posterior.sum() - 1.0) < 1e-10
        # Dark observation should shift toward storm states
        assert pomdp.p_storm(posterior) > pomdp.p_storm(predicted)

    # --- Optimal action tests ---
    def test_optimal_action_certain_no_storm(self, pomdp):
        """If certain no storm, should fish with max boats (any zone is safe)."""
        belief = np.array([0.5, 0.5, 0.0, 0.0])  # certain no storm
        zone, boats, er = pomdp.optimal_action(belief)
        assert boats == 3
        assert er == 21.0  # 7 * 3

    def test_optimal_action_certain_storm_north(self, pomdp):
        """If certain storm + wind=N, affected=A. Should fish B with 3 boats."""
        belief = np.array([0.0, 0.0, 1.0, 0.0])  # certain (1,N)
        zone, boats, er = pomdp.optimal_action(belief)
        assert zone == "B"
        assert boats == 3
        assert er == 21.0  # safe zone: 7 * 3

    def test_optimal_action_certain_storm_south(self, pomdp):
        """If certain storm + wind=S, affected=B. Should fish A with 3 boats."""
        belief = np.array([0.0, 0.0, 0.0, 1.0])  # certain (1,S)
        zone, boats, er = pomdp.optimal_action(belief)
        assert zone == "A"
        assert boats == 3
        assert er == 21.0

    def test_optimal_action_high_uncertainty(self, pomdp):
        """
        High storm probability but uncertain wind → should reduce boats.
        belief = [0.0, 0.0, 0.5, 0.5] means storm certain, wind 50/50.

        For zone A, boats b:
          E[R] = 0.5 * 7*b + 0.5 * (-18*b) = 0.5*b*(7-18) = -5.5*b
        For zone B, boats b:
          E[R] = 0.5 * (-18*b) + 0.5 * 7*b = -5.5*b

        Both zones give negative expected reward. Best is 1 boat (least loss).
        """
        belief = np.array([0.0, 0.0, 0.5, 0.5])
        zone, boats, er = pomdp.optimal_action(belief)
        assert boats == 1
        assert abs(er - (-5.5)) < 1e-10

    def test_p_storm_and_p_zone(self, pomdp):
        """Test convenience methods for extracting marginals."""
        belief = np.array([0.1, 0.2, 0.3, 0.4])
        assert abs(pomdp.p_storm(belief) - 0.7) < 1e-10
        # P(zone_A dangerous) = P(storm=1, wind=N) = belief[2]
        assert abs(pomdp.p_zone_a_dangerous(belief) - 0.3) < 1e-10


class TestHandCalculatedBeliefSequence:
    """
    Complete hand-calculated belief update sequence to validate
    the POMDP model end-to-end.

    Scenario: 3-step sequence with specific observations.
    """

    @pytest.fixture
    def pomdp(self):
        return FishingPOMDP()

    def test_three_step_belief_sequence(self, pomdp):
        """
        Day 1: Start uniform [0.25, 0.25, 0.25, 0.25]
               Observe sea_color = "green"

        Day 2: Predict with transition matrix, then observe "murky"

        Day 3: Predict, then observe "dark" + barometer=1000

        Verify each posterior by hand.
        """
        # --- Day 1 ---
        b0 = np.array([0.25, 0.25, 0.25, 0.25])

        # sea_color=green: L = [0.70, 0.70, 0.05, 0.05]
        # unnorm = [0.175, 0.175, 0.0125, 0.0125] sum=0.375
        b1 = pomdp.belief_update(b0, [("sea_color", "green")])
        expected_b1 = np.array([0.175, 0.175, 0.0125, 0.0125])
        expected_b1 /= expected_b1.sum()
        np.testing.assert_allclose(b1, expected_b1, atol=1e-10)

        # --- Day 2: predict then update ---
        b1_pred = pomdp.predict(b1)
        # b1_pred = T^T @ b1 -- computed from transition matrix
        # Verify it's a valid distribution
        assert abs(b1_pred.sum() - 1.0) < 1e-10
        assert all(p >= 0 for p in b1_pred)

        # After green on day 1, storm is unlikely. Prediction should keep it low
        # but transition can introduce some storm probability
        b2 = pomdp.belief_update(b1_pred, [("sea_color", "murky")])
        assert abs(b2.sum() - 1.0) < 1e-10

        # --- Day 3: predict then update with strong storm evidence ---
        b2_pred = pomdp.predict(b2)
        b3 = pomdp.belief_update(b2_pred, [("sea_color", "dark"), ("barometer", 1000.0)])
        assert abs(b3.sum() - 1.0) < 1e-10

        # After dark + barometer near storm mean, should strongly indicate storm
        assert pomdp.p_storm(b3) > 0.8, f"Expected high storm prob, got {pomdp.p_storm(b3)}"

    def test_belief_update_is_order_independent(self, pomdp):
        """
        Bayesian updates should be order-independent for observations
        at the same timestep (multiplication is commutative).
        """
        prior = np.array([0.25, 0.25, 0.25, 0.25])

        obs_a = [("sea_color", "dark"), ("barometer", 1000.0)]
        obs_b = [("barometer", 1000.0), ("sea_color", "dark")]

        post_a = pomdp.belief_update(prior, obs_a)
        post_b = pomdp.belief_update(prior, obs_b)

        np.testing.assert_allclose(post_a, post_b, atol=1e-10)


# =============================================================================
# Simulator Tests
# =============================================================================

class TestSimulator:
    """Test the FishingGameEnv simulator."""

    def test_reset_returns_observation_bundle(self):
        env = FishingGameEnv()
        obs = env.reset(seed=42)
        assert obs["day"] == 1
        assert obs["days_remaining"] == 19
        assert obs["sea_color"] in ("green", "murky", "dark")
        assert obs["cumulative_reward"] == 0.0
        assert "check_weather_reports" in obs["tools_available"]
        assert obs["tool_budget"]["check_weather_reports"] == 2

    def test_same_seed_identical_trajectory(self):
        """Same seed must produce identical trajectory."""
        def run_episode(seed):
            env = FishingGameEnv()
            obs = env.reset(seed=seed)
            trajectory = [obs["sea_color"]]
            for day in range(20):
                result = env.submit_decisions(
                    zone="A", boats=1,
                    beliefs={"storm_active": 0.5, "zone_a_is_dangerous": 0.5},
                )
                if result["done"]:
                    break
                trajectory.append(result["observation"]["sea_color"])
                trajectory.append(result["reward"])
            return trajectory

        t1 = run_episode(42)
        t2 = run_episode(42)
        assert t1 == t2, "Same seed must produce identical trajectory"

    def test_submit_decisions_advances_day(self):
        env = FishingGameEnv()
        obs = env.reset(seed=42)
        assert obs["day"] == 1

        result = env.submit_decisions(
            zone="A", boats=2,
            beliefs={"storm_active": 0.5, "zone_a_is_dangerous": 0.5},
        )
        assert not result["done"]
        assert result["observation"]["day"] == 2

    def test_tool_budget_enforced(self):
        env = FishingGameEnv()
        env.reset(seed=42)

        # check_weather_reports has budget 2
        r1 = env.check_weather_reports("storm")
        assert not isinstance(r1, dict) or "error" not in r1
        r2 = env.check_weather_reports("weather")
        assert not isinstance(r2, dict) or "error" not in r2
        r3 = env.check_weather_reports("wind")
        assert isinstance(r3, dict) and "error" in r3

    def test_query_fishing_log_read_only(self):
        env = FishingGameEnv()
        env.reset(seed=42)

        # Valid SELECT
        result = env.query_fishing_log("SELECT * FROM daily_log")
        assert isinstance(result, list)

        # Invalid write attempt
        result = env.query_fishing_log("DROP TABLE daily_log")
        assert isinstance(result, dict) and "error" in result

        result = env.query_fishing_log("INSERT INTO daily_log VALUES (1, 'g', 'A', 1, 1, 1)")
        assert isinstance(result, dict) and "error" in result

    def test_analyze_data(self):
        env = FishingGameEnv()
        env.reset(seed=42)
        result = env.analyze_data("print(2 + 3)")
        assert "5" in result

    def test_evaluate_options(self):
        env = FishingGameEnv()
        env.reset(seed=42)
        result = env.evaluate_options({
            "storm_probability": 0.0,
            "zone_a_danger_probability": 0.0,
        })
        # No "optimal" key — agent must pick from expected_rewards
        assert "optimal" not in result
        assert "expected_rewards" in result
        assert len(result["expected_rewards"]) == 6
        # With no storm, all options should yield 7*boats
        for r in result["expected_rewards"]:
            assert r["expected_reward"] == 7.0 * r["boats"]

    def test_evaluate_options_clamping_warning(self):
        env = FishingGameEnv()
        env.reset(seed=42)
        result = env.evaluate_options({
            "storm_probability": 0.2,
            "zone_a_danger_probability": 0.5,
        })
        assert "note" in result
        assert "clamped" in result["note"]
        assert result["belief_used"]["zone_a_danger_probability"] == 0.2

    def test_forecast_scenario_does_not_advance(self):
        env = FishingGameEnv()
        obs = env.reset(seed=42)
        day_before = obs["day"]

        env.forecast_scenario({
            "horizon_days": 5,
            "assume_storm_persists": True,
            "assume_zone": "A",
            "strategy": {"zone": "B", "boats": 2},
        })

        # Day should not have advanced
        assert env._day == day_before

    def test_ablation_disables_tools(self):
        ablation = {
            "check_weather_reports": False,
            "query_fishing_log": True,
            "read_barometer": True,
            "read_buoy": True,
            "analyze_data": True,
            "evaluate_options": True,
            "forecast_scenario": True,
        }
        env = FishingGameEnv(ablation=ablation)
        obs = env.reset(seed=42)

        assert "check_weather_reports" not in obs["tools_available"]
        result = env.check_weather_reports("storm")
        assert isinstance(result, dict) and "error" in result

    def test_visible_db_has_no_hidden_state(self):
        """Verify the visible database contains no hidden state columns."""
        env = FishingGameEnv()
        env.reset(seed=42)

        # Submit one decision to populate DB
        env.submit_decisions(
            zone="A", boats=2,
            beliefs={"storm_active": 0.5, "zone_a_is_dangerous": 0.5},
        )

        cur = env._db.cursor()
        # Check all tables for hidden columns
        for table in ["daily_log", "weather_signals", "catch_history"]:
            cur.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cur.fetchall()]
            for col in columns:
                assert "_storm" not in col, f"Hidden column found: {col} in {table}"
                assert "_wind" not in col, f"Hidden column found: {col} in {table}"
                assert "_affected_zone" not in col, f"Hidden column found: {col} in {table}"
                assert "_tier" not in col, f"Hidden column found: {col} in {table}"

    def test_hidden_state_not_in_search_results(self):
        """check_weather_reports must not return hidden fields."""
        env = FishingGameEnv()
        env.reset(seed=42)
        results = env.check_weather_reports("storm warning weather severe")
        if isinstance(results, list):
            for r in results:
                assert "_storm" not in r
                assert "_wind" not in r
                assert "_affected_zone" not in r
                assert "_tier" not in r

    def test_episode_trace_recorded(self):
        env = FishingGameEnv()
        env.reset(seed=42)
        env.submit_decisions(
            zone="A", boats=2,
            beliefs={"storm_active": 0.5, "zone_a_is_dangerous": 0.5},
        )
        trace = env.get_trace()
        assert len(trace) == 1
        assert trace[0]["day"] == 1
        assert trace[0]["action"] == {"zone": "A", "boats": 2}

    def test_full_episode_completes(self):
        """Run a full 20-day episode and verify completion."""
        env = FishingGameEnv()
        env.reset(seed=42)
        for day in range(20):
            result = env.submit_decisions(
                zone="A", boats=1,
                beliefs={"storm_active": 0.5, "zone_a_is_dangerous": 0.5},
            )
            if result["done"]:
                assert day == 19, f"Episode ended early on day {day+1}"
                break
        trace = env.get_trace()
        assert len(trace) == 20

    def test_signals_stored_in_db(self):
        env = FishingGameEnv()
        env.reset(seed=42)
        cur = env._db.cursor()
        cur.execute("SELECT COUNT(*) FROM weather_signals WHERE day=1")
        count = cur.fetchone()[0]
        assert count >= 2, "Should emit at least 2 signals per step"

    def test_read_barometer(self):
        env = FishingGameEnv()
        env.reset(seed=42)
        reading = env.read_barometer()
        assert isinstance(reading, float)
        # Budget should be exhausted after 1 call
        result = env.read_barometer()
        assert isinstance(result, dict) and "error" in result

    def test_read_buoy(self):
        env = FishingGameEnv()
        env.reset(seed=42)
        reading_a = env.read_buoy("A")
        assert isinstance(reading_a, float)
        reading_b = env.read_buoy("B")
        assert isinstance(reading_b, float)
        # Budget 2, both exhausted
        result = env.read_buoy("A")
        assert isinstance(result, dict) and "error" in result

    def test_yesterday_in_observation(self):
        env = FishingGameEnv()
        obs = env.reset(seed=42)
        assert obs["yesterday_zone"] is None
        assert obs["yesterday_boats"] is None
        result = env.submit_decisions(
            zone="B", boats=2,
            beliefs={"storm_active": 0.5, "zone_a_is_dangerous": 0.5},
        )
        obs2 = result["observation"]
        assert obs2["yesterday_zone"] == "B"
        assert obs2["yesterday_boats"] == 2

    def test_daily_log_populated_after_submit(self):
        env = FishingGameEnv()
        env.reset(seed=42)
        env.submit_decisions(
            zone="B", boats=3,
            beliefs={"storm_active": 0.2, "zone_a_is_dangerous": 0.1},
        )
        result = env.query_fishing_log("SELECT * FROM daily_log WHERE day=1")
        assert len(result) == 1
        assert result[0]["zone_fished"] == "B"
        assert result[0]["boats_sent"] == 3


# =============================================================================
# Evaluator Tests
# =============================================================================

class TestEvaluator:
    """Test the evaluator and the critical decomposition identity."""

    def _run_episode_with_trace(self, seed=42):
        """Helper: run a full episode with constant beliefs, return trace."""
        env = FishingGameEnv()
        env.reset(seed=seed)
        for _ in range(20):
            result = env.submit_decisions(
                zone="A", boats=2,
                beliefs={"storm_active": 0.5, "zone_a_is_dangerous": 0.5},
            )
            if result["done"]:
                break
        return env.get_trace()

    def test_decomposition_identity_every_step(self):
        """
        CRITICAL INVARIANT: At every step:
        tool_use_gap + inference_gap + planning_gap = oracle_reward - actual_reward
        """
        trace = self._run_episode_with_trace(seed=42)
        evaluator = Evaluator()
        result = evaluator.evaluate_episode(trace)

        for step in result["step_results"]:
            gap_sum = step["tool_use_gap"] + step["inference_gap"] + step["planning_gap"]
            expected = step["oracle_reward"] - step["actual_reward"]
            assert abs(gap_sum - expected) < 1e-10, (
                f"Day {step['day']}: decomposition failed! "
                f"gaps={gap_sum}, oracle-actual={expected}, "
                f"tool={step['tool_use_gap']}, inf={step['inference_gap']}, "
                f"plan={step['planning_gap']}"
            )

    def test_decomposition_identity_multiple_seeds(self):
        """Decomposition must hold across different seeds."""
        evaluator = Evaluator()
        for seed in [1, 7, 42, 100, 999]:
            trace = self._run_episode_with_trace(seed=seed)
            result = evaluator.evaluate_episode(trace)
            for step in result["step_results"]:
                gap_sum = (step["tool_use_gap"] + step["inference_gap"]
                           + step["planning_gap"])
                expected = step["oracle_reward"] - step["actual_reward"]
                assert abs(gap_sum - expected) < 1e-10, (
                    f"Seed {seed}, Day {step['day']}: decomposition failed"
                )

    def test_episode_level_metrics(self):
        """Verify episode-level metrics are computed."""
        trace = self._run_episode_with_trace(seed=42)
        evaluator = Evaluator()
        result = evaluator.evaluate_episode(trace)

        assert "total_reward" in result
        assert "mean_brier_storm" in result
        assert "mean_brier_zone" in result
        assert "mean_detection_lag" in result
        assert "total_tool_use_gap" in result
        assert "total_inference_gap" in result
        assert "total_planning_gap" in result
        assert "tool_usage_counts" in result
        assert "reward_per_quarter" in result
        assert len(result["reward_per_quarter"]) == 4
        assert len(result["step_results"]) == 20

    def test_episode_level_decomposition_sums(self):
        """Total gaps should sum correctly across the episode too."""
        trace = self._run_episode_with_trace(seed=42)
        evaluator = Evaluator()
        result = evaluator.evaluate_episode(trace)

        total_oracle = sum(s["oracle_reward"] for s in result["step_results"])
        total_actual = sum(s["actual_reward"] for s in result["step_results"])
        total_gaps = (result["total_tool_use_gap"]
                      + result["total_inference_gap"]
                      + result["total_planning_gap"])

        assert abs(total_gaps - (total_oracle - total_actual)) < 1e-10

    def test_brier_scores_bounded(self):
        """Brier scores must be in [0, 1]."""
        trace = self._run_episode_with_trace(seed=42)
        evaluator = Evaluator()
        result = evaluator.evaluate_episode(trace)

        for step in result["step_results"]:
            assert 0.0 <= step["brier_storm"] <= 1.0
            assert 0.0 <= step["brier_zone"] <= 1.0


# =============================================================================
# Baseline Tests
# =============================================================================

def _run_baseline(agent_cls, seed, config=None):
    """Helper: run a baseline agent for a full episode, return (total_reward, trace)."""
    import random as stdlib_random

    cfg = config or CONFIG
    env = FishingGameEnv(config=cfg)
    obs = env.reset(seed=seed)
    agent = agent_cls(config=cfg)
    if hasattr(agent, "reset"):
        agent.reset()

    rng = stdlib_random.Random(seed + 1000)  # separate rng for agent choices
    total_reward = 0.0
    for _ in range(cfg["episode_length"]):
        result = agent.act(env, obs, rng=rng)
        total_reward += result["reward"]
        if result["done"]:
            break
        obs = result["observation"]

    return total_reward, env.get_trace()


class TestBaselines:
    """Test all 5 baselines and verify ordering."""

    def test_random_agent_runs(self):
        reward, trace = _run_baseline(RandomAgent, seed=42)
        assert len(trace) == 20

    def test_no_tools_heuristic_runs(self):
        reward, trace = _run_baseline(NoToolsHeuristic, seed=42)
        assert len(trace) == 20

    def test_search_only_heuristic_runs(self):
        reward, trace = _run_baseline(SearchOnlyHeuristic, seed=42)
        assert len(trace) == 20

    def test_belief_aware_runs(self):
        reward, trace = _run_baseline(BeliefAwareBaseline, seed=42)
        assert len(trace) == 20

    def test_oracle_agent_runs(self):
        reward, trace = _run_baseline(OracleAgent, seed=42)
        assert len(trace) == 20

    def test_oracle_always_safe(self):
        """Oracle should always get positive reward (7*3=21 per day)."""
        reward, trace = _run_baseline(OracleAgent, seed=42)
        assert reward == 21.0 * 20  # 420

    def test_oracle_brier_zero(self):
        """Oracle Brier scores must be 0.0."""
        reward, trace = _run_baseline(OracleAgent, seed=42)
        evaluator = Evaluator()
        result = evaluator.evaluate_episode(trace)
        for step in result["step_results"]:
            assert abs(step["brier_storm"]) < 1e-10, (
                f"Day {step['day']}: Oracle brier_storm={step['brier_storm']}"
            )

    def test_belief_aware_brier_low(self):
        """BeliefAware Brier scores should be < 0.05 on average."""
        reward, trace = _run_baseline(BeliefAwareBaseline, seed=42)
        evaluator = Evaluator()
        result = evaluator.evaluate_episode(trace)
        assert result["mean_brier_storm"] < 0.05, (
            f"BeliefAware mean_brier_storm={result['mean_brier_storm']}"
        )

    def test_baseline_ordering_seed_42(self):
        """
        Critical test: Random < NoTools < SearchOnly < BeliefAware < Oracle
        averaged over multiple seeds for stability.
        """
        agents = [
            ("Random", RandomAgent),
            ("NoTools", NoToolsHeuristic),
            ("SearchOnly", SearchOnlyHeuristic),
            ("BeliefAware", BeliefAwareBaseline),
            ("Oracle", OracleAgent),
        ]

        seeds = [42, 123, 456, 789, 1024]
        avg_rewards = {}

        for name, cls in agents:
            total = 0.0
            for seed in seeds:
                reward, _ = _run_baseline(cls, seed=seed)
                total += reward
            avg_rewards[name] = total / len(seeds)

        # Verify strict ordering
        assert avg_rewards["Random"] < avg_rewards["NoTools"], (
            f"Random ({avg_rewards['Random']:.1f}) >= NoTools ({avg_rewards['NoTools']:.1f})"
        )
        assert avg_rewards["NoTools"] < avg_rewards["SearchOnly"], (
            f"NoTools ({avg_rewards['NoTools']:.1f}) >= SearchOnly ({avg_rewards['SearchOnly']:.1f})"
        )
        assert avg_rewards["SearchOnly"] < avg_rewards["BeliefAware"], (
            f"SearchOnly ({avg_rewards['SearchOnly']:.1f}) >= BeliefAware ({avg_rewards['BeliefAware']:.1f})"
        )
        assert avg_rewards["BeliefAware"] < avg_rewards["Oracle"], (
            f"BeliefAware ({avg_rewards['BeliefAware']:.1f}) >= Oracle ({avg_rewards['Oracle']:.1f})"
        )

    def test_decomposition_holds_for_all_baselines(self):
        """Decomposition identity must hold for every baseline."""
        agents = [RandomAgent, NoToolsHeuristic, SearchOnlyHeuristic,
                  BeliefAwareBaseline, OracleAgent]
        evaluator = Evaluator()

        for cls in agents:
            reward, trace = _run_baseline(cls, seed=42)
            result = evaluator.evaluate_episode(trace)
            for step in result["step_results"]:
                gap_sum = (step["tool_use_gap"] + step["inference_gap"]
                           + step["planning_gap"])
                expected = step["oracle_reward"] - step["actual_reward"]
                assert abs(gap_sum - expected) < 1e-10, (
                    f"{cls.__name__} Day {step['day']}: decomposition failed"
                )


# =============================================================================
# Ablation Runner Tests
# =============================================================================

class TestAblationRunner:
    """Test the full ablation runner."""

    def test_ablation_suite_runs(self):
        """Run a small ablation suite (2 seeds) and verify it completes."""
        from fishing_game.runner import run_ablation_suite, ABLATION_CONFIGS, BASELINES
        results, decomposition_ok = run_ablation_suite(seeds=[42, 123])
        assert decomposition_ok, "Decomposition identity failed in ablation suite"
        assert len(results) == len(ABLATION_CONFIGS)
        for config_name in results:
            assert len(results[config_name]) == len(BASELINES)

    def test_ablation_ordering(self):
        """Verify ordering holds across all ablation configs with 5 seeds."""
        from fishing_game.runner import run_ablation_suite, verify_ordering
        results, _ = run_ablation_suite(seeds=[42, 123, 456, 789, 1024])
        assert verify_ordering(results), "Baseline ordering violated in ablation suite"
