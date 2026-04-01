"""
Single test file for the entire fishing game v4 — discoverable causal structure.
Tests all components: config, POMDP, simulator, evaluator, baselines, runner.
"""

import json
import math
import numpy as np
import pytest
from fishing_game.config import CONFIG, HARD_CONFIG, _generate_states, _generate_valid_allocations
from fishing_game.pomdp import FishingPOMDP, _normal_pdf
from fishing_game.simulator import FishingGameEnv
from fishing_game.evaluator import Evaluator
from fishing_game.baselines import (
    RandomAgent, NaivePatternMatcher,
    CausalReasoner, OracleAgent,
)


# =============================================================================
# Config Tests
# =============================================================================

class TestConfig:
    """Verify config is well-formed."""

    def test_forty_states(self):
        assert len(CONFIG["states"]) == 40

    def test_states_are_3_tuples(self):
        for s in CONFIG["states"]:
            assert len(s) == 3
            storm, wind, equip = s
            assert storm in (0, 1)
            assert wind in ("N", "S", "E", "W")
            assert equip in (0, 1, 2, 3, 4)

    def test_storm_transition_rows_sum_to_one(self):
        T = np.array(CONFIG["storm_transition"])
        for i, row in enumerate(T):
            assert abs(sum(row) - 1.0) < 1e-10, f"Storm row {i} sums to {sum(row)}"

    def test_wind_transition_rows_sum_to_one(self):
        T = np.array(CONFIG["wind_transition"])
        for i, row in enumerate(T):
            assert abs(sum(row) - 1.0) < 1e-10, f"Wind row {i} sums to {sum(row)}"

    def test_equip_transition_rows_sum_to_one(self):
        T = np.array(CONFIG["equip_transition"])
        for i, row in enumerate(T):
            assert abs(sum(row) - 1.0) < 1e-10, f"Equip row {i} sums to {sum(row)}"

    def test_sea_color_probs_sum_to_one(self):
        for storm, probs in CONFIG["sea_color_probs"].items():
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-10

    def test_equip_indicator_probs_sum_to_one(self):
        for key, probs in CONFIG["equip_indicator_probs"].items():
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-10

    def test_initial_belief_sums_to_one(self):
        assert abs(sum(CONFIG["initial_belief"]) - 1.0) < 1e-10

    def test_initial_belief_length_40(self):
        assert len(CONFIG["initial_belief"]) == 40

    def test_four_zones(self):
        assert CONFIG["zones"] == ["A", "B", "C", "D"]

    def test_valid_allocations_count(self):
        allocs = _generate_valid_allocations()
        assert len(allocs) == 34

    def test_valid_allocations_sums(self):
        for alloc in CONFIG["valid_allocations"]:
            total = sum(alloc.values())
            assert 1 <= total <= 3

    def test_hard_config_transitions_valid(self):
        for name in ["storm_transition", "wind_transition", "equip_transition"]:
            T = np.array(HARD_CONFIG[name])
            for i, row in enumerate(T):
                assert abs(sum(row) - 1.0) < 1e-10, f"HARD {name} row {i}"

    def test_zone_adjacency_symmetric(self):
        adj = CONFIG["zone_adjacency"]
        for z1 in CONFIG["zones"]:
            for z2 in CONFIG["zones"]:
                assert adj[z1][z2] == adj[z2][z1], f"Asymmetric: {z1}-{z2}"

    def test_zone_adjacency_ring(self):
        """A-B-C-D-A ring: adjacent zones are distance 1, opposite is distance 2."""
        adj = CONFIG["zone_adjacency"]
        assert adj["A"]["B"] == 1
        assert adj["A"]["C"] == 2
        assert adj["A"]["D"] == 1
        assert adj["B"]["C"] == 1
        assert adj["B"]["D"] == 2

    def test_buoy_params_four_tiers(self):
        bp = CONFIG["buoy_params"]
        assert "normal" in bp
        assert "source" in bp
        assert "propagated" in bp
        assert "far_propagated" in bp

    def test_zone_infrastructure_ages(self):
        ages = CONFIG["zone_infrastructure_age"]
        assert ages["A"] > ages["B"] > ages["C"] > ages["D"]

    def test_fish_abundance_bonus(self):
        fab = CONFIG["fish_abundance_bonus"]
        assert fab[0] == 0  # storm zone: no bonus
        assert fab[1] == 3  # adjacent: +3
        assert fab[2] == 0  # far: no bonus

    def test_tool_budgets_no_sensors(self):
        """v3: raw sensors are free, not in tool_budgets."""
        budgets = CONFIG["tool_budgets"]
        assert "read_barometer" not in budgets
        assert "read_buoy" not in budgets
        assert "inspect_equipment" not in budgets
        assert "query_maintenance_log" in budgets


# =============================================================================
# POMDP Model Tests
# =============================================================================

class TestPOMDPModel:
    """Test the POMDP model: Kronecker product, belief_update, reward, optimal_action."""

    @pytest.fixture
    def pomdp(self):
        return FishingPOMDP()

    def test_transition_matrix_shape(self, pomdp):
        assert pomdp.T.shape == (40, 40)

    def test_transition_matrix_rows_sum_to_one(self, pomdp):
        for i in range(40):
            assert abs(pomdp.T[i].sum() - 1.0) < 1e-10, f"Row {i} sums to {pomdp.T[i].sum()}"

    def test_kronecker_product_correctness(self, pomdp):
        T_storm = np.array(CONFIG["storm_transition"])
        T_wind = np.array(CONFIG["wind_transition"])
        T_equip = np.array(CONFIG["equip_transition"])

        storm_map = {0: 0, 1: 1}
        wind_map = {"N": 0, "S": 1, "E": 2, "W": 3}
        equip_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

        for i in [0, 5, 10, 20, 30, 39]:
            s_i, w_i, e_i = CONFIG["states"][i]
            for j in [0, 7, 15, 25, 35, 39]:
                s_j, w_j, e_j = CONFIG["states"][j]
                expected = (T_storm[storm_map[s_i], storm_map[s_j]]
                            * T_wind[wind_map[w_i], wind_map[w_j]]
                            * T_equip[equip_map[e_i], equip_map[e_j]])
                assert abs(pomdp.T[i, j] - expected) < 1e-12

    def test_n_states(self, pomdp):
        assert pomdp.n_states == 40

    # --- Reward function ---
    def test_reward_safe(self, pomdp):
        idx = CONFIG["states"].index((0, "N", 0))
        alloc = {"A": 3, "B": 0, "C": 0, "D": 0}
        assert pomdp.reward(idx, alloc) == 21  # 7*3

    def test_reward_storm_only(self, pomdp):
        idx = CONFIG["states"].index((1, "N", 0))
        alloc = {"A": 3, "B": 0, "C": 0, "D": 0}
        assert pomdp.reward(idx, alloc) == -54  # -18*3

    def test_reward_equip_only(self, pomdp):
        idx = CONFIG["states"].index((0, "N", 2))
        alloc = {"A": 0, "B": 2, "C": 0, "D": 0}
        assert pomdp.reward(idx, alloc) == -20  # -10*2

    def test_reward_both_risks(self, pomdp):
        idx = CONFIG["states"].index((1, "N", 1))
        alloc = {"A": 2, "B": 0, "C": 0, "D": 0}
        assert pomdp.reward(idx, alloc) == -50  # -25*2

    def test_reward_fish_abundance_bonus(self, pomdp):
        """Zone adjacent to storm gets +3/boat fish bonus."""
        # Storm in A (wind=N), fish in B (adjacent, distance 1)
        idx = CONFIG["states"].index((1, "N", 0))
        alloc = {"A": 0, "B": 3, "C": 0, "D": 0}
        # B is adjacent to A (distance 1), bonus = +3/boat
        assert pomdp.reward(idx, alloc) == (7 + 3) * 3  # 30

    def test_reward_fish_bonus_far_zone(self, pomdp):
        """Zone opposite storm (distance 2) gets NO bonus."""
        # Storm in A (wind=N), fishing C (distance 2 from A)
        idx = CONFIG["states"].index((1, "N", 0))
        alloc = {"A": 0, "B": 0, "C": 3, "D": 0}
        assert pomdp.reward(idx, alloc) == 7 * 3  # 21, no bonus

    def test_reward_no_storm_no_bonus(self, pomdp):
        """No storm means no fish abundance bonus anywhere."""
        idx = CONFIG["states"].index((0, "N", 0))
        alloc = {"A": 3, "B": 0, "C": 0, "D": 0}
        assert pomdp.reward(idx, alloc) == 21

    # --- Belief update ---
    def test_belief_update_sea_color_dark(self, pomdp):
        prior = np.array(CONFIG["initial_belief"])
        posterior = pomdp.belief_update(prior, [("sea_color", "dark")])
        assert pomdp.p_storm(posterior) > 0.8
        assert abs(posterior.sum() - 1.0) < 1e-10

    def test_belief_update_sea_color_green(self, pomdp):
        prior = np.array(CONFIG["initial_belief"])
        posterior = pomdp.belief_update(prior, [("sea_color", "green")])
        assert pomdp.p_storm(posterior) < 0.1
        assert abs(posterior.sum() - 1.0) < 1e-10

    def test_belief_update_equip_indicator_critical(self, pomdp):
        prior = np.array(CONFIG["initial_belief"])
        posterior = pomdp.belief_update(prior, [("equip_indicator", "critical")])
        assert pomdp.p_equip_failure(posterior) > 0.8
        assert abs(posterior.sum() - 1.0) < 1e-10

    def test_belief_update_barometer_low(self, pomdp):
        prior = np.array(CONFIG["initial_belief"])
        posterior = pomdp.belief_update(prior, [("barometer", 998.0)])
        assert pomdp.p_storm(posterior) > 0.9

    def test_belief_update_barometer_high(self, pomdp):
        prior = np.array(CONFIG["initial_belief"])
        posterior = pomdp.belief_update(prior, [("barometer", 1013.0)])
        assert pomdp.p_storm(posterior) < 0.05

    # --- Wave propagation belief update ---
    def test_belief_update_buoy_source_zone(self, pomdp):
        """High buoy reading (4.5m) in zone A → storm source is A."""
        prior = np.array(CONFIG["initial_belief"])
        posterior = pomdp.belief_update(prior, [(("buoy", "A"), 4.5)])
        assert pomdp.p_storm_zone(posterior, "A") > 0.8

    def test_belief_update_buoy_propagated(self, pomdp):
        """Propagated buoy pattern: A=4.5, B=2.8 → storm in A, not B.
        Even though B is elevated, the SOURCE is A."""
        prior = np.array(CONFIG["initial_belief"])
        posterior = pomdp.belief_update(prior, [
            (("buoy", "A"), 4.5),
            (("buoy", "B"), 2.8),
            (("buoy", "C"), 1.5),
            (("buoy", "D"), 2.8),
        ])
        # Storm source should be A (highest buoy + consistent with propagation pattern)
        assert pomdp.p_storm_zone(posterior, "A") > pomdp.p_storm_zone(posterior, "B")
        assert pomdp.p_storm_zone(posterior, "A") > pomdp.p_storm_zone(posterior, "C")
        assert pomdp.p_storm_zone(posterior, "A") > pomdp.p_storm_zone(posterior, "D")

    def test_belief_update_buoy_normal_no_storm(self, pomdp):
        """All buoys reading ~1.2m → no storm."""
        prior = np.array(CONFIG["initial_belief"])
        posterior = pomdp.belief_update(prior, [
            (("buoy", "A"), 1.2),
            (("buoy", "B"), 1.1),
            (("buoy", "C"), 1.3),
            (("buoy", "D"), 1.2),
        ])
        assert pomdp.p_storm(posterior) < 0.1

    # --- Age-confounded equipment ---
    def test_belief_update_equip_inspection_age_confound(self, pomdp):
        """Old zone A reads high even when ok. Young zone D reads low when ok.
        Zone A ok: ~4.5 (2.0 + 25*0.1), Zone D broken: ~8.7 (8.5 + 2*0.1)
        If A reads 4.5 and D reads 8.7, failure should be in D, not A."""
        prior = np.array(CONFIG["initial_belief"])
        posterior = pomdp.belief_update(prior, [
            (("equip_inspection", "A"), 4.5),  # A reads high but it's just old
            (("equip_inspection", "D"), 8.7),  # D reads very high = actually broken
        ])
        assert pomdp.p_equip_zone(posterior, "D") > pomdp.p_equip_zone(posterior, "A")

    # --- Maintenance alerts (Poisson) ---
    def test_belief_update_maintenance_alerts(self, pomdp):
        """Zone A (age=25) with 8 alerts is normal (~7.5 baseline).
        Zone D (age=2) with 6 alerts is suspicious (~0.6 baseline, 5.6 if broken)."""
        prior = np.array(CONFIG["initial_belief"])
        posterior = pomdp.belief_update(prior, [
            (("maintenance_alerts", "A"), 8),
            (("maintenance_alerts", "D"), 6),
        ])
        assert pomdp.p_equip_zone(posterior, "D") > pomdp.p_equip_zone(posterior, "A")

    def test_belief_update_multiple_observations_v3(self, pomdp):
        """All v3 observations combined should yield well-formed posterior."""
        prior = np.array(CONFIG["initial_belief"])
        posterior = pomdp.belief_update(prior, [
            ("sea_color", "dark"),
            ("equip_indicator", "warning"),
            ("barometer", 1000.0),
            (("buoy", "A"), 4.5),
            (("buoy", "B"), 2.8),
            (("buoy", "C"), 1.5),
            (("buoy", "D"), 2.8),
            (("equip_inspection", "A"), 4.5),
            (("equip_inspection", "B"), 2.0),
            (("equip_inspection", "C"), 9.0),
            (("equip_inspection", "D"), 2.2),
            (("maintenance_alerts", "A"), 7),
            (("maintenance_alerts", "B"), 5),
            (("maintenance_alerts", "C"), 8),
            (("maintenance_alerts", "D"), 1),
        ])
        assert abs(posterior.sum() - 1.0) < 1e-10
        assert all(p >= 0 for p in posterior)

    def test_belief_update_preserves_normalization(self, pomdp):
        prior = np.array(CONFIG["initial_belief"])
        posterior = pomdp.belief_update(prior, [
            ("sea_color", "murky"),
            ("equip_indicator", "warning"),
            ("barometer", 1005.0),
            (("buoy", "B"), 2.0),
            (("equip_inspection", "D"), 3.0),
            (("maintenance_alerts", "A"), 5),
        ])
        assert abs(posterior.sum() - 1.0) < 1e-10
        assert all(p >= 0 for p in posterior)

    def test_belief_update_is_order_independent(self, pomdp):
        prior = np.array(CONFIG["initial_belief"])
        obs_a = [("sea_color", "dark"), ("barometer", 1000.0), ("equip_indicator", "warning")]
        obs_b = [("equip_indicator", "warning"), ("barometer", 1000.0), ("sea_color", "dark")]
        post_a = pomdp.belief_update(prior, obs_a)
        post_b = pomdp.belief_update(prior, obs_b)
        np.testing.assert_allclose(post_a, post_b, atol=1e-10)

    def test_predict_applies_transition(self, pomdp):
        belief = np.zeros(40)
        belief[0] = 1.0
        predicted = pomdp.predict(belief)
        np.testing.assert_allclose(predicted, pomdp.T[0, :], atol=1e-10)

    # --- Optimal action ---
    def test_optimal_action_certain_safe(self, pomdp):
        belief = np.zeros(40)
        belief[0] = 1.0
        alloc, er = pomdp.optimal_action(belief)
        assert sum(alloc.values()) == 3
        assert er == 21.0

    def test_optimal_action_storm_avoids_zone(self, pomdp):
        """Storm in zone A → avoid A."""
        idx = CONFIG["states"].index((1, "N", 0))
        belief = np.zeros(40)
        belief[idx] = 1.0
        alloc, er = pomdp.optimal_action(belief)
        assert alloc.get("A", 0) == 0

    def test_optimal_action_prefers_fish_bonus(self, pomdp):
        """Storm in A → optimal is fish in adjacent zones B or D (+10/boat)."""
        idx = CONFIG["states"].index((1, "N", 0))
        belief = np.zeros(40)
        belief[idx] = 1.0
        alloc, er = pomdp.optimal_action(belief)
        # Should prefer B or D (adjacent to A, +10/boat) over C (+7/boat)
        assert er == 30.0  # 10*3 = 30

    # --- Marginal helpers ---
    def test_p_storm(self, pomdp):
        belief = np.array(CONFIG["initial_belief"])
        assert abs(pomdp.p_storm(belief) - 0.5) < 1e-10

    def test_p_equip_failure(self, pomdp):
        belief = np.array(CONFIG["initial_belief"])
        assert abs(pomdp.p_equip_failure(belief) - 0.8) < 1e-10

    def test_storm_zone_probs_sum(self, pomdp):
        belief = np.array(CONFIG["initial_belief"])
        szp = pomdp.storm_zone_probs(belief)
        assert abs(sum(szp.values()) - pomdp.p_storm(belief)) < 1e-10

    def test_equip_zone_probs_sum(self, pomdp):
        belief = np.array(CONFIG["initial_belief"])
        ezp = pomdp.equip_zone_probs(belief)
        assert abs(sum(ezp.values()) - pomdp.p_equip_failure(belief)) < 1e-10


# =============================================================================
# Simulator Tests
# =============================================================================

class TestSimulator:
    """Test the FishingGameEnv simulator v3."""

    def test_reset_returns_observation_bundle(self):
        env = FishingGameEnv()
        obs = env.reset(seed=42)
        assert obs["day"] == 1
        assert obs["days_remaining"] == 19
        assert obs["sea_color"] in ("green", "murky", "dark")
        assert obs["equip_indicator"] in ("normal", "warning", "critical")
        assert obs["cumulative_reward"] == 0.0

    def test_free_sensors_in_observation(self):
        """v3: all raw sensors appear in observation bundle automatically."""
        env = FishingGameEnv()
        obs = env.reset(seed=42)
        assert "barometer" in obs
        assert isinstance(obs["barometer"], float)
        assert "buoy_readings" in obs
        assert set(obs["buoy_readings"].keys()) == {"A", "B", "C", "D"}
        assert "equipment_readings" in obs
        assert set(obs["equipment_readings"].keys()) == {"A", "B", "C", "D"}
        assert "maintenance_alerts" in obs
        assert set(obs["maintenance_alerts"].keys()) == {"A", "B", "C", "D"}
        assert "zone_infrastructure_ages" in obs

    def test_buoy_propagation_sampling(self):
        """Buoy readings follow propagation model."""
        # Run many episodes, collect storm-source buoy vs adjacent buoy
        source_readings = []
        adjacent_readings = []
        far_readings = []
        normal_readings = []

        np_rng = np.random.RandomState(42)
        for _ in range(500):
            env = FishingGameEnv()
            env.reset(seed=np_rng.randint(0, 100000))
            storm, wind, equip = env._hidden_state
            buoys = {z: env._available_buoys[z] for z in env.cfg["zones"]}
            if storm == 1:
                storm_zone = env.cfg["wind_to_zone"][wind]
                for z in env.cfg["zones"]:
                    dist = env.cfg["zone_adjacency"][storm_zone][z]
                    if dist == 0:
                        source_readings.append(buoys[z])
                    elif dist == 1:
                        adjacent_readings.append(buoys[z])
                    else:
                        far_readings.append(buoys[z])
            else:
                for z in env.cfg["zones"]:
                    normal_readings.append(buoys[z])

        # Source should be highest, then propagated, then far/normal
        if source_readings and adjacent_readings:
            assert np.mean(source_readings) > np.mean(adjacent_readings)
        if adjacent_readings and far_readings:
            assert np.mean(adjacent_readings) > np.mean(far_readings)

    def test_maintenance_alerts_age_correlated(self):
        """Older zones should have higher average maintenance alerts."""
        zone_alerts = {z: [] for z in CONFIG["zones"]}
        np_rng = np.random.RandomState(42)
        for _ in range(200):
            env = FishingGameEnv()
            env.reset(seed=np_rng.randint(0, 100000))
            for z in CONFIG["zones"]:
                zone_alerts[z].append(env._available_maintenance_alerts[z])

        # Zone A (age=25) should have higher avg alerts than Zone D (age=2)
        assert np.mean(zone_alerts["A"]) > np.mean(zone_alerts["D"])

    def test_historical_data_in_db(self):
        """Reset should pre-seed 30 days of historical data."""
        env = FishingGameEnv()
        env.reset(seed=42)
        cur = env._db.cursor()
        cur.execute("SELECT COUNT(*) FROM catch_history WHERE day < 0")
        assert cur.fetchone()[0] == 30  # 30 historical days
        cur.execute("SELECT COUNT(*) FROM maintenance_log WHERE day < 0")
        assert cur.fetchone()[0] == 30 * 4  # 30 days * 4 zones

    def test_sensor_log_table_exists(self):
        """sensor_log should have 120 rows (30 days x 4 zones) of historical data."""
        env = FishingGameEnv()
        env.reset(seed=42)
        cur = env._db.cursor()
        cur.execute("SELECT COUNT(*) FROM sensor_log WHERE day < 0")
        assert cur.fetchone()[0] == 30 * 4  # 120 rows

    def test_daily_conditions_table(self):
        """daily_conditions should have 30 historical rows with all columns populated."""
        env = FishingGameEnv()
        env.reset(seed=42)
        cur = env._db.cursor()
        cur.execute("SELECT COUNT(*) FROM daily_conditions WHERE day < 0")
        assert cur.fetchone()[0] == 30
        cur.execute("SELECT storm_active, storm_zone, equip_zone, barometer, sea_color "
                     "FROM daily_conditions WHERE day < 0 LIMIT 5")
        rows = cur.fetchall()
        assert len(rows) == 5
        for row in rows:
            assert row[0] in (0, 1)  # storm_active
            assert row[3] is not None  # barometer
            assert row[4] in ("green", "murky", "dark")  # sea_color

    def test_historical_buoy_propagation_discoverable(self):
        """SQL query on sensor_log + daily_conditions should reveal propagation pattern:
        source > adjacent > far buoy readings when storm is active."""
        env = FishingGameEnv()
        env.reset(seed=42)
        cur = env._db.cursor()
        # Get average buoy readings by distance from storm zone
        cur.execute("""
            SELECT dc.storm_zone, sl.zone, AVG(sl.buoy_reading) as avg_buoy
            FROM sensor_log sl
            JOIN daily_conditions dc ON sl.day = dc.day
            WHERE dc.storm_active = 1 AND dc.storm_zone IS NOT NULL
            GROUP BY dc.storm_zone, sl.zone
        """)
        rows = cur.fetchall()
        if len(rows) > 0:
            # Group by distance
            adj = CONFIG["zone_adjacency"]
            source_readings = []
            adjacent_readings = []
            far_readings = []
            for row in rows:
                storm_z, sensor_z, avg = row[0], row[1], row[2]
                dist = adj[storm_z][sensor_z]
                if dist == 0:
                    source_readings.append(avg)
                elif dist == 1:
                    adjacent_readings.append(avg)
                else:
                    far_readings.append(avg)
            if source_readings and adjacent_readings:
                assert np.mean(source_readings) > np.mean(adjacent_readings)
            if adjacent_readings and far_readings:
                assert np.mean(adjacent_readings) > np.mean(far_readings)

    def test_historical_age_confound_discoverable(self):
        """SQL query on sensor_log should reveal age-based equipment baselines:
        older zones have higher equipment readings even without failures."""
        env = FishingGameEnv()
        env.reset(seed=42)
        cur = env._db.cursor()
        # Get avg equipment reading per zone when NO equipment failure in that zone
        cur.execute("""
            SELECT sl.zone, AVG(sl.equipment_reading) as avg_equip
            FROM sensor_log sl
            JOIN daily_conditions dc ON sl.day = dc.day
            WHERE dc.equip_zone IS NULL OR dc.equip_zone != sl.zone
            GROUP BY sl.zone
        """)
        rows = {row[0]: row[1] for row in cur.fetchall()}
        if "A" in rows and "D" in rows:
            # Zone A (age=25) should have higher baseline than Zone D (age=2)
            assert rows["A"] > rows["D"]

    def test_maintenance_log_table_exists(self):
        env = FishingGameEnv()
        env.reset(seed=42)
        cur = env._db.cursor()
        cur.execute("SELECT * FROM maintenance_log LIMIT 1")
        assert cur.fetchone() is not None

    def test_same_seed_identical_trajectory(self):
        def run_episode(seed):
            env = FishingGameEnv()
            obs = env.reset(seed=seed)
            trajectory = [obs["sea_color"], obs["equip_indicator"], obs["barometer"]]
            for day in range(20):
                result = env.submit_decisions(
                    allocation={"A": 1, "B": 0, "C": 0, "D": 0},
                    beliefs={
                        "storm_active": 0.5,
                        "storm_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
                        "equip_failure_active": 0.2,
                        "equip_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
                    },
                )
                if result["done"]:
                    break
                trajectory.append(result["observation"]["sea_color"])
                trajectory.append(result["observation"]["barometer"])
                trajectory.append(result["reward"])
            return trajectory

        t1 = run_episode(42)
        t2 = run_episode(42)
        assert t1 == t2

    def test_submit_decisions_advances_day(self):
        env = FishingGameEnv()
        obs = env.reset(seed=42)
        result = env.submit_decisions(
            allocation={"A": 2, "B": 1, "C": 0, "D": 0},
            beliefs={
                "storm_active": 0.5,
                "storm_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
                "equip_failure_active": 0.2,
                "equip_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
            },
        )
        assert not result["done"]
        assert result["observation"]["day"] == 2

    def test_tool_budget_weather_reports(self):
        env = FishingGameEnv()
        env.reset(seed=42)
        r1 = env.check_weather_reports("storm")
        assert not isinstance(r1, dict) or "error" not in r1
        r2 = env.check_weather_reports("weather")
        assert not isinstance(r2, dict) or "error" not in r2
        r3 = env.check_weather_reports("wind")
        assert isinstance(r3, dict) and "error" in r3

    def test_tool_budget_maintenance_log(self):
        env = FishingGameEnv()
        env.reset(seed=42)
        r1 = env.query_maintenance_log("SELECT * FROM maintenance_log LIMIT 5")
        assert isinstance(r1, list)
        r2 = env.query_maintenance_log("SELECT COUNT(*) FROM maintenance_log")
        assert isinstance(r2, list)
        r3 = env.query_maintenance_log("SELECT 1")
        assert isinstance(r3, dict) and "error" in r3

    def test_query_fishing_log_read_only(self):
        env = FishingGameEnv()
        env.reset(seed=42)
        result = env.query_fishing_log("SELECT * FROM daily_log")
        assert isinstance(result, list)
        result = env.query_fishing_log("DROP TABLE daily_log")
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
            "storm_active": 0.0,
            "storm_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
            "equip_failure_active": 0.0,
            "equip_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
        })
        assert "top_allocations" in result
        best = result["top_allocations"][0]
        assert abs(best["expected_reward"] - 21.0) < 0.1

    def test_allocation_validation(self):
        env = FishingGameEnv()
        env.reset(seed=42)
        with pytest.raises(AssertionError):
            env.submit_decisions(
                allocation={"A": 2, "B": 2, "C": 0, "D": 0},
                beliefs={"storm_active": 0.5, "storm_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
                         "equip_failure_active": 0.2, "equip_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}},
            )

    def test_db_schema_mentions_tables(self):
        env = FishingGameEnv()
        obs = env.reset(seed=42)
        assert "maintenance_log" in obs["db_schema"]
        assert "sensor_log" in obs["db_schema"]
        assert "daily_conditions" in obs["db_schema"]

    def test_full_episode_completes(self):
        env = FishingGameEnv()
        env.reset(seed=42)
        for day in range(20):
            result = env.submit_decisions(
                allocation={"A": 1, "B": 0, "C": 0, "D": 0},
                beliefs={"storm_active": 0.5, "storm_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
                         "equip_failure_active": 0.2, "equip_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}},
            )
            if result["done"]:
                assert day == 19
                break
        assert len(env.get_trace()) == 20


# =============================================================================
# Evaluator Tests
# =============================================================================

class TestEvaluator:
    """Test the evaluator and the critical decomposition identity."""

    def _default_beliefs(self):
        return {
            "storm_active": 0.5,
            "storm_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
            "equip_failure_active": 0.2,
            "equip_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
        }

    def _run_episode_with_trace(self, seed=42):
        env = FishingGameEnv()
        env.reset(seed=seed)
        for _ in range(20):
            result = env.submit_decisions(
                allocation={"A": 1, "B": 0, "C": 0, "D": 0},
                beliefs=self._default_beliefs(),
            )
            if result["done"]:
                break
        return env.get_trace()

    def test_decomposition_identity_every_step(self):
        """CRITICAL: tool_use_gap + inference_gap + planning_gap = oracle - actual."""
        trace = self._run_episode_with_trace(seed=42)
        evaluator = Evaluator()
        result = evaluator.evaluate_episode(trace)
        for step in result["step_results"]:
            gap_sum = step["tool_use_gap"] + step["inference_gap"] + step["planning_gap"]
            expected = step["oracle_reward"] - step["actual_reward"]
            assert abs(gap_sum - expected) < 1e-10, (
                f"Day {step['day']}: gaps={gap_sum}, oracle-actual={expected}"
            )

    def test_decomposition_identity_multiple_seeds(self):
        evaluator = Evaluator()
        for seed in [1, 7, 42, 100, 999]:
            trace = self._run_episode_with_trace(seed=seed)
            result = evaluator.evaluate_episode(trace)
            for step in result["step_results"]:
                gap_sum = step["tool_use_gap"] + step["inference_gap"] + step["planning_gap"]
                expected = step["oracle_reward"] - step["actual_reward"]
                assert abs(gap_sum - expected) < 1e-10

    def test_tool_use_gap_positive_without_sql(self):
        """Without SQL tools, agent gets Tier 1 only → tool_use_gap > 0."""
        trace = self._run_episode_with_trace(seed=42)
        evaluator = Evaluator()
        result = evaluator.evaluate_episode(trace)
        total_tug = sum(s["tool_use_gap"] for s in result["step_results"])
        assert total_tug > 0, (
            f"Total tool_use_gap should be positive without SQL, got {total_tug}"
        )

    def test_tool_use_gap_zero_with_sql(self):
        """With SQL tools, agent gets Tier 1 + Tier 2 → tool_use_gap ≈ 0."""
        env = FishingGameEnv()
        obs = env.reset(seed=42)
        for _ in range(20):
            # Use SQL tool to trigger Tier 2 promotion
            env.query_fishing_log("SELECT 1")
            result = env.submit_decisions(
                allocation={"A": 1, "B": 0, "C": 0, "D": 0},
                beliefs=self._default_beliefs(),
            )
            if result["done"]:
                break
        trace = env.get_trace()
        evaluator = Evaluator()
        result = evaluator.evaluate_episode(trace)
        for step in result["step_results"]:
            assert abs(step["tool_use_gap"]) < 1e-10, (
                f"Day {step['day']}: tool_use_gap should be ~0 with SQL"
            )

    def test_episode_level_metrics(self):
        trace = self._run_episode_with_trace(seed=42)
        evaluator = Evaluator()
        result = evaluator.evaluate_episode(trace)
        assert "total_reward" in result
        assert "mean_brier_storm" in result
        assert "total_tool_use_gap" in result
        assert "total_inference_gap" in result
        assert "total_planning_gap" in result
        assert len(result["step_results"]) == 20

    def test_brier_scores_bounded(self):
        trace = self._run_episode_with_trace(seed=42)
        evaluator = Evaluator()
        result = evaluator.evaluate_episode(trace)
        for step in result["step_results"]:
            assert 0.0 <= step["brier_storm"] <= 1.0
            assert 0.0 <= step["brier_equip"] <= 1.0


# =============================================================================
# Baseline Tests
# =============================================================================

def _run_baseline(agent_cls, seed, config=None):
    """Helper: run a baseline agent for a full episode."""
    import random as stdlib_random
    cfg = config or CONFIG
    env = FishingGameEnv(config=cfg)
    obs = env.reset(seed=seed)
    agent = agent_cls(config=cfg)
    if hasattr(agent, "reset"):
        agent.reset()

    rng = stdlib_random.Random(seed + 1000)
    total_reward = 0.0
    for _ in range(cfg["episode_length"]):
        result = agent.act(env, obs, rng=rng)
        total_reward += result["reward"]
        if result["done"]:
            break
        obs = result["observation"]
    return total_reward, env.get_trace()


class TestBaselines:
    """Test all 4 baselines and verify ordering."""

    def test_random_agent_runs(self):
        reward, trace = _run_baseline(RandomAgent, seed=42)
        assert len(trace) == 20

    def test_naive_pattern_matcher_runs(self):
        reward, trace = _run_baseline(NaivePatternMatcher, seed=42)
        assert len(trace) == 20

    def test_causal_reasoner_runs(self):
        reward, trace = _run_baseline(CausalReasoner, seed=42)
        assert len(trace) == 20

    def test_oracle_agent_runs(self):
        reward, trace = _run_baseline(OracleAgent, seed=42)
        assert len(trace) == 20

    def test_oracle_optimal_with_fish_bonus(self):
        """Oracle should get >= 21/day average (fish bonus means some days > 21)."""
        rewards = []
        for seed in [42, 123, 456, 789, 1024]:
            reward, _ = _run_baseline(OracleAgent, seed=seed)
            rewards.append(reward)
        avg = np.mean(rewards)
        # Oracle gets 21/day on no-storm days, up to 30/day on storm days
        assert avg >= 420.0, f"Oracle average {avg} should be >= 420"

    def test_oracle_brier_zero(self):
        reward, trace = _run_baseline(OracleAgent, seed=42)
        evaluator = Evaluator()
        result = evaluator.evaluate_episode(trace)
        for step in result["step_results"]:
            assert abs(step["brier_storm"]) < 1e-10
            assert abs(step["brier_equip"]) < 1e-10

    def test_causal_reasoner_brier_low(self):
        """CausalReasoner with full data should have low Brier scores."""
        reward, trace = _run_baseline(CausalReasoner, seed=42)
        evaluator = Evaluator()
        result = evaluator.evaluate_episode(trace)
        assert result["mean_brier_storm"] < 0.15

    def test_baseline_ordering(self):
        """Random < NaivePattern < CausalReasoner <= Oracle averaged over seeds."""
        agents = [
            ("Random", RandomAgent),
            ("NaivePattern", NaivePatternMatcher),
            ("CausalReasoner", CausalReasoner),
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

        assert avg_rewards["Random"] < avg_rewards["NaivePattern"], (
            f"Random ({avg_rewards['Random']:.1f}) >= NaivePattern ({avg_rewards['NaivePattern']:.1f})"
        )
        assert avg_rewards["NaivePattern"] < avg_rewards["CausalReasoner"], (
            f"NaivePattern ({avg_rewards['NaivePattern']:.1f}) >= CausalReasoner ({avg_rewards['CausalReasoner']:.1f})"
        )
        assert avg_rewards["CausalReasoner"] <= avg_rewards["Oracle"], (
            f"CausalReasoner ({avg_rewards['CausalReasoner']:.1f}) > Oracle ({avg_rewards['Oracle']:.1f})"
        )

    def test_decomposition_holds_for_all_baselines(self):
        """Decomposition identity must hold for every baseline."""
        agents = [RandomAgent, NaivePatternMatcher, CausalReasoner, OracleAgent]
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

    def test_naive_has_higher_total_gap_than_causal(self):
        """NaivePatternMatcher should have higher total gap (tool_use + inference)
        than CausalReasoner. In v4, NaivePattern has large tool_use_gap (no SQL)
        plus inference errors, while CausalReasoner has ~0 for both."""
        evaluator = Evaluator()
        seeds = [42, 123, 456]

        naive_total = 0.0
        causal_total = 0.0

        for seed in seeds:
            _, trace = _run_baseline(NaivePatternMatcher, seed=seed)
            result = evaluator.evaluate_episode(trace)
            naive_total += result["total_tool_use_gap"] + result["total_inference_gap"]

            _, trace = _run_baseline(CausalReasoner, seed=seed)
            result = evaluator.evaluate_episode(trace)
            causal_total += result["total_tool_use_gap"] + result["total_inference_gap"]

        assert naive_total > causal_total, (
            f"Naive total gap ({naive_total:.1f}) should exceed "
            f"Causal ({causal_total:.1f})"
        )


# =============================================================================
# Ablation Runner Tests
# =============================================================================

class TestAblationRunner:

    def test_ablation_suite_runs(self):
        """Run a small ablation suite (2 seeds) and verify it completes."""
        from fishing_game.runner import run_ablation_suite, ABLATION_CONFIGS, BASELINES
        results, decomposition_ok = run_ablation_suite(seeds=[42, 123])
        assert decomposition_ok, "Decomposition identity failed"
        assert len(results) == len(ABLATION_CONFIGS)
        for config_name in results:
            assert len(results[config_name]) == len(BASELINES)

    def test_ablation_ordering_full(self):
        """Verify ordering holds for 'full' config."""
        from fishing_game.runner import run_ablation_suite, BASELINES
        results, _ = run_ablation_suite(seeds=[42, 123, 456, 789, 1024])
        agent_names = [name for name, _ in BASELINES]
        rewards = [results["full"][name]["reward_mean"] for name in agent_names]
        for i in range(len(rewards) - 1):
            if i == len(rewards) - 2:
                # CausalReasoner <= Oracle
                assert rewards[i] <= rewards[i + 1], (
                    f"full: {agent_names[i]} ({rewards[i]:.1f}) > {agent_names[i+1]} ({rewards[i+1]:.1f})"
                )
            else:
                assert rewards[i] < rewards[i + 1], (
                    f"full: {agent_names[i]} ({rewards[i]:.1f}) >= {agent_names[i+1]} ({rewards[i+1]:.1f})"
                )
