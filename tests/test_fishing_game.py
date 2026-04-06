"""
Single test file for the entire fishing game v5 — labels removed + tide/water temp.
Tests all components: config, POMDP, simulator, evaluator, baselines, runner.
Includes CausalLearner (learns POMDP params from historical DB via SQL).
"""

import json
import inspect
import math
import numpy as np
import pytest
from fishing_game.config import CONFIG, EASY_CONFIG, HARD_CONFIG, BENCHMARK_CONFIG, _generate_states, _generate_valid_allocations
from fishing_game.pomdp import FishingPOMDP, _normal_pdf
from fishing_game.simulator import FishingGameEnv
from fishing_game.evaluator import Evaluator
from fishing_game.coding_agent import FishingGameTools
from fishing_game.baselines import (
    RandomAgent, NaivePatternMatcher, CausalLearner,
    CausalReasoner, OracleAgent,
)


# =============================================================================
# Config Tests
# =============================================================================

class TestConfig:
    """Verify config is well-formed."""

    def test_eighty_states(self):
        assert len(CONFIG["states"]) == 80

    def test_states_are_4_tuples(self):
        for s in CONFIG["states"]:
            assert len(s) == 4
            storm, wind, equip, tide = s
            assert storm in (0, 1)
            assert wind in ("N", "S", "E", "W")
            assert equip in (0, 1, 2, 3, 4)
            assert tide in (0, 1)

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

    def test_tide_transition_rows_sum_to_one(self):
        T = np.array(CONFIG["tide_transition"])
        for i, row in enumerate(T):
            assert abs(sum(row) - 1.0) < 1e-10, f"Tide row {i} sums to {sum(row)}"

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

    def test_initial_belief_length_80(self):
        assert len(CONFIG["initial_belief"]) == 80

    def test_four_zones(self):
        assert CONFIG["zones"] == ["A", "B", "C", "D"]

    def test_max_boats_10(self):
        assert CONFIG["max_boats"] == 10

    def test_valid_allocations_sums(self):
        for alloc in CONFIG["valid_allocations"]:
            total = sum(alloc.values())
            assert 1 <= total <= 10

    def test_hard_config_transitions_valid(self):
        for name in ["storm_transition", "wind_transition", "equip_transition", "tide_transition"]:
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

    def test_tide_bonus(self):
        tb = CONFIG["tide_bonus"]
        assert tb[0] == 0   # low tide: no bonus
        assert tb[1] > 0    # high tide: positive bonus

    def test_water_temp_params(self):
        wt = CONFIG["water_temp_params"]
        assert "base" in wt
        assert "tide_effect" in wt
        assert wt["tide_effect"] > 0

    def test_zone_temp_offset(self):
        zt = CONFIG["zone_temp_offset"]
        assert zt["A"] > zt["D"]  # Zone A is warmest (age confound)

    def test_tool_budgets_no_sensors(self):
        """v5: raw sensors are free, not in tool_budgets."""
        budgets = CONFIG["tool_budgets"]
        assert "read_barometer" not in budgets
        assert "read_buoy" not in budgets
        assert "inspect_equipment" not in budgets
        assert "query_maintenance_log" in budgets

    def test_hard_config_tide_bonus_smaller(self):
        assert HARD_CONFIG["tide_bonus"][1] < EASY_CONFIG["tide_bonus"][1]

    def test_hard_config_water_temp_noisier(self):
        assert HARD_CONFIG["water_temp_params"]["base"]["std"] >= EASY_CONFIG["water_temp_params"]["base"]["std"]

    def test_hard_config_zone_temp_offset_stronger(self):
        assert HARD_CONFIG["zone_temp_offset"]["A"] >= EASY_CONFIG["zone_temp_offset"]["A"]

    def test_benchmark_config_has_sensor_zones(self):
        assert "sensor_zones_per_step" in BENCHMARK_CONFIG
        assert BENCHMARK_CONFIG["sensor_zones_per_step"] == 2

    def test_config_is_benchmark(self):
        assert CONFIG is BENCHMARK_CONFIG


# =============================================================================
# POMDP Model Tests
# =============================================================================

class TestPOMDPModel:
    """Test the POMDP model: Kronecker product, belief_update, reward, optimal_action.
    Uses EASY_CONFIG for deterministic reward value assertions."""

    @pytest.fixture
    def pomdp(self):
        return FishingPOMDP(EASY_CONFIG)

    def test_80_state_pomdp(self, pomdp):
        """Verify transition matrix is 80x80 and rows sum to 1."""
        assert pomdp.T.shape == (80, 80)
        for i in range(80):
            assert abs(pomdp.T[i].sum() - 1.0) < 1e-10, f"Row {i} sums to {pomdp.T[i].sum()}"

    def test_n_states(self, pomdp):
        assert pomdp.n_states == 80

    def test_kronecker_product_correctness(self, pomdp):
        T_storm = np.array(EASY_CONFIG["storm_transition"])
        T_wind = np.array(EASY_CONFIG["wind_transition"])
        T_equip = np.array(EASY_CONFIG["equip_transition"])
        T_tide = np.array(EASY_CONFIG["tide_transition"])

        storm_map = {0: 0, 1: 1}
        wind_map = {"N": 0, "S": 1, "E": 2, "W": 3}
        equip_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        tide_map = {0: 0, 1: 1}

        for i in [0, 5, 10, 20, 40, 60, 79]:
            s_i, w_i, e_i, t_i = EASY_CONFIG["states"][i]
            for j in [0, 7, 15, 30, 50, 70, 79]:
                s_j, w_j, e_j, t_j = EASY_CONFIG["states"][j]
                expected = (T_storm[storm_map[s_i], storm_map[s_j]]
                            * T_wind[wind_map[w_i], wind_map[w_j]]
                            * T_equip[equip_map[e_i], equip_map[e_j]]
                            * T_tide[tide_map[t_i], tide_map[t_j]])
                assert abs(pomdp.T[i, j] - expected) < 1e-12

    # --- Reward function ---
    def test_reward_safe_low_tide(self, pomdp):
        """Safe zone, low tide = +7/boat."""
        idx = EASY_CONFIG["states"].index((0, "N", 0, 0))
        alloc = {"A": 10, "B": 0, "C": 0, "D": 0}
        assert pomdp.reward(idx, alloc) == 70  # 7*10

    def test_reward_safe_high_tide(self, pomdp):
        """Safe zone, high tide = +9/boat (7 + 2 tide bonus)."""
        idx = EASY_CONFIG["states"].index((0, "N", 0, 1))
        alloc = {"A": 10, "B": 0, "C": 0, "D": 0}
        assert pomdp.reward(idx, alloc) == 90  # (7+2)*10

    def test_tide_bonus_applied(self, pomdp):
        """Reward includes +2/boat on safe zones when tide is high."""
        idx_low = EASY_CONFIG["states"].index((0, "N", 0, 0))
        idx_high = EASY_CONFIG["states"].index((0, "N", 0, 1))
        alloc = {"A": 5, "B": 0, "C": 0, "D": 0}
        reward_low = pomdp.reward(idx_low, alloc)
        reward_high = pomdp.reward(idx_high, alloc)
        assert reward_high - reward_low == 10  # 2*5 = 10

    def test_reward_storm_only(self, pomdp):
        idx = EASY_CONFIG["states"].index((1, "N", 0, 0))
        alloc = {"A": 10, "B": 0, "C": 0, "D": 0}
        assert pomdp.reward(idx, alloc) == -180  # -18*10

    def test_reward_equip_only(self, pomdp):
        idx = EASY_CONFIG["states"].index((0, "N", 2, 0))
        alloc = {"A": 0, "B": 5, "C": 0, "D": 0}
        assert pomdp.reward(idx, alloc) == -50  # -10*5

    def test_reward_both_risks(self, pomdp):
        idx = EASY_CONFIG["states"].index((1, "N", 1, 0))
        alloc = {"A": 4, "B": 0, "C": 0, "D": 0}
        assert pomdp.reward(idx, alloc) == -100  # -25*4

    def test_reward_fish_abundance_bonus(self, pomdp):
        """Zone adjacent to storm gets +3/boat fish bonus."""
        # Storm in A (wind=N), fish in B (adjacent, distance 1), low tide
        idx = EASY_CONFIG["states"].index((1, "N", 0, 0))
        alloc = {"A": 0, "B": 10, "C": 0, "D": 0}
        # B is adjacent to A (distance 1), bonus = +3/boat
        assert pomdp.reward(idx, alloc) == (7 + 3) * 10  # 100

    def test_reward_fish_bonus_plus_tide(self, pomdp):
        """Storm adjacent + high tide stacks bonuses."""
        idx = EASY_CONFIG["states"].index((1, "N", 0, 1))
        alloc = {"A": 0, "B": 10, "C": 0, "D": 0}
        # B adjacent to A: 7 + 3 (fish) + 2 (tide) = 12/boat
        assert pomdp.reward(idx, alloc) == 120  # 12*10

    def test_reward_storm_zone_no_tide_bonus(self, pomdp):
        """Storm zone gets loss regardless of tide."""
        idx = EASY_CONFIG["states"].index((1, "N", 0, 1))
        alloc = {"A": 5, "B": 0, "C": 0, "D": 0}
        assert pomdp.reward(idx, alloc) == -90  # -18*5

    # --- Belief update ---
    def test_belief_update_sea_color_dark(self, pomdp):
        prior = np.array(EASY_CONFIG["initial_belief"])
        posterior = pomdp.belief_update(prior, [("sea_color", "dark")])
        assert pomdp.p_storm(posterior) > 0.8
        assert abs(posterior.sum() - 1.0) < 1e-10

    def test_belief_update_sea_color_green(self, pomdp):
        prior = np.array(EASY_CONFIG["initial_belief"])
        posterior = pomdp.belief_update(prior, [("sea_color", "green")])
        assert pomdp.p_storm(posterior) < 0.1
        assert abs(posterior.sum() - 1.0) < 1e-10

    def test_belief_update_equip_indicator_critical(self, pomdp):
        prior = np.array(EASY_CONFIG["initial_belief"])
        posterior = pomdp.belief_update(prior, [("equip_indicator", "critical")])
        assert pomdp.p_equip_failure(posterior) > 0.8
        assert abs(posterior.sum() - 1.0) < 1e-10

    def test_belief_update_barometer_low(self, pomdp):
        prior = np.array(EASY_CONFIG["initial_belief"])
        posterior = pomdp.belief_update(prior, [("barometer", 998.0)])
        assert pomdp.p_storm(posterior) > 0.9

    def test_belief_update_barometer_high(self, pomdp):
        prior = np.array(EASY_CONFIG["initial_belief"])
        posterior = pomdp.belief_update(prior, [("barometer", 1013.0)])
        assert pomdp.p_storm(posterior) < 0.05

    # --- Wave propagation belief update ---
    def test_belief_update_buoy_source_zone(self, pomdp):
        """High buoy reading (4.5m) in zone A -> storm source is A."""
        prior = np.array(EASY_CONFIG["initial_belief"])
        posterior = pomdp.belief_update(prior, [(("buoy", "A"), 4.5)])
        assert pomdp.p_storm_zone(posterior, "A") > 0.8

    def test_belief_update_buoy_propagated(self, pomdp):
        """Propagated buoy pattern: A=4.5, B=2.8 -> storm in A, not B."""
        prior = np.array(EASY_CONFIG["initial_belief"])
        posterior = pomdp.belief_update(prior, [
            (("buoy", "A"), 4.5),
            (("buoy", "B"), 2.8),
            (("buoy", "C"), 1.5),
            (("buoy", "D"), 2.8),
        ])
        assert pomdp.p_storm_zone(posterior, "A") > pomdp.p_storm_zone(posterior, "B")
        assert pomdp.p_storm_zone(posterior, "A") > pomdp.p_storm_zone(posterior, "C")
        assert pomdp.p_storm_zone(posterior, "A") > pomdp.p_storm_zone(posterior, "D")

    def test_belief_update_buoy_normal_no_storm(self, pomdp):
        """All buoys reading ~1.2m -> no storm."""
        prior = np.array(EASY_CONFIG["initial_belief"])
        posterior = pomdp.belief_update(prior, [
            (("buoy", "A"), 1.2),
            (("buoy", "B"), 1.1),
            (("buoy", "C"), 1.3),
            (("buoy", "D"), 1.2),
        ])
        assert pomdp.p_storm(posterior) < 0.1

    # --- Age-confounded equipment ---
    def test_belief_update_equip_inspection_age_confound(self, pomdp):
        """Old zone A reads high even when ok. Young zone D reads low when ok."""
        prior = np.array(EASY_CONFIG["initial_belief"])
        posterior = pomdp.belief_update(prior, [
            (("equip_inspection", "A"), 4.5),
            (("equip_inspection", "D"), 8.7),
        ])
        assert pomdp.p_equip_zone(posterior, "D") > pomdp.p_equip_zone(posterior, "A")

    # --- Water temperature belief update ---
    def test_belief_update_water_temp_high_tide(self, pomdp):
        """Warm water across all zones suggests high tide."""
        prior = np.array(EASY_CONFIG["initial_belief"])
        # High tide adds +1.5 to all zones; observe ~16.5 everywhere
        posterior = pomdp.belief_update(prior, [
            (("water_temp", "A"), 17.5),
            (("water_temp", "B"), 17.0),
            (("water_temp", "C"), 16.5),
            (("water_temp", "D"), 16.3),
        ])
        assert pomdp.p_tide(posterior) > 0.7

    def test_belief_update_water_temp_low_tide(self, pomdp):
        """Cool water across all zones suggests low tide."""
        prior = np.array(EASY_CONFIG["initial_belief"])
        posterior = pomdp.belief_update(prior, [
            (("water_temp", "A"), 15.0),
            (("water_temp", "B"), 14.5),
            (("water_temp", "C"), 14.0),
            (("water_temp", "D"), 13.8),
        ])
        assert pomdp.p_tide(posterior) < 0.3

    # --- Maintenance alerts (Poisson) ---
    def test_belief_update_maintenance_alerts(self, pomdp):
        prior = np.array(EASY_CONFIG["initial_belief"])
        posterior = pomdp.belief_update(prior, [
            (("maintenance_alerts", "A"), 8),
            (("maintenance_alerts", "D"), 6),
        ])
        assert pomdp.p_equip_zone(posterior, "D") > pomdp.p_equip_zone(posterior, "A")

    def test_belief_update_multiple_observations_v5(self, pomdp):
        """All v5 observations combined should yield well-formed posterior."""
        prior = np.array(EASY_CONFIG["initial_belief"])
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
            (("water_temp", "A"), 16.5),
            (("water_temp", "B"), 16.0),
            (("water_temp", "C"), 15.5),
            (("water_temp", "D"), 15.3),
        ])
        assert abs(posterior.sum() - 1.0) < 1e-10
        assert all(p >= 0 for p in posterior)

    def test_belief_update_preserves_normalization(self, pomdp):
        prior = np.array(EASY_CONFIG["initial_belief"])
        posterior = pomdp.belief_update(prior, [
            ("sea_color", "murky"),
            ("equip_indicator", "warning"),
            ("barometer", 1005.0),
            (("buoy", "B"), 2.0),
            (("equip_inspection", "D"), 3.0),
            (("maintenance_alerts", "A"), 5),
            (("water_temp", "C"), 15.0),
        ])
        assert abs(posterior.sum() - 1.0) < 1e-10
        assert all(p >= 0 for p in posterior)

    def test_belief_update_is_order_independent(self, pomdp):
        prior = np.array(EASY_CONFIG["initial_belief"])
        obs_a = [("sea_color", "dark"), ("barometer", 1000.0), ("equip_indicator", "warning")]
        obs_b = [("equip_indicator", "warning"), ("barometer", 1000.0), ("sea_color", "dark")]
        post_a = pomdp.belief_update(prior, obs_a)
        post_b = pomdp.belief_update(prior, obs_b)
        np.testing.assert_allclose(post_a, post_b, atol=1e-10)

    def test_predict_applies_transition(self, pomdp):
        belief = np.zeros(80)
        belief[0] = 1.0
        predicted = pomdp.predict(belief)
        np.testing.assert_allclose(predicted, pomdp.T[0, :], atol=1e-10)

    # --- Optimal action ---
    def test_optimal_action_certain_safe_low_tide(self, pomdp):
        """Certain safe, low tide -> 10 boats all in one zone, 7*10=70."""
        idx = EASY_CONFIG["states"].index((0, "N", 0, 0))
        belief = np.zeros(80)
        belief[idx] = 1.0
        alloc, er = pomdp.optimal_action(belief)
        assert sum(alloc.values()) == 10
        assert er == 70.0

    def test_optimal_action_certain_safe_high_tide(self, pomdp):
        """Certain safe, high tide -> 10 boats, (7+2)*10=90."""
        idx = EASY_CONFIG["states"].index((0, "N", 0, 1))
        belief = np.zeros(80)
        belief[idx] = 1.0
        alloc, er = pomdp.optimal_action(belief)
        assert sum(alloc.values()) == 10
        assert er == 90.0

    def test_optimal_action_storm_avoids_zone(self, pomdp):
        """Storm in zone A -> avoid A."""
        idx = EASY_CONFIG["states"].index((1, "N", 0, 0))
        belief = np.zeros(80)
        belief[idx] = 1.0
        alloc, er = pomdp.optimal_action(belief)
        assert alloc.get("A", 0) == 0

    def test_optimal_action_prefers_fish_bonus(self, pomdp):
        """Storm in A, low tide -> optimal is fish in adjacent zones B or D (+10/boat)."""
        idx = EASY_CONFIG["states"].index((1, "N", 0, 0))
        belief = np.zeros(80)
        belief[idx] = 1.0
        alloc, er = pomdp.optimal_action(belief)
        # Should prefer B or D (adjacent to A, +10/boat) over C (+7/boat)
        assert er == 100.0  # 10*10 = 100

    # --- Marginal helpers ---
    def test_p_storm(self, pomdp):
        belief = np.array(EASY_CONFIG["initial_belief"])
        assert abs(pomdp.p_storm(belief) - 0.5) < 1e-10

    def test_p_equip_failure(self, pomdp):
        belief = np.array(EASY_CONFIG["initial_belief"])
        assert abs(pomdp.p_equip_failure(belief) - 0.8) < 1e-10

    def test_p_tide(self, pomdp):
        belief = np.array(EASY_CONFIG["initial_belief"])
        assert abs(pomdp.p_tide(belief) - 0.5) < 1e-10

    def test_storm_zone_probs_sum(self, pomdp):
        belief = np.array(EASY_CONFIG["initial_belief"])
        szp = pomdp.storm_zone_probs(belief)
        assert abs(sum(szp.values()) - pomdp.p_storm(belief)) < 1e-10

    def test_equip_zone_probs_sum(self, pomdp):
        belief = np.array(EASY_CONFIG["initial_belief"])
        ezp = pomdp.equip_zone_probs(belief)
        assert abs(sum(ezp.values()) - pomdp.p_equip_failure(belief)) < 1e-10


# =============================================================================
# Simulator Tests
# =============================================================================

class TestSimulator:
    """Test the FishingGameEnv simulator v5."""

    def test_reset_returns_observation_bundle(self):
        env = FishingGameEnv()
        obs = env.reset(seed=42)
        assert obs["day"] == 1
        assert obs["days_remaining"] == 19
        assert obs["sea_color"] in ("green", "murky", "dark")
        assert obs["equip_indicator"] in ("normal", "warning", "critical")
        assert obs["cumulative_reward"] == 0.0

    def test_free_sensors_in_observation(self):
        """Raw sensors appear in observation bundle, limited to sensor_zones."""
        env = FishingGameEnv()
        obs = env.reset(seed=42)
        assert "barometer" in obs
        assert isinstance(obs["barometer"], float)
        assert "sensor_zones" in obs
        sz = set(obs["sensor_zones"])
        assert len(sz) == CONFIG.get("sensor_zones_per_step", 4)
        assert sz <= {"A", "B", "C", "D"}
        assert "buoy_readings" in obs
        assert set(obs["buoy_readings"].keys()) == sz
        assert "equipment_readings" in obs
        assert set(obs["equipment_readings"].keys()) == sz
        assert "maintenance_alerts" in obs
        assert set(obs["maintenance_alerts"].keys()) == sz
        assert "zone_infrastructure_ages" in obs
        assert "water_temp_readings" in obs
        assert set(obs["water_temp_readings"].keys()) == sz

    def test_buoy_propagation_sampling(self):
        """Buoy readings follow propagation model."""
        source_readings = []
        adjacent_readings = []
        far_readings = []
        normal_readings = []

        np_rng = np.random.RandomState(42)
        for _ in range(500):
            env = FishingGameEnv()
            env.reset(seed=np_rng.randint(0, 100000))
            storm, wind, equip, tide = env._hidden_state
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

        assert np.mean(zone_alerts["A"]) > np.mean(zone_alerts["D"])

    def test_historical_data_in_db(self):
        """Reset should pre-seed 30 days of historical data."""
        env = FishingGameEnv()
        env.reset(seed=42)
        cur = env._db.cursor()
        cur.execute("SELECT COUNT(*) FROM catch_history WHERE day < 0")
        assert cur.fetchone()[0] == 30
        cur.execute("SELECT COUNT(*) FROM maintenance_log WHERE day < 0")
        assert cur.fetchone()[0] == 30 * 4

    def test_sensor_log_has_water_temp(self):
        """sensor_log should have water_temp column with historical data."""
        env = FishingGameEnv()
        env.reset(seed=42)
        cur = env._db.cursor()
        cur.execute("SELECT COUNT(*) FROM sensor_log WHERE day < 0")
        assert cur.fetchone()[0] == 30 * 4  # 120 rows
        cur.execute("SELECT water_temp FROM sensor_log WHERE day < 0 LIMIT 1")
        row = cur.fetchone()
        assert row[0] is not None  # water_temp column exists and has data

    def test_daily_conditions_table_has_tide(self):
        """daily_conditions should have tide column in historical data."""
        env = FishingGameEnv()
        env.reset(seed=42)
        cur = env._db.cursor()
        cur.execute("SELECT COUNT(*) FROM daily_conditions WHERE day < 0")
        assert cur.fetchone()[0] == 30
        cur.execute("SELECT tide FROM daily_conditions WHERE day < 0 LIMIT 1")
        row = cur.fetchone()
        assert row[0] in (0, 1)

    def test_daily_conditions_blocked(self):
        """Agent SQL queries referencing daily_conditions should be rejected."""
        env = FishingGameEnv()
        env.reset(seed=42)
        result = env.query_fishing_log("SELECT * FROM daily_conditions")
        assert isinstance(result, dict) and "error" in result
        assert "daily_conditions" in result["error"]

        # Also test via query_maintenance_log
        result = env.query_maintenance_log("SELECT * FROM daily_conditions")
        assert isinstance(result, dict) and "error" in result

        # Test JOIN referencing daily_conditions
        result = env.query_fishing_log(
            "SELECT s.* FROM sensor_log s JOIN daily_conditions dc ON s.day = dc.day"
        )
        assert isinstance(result, dict) and "error" in result

    def test_daily_conditions_blocked_case_insensitive(self):
        """daily_conditions blocking should be case-insensitive."""
        env = FishingGameEnv()
        env.reset(seed=42)
        result = env.query_fishing_log("SELECT * FROM DAILY_CONDITIONS")
        assert isinstance(result, dict) and "error" in result
        result = env.query_fishing_log("SELECT * FROM Daily_Conditions")
        assert isinstance(result, dict) and "error" in result

    def test_db_schema_does_not_mention_daily_conditions(self):
        """daily_conditions should NOT appear in agent-visible db_schema."""
        env = FishingGameEnv()
        obs = env.reset(seed=42)
        assert "daily_conditions" not in obs["db_schema"]

    def test_db_schema_mentions_water_temp(self):
        """sensor_log in schema should mention water_temp."""
        env = FishingGameEnv()
        obs = env.reset(seed=42)
        assert "water_temp" in obs["db_schema"]

    def test_db_schema_mentions_tables(self):
        env = FishingGameEnv()
        obs = env.reset(seed=42)
        assert "maintenance_log" in obs["db_schema"]
        assert "sensor_log" in obs["db_schema"]
        assert "catch_history" in obs["db_schema"]

    def test_water_temp_confound_discoverable(self):
        """SQL query on sensor_log should reveal zone A has warmer water temp (age confound)."""
        env = FishingGameEnv()
        env.reset(seed=42)
        cur = env._db.cursor()
        cur.execute("""
            SELECT zone, AVG(water_temp) as avg_temp
            FROM sensor_log
            WHERE day < 0
            GROUP BY zone
            ORDER BY avg_temp DESC
        """)
        rows = cur.fetchall()
        # Zone A should have highest average water temp (due to zone_temp_offset)
        assert rows[0][0] == "A"

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
            allocation={"A": 5, "B": 3, "C": 2, "D": 0},
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
        budget = CONFIG["tool_budgets"]["check_weather_reports"]
        for i in range(budget):
            r = env.check_weather_reports("storm")
            assert not isinstance(r, dict) or "error" not in r
        # Budget exhausted
        r_over = env.check_weather_reports("wind")
        assert isinstance(r_over, dict) and "error" in r_over

    def test_tool_budget_maintenance_log(self):
        env = FishingGameEnv()
        env.reset(seed=42)
        budget = CONFIG["tool_budgets"]["query_maintenance_log"]
        for i in range(budget):
            r = env.query_maintenance_log("SELECT * FROM maintenance_log LIMIT 5")
            assert isinstance(r, list)
        r_over = env.query_maintenance_log("SELECT 1")
        assert isinstance(r_over, dict) and "error" in r_over

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
            "tide_high": 0.5,
        })
        assert "top_allocations" in result
        best = result["top_allocations"][0]
        # With 50% tide: 0.5*(safe*10) + 0.5*((safe+tide_bonus)*10)
        safe = CONFIG["safe_profit_per_boat"]
        tb = CONFIG["tide_bonus"][1]
        expected = 0.5 * safe * 10 + 0.5 * (safe + tb) * 10
        assert abs(best["expected_reward"] - expected) < 0.1

    def test_allocation_validation_max_boats(self):
        env = FishingGameEnv()
        env.reset(seed=42)
        with pytest.raises(AssertionError):
            env.submit_decisions(
                allocation={"A": 6, "B": 6, "C": 0, "D": 0},  # 12 > 10
                beliefs={"storm_active": 0.5, "storm_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
                         "equip_failure_active": 0.2, "equip_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}},
            )

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
        """Without SQL tools, agent gets Tier 1 only -> tool_use_gap > 0."""
        trace = self._run_episode_with_trace(seed=42)
        evaluator = Evaluator()
        result = evaluator.evaluate_episode(trace)
        total_tug = sum(s["tool_use_gap"] for s in result["step_results"])
        assert total_tug > 0, (
            f"Total tool_use_gap should be positive without SQL, got {total_tug}"
        )

    def test_tool_use_gap_zero_with_sql(self):
        """With SQL tools, agent gets Tier 1 + Tier 2 -> tool_use_gap ~ 0."""
        env = FishingGameEnv()
        obs = env.reset(seed=42)
        for _ in range(20):
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
    """Test all 5 baselines and verify ordering."""

    def test_random_agent_runs(self):
        reward, trace = _run_baseline(RandomAgent, seed=42)
        assert len(trace) == 20

    def test_naive_pattern_matcher_runs(self):
        reward, trace = _run_baseline(NaivePatternMatcher, seed=42)
        assert len(trace) == 20

    def test_causal_learner_runs(self):
        reward, trace = _run_baseline(CausalLearner, seed=42)
        assert len(trace) == 20

    def test_causal_reasoner_runs(self):
        reward, trace = _run_baseline(CausalReasoner, seed=42)
        assert len(trace) == 20

    def test_oracle_agent_runs(self):
        reward, trace = _run_baseline(OracleAgent, seed=42)
        assert len(trace) == 20

    def test_oracle_optimal_with_tide_bonus(self):
        """Oracle should get high rewards."""
        rewards = []
        for seed in [42, 123, 456, 789, 1024]:
            reward, _ = _run_baseline(OracleAgent, seed=seed)
            rewards.append(reward)
        avg = np.mean(rewards)
        # Oracle reads hidden state — should still get good rewards.
        # With BENCHMARK_CONFIG tide_bonus=1, safe=7/boat, 10 boats, ~1200+ avg expected
        assert avg >= 1200.0, f"Oracle average {avg} should be >= 1200"

    def test_oracle_brier_zero(self):
        reward, trace = _run_baseline(OracleAgent, seed=42)
        evaluator = Evaluator()
        result = evaluator.evaluate_episode(trace)
        for step in result["step_results"]:
            assert abs(step["brier_storm"]) < 1e-10
            assert abs(step["brier_equip"]) < 1e-10

    def test_causal_reasoner_brier_low(self):
        """CausalReasoner should have lower Brier scores than NaivePattern."""
        reward_c, trace_c = _run_baseline(CausalReasoner, seed=42)
        reward_n, trace_n = _run_baseline(NaivePatternMatcher, seed=42)
        evaluator = Evaluator()
        result_c = evaluator.evaluate_episode(trace_c)
        result_n = evaluator.evaluate_episode(trace_n)
        assert result_c["mean_brier_storm"] < result_n["mean_brier_storm"]

    def test_baseline_ordering(self):
        """Random ≈ NaivePattern << CausalLearner <= CausalReasoner <= Oracle."""
        agents = [
            ("Random", RandomAgent),
            ("NaivePattern", NaivePatternMatcher),
            ("CausalLearner", CausalLearner),
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

        # Random ≈ NaivePattern with tight distributions + 2 sensor zones
        # NaivePattern's heuristics are unreliable with partial observability
        assert avg_rewards["Random"] <= avg_rewards["NaivePattern"] + 50, (
            f"Random ({avg_rewards['Random']:.1f}) >> NaivePattern ({avg_rewards['NaivePattern']:.1f})"
        )
        assert avg_rewards["NaivePattern"] < avg_rewards["CausalLearner"], (
            f"NaivePattern ({avg_rewards['NaivePattern']:.1f}) >= CausalLearner ({avg_rewards['CausalLearner']:.1f})"
        )
        assert avg_rewards["CausalLearner"] <= avg_rewards["CausalReasoner"], (
            f"CausalLearner ({avg_rewards['CausalLearner']:.1f}) > CausalReasoner ({avg_rewards['CausalReasoner']:.1f})"
        )
        assert avg_rewards["CausalReasoner"] <= avg_rewards["Oracle"], (
            f"CausalReasoner ({avg_rewards['CausalReasoner']:.1f}) > Oracle ({avg_rewards['Oracle']:.1f})"
        )

    def test_causal_learner_uses_sql(self):
        """CausalLearner should use SQL tools (query_fishing_log, query_maintenance_log)."""
        reward, trace = _run_baseline(CausalLearner, seed=42)
        all_tools = []
        for step in trace:
            all_tools.extend(step.get("tools_used", []))
        sql_tools = {"query_fishing_log", "query_maintenance_log"}
        assert sql_tools & set(all_tools), (
            f"CausalLearner should use SQL tools, but used: {set(all_tools)}"
        )

    def test_causal_learner_day_classification(self):
        """CausalLearner should correctly classify reward-based day labels."""
        env = FishingGameEnv(config=CONFIG)
        obs = env.reset(seed=42)
        agent = CausalLearner(config=CONFIG)
        agent.reset()

        # Run SQL to get catch data
        catch_data = env.query_fishing_log(
            "SELECT day, zone, boats, reward FROM catch_history WHERE day < 0 ORDER BY day"
        )
        labels = agent._classify_days(catch_data)

        # Verify all classified days have valid types
        valid_types = {"storm_hit", "equip_hit", "both_hit", "safe", "safe_adjacent", "unknown"}
        for day, label in labels.items():
            assert label["type"] in valid_types, f"Day {day} has invalid type {label['type']}"
            # Storm days should identify storm zone
            if label["type"] in ("storm_hit", "both_hit"):
                assert label["storm_zone"] is not None
            # Equip days should identify equip zone
            if label["type"] in ("equip_hit", "both_hit"):
                assert label["equip_zone"] is not None
            # Safe days should identify tide
            if label["type"] in ("safe", "safe_adjacent"):
                assert label["tide"] in (0, 1)

    def test_decomposition_holds_for_all_baselines(self):
        """Decomposition identity must hold for every baseline."""
        agents = [RandomAgent, NaivePatternMatcher, CausalLearner, CausalReasoner, OracleAgent]
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
        than CausalReasoner."""
        evaluator = Evaluator()
        seeds = [42]

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
        """Run a small ablation suite (2 seeds) and verify structure + decomposition."""
        from fishing_game.runner import run_ablation_suite, ABLATION_CONFIGS, BASELINES
        results, decomposition_ok = run_ablation_suite(seeds=[42, 123])
        assert decomposition_ok, "Decomposition identity failed"
        assert len(results) == len(ABLATION_CONFIGS)
        for config_name in results:
            assert len(results[config_name]) == len(BASELINES)


# =============================================================================
# LLM+Solver Tests
# =============================================================================

from fishing_game.llm_solver_agent import (
    LLMSolverAgent, MockLLMSolverAgent,
    _parse_config_patch, _deep_merge,
)
from fishing_game.llm_agent import execute_tool_call


def _run_llm_solver(agent, seed, config=None):
    """Helper: run an LLMSolverAgent for a full episode."""
    import random as stdlib_random
    cfg = config or CONFIG
    env = FishingGameEnv(config=cfg)
    obs = env.reset(seed=seed)
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


class TestLLMSolver:

    def test_llm_solver_runs(self):
        """MockLLMSolverAgent completes a full 20-day episode."""
        agent = MockLLMSolverAgent(config=CONFIG)
        reward, trace = _run_llm_solver(agent, seed=42)
        assert len(trace) == 20

    def test_llm_solver_uses_sql(self):
        """LLM+Solver should use SQL tools (for data queries + Tier 2 promotion)."""
        agent = MockLLMSolverAgent(config=CONFIG)
        reward, trace = _run_llm_solver(agent, seed=42)
        all_tools = []
        for step in trace:
            all_tools.extend(step.get("tools_used", []))
        sql_tools = {"query_fishing_log", "query_maintenance_log"}
        assert sql_tools & set(all_tools), (
            f"LLM+Solver should use SQL tools, but used: {set(all_tools)}"
        )

    def test_llm_solver_invalid_json_fallback(self):
        """LLM returns garbage → agent falls back to defaults, episode completes."""
        def bad_llm(messages):
            return "I don't understand the question. Here's a poem about fish."

        agent = LLMSolverAgent(llm_fn=bad_llm, config=CONFIG)
        reward, trace = _run_llm_solver(agent, seed=42)
        assert len(trace) == 20
        assert agent._parsed_config_patch == {}

    def test_llm_solver_partial_json_fallback(self):
        """LLM returns JSON missing some fields → defaults used for missing."""
        def partial_llm(messages):
            return json.dumps({
                "buoy_params": {
                    "normal": {"mean": 1.5, "std": 0.6},
                    "source": {"mean": 3.5, "std": 0.8},
                    "propagated": {"mean": 2.5, "std": 0.7},
                    "far_propagated": {"mean": 1.8, "std": 0.6},
                },
                # Missing: equipment, maintenance, water_temp, transitions
            })

        agent = LLMSolverAgent(llm_fn=partial_llm, config=CONFIG)
        reward, trace = _run_llm_solver(agent, seed=42)
        assert len(trace) == 20
        assert "buoy_params" in agent._parsed_config_patch
        assert "equipment_inspection_params" not in agent._parsed_config_patch

    def test_llm_solver_true_patch_matches_causal_reasoner(self):
        """Feed true CONFIG values as patch → should match CausalReasoner rewards."""
        # Build a patch equal to the true config values
        true_patch = {
            "buoy_params": CONFIG["buoy_params"],
            "equipment_inspection_params": CONFIG["equipment_inspection_params"],
            "equipment_age_offset_factor": CONFIG["equipment_age_offset_factor"],
            "maintenance_alert_params": CONFIG["maintenance_alert_params"],
            "water_temp_params": CONFIG["water_temp_params"],
            "zone_temp_offset": CONFIG["zone_temp_offset"],
            "storm_transition": CONFIG["storm_transition"],
            "wind_transition": CONFIG["wind_transition"],
            "equip_transition": CONFIG["equip_transition"],
            "tide_transition": CONFIG["tide_transition"],
        }

        solver_agent = MockLLMSolverAgent(config=CONFIG, patch=true_patch)
        solver_reward, _ = _run_llm_solver(solver_agent, seed=42)

        reasoner_reward, _ = _run_baseline(CausalReasoner, seed=42)

        assert solver_reward == reasoner_reward, (
            f"LLM+Solver with true params ({solver_reward}) != "
            f"CausalReasoner ({reasoner_reward})"
        )

    def test_parse_config_patch_valid(self):
        """Valid JSON patch is parsed correctly."""
        text = '```json\n{"buoy_params": {"normal": {"mean": 1.5, "std": 0.6}, "source": {"mean": 3.5, "std": 0.8}, "propagated": {"mean": 2.5, "std": 0.7}, "far_propagated": {"mean": 1.8, "std": 0.6}}}\n```'
        patch = _parse_config_patch(text)
        assert "buoy_params" in patch
        assert patch["buoy_params"]["source"]["mean"] == 3.5

    def test_parse_config_patch_std_floor(self):
        """Std values below 0.3 are clamped."""
        text = json.dumps({
            "buoy_params": {
                "normal": {"mean": 1.5, "std": 0.1},
                "source": {"mean": 3.5, "std": 0.05},
                "propagated": {"mean": 2.5, "std": 0.7},
                "far_propagated": {"mean": 1.8, "std": 0.6},
            }
        })
        patch = _parse_config_patch(text)
        assert patch["buoy_params"]["normal"]["std"] == 0.3
        assert patch["buoy_params"]["source"]["std"] == 0.3

    def test_deep_merge_nested(self):
        """Deep merge correctly handles nested dicts."""
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        patch = {"a": {"b": 10}, "d": 30}
        result = _deep_merge(base, patch)
        assert result["a"]["b"] == 10
        assert result["a"]["c"] == 2  # preserved
        assert result["d"] == 30

    def test_deep_merge_ignores_unknown_keys(self):
        """Deep merge ignores keys not in base."""
        base = {"a": 1}
        patch = {"a": 2, "z": 99}
        result = _deep_merge(base, patch)
        assert result["a"] == 2
        assert "z" not in result


class TestLLMAgentSubmitValidation:

    def test_submit_decisions_requires_full_belief_fields(self):
        env = FishingGameEnv(config=CONFIG)
        env.reset(seed=42)
        result_str, is_submit = execute_tool_call(env, "submit_decisions", {
            "allocation": {"A": 1, "B": 0, "C": 0, "D": 0},
            "storm_active": 0.5,
        })
        result = json.loads(result_str)
        assert not is_submit
        assert "error" in result
        assert "Missing" in result["error"]


class TestCodingAgentSubmitValidation:

    def test_coding_agent_submit_requires_tide_high_argument(self):
        sig = inspect.signature(FishingGameTools.submit_decisions)
        assert "tide_high" in sig.parameters
        assert sig.parameters["tide_high"].default is inspect.Signature.empty

    def test_coding_agent_submit_records_full_beliefs(self):
        env = FishingGameEnv(config=CONFIG)
        env.reset(seed=42)
        tools = FishingGameTools(env)

        result_str = tools.submit_decisions(
            allocation_A=1,
            allocation_B=0,
            allocation_C=0,
            allocation_D=0,
            storm_active=0.4,
            equip_failure_active=0.3,
            storm_zone_A=0.7,
            storm_zone_B=0.2,
            storm_zone_C=0.1,
            storm_zone_D=0.0,
            equip_zone_A=0.1,
            equip_zone_B=0.2,
            equip_zone_C=0.3,
            equip_zone_D=0.4,
            tide_high=0.6,
            reasoning="test",
        )

        result = json.loads(result_str)
        assert "reward" in result
        assert "done" in result
        assert env.get_trace()[-1]["beliefs"]["tide_high"] == pytest.approx(0.6)
