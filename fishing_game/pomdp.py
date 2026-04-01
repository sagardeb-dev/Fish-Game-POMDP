"""
POMDP model for the Fishing Game v3 — Causal reasoning traps.

40 states = 2(storm) x 4(wind) x 5(equip_failure).
Transition matrix built as Kronecker product of 3 factored sub-matrices.
Belief update handles: sea_color, equip_indicator, barometer, buoy(zone)
with wave propagation, equip_inspection(zone) with age confound,
maintenance_alerts(zone) with Poisson likelihood.
Reward: dual-risk + fish abundance bonus for zones adjacent to storm.

All parameterized from config.
"""

import math
import numpy as np
from fishing_game.config import CONFIG


class FishingPOMDP:
    def __init__(self, config=None):
        cfg = config or CONFIG
        self.cfg = cfg
        self.states = cfg["states"]  # list of (storm, wind, equip) tuples
        self.n_states = len(self.states)
        self.sea_color_probs = cfg["sea_color_probs"]
        self.equip_indicator_probs = cfg["equip_indicator_probs"]
        self.barometer_params = cfg["barometer_params"]
        self.buoy_params = cfg["buoy_params"]
        self.equip_inspection_params = cfg["equipment_inspection_params"]
        self.safe_profit = cfg["safe_profit_per_boat"]
        self.danger_loss = cfg["danger_loss_per_boat"]
        self.danger_loss_equip = cfg["danger_loss_equip_per_boat"]
        self.danger_loss_both = cfg["danger_loss_both_per_boat"]
        self.zones = cfg["zones"]
        self.max_boats = cfg["max_boats"]
        self.initial_belief = np.array(cfg["initial_belief"], dtype=np.float64)
        self.wind_to_zone = cfg["wind_to_zone"]
        self.equip_to_zone = cfg["equip_to_zone"]
        self.valid_allocations = cfg["valid_allocations"]
        self.zone_adjacency = cfg["zone_adjacency"]
        self.fish_abundance_bonus = cfg["fish_abundance_bonus"]
        self.age_offset_factor = cfg["equipment_age_offset_factor"]
        self.zone_ages = cfg["zone_infrastructure_age"]
        self.maintenance_params = cfg["maintenance_alert_params"]

        # Build state index for fast lookup
        self._state_idx = {s: i for i, s in enumerate(self.states)}

        # Build 40x40 transition matrix from Kronecker product
        self.T = self._build_transition_matrix(cfg)

    def _build_transition_matrix(self, cfg):
        """Build full transition matrix as Kronecker product of 3 sub-matrices."""
        T_storm = np.array(cfg["storm_transition"], dtype=np.float64)
        T_wind = np.array(cfg["wind_transition"], dtype=np.float64)
        T_equip = np.array(cfg["equip_transition"], dtype=np.float64)

        # Kronecker product: T = T_storm (x) T_wind (x) T_equip
        T = np.kron(T_storm, np.kron(T_wind, T_equip))

        # Verify shape and row sums
        assert T.shape == (self.n_states, self.n_states), f"T shape {T.shape} != ({self.n_states}, {self.n_states})"
        return T

    # =========================================================================
    # State property helpers
    # =========================================================================

    def _storm_zone(self, state):
        """Which zone the storm hits, given state tuple."""
        _, wind, _ = state
        return self.wind_to_zone[wind]

    def _equip_zone(self, state):
        """Which zone has equipment failure, or None."""
        _, _, equip = state
        return self.equip_to_zone[equip]

    # =========================================================================
    # Observation likelihoods
    # =========================================================================

    def _obs_likelihood_sea_color(self, color, state_idx):
        """P(sea_color | state). Depends only on storm."""
        storm = self.states[state_idx][0]
        return self.sea_color_probs[storm].get(color, 0.0)

    def _obs_likelihood_equip_indicator(self, level, state_idx):
        """P(equip_indicator | state). Depends on whether equip_failure > 0."""
        equip = self.states[state_idx][2]
        key = 1 if equip > 0 else 0
        return self.equip_indicator_probs[key].get(level, 0.0)

    def _obs_likelihood_barometer(self, reading, state_idx):
        """P(barometer_reading | state). Depends only on storm."""
        storm = self.states[state_idx][0]
        params = self.barometer_params[storm]
        return _normal_pdf(reading, params["mean"], params["std"])

    def _obs_likelihood_buoy(self, reading, state_idx, buoy_zone):
        """P(buoy_reading | state, buoy_zone). Uses wave propagation model."""
        storm, wind, _ = self.states[state_idx]
        if storm == 0:
            params = self.buoy_params["normal"]
        else:
            storm_zone = self.wind_to_zone[wind]
            distance = self.zone_adjacency[storm_zone][buoy_zone]
            if distance == 0:
                params = self.buoy_params["source"]
            elif distance == 1:
                params = self.buoy_params["propagated"]
            else:
                params = self.buoy_params["far_propagated"]
        return _normal_pdf(reading, params["mean"], params["std"])

    def _obs_likelihood_equip_inspection(self, reading, state_idx, zone):
        """P(inspection_reading | state, zone). Age-confounded: older zones read higher."""
        equip = self.states[state_idx][2]
        equip_zone = self.equip_to_zone[equip]
        if equip_zone == zone:
            params = self.equip_inspection_params["broken"]
        else:
            params = self.equip_inspection_params["ok"]
        age_offset = self.zone_ages[zone] * self.age_offset_factor
        return _normal_pdf(reading, params["mean"] + age_offset, params["std"])

    def _obs_likelihood_maintenance(self, alert_count, state_idx, zone):
        """P(maintenance_alerts | state, zone). Poisson with age confound."""
        equip = self.states[state_idx][2]
        equip_zone = self.equip_to_zone[equip]
        actually_broken = (equip_zone == zone)
        age = self.zone_ages[zone]
        rate = age * self.maintenance_params["age_rate_factor"]
        if actually_broken:
            rate += self.maintenance_params["failure_signal"]
        # Poisson PMF: rate^k * exp(-rate) / k!
        k = int(alert_count)
        if rate == 0:
            return 1.0 if k == 0 else 0.0
        return (rate ** k) * math.exp(-rate) / math.factorial(k)

    # =========================================================================
    # Belief operations
    # =========================================================================

    def predict(self, belief):
        """Predict next belief after transition: b' = T^T @ b."""
        return self.T.T @ belief

    def belief_update(self, prior, observations):
        """
        Exact Bayesian filtering over 40-state space.

        Args:
            prior: length-40 probability vector over states
            observations: list of (obs_type, obs_value) tuples where
                obs_type is one of:
                  "sea_color", "equip_indicator", "barometer",
                  ("buoy", zone), ("equip_inspection", zone),
                  ("maintenance_alerts", zone)

        Returns:
            posterior: length-40 probability vector
        """
        belief = np.array(prior, dtype=np.float64).copy()

        for obs_type, obs_value in observations:
            likelihoods = np.zeros(self.n_states, dtype=np.float64)
            for i in range(self.n_states):
                if obs_type == "sea_color":
                    likelihoods[i] = self._obs_likelihood_sea_color(obs_value, i)
                elif obs_type == "equip_indicator":
                    likelihoods[i] = self._obs_likelihood_equip_indicator(obs_value, i)
                elif obs_type == "barometer":
                    likelihoods[i] = self._obs_likelihood_barometer(obs_value, i)
                elif isinstance(obs_type, tuple) and obs_type[0] == "buoy":
                    likelihoods[i] = self._obs_likelihood_buoy(obs_value, i, obs_type[1])
                elif isinstance(obs_type, tuple) and obs_type[0] == "equip_inspection":
                    likelihoods[i] = self._obs_likelihood_equip_inspection(obs_value, i, obs_type[1])
                elif isinstance(obs_type, tuple) and obs_type[0] == "maintenance_alerts":
                    likelihoods[i] = self._obs_likelihood_maintenance(obs_value, i, obs_type[1])
                else:
                    raise ValueError(f"Unknown observation type: {obs_type}")

            belief = belief * likelihoods
            total = belief.sum()
            if total > 0:
                belief /= total
            else:
                belief = np.ones(self.n_states, dtype=np.float64) / self.n_states

        return belief

    # =========================================================================
    # Reward function
    # =========================================================================

    def reward(self, state_idx, allocation):
        """
        R(s, a): reward for a given allocation dict and state.

        allocation: {"A": n, "B": n, "C": n, "D": n} where sum in [1, max_boats]

        Per zone, per boat:
          - Safe (no risk):           +7 (+ fish abundance bonus if adjacent to storm)
          - Storm only:              -18
          - Equipment failure only:  -10
          - Both storm + equip:      -25
        """
        state = self.states[state_idx]
        storm, wind, equip = state
        storm_zone = self.wind_to_zone[wind] if storm == 1 else None
        equip_zone = self.equip_to_zone[equip]

        total = 0.0
        for zone in self.zones:
            boats = allocation.get(zone, 0)
            if boats == 0:
                continue

            has_storm = (storm_zone == zone)
            has_equip = (equip_zone == zone)

            if has_storm and has_equip:
                total += self.danger_loss_both * boats
            elif has_storm:
                total += self.danger_loss * boats
            elif has_equip:
                total += self.danger_loss_equip * boats
            else:
                profit = self.safe_profit
                # Fish abundance bonus for zones near storm (currents bring fish)
                if storm == 1:
                    distance = self.zone_adjacency[storm_zone][zone]
                    profit += self.fish_abundance_bonus.get(distance, 0)
                total += profit * boats

        return total

    def expected_reward(self, belief, allocation):
        """E[R(s, a)] under belief distribution."""
        er = 0.0
        for i in range(self.n_states):
            er += belief[i] * self.reward(i, allocation)
        return er

    def optimal_action(self, belief):
        """
        Enumerate all 35 valid allocations and return the best.

        Returns: (allocation_dict, expected_reward)
        """
        best_alloc = None
        best_er = -float("inf")
        for alloc in self.valid_allocations:
            er = self.expected_reward(belief, alloc)
            if er > best_er:
                best_er = er
                best_alloc = alloc
        return best_alloc, best_er

    # =========================================================================
    # Marginal probability helpers
    # =========================================================================

    def p_storm(self, belief):
        """P(storm=1) from belief vector."""
        total = 0.0
        for i, (storm, _, _) in enumerate(self.states):
            if storm == 1:
                total += belief[i]
        return total

    def p_equip_failure(self, belief):
        """P(equip_failure > 0) from belief vector."""
        total = 0.0
        for i, (_, _, equip) in enumerate(self.states):
            if equip > 0:
                total += belief[i]
        return total

    def p_storm_zone(self, belief, zone):
        """P(storm hits this zone) = P(storm=1 AND wind maps to zone)."""
        # Reverse lookup: which wind values map to this zone
        total = 0.0
        for i, (storm, wind, _) in enumerate(self.states):
            if storm == 1 and self.wind_to_zone[wind] == zone:
                total += belief[i]
        return total

    def p_equip_zone(self, belief, zone):
        """P(equipment failure in this zone)."""
        zone_equip = self.cfg["zone_to_equip"][zone]
        total = 0.0
        for i, (_, _, equip) in enumerate(self.states):
            if equip == zone_equip:
                total += belief[i]
        return total

    def storm_zone_probs(self, belief):
        """Dict of P(storm hits zone) for each zone. Sums to P(storm)."""
        return {z: self.p_storm_zone(belief, z) for z in self.zones}

    def equip_zone_probs(self, belief):
        """Dict of P(equip failure in zone) for each zone. Sums to P(equip>0)."""
        return {z: self.p_equip_zone(belief, z) for z in self.zones}


def _normal_pdf(x, mu, sigma):
    """Standard normal probability density function."""
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )
