"""
POMDP model for the Fishing Game. Pure math, no simulation.

States, transition matrix, observation distributions, reward function,
belief_update(), optimal_action(). All parameterized from config.
"""

import math
import numpy as np
from fishing_game.config import CONFIG


class FishingPOMDP:
    def __init__(self, config=None):
        cfg = config or CONFIG
        self.states = cfg["states"]  # [(0,"N"), (0,"S"), (1,"N"), (1,"S")]
        self.n_states = len(self.states)
        self.T = np.array(cfg["transition_matrix"], dtype=np.float64)
        self.sea_color_probs = cfg["sea_color_probs"]
        self.barometer_params = cfg["barometer_params"]
        self.buoy_params = cfg["buoy_params"]
        self.safe_profit = cfg["safe_profit_per_boat"]
        self.danger_loss = cfg["danger_loss_per_boat"]
        self.zones = cfg["zones"]
        self.max_boats = cfg["max_boats"]
        self.initial_belief = np.array(cfg["initial_belief"], dtype=np.float64)

        # Build state index for fast lookup
        self._state_idx = {s: i for i, s in enumerate(self.states)}

    def _affected_zone(self, state):
        """Derive affected zone from wind direction."""
        _, wind = state
        return "A" if wind == "N" else "B"

    def _obs_likelihood_sea_color(self, color, state_idx):
        """P(sea_color | state)."""
        storm = self.states[state_idx][0]
        probs = self.sea_color_probs[storm]
        return probs.get(color, 0.0)

    def _obs_likelihood_barometer(self, reading, state_idx):
        """P(barometer_reading | state) using Normal pdf."""
        storm = self.states[state_idx][0]
        params = self.barometer_params[storm]
        mu, sigma = params["mean"], params["std"]
        return _normal_pdf(reading, mu, sigma)

    def _obs_likelihood_buoy(self, reading, state_idx, buoy_zone):
        """P(buoy_reading | state, buoy_zone)."""
        storm, wind = self.states[state_idx]
        affected = "A" if wind == "N" else "B"
        if storm == 1 and buoy_zone == affected:
            params = self.buoy_params["danger"]
        else:
            params = self.buoy_params["normal"]
        mu, sigma = params["mean"], params["std"]
        return _normal_pdf(reading, mu, sigma)

    def predict(self, belief):
        """Predict next belief after transition: b' = T^T @ b."""
        return self.T.T @ belief

    def belief_update(self, prior, observations):
        """
        Exact Bayesian filtering.

        Args:
            prior: length-4 probability vector over states
            observations: list of (obs_type, obs_value) tuples where
                obs_type is "sea_color", "barometer", or ("buoy", zone)
                obs_value is the observed value

        Returns:
            posterior: length-4 probability vector
        """
        belief = np.array(prior, dtype=np.float64).copy()

        for obs_type, obs_value in observations:
            likelihoods = np.zeros(self.n_states, dtype=np.float64)
            for i in range(self.n_states):
                if obs_type == "sea_color":
                    likelihoods[i] = self._obs_likelihood_sea_color(obs_value, i)
                elif obs_type == "barometer":
                    likelihoods[i] = self._obs_likelihood_barometer(obs_value, i)
                elif isinstance(obs_type, tuple) and obs_type[0] == "buoy":
                    buoy_zone = obs_type[1]
                    likelihoods[i] = self._obs_likelihood_buoy(obs_value, i, buoy_zone)
                else:
                    raise ValueError(f"Unknown observation type: {obs_type}")

            belief = belief * likelihoods
            total = belief.sum()
            if total > 0:
                belief /= total
            else:
                # Degenerate case: no state explains the observation
                belief = np.ones(self.n_states, dtype=np.float64) / self.n_states

        return belief

    def reward(self, state_idx, zone, boats):
        """
        R(s, a): reward for fishing in zone with boats given state.

        zone != affected_zone OR storm == 0: +7 * boats
        zone == affected_zone AND storm == 1: -18 * boats
        """
        storm, wind = self.states[state_idx]
        affected = "A" if wind == "N" else "B"
        if storm == 1 and zone == affected:
            return self.danger_loss * boats
        else:
            return self.safe_profit * boats

    def expected_reward(self, belief, zone, boats):
        """E[R(s, a)] under belief distribution."""
        er = 0.0
        for i in range(self.n_states):
            er += belief[i] * self.reward(i, zone, boats)
        return er

    def optimal_action(self, belief):
        """
        Compute expected reward for all 6 action combos
        (2 zones x 3 boat counts 1,2,3) and return the best.

        Returns: (zone, boats, expected_reward)
        """
        best_zone, best_boats, best_er = None, None, -float("inf")
        for zone in self.zones:
            for boats in range(1, self.max_boats + 1):
                er = self.expected_reward(belief, zone, boats)
                if er > best_er:
                    best_er = er
                    best_zone = zone
                    best_boats = boats
                    best_er_val = er
        return best_zone, best_boats, best_er_val

    def p_storm(self, belief):
        """P(storm=1) from belief vector."""
        # states 2 and 3 have storm=1
        return belief[2] + belief[3]

    def p_zone_a_dangerous(self, belief):
        """P(affected_zone=A) = P(storm=1 AND wind=N) from belief."""
        # state (1,N) is index 2
        return belief[2]

    def p_zone_a_is_dangerous_given_storm(self, belief):
        """P(zone_A is dangerous) = P(storm=1 AND wind=N)."""
        return belief[2]


def _normal_pdf(x, mu, sigma):
    """Standard normal probability density function."""
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )
