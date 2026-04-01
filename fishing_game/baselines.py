"""
Five baseline agents for the Fishing Game.

All submit mandatory beliefs. All use the same simulator interface.

1. RandomAgent — never calls tools, random zone, 1 boat
2. NoToolsHeuristic — uses only sea_color, never calls tools
3. SearchOnlyHeuristic — calls check_weather_reports, keyword detection
4. BeliefAwareBaseline — full Bayesian inference with strategic tool use
5. OracleAgent — reads true hidden state (cheats)
"""

import random as stdlib_random
import numpy as np
from fishing_game.pomdp import FishingPOMDP
from fishing_game.config import CONFIG


class RandomAgent:
    """Never calls tools. Random zone, 1 boat. Beliefs: 0.5/0.5."""

    def __init__(self, config=None):
        self.cfg = config or CONFIG

    def act(self, env, obs, rng=None):
        rng = rng or stdlib_random
        zone = rng.choice(self.cfg["zones"])
        return env.submit_decisions(
            zone=zone, boats=1,
            beliefs={"storm_active": 0.5, "zone_a_is_dangerous": 0.5},
            reasoning="Random agent: no tools, random zone, 1 boat.",
        )


class NoToolsHeuristic:
    """
    Never calls tools. Uses only sea_color for decisions.
    dark → 1 boat random zone, murky → 2 boats random zone, green → 3 boats random zone.
    Beliefs derived from sea_color likelihood ratio only.
    """

    def __init__(self, config=None):
        self.cfg = config or CONFIG

    def act(self, env, obs, rng=None):
        rng = rng or stdlib_random
        color = obs["sea_color"]
        zone = rng.choice(self.cfg["zones"])

        if color == "dark":
            boats = 1
        elif color == "murky":
            boats = 2
        else:
            boats = 3

        # Beliefs from sea_color likelihood ratio
        p_color_storm0 = self.cfg["sea_color_probs"][0][color]
        p_color_storm1 = self.cfg["sea_color_probs"][1][color]
        # P(storm|color) ∝ P(color|storm) * P(storm)
        # Assume P(storm) = 0.5 prior
        p_storm = p_color_storm1 / (p_color_storm0 + p_color_storm1)

        return env.submit_decisions(
            zone=zone, boats=boats,
            beliefs={"storm_active": p_storm, "zone_a_is_dangerous": p_storm * 0.5},
            reasoning=f"NoTools: sea_color={color}, boats={boats}, random zone.",
        )


class SearchOnlyHeuristic:
    """
    Calls search_signals("storm warning weather") each day.
    If any result contains alert keywords, reduces to 1 boat random zone.
    Otherwise 3 boats random zone.
    Beliefs: binary based on keyword match.
    """

    ALERT_KEYWORDS = {"storm", "gale", "severe", "warning"}

    def __init__(self, config=None):
        self.cfg = config or CONFIG

    def act(self, env, obs, rng=None):
        rng = rng or stdlib_random
        current_day = obs["day"]

        # Search for storm signals
        alert_found = False
        if "check_weather_reports" in obs["tools_available"] and obs["tool_budget"].get("check_weather_reports", 0) > 0:
            results = env.check_weather_reports("storm warning weather severe gale")
            if isinstance(results, list):
                for r in results:
                    # Only consider today's signals
                    if r.get("day") != current_day:
                        continue
                    headline = r.get("headline", "").lower()
                    body = r.get("body", "").lower()
                    text = headline + " " + body
                    if any(kw in text for kw in self.ALERT_KEYWORDS):
                        alert_found = True
                        break

        # Also use sea_color as a secondary signal
        color = obs["sea_color"]

        zone = rng.choice(self.cfg["zones"])
        if alert_found:
            boats = 1
            p_storm = 0.8
        elif color == "dark":
            boats = 1
            p_storm = 0.6
        else:
            boats = 3
            p_storm = 0.15

        return env.submit_decisions(
            zone=zone, boats=boats,
            beliefs={"storm_active": p_storm, "zone_a_is_dangerous": p_storm * 0.5},
            reasoning=f"SearchOnly: alert={'found' if alert_found else 'none'}, "
                      f"color={color}, boats={boats}.",
        )


class BeliefAwareBaseline:
    """
    Runs POMDP.belief_update internally from all observations it gathers.
    Uses tools strategically: always reads barometer, reads buoy if P(storm) > 0.4.
    Calls POMDP.optimal_action on its posterior.
    Submits exact posterior as beliefs.
    """

    def __init__(self, config=None):
        self.cfg = config or CONFIG
        self.pomdp = FishingPOMDP(self.cfg)
        self.belief = None

    def reset(self):
        self.belief = np.array(self.cfg["initial_belief"], dtype=np.float64)

    def act(self, env, obs, rng=None):
        rng = rng or stdlib_random

        if self.belief is None:
            self.reset()

        # Prediction step (transition) — advance belief through time
        if obs["day"] > 1:
            self.belief = self.pomdp.predict(self.belief)

        # Save the predicted (pre-update) belief as the base for all updates
        base_belief = self.belief.copy()

        # Phase 1: Collect always-available observations
        observations = [("sea_color", obs["sea_color"])]

        # Read barometer if available (uses budget-tracked method)
        if "read_barometer" in obs["tools_available"] and obs["tool_budget"].get("read_barometer", 0) > 0:
            barometer = env.read_barometer()
            if not isinstance(barometer, dict):  # not an error
                observations.append(("barometer", barometer))

        # Interim belief to decide whether to gather buoy data
        interim_belief = self.pomdp.belief_update(base_belief, observations)

        # Phase 2: Conditionally gather buoy data
        if self.pomdp.p_storm(interim_belief) > 0.4:
            if "read_buoy" in obs["tools_available"] and obs["tool_budget"].get("read_buoy", 0) > 0:
                buoy_a = env.read_buoy("A")
                if not isinstance(buoy_a, dict):
                    observations.append((("buoy", "A"), buoy_a))
            if "read_buoy" in obs["tools_available"] and obs["tool_budget"].get("read_buoy", 0) > 0:
                buoy_b = env.read_buoy("B")
                if not isinstance(buoy_b, dict):
                    observations.append((("buoy", "B"), buoy_b))

        # Phase 3: Final belief = base + ALL observations (single clean update)
        self.belief = self.pomdp.belief_update(base_belief, observations)

        # Optimal action under final belief
        zone, boats, er = self.pomdp.optimal_action(self.belief)

        p_storm = float(self.pomdp.p_storm(self.belief))
        p_zone_a = float(self.pomdp.p_zone_a_dangerous(self.belief))

        return env.submit_decisions(
            zone=zone, boats=boats,
            beliefs={"storm_active": p_storm, "zone_a_is_dangerous": p_zone_a},
            reasoning=f"BeliefAware: P(storm)={p_storm:.3f}, P(zone_A_danger)={p_zone_a:.3f}, "
                      f"action=({zone},{boats}), E[R]={er:.1f}.",
        )


class OracleAgent:
    """
    Reads true hidden state directly (cheats).
    Always fishes safe zone with 3 boats.
    Submits exact true state as beliefs (Brier = 0.0).
    """

    def __init__(self, config=None):
        self.cfg = config or CONFIG

    def act(self, env, obs, rng=None):
        # Read true hidden state (cheating)
        storm, wind = env._hidden_state
        affected = "A" if wind == "N" else "B"

        # Fish the safe zone with max boats
        if storm == 1:
            zone = "B" if affected == "A" else "A"
        else:
            zone = "A"  # Any zone is safe when no storm
        boats = 3

        # Exact beliefs
        p_storm = 1.0 if storm == 1 else 0.0
        if storm == 1:
            p_zone_a = 1.0 if affected == "A" else 0.0
        else:
            p_zone_a = 0.0

        return env.submit_decisions(
            zone=zone, boats=boats,
            beliefs={"storm_active": p_storm, "zone_a_is_dangerous": p_zone_a},
            reasoning=f"Oracle: storm={storm}, wind={wind}, affected={affected}, "
                      f"fishing safe zone {zone} with {boats} boats.",
        )
