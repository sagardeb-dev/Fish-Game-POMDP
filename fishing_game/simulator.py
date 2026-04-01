"""
Fishing Game Simulator — OpenEnv-compatible in-process environment.

reset(), tool methods (check_weather_reports, query_fishing_log, analyze_data,
evaluate_options, forecast_scenario), submit_decisions(). Episode-local SQLite
database created on reset, updated on submit. Signal emission with tiers.

All parameters derive from the shared config.
"""

import math
import random
import re
import sqlite3
import uuid
from io import StringIO
from contextlib import redirect_stdout

import numpy as np

from fishing_game.config import CONFIG
from fishing_game.pomdp import FishingPOMDP


class FishingGameEnv:
    """
    OpenEnv-compatible fishing game environment.
    All interaction through reset(), tool calls, and submit_decisions().
    """

    def __init__(self, config=None, ablation=None):
        """
        Args:
            config: Config dict. Defaults to CONFIG.
            ablation: Dict of tool_name -> bool indicating which tools are enabled.
                      Defaults to all enabled.
        """
        self.cfg = config or CONFIG
        self.pomdp = FishingPOMDP(self.cfg)
        self.ablation = ablation or {t: True for t in self.cfg["tool_budgets"]}
        self._episode_active = False

    def reset(self, seed=None):
        """
        Initialize a new episode. Creates fresh SQLite database.
        Returns initial observation bundle.
        """
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(
            seed if seed is not None else self._rng.randint(0, 2**31)
        )

        self._day = 1
        self._cumulative_reward = 0.0
        self._yesterday_reward = 0.0
        self._yesterday_zone = None
        self._yesterday_boats = None
        self._episode_active = True

        # Sample initial hidden state
        self._hidden_state_idx = self._sample_initial_state()
        self._hidden_state = self.cfg["states"][self._hidden_state_idx]

        # Episode trace for evaluator
        self._trace = []

        # Create episode-local SQLite database (in-memory)
        self._db = sqlite3.connect(":memory:")
        self._db.row_factory = sqlite3.Row
        self._init_db()

        # Tool budgets for current day
        self._reset_daily_budgets()

        # Emit observations for day 1
        self._current_sea_color = self._sample_sea_color()
        self._current_signals = self._emit_signals()
        self._store_signals(self._current_signals)

        # Track tool usage and observations for this step
        self._step_tool_usage = []
        self._step_observations = [("sea_color", self._current_sea_color)]

        # Available tool observations (what COULD be gathered)
        self._available_barometer = self._sample_barometer()
        self._available_buoy_a = self._sample_buoy("A")
        self._available_buoy_b = self._sample_buoy("B")

        return self._make_observation_bundle()

    def _sample_initial_state(self):
        """Sample initial state from uniform distribution."""
        return self._rng.randint(0, 3)

    def _init_db(self):
        """Create the visible database tables."""
        cur = self._db.cursor()
        cur.execute("""
            CREATE TABLE daily_log (
                day          INTEGER PRIMARY KEY,
                sea_color    TEXT,
                zone_fished  TEXT,
                boats_sent   INTEGER,
                reward       REAL,
                cumulative   REAL
            )
        """)
        cur.execute("""
            CREATE TABLE weather_signals (
                signal_id    TEXT PRIMARY KEY,
                day          INTEGER,
                source_type  TEXT,
                headline     TEXT,
                body         TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE catch_history (
                day          INTEGER,
                zone         TEXT,
                boats        INTEGER,
                reward       REAL
            )
        """)
        self._db.commit()

    def _reset_daily_budgets(self):
        """Reset tool budgets for the current day."""
        self._budgets = dict(self.cfg["tool_budgets"])

    def _sample_sea_color(self):
        """Sample sea color from observation distribution given hidden state."""
        storm = self._hidden_state[0]
        probs = self.cfg["sea_color_probs"][storm]
        colors = list(probs.keys())
        weights = [probs[c] for c in colors]
        return self._rng.choices(colors, weights=weights, k=1)[0]

    def _sample_barometer(self):
        """Pre-sample barometer reading for this step."""
        storm = self._hidden_state[0]
        params = self.cfg["barometer_params"][storm]
        return self._np_rng.normal(params["mean"], params["std"])

    def _sample_buoy(self, zone):
        """Pre-sample buoy reading for a given zone."""
        storm, wind = self._hidden_state
        affected = "A" if wind == "N" else "B"
        if storm == 1 and zone == affected:
            params = self.cfg["buoy_params"]["danger"]
        else:
            params = self.cfg["buoy_params"]["normal"]
        return self._np_rng.normal(params["mean"], params["std"])

    def _emit_signals(self):
        """
        Emit 2-5 weather signals based on hidden state.
        Each signal has a tier (hidden) and visible attributes.
        """
        lo, hi = self.cfg["signals_per_step_range"]
        n_signals = self._rng.randint(lo, hi)
        storm = self._hidden_state[0]
        signals = []
        tiers_used = []

        for _ in range(n_signals):
            # Pick a tier to attempt emission
            tier = self._rng.choice([1, 2, 3])
            tier_cfg = self.cfg["signal_tiers"][tier]

            if tier == 3:
                emit_prob = tier_cfg["emission_prob_always"]
            else:
                emit_prob = tier_cfg["emission_prob"][storm]

            if self._rng.random() < emit_prob:
                headline = self._rng.choice(tier_cfg["headlines"])
                source = self._rng.choice(self.cfg["signal_sources"])
                signal = {
                    "signal_id": str(uuid.uuid4())[:8],
                    "day": self._day,
                    "source_type": source,
                    "headline": headline,
                    "body": f"Detailed report: {headline.lower()}. "
                            f"Source: {source}. Day {self._day} update.",
                    # Hidden fields — never exposed to agent
                    "_tier": tier,
                    "_storm": storm,
                    "_wind": self._hidden_state[1],
                    "_affected_zone": "A" if self._hidden_state[1] == "N" else "B",
                }
                signals.append(signal)

        # Ensure at least min signals by adding noise tier if needed
        while len(signals) < lo:
            tier_cfg = self.cfg["signal_tiers"][3]
            headline = self._rng.choice(tier_cfg["headlines"])
            source = self._rng.choice(self.cfg["signal_sources"])
            signals.append({
                "signal_id": str(uuid.uuid4())[:8],
                "day": self._day,
                "source_type": source,
                "headline": headline,
                "body": f"Detailed report: {headline.lower()}. "
                        f"Source: {source}. Day {self._day} update.",
                "_tier": 3,
                "_storm": storm,
                "_wind": self._hidden_state[1],
                "_affected_zone": "A" if self._hidden_state[1] == "N" else "B",
            })

        return signals

    def _store_signals(self, signals):
        """Store signals in the visible database (without hidden fields)."""
        cur = self._db.cursor()
        for sig in signals:
            cur.execute(
                "INSERT INTO weather_signals (signal_id, day, source_type, headline, body) "
                "VALUES (?, ?, ?, ?, ?)",
                (sig["signal_id"], sig["day"], sig["source_type"],
                 sig["headline"], sig["body"]),
            )
        self._db.commit()

    def _make_observation_bundle(self):
        """Create the observation bundle the agent receives each day."""
        available_tools = [t for t in self.cfg["tool_budgets"] if self.ablation.get(t, True)]
        budget = {t: self._budgets[t] for t in available_tools}
        bundle = {
            "day": self._day,
            "days_remaining": self.cfg["episode_length"] - self._day,
            "sea_color": self._current_sea_color,
            "yesterday_reward": self._yesterday_reward,
            "yesterday_zone": self._yesterday_zone,
            "yesterday_boats": self._yesterday_boats,
            "cumulative_reward": self._cumulative_reward,
            "tools_available": available_tools,
            "tool_budget": budget,
            "db_schema": (
                "Available tables:\n\n"
                "daily_log (day, sea_color, zone_fished, boats_sent, reward, cumulative)\n"
                "  - Your complete fishing log\n\n"
                "weather_signals (signal_id, day, source_type, headline, body)\n"
                "  - Weather and maritime intelligence reports\n\n"
                "catch_history (day, zone, boats, reward)\n"
                "  - Historical catch outcomes by zone"
            ),
        }
        return bundle

    # =========================================================================
    # Tool methods
    # =========================================================================

    def check_weather_reports(self, query, max_results=3):
        """
        Keyword search over the weather_signals table.
        Budget: 2 per day.
        """
        if not self.ablation.get("check_weather_reports", True):
            return {"error": "check_weather_reports is disabled in this configuration"}
        if self._budgets.get("check_weather_reports", 0) <= 0:
            return {"error": "check_weather_reports budget exhausted for today"}
        self._budgets["check_weather_reports"] -= 1
        self._step_tool_usage.append("check_weather_reports")

        cur = self._db.cursor()
        cur.execute("SELECT signal_id, day, source_type, headline, body FROM weather_signals")
        rows = cur.fetchall()

        # Simple keyword matching + recency boost
        keywords = query.lower().split()
        scored = []
        for row in rows:
            text = f"{row['headline']} {row['body']}".lower()
            keyword_score = sum(1 for kw in keywords if kw in text)
            recency_boost = row["day"] / self._day if self._day > 0 else 1.0
            score = keyword_score + recency_boost * 0.5
            if keyword_score > 0:
                scored.append((score, dict(row)))

        scored.sort(key=lambda x: -x[0])
        results = [item[1] for item in scored[:max_results]]
        return results

    def query_fishing_log(self, query):
        """
        Read-only SQL against the visible database.
        Budget: 2 per day. Only SELECT/WITH allowed.
        """
        if not self.ablation.get("query_fishing_log", True):
            return {"error": "query_fishing_log is disabled in this configuration"}
        if self._budgets.get("query_fishing_log", 0) <= 0:
            return {"error": "query_fishing_log budget exhausted for today"}

        # Validate: only SELECT and WITH
        stripped = query.strip().upper()
        if not (stripped.startswith("SELECT") or stripped.startswith("WITH")):
            return {"error": "Only SELECT and WITH statements are allowed"}

        # Reject write operations even if embedded
        write_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "REPLACE"]
        for kw in write_keywords:
            if kw in stripped:
                return {"error": f"Write operation '{kw}' not allowed"}

        self._budgets["query_fishing_log"] -= 1
        self._step_tool_usage.append("query_fishing_log")

        try:
            cur = self._db.cursor()
            cur.execute(query)
            rows = cur.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            return {"error": str(e)}

    def analyze_data(self, code):
        """
        Sandboxed Python execution with math, statistics, random.
        Budget: 1 per day. Returns stdout.
        """
        if not self.ablation.get("analyze_data", True):
            return {"error": "analyze_data is disabled in this configuration"}
        if self._budgets.get("analyze_data", 0) <= 0:
            return {"error": "analyze_data budget exhausted for today"}
        self._budgets["analyze_data"] -= 1
        self._step_tool_usage.append("analyze_data")

        import math as math_mod
        import statistics as stats_mod

        safe_globals = {
            "__builtins__": {
                "print": print, "range": range, "len": len, "sum": sum,
                "min": min, "max": max, "abs": abs, "round": round,
                "sorted": sorted, "enumerate": enumerate, "zip": zip,
                "map": map, "filter": filter, "list": list, "dict": dict,
                "tuple": tuple, "set": set, "int": int, "float": float,
                "str": str, "bool": bool, "True": True, "False": False,
                "None": None, "isinstance": isinstance, "type": type,
            },
            "math": math_mod,
            "statistics": stats_mod,
            "random": random,
        }

        stdout_capture = StringIO()
        try:
            with redirect_stdout(stdout_capture):
                exec(code, safe_globals)
            return stdout_capture.getvalue()
        except Exception as e:
            return f"Error: {e}"

    def evaluate_options(self, params):
        """
        Expected value calculator. Takes storm_probability, zone_a_danger_probability.
        Returns expected reward for all 6 action combinations.
        Budget: 1 per day.
        """
        if not self.ablation.get("evaluate_options", True):
            return {"error": "evaluate_options is disabled in this configuration"}
        if self._budgets.get("evaluate_options", 0) <= 0:
            return {"error": "evaluate_options budget exhausted for today"}
        self._budgets["evaluate_options"] -= 1
        self._step_tool_usage.append("evaluate_options")

        p_storm_raw = params.get("storm_probability", 0.5)
        p_zone_a_raw = params.get("zone_a_danger_probability", 0.5)

        p_storm = max(0.0, min(1.0, p_storm_raw))
        p_zone_a_danger = max(0.0, min(p_storm, p_zone_a_raw))
        p_zone_b_danger = p_storm - p_zone_a_danger
        p_no_storm = 1.0 - p_storm

        # Build note if values were clamped
        note = None
        if abs(p_zone_a_danger - p_zone_a_raw) > 1e-6:
            note = (
                f"zone_a_danger_probability was clamped from {p_zone_a_raw} to "
                f"{p_zone_a_danger} (cannot exceed storm_probability={p_storm})"
            )

        # Convert to 4-state belief vector
        if p_storm > 0:
            wind_n_ratio = p_zone_a_danger / p_storm
        else:
            wind_n_ratio = 0.5
        belief = np.array([
            p_no_storm * wind_n_ratio,
            p_no_storm * (1 - wind_n_ratio),
            p_zone_a_danger,
            p_zone_b_danger,
        ])
        if belief.sum() > 0:
            belief /= belief.sum()

        # Compute expected reward for all 6 combos (unsorted)
        results = []
        for zone in self.cfg["zones"]:
            for boats in range(1, self.cfg["max_boats"] + 1):
                er = self.pomdp.expected_reward(belief, zone, boats)
                results.append({
                    "zone": zone,
                    "boats": boats,
                    "expected_reward": round(er, 4),
                })

        output = {
            "expected_rewards": results,
            "belief_used": {
                "storm_probability": p_storm,
                "zone_a_danger_probability": p_zone_a_danger,
            },
        }
        if note:
            output["note"] = note
        return output

    def forecast_scenario(self, params):
        """
        Projects forward N days under a specified scenario.
        Uses the transition model but does NOT advance the live episode.
        Budget: 1 per day.
        """
        if not self.ablation.get("forecast_scenario", True):
            return {"error": "forecast_scenario is disabled in this configuration"}
        if self._budgets.get("forecast_scenario", 0) <= 0:
            return {"error": "forecast_scenario budget exhausted for today"}
        self._budgets["forecast_scenario"] -= 1
        self._step_tool_usage.append("forecast_scenario")

        horizon = params.get("horizon_days", 5)
        assume_storm = params.get("assume_storm_persists", True)
        assume_zone = params.get("assume_zone", "A")
        strategy = params.get("strategy", {"zone": "A", "boats": 2})

        projected_days = []
        cumulative = 0.0

        for d in range(horizon):
            # Under the scenario assumption
            if assume_storm:
                storm = 1
                affected = assume_zone
            else:
                storm = 0
                affected = assume_zone  # doesn't matter if no storm

            s_zone = strategy["zone"]
            s_boats = strategy["boats"]

            if storm == 1 and s_zone == affected:
                reward = self.cfg["danger_loss_per_boat"] * s_boats
            else:
                reward = self.cfg["safe_profit_per_boat"] * s_boats

            cumulative += reward
            projected_days.append({
                "projected_day": self._day + d + 1,
                "reward": reward,
                "cumulative": cumulative,
            })

        return {
            "scenario": params,
            "projected_daily_outcomes": projected_days,
            "projected_cumulative_reward": cumulative,
        }

    def read_barometer(self):
        """
        LLM-accessible: read today's barometric pressure.
        Budget: 1 per day.
        """
        if not self.ablation.get("read_barometer", True):
            return {"error": "read_barometer is disabled in this configuration"}
        if self._budgets.get("read_barometer", 0) <= 0:
            return {"error": "read_barometer budget exhausted for today"}
        self._budgets["read_barometer"] -= 1
        self._step_tool_usage.append("read_barometer")
        reading = float(self._available_barometer)
        self._step_observations.append(("barometer", reading))
        return reading

    def read_buoy(self, zone):
        """
        LLM-accessible: read wave height from a buoy in the specified zone.
        Budget: 2 per day.
        """
        if not self.ablation.get("read_buoy", True):
            return {"error": "read_buoy is disabled in this configuration"}
        if self._budgets.get("read_buoy", 0) <= 0:
            return {"error": "read_buoy budget exhausted for today"}
        self._budgets["read_buoy"] -= 1
        self._step_tool_usage.append("read_buoy")
        if zone == "A":
            reading = float(self._available_buoy_a)
        else:
            reading = float(self._available_buoy_b)
        self._step_observations.append((("buoy", zone), reading))
        return reading

    def get_barometer(self):
        """
        Internal method for baselines (no budget tracking).
        Records as an observation for the evaluator.
        """
        reading = float(self._available_barometer)
        self._step_observations.append(("barometer", reading))
        return reading

    def get_buoy(self, zone):
        """
        Internal method for baselines (no budget tracking).
        Records as an observation for the evaluator.
        """
        if zone == "A":
            reading = float(self._available_buoy_a)
        else:
            reading = float(self._available_buoy_b)
        self._step_observations.append((("buoy", zone), reading))
        return reading

    # =========================================================================
    # Submit decisions — the ONLY action that advances the day
    # =========================================================================

    def submit_decisions(self, zone, boats, beliefs, reasoning=""):
        """
        Submit fishing decisions. This advances the day.

        Args:
            zone: "A" or "B"
            boats: 1, 2, or 3
            beliefs: dict with "storm_active" and "zone_a_is_dangerous" (floats 0-1)
            reasoning: optional string

        Returns:
            dict with observation, reward, done, info
        """
        assert self._episode_active, "Episode not active. Call reset() first."
        assert zone in self.cfg["zones"], f"Invalid zone: {zone}"
        assert 1 <= boats <= self.cfg["max_boats"], f"Invalid boats: {boats}"

        # Compute reward from hidden state
        reward = self.pomdp.reward(self._hidden_state_idx, zone, boats)

        # Update visible database
        self._cumulative_reward += reward
        cur = self._db.cursor()
        cur.execute(
            "INSERT INTO daily_log (day, sea_color, zone_fished, boats_sent, reward, cumulative) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (self._day, self._current_sea_color, zone, boats, reward, self._cumulative_reward),
        )
        cur.execute(
            "INSERT INTO catch_history (day, zone, boats, reward) VALUES (?, ?, ?, ?)",
            (self._day, zone, boats, reward),
        )
        self._db.commit()

        # Record trace entry for evaluator
        self._trace.append({
            "day": self._day,
            "hidden_state_idx": self._hidden_state_idx,
            "hidden_state": self._hidden_state,
            "sea_color": self._current_sea_color,
            "observations": list(self._step_observations),
            "available_observations": self._get_all_available_observations(),
            "signals": self._current_signals,
            "tools_used": list(self._step_tool_usage),
            "action": {"zone": zone, "boats": boats},
            "beliefs": dict(beliefs),
            "reward": reward,
            "reasoning": reasoning,
        })

        # Advance the day
        self._yesterday_reward = reward
        self._yesterday_zone = zone
        self._yesterday_boats = boats
        self._day += 1
        done = self._day > self.cfg["episode_length"]

        if done:
            self._episode_active = False
            return {
                "observation": None,
                "reward": reward,
                "done": True,
                "info": {"cumulative_reward": self._cumulative_reward},
            }

        # Transition hidden state
        self._transition_hidden_state()

        # Reset daily budgets and observations
        self._reset_daily_budgets()
        self._step_tool_usage = []

        # Sample new observations
        self._current_sea_color = self._sample_sea_color()
        self._step_observations = [("sea_color", self._current_sea_color)]
        self._current_signals = self._emit_signals()
        self._store_signals(self._current_signals)

        # Pre-sample tool observations for new day
        self._available_barometer = self._sample_barometer()
        self._available_buoy_a = self._sample_buoy("A")
        self._available_buoy_b = self._sample_buoy("B")

        obs = self._make_observation_bundle()
        return {
            "observation": obs,
            "reward": reward,
            "done": False,
            "info": {},
        }

    def _transition_hidden_state(self):
        """Sample next hidden state from transition matrix."""
        row = self.cfg["transition_matrix"][self._hidden_state_idx]
        self._hidden_state_idx = self._rng.choices(range(4), weights=row, k=1)[0]
        self._hidden_state = self.cfg["states"][self._hidden_state_idx]

    def _get_all_available_observations(self):
        """
        Return all observations that COULD have been gathered this step
        (for the evaluator's oracle computation).
        """
        return [
            ("sea_color", self._current_sea_color),
            ("barometer", float(self._available_barometer)),
            (("buoy", "A"), float(self._available_buoy_a)),
            (("buoy", "B"), float(self._available_buoy_b)),
        ]

    def get_trace(self):
        """Return the full episode trace for the evaluator."""
        return list(self._trace)
