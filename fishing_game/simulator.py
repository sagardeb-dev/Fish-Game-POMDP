"""
Fishing Game Simulator v4 — Discoverable causal structure.

40 hidden states = 2(storm) x 4(wind) x 5(equip_failure).
4 zones: A, B, C, D (ring topology for wave propagation).
2 independent risks: storm + equipment failure.

Two-tier observation model:
  Tier 1 (free): sea_color, equip_indicator, barometer, 4x maintenance_alerts (7 obs)
  Tier 2 (SQL-discoverable): 4x buoy readings, 4x equipment inspections (8 obs)

Tier 2 observations appear in the text bundle but only count for the evaluator's
POMDP belief computation if the agent used SQL tools to discover the causal structure.
The enriched historical DB (sensor_log + catch_history) makes wave propagation and
age confound patterns discoverable through data analysis.

Budget-gated tools: search, SQL queries, analysis, optimizer, forecast.
All parameters derive from the shared config.
"""

import json
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
    OpenEnv-compatible fishing game environment v4.
    All interaction through reset(), tool calls, and submit_decisions().
    Two-tier observation model: Tier 1 (free) + Tier 2 (SQL-discoverable).
    """

    def __init__(self, config=None, ablation=None):
        self.cfg = config or CONFIG
        self.pomdp = FishingPOMDP(self.cfg)
        self.ablation = ablation or {t: True for t in self.cfg["tool_budgets"]}
        self._episode_active = False

    def reset(self, seed=None):
        """Initialize a new episode. Returns initial observation bundle."""
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(
            seed if seed is not None else self._rng.randint(0, 2**31)
        )

        self._day = 1
        self._cumulative_reward = 0.0
        self._yesterday_reward = 0.0
        self._yesterday_allocation = None
        self._episode_active = True

        # Sample initial hidden state (uniform over 40)
        self._hidden_state_idx = self._sample_initial_state()
        self._hidden_state = self.cfg["states"][self._hidden_state_idx]

        # Episode trace for evaluator
        self._trace = []

        # Create episode-local SQLite database (in-memory)
        self._db = sqlite3.connect(":memory:")
        self._db.row_factory = sqlite3.Row
        self._init_db()
        self._generate_historical_data()

        # Tool budgets for current day
        self._reset_daily_budgets()

        # Emit observations for day 1
        self._current_sea_color = self._sample_sea_color()
        self._current_equip_indicator = self._sample_equip_indicator()
        self._current_weather_signals = self._emit_weather_signals()
        self._current_equip_signals = self._emit_equipment_signals()
        self._store_signals(self._current_weather_signals, "weather")
        self._store_signals(self._current_equip_signals, "equipment")

        # Pre-sample all free sensor readings for this day
        self._available_barometer = self._sample_barometer()
        self._available_buoys = {z: self._sample_buoy(z) for z in self.cfg["zones"]}
        self._available_inspections = {z: self._sample_inspection(z) for z in self.cfg["zones"]}
        self._available_maintenance_alerts = self._sample_maintenance_alerts()
        self._store_maintenance_alerts(self._available_maintenance_alerts)

        # Track tool usage for this step
        self._step_tool_usage = []

        return self._make_observation_bundle()

    def _sample_initial_state(self):
        """Sample initial state from uniform distribution over 40 states."""
        return self._rng.randint(0, self.pomdp.n_states - 1)

    def _init_db(self):
        """Create the visible database tables."""
        cur = self._db.cursor()
        cur.execute("""
            CREATE TABLE daily_log (
                day              INTEGER PRIMARY KEY,
                sea_color        TEXT,
                equip_indicator  TEXT,
                allocation       TEXT,
                reward           REAL,
                cumulative       REAL
            )
        """)
        cur.execute("""
            CREATE TABLE weather_signals (
                signal_id    TEXT PRIMARY KEY,
                day          INTEGER,
                source_type  TEXT,
                report_type  TEXT,
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
        cur.execute("""
            CREATE TABLE daily_conditions (
                day            INTEGER PRIMARY KEY,
                storm_active   INTEGER,
                storm_zone     TEXT,
                equip_zone     TEXT,
                barometer      REAL,
                sea_color      TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE maintenance_log (
                day          INTEGER,
                zone         TEXT,
                alerts       INTEGER
            )
        """)
        cur.execute("""
            CREATE TABLE sensor_log (
                day                INTEGER,
                zone               TEXT,
                buoy_reading       REAL,
                equipment_reading  REAL
            )
        """)
        self._db.commit()

    def _generate_historical_data(self, n_days=30):
        """Pre-seed catch_history, maintenance_log, and sensor_log with historical data."""
        hist_rng = random.Random(self._rng.randint(0, 2**31))
        hist_np_rng = np.random.RandomState(hist_rng.randint(0, 2**31))

        state_idx = hist_rng.randint(0, self.pomdp.n_states - 1)
        cur = self._db.cursor()

        for day in range(-n_days, 0):
            state = self.cfg["states"][state_idx]
            storm, wind, equip = state
            storm_zone = self.cfg["wind_to_zone"][wind] if storm == 1 else None
            equip_zone = self.cfg["equip_to_zone"][equip]

            # Sample barometer and sea_color for this historical day
            baro_params = self.cfg["barometer_params"][storm]
            hist_barometer = float(hist_np_rng.normal(baro_params["mean"], baro_params["std"]))
            sc_probs = self.cfg["sea_color_probs"][storm]
            sc_colors = list(sc_probs.keys())
            sc_weights = [sc_probs[c] for c in sc_colors]
            hist_sea_color = hist_rng.choices(sc_colors, weights=sc_weights, k=1)[0]

            # Daily conditions (1 row per day — joinable to any per-zone table)
            cur.execute(
                "INSERT INTO daily_conditions "
                "(day, storm_active, storm_zone, equip_zone, barometer, sea_color) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (day, storm, storm_zone, equip_zone,
                 round(hist_barometer, 2), hist_sea_color),
            )

            # Prior captain picks a random zone and sends 3 boats
            zone = hist_rng.choice(self.cfg["zones"])
            alloc = {z: 0 for z in self.cfg["zones"]}
            alloc[zone] = 3
            reward = self.pomdp.reward(state_idx, alloc)
            cur.execute(
                "INSERT INTO catch_history (day, zone, boats, reward) VALUES (?, ?, ?, ?)",
                (day, zone, 3, reward),
            )

            # Sensor log: buoy + equipment readings for all 4 zones
            for z in self.cfg["zones"]:
                buoy_val = self._sample_buoy_for_state(z, state, hist_np_rng)
                equip_val = self._sample_inspection_for_state(z, state, hist_np_rng)
                cur.execute(
                    "INSERT INTO sensor_log (day, zone, buoy_reading, equipment_reading) "
                    "VALUES (?, ?, ?, ?)",
                    (day, z, round(float(buoy_val), 2), round(float(equip_val), 2)),
                )

            # Maintenance alerts with age confound
            maint_params = self.cfg["maintenance_alert_params"]
            for z in self.cfg["zones"]:
                age = self.cfg["zone_infrastructure_age"][z]
                rate = age * maint_params["age_rate_factor"]
                if equip_zone == z:
                    rate += maint_params["failure_signal"]
                alerts = int(hist_np_rng.poisson(rate))
                cur.execute(
                    "INSERT INTO maintenance_log (day, zone, alerts) VALUES (?, ?, ?)",
                    (day, z, alerts),
                )

            # Transition
            row = self.pomdp.T[state_idx]
            state_idx = hist_rng.choices(range(self.pomdp.n_states), weights=row.tolist(), k=1)[0]

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

    def _sample_equip_indicator(self):
        """Sample equipment indicator from distribution given hidden state."""
        equip = self._hidden_state[2]
        key = 1 if equip > 0 else 0
        probs = self.cfg["equip_indicator_probs"][key]
        levels = list(probs.keys())
        weights = [probs[l] for l in levels]
        return self._rng.choices(levels, weights=weights, k=1)[0]

    def _sample_barometer(self):
        """Pre-sample barometer reading for this step."""
        storm = self._hidden_state[0]
        params = self.cfg["barometer_params"][storm]
        return self._np_rng.normal(params["mean"], params["std"])

    def _sample_buoy(self, zone):
        """Pre-sample buoy reading with wave propagation model."""
        storm, wind, _ = self._hidden_state
        if storm == 0:
            params = self.cfg["buoy_params"]["normal"]
        else:
            storm_zone = self.cfg["wind_to_zone"][wind]
            distance = self.cfg["zone_adjacency"][storm_zone][zone]
            if distance == 0:
                params = self.cfg["buoy_params"]["source"]
            elif distance == 1:
                params = self.cfg["buoy_params"]["propagated"]
            else:
                params = self.cfg["buoy_params"]["far_propagated"]
        return self._np_rng.normal(params["mean"], params["std"])

    def _sample_inspection(self, zone):
        """Pre-sample equipment inspection reading with age confound."""
        _, _, equip = self._hidden_state
        equip_zone = self.cfg["equip_to_zone"][equip]
        if equip_zone == zone:
            params = self.cfg["equipment_inspection_params"]["broken"]
        else:
            params = self.cfg["equipment_inspection_params"]["ok"]
        age_offset = self.cfg["zone_infrastructure_age"][zone] * self.cfg["equipment_age_offset_factor"]
        return self._np_rng.normal(params["mean"] + age_offset, params["std"])

    def _sample_maintenance_alerts(self):
        """Sample Poisson maintenance alerts per zone with age confound."""
        _, _, equip = self._hidden_state
        equip_zone = self.cfg["equip_to_zone"][equip]
        maint_params = self.cfg["maintenance_alert_params"]
        alerts = {}
        for zone in self.cfg["zones"]:
            age = self.cfg["zone_infrastructure_age"][zone]
            rate = age * maint_params["age_rate_factor"]
            if equip_zone == zone:
                rate += maint_params["failure_signal"]
            alerts[zone] = int(self._np_rng.poisson(rate))
        return alerts

    def _sample_buoy_for_state(self, zone, state, np_rng):
        """Sample a buoy reading for a given zone and explicit state tuple."""
        storm, wind, _ = state
        if storm == 0:
            params = self.cfg["buoy_params"]["normal"]
        else:
            storm_zone = self.cfg["wind_to_zone"][wind]
            distance = self.cfg["zone_adjacency"][storm_zone][zone]
            if distance == 0:
                params = self.cfg["buoy_params"]["source"]
            elif distance == 1:
                params = self.cfg["buoy_params"]["propagated"]
            else:
                params = self.cfg["buoy_params"]["far_propagated"]
        return np_rng.normal(params["mean"], params["std"])

    def _sample_inspection_for_state(self, zone, state, np_rng):
        """Sample an equipment inspection reading for a given zone and explicit state tuple."""
        _, _, equip = state
        equip_zone = self.cfg["equip_to_zone"][equip]
        if equip_zone == zone:
            params = self.cfg["equipment_inspection_params"]["broken"]
        else:
            params = self.cfg["equipment_inspection_params"]["ok"]
        age_offset = self.cfg["zone_infrastructure_age"][zone] * self.cfg["equipment_age_offset_factor"]
        return np_rng.normal(params["mean"] + age_offset, params["std"])

    def _store_maintenance_alerts(self, alerts):
        """Store today's maintenance alerts in the database."""
        cur = self._db.cursor()
        for zone, count in alerts.items():
            cur.execute(
                "INSERT INTO maintenance_log (day, zone, alerts) VALUES (?, ?, ?)",
                (self._day, zone, count),
            )
        self._db.commit()

    def _build_free_observations(self):
        """Build Tier 1 observations — always free, no SQL needed (7 obs)."""
        obs = [
            ("sea_color", self._current_sea_color),
            ("equip_indicator", self._current_equip_indicator),
            ("barometer", float(self._available_barometer)),
        ]
        for zone in self.cfg["zones"]:
            obs.append((("maintenance_alerts", zone), self._available_maintenance_alerts[zone]))
        return obs

    def _build_sql_discoverable_observations(self):
        """Build Tier 2 observations — only count if agent used SQL tools (8 obs)."""
        obs = []
        for zone in self.cfg["zones"]:
            obs.append((("buoy", zone), float(self._available_buoys[zone])))
        for zone in self.cfg["zones"]:
            obs.append((("equip_inspection", zone), float(self._available_inspections[zone])))
        return obs

    def _emit_weather_signals(self):
        """Emit weather signals based on storm state."""
        lo, hi = self.cfg["signals_per_step_range"]
        n_signals = self._rng.randint(lo, hi)
        storm = self._hidden_state[0]
        signals = []

        for _ in range(n_signals):
            tier = self._rng.choice([1, 2, 3])
            tier_cfg = self.cfg["signal_tiers"][tier]

            if tier == 3:
                emit_prob = tier_cfg["emission_prob_always"]
            else:
                emit_prob = tier_cfg["emission_prob"][storm]

            if self._rng.random() < emit_prob:
                headline = self._rng.choice(tier_cfg["headlines"])
                source = self._rng.choice(self.cfg["signal_sources"])
                signals.append({
                    "signal_id": str(uuid.uuid4())[:8],
                    "day": self._day,
                    "source_type": source,
                    "report_type": "weather",
                    "headline": headline,
                    "body": f"Detailed report: {headline.lower()}. Source: {source}. Day {self._day} update.",
                    "_tier": tier,
                    "_storm": storm,
                    "_wind": self._hidden_state[1],
                })

        # Ensure minimum signals
        while len(signals) < lo:
            tier_cfg = self.cfg["signal_tiers"][3]
            headline = self._rng.choice(tier_cfg["headlines"])
            source = self._rng.choice(self.cfg["signal_sources"])
            signals.append({
                "signal_id": str(uuid.uuid4())[:8],
                "day": self._day,
                "source_type": source,
                "report_type": "weather",
                "headline": headline,
                "body": f"Detailed report: {headline.lower()}. Source: {source}. Day {self._day} update.",
                "_tier": 3,
                "_storm": storm,
                "_wind": self._hidden_state[1],
            })

        return signals

    def _emit_equipment_signals(self):
        """Emit equipment signals based on equipment failure state."""
        lo, hi = self.cfg["signals_per_step_range"]
        n_signals = self._rng.randint(lo, hi)
        equip = self._hidden_state[2]
        equip_active = 1 if equip > 0 else 0
        signals = []

        for _ in range(n_signals):
            tier = self._rng.choice([1, 2, 3])
            tier_cfg = self.cfg["equipment_signal_tiers"][tier]

            if tier == 3:
                emit_prob = tier_cfg["emission_prob_always"]
            else:
                emit_prob = tier_cfg["emission_prob"][equip_active]

            if self._rng.random() < emit_prob:
                headline = self._rng.choice(tier_cfg["headlines"])
                source = self._rng.choice(self.cfg["signal_sources"])
                signals.append({
                    "signal_id": str(uuid.uuid4())[:8],
                    "day": self._day,
                    "source_type": source,
                    "report_type": "equipment",
                    "headline": headline,
                    "body": f"Detailed report: {headline.lower()}. Source: {source}. Day {self._day} update.",
                    "_tier": tier,
                    "_equip": equip,
                })

        # Ensure minimum signals
        while len(signals) < lo:
            tier_cfg = self.cfg["equipment_signal_tiers"][3]
            headline = self._rng.choice(tier_cfg["headlines"])
            source = self._rng.choice(self.cfg["signal_sources"])
            signals.append({
                "signal_id": str(uuid.uuid4())[:8],
                "day": self._day,
                "source_type": source,
                "report_type": "equipment",
                "headline": headline,
                "body": f"Detailed report: {headline.lower()}. Source: {source}. Day {self._day} update.",
                "_tier": 3,
                "_equip": equip,
            })

        return signals

    def _store_signals(self, signals, report_type):
        """Store signals in the visible database (without hidden fields)."""
        cur = self._db.cursor()
        for sig in signals:
            cur.execute(
                "INSERT INTO weather_signals (signal_id, day, source_type, report_type, headline, body) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (sig["signal_id"], sig["day"], sig["source_type"],
                 report_type, sig["headline"], sig["body"]),
            )
        self._db.commit()

    def _make_observation_bundle(self):
        """Create the observation bundle the agent receives each day."""
        available_tools = [t for t in self.cfg["tool_budgets"] if self.ablation.get(t, True)]
        budget = {t: self._budgets[t] for t in available_tools}

        return {
            "day": self._day,
            "days_remaining": self.cfg["episode_length"] - self._day,
            # Free categorical observations
            "sea_color": self._current_sea_color,
            "equip_indicator": self._current_equip_indicator,
            # Free continuous sensors
            "barometer": round(float(self._available_barometer), 2),
            "buoy_readings": {z: round(float(self._available_buoys[z]), 2) for z in self.cfg["zones"]},
            "equipment_readings": {z: round(float(self._available_inspections[z]), 2) for z in self.cfg["zones"]},
            "maintenance_alerts": dict(self._available_maintenance_alerts),
            # Zone metadata (constant, provided for convenience)
            "zone_infrastructure_ages": dict(self.cfg["zone_infrastructure_age"]),
            # Episode context
            "yesterday_reward": self._yesterday_reward,
            "yesterday_allocation": self._yesterday_allocation,
            "cumulative_reward": self._cumulative_reward,
            "tools_available": available_tools,
            "tool_budget": budget,
            "db_schema": (
                "Available tables:\n\n"
                "daily_log (day, sea_color, equip_indicator, allocation, reward, cumulative)\n"
                "  - Your complete fishing log\n\n"
                "weather_signals (signal_id, day, source_type, report_type, headline, body)\n"
                "  - Weather and equipment intelligence reports (report_type: 'weather' or 'equipment')\n\n"
                "daily_conditions (day, storm_active, storm_zone, equip_zone, barometer, sea_color)\n"
                "  - One row per day: actual conditions (30 days of pre-episode history + current season)\n"
                "  - JOIN to sensor_log or catch_history on day\n\n"
                "catch_history (day, zone, boats, reward)\n"
                "  - Historical catch outcomes by zone (includes 30 days of pre-episode history)\n\n"
                "maintenance_log (day, zone, alerts)\n"
                "  - Maintenance alert counts per zone per day (includes 30 days of pre-episode history)\n\n"
                "sensor_log (day, zone, buoy_reading, equipment_reading)\n"
                "  - Historical sensor readings per zone per day (includes 30 days of pre-episode history)"
            ),
        }

    # =========================================================================
    # Tool methods
    # =========================================================================

    def check_weather_reports(self, query, max_results=3):
        """
        Keyword search over weather-type signals.
        Budget: 2 per day.
        """
        if not self.ablation.get("check_weather_reports", True):
            return {"error": "check_weather_reports is disabled in this configuration"}
        if self._budgets.get("check_weather_reports", 0) <= 0:
            return {"error": "check_weather_reports budget exhausted for today"}
        self._budgets["check_weather_reports"] -= 1
        self._step_tool_usage.append("check_weather_reports")

        cur = self._db.cursor()
        cur.execute(
            "SELECT signal_id, day, source_type, report_type, headline, body "
            "FROM weather_signals WHERE report_type='weather'"
        )
        rows = cur.fetchall()

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
        return [item[1] for item in scored[:max_results]]

    def check_equipment_reports(self, query, max_results=3):
        """
        Keyword search over equipment-type signals.
        Budget: 2 per day.
        """
        if not self.ablation.get("check_equipment_reports", True):
            return {"error": "check_equipment_reports is disabled in this configuration"}
        if self._budgets.get("check_equipment_reports", 0) <= 0:
            return {"error": "check_equipment_reports budget exhausted for today"}
        self._budgets["check_equipment_reports"] -= 1
        self._step_tool_usage.append("check_equipment_reports")

        cur = self._db.cursor()
        cur.execute(
            "SELECT signal_id, day, source_type, report_type, headline, body "
            "FROM weather_signals WHERE report_type='equipment'"
        )
        rows = cur.fetchall()

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
        return [item[1] for item in scored[:max_results]]

    def query_fishing_log(self, query):
        """
        Read-only SQL against the visible database.
        Budget: 2 per day. Only SELECT/WITH allowed.
        """
        if not self.ablation.get("query_fishing_log", True):
            return {"error": "query_fishing_log is disabled in this configuration"}
        if self._budgets.get("query_fishing_log", 0) <= 0:
            return {"error": "query_fishing_log budget exhausted for today"}

        stripped = query.strip().upper()
        if not (stripped.startswith("SELECT") or stripped.startswith("WITH")):
            return {"error": "Only SELECT and WITH statements are allowed"}

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

    def query_maintenance_log(self, query):
        """
        Read-only SQL against the visible database (maintenance_log + others).
        Budget: 2 per day. Only SELECT/WITH allowed.
        """
        if not self.ablation.get("query_maintenance_log", True):
            return {"error": "query_maintenance_log is disabled in this configuration"}
        if self._budgets.get("query_maintenance_log", 0) <= 0:
            return {"error": "query_maintenance_log budget exhausted for today"}

        stripped = query.strip().upper()
        if not (stripped.startswith("SELECT") or stripped.startswith("WITH")):
            return {"error": "Only SELECT and WITH statements are allowed"}

        write_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "REPLACE"]
        for kw in write_keywords:
            if kw in stripped:
                return {"error": f"Write operation '{kw}' not allowed"}

        self._budgets["query_maintenance_log"] -= 1
        self._step_tool_usage.append("query_maintenance_log")

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
        Budget: 1 per day.
        """
        if not self.ablation.get("analyze_data", True):
            return {"error": "analyze_data is disabled in this configuration"}
        if self._budgets.get("analyze_data", 0) <= 0:
            return {"error": "analyze_data budget exhausted for today"}
        self._budgets["analyze_data"] -= 1
        self._step_tool_usage.append("analyze_data")

        import math as math_mod
        import statistics as stats_mod
        import collections as collections_mod
        import itertools as itertools_mod
        import functools as functools_mod

        ALLOWED_MODULES = {
            "math": math_mod,
            "statistics": stats_mod,
            "random": random,
            "collections": collections_mod,
            "itertools": itertools_mod,
            "functools": functools_mod,
        }

        def _safe_import(name, *args, **kwargs):
            if name in ALLOWED_MODULES:
                return ALLOWED_MODULES[name]
            raise ImportError(f"Module '{name}' is not available in sandbox. "
                              f"Available: {', '.join(sorted(ALLOWED_MODULES))}")

        safe_globals = {
            "__builtins__": {
                "__import__": _safe_import,
                "print": print, "range": range, "len": len, "sum": sum,
                "min": min, "max": max, "abs": abs, "round": round,
                "sorted": sorted, "enumerate": enumerate, "zip": zip,
                "map": map, "filter": filter, "list": list, "dict": dict,
                "tuple": tuple, "set": set, "int": int, "float": float,
                "str": str, "bool": bool, "True": True, "False": False,
                "None": None, "isinstance": isinstance, "type": type,
                "any": any, "all": all, "next": next, "iter": iter,
                "reversed": reversed, "hash": hash, "repr": repr,
                "pow": pow, "divmod": divmod, "frozenset": frozenset,
                "getattr": getattr, "setattr": setattr, "hasattr": hasattr,
                "callable": callable, "format": format,
            },
            **ALLOWED_MODULES,
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
        Expected value calculator for allocations.
        Takes expanded beliefs, returns expected reward for top allocations.
        Budget: 1 per day.
        """
        if not self.ablation.get("evaluate_options", True):
            return {"error": "evaluate_options is disabled in this configuration"}
        if self._budgets.get("evaluate_options", 0) <= 0:
            return {"error": "evaluate_options budget exhausted for today"}
        self._budgets["evaluate_options"] -= 1
        self._step_tool_usage.append("evaluate_options")

        # Extract beliefs from params
        p_storm = max(0.0, min(1.0, params.get("storm_active", 0.5)))
        storm_zone_probs = params.get("storm_zone_probs", {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        p_equip = max(0.0, min(1.0, params.get("equip_failure_active", 0.2)))
        equip_zone_probs = params.get("equip_zone_probs", {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})

        # Build belief vector from marginals
        belief = self._beliefs_to_vector(p_storm, storm_zone_probs, p_equip, equip_zone_probs)

        # Compute expected reward for all allocations, return top 10
        results = []
        for alloc in self.cfg["valid_allocations"]:
            er = self.pomdp.expected_reward(belief, alloc)
            results.append({
                "allocation": alloc,
                "expected_reward": round(er, 4),
            })
        results.sort(key=lambda x: -x["expected_reward"])

        return {
            "top_allocations": results[:10],
            "total_allocations_evaluated": len(results),
            "beliefs_used": {
                "storm_active": p_storm,
                "storm_zone_probs": storm_zone_probs,
                "equip_failure_active": p_equip,
                "equip_zone_probs": equip_zone_probs,
            },
        }

    def _beliefs_to_vector(self, p_storm, storm_zone_probs, p_equip, equip_zone_probs):
        """Convert 10 marginal beliefs to 40-element belief vector under independence."""
        wind_labels = ["N", "S", "E", "W"]
        zone_to_wind = {v: k for k, v in self.cfg["wind_to_zone"].items()}

        belief = np.zeros(self.pomdp.n_states, dtype=np.float64)
        for i, (storm, wind, equip) in enumerate(self.cfg["states"]):
            # P(storm)
            p_s = p_storm if storm == 1 else (1.0 - p_storm)

            # P(wind | storm) -- use storm_zone_probs for conditional
            zone_for_wind = self.cfg["wind_to_zone"][wind]
            szp = storm_zone_probs.get(zone_for_wind, 0.25)
            # Normalize storm_zone_probs to sum to 1
            szp_sum = sum(storm_zone_probs.get(z, 0.25) for z in self.cfg["zones"])
            if szp_sum > 0:
                p_w = szp / szp_sum
            else:
                p_w = 0.25

            # P(equip)
            if equip == 0:
                p_e = 1.0 - p_equip
            else:
                equip_zone = self.cfg["equip_to_zone"][equip]
                ezp = equip_zone_probs.get(equip_zone, 0.25)
                ezp_sum = sum(equip_zone_probs.get(z, 0.25) for z in self.cfg["zones"])
                if ezp_sum > 0:
                    p_e = p_equip * (ezp / ezp_sum)
                else:
                    p_e = p_equip * 0.25

            belief[i] = p_s * p_w * p_e

        total = belief.sum()
        if total > 0:
            belief /= total
        else:
            belief = np.ones(self.pomdp.n_states, dtype=np.float64) / self.pomdp.n_states
        return belief

    def forecast_scenario(self, params):
        """
        Projects forward N days under a specified dual-risk scenario.
        Does NOT advance the live episode.
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
        assume_storm_zone = params.get("assume_storm_zone", "A")
        assume_equip_failure = params.get("assume_equip_failure", False)
        assume_equip_zone = params.get("assume_equip_zone", "A")
        strategy = params.get("strategy", {"A": 3, "B": 0, "C": 0, "D": 0})

        projected_days = []
        cumulative = 0.0

        for d in range(horizon):
            day_reward = 0.0
            for zone in self.cfg["zones"]:
                boats = strategy.get(zone, 0)
                if boats == 0:
                    continue

                has_storm = assume_storm and (zone == assume_storm_zone)
                has_equip = assume_equip_failure and (zone == assume_equip_zone)

                if has_storm and has_equip:
                    day_reward += self.cfg["danger_loss_both_per_boat"] * boats
                elif has_storm:
                    day_reward += self.cfg["danger_loss_per_boat"] * boats
                elif has_equip:
                    day_reward += self.cfg["danger_loss_equip_per_boat"] * boats
                else:
                    day_reward += self.cfg["safe_profit_per_boat"] * boats

            cumulative += day_reward
            projected_days.append({
                "projected_day": self._day + d + 1,
                "reward": day_reward,
                "cumulative": cumulative,
            })

        return {
            "scenario": params,
            "projected_daily_outcomes": projected_days,
            "projected_cumulative_reward": cumulative,
        }

    def _agent_used_sql_tools(self):
        """Check if the agent used any SQL tools this step."""
        sql_tools = {"query_fishing_log", "query_maintenance_log"}
        return bool(sql_tools & set(self._step_tool_usage))

    # Internal accessors for baselines (no budget — sensors are free in v3)

    def get_barometer(self):
        """Internal: read barometer. Already in step_observations."""
        return float(self._available_barometer)

    def get_buoy(self, zone):
        """Internal: read buoy. Already in step_observations."""
        return float(self._available_buoys[zone])

    def get_inspection(self, zone):
        """Internal: read equipment inspection. Already in step_observations."""
        return float(self._available_inspections[zone])

    def get_maintenance_alerts(self):
        """Internal: read maintenance alerts dict. Already in step_observations."""
        return dict(self._available_maintenance_alerts)

    # =========================================================================
    # Submit decisions — the ONLY action that advances the day
    # =========================================================================

    def submit_decisions(self, allocation, beliefs, reasoning=""):
        """
        Submit fishing decisions. This advances the day.

        Args:
            allocation: dict {"A": n, "B": n, "C": n, "D": n} where
                        values >= 0 and 1 <= sum <= max_boats.
            beliefs: dict with:
                "storm_active": float (P(storm=1))
                "storm_zone_probs": {"A": f, "B": f, "C": f, "D": f}
                "equip_failure_active": float (P(equip>0))
                "equip_zone_probs": {"A": f, "B": f, "C": f, "D": f}
            reasoning: optional string

        Returns:
            dict with observation, reward, done, info
        """
        assert self._episode_active, "Episode not active. Call reset() first."

        # Validate allocation
        total_boats = sum(allocation.get(z, 0) for z in self.cfg["zones"])
        assert 1 <= total_boats <= self.cfg["max_boats"], \
            f"Total boats must be 1-{self.cfg['max_boats']}, got {total_boats}"
        for z in allocation:
            assert z in self.cfg["zones"], f"Invalid zone in allocation: {z}"
            assert allocation[z] >= 0, f"Negative boats in zone {z}"

        # Compute reward from hidden state
        reward = self.pomdp.reward(self._hidden_state_idx, allocation)

        # Update visible database
        self._cumulative_reward += reward
        cur = self._db.cursor()
        alloc_json = json.dumps(allocation)
        cur.execute(
            "INSERT INTO daily_log (day, sea_color, equip_indicator, allocation, reward, cumulative) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (self._day, self._current_sea_color, self._current_equip_indicator,
             alloc_json, reward, self._cumulative_reward),
        )
        # Insert daily conditions (1 row per day)
        storm, wind, equip = self._hidden_state
        storm_zone = self.cfg["wind_to_zone"][wind] if storm == 1 else None
        equip_zone = self.cfg["equip_to_zone"][equip]
        cur.execute(
            "INSERT INTO daily_conditions "
            "(day, storm_active, storm_zone, equip_zone, barometer, sea_color) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (self._day, storm, storm_zone, equip_zone,
             round(float(self._available_barometer), 2), self._current_sea_color),
        )
        # Insert per-zone catch history
        for zone in self.cfg["zones"]:
            boats = allocation.get(zone, 0)
            if boats > 0:
                zone_reward = self.pomdp.reward(self._hidden_state_idx, {zone: boats})
                cur.execute(
                    "INSERT INTO catch_history (day, zone, boats, reward) VALUES (?, ?, ?, ?)",
                    (self._day, zone, boats, zone_reward),
                )
        # Insert today's sensor readings into sensor_log
        for zone in self.cfg["zones"]:
            cur.execute(
                "INSERT INTO sensor_log (day, zone, buoy_reading, equipment_reading) "
                "VALUES (?, ?, ?, ?)",
                (self._day, zone,
                 round(float(self._available_buoys[zone]), 2),
                 round(float(self._available_inspections[zone]), 2)),
            )
        self._db.commit()

        # Record trace entry
        # Observations = Tier 1 + Tier 2 if agent used SQL, else Tier 1 only
        if self._agent_used_sql_tools():
            step_obs = self._build_free_observations() + self._build_sql_discoverable_observations()
        else:
            step_obs = list(self._build_free_observations())

        self._trace.append({
            "day": self._day,
            "hidden_state_idx": self._hidden_state_idx,
            "hidden_state": self._hidden_state,
            "sea_color": self._current_sea_color,
            "equip_indicator": self._current_equip_indicator,
            "observations": step_obs,
            "available_observations": self._get_all_available_observations(),
            "weather_signals": self._current_weather_signals,
            "equipment_signals": self._current_equip_signals,
            "tools_used": list(self._step_tool_usage),
            "action": dict(allocation),
            "beliefs": dict(beliefs),
            "reward": reward,
            "reasoning": reasoning,
        })

        # Advance the day
        self._yesterday_reward = reward
        self._yesterday_allocation = dict(allocation)
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
        self._current_equip_indicator = self._sample_equip_indicator()
        self._current_weather_signals = self._emit_weather_signals()
        self._current_equip_signals = self._emit_equipment_signals()
        self._store_signals(self._current_weather_signals, "weather")
        self._store_signals(self._current_equip_signals, "equipment")

        # Pre-sample all free sensor readings for new day
        self._available_barometer = self._sample_barometer()
        self._available_buoys = {z: self._sample_buoy(z) for z in self.cfg["zones"]}
        self._available_inspections = {z: self._sample_inspection(z) for z in self.cfg["zones"]}
        self._available_maintenance_alerts = self._sample_maintenance_alerts()
        self._store_maintenance_alerts(self._available_maintenance_alerts)

        obs = self._make_observation_bundle()
        return {
            "observation": obs,
            "reward": reward,
            "done": False,
            "info": {},
        }

    def _transition_hidden_state(self):
        """Sample next hidden state from 40x40 transition matrix."""
        row = self.pomdp.T[self._hidden_state_idx]
        self._hidden_state_idx = self._rng.choices(
            range(self.pomdp.n_states), weights=row.tolist(), k=1
        )[0]
        self._hidden_state = self.cfg["states"][self._hidden_state_idx]

    def _get_all_available_observations(self):
        """
        Return all observations that COULD have been gathered this step
        (for the evaluator's oracle computation). Tier 1 + Tier 2 = 15 obs.
        """
        return self._build_free_observations() + self._build_sql_discoverable_observations()

    def get_trace(self):
        """Return the full episode trace for the evaluator."""
        return list(self._trace)
