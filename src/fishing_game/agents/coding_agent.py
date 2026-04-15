"""
Coding Agent for the Fishing Game.

Uses Agno framework with PythonTools for persistent code execution.
The agent writes and runs Python code to analyze historical data,
discover patterns, and make daily fishing decisions.

No solver, no POMDP — pure LLM reasoning with code as a thinking tool.
"""

import json
from pathlib import Path

from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.models.openai import OpenAIChat
from agno.tools import Toolkit
from agno.tools.python import PythonTools

from fishing_game.config import CONFIG
from fishing_game.llm_agent import format_observation_message


SYSTEM_PROMPT = """\
You are a data scientist managing a fishing fleet across 4 zones (A, B, C, D) for a 20-day season.

Each day you receive sensor readings and must allocate 1-10 boats across zones. Hidden risks (storms, equipment failures) can destroy your catch. You are not told what the risks are or how sensors relate to them — you must discover this from data.

CRITICAL RULE: You MUST use run_python_code on EVERY day. Never reason about numbers in text — always compute in Python. Variables persist across all days.

DAY 1 — MANDATORY ANALYSIS (do ALL of these before submitting):
1. Query catch_history: SELECT * FROM catch_history ORDER BY day, zone
2. Query sensor_log: SELECT * FROM sensor_log ORDER BY day, zone
3. Query maintenance_log: SELECT * FROM maintenance_log ORDER BY day, zone
4. In Python, compute:
   - Per-zone average reward (reward per boat) for each zone
   - Classify each day as "storm" (any zone reward <= -18) or "no_storm"
   - For storm vs no-storm days: mean buoy, mean equipment_reading, mean water_temp per zone
   - For storm vs no-storm days: mean barometer (from daily_log table)
   - Correlation between maintenance alerts and negative rewards
   - Store all thresholds in a dict called `model` for later use

DAYS 2-20 — DECISION LOOP:
1. Run Python code to score today's sensor readings against your `model` thresholds
2. Compute P(storm) and P(equipment_failure) numerically, not by gut feel
3. For each zone with sensor data, compute a risk score
4. Allocate boats: concentrate on lowest-risk zones, avoid highest-risk
5. submit_decisions with computed probabilities for storm, equipment failure, zone locations, and tide_high
6. If the evidence is mixed, still submit explicit non-uniform probabilities when your code supports a directional hypothesis.

AVAILABLE SENSORS (free, every day):
- Sea color: green/murky/dark (global)
- Equipment indicator: normal/warning/critical (global)
- Barometer: pressure reading in hPa (global)
- Buoy readings: wave height per zone (only 2 zones visible per day)
- Equipment readings: condition score per zone (only 2 zones visible per day)
- Maintenance alerts: alert count per zone (only 2 zones visible per day)
- Water temperature: degrees C per zone (only 2 zones visible per day)
- Zone infrastructure ages: A=25yr, B=15yr, C=5yr, D=2yr (constant)

REWARDS PER BOAT:
- Safe zone: +7 (plus possible bonuses)
- Storm zone: -18
- Equipment failure zone: -10
- Both risks: -25

DATABASE TABLES (queryable via query_fishing_log/query_maintenance_log):
- catch_history (day, zone, boats, reward) — 30 days of historical outcomes
- sensor_log (day, zone, buoy_reading, equipment_reading, water_temp) — historical sensor data
- maintenance_log (day, zone, alerts) — historical maintenance alerts
- daily_log (day, sea_color, equip_indicator, allocation, reward, cumulative) — daily summaries

KEY PATTERNS TO DISCOVER:
- Storms affect ONE zone + propagate to neighbors (ring: A-B-C-D-A). High buoy = storm nearby.
- Equipment failures correlate with high equipment_reading and high maintenance alerts.
- Barometer drops when storms are active. Sea color "dark" suggests storm.
- Equipment indicator "critical" suggests equipment failure somewhere.
- Zone A (25yr old) has higher baseline maintenance than Zone D (2yr old) — don't confuse age with failure.

IMPORTANT:
- Only 2 of 4 zones have sensor data each day. Unknown zones are riskier.
- ALWAYS run Python code to compute decisions. Never eyeball numbers.
- When uncertain, still submit explicit probabilities that sum to 1.0 for each zone distribution.
- Be concise — save context for later days.
"""


class FishingGameTools(Toolkit):
    """Toolkit wrapping the FishingGameEnv tools for use with Agno agents."""

    def __init__(self, env):
        self.env = env
        self._last_result = None
        tools = [
            self.query_fishing_log,
            self.query_maintenance_log,
            self.check_weather_reports,
            self.check_equipment_reports,
            self.submit_decisions,
        ]
        super().__init__(name="fishing_game", tools=tools)

    def query_fishing_log(self, query: str) -> str:
        """Run a read-only SQL query against the fishing database.

        Available tables:
        - catch_history (day, zone, boats, reward)
        - sensor_log (day, zone, buoy_reading, equipment_reading, water_temp)
        - maintenance_log (day, zone, alerts)
        - daily_log (day, sea_color, equip_indicator, allocation, reward, cumulative)
        - weather_signals (signal_id, day, source_type, report_type, headline, body)

        Args:
            query: SQL SELECT statement.

        Returns:
            JSON array of result rows, or error message.
        """
        result = self.env.query_fishing_log(query)
        return json.dumps(result, default=str)

    def query_maintenance_log(self, query: str) -> str:
        """Run a read-only SQL query against the maintenance database.

        Available tables: same as query_fishing_log.

        Args:
            query: SQL SELECT statement.

        Returns:
            JSON array of result rows, or error message.
        """
        result = self.env.query_maintenance_log(query)
        return json.dumps(result, default=str)

    def check_weather_reports(self, query: str) -> str:
        """Search weather intelligence reports for relevant signals.

        Args:
            query: Keywords to search for (e.g., 'storm warning severe').

        Returns:
            JSON array of matching weather reports.
        """
        result = self.env.check_weather_reports(query)
        return json.dumps(result, default=str)

    def check_equipment_reports(self, query: str) -> str:
        """Search equipment condition reports for relevant signals.

        Args:
            query: Keywords to search for (e.g., 'equipment failure critical').

        Returns:
            JSON array of matching equipment reports.
        """
        result = self.env.check_equipment_reports(query)
        return json.dumps(result, default=str)

    def submit_decisions(
        self,
        allocation_A: int,
        allocation_B: int,
        allocation_C: int,
        allocation_D: int,
        storm_active: float,
        equip_failure_active: float,
        storm_zone_A: float,
        storm_zone_B: float,
        storm_zone_C: float,
        storm_zone_D: float,
        equip_zone_A: float,
        equip_zone_B: float,
        equip_zone_C: float,
        equip_zone_D: float,
        tide_high: float,
        reasoning: str = "",
    ) -> str:
        """Submit your daily fishing decisions. This ends the current day.

        Args:
            allocation_A: Number of boats to send to zone A (0-10).
            allocation_B: Number of boats to send to zone B (0-10).
            allocation_C: Number of boats to send to zone C (0-10).
            allocation_D: Number of boats to send to zone D (0-10).
            storm_active: Your estimate of P(storm is active) from 0.0 to 1.0.
            equip_failure_active: Your estimate of P(equipment failure) from 0.0 to 1.0.
            storm_zone_A: P(storm is in zone A | storm active). All 4 should sum to 1.0.
            storm_zone_B: P(storm is in zone B | storm active).
            storm_zone_C: P(storm is in zone C | storm active).
            storm_zone_D: P(storm is in zone D | storm active).
            equip_zone_A: P(equip failure in zone A | failure active). All 4 should sum to 1.0.
            equip_zone_B: P(equip failure in zone B | failure active).
            equip_zone_C: P(equip failure in zone C | failure active).
            equip_zone_D: P(equip failure in zone D | failure active).
            tide_high: P(tide is high) from 0.0 to 1.0.
            reasoning: Brief explanation of your decision.

        Returns:
            JSON with reward and done status.
        """
        allocation = {
            "A": int(allocation_A),
            "B": int(allocation_B),
            "C": int(allocation_C),
            "D": int(allocation_D),
        }
        beliefs = {
            "storm_active": float(storm_active),
            "storm_zone_probs": {
                "A": float(storm_zone_A), "B": float(storm_zone_B),
                "C": float(storm_zone_C), "D": float(storm_zone_D),
            },
            "equip_failure_active": float(equip_failure_active),
            "equip_zone_probs": {
                "A": float(equip_zone_A), "B": float(equip_zone_B),
                "C": float(equip_zone_C), "D": float(equip_zone_D),
            },
            "tide_high": float(tide_high),
        }

        result = self.env.submit_decisions(
            allocation=allocation,
            beliefs=beliefs,
            reasoning=reasoning,
        )
        self._last_result = result
        return json.dumps({"reward": result["reward"], "done": result["done"]})


class CodingAgent:
    """
    Coding agent that uses Agno framework with PythonTools + FishingGameTools.

    The agent writes and executes Python code to analyze data and make decisions.
    Variables persist across all 20 days via PythonTools.
    """

    def __init__(self, model="gpt-5.4", config=None):
        self.cfg = config or CONFIG
        self.model_id = model
        self._agent = None
        self._game_tools = None

    def reset(self):
        self._agent = None
        self._game_tools = None

    def _build_agent(self, env):
        """Build the Agno agent with game tools + Python REPL."""
        self._game_tools = FishingGameTools(env)
        self._agent = Agent(
            model=OpenAIChat(id=self.model_id),
            tools=[
                PythonTools(
                    base_dir=Path("tmp/coding_agent"),
                    exclude_tools=["pip_install_package", "uv_pip_install_package"],
                ),
                self._game_tools,
            ],
            instructions=[SYSTEM_PROMPT],
            db=InMemoryDb(),
            add_history_to_context=True,
            num_history_runs=20,
            markdown=True,
        )

    def act(self, env, obs, rng=None):
        if self._agent is None:
            self._build_agent(env)

        # Reset last result so we can detect if submit was called
        self._game_tools._last_result = None

        obs_message = format_observation_message(obs)

        # print_response shows tool calls, LLM reasoning, and output to CLI
        self._agent.print_response(obs_message, stream=True)

        if self._game_tools._last_result is not None:
            return self._game_tools._last_result

        # Fallback if agent didn't submit
        print("[CodingAgent] WARNING: Agent didn't call submit_decisions, using fallback")
        result = env.submit_decisions(
            allocation={"A": 3, "B": 3, "C": 2, "D": 2},
            beliefs={
                "storm_active": 0.5,
                "storm_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
                "equip_failure_active": 0.5,
                "equip_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
                "tide_high": 0.5,
            },
            reasoning="CodingAgent fallback — agent didn't submit.",
        )
        return result
