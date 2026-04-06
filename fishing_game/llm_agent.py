"""
LLM Agent adapter for the Fishing Game v5 — Labels removed + tide/water temp.

Bridges any tool-calling LLM to the FishingGameEnv v5.
The LLM sees the observation bundle, the tool schemas, and must end each
turn by calling submit_decisions. Causal structure must be discovered
through historical database analysis, not explained in the prompt.
daily_conditions table is hidden — no ground truth labels available.
"""

import json
from fishing_game.config import CONFIG


# ============================================================================
# Tool schemas — the LLM sees these as callable functions
# ============================================================================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "check_weather_reports",
            "description": (
                "Search today's and historical weather intelligence reports. "
                "Returns matching signals sorted by relevance. Each result includes "
                "the day it was issued — check the day field to distinguish fresh vs. "
                "stale reports. Budget: 2 calls per day."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords (e.g. 'storm warning severe weather')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 3)",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_equipment_reports",
            "description": (
                "Search today's and historical equipment intelligence reports. "
                "Returns matching signals about equipment condition, maintenance, "
                "and failures. Budget: 2 calls per day."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords (e.g. 'equipment failure malfunction')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 3)",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_fishing_log",
            "description": (
                "Run a read-only SQL query against your fishing database. "
                "Available tables: daily_log, weather_signals, "
                "catch_history, maintenance_log, sensor_log. "
                "sensor_log (day, zone, buoy_reading, equipment_reading, water_temp) has per-zone readings. "
                "catch_history (day, zone, boats, reward) has historical catch outcomes. "
                "Only SELECT/WITH allowed. Budget: 2 calls per day."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL SELECT query",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_maintenance_log",
            "description": (
                "Run a read-only SQL query against the maintenance database. "
                "Table: maintenance_log (day, zone, alerts). Contains 30 days of "
                "pre-season historical data (days -30 to -1) plus current season data. "
                "Only SELECT/WITH allowed. Budget: 2 calls per day."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL SELECT query",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_data",
            "description": (
                "Execute Python code in a sandbox (math, statistics, random, collections, itertools, functools available). "
                "Use for computing posteriors, analyzing patterns. Returns stdout. "
                "Budget: 1 call per day."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_options",
            "description": (
                "Compute expected rewards for top boat allocations given your beliefs "
                "about storm and equipment risks. Returns top 10 allocations ranked "
                "by expected reward. Budget: 1 call per day."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "storm_active": {
                        "type": "number",
                        "description": "Probability that a storm is active (0.0 to 1.0)",
                    },
                    "storm_zone_probs": {
                        "type": "object",
                        "description": "Probability for each zone being the storm zone (should sum to 1)",
                        "properties": {
                            "A": {"type": "number"}, "B": {"type": "number"},
                            "C": {"type": "number"}, "D": {"type": "number"},
                        },
                    },
                    "equip_failure_active": {
                        "type": "number",
                        "description": "Probability of equipment failure (0.0 to 1.0)",
                    },
                    "equip_zone_probs": {
                        "type": "object",
                        "description": "Probability for each zone having equipment failure (should sum to 1)",
                        "properties": {
                            "A": {"type": "number"}, "B": {"type": "number"},
                            "C": {"type": "number"}, "D": {"type": "number"},
                        },
                    },
                    "tide_high": {
                        "type": "number",
                        "description": "Probability that tide is high (0.0 to 1.0)",
                    },
                },
                "required": ["storm_active"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forecast_scenario",
            "description": (
                "Project forward N days under a hypothetical dual-risk scenario. "
                "Useful for comparing strategies. Does NOT change actual game state. "
                "Budget: 1 call per day."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "horizon_days": {
                        "type": "integer",
                        "description": "Number of days to project forward",
                    },
                    "assume_storm_persists": {
                        "type": "boolean",
                        "description": "Whether to assume the storm continues",
                    },
                    "assume_storm_zone": {
                        "type": "string",
                        "enum": ["A", "B", "C", "D"],
                        "description": "Which zone the storm affects",
                    },
                    "assume_equip_failure": {
                        "type": "boolean",
                        "description": "Whether to assume equipment failure continues",
                    },
                    "assume_equip_zone": {
                        "type": "string",
                        "enum": ["A", "B", "C", "D"],
                        "description": "Which zone has equipment failure",
                    },
                    "assume_tide_high": {
                        "type": "boolean",
                        "description": "Whether to assume high tide",
                    },
                    "strategy": {
                        "type": "object",
                        "description": "Boat allocation to simulate",
                        "properties": {
                            "A": {"type": "integer"}, "B": {"type": "integer"},
                            "C": {"type": "integer"}, "D": {"type": "integer"},
                        },
                    },
                },
                "required": ["horizon_days", "assume_storm_persists", "strategy"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_decisions",
            "description": (
                "Submit your fishing decision for today. This is MANDATORY and "
                "ends your turn. You must provide a boat allocation across zones "
                "and your beliefs about both storm and equipment risks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "allocation": {
                        "type": "object",
                        "description": "Boats per zone, e.g. {\"A\": 5, \"B\": 3, \"C\": 2, \"D\": 0}. Total must be 1-10.",
                        "properties": {
                            "A": {"type": "integer", "minimum": 0, "maximum": 10},
                            "B": {"type": "integer", "minimum": 0, "maximum": 10},
                            "C": {"type": "integer", "minimum": 0, "maximum": 10},
                            "D": {"type": "integer", "minimum": 0, "maximum": 10},
                        },
                    },
                    "storm_active": {
                        "type": "number",
                        "description": "Probability that a storm is active (0.0 to 1.0)",
                    },
                    "storm_zone_probs": {
                        "type": "object",
                        "description": "Probability each zone is the storm zone (should sum to 1)",
                        "properties": {
                            "A": {"type": "number"}, "B": {"type": "number"},
                            "C": {"type": "number"}, "D": {"type": "number"},
                        },
                    },
                    "equip_failure_active": {
                        "type": "number",
                        "description": "Probability of equipment failure (0.0 to 1.0)",
                    },
                    "equip_zone_probs": {
                        "type": "object",
                        "description": "Probability each zone has equipment failure (should sum to 1)",
                        "properties": {
                            "A": {"type": "number"}, "B": {"type": "number"},
                            "C": {"type": "number"}, "D": {"type": "number"},
                        },
                    },
                    "tide_high": {
                        "type": "number",
                        "description": "Probability that tide is high (0.0 to 1.0)",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of your decision",
                    },
                },
                "required": [
                    "allocation",
                    "storm_active",
                    "storm_zone_probs",
                    "equip_failure_active",
                    "equip_zone_probs",
                    "tide_high",
                ],
            },
        },
    },
]


SYSTEM_PROMPT = """\
You are an expert fisher managing a fleet over a 20-day season across 4 fishing zones (A, B, C, D).

## Situation
Each day, TWO independent hidden risks may be active:
1. **Storm**: A hidden storm may be active, hitting ONE of the four zones.
2. **Equipment failure**: Fishing equipment in ONE zone may be broken, causing losses.

Additionally, **tide** (high or low) affects fishing conditions but is not directly observable.

## Rewards (per boat, per zone)
- Zone is SAFE (no risks): +7 base (+ possible tide bonus)
- Zone has STORM only: -18
- Zone has EQUIPMENT FAILURE only: -10
- Zone has BOTH storm AND equipment failure: -25

## Sensor data (provided every day)
You receive the following automatically:
- **Sea color** (green/murky/dark)
- **Equipment indicator** (normal/warning/critical)
- **Barometer** (hPa)
- **Buoy readings** (4 zones, meters)
- **Equipment readings** (4 zones, score)
- **Maintenance alerts** (4 zones, count)
- **Water temperature** (4 zones, degrees C)
- **Zone infrastructure ages**: A=25yr, B=15yr, C=5yr, D=2yr (constant)

**Important**: Sensor readings may be correlated or confounded. Investigate historical
patterns in the database to understand how readings relate to actual conditions.
The historical database contains 30 days of pre-season sensor and catch data.

## Budget-gated tools
- **check_weather_reports**: Search weather intelligence (budget: 2/day)
- **check_equipment_reports**: Search equipment intelligence (budget: 2/day)
- **query_fishing_log**: SQL query on fishing database (budget: 2/day)
- **query_maintenance_log**: SQL query on maintenance database (budget: 2/day)
- **analyze_data**: Run Python calculations (budget: 1/day)
- **evaluate_options**: See expected rewards for allocations given beliefs (budget: 1/day)
- **forecast_scenario**: Project multi-day scenarios (budget: 1/day)

## Boat allocation
Each day, allocate 1-10 boats across the 4 zones. Example: {"A": 5, "B": 3, "C": 2, "D": 0}

## Required output
Every turn MUST end with submit_decisions including your full beliefs:
- storm_active: P(storm is active)
- storm_zone_probs: {"A": p, "B": p, "C": p, "D": p} summing to 1
- equip_failure_active: P(equipment is broken somewhere)
- equip_zone_probs: {"A": p, "B": p, "C": p, "D": p} summing to 1
- tide_high: P(tide is high)

Do not omit any belief fields. If you are uncertain, still provide explicit probabilities.
If submit_decisions is missing any required belief field, the action will fail.

## Tool discipline
Use analyze_data when you need to compute, compare hypotheses, deconfound signals, estimate probabilities, or check whether an allocation is justified.
Do not rely only on narrative reasoning when a short calculation would reduce uncertainty.
When you submit, make sure your stated zone probabilities reflect your reasoning; do not leave them uniform unless you truly believe all zones are equally likely.
"""


def format_observation_message(obs):
    """Convert an observation bundle into a user message for the LLM."""
    yesterday = ""
    if obs.get("yesterday_allocation"):
        yesterday = (
            f"Yesterday: allocated {json.dumps(obs['yesterday_allocation'])}, "
            f"reward: {obs['yesterday_reward']}\n"
        )
    else:
        yesterday = f"Yesterday's reward: {obs['yesterday_reward']}\n"

    # Format free sensor data
    buoys = obs.get("buoy_readings", {})
    equip = obs.get("equipment_readings", {})
    maint = obs.get("maintenance_alerts", {})
    ages = obs.get("zone_infrastructure_ages", {})
    water_temps = obs.get("water_temp_readings", {})

    buoy_str = ", ".join(f"{z}={buoys.get(z, '?')}m" for z in ["A", "B", "C", "D"])
    equip_str = ", ".join(f"{z}={equip.get(z, '?')}" for z in ["A", "B", "C", "D"])
    maint_str = ", ".join(f"{z}={maint.get(z, '?')}" for z in ["A", "B", "C", "D"])
    age_str = ", ".join(f"{z}={ages.get(z, '?')}yr" for z in ["A", "B", "C", "D"])
    wtemp_str = ", ".join(f"{z}={water_temps.get(z, '?')}C" for z in ["A", "B", "C", "D"])

    return (
        f"=== DAY {obs['day']} ({obs['days_remaining']} days remaining) ===\n"
        f"Sea color: {obs['sea_color']}\n"
        f"Equipment indicator: {obs['equip_indicator']}\n"
        f"Barometer: {obs.get('barometer', '?')} hPa\n"
        f"Buoy readings: {buoy_str}\n"
        f"Equipment readings: {equip_str}\n"
        f"Maintenance alerts: {maint_str}\n"
        f"Water temperature: {wtemp_str}\n"
        f"Zone ages: {age_str}\n"
        f"{yesterday}"
        f"Cumulative reward: {obs['cumulative_reward']}\n"
        f"Tools available: {', '.join(obs['tools_available'])}\n"
        f"Tool budget: {json.dumps(obs['tool_budget'])}\n\n"
        f"Database schema:\n{obs['db_schema']}\n\n"
        f"Decide: analyze the sensor data, use tools if needed, then call submit_decisions."
    )


def get_active_tool_schemas(obs):
    """Filter tool schemas to only include tools available this turn."""
    available = set(obs["tools_available"]) | {"submit_decisions"}
    return [s for s in TOOL_SCHEMAS if s["function"]["name"] in available]


def execute_tool_call(env, tool_name, tool_args):
    """
    Execute a tool call against the environment.
    Returns (result_str, is_submit).
    """
    if tool_name == "check_weather_reports":
        result = env.check_weather_reports(
            query=tool_args["query"],
            max_results=tool_args.get("max_results", 3),
        )
        return json.dumps(result, indent=2), False

    elif tool_name == "check_equipment_reports":
        result = env.check_equipment_reports(
            query=tool_args["query"],
            max_results=tool_args.get("max_results", 3),
        )
        return json.dumps(result, indent=2), False

    elif tool_name == "query_fishing_log":
        result = env.query_fishing_log(query=tool_args["query"])
        return json.dumps(result, indent=2), False

    elif tool_name == "query_maintenance_log":
        result = env.query_maintenance_log(query=tool_args["query"])
        return json.dumps(result, indent=2), False

    elif tool_name == "analyze_data":
        result = env.analyze_data(code=tool_args["code"])
        return str(result), False

    elif tool_name == "evaluate_options":
        result = env.evaluate_options(params={
            "storm_active": tool_args.get("storm_active", 0.5),
            "storm_zone_probs": tool_args.get("storm_zone_probs", {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}),
            "equip_failure_active": tool_args.get("equip_failure_active", 0.2),
            "equip_zone_probs": tool_args.get("equip_zone_probs", {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}),
            "tide_high": tool_args.get("tide_high", 0.5),
        })
        return json.dumps(result, indent=2), False

    elif tool_name == "forecast_scenario":
        result = env.forecast_scenario(params={
            "horizon_days": tool_args["horizon_days"],
            "assume_storm_persists": tool_args["assume_storm_persists"],
            "assume_storm_zone": tool_args.get("assume_storm_zone", "A"),
            "assume_equip_failure": tool_args.get("assume_equip_failure", False),
            "assume_equip_zone": tool_args.get("assume_equip_zone", "A"),
            "assume_tide_high": tool_args.get("assume_tide_high", False),
            "strategy": tool_args["strategy"],
        })
        return json.dumps(result, indent=2), False

    elif tool_name == "submit_decisions":
        required_fields = [
            "allocation",
            "storm_active",
            "storm_zone_probs",
            "equip_failure_active",
            "equip_zone_probs",
            "tide_high",
        ]
        missing = [field for field in required_fields if field not in tool_args]
        if missing:
            return json.dumps({
                "error": (
                    "submit_decisions requires all belief fields. "
                    f"Missing: {', '.join(missing)}"
                )
            }, indent=2), False

        allocation = tool_args["allocation"]
        beliefs = {
            "storm_active": tool_args["storm_active"],
            "storm_zone_probs": tool_args["storm_zone_probs"],
            "equip_failure_active": tool_args["equip_failure_active"],
            "equip_zone_probs": tool_args["equip_zone_probs"],
            "tide_high": tool_args["tide_high"],
        }
        result = env.submit_decisions(
            allocation=allocation,
            beliefs=beliefs,
            reasoning=tool_args.get("reasoning", ""),
        )
        env._last_submit_result = result
        return json.dumps({
            "reward": result["reward"],
            "done": result["done"],
        }), True

    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"}), False


class LLMAgent:
    """
    Generic LLM agent that works with any tool-calling model.
    Subclass and implement _call_llm(messages, tools).
    """

    def __init__(self, config=None):
        self.cfg = config or CONFIG
        self.conversation_history = []

    def reset(self):
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def act(self, env, obs, rng=None):
        obs_message = format_observation_message(obs)
        self.conversation_history.append({"role": "user", "content": obs_message})

        tools = get_active_tool_schemas(obs)
        max_iterations = 10

        for _ in range(max_iterations):
            tool_calls = self._call_llm(self.conversation_history, tools)

            if tool_calls is None:
                result = env.submit_decisions(
                    allocation={"A": 1, "B": 0, "C": 0, "D": 0},
                    beliefs={
                        "storm_active": 0.5,
                        "storm_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
                        "equip_failure_active": 0.2,
                        "equip_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
                    },
                    reasoning="LLM failed to call submit_decisions.",
                )
                return result

            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc["arguments"]
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)

                result_str, is_submit = execute_tool_call(env, tool_name, tool_args)

                tc_id = tc.get("id", f"call_{tool_name}")
                tc_args = tc.get("arguments", "{}")
                if isinstance(tc_args, dict):
                    tc_args = json.dumps(tc_args)
                self.conversation_history.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tc_args,
                        },
                    }],
                })
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result_str,
                })

                if is_submit:
                    return env._last_submit_result

        # Safety fallback
        result = env.submit_decisions(
            allocation={"A": 1, "B": 0, "C": 0, "D": 0},
            beliefs={
                "storm_active": 0.5,
                "storm_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
                "equip_failure_active": 0.2,
                "equip_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
            },
            reasoning="Max iterations reached.",
        )
        return result

    def _call_llm(self, messages, tools):
        raise NotImplementedError("Subclass and implement _call_llm()")


# ============================================================================
# Simulated LLM agent (for testing without an actual LLM)
# ============================================================================

class SimulatedLLMAgent(LLMAgent):
    """
    Mock LLM agent with scripted strategy for v5.
    Strategy: search weather reports -> search equipment reports -> submit.
    Uses naive pattern matching (falls for causal traps).
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._turn = 0

    def _call_llm(self, messages, tools):
        last_msg = messages[-1]

        if last_msg["role"] == "user" and "DAY" in last_msg["content"]:
            self._turn = 0
            content = last_msg["content"]
            self._sea_color = "green"
            if "dark" in content:
                self._sea_color = "dark"
            elif "murky" in content:
                self._sea_color = "murky"

            self._equip_ind = "normal"
            if "critical" in content:
                self._equip_ind = "critical"
            elif "warning" in content:
                self._equip_ind = "warning"

            return [{"name": "check_weather_reports", "arguments": json.dumps({
                "query": "storm warning severe weather gale"
            }), "id": "call_weather"}]

        if last_msg["role"] == "tool" and "weather" in last_msg.get("tool_call_id", ""):
            self._turn += 1
            self._weather_alert = any(
                kw in last_msg["content"].lower()
                for kw in ["storm", "gale", "severe", "warning"]
            )
            return [{"name": "check_equipment_reports", "arguments": json.dumps({
                "query": "equipment failure malfunction critical alert"
            }), "id": "call_equip"}]

        if last_msg["role"] == "tool" and "equip" in last_msg.get("tool_call_id", ""):
            self._equip_alert = any(
                kw in last_msg["content"].lower()
                for kw in ["equipment", "failure", "malfunction", "critical"]
            )

            if self._weather_alert or self._sea_color == "dark":
                p_storm = 0.75
                boats = 1
            elif self._sea_color == "murky":
                p_storm = 0.4
                boats = 5
            else:
                p_storm = 0.15
                boats = 10

            p_equip = 0.6 if (self._equip_alert or self._equip_ind == "critical") else 0.15

            return [{"name": "submit_decisions", "arguments": json.dumps({
                "allocation": {"A": boats, "B": 0, "C": 0, "D": 0},
                "storm_active": p_storm,
                "storm_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
                "equip_failure_active": p_equip,
                "equip_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
                "reasoning": f"Sea={self._sea_color}, equip={self._equip_ind}, "
                             f"weather_alert={self._weather_alert}, equip_alert={self._equip_alert}",
            }), "id": "call_submit"}]

        # Fallback
        return [{"name": "submit_decisions", "arguments": json.dumps({
            "allocation": {"A": 1, "B": 0, "C": 0, "D": 0},
            "storm_active": 0.5,
            "storm_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
            "equip_failure_active": 0.2,
            "equip_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
            "reasoning": "Fallback.",
        }), "id": "call_fallback"}]
