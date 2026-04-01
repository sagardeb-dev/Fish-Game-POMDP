"""
LLM Agent adapter for the Fishing Game.

This module bridges any tool-calling LLM to the FishingGameEnv.
The LLM sees the observation bundle as a system/user message, the 5 tools
as function definitions, and must end each turn by calling submit_decisions.

Works with any provider that supports tool/function calling:
  - OpenAI (gpt-*), Anthropic (claude-*), Google (gemini-*), etc.

The adapter handles:
  1. Formatting observations into LLM messages
  2. Converting env tool methods into tool schemas (JSON)
  3. Running the tool-call loop: LLM picks a tool -> env executes -> result
     returned to LLM -> repeat until submit_decisions is called
  4. Parsing the final decision and beliefs from the LLM's submit call

No LLM SDK is imported here — the adapter defines the interface.
A concrete runner plugs in any SDK.
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
                "Search today's and historical weather/maritime intelligence reports. "
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
            "name": "query_fishing_log",
            "description": (
                "Run a read-only SQL query against your fishing database. "
                "Available tables: daily_log (day, sea_color, zone_fished, boats_sent, "
                "reward, cumulative), weather_signals (signal_id, day, source_type, "
                "headline, body), catch_history (day, zone, boats, reward). "
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
            "name": "read_barometer",
            "description": (
                "Read today's barometer pressure (hPa). Normal calm weather reads "
                "around 1013 hPa. Storms cause pressure to drop significantly. "
                "This is a strong indicator of whether a storm is active today. "
                "Budget: 1 call per day."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_buoy",
            "description": (
                "Read the wave-height sensor on a buoy in a specific zone. "
                "Normal conditions read around 1.2m. If a storm is active AND "
                "this zone is the dangerous one, waves spike to around 4.0m. "
                "This is the ONLY way to determine WHICH zone is dangerous. "
                "Budget: 2 calls per day (one per zone)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "zone": {
                        "type": "string",
                        "enum": ["A", "B"],
                        "description": "Which zone's buoy to read",
                    },
                },
                "required": ["zone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_data",
            "description": (
                "Execute Python code in a sandbox (math, statistics, random available). "
                "Use for computing posteriors, analyzing patterns, running correlations "
                "on your historical data. Returns stdout. Budget: 1 call per day."
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
                "Compute expected rewards for all 6 fishing options (2 zones x "
                "3 boat counts) given your beliefs about storm probability and "
                "which zone is dangerous. You must still decide which option to "
                "pick — this tool only shows the math. Budget: 1 call per day."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "storm_probability": {
                        "type": "number",
                        "description": "Your estimated probability that a storm is active (0.0 to 1.0)",
                    },
                    "zone_a_danger_probability": {
                        "type": "number",
                        "description": "Your estimated probability that zone A is the dangerous zone (0.0 to 1.0)",
                    },
                },
                "required": ["storm_probability", "zone_a_danger_probability"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forecast_scenario",
            "description": (
                "Compare the cost of different strategies over the next N days "
                "under a hypothetical scenario (e.g. 'what if the storm persists "
                "in zone A for 5 more days?'). Useful for deciding whether to play "
                "it safe with fewer boats or go aggressive. Does NOT change the "
                "actual game state. Budget: 1 call per day."
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
                    "assume_zone": {
                        "type": "string",
                        "enum": ["A", "B"],
                        "description": "Which zone to assume is affected",
                    },
                    "strategy": {
                        "type": "object",
                        "description": "The fishing strategy to simulate",
                        "properties": {
                            "zone": {"type": "string", "enum": ["A", "B"]},
                            "boats": {"type": "integer", "minimum": 1, "maximum": 3},
                        },
                        "required": ["zone", "boats"],
                    },
                },
                "required": ["horizon_days", "assume_storm_persists", "assume_zone", "strategy"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_decisions",
            "description": (
                "Submit your fishing decision for today. This is MANDATORY and "
                "ends your turn. You must provide your beliefs about the storm state."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "zone": {
                        "type": "string",
                        "enum": ["A", "B"],
                        "description": "Which zone to fish in",
                    },
                    "boats": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 3,
                        "description": "How many boats to send (1-3)",
                    },
                    "storm_active": {
                        "type": "number",
                        "description": "Your belief that a storm is active (0.0 to 1.0)",
                    },
                    "zone_a_is_dangerous": {
                        "type": "number",
                        "description": (
                            "Your belief that zone A is the dangerous zone, "
                            "conditional on a storm being active (0.0 to 1.0). "
                            "Only meaningful when storm_active > 0."
                        ),
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of your decision",
                    },
                },
                "required": ["zone", "boats", "storm_active", "zone_a_is_dangerous"],
            },
        },
    },
]


SYSTEM_PROMPT = """\
You are an expert fisher managing a fleet over a 20-day season.

## Situation
Each day, a hidden storm may be active, hitting one of two fishing zones (A or B).
You cannot observe the storm directly. You receive a sea color observation for free,
and can use tools to gather more information before deciding where to fish.

## Rewards
- Fishing a SAFE zone: +7 per boat
- Fishing the DANGEROUS zone during a storm: -18 per boat
- No storm: both zones are safe (+7 per boat)

## Your tools (use before submitting your decision)
- check_weather_reports: Search weather intelligence reports (budget: 2/day)
- query_fishing_log: Query your historical fishing database (budget: 2/day)
- read_barometer: Read today's barometric pressure — strong storm indicator (budget: 1/day)
- read_buoy: Read wave height in a specific zone — the ONLY way to tell which zone is dangerous (budget: 2/day)
- analyze_data: Run Python calculations (budget: 1/day)
- evaluate_options: See expected rewards for all 6 fishing options given your beliefs (budget: 1/day)
- forecast_scenario: Compare strategies over multiple days under a hypothetical (budget: 1/day)

## Sea color clues (free each day)
- "green" = likely no storm (70% if calm, 5% if storm)
- "murky" = ambiguous (25% if calm, 35% if storm)
- "dark" = likely storm (5% if calm, 60% if storm)

## Signal reliability
Weather reports are noisy — storm warnings occasionally appear even during calm
conditions (~3% false positive rate). Treat them as one input among many, not as
ground truth.

## Strategy tips
- Use tools to reduce uncertainty before committing boats
- When uncertain about storm, send fewer boats to limit downside
- Track patterns across days — storms tend to persist (80% chance)
- Read the barometer to confirm storm presence, then read buoys in BOTH zones to identify which is dangerous
- The dangerous zone depends on wind direction, which can shift even while a storm persists

## Required output
Every turn MUST end with a submit_decisions call including your beliefs:
- storm_active: probability 0.0 to 1.0
- zone_a_is_dangerous: probability that zone A is the dangerous zone IF a storm is active (0.0 to 1.0)
"""


def format_observation_message(obs):
    """Convert an observation bundle into a user message for the LLM."""
    yesterday = ""
    if obs.get("yesterday_zone"):
        yesterday = (
            f"Yesterday: fished zone {obs['yesterday_zone']} with "
            f"{obs['yesterday_boats']} boats, reward: {obs['yesterday_reward']}\n"
        )
    else:
        yesterday = f"Yesterday's reward: {obs['yesterday_reward']}\n"

    return (
        f"=== DAY {obs['day']} ({obs['days_remaining']} days remaining) ===\n"
        f"Sea color: {obs['sea_color']}\n"
        f"{yesterday}"
        f"Cumulative reward: {obs['cumulative_reward']}\n"
        f"Tools available: {', '.join(obs['tools_available'])}\n"
        f"Tool budget: {json.dumps(obs['tool_budget'])}\n\n"
        f"Database schema:\n{obs['db_schema']}\n\n"
        f"Decide: use tools to investigate, then call submit_decisions."
    )


def get_active_tool_schemas(obs):
    """Filter tool schemas to only include tools available this turn."""
    available = set(obs["tools_available"]) | {"submit_decisions"}
    return [s for s in TOOL_SCHEMAS if s["function"]["name"] in available]


def execute_tool_call(env, tool_name, tool_args):
    """
    Execute a tool call against the environment.
    Returns (result_str, is_submit) — is_submit is True if this was submit_decisions.
    """
    if tool_name == "check_weather_reports":
        result = env.check_weather_reports(
            query=tool_args["query"],
            max_results=tool_args.get("max_results", 3),
        )
        return json.dumps(result, indent=2), False

    elif tool_name == "query_fishing_log":
        result = env.query_fishing_log(query=tool_args["query"])
        return json.dumps(result, indent=2), False

    elif tool_name == "read_barometer":
        reading = env.read_barometer()
        return json.dumps({"pressure_hpa": round(reading, 1)}), False

    elif tool_name == "read_buoy":
        reading = env.read_buoy(zone=tool_args["zone"])
        return json.dumps({"zone": tool_args["zone"], "wave_height_m": round(reading, 2)}), False

    elif tool_name == "analyze_data":
        result = env.analyze_data(code=tool_args["code"])
        return str(result), False

    elif tool_name == "evaluate_options":
        result = env.evaluate_options(params={
            "storm_probability": tool_args["storm_probability"],
            "zone_a_danger_probability": tool_args.get("zone_a_danger_probability", 0.5),
        })
        return json.dumps(result, indent=2), False

    elif tool_name == "forecast_scenario":
        result = env.forecast_scenario(params={
            "horizon_days": tool_args["horizon_days"],
            "assume_storm_persists": tool_args["assume_storm_persists"],
            "assume_zone": tool_args["assume_zone"],
            "strategy": tool_args["strategy"],
        })
        return json.dumps(result, indent=2), False

    elif tool_name == "submit_decisions":
        beliefs = {
            "storm_active": tool_args.get("storm_active", 0.5),
            "zone_a_is_dangerous": tool_args.get("zone_a_is_dangerous", 0.5),
        }
        result = env.submit_decisions(
            zone=tool_args["zone"],
            boats=tool_args["boats"],
            beliefs=beliefs,
            reasoning=tool_args.get("reasoning", ""),
        )
        # Stash the full env result for the caller
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

    Subclass and implement `_call_llm(messages, tools)` to connect
    to your LLM provider. It should return a list of tool calls
    or None if the model wants to just respond with text.
    """

    def __init__(self, config=None):
        self.cfg = config or CONFIG
        self.conversation_history = []

    def reset(self):
        """Reset conversation for a new episode."""
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def act(self, env, obs, rng=None):
        """
        Run one turn: send observation to LLM, execute tool calls in a loop
        until submit_decisions is called.
        """
        # Add observation as user message
        obs_message = format_observation_message(obs)
        self.conversation_history.append({"role": "user", "content": obs_message})

        tools = get_active_tool_schemas(obs)
        max_iterations = 10  # safety limit

        for _ in range(max_iterations):
            # Call the LLM
            tool_calls = self._call_llm(self.conversation_history, tools)

            if tool_calls is None:
                # LLM responded with text only — no tool call.
                # Force a default submission.
                result = env.submit_decisions(
                    zone="A", boats=1,
                    beliefs={"storm_active": 0.5, "zone_a_is_dangerous": 0.5},
                    reasoning="LLM failed to call submit_decisions.",
                )
                return result

            # Process each tool call
            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc["arguments"]
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)

                result_str, is_submit = execute_tool_call(env, tool_name, tool_args)

                # Add tool call and result to conversation
                # Normalize to OpenAI format
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
                    # submit_decisions was called — turn is over
                    return env._last_submit_result

        # Safety: if loop exhausted, force submit
        result = env.submit_decisions(
            zone="A", boats=1,
            beliefs={"storm_active": 0.5, "zone_a_is_dangerous": 0.5},
            reasoning="Max iterations reached.",
        )
        return result

    def _call_llm(self, messages, tools):
        """
        Override this method to connect to your LLM provider.

        Args:
            messages: List of message dicts (OpenAI chat format)
            tools: List of tool schema dicts

        Returns:
            List of tool call dicts [{"name": ..., "arguments": ..., "id": ...}]
            or None if the model responded with text only.
        """
        raise NotImplementedError("Subclass and implement _call_llm()")


# ============================================================================
# Example: Simulated LLM agent (for testing without an actual LLM)
# ============================================================================

class SimulatedLLMAgent(LLMAgent):
    """
    A mock LLM agent that follows a scripted strategy.
    Demonstrates the exact message flow without requiring an API key.

    Strategy: search for storm signals, check SQL history, then decide.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._turn = 0

    def _call_llm(self, messages, tools):
        """Simulate LLM tool-calling behavior with a simple heuristic."""
        last_msg = messages[-1]

        # If the last message is the observation (user message), start investigating
        if last_msg["role"] == "user" and "DAY" in last_msg["content"]:
            self._turn = 0
            # Parse sea color from the observation
            content = last_msg["content"]
            if "dark" in content:
                self._sea_color = "dark"
            elif "murky" in content:
                self._sea_color = "murky"
            else:
                self._sea_color = "green"

            # Step 1: Search for storm signals
            return [{"name": "check_weather_reports", "arguments": json.dumps({
                "query": "storm warning severe weather gale"
            }), "id": "call_check_weather_reports"}]

        # If we just got search results back, check them and decide
        if last_msg["role"] == "tool" and "check_weather_reports" in last_msg.get("tool_call_id", ""):
            self._turn += 1
            search_results = last_msg["content"]
            alert_keywords = ["storm", "gale", "severe", "warning"]
            self._alert = any(kw in search_results.lower() for kw in alert_keywords)

            # Step 2: Check historical data
            return [{"name": "query_fishing_log", "arguments": json.dumps({
                "query": "SELECT zone, AVG(reward) as avg_reward, COUNT(*) as days "
                         "FROM catch_history GROUP BY zone"
            }), "id": "call_query_fishing_log"}]

        # After SQL, make decision
        if last_msg["role"] == "tool" and "query_fishing_log" in last_msg.get("tool_call_id", ""):
            if self._alert or self._sea_color == "dark":
                p_storm = 0.75
                boats = 1
            elif self._sea_color == "murky":
                p_storm = 0.4
                boats = 2
            else:
                p_storm = 0.15
                boats = 3

            return [{"name": "submit_decisions", "arguments": json.dumps({
                "zone": "A",
                "boats": boats,
                "storm_active": p_storm,
                "zone_a_is_dangerous": p_storm * 0.5,
                "reasoning": f"Sea={self._sea_color}, alert={self._alert}, "
                             f"P(storm)={p_storm}, sending {boats} boats.",
            }), "id": "call_3"}]

        # Fallback
        return [{"name": "submit_decisions", "arguments": json.dumps({
            "zone": "A", "boats": 1,
            "storm_active": 0.5, "zone_a_is_dangerous": 0.25,
            "reasoning": "Fallback.",
        }), "id": "call_fallback"}]
