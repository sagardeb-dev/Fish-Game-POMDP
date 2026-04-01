"""
Traced LLM agent runner.

Captures every input/output at every step of the agent-environment interaction
and saves the full trace to a JSON file.

Each step records:
  - The observation the agent received
  - Every LLM call (messages sent, tools available)
  - Every tool call the LLM made (name, arguments)
  - Every tool result the environment returned
  - The final decision (zone, boats, beliefs, reasoning)
  - The environment's response (reward, done, next observation)
  - Hidden state (for post-hoc analysis, never shown to agent)
"""

import json
import os
from datetime import datetime

from fishing_game.config import CONFIG
from fishing_game.simulator import FishingGameEnv
from fishing_game.evaluator import Evaluator
from fishing_game.llm_agent import (
    LLMAgent,
    SimulatedLLMAgent,
    format_observation_message,
    get_active_tool_schemas,
    execute_tool_call,
    SYSTEM_PROMPT,
)


class TracedLLMAgent(LLMAgent):
    """
    Wraps any LLM agent with full input/output tracing.
    Every message sent to the LLM and every result returned
    is captured in a structured trace.
    """

    def __init__(self, inner_agent, config=None):
        """
        Args:
            inner_agent: An LLMAgent subclass instance (e.g., SimulatedLLMAgent)
                         that implements _call_llm().
        """
        super().__init__(config)
        self._inner = inner_agent
        self.step_traces = []  # one entry per day

    def reset(self):
        super().reset()
        self._inner.reset()
        self.step_traces = []

    def act(self, env, obs, rng=None):
        """Run one turn with full tracing."""
        step_trace = {
            "day": obs["day"],
            "hidden_state": {
                "storm": env._hidden_state[0],
                "wind": env._hidden_state[1],
                "affected_zone": "A" if env._hidden_state[1] == "N" else "B",
                "state_idx": env._hidden_state_idx,
            },
            "observation_bundle": obs,
            "observation_message": format_observation_message(obs),
            "tool_calls": [],
            "decision": None,
            "env_result": None,
        }

        # Sync conversation history
        self._inner.conversation_history = list(self.conversation_history)

        # Add observation
        obs_msg = format_observation_message(obs)
        self.conversation_history.append({"role": "user", "content": obs_msg})
        self._inner.conversation_history.append({"role": "user", "content": obs_msg})

        tools = get_active_tool_schemas(obs)
        max_iterations = 10

        for iteration in range(max_iterations):
            # Record what we're sending to the LLM
            llm_input = {
                "iteration": iteration,
                "messages_count": len(self._inner.conversation_history),
                "last_message_role": self._inner.conversation_history[-1]["role"],
                "tools_available": [t["function"]["name"] for t in tools],
            }

            # Call the inner LLM
            tool_calls = self._inner._call_llm(
                self._inner.conversation_history, tools
            )

            if tool_calls is None:
                step_trace["tool_calls"].append({
                    "llm_input": llm_input,
                    "llm_output": "NO_TOOL_CALL (text response)",
                    "tool_name": None,
                    "tool_args": None,
                    "tool_result": None,
                })
                # Force default submission
                result = env.submit_decisions(
                    zone="A", boats=1,
                    beliefs={"storm_active": 0.5, "zone_a_is_dangerous": 0.5},
                    reasoning="LLM failed to call submit_decisions.",
                )
                step_trace["decision"] = {
                    "zone": "A", "boats": 1,
                    "beliefs": {"storm_active": 0.5, "zone_a_is_dangerous": 0.5},
                    "reasoning": "LLM failed to call submit_decisions.",
                }
                step_trace["env_result"] = {
                    "reward": result["reward"],
                    "done": result["done"],
                }
                self.step_traces.append(step_trace)
                return result

            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc["arguments"]
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)

                # Execute tool
                result_str, is_submit = execute_tool_call(env, tool_name, tool_args)

                # Record the full exchange
                call_record = {
                    "llm_input": llm_input,
                    "llm_output": f"tool_call: {tool_name}",
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "tool_result": _safe_parse_json(result_str),
                    "is_submit": is_submit,
                }
                step_trace["tool_calls"].append(call_record)

                # Add to both conversation histories
                # Normalize tool call format for OpenAI compatibility
                tc_id = tc.get("id", f"call_{tool_name}")
                tc_args = tc.get("arguments", "{}")
                if isinstance(tc_args, dict):
                    tc_args = json.dumps(tc_args)
                normalized_tc = {
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tc_args,
                    },
                }
                assistant_msg = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [normalized_tc],
                }
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result_str,
                }
                self.conversation_history.append(assistant_msg)
                self.conversation_history.append(tool_msg)
                self._inner.conversation_history.append(assistant_msg)
                self._inner.conversation_history.append(tool_msg)

                if is_submit:
                    step_trace["decision"] = {
                        "zone": tool_args["zone"],
                        "boats": tool_args["boats"],
                        "beliefs": {
                            "storm_active": tool_args.get("storm_active", 0.5),
                            "zone_a_is_dangerous": tool_args.get(
                                "zone_a_is_dangerous", 0.5
                            ),
                        },
                        "reasoning": tool_args.get("reasoning", ""),
                    }
                    env_result = env._last_submit_result
                    step_trace["env_result"] = {
                        "reward": env_result["reward"],
                        "done": env_result["done"],
                    }
                    self.step_traces.append(step_trace)
                    return env_result

        # Safety fallback
        result = env.submit_decisions(
            zone="A", boats=1,
            beliefs={"storm_active": 0.5, "zone_a_is_dangerous": 0.5},
            reasoning="Max iterations.",
        )
        step_trace["decision"] = {
            "zone": "A", "boats": 1,
            "beliefs": {"storm_active": 0.5, "zone_a_is_dangerous": 0.5},
            "reasoning": "Max iterations.",
        }
        step_trace["env_result"] = {"reward": result["reward"], "done": result["done"]}
        self.step_traces.append(step_trace)
        return result

    def _call_llm(self, messages, tools):
        return self._inner._call_llm(messages, tools)


def run_traced_episode(agent_cls=None, seed=42, config=None, save_path=None):
    """
    Run a full episode with tracing. Returns the trace dict.

    Args:
        agent_cls: An LLMAgent subclass. Defaults to SimulatedLLMAgent.
        seed: Random seed for reproducibility.
        config: Config dict. Defaults to CONFIG.
        save_path: If provided, saves the trace as JSON to this path.
    """
    cfg = config or CONFIG
    inner = (agent_cls or SimulatedLLMAgent)(config=cfg)
    agent = TracedLLMAgent(inner, config=cfg)
    agent.reset()

    env = FishingGameEnv(config=cfg)
    obs = env.reset(seed=seed)

    for _ in range(cfg["episode_length"]):
        result = agent.act(env, obs)
        if result["done"]:
            break
        obs = result["observation"]

    # Evaluate
    trace = env.get_trace()
    evaluator = Evaluator(config=cfg)
    eval_result = evaluator.evaluate_episode(trace)

    # Build full output
    output = {
        "metadata": {
            "seed": seed,
            "agent": type(inner).__name__,
            "timestamp": datetime.now().isoformat(),
            "episode_length": cfg["episode_length"],
        },
        "steps": agent.step_traces,
        "evaluation": {
            "total_reward": eval_result["total_reward"],
            "mean_brier_storm": eval_result["mean_brier_storm"],
            "mean_brier_zone": eval_result["mean_brier_zone"],
            "mean_detection_lag": eval_result["mean_detection_lag"],
            "total_tool_use_gap": eval_result["total_tool_use_gap"],
            "total_inference_gap": eval_result["total_inference_gap"],
            "total_planning_gap": eval_result["total_planning_gap"],
            "tool_usage_counts": eval_result["tool_usage_counts"],
            "reward_per_quarter": eval_result["reward_per_quarter"],
            "per_step": eval_result["step_results"],
        },
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

    return output


def print_traced_episode(output):
    """Pretty-print a traced episode to stdout."""
    meta = output["metadata"]
    print(f"Agent: {meta['agent']}, Seed: {meta['seed']}")
    print("=" * 90)

    for step in output["steps"]:
        day = step["day"]
        hs = step["hidden_state"]
        obs = step["observation_bundle"]
        decision = step["decision"]
        env_r = step["env_result"]

        print(f"\n{'='*90}")
        print(f"DAY {day}")
        print(f"{'='*90}")

        # Hidden state
        print(f"  [HIDDEN] storm={hs['storm']}, wind={hs['wind']}, "
              f"affected_zone={hs['affected_zone']}")
        print()

        # What agent sees
        print(f"  [INPUT] Observation bundle:")
        print(f"    sea_color:    {obs['sea_color']}")
        print(f"    yesterday_rw: {obs['yesterday_reward']}")
        print(f"    cumulative:   {obs['cumulative_reward']}")
        print(f"    tool_budget:  {obs['tool_budget']}")
        print()

        # Tool call loop
        print(f"  [TOOL CALLS] ({len(step['tool_calls'])} calls this turn):")
        for i, tc in enumerate(step["tool_calls"]):
            name = tc["tool_name"]
            args = tc["tool_args"]
            result = tc["tool_result"]

            if name == "submit_decisions":
                continue  # show separately below

            args_short = json.dumps(args, separators=(",", ":"))
            if len(args_short) > 80:
                args_short = args_short[:77] + "..."

            result_short = json.dumps(result, separators=(",", ":"))
            if len(result_short) > 120:
                result_short = result_short[:117] + "..."

            print(f"    [{i+1}] {name}({args_short})")
            print(f"        -> {result_short}")
        print()

        # Decision
        print(f"  [OUTPUT] Decision:")
        print(f"    zone={decision['zone']}, boats={decision['boats']}")
        print(f"    beliefs: storm={decision['beliefs']['storm_active']:.2f}, "
              f"zone_a={decision['beliefs']['zone_a_is_dangerous']:.2f}")
        print(f"    reasoning: {decision['reasoning']}")
        print()

        # Environment response
        print(f"  [RESULT] reward={env_r['reward']}, done={env_r['done']}")

    # Summary
    ev = output["evaluation"]
    print(f"\n{'='*90}")
    print("EPISODE SUMMARY")
    print(f"{'='*90}")
    print(f"  Total reward:        {ev['total_reward']}")
    print(f"  Mean Brier (storm):  {ev['mean_brier_storm']:.4f}")
    print(f"  Mean Brier (zone):   {ev['mean_brier_zone']:.4f}")
    print(f"  Detection lag:       {ev['mean_detection_lag']}")
    print(f"  Tool use gap:        {ev['total_tool_use_gap']:.1f}")
    print(f"  Inference gap:       {ev['total_inference_gap']:.1f}")
    print(f"  Planning gap:        {ev['total_planning_gap']:.1f}")
    print(f"  Tool usage:          {ev['tool_usage_counts']}")


def _safe_parse_json(s):
    """Try to parse JSON, return raw string on failure."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s


if __name__ == "__main__":
    output = run_traced_episode(
        seed=42,
        save_path="traces/simulated_llm_seed42.json",
    )
    print_traced_episode(output)
    print(f"\nTrace saved to traces/simulated_llm_seed42.json")
