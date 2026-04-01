"""
Traced LLM agent runner for Fishing Game v2.

Captures every input/output at every step of the agent-environment interaction
and saves the full trace to a JSON file.
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
    """

    def __init__(self, inner_agent, config=None):
        super().__init__(config)
        self._inner = inner_agent
        self.step_traces = []

    def reset(self):
        super().reset()
        self._inner.reset()
        self.step_traces = []

    def act(self, env, obs, rng=None):
        step_trace = {
            "day": obs["day"],
            "hidden_state": {
                "storm": env._hidden_state[0],
                "wind": env._hidden_state[1],
                "equip": env._hidden_state[2],
                "storm_zone": env.cfg["wind_to_zone"][env._hidden_state[1]] if env._hidden_state[0] == 1 else None,
                "equip_zone": env.cfg["equip_to_zone"][env._hidden_state[2]],
                "state_idx": env._hidden_state_idx,
            },
            "observation_bundle": obs,
            "observation_message": format_observation_message(obs),
            "tool_calls": [],
            "decision": None,
            "env_result": None,
        }

        self._inner.conversation_history = list(self.conversation_history)

        obs_msg = format_observation_message(obs)
        self.conversation_history.append({"role": "user", "content": obs_msg})
        self._inner.conversation_history.append({"role": "user", "content": obs_msg})

        tools = get_active_tool_schemas(obs)
        max_iterations = 10

        for iteration in range(max_iterations):
            llm_input = {
                "iteration": iteration,
                "messages_count": len(self._inner.conversation_history),
                "last_message_role": self._inner.conversation_history[-1]["role"],
                "tools_available": [t["function"]["name"] for t in tools],
            }

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
                step_trace["decision"] = {"allocation": {"A": 1}, "reasoning": "LLM fallback"}
                step_trace["env_result"] = {"reward": result["reward"], "done": result["done"]}
                self.step_traces.append(step_trace)
                return result

            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc["arguments"]
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)

                result_str, is_submit = execute_tool_call(env, tool_name, tool_args)

                call_record = {
                    "llm_input": llm_input,
                    "llm_output": f"tool_call: {tool_name}",
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "tool_result": _safe_parse_json(result_str),
                    "is_submit": is_submit,
                }
                step_trace["tool_calls"].append(call_record)

                tc_id = tc.get("id", f"call_{tool_name}")
                tc_args = tc.get("arguments", "{}")
                if isinstance(tc_args, dict):
                    tc_args = json.dumps(tc_args)
                normalized_tc = {
                    "id": tc_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": tc_args},
                }
                assistant_msg = {"role": "assistant", "content": None, "tool_calls": [normalized_tc]}
                tool_msg = {"role": "tool", "tool_call_id": tc_id, "content": result_str}
                self.conversation_history.append(assistant_msg)
                self.conversation_history.append(tool_msg)
                self._inner.conversation_history.append(assistant_msg)
                self._inner.conversation_history.append(tool_msg)

                if is_submit:
                    step_trace["decision"] = {
                        "allocation": tool_args.get("allocation", {}),
                        "beliefs": {
                            "storm_active": tool_args.get("storm_active", 0.5),
                            "storm_zone_probs": tool_args.get("storm_zone_probs", {}),
                            "equip_failure_active": tool_args.get("equip_failure_active", 0.2),
                            "equip_zone_probs": tool_args.get("equip_zone_probs", {}),
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
            allocation={"A": 1, "B": 0, "C": 0, "D": 0},
            beliefs={
                "storm_active": 0.5,
                "storm_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
                "equip_failure_active": 0.2,
                "equip_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
            },
            reasoning="Max iterations.",
        )
        step_trace["decision"] = {"allocation": {"A": 1}, "reasoning": "Max iterations"}
        step_trace["env_result"] = {"reward": result["reward"], "done": result["done"]}
        self.step_traces.append(step_trace)
        return result

    def _call_llm(self, messages, tools):
        return self._inner._call_llm(messages, tools)


def run_traced_episode(agent_cls=None, seed=42, config=None, save_path=None):
    """Run a full episode with tracing."""
    cfg = config or CONFIG
    inner = (agent_cls or SimulatedLLMAgent)(config=cfg)
    agent = TracedLLMAgent(inner, config=cfg)
    agent.reset()

    env = FishingGameEnv(config=cfg)
    obs = env.reset(seed=seed)

    import time
    for day_idx in range(cfg["episode_length"]):
        t0 = time.time()
        result = agent.act(env, obs)
        elapsed = time.time() - t0
        ctx_len = sum(len(json.dumps(m)) for m in agent._inner.conversation_history)
        print(f"  Day {day_idx+1}/{cfg['episode_length']}: "
              f"reward={result['reward']}, {elapsed:.1f}s, ctx~{ctx_len//1000}k chars",
              flush=True)
        if result["done"]:
            break
        obs = result["observation"]

    trace = env.get_trace()
    evaluator = Evaluator(config=cfg)
    eval_result = evaluator.evaluate_episode(trace)

    output = {
        "metadata": {
            "seed": seed,
            "agent": type(inner).__name__,
            "model": getattr(inner, "model", None),
            "timestamp": datetime.now().isoformat(),
            "episode_length": cfg["episode_length"],
        },
        "steps": agent.step_traces,
        "evaluation": {
            "total_reward": eval_result["total_reward"],
            "mean_brier_storm": eval_result["mean_brier_storm"],
            "mean_brier_storm_zone": eval_result["mean_brier_storm_zone"],
            "mean_brier_equip": eval_result["mean_brier_equip"],
            "mean_brier_equip_zone": eval_result["mean_brier_equip_zone"],
            "mean_storm_detection_lag": eval_result["mean_storm_detection_lag"],
            "mean_equip_detection_lag": eval_result["mean_equip_detection_lag"],
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
    """Pretty-print a traced episode."""
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

        print(f"  [HIDDEN] storm={hs['storm']}, wind={hs['wind']}, equip={hs['equip']}, "
              f"storm_zone={hs['storm_zone']}, equip_zone={hs['equip_zone']}")
        print()

        print(f"  [INPUT] sea_color={obs['sea_color']}, equip_indicator={obs['equip_indicator']}")
        print(f"    cumulative: {obs['cumulative_reward']}, tool_budget: {obs['tool_budget']}")
        print()

        print(f"  [TOOL CALLS] ({len(step['tool_calls'])} calls):")
        for i, tc in enumerate(step["tool_calls"]):
            name = tc["tool_name"]
            if name == "submit_decisions":
                continue
            args_short = json.dumps(tc["tool_args"], separators=(",", ":"))
            if len(args_short) > 80:
                args_short = args_short[:77] + "..."
            result_short = json.dumps(tc["tool_result"], separators=(",", ":"))
            if len(result_short) > 120:
                result_short = result_short[:117] + "..."
            print(f"    [{i+1}] {name}({args_short})")
            print(f"        -> {result_short}")
        print()

        print(f"  [OUTPUT] allocation={decision.get('allocation', {})}")
        if "beliefs" in decision:
            b = decision["beliefs"]
            print(f"    storm={b.get('storm_active', '?')}, equip={b.get('equip_failure_active', '?')}")
        print(f"    reasoning: {decision.get('reasoning', '')}")
        print(f"  [RESULT] reward={env_r['reward']}, done={env_r['done']}")

    ev = output["evaluation"]
    print(f"\n{'='*90}")
    print("EPISODE SUMMARY")
    print(f"{'='*90}")
    print(f"  Total reward:        {ev['total_reward']}")
    print(f"  Brier (storm):       {ev['mean_brier_storm']:.4f}")
    print(f"  Brier (equip):       {ev['mean_brier_equip']:.4f}")
    print(f"  Tool use gap:        {ev['total_tool_use_gap']:.1f}")
    print(f"  Inference gap:       {ev['total_inference_gap']:.1f}")
    print(f"  Planning gap:        {ev['total_planning_gap']:.1f}")
    print(f"  Tool usage:          {ev['tool_usage_counts']}")


def _safe_parse_json(s):
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
