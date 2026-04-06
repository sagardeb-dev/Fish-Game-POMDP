from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib import error, request

from dotenv import load_dotenv

from fishing_game.config import CONFIG
from fishing_game.evaluator import Evaluator
from fishing_game.llm_agent import (
    SYSTEM_PROMPT,
    execute_tool_call,
    format_observation_message,
    get_active_tool_schemas,
)
from fishing_game.simulator import FishingGameEnv


def _load_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)


def _debug_enabled() -> bool:
    return os.getenv("ANTHROPIC_DEBUG_TRACE", "1") == "1"


def _debug_dir() -> Path:
    path = Path(__file__).resolve().parent / "traces" / "anthropic_debug"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_debug_json(name: str, payload: object) -> None:
    if not _debug_enabled():
        return
    out_path = _debug_dir() / name
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _convert_tools(obs: dict) -> list[dict]:
    anthropic_tools = []
    for tool in get_active_tool_schemas(obs):
        fn = tool["function"]
        anthropic_tools.append({
            "name": fn["name"],
            "description": fn["description"],
            "input_schema": fn["parameters"],
        })
    return anthropic_tools


def _anthropic_request(api_key: str, model: str, system: str, messages: list[dict], tools: list[dict]) -> dict:
    payload = {
        "model": model,
        "system": system,
        "max_tokens": 2048,
        "messages": messages,
        "tools": tools,
    }
    req = request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    with request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


class AnthropicFishingAgent:
    def __init__(self, model: str) -> None:
        self.model = model
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not found in parent .env")
        self.messages: list[dict] = []
        self.request_idx = 0

    def reset(self) -> None:
        self.messages = []
        self.request_idx = 0

    def act(self, env: FishingGameEnv, obs: dict) -> dict:
        obs_message = format_observation_message(obs)
        self.messages.append({
            "role": "user",
            "content": [{"type": "text", "text": obs_message}],
        })

        tools = _convert_tools(obs)

        for _ in range(12):
            self.request_idx += 1
            _write_debug_json(
                f"request_{self.request_idx:03d}.json",
                {
                    "system": SYSTEM_PROMPT,
                    "messages": self.messages,
                    "tools": tools,
                },
            )
            response = _anthropic_request(
                api_key=self.api_key,
                model=self.model,
                system=SYSTEM_PROMPT,
                messages=self.messages,
                tools=tools,
            )
            _write_debug_json(f"response_{self.request_idx:03d}.json", response)
            content = response.get("content", [])
            stop_reason = response.get("stop_reason")
            has_tool_use = any(block.get("type") == "tool_use" for block in content)

            # Anthropic may still return tool_use blocks when stop_reason is max_tokens.
            # Tool-use content must always be handled first.
            if has_tool_use:
                result = self._handle_tool_use_turn(env, content)
                if result is not None:
                    return result
                continue

            self.messages.append({
                "role": "assistant",
                "content": content,
            })

            if stop_reason == "end_turn":
                text = "".join(
                    block.get("text", "") for block in content if block.get("type") == "text"
                ).strip()
                self.messages.append({
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": (
                            "You must finish this turn by calling submit_decisions. "
                            f"Your last response was: {text}"
                        ),
                    }],
                })
                continue

            if stop_reason == "max_tokens":
                self.messages.append({
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "Continue and finish by calling submit_decisions.",
                    }],
                })
                continue

            break

        return env.submit_decisions(
            allocation={"A": 1, "B": 0, "C": 0, "D": 0},
            beliefs={
                "storm_active": 0.5,
                "storm_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
                "equip_failure_active": 0.2,
                "equip_zone_probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
                "tide_high": 0.5,
            },
            reasoning="Anthropic runner fallback: model did not call submit_decisions.",
        )

    def _handle_tool_use_turn(self, env: FishingGameEnv, content: list[dict]) -> dict | None:
        tool_use_blocks = [block for block in content if block.get("type") == "tool_use"]
        if not tool_use_blocks:
            self.messages.append({
                "role": "assistant",
                "content": content,
            })
            return None

        # Anthropic requires the assistant tool_use message to be followed
        # immediately by a user message containing matching tool_result blocks.
        assistant_tool_message = {
            "role": "assistant",
            "content": tool_use_blocks,
        }
        self.messages.append(assistant_tool_message)

        tool_results = []
        saw_submit = False
        for block in tool_use_blocks:
            tool_name = block["name"]
            tool_args = block.get("input", {})
            try:
                result_str, is_submit = execute_tool_call(env, tool_name, tool_args)
            except Exception as exc:
                result_str = json.dumps({
                    "error": f"Tool execution failed for {tool_name}: {exc}",
                    "received_args": tool_args,
                })
                is_submit = False
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block["id"],
                "content": result_str,
            })
            if is_submit:
                saw_submit = True

        _write_debug_json(
            f"tool_pair_{self.request_idx:03d}.json",
            {
                "assistant_tool_use": assistant_tool_message,
                "user_tool_results": {
                    "role": "user",
                    "content": tool_results,
                },
            },
        )

        self.messages.append({
            "role": "user",
            "content": tool_results,
        })

        if saw_submit:
            return env._last_submit_result
        return None


def run_episode(seed: int, model: str) -> dict:
    env = FishingGameEnv(config=CONFIG)
    obs = env.reset(seed=seed)
    agent = AnthropicFishingAgent(model=model)
    agent.reset()

    total_reward = 0.0
    for day in range(CONFIG["episode_length"]):
        result = agent.act(env, obs)
        total_reward += result["reward"]
        print(
            f"Day {day + 1}/{CONFIG['episode_length']}: reward={result['reward']}, total={total_reward}",
            flush=True,
        )
        if result["done"]:
            break
        obs = result["observation"]

    trace = env.get_trace()
    evaluation = Evaluator(config=CONFIG).evaluate_episode(trace)
    return {
        "seed": seed,
        "model": model,
        "total_reward": evaluation["total_reward"],
        "mean_brier_storm": evaluation["mean_brier_storm"],
        "mean_brier_equip": evaluation["mean_brier_equip"],
        "total_tool_use_gap": evaluation["total_tool_use_gap"],
        "total_inference_gap": evaluation["total_inference_gap"],
        "total_planning_gap": evaluation["total_planning_gap"],
        "tool_usage_counts": evaluation["tool_usage_counts"],
    }


def main() -> int:
    _load_env()
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    try:
        result = run_episode(seed=seed, model=model)
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        _write_debug_json(
            "last_http_error.json",
            {
                "status": exc.code,
                "error": details,
            },
        )
        print(json.dumps({
            "ok": False,
            "status": exc.code,
            "error": details,
        }, indent=2))
        return 1
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "error": str(exc),
        }, indent=2))
        return 2

    print("\nEpisode Summary")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
