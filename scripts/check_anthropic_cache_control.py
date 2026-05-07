"""Smoke-test Anthropic prompt caching without touching benchmark code.

This script makes two Anthropic Messages API calls with the same cached static
prefix and different user suffixes. A successful cache test should show:

1. First call: cache_creation_input_tokens > 0
2. Second call: cache_read_input_tokens > 0

It intentionally uses urllib instead of the Anthropic SDK so that this checks
the raw API parameter path and does not add project dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def make_static_prefix(repetitions: int) -> str:
    base = (
        "CACHE_CONTROL_SMOKE_TEST_STATIC_PREFIX. "
        "This line is intentionally repetitive and stable across requests. "
        "It exists only to exceed Anthropic's minimum cacheable token threshold "
        "for Claude Sonnet 4.6. Do not infer benchmark behavior from it. "
    )
    return base * repetitions


def post_message(api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic API HTTP {exc.code}: {body}") from exc


def usage_summary(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage", {})
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cache_creation": usage.get("cache_creation", {}),
    }


def build_payload(model: str, static_prefix: str, question: str) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": 32,
        "temperature": 0,
        "system": [
            {
                "type": "text",
                "text": static_prefix,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": question}],
            }
        ],
    }


def main() -> int:
    default_env = Path.cwd().parent / ".env"
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=default_env)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument(
        "--prefix-repetitions",
        type=int,
        default=120,
        help="Increase if cache_creation_input_tokens stays 0 due token threshold.",
    )
    args = parser.parse_args()

    load_env_file(args.env_file)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY missing. Pass --env-file or set the env var.", file=sys.stderr)
        return 2

    static_prefix = make_static_prefix(args.prefix_repetitions)
    payload_1 = build_payload(args.model, static_prefix, "Reply with exactly: first")
    payload_2 = build_payload(args.model, static_prefix, "Reply with exactly: second")

    print(f"model={args.model}")
    print(f"env_file={args.env_file}")
    print(f"static_prefix_chars={len(static_prefix)}")
    print("sending first request; expected cache_creation_input_tokens > 0")
    first = post_message(api_key, payload_1)
    first_usage = usage_summary(first)
    print("first_usage=" + json.dumps(first_usage, sort_keys=True))

    time.sleep(2)

    print("sending second request; expected cache_read_input_tokens > 0")
    second = post_message(api_key, payload_2)
    second_usage = usage_summary(second)
    print("second_usage=" + json.dumps(second_usage, sort_keys=True))

    creation = int(first_usage["cache_creation_input_tokens"] or 0)
    read = int(second_usage["cache_read_input_tokens"] or 0)
    if creation > 0 and read > 0:
        print("PASS: cache_control was accepted and the second request read from cache.")
        return 0
    if creation > 0:
        print("PARTIAL: cache_control was accepted, but second request did not hit cache.")
        return 1
    print("FAIL: cache_control request succeeded, but no cache was created.")
    print("Likely causes: cached prefix below model threshold, unsupported model, or cache skipped.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
