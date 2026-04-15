"""
GPT-based LLM agent for the Fishing Game.

Uses OpenAI's tool-calling API to play the fishing game.
Loads API key from .env file via python-dotenv.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from fishing_game.config import CONFIG
from fishing_game.llm_agent import LLMAgent, TOOL_SCHEMAS, SYSTEM_PROMPT


# Load .env from the parent benchmark directory
_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_env_path)


class GPTAgent(LLMAgent):
    """
    OpenAI GPT agent that uses tool calling to play the fishing game.
    """

    def __init__(self, model="gpt-5.4", config=None):
        super().__init__(config)
        self.model = model
        self.client = OpenAI(timeout=120.0)  # 120s per request

    def _call_llm(self, messages, tools):
        """Call OpenAI chat completions with tool definitions."""
        # Convert tool schemas to OpenAI format
        openai_tools = []
        for t in tools:
            openai_tools.append({
                "type": "function",
                "function": t["function"],
            })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
        )

        choice = response.choices[0]

        if choice.message.tool_calls:
            return [
                {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                    "id": tc.id,
                }
                for tc in choice.message.tool_calls
            ]

        # If the model responded with text instead of a tool call,
        # append its text to the conversation and return None
        if choice.message.content:
            messages.append({
                "role": "assistant",
                "content": choice.message.content,
            })

        return None
