"""
POMDP Discovery Agent for the Fishing Game.

Causal DISCOVERY (not inference): the agent is given only the tool API,
database schema, reward schedule, and action space. Hidden-state structure,
causal DAG, and observation models are NOT revealed — the agent must infer
them from historical data.

Contrast with:
  - LLMSolverAgent / PomdpCodingAgent: given full parameter schema
    (ESTIMATION_PROMPT). Must only learn numbers.
  - LLMAgent / CodingAgent (existing): given the causal mechanisms in natural
    language in the prompt. Must decide each day, no model.

This agent reuses the existing CodingAgent's Agno + PythonTools wiring — the
REPL, the FishingGameTools toolkit, the InMemoryDb, and the 20-day history
retention — and only swaps the system prompt for DISCOVERY_PROMPT.
"""

from fishing_game.agents.coding_agent import CodingAgent, FishingGameTools  # noqa: F401
from fishing_game.prompts.llm_solver import DISCOVERY_PROMPT

from pathlib import Path

from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.models.openai import OpenAIChat
from agno.tools.python import PythonTools


class PomdpDiscoveryAgent(CodingAgent):
    """CodingAgent with DISCOVERY_PROMPT instead of CodingAgent.SYSTEM_PROMPT.

    The only behavioral override is the system prompt: the agent starts without
    being told the hidden-state factorization. Everything else (Agno wiring,
    persistent REPL, 20-day history, fallback submit) is inherited.
    """

    def __init__(self, model="gpt-5.4", config=None):
        super().__init__(model=model, config=config)

    def _build_agent(self, env):
        """Same as CodingAgent._build_agent but uses DISCOVERY_PROMPT."""
        self._game_tools = FishingGameTools(env)
        self._agent = Agent(
            model=OpenAIChat(id=self.model_id),
            tools=[
                PythonTools(
                    base_dir=Path("tmp/pomdp_discovery_agent"),
                    exclude_tools=["pip_install_package", "uv_pip_install_package"],
                ),
                self._game_tools,
            ],
            instructions=[DISCOVERY_PROMPT],
            db=InMemoryDb(),
            add_history_to_context=True,
            num_history_runs=20,
            markdown=True,
        )
