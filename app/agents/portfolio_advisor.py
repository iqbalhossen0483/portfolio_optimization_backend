"""
Portfolio Advisor Agent factory.

Rebuilt per ChatService instance so that tool closures can capture per-request state.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool

from app.config import get_settings
from app.agents.instructions import PORTFOLIO_ADVISOR_INSTRUCTION
from app.agents.market_intelligence import market_agent
from app.agents.esg_research import esg_research_agent
from app.agents.tools import make_generate_portfolio, make_list_available_models

if TYPE_CHECKING:
    from app.services.chat_service import ChatService

cfg = get_settings()


def build_portfolio_advisor(service: "ChatService", username: str) -> LlmAgent:
    personalized_instruction = (
        f"The user's name is {username}. "
        "Greet them by name on the first response and address them occasionally "
        "throughout the conversation.\n\n"
    ) + PORTFOLIO_ADVISOR_INSTRUCTION

    return LlmAgent.model_validate({
        "name":        "portfolio_advisor",
        "model":       cfg.adk_model,
        "instruction": personalized_instruction,
        "tools": [
            make_generate_portfolio(service),
            make_list_available_models(service),
            AgentTool(agent=market_agent),
            AgentTool(agent=esg_research_agent),
        ],
    })
