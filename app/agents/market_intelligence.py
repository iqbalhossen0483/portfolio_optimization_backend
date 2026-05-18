"""
Market Intelligence Agent — module-level singleton.

Built once at import time; no per-request state.
Uses google_search (grounded web search) and url_context (page fetch) as tools.
"""
import os

from google.adk.agents import LlmAgent
from google.adk.tools import google_search, url_context

from app.config import get_settings
from app.agents.instructions import MARKET_INTELLIGENCE_INSTRUCTION

cfg = get_settings()

if cfg.google_api_key:
    os.environ.setdefault("GOOGLE_API_KEY", cfg.google_api_key)

market_agent = LlmAgent(
    name="market_intelligence",
    model=cfg.adk_model_market,
    instruction=MARKET_INTELLIGENCE_INSTRUCTION,
    tools=[google_search, url_context],
)
