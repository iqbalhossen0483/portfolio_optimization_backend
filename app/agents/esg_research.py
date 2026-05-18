"""
ESG Research Analyst Agent — module-level singleton.

Built once at import time; no per-request state.
Focused exclusively on ESG ratings, controversies, Bloomberg vs LESG divergence,
and the real-world drivers behind high-ΔESG assets in the MASAC portfolio.
"""
import os

from google.adk.agents import LlmAgent
from google.adk.tools import google_search, url_context

from app.config import get_settings
from app.agents.instructions import ESG_RESEARCH_ANALYST_INSTRUCTION

cfg = get_settings()

if cfg.google_api_key:
    os.environ.setdefault("GOOGLE_API_KEY", cfg.google_api_key)

esg_research_agent = LlmAgent(
    name="esg_research",
    model=cfg.adk_model_market,
    instruction=ESG_RESEARCH_ANALYST_INSTRUCTION,
    tools=[google_search, url_context],
)
