"""
ChatService — Google ADK-powered natural language interface to the portfolio system.

The ADK agent parses user intent (portfolio model + investment amount) and calls
generate_portfolio(), which runs InferenceService and returns three topology panels.

Session history is maintained via InMemorySessionService (process-level singleton).
Each session_id maps to a full conversation history across multiple requests.
"""
from __future__ import annotations

import json
import os
import structlog

from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.genai.types import Content, Part

from app.config import get_settings
from app.services.inference_service import InferenceService

log = structlog.get_logger(__name__)
cfg = get_settings()

# ADK reads the key from os.environ directly — pydantic-settings does not set it there.
if cfg.google_api_key:
    os.environ.setdefault("GOOGLE_API_KEY", cfg.google_api_key)

# prevent asyncio context conflicts with fastapi async context when ADK calls tools that access the database or other shared resources
os.environ.setdefault("OTEL_SDK_DISABLED", "true") 


_session_service = DatabaseSessionService(db_url=cfg.postgres_dsn)
_APP_NAME = "madrl_portfolio"

_AGENT_INSTRUCTION = """
You are the official portfolio advisor for the MADRL (Multi-Agent Deep Reinforcement Learning) Portfolio System.

The system uses three MASAC-based agents:
- Bloomberg ESG Agent
- LESG ESG Agent
- Financial Return Agent

The system generates portfolio allocations across three game-theoretic interaction topologies:
- Cooperative
- Competitive
- Mixed

Your role is to:
- interpret user investment requests,
- call the correct tools,
- explain portfolio outputs,
- compare topology behavior,
- and summarize ESG disagreement effects professionally.

━━━━━━━━━━━━━━━━━━
TOOL USAGE POLICY
━━━━━━━━━━━━━━━━━━

AVAILABLE TOOLS:
1. generate_portfolio
2. list_available_models

Use tools instead of reasoning manually whenever portfolio data is required.

Do not fabricate:
- portfolio allocations,
- Sharpe ratios,
- ESG metrics,
- trained model availability,
- topology outputs,
- or performance statistics.

━━━━━━━━━━━━━━━━━━
MODEL SELECTION RULES
━━━━━━━━━━━━━━━━━━

Portfolio models:

A:
- ESG consensus model
- no disagreement penalty

B:
- signed ESG disagreement model
- agents bet against opposing ESG providers

C:
- consensus + uncertainty penalty
- full model
- recommended default

MODEL ROUTING:

- If the user explicitly requests model "A":
  use portfolio_model="A"

- If the user explicitly requests model "B":
  use portfolio_model="B"

- In ALL other cases:
  use portfolio_model="C"

This includes:
- "best"
- "recommended"
- "default"
- "optimal"
- "full model"
- ambiguous requests
- unspecified requests

Never ask the user which model to use.

Never explain routing logic unless asked.

━━━━━━━━━━━━━━━━━━
INVESTMENT AMOUNT RULES
━━━━━━━━━━━━━━━━━━

Call generate_portfolio immediately if the investment amount is provided.

Automatically parse:
- "$10M"
- "$10 million"
- "10000000"
- "ten million"
- similar formats

Only ask for investment_amount if missing.

Do not ask unnecessary clarification questions.

━━━━━━━━━━━━━━━━━━
MAX ASSET SELECTION RULES
━━━━━━━━━━━━━━━━━━

Default:
- max_assets = 3

If the user explicitly requests:
- top 5
- 7 assets
- more holdings
→ use that value.

Otherwise:
- prefer concentrated portfolios
- only increase asset count when clearly justified

Never exceed 7 unless explicitly requested.

━━━━━━━━━━━━━━━━━━
WHEN TO CALL list_available_models
━━━━━━━━━━━━━━━━━━

Call list_available_models ONLY when the user asks:
- what models are trained
- what is available
- training status
- completed jobs
- available portfolio models
- inference availability

Do NOT call it before generate_portfolio.

The generate_portfolio tool already handles fallback behavior internally.

━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT
━━━━━━━━━━━━━━━━━━

When generate_portfolio returns results:

Present:
- Cooperative panel
- Competitive panel
- Mixed panel

For EACH topology panel include:

1. Topology title
2. Holdings table
3. Strategic summary

━━━━━━━━━━━━━━━━━━
HOLDINGS TABLE FORMAT
━━━━━━━━━━━━━━━━━━

Include:
- ISIN
- Company
- Weight %
- Allocation ($)
- Sharpe
- μESG
- ΔESG

Do not invent missing columns.

Use percentages clearly.

━━━━━━━━━━━━━━━━━━
STRATEGIC SUMMARY RULES
━━━━━━━━━━━━━━━━━━

For every topology:
- explain why allocations differ
- identify high-ΔESG assets
- explain disagreement effects
- explain risk-return tradeoffs

Key interpretation logic:

Cooperative:
- suppresses high disagreement assets

Competitive:
- may overweight high-return/high-ΔESG assets

Mixed:
- intermediate behavior

Keep summaries concise and quantitative.

━━━━━━━━━━━━━━━━━━
CROSS-TOPOLOGY ANALYSIS
━━━━━━━━━━━━━━━━━━

After all topology panels:
- compare allocation differences
- identify highest Sharpe profile
- identify highest ESG consensus profile
- identify highest disagreement exposure
- summarize how topology changed allocations

━━━━━━━━━━━━━━━━━━
INFORMATIONAL QUERIES
━━━━━━━━━━━━━━━━━━

For informational questions:
- explain system behavior directly
- explain MASAC behavior concisely
- explain ESG disagreement simply
- explain topology mechanics professionally

Do NOT generate portfolios unless requested.

━━━━━━━━━━━━━━━━━━
IMPORTANT CONSTRAINTS
━━━━━━━━━━━━━━━━━━

Never:
- fabricate metrics
- fabricate training jobs
- fabricate topology outputs
- merge topology panels
- ask unnecessary questions
- expose raw internal JSON
- expose backend implementation details
- explain internal fallback logic unless asked

Always:
- remain concise
- remain professional
- remain quantitative
- focus on allocation logic
- explain ESG disagreement effects clearly

Tone:
- institutional
- analytical
- portfolio-management focused
- concise
- professional
"""


class ChatService:
    """
    One instance per request (via FastAPI Depends).
    _portfolio_result is written by the tool closure and read by the route after chat() returns.
    No shared mutable state across concurrent requests.
    """

    def __init__(self, dsn: str, username: str = "user") -> None:
        self._dsn = dsn
        self._username = username
        self._inference = InferenceService(dsn)
        self._portfolio_result: dict = {}
        self._runner = self._build_runner()

    def _build_runner(self) -> Runner:
        service = self   # captured by tool closures

        async def generate_portfolio(
            portfolio_model: str,
            investment_amount: float,
            max_assets: int = 3,
        ) -> str:
            """
            Generate portfolio allocations for the given model across all three topologies.

            Args:
                portfolio_model: Must be "A", "B", or "C".
                    A = ESG consensus only (no disagreement penalty)
                    B = signed ESG disagreement (each agent bets its ESG source is correct)
                    C = full model — consensus + uncertainty penalty (recommended)
                investment_amount: Total investment in USD (e.g. 10000000.0 for $10 million)
                max_assets: How many top assets to return per topology panel (default 3).
                    Choose this dynamically: use 3 when assets drop off in Sharpe quality,
                    up to 5–7 when many assets show positive Sharpe and meaningful weights.
                    Use whatever the user requested if they specified a number.

            Returns:
                JSON with panel summaries for cooperative, competitive, and mixed topologies.
            """
            try:
                model_key = portfolio_model.upper()
                if model_key not in ("A", "B", "C"):
                    model_key = "C"

                # If requested model has no training job, fall back to C silently
                try:
                    result = await service._inference.run(model_key, investment_amount)
                except ValueError as ve:
                    if "No completed training job" in str(ve) and model_key != "C":
                        log.warning("model_not_trained_falling_back", requested=model_key)
                        model_key = "C"
                        result = await service._inference.run(model_key, investment_amount)
                    else:
                        raise

                n = max(1, int(max_assets))

                # Accumulate trimmed panels — keys: "{MODEL}_{topology}"
                if not service._portfolio_result:
                    service._portfolio_result = {
                        "job_id": result["job_id"],
                        "portfolio_model": model_key,
                        "panels": {},
                    }
                else:
                    service._portfolio_result["portfolio_model"] = "ALL"

                for topology, assets in result["panels"].items():
                    service._portfolio_result["panels"][f"{model_key}_{topology}"] = assets[:n]

                # Return a concise summary for the LLM to narrate — not the full panel
                summaries: dict = {}
                for topology, assets in result["panels"].items():
                    total_w = sum(a["weight"] for a in assets)
                    top_n = assets[:n]
                    summaries[topology] = {
                        "top_holdings": [
                            {
                                "isin":       a["isin"],
                                "company":    a["company"],
                                "weight_pct": round(a["weight"] * 100, 1),
                                "allocation": a["allocation"],
                                "sharpe":     a["sharpe"],
                                "mu_esg":     a["mu_esg"],
                                "delta_esg":  a["delta_esg"],
                            }
                            for a in top_n
                        ],
                        "total_weight_check": round(total_w, 4),
                    }

                return json.dumps({
                    "success": True,
                    "portfolio_model": model_key,
                    "job_id": result["job_id"],
                    "n_assets": len(next(iter(result["panels"].values()))),
                    "panel_summaries": summaries,
                })

            except Exception as exc:
                service._portfolio_result = {}
                log.error("generate_portfolio_failed", error=str(exc))
                return json.dumps({"error": str(exc)})

        async def list_available_models() -> str:
            """
            List all completed training jobs available for inference.
            Call this when the user asks what models have been trained,
            or before generating a portfolio to confirm one exists.
            """
            from sqlalchemy import select
            from sqlalchemy.ext.asyncio import (
                AsyncSession, async_sessionmaker, create_async_engine
            )
            from app.models.domain import TrainingJob

            engine = create_async_engine(service._dsn)
            try:
                SessionLocal = async_sessionmaker(
                    engine, class_=AsyncSession, expire_on_commit=False
                )
                async with SessionLocal() as db:
                    result = await db.execute(
                        select(TrainingJob)
                        .where(TrainingJob.status == "completed")
                        .order_by(TrainingJob.id.desc())
                    )
                    jobs = result.scalars().all()

                if not jobs:
                    return json.dumps({
                        "message": (
                            "No completed training jobs found. "
                            "Train a model first via POST /api/v1/training/start."
                        )
                    })

                return json.dumps({
                    "available_models": [
                        {
                            "job_id":          j.id,
                            "portfolio_model": j.portfolio_model,
                            "topology":        j.topology,
                            "best_sharpe":     j.best_sharpe,
                            "best_mu_esg":     j.best_mu_esg,
                        }
                        for j in jobs
                    ]
                })
            finally:
                await engine.dispose()

        agent = LlmAgent.model_validate({
            "name":  "portfolio_advisor",
            "model": cfg.adk_model,
            "instruction": (
                f"The user's name is {self._username}. "
                "Greet them by name on the first response and address them occasionally "
                "throughout the conversation.\n\n"
            ) + _AGENT_INSTRUCTION,
            "tools": [generate_portfolio, list_available_models],
        })
        return Runner(
            agent=agent,
            app_name="madrl_portfolio",
            session_service=_session_service,
        )

    async def chat(self, session_id: str, message: str, user_id: str = "anonymous") -> dict:
        """
        Send a message and return the agent's response + any generated portfolio data.
        """
        self._portfolio_result = {}

        existing = await _session_service.get_session(
            app_name=_APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        if existing is None:
            await _session_service.create_session(
                app_name=_APP_NAME,
                user_id=user_id,
                session_id=session_id,
            )

        new_message = Content(role="user", parts=[Part(text=message)])
        final_text = ""

        async for event in self._runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=new_message,
            run_config=RunConfig(max_llm_calls=10),
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_text = "".join(
                    p.text
                    for p in event.content.parts
                    if hasattr(p, "text") and p.text
                )

        return {
            "session_id": session_id,
            "response":   final_text,
            "portfolio_result": self._portfolio_result if self._portfolio_result else None,
        }
