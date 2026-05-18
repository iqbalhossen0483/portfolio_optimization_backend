"""
ChatService — Google ADK-powered natural language interface to the portfolio system.

The ADK agent parses user intent (portfolio model + investment amount) and calls
generate_portfolio(), which runs InferenceService and returns three topology panels.

Session history is maintained via DatabaseSessionService (process-level singleton).
Each session_id maps to a full conversation history across multiple requests.
"""
from __future__ import annotations

import os
import structlog

from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.genai.types import Content, Part

from app.config import get_settings
from app.agents import build_portfolio_advisor
from app.services.inference_service import InferenceService

log = structlog.get_logger(__name__)
cfg = get_settings()

# ADK reads the key from os.environ directly
if cfg.google_api_key:
    os.environ.setdefault("GOOGLE_API_KEY", cfg.google_api_key)

# prevent asyncio context conflicts with FastAPI async context
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

_session_service = DatabaseSessionService(db_url=cfg.postgres_dsn)
_APP_NAME = "madrl_portfolio"


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
        agent = build_portfolio_advisor(self, self._username)
        return Runner(
            agent=agent,
            app_name=_APP_NAME,
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
            run_config=RunConfig(max_llm_calls=25),
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
