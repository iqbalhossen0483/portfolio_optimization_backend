"""
ChatService — Google ADK-powered natural language interface to the portfolio system.

The ADK agent parses user intent (portfolio model + investment amount) and calls
generate_portfolio(), which runs InferenceService and returns three topology panels.

Session history is maintained via DatabaseSessionService (process-level singleton).
Each session_id maps to a full conversation history across multiple requests.
"""
from __future__ import annotations

import os
from typing import AsyncGenerator

import structlog
from google import genai as _genai

from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.genai.types import Content, Part

from app.config import get_settings
from app.agents import build_portfolio_advisor
from app.services.inference_service import InferenceService
from app.agents.instructions import _GUARD_PROMPT

log = structlog.get_logger(__name__)
cfg = get_settings()

# ADK reads the key from os.environ directly
if cfg.google_api_key:
    os.environ.setdefault("GOOGLE_API_KEY", cfg.google_api_key)

#  prevent asyncio context conflicts with FastAPI async context
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

_session_service = DatabaseSessionService(db_url=cfg.postgres_dsn)
_APP_NAME = "madrl_portfolio"

_TOOL_LABELS: dict[str, str] = {
    "generate_portfolio":    "Calculating portfolio...",
    "list_available_models": "Checking trained models...",
    "market_intelligence":   "Fetching market intelligence...",
    "esg_research":          "Researching ESG data...",
}

_WORKING_LABELS: dict[str, str] = {
    "portfolio_advisor":   "Thinking...",
    "market_intelligence": "Fetching market intelligence...",
    "esg_research":        "Researching ESG data...",
}


_BLOCKED_RESPONSES: dict[str, str] = {
    "off_topic": (
        "I'm a MASAC portfolio advisor focused on portfolio construction, market "
        "intelligence, and ESG research. I'm not able to help with that topic — "
        "how can I assist you with a portfolio or investment question?"
    ),
    "abusive": (
        "I'm not able to respond to that kind of message. I'm here to help with "
        "portfolio construction and investment analysis."
    ),
    "system_probe": (
        "I'm not able to share information about my internal configuration or architecture."
    ),
    "jailbreak": (
        "I'm a MASAC portfolio advisor focused on portfolio construction. "
        "How can I assist you today?"
    ),
}


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
        self._guard_client = _genai.Client(api_key=cfg.google_api_key)
        self._runner = self._build_runner()

    async def _classify_input(self, message: str) -> str:
        """
        Fast pre-check using Gemini Flash-Lite.
        Returns: relevant | off_topic | abusive | system_probe | jailbreak
        Fails open so legitimate users are never blocked by API errors.
        """
        try:
            resp = await self._guard_client.aio.models.generate_content(
                model=cfg.adk_model_guard,
                contents=_GUARD_PROMPT.format(message=message),
            )
            category = (resp.text or "").strip().lower()
            return category if category in _BLOCKED_RESPONSES else "relevant"
        except Exception as exc:
            log.warning("input_rail_failed", error=str(exc))
            return "relevant"

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

        category = await self._classify_input(message)
        if category in _BLOCKED_RESPONSES:
            return {
                "session_id": session_id,
                "response": _BLOCKED_RESPONSES[category],
                "portfolio_result": None,
            }

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

    async def stream_chat(
        self, session_id: str, message: str, user_id: str = "anonymous"
    ) -> AsyncGenerator[dict, None]:
        """
        Async generator yielding SSE-ready dicts as the agent pipeline progresses.

        Event shapes:
          {"type": "status",     "status": "thinking"|"calling_tool", "agent": str, "tool": str|None, "label": str, "content": ""}
          {"type": "text_chunk", "agent": str, "label": str, "content": str}
          {"type": "done",       "session_id": str, "response": str, "portfolio_result": dict|None}
        """
        self._portfolio_result = {}

        category = await self._classify_input(message)
        if category in _BLOCKED_RESPONSES:
            yield {
                "type": "done",
                "session_id": session_id,
                "response": _BLOCKED_RESPONSES[category],
                "portfolio_result": None,
            }
            return

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

        yield {
            "type": "status", "status": "thinking",
            "agent": "portfolio_advisor", "label": "Thinking...", "content": "",
        }

        new_message = Content(role="user", parts=[Part(text=message)])
        final_text = ""

        async for event in self._runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=new_message,
            run_config=RunConfig(max_llm_calls=25),
        ):
            # Tool calls being dispatched
            for fc in event.get_function_calls():
                tool_name = fc.name or ""
                label = _TOOL_LABELS.get(tool_name, f"Calling {tool_name}...")
                yield {
                    "type": "status", "status": "calling_tool",
                    "tool": tool_name, "agent": event.author,
                    "label": label, "content": "",
                }

            # Partial text chunks — tagged with whichever agent is streaming
            if event.partial and event.content and event.content.parts:
                label = _WORKING_LABELS.get(event.author, "Working...")
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        yield {
                            "type": "text_chunk",
                            "agent": event.author,
                            "label": label,
                            "content": part.text,
                        }

            # Final response — capture the orchestrator's complete text
            if event.is_final_response() and event.content and event.content.parts:
                if event.author == "portfolio_advisor":
                    final_text = "".join(
                        p.text for p in event.content.parts
                        if hasattr(p, "text") and p.text
                    )

        yield {
            "type": "done",
            "session_id": session_id,
            "response": final_text,
            "portfolio_result": self._portfolio_result if self._portfolio_result else None,
        }
