from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_chat_service, get_current_user, get_db
from app.models.domain import ChatMessage, ChatSession, User
from app.models.schemas import (
    ChatMessageInfo,
    ChatRequest,
    ChatResponse,
    ChatSessionDetail,
    ChatSessionInfo,
    ChatSessionListResponse,
    PortfolioAssetPanel,
    RenameChatSessionRequest,
)
from app.services.chat_service import ChatService

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _auto_name_session(message: str, max_len: int = 60) -> str:
    text = message.strip()
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated + "..."


async def _get_or_create_session(
    db: AsyncSession, user_id: int, session_id: str, first_message: str
) -> ChatSession:
    result = await db.execute(
        select(ChatSession).where(ChatSession.session_id == session_id)
    )
    existing = result.scalar_one_or_none()
    if existing:
        return existing
    new_session = ChatSession(
        user_id=user_id,
        session_id=session_id,
        name=_auto_name_session(first_message),
    )
    db.add(new_session)
    await db.flush()  # assign PK without committing — message FKs resolved in same tx
    return new_session


# ── Session endpoints ─────────────────────────────────────────────────────────

@router.get(
    "/sessions",
    response_model=ChatSessionListResponse,
    summary="List your chat sessions",
    description="Returns all chat sessions for the authenticated user, ordered by most recently active.",
    status_code=200,
)
async def list_sessions(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ChatSessionListResponse:
    msg_count_subq = (
        select(
            ChatMessage.chat_session_id,
            func.count(ChatMessage.id).label("message_count"),
        )
        .group_by(ChatMessage.chat_session_id)
        .subquery()
    )
    stmt = (
        select(
            ChatSession,
            func.coalesce(msg_count_subq.c.message_count, 0).label("message_count"),
        )
        .outerjoin(msg_count_subq, msg_count_subq.c.chat_session_id == ChatSession.id)
        .where(ChatSession.user_id == user.id)
        .order_by(ChatSession.updated_at.desc())
    )
    rows = (await db.execute(stmt)).all()
    sessions = [
        ChatSessionInfo(
            id=row.ChatSession.id,
            session_id=row.ChatSession.session_id,
            name=row.ChatSession.name,
            created_at=row.ChatSession.created_at,
            updated_at=row.ChatSession.updated_at,
            message_count=row.message_count,
        )
        for row in rows
    ]
    return ChatSessionListResponse(sessions=sessions, total=len(sessions))


@router.get(
    "/sessions/{session_id}",
    response_model=ChatSessionDetail,
    summary="Get a chat session with full message history",
    description="Returns the session metadata and all messages in chronological order.",
    responses={
        403: {"description": "Session belongs to another user"},
        404: {"description": "Session not found"},
    },
    status_code=200,
)
async def get_session(
    session_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ChatSessionDetail:
    result = await db.execute(
        select(ChatSession).where(ChatSession.session_id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this session")

    msg_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.chat_session_id == session.id)
        .order_by(ChatMessage.id)
    )
    messages = msg_result.scalars().all()

    return ChatSessionDetail(
        id=session.id,
        session_id=session.session_id,
        name=session.name,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=[ChatMessageInfo.model_validate(m) for m in messages],
    )


@router.patch(
    "/sessions/{session_id}",
    response_model=ChatSessionInfo,
    summary="Rename a chat session",
    responses={
        403: {"description": "Session belongs to another user"},
        404: {"description": "Session not found"},
    },
    status_code=200,
)
async def rename_session(
    session_id: str,
    body: RenameChatSessionRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ChatSessionInfo:
    result = await db.execute(
        select(ChatSession).where(ChatSession.session_id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to modify this session")

    session.name = body.name
    await db.commit()
    await db.refresh(session)

    count_result = await db.execute(
        select(func.count(ChatMessage.id)).where(ChatMessage.chat_session_id == session.id)
    )
    message_count = count_result.scalar_one()

    return ChatSessionInfo(
        id=session.id,
        session_id=session.session_id,
        name=session.name,
        created_at=session.created_at,
        updated_at=session.updated_at,
        message_count=message_count,
    )


@router.delete(
    "/sessions/{session_id}",
    summary="Delete a chat session and all its messages",
    description="Deletes the session and all associated messages. Best-effort ADK session cleanup is also attempted.",
    responses={
        403: {"description": "Session belongs to another user"},
        404: {"description": "Session not found"},
    },
    status_code=204,
)
async def delete_session(
    session_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    result = await db.execute(
        select(ChatSession).where(ChatSession.session_id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this session")

    await db.delete(session)  # CASCADE handles chat_messages at DB level
    await db.commit()

    try:
        from app.services.chat_service import _APP_NAME, _session_service
        await _session_service.delete_session(
            app_name=_APP_NAME,
            user_id=str(user.id),
            session_id=session_id,
        )
    except Exception as exc:
        log.warning("adk_session_delete_failed", error=str(exc), session_id=session_id)


# ── Chat endpoint ─────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=ChatResponse,
    summary="Chat with the portfolio advisor",
    description=(
        "Send a natural language message to the MASAC portfolio advisor agent.\n\n"
        "**Example queries:**\n"
        "- *'I have $10,000,000 to allocate. Use Portfolio C.'*\n"
        "- *'What portfolio models have been trained?'*\n"
        "- *'Explain the difference between the cooperative and competitive panels.'*\n\n"
        "On the **first message**, omit `session_id` — the server generates one.\n"
        "Include the returned `session_id` in all follow-up messages to continue the conversation.\n\n"
        "When a portfolio is generated, `panels` contains three topology results "
        "(cooperative, competitive, mixed) each with per-asset weights and allocations.\n"
        "`panels` is `null` for purely conversational messages."
    ),
    responses={
        202: {"description": "Agent response with optional portfolio panels"},
        503: {"description": "Google ADK or Gemini API unavailable"},
    },
    status_code=202,
)
async def chat(
    request: ChatRequest,
    user: User = Depends(get_current_user),
    service: ChatService = Depends(get_chat_service),
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    session_id = request.session_id or str(uuid.uuid4())

    try:
        result = await service.chat(session_id, request.message, str(user.id))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    panels = None
    pr = result.get("portfolio_result")
    if pr and pr.get("panels"):
        panels = {
            topology: [PortfolioAssetPanel(**asset) for asset in assets]
            for topology, assets in pr["panels"].items()
        }

    response = ChatResponse(
        session_id=session_id,
        response=result["response"],
        job_id=pr.get("job_id") if pr else None,
        portfolio_model=pr.get("portfolio_model") if pr else None,
        panels=panels,
    )

    # Persist to our own tables — failure-safe so DB errors never break the chat response
    try:
        chat_session = await _get_or_create_session(db, user.id, session_id, request.message)
        db.add(ChatMessage(chat_session_id=chat_session.id, role="user",      content=request.message))
        db.add(ChatMessage(chat_session_id=chat_session.id, role="assistant", content=result["response"]))
        await db.execute(
            sa_update(ChatSession)
            .where(ChatSession.id == chat_session.id)
            .values(updated_at=func.now())
        )
        await db.commit()
    except Exception as persist_exc:
        log.warning("chat_persistence_failed", error=str(persist_exc), session_id=session_id)
        await db.rollback()

    return response
