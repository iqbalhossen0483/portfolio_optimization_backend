"""
FastAPI dependency injection — database session, Redis, service factories, and auth guards.
"""
from __future__ import annotations
from typing import AsyncGenerator

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session
from app.config import get_settings

cfg = get_settings()

_bearer = HTTPBearer(auto_error=False)


# ── Database ──────────────────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_async_session():
        yield session


# ── Redis ─────────────────────────────────────────────────────────────────────

_redis_pool: aioredis.Redis | None = None


def get_redis_pool() -> aioredis.Redis:
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = aioredis.from_url(cfg.redis_url, decode_responses=True)
    return _redis_pool


async def get_redis() -> aioredis.Redis:
    return get_redis_pool()


# ── Auth ──────────────────────────────────────────────────────────────────────

async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
):
    """Resolves the current user from the Bearer token. Raises 401 if missing/invalid."""
    from app.core.security import decode_token
    from app.models.domain import User

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = decode_token(credentials.credentials)
        user_id: str | None = payload.get("sub")
        if not user_id:
            raise ValueError("Missing subject")
    except (JWTError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    result = await db.execute(select(User).where(User.id == int(user_id)))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or disabled")
    return user


async def require_admin(current_user=Depends(get_current_user)):
    """Raises 403 if the authenticated user is not an admin."""
    if current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return current_user


# ── WebSocket token resolver (query-param based, no HTTP headers over WS) ────

async def get_ws_user(token: str, db: AsyncSession):
    """Validate a JWT token string and return the user (for WebSocket endpoints)."""
    from app.core.security import decode_token
    from app.models.domain import User

    try:
        payload = decode_token(token)
        user_id = payload.get("sub")
        if not user_id:
            return None
    except JWTError:
        return None

    result = await db.execute(select(User).where(User.id == int(user_id)))
    user = result.scalar_one_or_none()
    return user if user and user.is_active else None


# ── Services ──────────────────────────────────────────────────────────────────

async def get_training_service(
    db: AsyncSession = Depends(get_db),
    redis_client: aioredis.Redis = Depends(get_redis),
):
    from app.services.training_service import TrainingService
    return TrainingService(db=db, redis_client=redis_client)


async def get_chat_service(
    user=Depends(get_current_user),
):
    from app.services.chat_service import ChatService
    return ChatService(dsn=cfg.postgres_dsn, username=user.username)
