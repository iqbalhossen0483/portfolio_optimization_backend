"""
FastAPI dependency injection — database session, Redis, and service factories.
"""
from __future__ import annotations
from typing import AsyncGenerator

import redis.asyncio as aioredis
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session
from app.config import get_settings

cfg = get_settings()


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


# ── Services ──────────────────────────────────────────────────────────────────

async def get_portfolio_service(
    db: AsyncSession = Depends(get_db),
    redis_client: aioredis.Redis = Depends(get_redis),
):
    from app.services.portfolio_service import PortfolioService
    return PortfolioService(db=db, redis_client=redis_client)


async def get_training_service(
    db: AsyncSession = Depends(get_db),
    redis_client: aioredis.Redis = Depends(get_redis),
):
    from app.services.training_service import TrainingService
    return TrainingService(db=db, redis_client=redis_client)
