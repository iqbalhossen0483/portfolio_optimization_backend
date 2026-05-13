from __future__ import annotations
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.config import get_settings

cfg = get_settings()

engine = create_async_engine(
    cfg.postgres_dsn,
    pool_size=cfg.postgres_pool_size,
    echo=cfg.debug,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


async def create_tables() -> None:
    from app.models.domain import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
