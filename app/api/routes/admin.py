from __future__ import annotations

import base64
import json
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, or_, and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, require_admin
from app.models.domain import Asset, MarketData, TrainingJob, User
from app.models.schemas import (
    AssetListItem,
    AssetListResponse,
    DashboardMetrics,
)

router = APIRouter(prefix="/admin", tags=["admin"])


def _encode_cursor(created_at: datetime, asset_id: int) -> str:
    payload = json.dumps({"dt": created_at.isoformat(), "id": asset_id})
    return base64.urlsafe_b64encode(payload.encode()).decode()


def _decode_cursor(cursor: str) -> tuple[datetime, int]:
    data = json.loads(base64.urlsafe_b64decode(cursor + "==").decode())
    return datetime.fromisoformat(data["dt"]), data["id"]


@router.get(
    "/dashboard",
    summary="Admin dashboard metrics",
    description=(
        "Returns aggregate counts for assets, training jobs, and users, "
        "plus a flag indicating whether any training job is currently running."
    ),
    response_model=DashboardMetrics,
)
async def get_dashboard(
    _: object = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> DashboardMetrics:
    assets_count = await db.scalar(select(func.count()).select_from(Asset))
    jobs_count = await db.scalar(select(func.count()).select_from(TrainingJob))
    users_count = await db.scalar(select(func.count()).select_from(User))
    market_data_count = await db.scalar(select(func.count()).select_from(MarketData))
    running_count = await db.scalar(
        select(func.count())
        .select_from(TrainingJob)
        .where(TrainingJob.status == "running")
    )
    return DashboardMetrics(
        assets_count=assets_count or 0,
        jobs_count=jobs_count or 0,
        users_count=users_count or 0,
        market_data_count=market_data_count or 0,
        training_running=(running_count or 0) > 0,
    )


@router.get(
    "/assets",
    summary="Paginated asset list",
    description=(
        "Returns a keyset-paginated list of assets ordered by `created_at DESC`.\n\n"
        "**Search** (`q`): case-insensitive substring match on company name, ISIN, or sector.\n\n"
        "**Pagination**: pass the `next_cursor` from the previous response as `cursor` to fetch "
        "the next page. Omit `cursor` to start from the first page.\n\n"
        "`market_data_count` is the total number of market-data rows ingested for each asset."
    ),
    response_model=AssetListResponse,
)
async def list_assets_paginated(
    q: str | None = Query(
        None,
        description="Search term — matched case-insensitively against name, ISIN, and sector.",
    ),
    limit: int = Query(
        20,
        ge=1,
        le=100,
        description="Number of items per page (1–100).",
    ),
    cursor: str | None = Query(
        None,
        description="Opaque keyset cursor returned by the previous page. Omit for first page.",
    ),
    _: object = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> AssetListResponse:
    md_count_sq = (
        select(func.count())
        .select_from(MarketData)
        .where(MarketData.asset_id == Asset.id)
        .correlate(Asset)
        .scalar_subquery()
    ).label("market_data_count")

    stmt = select(
        Asset.id,
        Asset.isin,
        Asset.name,
        Asset.sector,
        Asset.created_at,
        md_count_sq,
    )

    if q:
        stmt = stmt.where(
            or_(
                Asset.name.ilike(f"%{q}%"),
                Asset.isin.ilike(f"%{q}%"),
                Asset.sector.ilike(f"%{q}%"),
            )
        )

    if cursor:
        cursor_dt, cursor_id = _decode_cursor(cursor)
        stmt = stmt.where(
            or_(
                Asset.created_at < cursor_dt,
                and_(Asset.created_at == cursor_dt, Asset.id < cursor_id),
            )
        )

    stmt = stmt.order_by(Asset.created_at.desc(), Asset.id.desc()).limit(limit + 1)

    rows = (await db.execute(stmt)).fetchall()

    has_more = len(rows) > limit
    page = rows[:limit]

    items = [
        AssetListItem(
            id=r.id,
            isin=r.isin,
            name=r.name,
            sector=r.sector,
            created_at=r.created_at,
            market_data_count=r.market_data_count or 0,
        )
        for r in page
    ]

    next_cursor: str | None = None
    if has_more:
        last = page[-1]
        next_cursor = _encode_cursor(last.created_at, last.id)

    return AssetListResponse(items=items, next_cursor=next_cursor, has_more=has_more)
