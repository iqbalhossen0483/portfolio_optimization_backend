from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_current_user, get_db
from app.models.domain import Asset
from app.models.schemas import AssetsResponse, AssetInfo

router = APIRouter(prefix="/data", tags=["data"])


@router.get(
    "/assets",
    summary="List all ingested assets",
    description=(
        "Returns every asset that has been ingested into the database, "
        "optionally filtered by sector.\n\n"
        "Assets are populated automatically when `.xlsx` files are uploaded to "
        "`POST /training/start`. The N returned here equals the number of distinct "
        "ISINs across all uploaded files."
    ),
    response_model=AssetsResponse,
    response_description="List of assets with ISIN, name, and sector",
    responses={},
)
async def list_assets(
    sector: str | None = Query(
        None,
        description="Filter by sector name (case-sensitive). Omit to return all sectors.",
        examples=["Technology"],
    ),
    _: object = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AssetsResponse:
    q = select(Asset)
    if sector:
        q = q.where(Asset.sector == sector)
    result = await db.execute(q)
    assets = result.scalars().all()
    return AssetsResponse(
        assets=[AssetInfo(isin=a.isin, name=a.name, sector=a.sector) for a in assets],
        total=len(assets),
    )


@router.get(
    "/health",
    summary="Data layer health check",
    description="Verifies the API can reach PostgreSQL. Returns `ok` if a test query succeeds.",
    response_description="Database connectivity status",
    tags=["system"],
)
async def data_health(db: AsyncSession = Depends(get_db)) -> dict:
    from sqlalchemy import text
    await db.execute(text("SELECT 1"))
    return {"status": "ok", "database": "connected"}
