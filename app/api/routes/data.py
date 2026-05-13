from __future__ import annotations
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db
from app.models.domain import Asset
from app.models.schemas import AssetsResponse, AssetInfo, DataIngestionRequest, DataIngestionResponse
import uuid

router = APIRouter(prefix="/data", tags=["data"])


@router.get(
    "/assets",
    summary="List available assets",
)
async def list_assets(
    sector: str | None = Query(None, description="Filter by sector"),
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


@router.post(
    "/ingest",
    summary="Trigger data ingestion for a list of assets",
    description="Fetches OHLCV and ESG data from external APIs and stores in the database.",
)
async def ingest_data(
    request: DataIngestionRequest,
    db: AsyncSession = Depends(get_db),
) -> DataIngestionResponse:
    from app.workers.tasks import celery_app
    job_id = str(uuid.uuid4())
    # Enqueue data ingestion task
    celery_app.send_task(
        "ingest_data",
        args=[job_id, request.model_dump(mode="json")],
    )
    return DataIngestionResponse(
        job_id=job_id,
        assets_queued=len(request.assets),
        status="queued",
    )


@router.get(
    "/health",
    summary="Data layer health check",
)
async def data_health(db: AsyncSession = Depends(get_db)) -> dict:
    from sqlalchemy import text
    await db.execute(text("SELECT 1"))
    return {"status": "ok", "database": "connected"}
