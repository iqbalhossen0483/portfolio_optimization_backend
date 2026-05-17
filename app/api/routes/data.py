from __future__ import annotations
import uuid

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db
from app.models.domain import Asset
from app.models.schemas import (
    AssetsResponse, AssetInfo,
    DataIngestionRequest, DataIngestionResponse,
)

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
    summary="Trigger legacy data ingestion (yfinance / ESG stub)",
    description=(
        "Queues a Celery task to fetch OHLCV and ESG data from external APIs "
        "(yfinance for market data; Bloomberg or synthetic stub for ESG) "
        "and store the results in the database.\n\n"
        "> **Note:** This is the **legacy ingestion path**. "
        "The primary ingestion path is `POST /training/start` which accepts `.xlsx` files "
        "and runs the full 4-stage pipeline with real data. "
        "Use this endpoint only if you want to pre-populate the DB from yfinance / Bloomberg API."
    ),
    response_model=DataIngestionResponse,
    response_description="Job queued — use job_id to track progress",
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        422: {"description": "Validation error — invalid date range or empty asset list"},
    },
)
async def ingest_data(
    request: DataIngestionRequest,
    db: AsyncSession = Depends(get_db),
) -> DataIngestionResponse:
    from app.workers.tasks import celery_app
    job_id = str(uuid.uuid4())
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
    description="Verifies the API can reach PostgreSQL. Returns `ok` if a test query succeeds.",
    response_description="Database connectivity status",
    tags=["system"],
)
async def data_health(db: AsyncSession = Depends(get_db)) -> dict:
    from sqlalchemy import text
    await db.execute(text("SELECT 1"))
    return {"status": "ok", "database": "connected"}
