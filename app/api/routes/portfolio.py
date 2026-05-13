from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_portfolio_service
from app.models.schemas import PortfolioGenerateRequest, PortfolioGenerateResponse
from app.services.portfolio_service import PortfolioService

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.post(
    "/generate",
    summary="Generate portfolio comparison across all three topologies",
    description=(
        "Runs three game-theoretic topologies concurrently (Cooperative, Competitive, Mixed). "
        "Returns three independent portfolio panels for side-by-side comparison. "
        "Uses the best available trained MASAC model for the requested portfolio model (A/B/C)."
    ),
)
async def generate_portfolio(
    request: PortfolioGenerateRequest,
    service: PortfolioService = Depends(get_portfolio_service),
) -> dict:
    result = await service.generate(request)
    return result


@router.get(
    "/{query_id}/comparison",
    summary="Retrieve a previously generated portfolio comparison",
)
async def get_portfolio_comparison(
    query_id: str,
    service: PortfolioService = Depends(get_portfolio_service),
) -> dict:
    result = await service.get_comparison(query_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio comparison {query_id} not found",
        )
    return result
