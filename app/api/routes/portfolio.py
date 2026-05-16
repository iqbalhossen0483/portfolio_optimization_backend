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
        "Runs **Cooperative**, **Competitive**, and **Mixed** topologies concurrently "
        "using the best available trained MASAC model for the requested portfolio model.\n\n"
        "Returns three independent portfolio panels side-by-side so you can directly compare "
        "how game-theoretic framing changes asset weights and ESG metrics.\n\n"
        "**Portfolio models:**\n"
        "- `A` — ESG consensus: reward = α₁ · ESG_B_norm + α₂ · ESG_L_norm + financial return\n"
        "- `B` — Signed disagreement: Bloomberg agent bets ESG_B, LESG agent bets ESG_L\n"
        "- `C` — Full model: consensus + β · ΔESGᵢₜ uncertainty penalty (recommended)\n\n"
        "**Topology β effect on weights:**\n\n"
        "| Topology | β applied | Allocation behaviour |\n"
        "|---|---|---|\n"
        "| Cooperative | full β | Penalises ESG-ambiguous assets → conservative |\n"
        "| Competitive | β = 0 | Each agent maximises its own ESG source → divergent |\n"
        "| Mixed | partial β | Balanced — intermediate between the two |"
    ),
    response_model=PortfolioGenerateResponse,
    response_description="Three topology panels with per-asset allocations and aggregate metrics",
    responses={
        404: {"description": "No trained model found for the requested portfolio_model"},
        422: {"description": "Validation error — invalid ISIN list, negative allocation_amount, or unknown portfolio_model"},
    },
    status_code=status.HTTP_200_OK,
)
async def generate_portfolio(
    request: PortfolioGenerateRequest,
    service: PortfolioService = Depends(get_portfolio_service),
) -> PortfolioGenerateResponse:
    result = await service.generate(request)
    return result


@router.get(
    "/{query_id}/comparison",
    summary="Retrieve a previously generated portfolio comparison",
    description=(
        "Fetches a stored portfolio comparison by its `query_id` (returned by `POST /portfolio/generate`). "
        "All three topology panels are returned exactly as they were at generation time."
    ),
    response_description="Three topology panels for the requested query_id",
    responses={
        404: {"description": "No portfolio comparison found for the given query_id"},
    },
    status_code=status.HTTP_200_OK,
)
async def get_portfolio_comparison(
    query_id: str,
    service: PortfolioService = Depends(get_portfolio_service),
) -> dict:
    result = await service.get_comparison(query_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio comparison '{query_id}' not found",
        )
    return result
