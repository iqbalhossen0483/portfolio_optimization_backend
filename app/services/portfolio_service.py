"""
PortfolioService — business logic for portfolio generation.
Loads latest trained model, runs DataPipeline, calls OrchestratorAgent.
"""
from __future__ import annotations
import uuid
from datetime import date
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.domain import Asset, PortfolioResult, TrainingJob
from app.models.schemas import PortfolioGenerateRequest
from app.agents.portfolio_orchestrator import PortfolioOrchestratorAgent
from app.data.pipeline import DataPipeline
from app.data.sources.market import MarketDataSource
from app.data.sources.esg import ESGDataSource
from app.config import get_settings

log = structlog.get_logger(__name__)
cfg = get_settings()


class PortfolioService:

    def __init__(
        self,
        db: AsyncSession,
        redis_client=None,
        market_source: MarketDataSource | None = None,
        esg_source: ESGDataSource | None = None,
    ) -> None:
        self._db = db
        self._redis = redis_client
        self._market = market_source or MarketDataSource(redis_client)
        self._esg    = esg_source    or ESGDataSource(
            bloomberg_api_key=cfg.bloomberg_api_key,
            lesg_api_key=cfg.lesg_api_key,
            redis_client=redis_client,
            use_stub=(not cfg.bloomberg_api_key),
        )
        self._pipeline = DataPipeline(self._market, self._esg)

    async def generate(self, request: PortfolioGenerateRequest) -> dict[str, Any]:
        """
        Full portfolio generation flow:
        1. Resolve best trained model for requested portfolio_model
        2. Fetch and preprocess market data as of request date
        3. Run orchestrator for all three topologies
        4. Persist results and return comparison panels
        """
        query_id = str(uuid.uuid4())
        as_of = request.as_of_date or date.today()

        # ── 1. Resolve best model ─────────────────────────────────────────────
        job_id = await self._get_best_job(request.portfolio_model.value)
        if not job_id:
            log.warning("No trained model found — using random actors",
                        portfolio_model=request.portfolio_model)
            job_id = "untrained"

        # ── 2. Prepare data ───────────────────────────────────────────────────
        # Use 1-year lookback for normalization context
        from datetime import timedelta
        train_start = date(as_of.year - 1, as_of.month, as_of.day)
        dataset = await self._pipeline.prepare(
            isins=request.assets,
            start=train_start,
            end=as_of,
            fit=True,
        )

        # ── 3. Fetch asset metadata ───────────────────────────────────────────
        sectors = await self._get_sectors(request.assets)

        # ── 4. Run orchestrator ───────────────────────────────────────────────
        orchestrator = PortfolioOrchestratorAgent(
            model_store_path=cfg.model_store_path,
            job_id=job_id,
            n_assets=len(request.assets),
            hidden=cfg.masac_hidden_size,
        )

        t = dataset.n_timesteps - 1   # latest available day
        panels = await orchestrator.generate_comparison(dataset, t, request, sectors)

        # ── 5. Persist results ────────────────────────────────────────────────
        for topology_key, panel in panels.items():
            pr = PortfolioResult(
                id=str(uuid.uuid4()),
                query_id=query_id,
                topology=topology_key,
                portfolio_model=request.portfolio_model.value,
                allocation_json={"portfolio": panel["portfolio"]},
                metrics_json=panel["aggregate_metrics"],
            )
            self._db.add(pr)
        await self._db.commit()

        return {
            "query_id": query_id,
            "portfolio_model": request.portfolio_model.value,
            "allocation_amount": request.allocation_amount,
            "as_of_date": str(as_of),
            **panels,
        }

    async def get_comparison(self, query_id: str) -> dict[str, Any] | None:
        result = await self._db.execute(
            select(PortfolioResult).where(PortfolioResult.query_id == query_id)
        )
        rows = result.scalars().all()
        if not rows:
            return None
        out: dict[str, Any] = {"query_id": query_id}
        for row in rows:
            out[row.topology] = {
                "topology": row.topology,
                "portfolio": row.allocation_json.get("portfolio", []),
                "aggregate_metrics": row.metrics_json,
            }
        return out

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _get_best_job(self, portfolio_model: str) -> str | None:
        """Return the job_id of the best completed training run for this model."""
        result = await self._db.execute(
            select(TrainingJob)
            .where(
                TrainingJob.portfolio_model == portfolio_model,
                TrainingJob.status == "completed",
            )
            .order_by(TrainingJob.best_sharpe.desc().nulls_last())
            .limit(1)
        )
        job = result.scalar_one_or_none()
        return job.id if job else None

    async def _get_sectors(self, isins: list[str]) -> list[str]:
        result = await self._db.execute(
            select(Asset).where(Asset.isin.in_(isins))
        )
        assets = {a.isin: a.sector for a in result.scalars().all()}
        return [assets.get(isin, "Unknown") for isin in isins]
