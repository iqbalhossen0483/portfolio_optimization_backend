"""
PortfolioService — business logic for portfolio generation.
Loads latest trained model, runs DataPipeline with frozen normalizer, calls OrchestratorAgent.
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
from app.data.sources.database import DatabaseMarketDataSource, DatabaseESGDataSource
from app.data.preprocessing.normalizer import DataNormalizer
from app.config import get_settings

log = structlog.get_logger(__name__)
cfg = get_settings()


class PortfolioService:

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def generate(self, request: PortfolioGenerateRequest) -> dict[str, Any]:
        """
        Full portfolio generation flow:
        1. Resolve best trained job for requested portfolio_model
        2. Load frozen normalizer params from training_normalizer_params (DB)
        3. Fetch pre-computed features from market_data + esg_scores (DB)
        4. Apply frozen normalizer → build state vectors (no look-ahead)
        5. Run orchestrator for all three topologies
        6. Persist results and return comparison panels
        """
        query_id = str(uuid.uuid4())
        as_of = request.as_of_date or date.today()

        # ── 1. Resolve best completed training job ────────────────────────────
        job = await self._get_best_job(request.portfolio_model.value)
        if job is None:
            raise ValueError(
                f"No completed training job found for portfolio_model="
                f"{request.portfolio_model.value}. Train a model first."
            )
        job_id: int = job.id
        config = job.config_json or {}

        # ── 2. Load frozen normalizer from DB (training_normalizer_params) ────
        # This reconstructs the exact min/max fitted on the training window.
        # Survives server restarts — params are persisted in DB, not in memory.
        frozen_normalizer = await DataNormalizer.load_from_db(job_id, cfg.postgres_dsn)

        # ── 3. Determine inference date window ────────────────────────────────
        # Use the validation window from the original job config so the
        # inference data is outside the training window (no look-ahead).
        val_start_str = config.get("val_start")
        val_end_str   = config.get("val_end")
        if val_start_str and val_end_str:
            infer_start = date.fromisoformat(val_start_str)
            infer_end   = date.fromisoformat(val_end_str)
        else:
            # Fallback: use 1-year lookback ending today
            from datetime import timedelta
            infer_end   = as_of
            infer_start = date(as_of.year - 1, as_of.month, as_of.day)

        # ── 4. Fetch pre-computed features from DB + apply frozen normalizer ──
        market_src = DatabaseMarketDataSource(cfg.postgres_dsn)
        esg_src    = DatabaseESGDataSource(cfg.postgres_dsn)
        pipeline   = DataPipeline(market_src, esg_src)

        dataset = await pipeline.prepare(
            isins=request.assets,
            start=infer_start,
            end=infer_end,
            fit=False,               # never refit — use frozen training-window params
            normalizer=frozen_normalizer,
        )

        # ── 5. Fetch asset sector metadata ────────────────────────────────────
        sectors = await self._get_sectors(request.assets)

        # ── 6. Run orchestrator (loads saved actor weights from model store) ──
        orchestrator = PortfolioOrchestratorAgent(
            model_store_path=cfg.model_store_path,
            job_id=job_id,
            n_assets=len(request.assets),
            hidden=cfg.masac_hidden_size,
        )

        t = dataset.n_timesteps - 1   # latest available timestep
        panels = await orchestrator.generate_comparison(dataset, t, request, sectors)

        # ── 7. Persist results ────────────────────────────────────────────────
        for topology_key, panel in panels.items():
            pr = PortfolioResult(
                query_id=query_id,
                job_id=job_id,
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

    async def _get_best_job(self, portfolio_model: str) -> TrainingJob | None:
        """Return the best completed TrainingJob for this portfolio_model (highest Sharpe)."""
        result = await self._db.execute(
            select(TrainingJob)
            .where(
                TrainingJob.portfolio_model == portfolio_model,
                TrainingJob.status == "completed",
            )
            .order_by(TrainingJob.best_sharpe.desc().nulls_last())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def _get_sectors(self, isins: list[str]) -> list[str]:
        result = await self._db.execute(
            select(Asset).where(Asset.isin.in_(isins))
        )
        assets = {a.isin: a.sector for a in result.scalars().all()}
        return [assets.get(isin, "Unknown") for isin in isins]
