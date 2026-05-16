"""
TrainingService — orchestrates background training jobs via Celery.

start_training()           — legacy JSON-body path (backward compat)
start_training_from_xlsx() — new XLSX multipart path:
    Stage 1: parse XLSX → upsert assets / market_data / esg_scores
    Stage 2: cross-sectional ESG normalization → update esg_scores
    Stage 3: fit time-series normalizer → insert training_normalizer_params
    Stage 4: enqueued to Celery (DB-backed sources, frozen normalizer)
"""
from __future__ import annotations
from datetime import datetime, date
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.domain import TrainingJob
from app.models.schemas import TrainingRequest, TrainingStatus
from app.config import get_settings

log = structlog.get_logger(__name__)
cfg = get_settings()


class TrainingService:

    def __init__(self, db: AsyncSession, redis_client=None) -> None:
        self._db = db
        self._redis = redis_client

    # ── Legacy JSON-body path (backward compat) ───────────────────────────────

    async def start_training(self, request: TrainingRequest) -> int:
        """Create a DB record and enqueue Celery tasks. Returns job_id."""
        topologies = (
            ["cooperative", "competitive", "mixed"]
            if request.topology.value == "all"
            else [request.topology.value]
        )

        config = {
            "portfolio_model": request.portfolio_model.value,
            "topologies": topologies,
            "assets": request.assets,
            "train_start": str(request.train_start),
            "train_end": str(request.train_end),
            "val_start": str(request.val_start),
            "val_end": str(request.val_end),
            "hyperparams": request.hyperparams.model_dump(),
        }

        job = TrainingJob(
            status="queued",
            portfolio_model=request.portfolio_model.value,
            topology=request.topology.value,
            config_json=config,
            started_at=None,
        )
        self._db.add(job)
        await self._db.flush()   # populates job.id from DB autoincrement
        job_id: int = job.id
        await self._db.commit()

        from app.workers.tasks import run_training_job
        run_training_job.delay(job_id, config)  # type: ignore[union-attr]

        log.info("training_job_queued", job_id=job_id, topologies=topologies)
        return job_id

    # ── XLSX multipart path (4-stage pipeline) ────────────────────────────────

    async def start_training_from_xlsx(
        self,
        tmp_paths: list[str],
        portfolio_model: str,
        topology: str,
        train_start: date | None,
        train_end: date | None,
        val_start: date | None,
        val_end: date | None,
        hyperparams: dict,
    ) -> int:
        from app.data.sources.xlsx import XLSXDataSource
        from app.data.sources.database import DatabaseMarketDataSource, DatabaseESGDataSource
        from app.data.pipeline import DataPipeline

        # ── Stage 1: Parse XLSX → compute return_pct + macd_hist ─────────────
        log.info("stage1_start", files=len(tmp_paths))
        parsed = XLSXDataSource.parse_files(tmp_paths)
        isins = [a["isin"] for a in parsed.assets]  # N — dynamic from XLSX

        await self._upsert_assets(parsed.assets)
        await self._upsert_market_data(parsed.market_df)
        await self._upsert_esg_raw(parsed.esg_df)
        log.info("stage1_done", n_assets=len(isins), n_timesteps=parsed.n_timesteps)

        # ── Stage 2: Cross-sectional ESG normalization (per date, all N) ─────
        log.info("stage2_start")
        await self._compute_esg_normalization(isins)
        log.info("stage2_done")

        # ── Derive date ranges if not supplied ────────────────────────────────
        if None in (train_start, train_end, val_start, val_end):
            train_start, train_end, val_start, val_end = (
                XLSXDataSource.derive_date_split(parsed.all_dates())
            )
        assert train_start and train_end and val_start and val_end

        # ── Create job record (must exist before normalizer params FK insert) ───
        topologies = (
            ["cooperative", "competitive", "mixed"]
            if topology == "all"
            else [topology]
        )
        config: dict[str, Any] = {
            "data_source": "database",
            "portfolio_model": portfolio_model,
            "topologies": topologies,
            "assets": isins,
            "n_assets": len(isins),
            "train_start": str(train_start),
            "train_end": str(train_end),
            "val_start": str(val_start),
            "val_end": str(val_end),
            "hyperparams": hyperparams,
        }
        job_id: int = await self._create_job_record(portfolio_model, topology, config)

        # ── Stage 3: Fit time-series normalizer → persist params ─────────────
        log.info("stage3_start", job_id=job_id,
                 train_start=str(train_start), train_end=str(train_end))
        market_src = DatabaseMarketDataSource(cfg.postgres_dsn)
        esg_src    = DatabaseESGDataSource(cfg.postgres_dsn)
        pipeline   = DataPipeline(market_src, esg_src)
        train_ds   = await pipeline.prepare(isins, train_start, train_end, fit=True)
        param_records = train_ds.normalizer.to_param_records(job_id, isins)
        await self._upsert_normalizer_params(param_records)
        log.info("stage3_done", job_id=job_id, param_rows=len(param_records))

        from app.workers.tasks import run_training_job
        run_training_job.delay(job_id, config)  # type: ignore[union-attr]

        log.info("stage4_queued", job_id=job_id, n_assets=len(isins),
                 topologies=topologies)
        return job_id

    # ── Stage 1 helpers ───────────────────────────────────────────────────────

    async def _upsert_assets(self, assets: list[dict]) -> None:
        """Bulk upsert N assets — single statement, ON CONFLICT DO UPDATE."""
        if not assets:
            return
        await self._db.execute(
            text("""
                INSERT INTO assets (isin, name, sector)
                VALUES (:isin, :name, :sector)
                ON CONFLICT (isin) DO UPDATE
                    SET name = EXCLUDED.name,
                        sector = EXCLUDED.sector
            """),
            assets,
        )
        await self._db.commit()

    async def _upsert_market_data(self, df: pd.DataFrame) -> None:
        """
        Bulk upsert market_data rows (T×N rows, N is dynamic).
        Fetches asset_id via a single IN-query, then bulk-inserts.
        ON CONFLICT updates all computed columns (handles re-upload).
        """
        isins = df["isin"].unique().tolist()
        result = await self._db.execute(
            text("SELECT id, isin FROM assets WHERE isin = ANY(:isins)"),
            {"isins": isins},
        )
        isin_to_id = {row.isin: row.id for row in result}

        records = []
        for row in df.itertuples(index=False):
            asset_id = isin_to_id.get(row.isin)
            if not asset_id:
                continue
            records.append({
                "asset_id": asset_id,
                "date": row.date,
                "open": _safe_float(row.open),
                "high": _safe_float(row.high),
                "low": _safe_float(row.low),
                "close": _safe_float(row.close),
                "volume": _safe_float(row.volume),
                "rsi": _safe_float(row.rsi),
                "return_pct": _safe_float(row.return_pct),
                "macd_hist": _safe_float(row.macd_hist),
            })

        if not records:
            return

        await self._db.execute(
            text("""
                INSERT INTO market_data
                    (asset_id, date, open, high, low, close, volume,
                     rsi, return_pct, macd_hist)
                VALUES
                    (:asset_id, :date, :open, :high, :low, :close, :volume,
                     :rsi, :return_pct, :macd_hist)
                ON CONFLICT (asset_id, date) DO UPDATE SET
                    open       = EXCLUDED.open,
                    high       = EXCLUDED.high,
                    low        = EXCLUDED.low,
                    close      = EXCLUDED.close,
                    volume     = EXCLUDED.volume,
                    rsi        = EXCLUDED.rsi,
                    return_pct = EXCLUDED.return_pct,
                    macd_hist  = EXCLUDED.macd_hist
            """),
            records,
        )
        await self._db.commit()
        log.info("market_data_upserted", rows=len(records))

    async def _upsert_esg_raw(self, df: pd.DataFrame) -> None:
        """Bulk upsert raw bloomberg_score + lesg_score (T×N rows)."""
        isins = df["isin"].unique().tolist()
        result = await self._db.execute(
            text("SELECT id, isin FROM assets WHERE isin = ANY(:isins)"),
            {"isins": isins},
        )
        isin_to_id = {row.isin: row.id for row in result}

        records = []
        for row in df.itertuples(index=False):
            asset_id = isin_to_id.get(row.isin)
            if not asset_id:
                continue
            records.append({
                "asset_id": asset_id,
                "date": row.date,
                "bloomberg_score": _safe_float(row.bloomberg_score),
                "lesg_score": _safe_float(row.lesg_score),
            })

        if not records:
            return

        await self._db.execute(
            text("""
                INSERT INTO esg_scores (asset_id, date, bloomberg_score, lesg_score)
                VALUES (:asset_id, :date, :bloomberg_score, :lesg_score)
                ON CONFLICT (asset_id, date) DO UPDATE SET
                    bloomberg_score = EXCLUDED.bloomberg_score,
                    lesg_score      = EXCLUDED.lesg_score
            """),
            records,
        )
        await self._db.commit()
        log.info("esg_raw_upserted", rows=len(records))

    # ── Stage 2 helper ────────────────────────────────────────────────────────

    async def _compute_esg_normalization(self, isins: list[str]) -> None:
        """
        Cross-sectional ESG normalization per date across all N ISINs.
        Vectorized pandas groupby — single DB read, one groupby pass, single bulk update.

        For each date t:
            esg_b_norm(i,t) = (ESG_B(i,t) - min_i ESG_B(t)) / (max_i ESG_B(t) - min_i ESG_B(t))
            esg_l_norm(i,t) = same for LESG
            delta_esg(i,t)  = |esg_b_norm - esg_l_norm|
            mu_esg(i,t)     = (esg_b_norm + esg_l_norm) / 2
        """
        result = await self._db.execute(
            text("""
                SELECT e.id, a.isin, e.date, e.bloomberg_score, e.lesg_score
                FROM esg_scores e
                JOIN assets a ON a.id = e.asset_id
                WHERE a.isin = ANY(:isins)
                ORDER BY e.date, a.isin
            """),
            {"isins": isins},
        )
        df = pd.DataFrame(result.fetchall(), columns=["id", "isin", "date",
                                                        "bloomberg_score", "lesg_score"])
        if df.empty:
            return

        # Per-date cross-sectional min-max (vectorized groupby)
        def _cs_norm(series: pd.Series) -> pd.Series:
            mn = series.min()
            mx = series.max()
            denom = mx - mn
            if denom == 0:
                return pd.Series(0.5, index=series.index)
            return (series - mn) / denom

        df["esg_b_norm"] = df.groupby("date")["bloomberg_score"].transform(_cs_norm)
        df["esg_l_norm"] = df.groupby("date")["lesg_score"].transform(_cs_norm)
        df["delta_esg"]  = (df["esg_b_norm"] - df["esg_l_norm"]).abs()
        df["mu_esg"]     = (df["esg_b_norm"] + df["esg_l_norm"]) / 2.0

        update_records = df[["id", "esg_b_norm", "esg_l_norm",
                              "delta_esg", "mu_esg"]].to_dict(orient="records")

        await self._db.execute(  # type: ignore[call-overload]
            text("""
                UPDATE esg_scores SET
                    esg_b_norm = :esg_b_norm,
                    esg_l_norm = :esg_l_norm,
                    delta_esg  = :delta_esg,
                    mu_esg     = :mu_esg
                WHERE id = :id
            """),
            update_records,  # type: ignore[arg-type]
        )
        await self._db.commit()
        log.info("esg_normalization_done", rows=len(update_records))

    # ── Stage 3 helper ────────────────────────────────────────────────────────

    async def _upsert_normalizer_params(self, records: list[dict]) -> None:
        """Bulk insert 8×N normalizer param rows (N is dynamic)."""
        if not records:
            return
        await self._db.execute(
            text("""
                INSERT INTO training_normalizer_params
                    (job_id, isin, feature_name, min_val, max_val)
                VALUES (:job_id, :isin, :feature_name, :min_val, :max_val)
                ON CONFLICT (job_id, isin, feature_name) DO UPDATE SET
                    min_val = EXCLUDED.min_val,
                    max_val = EXCLUDED.max_val
            """),
            records,
        )
        await self._db.commit()
        log.info("normalizer_params_upserted", rows=len(records))

    # ── Job record helper ─────────────────────────────────────────────────────

    async def _create_job_record(
        self,
        portfolio_model: str,
        topology: str,
        config: dict,
    ) -> int:
        job = TrainingJob(
            status="queued",
            portfolio_model=portfolio_model,
            topology=topology,
            config_json=config,
            started_at=None,
        )
        self._db.add(job)
        await self._db.flush()   # populates job.id from DB autoincrement
        job_id: int = job.id
        await self._db.commit()
        return job_id

    # ── Status / stop ─────────────────────────────────────────────────────────

    async def get_status(self, job_id: int) -> dict[str, Any]:
        from sqlalchemy import select
        result = await self._db.execute(
            select(TrainingJob).where(TrainingJob.id == job_id)
        )
        job = result.scalar_one_or_none()
        if not job:
            return {"error": "Job not found"}

        elapsed = None
        if job.started_at:
            end = job.completed_at or datetime.utcnow()
            elapsed = (end - job.started_at).total_seconds()

        return {
            "job_id": job.id,
            "status": job.status,
            "step": job.current_step,
            "max_steps": cfg.masac_max_steps,
            "progress_pct": round(job.current_step / cfg.masac_max_steps * 100, 1),
            "entropy_rolling_std": None,
            "best_sharpe": job.best_sharpe,
            "best_mu_esg": job.best_mu_esg,
            "current_rewards": {},
            "elapsed_seconds": elapsed,
            "error_message": job.error_message,
        }

    async def stop_training(self, job_id: int) -> bool:
        """Signal the Celery worker to stop the job gracefully."""
        if self._redis:
            await self._redis.set(f"stop:{job_id}", "1", ex=3600)
        return True


def _safe_float(v: Any) -> float | None:
    """Convert to float; returns None for NaN/None (stored as SQL NULL)."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (f != f) else f   # NaN check without math import
    except (TypeError, ValueError):
        return None
