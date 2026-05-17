"""
Celery tasks for background training.
Each task manages one (job_id, topology) training run.

Data source: DatabaseMarketDataSource + DatabaseESGDataSource (all pre-computed, no API calls).
Normalizer: loaded from training_normalizer_params table (frozen, no refit).
"""
from __future__ import annotations
import asyncio
import os
import sys
from datetime import datetime, date, timezone
import structlog
from celery import Celery

from app.config import get_settings


_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)



log = structlog.get_logger(__name__)
cfg = get_settings()

celery_app = Celery(
    "madrl_portfolio",
    broker=cfg.celery_broker_url,
    backend=cfg.celery_result_backend,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    worker_pool="solo",       # Windows: prefork uses fork() which is unsupported
)


@celery_app.task(name="run_training_job", max_retries=0)
def run_training_job(job_id: int, config: dict) -> dict:
    """
    Main training entry point.  Runs each topology sequentially.
    """
    return asyncio.run(_async_run_training(job_id, config))


async def _async_run_training(job_id: int, config: dict) -> dict:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from app.models.domain import TrainingJob
    from app.data.pipeline import DataPipeline
    from app.rl.trainer import TrainingOrchestrator
    import redis.asyncio as aioredis

    engine = create_async_engine(cfg.postgres_dsn)
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    redis_client = aioredis.from_url(cfg.redis_url)

    hp = config["hyperparams"]
    train_start = date.fromisoformat(config["train_start"])
    train_end   = date.fromisoformat(config["train_end"])
    isins       = config["assets"]

    # ── Data sources (DB-backed, pre-computed features) ──────────────────────
    from app.data.sources.database import DatabaseMarketDataSource, DatabaseESGDataSource
    from app.data.preprocessing.normalizer import DataNormalizer

    market_src = DatabaseMarketDataSource(cfg.postgres_dsn)
    esg_src    = DatabaseESGDataSource(cfg.postgres_dsn)
    pipeline   = DataPipeline(market_src, esg_src)

    frozen_normalizer = await DataNormalizer.load_from_db(job_id, cfg.postgres_dsn)
    log.info("stage4_db_sources", job_id=job_id, n_assets=len(isins))

    train_ds = await pipeline.prepare(
        isins, train_start, train_end,
        fit=False, normalizer=frozen_normalizer,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    results = []
    async with SessionLocal() as db:
        from sqlalchemy import update
        await db.execute(
            update(TrainingJob)
            .where(TrainingJob.id == job_id)
            .values(status="running", started_at=datetime.now(timezone.utc))
        )
        await db.commit()

        try:
            for topology in config["topologies"]:
                stop = await redis_client.get(f"stop:{job_id}")
                if stop:
                    log.info("stop_requested", job_id=job_id, topology=topology)
                    break

                trainer = TrainingOrchestrator(
                    job_id=job_id,
                    dataset=train_ds,
                    portfolio_model=config["portfolio_model"],
                    topology=topology,
                    hyperparams=hp,
                    model_store_path=cfg.model_store_path,
                    redis_client=redis_client,
                    db=db,
                )
                result = await trainer.run()
                results.append(result)

                await db.execute(
                    update(TrainingJob)
                    .where(TrainingJob.id == job_id)
                    .values(
                        current_step=result["steps_completed"],
                        best_sharpe=result.get("best_sharpe"),
                        best_mu_esg=result.get("best_mu_esg"),
                    )
                )
                await db.commit()

            await db.execute(
                update(TrainingJob)
                .where(TrainingJob.id == job_id)
                .values(status="completed", completed_at=datetime.now(timezone.utc))
            )
            await db.commit()

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            log.error("training_failed", job_id=job_id, error=error_msg)
            await db.execute(
                update(TrainingJob)
                .where(TrainingJob.id == job_id)
                .values(status="failed", error_message=error_msg[:1024],
                        completed_at=datetime.now(timezone.utc))
            )
            await db.commit()
            raise   # re-raise so Celery marks the task as FAILURE too

    await redis_client.aclose()
    return {"job_id": job_id, "results": results}
