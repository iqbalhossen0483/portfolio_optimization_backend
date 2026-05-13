"""
Celery tasks for background training.
Each task manages one (job_id, topology) training run.
"""
from __future__ import annotations
import asyncio
import os
import sys
from datetime import datetime, date

# Ensure project root is importable in worker subprocesses that re-import this
# module with a clean sys.path (common on Windows and with prefork pools).
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

import structlog
from celery import Celery
from celery.signals import task_failure, task_success

from app.config import get_settings

log = structlog.get_logger(__name__)
cfg = get_settings()

celery_app = Celery(
    "madrl_portfolio",
    broker=cfg.celery_broker_url,
    backend=cfg.celery_result_backend,
)
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.accept_content = ["json"]
celery_app.conf.timezone = "UTC"


@celery_app.task(bind=True, name="run_training_job", max_retries=0)
def run_training_job(self, job_id: str, config: dict) -> dict:
    """
    Main training entry point.  Runs each topology sequentially in the worker
    (parallel topology runs can be achieved by spawning separate tasks).
    """
    return asyncio.get_event_loop().run_until_complete(
        _async_run_training(job_id, config)
    )


async def _async_run_training(job_id: str, config: dict) -> dict:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from app.models.domain import TrainingJob
    from app.data.pipeline import DataPipeline
    from app.data.sources.market import MarketDataSource
    from app.data.sources.esg import ESGDataSource
    from app.rl.trainer import TrainingOrchestrator
    import redis.asyncio as aioredis

    engine = create_async_engine(cfg.postgres_dsn)
    SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    redis_client = aioredis.from_url(cfg.redis_url)
    market_src = MarketDataSource(redis_client, ttl=cfg.redis_ttl_market)
    esg_src    = ESGDataSource(
        bloomberg_api_key=cfg.bloomberg_api_key,
        lesg_api_key=cfg.lesg_api_key,
        redis_client=redis_client,
        use_stub=(not cfg.bloomberg_api_key),
    )
    pipeline = DataPipeline(market_src, esg_src)

    hp = config["hyperparams"]
    train_start = date.fromisoformat(config["train_start"])
    train_end   = date.fromisoformat(config["train_end"])
    val_start   = date.fromisoformat(config["val_start"])
    val_end     = date.fromisoformat(config["val_end"])

    # Prepare datasets
    train_ds = await pipeline.prepare(config["assets"], train_start, train_end, fit=True)
    val_ds   = await pipeline.prepare(
        config["assets"], val_start, val_end,
        fit=False, normalizer=train_ds.normalizer
    )

    results = []
    async with SessionLocal() as db:
        from sqlalchemy import select, update
        await db.execute(
            update(TrainingJob)
            .where(TrainingJob.id == job_id)
            .values(status="running", started_at=datetime.utcnow())
        )
        await db.commit()

        for topology in config["topologies"]:
            # Check stop signal
            stop = await redis_client.get(f"stop:{job_id}")
            if stop:
                log.info("Stop requested", job_id=job_id, topology=topology)
                break

            trainer = TrainingOrchestrator(
                job_id=job_id,
                dataset=train_ds,
                portfolio_model=config["portfolio_model"],
                topology=topology,
                hyperparams=hp,
                model_store_path=cfg.model_store_path,
                redis_client=redis_client,
            )
            result = await trainer.run()
            results.append(result)

            # Update job record with best metrics
            best_sharpe = result.get("best_sharpe")
            best_mu_esg = result.get("best_mu_esg")
            await db.execute(
                update(TrainingJob)
                .where(TrainingJob.id == job_id)
                .values(
                    current_step=result["steps_completed"],
                    best_sharpe=best_sharpe,
                    best_mu_esg=best_mu_esg,
                )
            )
            await db.commit()

        # Mark completed
        await db.execute(
            update(TrainingJob)
            .where(TrainingJob.id == job_id)
            .values(status="completed", completed_at=datetime.utcnow())
        )
        await db.commit()

    await redis_client.aclose()
    return {"job_id": job_id, "results": results}
