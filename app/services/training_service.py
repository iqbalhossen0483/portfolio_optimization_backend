"""
TrainingService — orchestrates background training jobs via Celery.
Supports single topology or all-three concurrent training.
"""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Any

import structlog
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

    async def start_training(self, request: TrainingRequest) -> str:
        """
        Create a DB record and enqueue Celery tasks.
        Returns job_id.
        """
        job_id = str(uuid.uuid4())
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
            id=job_id,
            status="queued",
            portfolio_model=request.portfolio_model.value,
            topology=request.topology.value,
            config_json=config,
            started_at=None,
        )
        self._db.add(job)
        await self._db.commit()

        # Enqueue Celery task
        from app.workers.tasks import run_training_job
        run_training_job.delay(job_id, config)

        log.info("Training job queued", job_id=job_id, topologies=topologies)
        return job_id

    async def get_status(self, job_id: str) -> dict[str, Any]:
        from sqlalchemy import select
        result = await self._db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
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

    async def stop_training(self, job_id: str) -> bool:
        """Signal the Celery worker to stop the job gracefully."""
        if self._redis:
            await self._redis.set(f"stop:{job_id}", "1", ex=3600)
        return True
