from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_training_service
from app.models.schemas import (
    TrainingRequest, TrainingJobResponse, TrainingStatusResponse, TrainingStatus
)
from app.services.training_service import TrainingService

router = APIRouter(prefix="/training", tags=["training"])


@router.post(
    "/start",
    summary="Start a MASAC training job",
    description=(
        "Queues a background Celery task to train MASAC agents for the specified "
        "portfolio model and topology. Returns job_id for status polling or WebSocket streaming."
    ),
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_training(
    request: TrainingRequest,
    service: TrainingService = Depends(get_training_service),
) -> TrainingJobResponse:
    job_id = await service.start_training(request)
    return TrainingJobResponse(
        job_id=job_id,
        status=TrainingStatus.QUEUED,
        message="Training job queued successfully",
    )


@router.get(
    "/{job_id}/status",
    summary="Get training job status and live metrics",
)
async def get_training_status(
    job_id: str,
    service: TrainingService = Depends(get_training_service),
) -> TrainingStatusResponse:
    data = await service.get_status(job_id)
    if "error" in data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=data["error"])
    return TrainingStatusResponse(**data)


@router.post(
    "/{job_id}/stop",
    summary="Request graceful stop of a running training job",
)
async def stop_training(
    job_id: str,
    service: TrainingService = Depends(get_training_service),
) -> dict:
    ok = await service.stop_training(job_id)
    return {"job_id": job_id, "stop_requested": ok}
