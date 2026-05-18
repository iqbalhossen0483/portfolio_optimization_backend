"""
Training API routes.

POST /training/start            — multipart: upload .xlsx files + form fields
GET  /training/{job_id}/status  — poll training progress and live metrics
POST /training/{job_id}/stop    — request graceful stop
"""
from __future__ import annotations
import json
import os
import shutil
import tempfile
from datetime import date

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.deps import get_training_service, require_admin
from app.models.schemas import (
    HyperParams, TrainingJobResponse, TrainingStatusResponse, TrainingStatus,
    PortfolioModel, Topology,
)
from app.services.training_service import TrainingService

router = APIRouter(prefix="/training", tags=["training"])

_VALID_PORTFOLIO_MODELS = {m.value for m in PortfolioModel}
_VALID_TOPOLOGIES = {t.value for t in Topology}


@router.post(
    "/start",
    summary="Start a MASAC training job from uploaded XLSX files",
    description=(
        "**Accepts:** one or more `.xlsx` files (sheet name: `Stock_ESG_Dataset`) "
        "plus training configuration as multipart form fields.\n\n"
        "**Stages 1–3** run synchronously before the response is returned:\n"
        "1. Parse XLSX → compute `return_pct` (pct_change of Close) and `macd_hist` "
        "(MACD histogram from Close using `macd_fast/slow/signal` from config) "
        "→ bulk-upsert to `assets`, `market_data`, `esg_scores`\n"
        "2. Cross-sectional ESG normalisation per date across all N ISINs: "
        "`esg_b_norm`, `esg_l_norm`, `delta_esg = |b-l|`, `mu_esg = (b+l)/2` "
        "→ update `esg_scores`\n"
        "3. Fit time-series min/max normaliser on the training window "
        "→ insert 8 × N rows into `training_normalizer_params`\n\n"
        "**Stage 4** (MASAC training) is enqueued to Celery and runs in the background. "
        "Stream live metrics via `WS /ws/training/{job_id}` or poll "
        "`GET /training/{job_id}/status`.\n\n"
        "**Date auto-split:** if `train_start / train_end / val_start / val_end` are omitted, "
        "dates are derived automatically at an 80/20 split from the XLSX contents.\n\n"
        "**Multi-file upload:** multiple `.xlsx` files covering different ISINs or date ranges "
        "are merged and deduplicated on `(ISIN, Date)` before processing."
    ),
    response_model=TrainingJobResponse,
    response_description="Job accepted — job_id for polling or WebSocket streaming",
    responses={
        422: {
            "description": (
                "Validation error. Common causes:\n"
                "- `portfolio_model` not in {A, B, C}\n"
                "- `topology` not in {cooperative, competitive, mixed, all}\n"
                "- Non-`.xlsx` file uploaded\n"
                "- Invalid date format (expected YYYY-MM-DD)\n"
                "- Malformed `hyperparams_json`"
            )
        },
    },
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_training(
    _: object = Depends(require_admin),
    files: list[UploadFile] = File(
        ...,
        description=(
            "One or more `.xlsx` files with sheet `Stock_ESG_Dataset`. "
            "Columns: Date, ISIN, Company name, Sector, Open, High, Low, Close, "
            "Volume (accepts `10.5M` / `2.3K` / `1B` / `1T`), RSI, "
            "Bloom. ESG (0-100), LESG ESG (0-10)."
        ),
    ),
    portfolio_model: str = Form(
        ...,
        description="Portfolio reward model. One of: `A` (consensus), `B` (disagreement), `C` (full — recommended).",
    ),
    topology: str = Form(
        default="all",
        description="Game topology to train. One of: `cooperative`, `competitive`, `mixed`, `all` (trains all three).",
    ),
    train_start: str | None = Form(
        default=None,
        description="Training window start date (YYYY-MM-DD). Auto-derived at 80% split if omitted.",
    ),
    train_end: str | None = Form(
        default=None,
        description="Training window end date (YYYY-MM-DD).",
    ),
    val_start: str | None = Form(
        default=None,
        description="Validation window start date (YYYY-MM-DD). Must be after `train_end`.",
    ),
    val_end: str | None = Form(
        default=None,
        description="Validation window end date (YYYY-MM-DD).",
    ),
    hyperparams_json: str = Form(
        default="{}",
        description=(
            "JSON string of hyperparameter overrides. "
            'Example: `{"alpha_1": 0.6, "beta": 0.4}`. '
            "Unspecified fields use defaults: alpha_1=0.5, alpha_2=0.5, alpha_3=0.01, beta=0.3, lam=0.4."
        ),
    ),
    service: TrainingService = Depends(get_training_service),
) -> TrainingJobResponse:

    if portfolio_model not in _VALID_PORTFOLIO_MODELS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"portfolio_model must be one of {sorted(_VALID_PORTFOLIO_MODELS)}",
        )
    if topology not in _VALID_TOPOLOGIES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"topology must be one of {sorted(_VALID_TOPOLOGIES)}",
        )

    try:
        hp_dict = json.loads(hyperparams_json)
        hyperparams = HyperParams(**hp_dict).model_dump()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid hyperparams_json: {exc}",
        )

    def _parse_date(s: str | None) -> date | None:
        if not s:
            return None
        try:
            return date.fromisoformat(s)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid date '{s}': expected YYYY-MM-DD — {exc}",
            )

    t_start = _parse_date(train_start)
    t_end   = _parse_date(train_end)
    v_start = _parse_date(val_start)
    v_end   = _parse_date(val_end)

    for f in files:
        if not (f.filename or "").lower().endswith(".xlsx"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Only .xlsx files are accepted, got: '{f.filename}'",
            )

    tmp_dir = tempfile.mkdtemp(prefix="madrl_xlsx_")
    tmp_paths: list[str] = []
    try:
        for upload in files:
            dest = os.path.join(tmp_dir, upload.filename or f"data_{len(tmp_paths)}.xlsx")
            with open(dest, "wb") as out:
                shutil.copyfileobj(upload.file, out)
            tmp_paths.append(dest)

        job_id = await service.start_training_from_xlsx(
            tmp_paths=tmp_paths,
            portfolio_model=portfolio_model,
            topology=topology,
            train_start=t_start,
            train_end=t_end,
            val_start=v_start,
            val_end=v_end,
            hyperparams=hyperparams,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return TrainingJobResponse(
        job_id=job_id,
        status=TrainingStatus.QUEUED,
        message="Stages 1-3 complete. MASAC training queued.",
    )


@router.get(
    "/{job_id}/status",
    summary="Get training job status and live metrics",
    description=(
        "Returns the current status and best metrics for a training job.\n\n"
        "**Status values:**\n"
        "- `queued` — waiting in Celery queue\n"
        "- `running` — actively training\n"
        "- `completed` — finished (converged or max_steps reached)\n"
        "- `failed` — error during training (see `error_message`)\n"
        "- `stopped` — gracefully stopped via `POST /training/{job_id}/stop`\n\n"
        "For real-time step-level metrics (entropy, rewards, losses) use the WebSocket "
        "endpoint `WS /ws/training/{job_id}` instead."
    ),
    response_model=TrainingStatusResponse,
    response_description="Current job status with progress and best metrics",
    responses={
        404: {"description": "No training job found for the given job_id"},
    },
)
async def get_training_status(
    job_id: int,
    _: object = Depends(require_admin),
    service: TrainingService = Depends(get_training_service),
) -> TrainingStatusResponse:
    data = await service.get_status(job_id)
    if "error" in data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=data["error"])
    return TrainingStatusResponse(**data)


@router.post(
    "/{job_id}/stop",
    summary="Request graceful stop of a running training job",
    description=(
        "Sets a stop flag in Redis (`stop:{job_id}`). The Celery worker checks this flag "
        "between topologies and exits cleanly after finishing the current topology. "
        "The job status changes to `stopped` once the worker acknowledges the signal.\n\n"
        "This does **not** kill the process immediately — in-progress topology training "
        "completes before the worker exits."
    ),
    response_description="Confirms the stop signal was sent",
    responses={
        404: {"description": "No training job found for the given job_id"},
    },
)
async def stop_training(
    job_id: int,
    _: object = Depends(require_admin),
    service: TrainingService = Depends(get_training_service),
) -> dict:
    ok = await service.stop_training(job_id)
    return {"job_id": job_id, "stop_requested": ok}
