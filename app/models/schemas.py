from __future__ import annotations
from datetime import date
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, model_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class PortfolioModel(str, Enum):
    A = "A"   # ESG consensus only
    B = "B"   # signed disagreement
    C = "C"   # consensus + uncertainty penalty


class Topology(str, Enum):
    COOPERATIVE = "cooperative"
    COMPETITIVE = "competitive"
    MIXED = "mixed"
    ALL = "all"


class TrainingStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


# ── Hyperparameters ───────────────────────────────────────────────────────────

class HyperParams(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "alpha_1": 0.5,
            "alpha_2": 0.5,
            "alpha_3": 0.01,
            "beta": 0.3,
            "lam": 0.4,
        }
    })

    alpha_1: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Bloomberg ESG weight in the reward function (Portfolios A and C).",
    )
    alpha_2: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="LESG ESG weight in the reward function (Portfolios A and C).",
    )
    alpha_3: float = Field(
        default=0.01, ge=0.0, le=0.1,
        description="Financial agent ESG bias — kept near 0 by design so the financial agent focuses on returns.",
    )
    beta: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Shared ambiguity penalty strength β · ΔESGᵢₜ applied in the Cooperative topology (Portfolio C only).",
    )
    lam: float = Field(
        default=0.4, ge=0.0, le=1.0,
        description="Signed disagreement sensitivity λ (Portfolio B only).",
    )


# ── Training ──────────────────────────────────────────────────────────────────

class TrainingRequest(BaseModel):
    """Legacy JSON-body training request (non-XLSX path, kept for backward compat)."""
    portfolio_model: PortfolioModel
    topology: Topology = Topology.ALL
    assets: list[str] = Field(..., min_length=2)
    train_start: date
    train_end: date
    val_start: date
    val_end: date
    hyperparams: HyperParams = Field(default_factory=HyperParams)

    @model_validator(mode="after")
    def validate_date_order(self) -> TrainingRequest:
        if self.train_end <= self.train_start:
            raise ValueError("train_end must be after train_start")
        if self.val_start <= self.train_end:
            raise ValueError("val_start must be after train_end (no leakage)")
        if self.val_end <= self.val_start:
            raise ValueError("val_end must be after val_start")
        return self


class TrainingJobResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "status": "queued",
            "message": "Stages 1-3 complete. MASAC training queued.",
        }
    })

    job_id: int = Field(description="ID of the training job — use for status polling and WebSocket streaming")
    status: TrainingStatus
    message: str = Field(default="", description="Human-readable status message")


class TrainingStatusResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "status": "running",
            "step": 45000,
            "max_steps": 500000,
            "progress_pct": 9.0,
            "entropy_rolling_std": 0.082,
            "best_sharpe": 1.31,
            "best_mu_esg": 0.68,
            "current_rewards": {
                "bloomberg": 0.018,
                "lesg": 0.014,
                "financial": 0.021,
            },
            "elapsed_seconds": 183.4,
            "error_message": None,
        }
    })

    job_id: int
    status: TrainingStatus
    step: int = Field(description="Current training step")
    max_steps: int = Field(description="Maximum training steps configured (default 500 000)")
    progress_pct: float = Field(description="Completion percentage (step / max_steps × 100)")
    entropy_rolling_std: float | None = Field(
        description="100-step rolling std of mean policy entropy. Training stops when < 0.01."
    )
    best_sharpe: float | None = Field(description="Best Sharpe ratio seen so far across all steps")
    best_mu_esg: float | None = Field(description="Best μESG value seen so far")
    current_rewards: dict[str, float] = Field(
        default_factory=dict,
        description="Per-agent rewards at the latest step: bloomberg, lesg, financial",
    )
    elapsed_seconds: float | None = Field(description="Wall-clock seconds since training started")
    error_message: str | None = Field(default=None, description="Error detail if status is 'failed'")


# ── Data ─────────────────────────────────────────────────────────────────────

class AssetInfo(BaseModel):
    isin: str = Field(description="ISIN code")
    name: str = Field(description="Company name")
    sector: str = Field(description="Sector classification")


class AssetsResponse(BaseModel):
    assets: list[AssetInfo]
    total: int = Field(description="Total number of assets returned")


class DataIngestionRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "assets": ["GB0002875804", "US0378331005"],
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "sources": ["market", "bloomberg", "lesg"],
        }
    })

    assets: list[str] = Field(description="List of ISINs to ingest")
    start_date: date
    end_date: date
    sources: list[str] = Field(
        default_factory=lambda: ["market", "bloomberg", "lesg"],
        description="Data sources to fetch: market (OHLCV), bloomberg (ESG 0-100), lesg (ESG 0-10)",
    )


class DataIngestionResponse(BaseModel):
    job_id: str  # legacy data ingestion — keeps str for backward compat
    assets_queued: int
    status: str


# ── WebSocket message schemas (documented for reference) ─────────────────────

class WsTrainingStep(BaseModel):
    """Emitted every 500 training steps via WS /ws/training/{job_id}."""
    type: str = Field(default="step")
    step: int
    entropy: float = Field(description="Mean policy entropy across all agents at this step")
    entropy_rolling_std: float = Field(description="100-step rolling std — convergence signal")
    reward_bloomberg: float
    reward_lesg: float
    reward_financial: float
    loss_actor: float
    loss_critic: float
    alpha_t: float = Field(description="Current temperature parameter αₜ (entropy regularisation)")


class WsTrainingConverged(BaseModel):
    """Emitted once when entropy converges or max_steps is reached."""
    type: str = Field(default="converged")
    step: int
    final_sharpe: float
    mu_esg: float
    message: str


class WsError(BaseModel):
    type: str = Field(default="error")
    message: str
    details: Any = None
