from __future__ import annotations
from datetime import date
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, model_validator


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
    alpha_1: float = Field(default=0.5,  ge=0.0, le=1.0, description="Bloomberg ESG weight (Agents A, C)")
    alpha_2: float = Field(default=0.5,  ge=0.0, le=1.0, description="LESG ESG weight (Agents A, C)")
    alpha_3: float = Field(default=0.01, ge=0.0, le=0.1, description="Financial agent ESG bias (≈0)")
    beta:    float = Field(default=0.3,  ge=0.0, le=1.0, description="Shared ambiguity penalty (Portfolio C)")
    lam:     float = Field(default=0.4,  ge=0.0, le=1.0, description="Signed disagreement sensitivity (Portfolio B)")


# ── Portfolio generation ──────────────────────────────────────────────────────

class PortfolioGenerateRequest(BaseModel):
    assets: list[str] = Field(..., min_length=2, description="List of ISINs")
    portfolio_model: PortfolioModel = PortfolioModel.C
    allocation_amount: float = Field(..., gt=0, description="Total capital in base currency")
    hyperparams: HyperParams = Field(default_factory=HyperParams)
    as_of_date: date | None = None


class AssetAllocation(BaseModel):
    isin: str
    sector: str
    weight: float = Field(..., ge=0.0, le=1.0)
    allocation: float
    return_ann: float
    risk_ann: float
    sharpe: float
    mu_esg: float
    delta_esg: float


class AggregateMetrics(BaseModel):
    portfolio_return: float
    portfolio_risk: float
    portfolio_sharpe: float
    portfolio_mu_esg: float
    portfolio_delta_esg: float


class TopologyPanel(BaseModel):
    topology: Topology
    portfolio: list[AssetAllocation]
    aggregate_metrics: AggregateMetrics
    strategic_summary: str


class PortfolioGenerateResponse(BaseModel):
    query_id: str
    portfolio_model: PortfolioModel
    allocation_amount: float
    as_of_date: date | None
    cooperative: TopologyPanel
    competitive: TopologyPanel
    mixed: TopologyPanel


class PortfolioGetResponse(BaseModel):
    id: str
    query_id: str
    topology: Topology
    portfolio_model: PortfolioModel
    allocations: list[AssetAllocation]
    metrics: AggregateMetrics


# ── Training ──────────────────────────────────────────────────────────────────

class TrainingRequest(BaseModel):
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
    job_id: str
    status: TrainingStatus
    message: str = ""


class TrainingStatusResponse(BaseModel):
    job_id: str
    status: TrainingStatus
    step: int
    max_steps: int
    progress_pct: float
    entropy_rolling_std: float | None
    best_sharpe: float | None
    best_mu_esg: float | None
    current_rewards: dict[str, float] = Field(default_factory=dict)
    elapsed_seconds: float | None
    error_message: str | None = None


# ── Data ─────────────────────────────────────────────────────────────────────

class AssetInfo(BaseModel):
    isin: str
    name: str
    sector: str


class AssetsResponse(BaseModel):
    assets: list[AssetInfo]
    total: int


class DataIngestionRequest(BaseModel):
    assets: list[str]
    start_date: date
    end_date: date
    sources: list[str] = Field(default_factory=lambda: ["market", "bloomberg", "lesg"])


class DataIngestionResponse(BaseModel):
    job_id: str
    assets_queued: int
    status: str


# ── WebSocket messages ────────────────────────────────────────────────────────

class WsTrainingStep(BaseModel):
    type: str = "step"
    step: int
    entropy: float
    entropy_rolling_std: float
    reward_bloomberg: float
    reward_lesg: float
    reward_financial: float
    loss_actor: float
    loss_critic: float
    alpha_t: float


class WsTrainingConverged(BaseModel):
    type: str = "converged"
    step: int
    final_sharpe: float
    mu_esg: float
    message: str


class WsError(BaseModel):
    type: str = "error"
    message: str
    details: Any = None
