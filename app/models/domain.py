from __future__ import annotations
import uuid
from datetime import datetime
from sqlalchemy import (
    String, Float, Integer, DateTime, JSON, ForeignKey, UniqueConstraint, func
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def new_uuid() -> str:
    return str(uuid.uuid4())


class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    isin: Mapped[str] = mapped_column(String(12), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    sector: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    market_data: Mapped[list[MarketData]] = relationship(back_populates="asset")
    esg_scores: Mapped[list[ESGScore]] = relationship(back_populates="asset")


class MarketData(Base):
    __tablename__ = "market_data"
    __table_args__ = (UniqueConstraint("asset_id", "date"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id"), index=True)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)

    asset: Mapped[Asset] = relationship(back_populates="market_data")


class ESGScore(Base):
    __tablename__ = "esg_scores"
    __table_args__ = (UniqueConstraint("asset_id", "date"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id"), index=True)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    bloomberg_score: Mapped[float | None] = mapped_column(Float, nullable=True)  # 0-100
    lesg_score: Mapped[float | None] = mapped_column(Float, nullable=True)       # 0-10

    asset: Mapped[Asset] = relationship(back_populates="esg_scores")


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    portfolio_model: Mapped[str] = mapped_column(String(8), nullable=False)
    topology: Mapped[str] = mapped_column(String(32), nullable=False)
    config_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    current_step: Mapped[int] = mapped_column(Integer, default=0)
    best_sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_mu_esg: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    checkpoints: Mapped[list[ModelCheckpoint]] = relationship(back_populates="job")


class ModelCheckpoint(Base):
    __tablename__ = "model_checkpoints"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("training_jobs.id"), index=True)
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    path: Mapped[str] = mapped_column(String(512), nullable=False)
    sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    mu_esg: Mapped[float | None] = mapped_column(Float, nullable=True)
    entropy: Mapped[float | None] = mapped_column(Float, nullable=True)
    saved_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    job: Mapped[TrainingJob] = relationship(back_populates="checkpoints")


class PortfolioResult(Base):
    __tablename__ = "portfolios"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    query_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("training_jobs.id"), nullable=True)
    topology: Mapped[str] = mapped_column(String(32), nullable=False)
    portfolio_model: Mapped[str] = mapped_column(String(8), nullable=False)
    allocation_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    metrics_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
