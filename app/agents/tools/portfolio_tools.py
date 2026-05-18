"""
Portfolio agent tools: generate_portfolio and list_available_models.

Both are factory functions because the tool closures must capture a per-request
ChatService instance (for _inference and _portfolio_result state).
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Callable

import structlog

if TYPE_CHECKING:
    from app.services.chat_service import ChatService

log = structlog.get_logger(__name__)


def make_generate_portfolio(service: "ChatService") -> Callable:
    async def generate_portfolio(
        portfolio_model: str,
        investment_amount: float,
        max_assets: int = 3,
    ) -> str:
        """
        Generate portfolio allocations for the given model across all three topologies.

        Args:
            portfolio_model: Must be "A", "B", or "C".
                A = ESG consensus only (no disagreement penalty)
                B = signed ESG disagreement (each agent bets its ESG source is correct)
                C = full model — consensus + uncertainty penalty (recommended)
            investment_amount: Total investment in USD (e.g. 10000000.0 for $10 million)
            max_assets: How many top assets to return per topology panel (default 3).
                Choose this dynamically: use 3 when assets drop off in Sharpe quality,
                up to 5–7 when many assets show positive Sharpe and meaningful weights.
                Use whatever the user requested if they specified a number.

        Returns:
            JSON with panel summaries for cooperative, competitive, and mixed topologies.
        """
        try:
            model_key = portfolio_model.upper()
            if model_key not in ("A", "B", "C"):
                model_key = "C"

            try:
                result = await service._inference.run(model_key, investment_amount)
            except ValueError as ve:
                if "No completed training job" in str(ve) and model_key != "C":
                    log.warning("model_not_trained_falling_back", requested=model_key)
                    model_key = "C"
                    result = await service._inference.run(model_key, investment_amount)
                else:
                    raise

            n = max(1, int(max_assets))

            if not service._portfolio_result:
                service._portfolio_result = {
                    "job_id": result["job_id"],
                    "portfolio_model": model_key,
                    "panels": {},
                }
            else:
                service._portfolio_result["portfolio_model"] = "ALL"

            for topology, assets in result["panels"].items():
                service._portfolio_result["panels"][f"{model_key}_{topology}"] = assets[:n]

            summaries: dict = {}
            for topology, assets in result["panels"].items():
                total_w = sum(a["weight"] for a in assets)
                top_n = assets[:n]
                summaries[topology] = {
                    "top_holdings": [
                        {
                            "isin":       a["isin"],
                            "company":    a["company"],
                            "weight_pct": round(a["weight"] * 100, 1),
                            "allocation": a["allocation"],
                            "sharpe":     a["sharpe"],
                            "mu_esg":     a["mu_esg"],
                            "delta_esg":  a["delta_esg"],
                        }
                        for a in top_n
                    ],
                    "total_weight_check": round(total_w, 4),
                }

            return json.dumps({
                "success": True,
                "portfolio_model": model_key,
                "job_id": result["job_id"],
                "n_assets": len(next(iter(result["panels"].values()))),
                "panel_summaries": summaries,
            })

        except Exception as exc:
            service._portfolio_result = {}
            log.error("generate_portfolio_failed", error=str(exc))
            return json.dumps({"error": str(exc)})

    return generate_portfolio


def make_list_available_models(service: "ChatService") -> Callable:
    async def list_available_models() -> str:
        """
        List all completed training jobs available for inference.
        Call this when the user asks what models have been trained,
        or before generating a portfolio to confirm one exists.
        """
        from sqlalchemy import select
        from sqlalchemy.ext.asyncio import (
            AsyncSession, async_sessionmaker, create_async_engine,
        )
        from app.models.domain import TrainingJob

        engine = create_async_engine(service._dsn)
        try:
            SessionLocal = async_sessionmaker(
                engine, class_=AsyncSession, expire_on_commit=False
            )
            async with SessionLocal() as db:
                result = await db.execute(
                    select(TrainingJob)
                    .where(TrainingJob.status == "completed")
                    .order_by(TrainingJob.id.desc())
                )
                jobs = result.scalars().all()

            if not jobs:
                return json.dumps({
                    "message": (
                        "No completed training jobs found. "
                        "Train a model first via POST /api/v1/training/start."
                    )
                })

            return json.dumps({
                "available_models": [
                    {
                        "job_id":          j.id,
                        "portfolio_model": j.portfolio_model,
                        "topology":        j.topology,
                        "best_sharpe":     j.best_sharpe,
                        "best_mu_esg":     j.best_mu_esg,
                    }
                    for j in jobs
                ]
            })
        finally:
            await engine.dispose()

    return list_available_models
