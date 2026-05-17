"""
InferenceService — loads trained MASAC checkpoints and runs deterministic inference.

Flow:
  1. Find the latest completed training_job for the requested portfolio_model
  2. Load the frozen DataNormalizer from training_normalizer_params
  3. Build a 10N state vector from the most recent market_data + esg_scores rows
  4. For each topology (cooperative, competitive, mixed):
       - Load best MASAC checkpoint (highest sharpe in model_checkpoints)
       - Run deterministic actor forward pass → joint softmax → weights
  5. Compute per-asset metrics (annualised return, risk, Sharpe, μESG, ΔESG)
  6. Return three panels ready for the chat response
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import structlog
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings
from app.data.preprocessing.normalizer import DataNormalizer
from app.models.domain import ModelCheckpoint, TrainingJob
from app.rl.masac import MASAC

log = structlog.get_logger(__name__)
cfg = get_settings()


class InferenceService:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    async def run(
        self,
        portfolio_model: str,
        investment_amount: float,
    ) -> dict[str, Any]:
        """
        Full inference pipeline. Returns dict with keys:
          job_id, portfolio_model, panels (cooperative / competitive / mixed)
        """
        engine = create_async_engine(self._dsn)
        SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        try:
            async with SessionLocal() as db:
                job = await self._get_latest_job(portfolio_model, db)
                if not job:
                    raise ValueError(
                        f"No completed training job found for portfolio_model='{portfolio_model}'. "
                        "Upload an XLSX file and start training via POST /api/v1/training/start."
                    )

                isins: list[str] = job.config_json["assets"]
                n = len(isins)

                normalizer = await DataNormalizer.load_from_db(job.id, self._dsn)
                state_vector, asset_metrics = await self._build_current_state(
                    isins, normalizer, db
                )

                panels: dict[str, list[dict]] = {}
                for topology in ("cooperative", "competitive", "mixed"):
                    ckpt_path = await self._get_best_checkpoint_path(job.id, topology, db)
                    masac = MASAC(
                        n_assets=n,
                        gamma=cfg.masac_gamma,
                        tau=cfg.masac_tau,
                        hidden=cfg.masac_hidden_size,
                        lr_actor=cfg.masac_lr_actor,
                        lr_critic=cfg.masac_lr_critic,
                        lr_alpha=cfg.masac_lr_alpha,
                        batch_size=cfg.masac_batch_size,
                        buffer_capacity=cfg.masac_buffer_capacity,
                    )
                    masac.load(ckpt_path)

                    actions = masac.select_actions(state_vector, deterministic=True)
                    z_joint = (
                        actions["bloomberg"] + actions["lesg"] + actions["financial"]
                    ) / 3.0
                    weights = _softmax(z_joint)

                    panels[topology] = self._build_panel(
                        isins, weights, asset_metrics, investment_amount
                    )

                log.info("inference_done", job_id=job.id, portfolio_model=portfolio_model)
                return {
                    "job_id": job.id,
                    "portfolio_model": portfolio_model,
                    "panels": panels,
                }
        finally:
            await engine.dispose()

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _get_latest_job(
        self, portfolio_model: str, db: AsyncSession
    ) -> TrainingJob | None:
        result = await db.execute(
            select(TrainingJob)
            .where(TrainingJob.portfolio_model == portfolio_model)
            .where(TrainingJob.status == "completed")
            .order_by(TrainingJob.id.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def _get_best_checkpoint_path(
        self, job_id: int, topology: str, db: AsyncSession
    ) -> str:
        """Returns the path of the checkpoint with the highest Sharpe for this topology.
        Falls back to the default path if the table has no rows yet."""
        result = await db.execute(
            select(ModelCheckpoint)
            .where(ModelCheckpoint.job_id == job_id)
            .where(ModelCheckpoint.topology == topology)
            .order_by(ModelCheckpoint.sharpe.desc().nullslast())
            .limit(1)
        )
        ckpt = result.scalar_one_or_none()
        if ckpt:
            return ckpt.path
        # Fall back to default path (training may have just finished)
        return os.path.join(cfg.model_store_path, str(job_id), topology)

    async def _build_current_state(
        self,
        isins: list[str],
        normalizer: DataNormalizer,
        db: AsyncSession,
    ) -> tuple[np.ndarray, dict[str, dict]]:
        """
        Queries the most recent row per ISIN from market_data + esg_scores,
        applies the frozen normalizer, and returns:
          - state_vector: (10N,) ready for MASAC actor input
          - asset_metrics: dict[isin → {name, sector, return_ann, risk, sharpe, mu_esg, delta_esg}]
        """
        n = len(isins)
        isin_idx = {isin: i for i, isin in enumerate(isins)}

        # ── Most-recent-date snapshot ─────────────────────────────────────────
        snapshot = await db.execute(
            text("""
                SELECT a.isin, m.open, m.high, m.low, m.close, m.volume,
                       m.rsi, m.return_pct, m.macd_hist,
                       COALESCE(e.delta_esg, 0.0) AS delta_esg,
                       COALESCE(e.mu_esg,    0.0) AS mu_esg
                FROM market_data m
                JOIN assets a ON a.id = m.asset_id
                LEFT JOIN esg_scores e
                       ON e.asset_id = m.asset_id AND e.date = m.date
                WHERE a.isin = ANY(:isins)
                  AND m.date = (
                      SELECT MAX(m2.date)
                      FROM market_data m2
                      JOIN assets a2 ON a2.id = m2.asset_id
                      WHERE a2.isin = ANY(:isins)
                  )
            """),
            {"isins": isins},
        )
        rows = snapshot.fetchall()

        open_v   = np.zeros(n)
        high_v   = np.zeros(n)
        low_v    = np.zeros(n)
        close_v  = np.zeros(n)
        volume_v = np.zeros(n)
        rsi_v    = np.zeros(n)
        ret_v    = np.zeros(n)
        macd_v   = np.zeros(n)
        delta_v  = np.zeros(n)
        mu_v     = np.zeros(n)

        for row in rows:
            j = isin_idx.get(row.isin)
            if j is None:
                continue
            open_v[j]   = row.open   or 0.0
            high_v[j]   = row.high   or 0.0
            low_v[j]    = row.low    or 0.0
            close_v[j]  = row.close  or 0.0
            volume_v[j] = row.volume or 0.0
            rsi_v[j]    = row.rsi         or 0.0
            ret_v[j]    = row.return_pct  or 0.0
            macd_v[j]   = row.macd_hist   or 0.0
            delta_v[j]  = row.delta_esg
            mu_v[j]     = row.mu_esg

        # Apply frozen time-series normalizer (shape (1, N) for each feature)
        def _col(arr: np.ndarray) -> np.ndarray:
            return arr[np.newaxis, :]   # (1, N)

        market = {
            "open":      _col(open_v),
            "high":      _col(high_v),
            "low":       _col(low_v),
            "close":     _col(close_v),
            "volume":    _col(volume_v),
            "rsi":       _col(rsi_v),
            "macd_hist": _col(macd_v),
            "returns":   _col(ret_v),
        }
        normed = normalizer.transform_market_only(market)
        normed["delta_esg"] = _col(delta_v)
        normed["mu_esg"]    = _col(mu_v)

        state_vector = DataNormalizer.build_state_vector(normed, t=0)   # (10N,)

        # ── Asset performance metrics from last 252 trading days ──────────────
        hist = await db.execute(
            text("""
                SELECT a.isin, a.name, a.sector, m.return_pct
                FROM market_data m
                JOIN assets a ON a.id = m.asset_id
                WHERE a.isin = ANY(:isins)
                  AND m.return_pct IS NOT NULL
                  AND m.date >= (
                      SELECT MAX(date) - INTERVAL '365 days' FROM market_data
                  )
                ORDER BY a.isin, m.date
            """),
            {"isins": isins},
        )

        raw: dict[str, dict] = {}
        for row in hist.fetchall():
            if row.isin not in raw:
                raw[row.isin] = {"name": row.name, "sector": row.sector, "returns": []}
            raw[row.isin]["returns"].append(row.return_pct)

        asset_metrics: dict[str, dict] = {}
        for isin in isins:
            j = isin_idx[isin]
            d = raw.get(isin, {})
            r = np.array(d.get("returns", []), dtype=float)
            r = r[~np.isnan(r)]
            if len(r) >= 2:
                ann_return = float(r.mean() * 252)
                risk       = float(r.std() * np.sqrt(252))
            else:
                ann_return, risk = 0.0, 1e-4
            sharpe = ann_return / (risk + 1e-8)
            asset_metrics[isin] = {
                "name":       d.get("name"),
                "sector":     d.get("sector"),
                "return_ann": round(ann_return, 4),
                "risk":       round(risk, 4),
                "sharpe":     round(sharpe, 4),
                "mu_esg":     round(float(mu_v[j]), 4),
                "delta_esg":  round(float(delta_v[j]), 4),
            }

        return state_vector, asset_metrics

    def _build_panel(
        self,
        isins: list[str],
        weights: np.ndarray,
        asset_metrics: dict[str, dict],
        investment_amount: float,
    ) -> list[dict]:
        panel = []
        for i, isin in enumerate(isins):
            w = float(weights[i])
            m = asset_metrics.get(isin, {})
            panel.append({
                "isin":       isin,
                "company":    m.get("name"),
                "sector":     m.get("sector"),
                "return_ann": m.get("return_ann"),
                "risk":       m.get("risk"),
                "sharpe":     m.get("sharpe"),
                "mu_esg":     m.get("mu_esg"),
                "delta_esg":  m.get("delta_esg"),
                "weight":     round(w, 4),
                "allocation": round(w * investment_amount, 2),
            })
        panel.sort(key=lambda x: x["weight"] or 0.0, reverse=True)  # type: ignore[return-value]
        return panel


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()
