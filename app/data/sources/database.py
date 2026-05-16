"""
DB-backed data sources for MASAC training (Stage 4).

DatabaseMarketDataSource and DatabaseESGDataSource fetch all N ISINs in a
single SQL query each — no per-ISIN loops, no external API calls.
They return DataFrames that carry the pre-computed columns (return_pct, rsi,
macd_hist, esg_b_norm, esg_l_norm, delta_esg, mu_esg) so that DataPipeline
detects them and skips recomputation.

Both classes create a short-lived async engine per call, which is safe for
Celery worker subprocesses that cannot share a connection pool from the main
FastAPI process.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

log = structlog.get_logger(__name__)


class DatabaseMarketDataSource:
    """
    Reads pre-computed OHLCV + RSI + return_pct + macd_hist from market_data table.
    Single query for all N ISINs — result pivoted to dict[isin → DataFrame].
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    async def fetch_ohlcv_batch(
        self,
        isins: list[str],
        start: date,
        end: date,
    ) -> dict[str, pd.DataFrame]:
        """
        Returns dict[isin → DataFrame] with columns:
          date, open, high, low, close, volume, rsi, return_pct, macd_hist
        Single SQL query (IN clause) — no per-ISIN loops.
        """
        engine = create_async_engine(self._dsn, future=True)
        try:
            async with engine.connect() as conn:
                rows = await conn.execute(
                    text("""
                        SELECT
                            a.isin,
                            m.date,
                            m.open, m.high, m.low, m.close, m.volume,
                            m.rsi, m.return_pct, m.macd_hist
                        FROM market_data m
                        JOIN assets a ON a.id = m.asset_id
                        WHERE a.isin = ANY(:isins)
                          AND m.date BETWEEN :start AND :end
                        ORDER BY a.isin, m.date
                    """),
                    {"isins": isins, "start": start, "end": end},
                )
                df = pd.DataFrame(rows.fetchall(), columns=list(rows.keys()))
        finally:
            await engine.dispose()

        if df.empty:
            log.warning("db_market_empty", isins=isins, start=start, end=end)
            return {isin: pd.DataFrame() for isin in isins}

        df["date"] = pd.to_datetime(df["date"])

        result: dict[str, pd.DataFrame] = {}
        for isin, group in df.groupby("isin", sort=False):
            result[str(isin)] = group.drop(columns=["isin"]).reset_index(drop=True)

        log.info(
            "db_market_fetched",
            n_isins=len(result),
            total_rows=len(df),
            start=str(start),
            end=str(end),
        )
        return result

    # Alias so the pipeline can also call fetch_ohlcv (single ISIN) if needed
    async def fetch_ohlcv(
        self, isin: str, start: date, end: date
    ) -> pd.DataFrame:
        batch = await self.fetch_ohlcv_batch([isin], start, end)
        return batch.get(isin, pd.DataFrame())


class DatabaseESGDataSource:
    """
    Reads raw ESG scores + Stage-2 cross-sectional normalization results
    from esg_scores table.
    Single query for all N ISINs — result pivoted to dict[isin → DataFrame].
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    async def fetch_esg_batch(
        self,
        isins: list[str],
        start: date,
        end: date,
    ) -> dict[str, pd.DataFrame]:
        """
        Returns dict[isin → DataFrame] with columns:
          date, bloomberg_score, lesg_score,
          esg_b_norm, esg_l_norm, delta_esg, mu_esg
        Single SQL query (IN clause) — no per-ISIN loops.
        """
        engine = create_async_engine(self._dsn, future=True)
        try:
            async with engine.connect() as conn:
                rows = await conn.execute(
                    text("""
                        SELECT
                            a.isin,
                            e.date,
                            e.bloomberg_score,
                            e.lesg_score,
                            e.esg_b_norm,
                            e.esg_l_norm,
                            e.delta_esg,
                            e.mu_esg
                        FROM esg_scores e
                        JOIN assets a ON a.id = e.asset_id
                        WHERE a.isin = ANY(:isins)
                          AND e.date BETWEEN :start AND :end
                        ORDER BY a.isin, e.date
                    """),
                    {"isins": isins, "start": start, "end": end},
                )
                df = pd.DataFrame(rows.fetchall(), columns=list(rows.keys()))
        finally:
            await engine.dispose()

        if df.empty:
            log.warning("db_esg_empty", isins=isins, start=start, end=end)
            return {isin: pd.DataFrame() for isin in isins}

        df["date"] = pd.to_datetime(df["date"])

        result: dict[str, pd.DataFrame] = {}
        for isin, group in df.groupby("isin", sort=False):
            result[str(isin)] = group.drop(columns=["isin"]).reset_index(drop=True)

        log.info(
            "db_esg_fetched",
            n_isins=len(result),
            total_rows=len(df),
            start=str(start),
            end=str(end),
        )
        return result

    # Alias so the pipeline can also call fetch_esg (single ISIN) if needed
    async def fetch_esg(
        self, isin: str, start: date, end: date
    ) -> pd.DataFrame:
        batch = await self.fetch_esg_batch([isin], start, end)
        return batch.get(isin, pd.DataFrame())
