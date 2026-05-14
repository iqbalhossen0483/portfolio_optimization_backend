"""
ESG data source — Bloomberg (0-100) and LESG (0-10) score fetcher.
Production adapters plug in via _fetch_bloomberg / _fetch_lesg.
Stub implementation returns synthetic data for development/testing.
"""
from __future__ import annotations
import asyncio
from datetime import date
import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)


class ESGDataSource:
    """
    Async ESG score fetcher.
    Returns DataFrames with columns: date, bloomberg_score (0-100), lesg_score (0-10).
    Scores are forward-filled in the pipeline; missing scores use neutral fallbacks.
    """

    def __init__(
        self,
        bloomberg_api_key: str = "",
        lesg_api_key: str = "",
        redis_client=None,
        ttl: int = 86400,
        use_stub: bool = True,
    ) -> None:
        self._bloomberg_key = bloomberg_api_key
        self._lesg_key = lesg_api_key
        self._redis = redis_client
        self._ttl = ttl
        self._use_stub = use_stub

    async def fetch_esg(self, isin: str, start: date, end: date) -> pd.DataFrame:
        cache_key = f"cache:esg:{isin}:{start}:{end}"
        if self._redis:
            cached = await self._redis.get(cache_key)
            if cached:
                return pd.read_json(cached)

        if self._use_stub:
            df = await asyncio.to_thread(self._stub_esg, isin, start, end)
        else:
            df = await asyncio.gather(
                asyncio.to_thread(self._fetch_bloomberg, isin, start, end),
                asyncio.to_thread(self._fetch_lesg, isin, start, end),
            )
            df = self._merge_sources(*df)

        if self._redis and not df.empty:
            await self._redis.setex(cache_key, self._ttl, df.to_json())
        return df

    async def fetch_esg_batch(
        self, isins: list[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        tasks = [self.fetch_esg(isin, start, end) for isin in isins]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: dict[str, pd.DataFrame] = {}
        for isin, result in zip(isins, results):
            if isinstance(result, BaseException):
                log.error("Failed to fetch ESG", isin=isin, error=str(result))
                out[isin] = pd.DataFrame()
            else:
                out[isin] = result
        return out

    # ── Stub (development) ────────────────────────────────────────────────────

    @staticmethod
    def _stub_esg(isin: str, start: date, end: date) -> pd.DataFrame:
        """
        Deterministic synthetic ESG data keyed on ISIN hash so the same ISIN
        always produces the same score (useful for tests / demo).
        ESG scores are relatively stable over time — emitted as quarterly values
        and forward-filled by the pipeline.
        """
        rng = np.random.default_rng(abs(hash(isin)) % (2**31))
        b_score = float(rng.uniform(20, 98))     # Bloomberg 0-100
        l_score = float(rng.uniform(2, 9.8))     # LESG 0-10
        dates = pd.bdate_range(str(start), str(end), freq="QS")  # quarterly signal
        if len(dates) == 0:
            dates = pd.bdate_range(str(start), str(end))[:1]
        return pd.DataFrame({
            "date": dates,
            "bloomberg_score": b_score + rng.normal(0, 2, len(dates)),
            "lesg_score": l_score + rng.normal(0, 0.2, len(dates)),
        })

    # ── Production adapters (to be implemented with real API clients) ────────

    def _fetch_bloomberg(self, isin: str, start: date, end: date) -> pd.DataFrame:
        raise NotImplementedError("Implement Bloomberg ESG API adapter")

    def _fetch_lesg(self, isin: str, start: date, end: date) -> pd.DataFrame:
        raise NotImplementedError("Implement LESG API adapter")

    @staticmethod
    def _merge_sources(b_df: pd.DataFrame, l_df: pd.DataFrame) -> pd.DataFrame:
        merged = pd.merge(b_df, l_df, on="date", how="outer", suffixes=("_b", "_l"))
        return merged.rename(columns={"score_b": "bloomberg_score", "score_l": "lesg_score"})
