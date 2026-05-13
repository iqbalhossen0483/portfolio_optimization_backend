"""
Market data source — OHLCV fetcher.
Primary: yfinance (free, for prototyping).
Production: swap _fetch_yfinance for Bloomberg / Refinitiv adapter.
"""
from __future__ import annotations
import asyncio
from datetime import date
import pandas as pd
import structlog

log = structlog.get_logger(__name__)


class MarketDataSource:
    """
    Async OHLCV data fetcher with Redis-backed caching.
    Each ISIN returns a DataFrame with columns: date, open, high, low, close, volume.
    """

    def __init__(self, redis_client=None, ttl: int = 3600) -> None:
        self._redis = redis_client
        self._ttl = ttl

    async def fetch_ohlcv(
        self, isin: str, start: date, end: date
    ) -> pd.DataFrame:
        cache_key = f"cache:market:{isin}:{start}:{end}"
        if self._redis:
            cached = await self._redis.get(cache_key)
            if cached:
                return pd.read_json(cached)

        df = await asyncio.to_thread(self._fetch_yfinance, isin, start, end)
        if self._redis and not df.empty:
            await self._redis.setex(cache_key, self._ttl, df.to_json())
        return df

    async def fetch_ohlcv_batch(
        self, isins: list[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        tasks = [self.fetch_ohlcv(isin, start, end) for isin in isins]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: dict[str, pd.DataFrame] = {}
        for isin, result in zip(isins, results):
            if isinstance(result, Exception):
                log.error("Failed to fetch OHLCV", isin=isin, error=str(result))
                out[isin] = pd.DataFrame()
            else:
                out[isin] = result
        return out

    @staticmethod
    def _fetch_yfinance(isin: str, start: date, end: date) -> pd.DataFrame:
        try:
            import yfinance as yf
            ticker = yf.Ticker(isin)
            hist = ticker.history(start=str(start), end=str(end))
            if hist.empty:
                return pd.DataFrame()
            hist = hist.reset_index()
            hist.columns = [c.lower() for c in hist.columns]
            hist = hist.rename(columns={"datetime": "date", "stock splits": "splits"})
            return hist[["date", "open", "high", "low", "close", "volume"]].copy()
        except Exception as exc:
            log.warning("yfinance fetch failed", isin=isin, error=str(exc))
            return pd.DataFrame()
