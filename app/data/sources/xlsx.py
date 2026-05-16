"""
XLSXDataSource — parses one or more .xlsx files in Stock_ESG_Dataset format.

Computed from Close:
  - return_pct  : per-ISIN pct_change of Close
  - macd_hist   : MACD histogram per-ISIN using cfg.macd_fast/slow/signal

N (number of ISINs) and T (number of dates) are always dynamic —
derived at parse time from the actual file contents.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import structlog
from ta.trend import MACD

from app.config import get_settings

log = structlog.get_logger(__name__)
cfg = get_settings()

# Maps XLSX column headers → internal names.
COLUMN_MAP: dict[str, str] = {
    "Date": "date",
    "ISIN": "isin",
    "Company name": "name",
    "Sector": "sector",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "RSI": "rsi",
    "Bloom. ESG (0-100)": "bloomberg_score",
    "LESG ESG (0-10)": "lesg_score",
}

_VOLUME_MULTIPLIERS: dict[str, float] = {
    "K": 1_000.0,
    "M": 1_000_000.0,
    "B": 1_000_000_000.0,
    "T": 1_000_000_000_000.0,
}


def _parse_volume(v: Any) -> float:
    """
    Parses volume strings such as '10.5M', '2.3k', '1B', or plain numbers.
    Returns NaN on any parse failure (never raises).
    """
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return float("nan")
    suffix = s[-1].upper()
    if suffix in _VOLUME_MULTIPLIERS:
        try:
            return float(s[:-1]) * _VOLUME_MULTIPLIERS[suffix]
        except ValueError:
            log.warning("volume_parse_failed", raw=v)
            return float("nan")
    try:
        return float(s)
    except ValueError:
        log.warning("volume_parse_failed", raw=v)
        return float("nan")


def _macd_hist_series(close: pd.Series) -> pd.Series:
    """Computes MACD histogram for a single ISIN's Close series using config params."""
    indicator = MACD(
        close=close,
        window_fast=cfg.macd_fast,
        window_slow=cfg.macd_slow,
        window_sign=cfg.macd_signal,
        fillna=False,
    )
    return indicator.macd_diff()


@dataclass
class ParsedXLSX:
    """
    Output of XLSXDataSource.parse_files().

    assets     : deduplicated list of {isin, name, sector} dicts — N entries
    market_df  : DataFrame with columns [isin, date, open, high, low, close,
                 volume, rsi, return_pct, macd_hist] — shape (T×N, 10)
    esg_df     : DataFrame with columns [isin, date, bloomberg_score,
                 lesg_score] — shape (T×N, 4)
    n_assets   : N (dynamic — from file)
    n_timesteps: T (unique trading dates after deduplication)
    """
    assets: list[dict]
    market_df: pd.DataFrame
    esg_df: pd.DataFrame
    n_assets: int
    n_timesteps: int

    def all_dates(self) -> np.ndarray:
        """Sorted unique trading dates across all ISINs."""
        return np.sort(self.market_df["date"].unique())


class XLSXDataSource:

    @staticmethod
    def parse_files(paths: list[str]) -> ParsedXLSX:
        """
        Reads one or more .xlsx files (sheet: Stock_ESG_Dataset), concatenates,
        deduplicates on (ISIN, Date), computes return_pct + macd_hist from Close,
        and returns a ParsedXLSX dataclass.

        All N assets processed in a single vectorized pass — no Python loops over ISINs.
        """
        frames: list[pd.DataFrame] = []
        for path in paths:
            try:
                raw = pd.read_excel(path, sheet_name="Stock_ESG_Dataset")
            except Exception as exc:
                log.error("xlsx_read_failed", path=path, error=str(exc))
                raise

            # Rename columns using COLUMN_MAP; drop any unmapped columns
            present = {k: v for k, v in COLUMN_MAP.items() if k in raw.columns}
            raw = raw.rename(columns=present)[list(present.values())]
            frames.append(raw)

        df = pd.concat(frames, ignore_index=True)

        # Parse dates
        df["date"] = pd.to_datetime(df["date"])

        # Parse volume
        df["volume"] = df["volume"].apply(_parse_volume)

        # Deduplicate on (isin, date) — keep first occurrence across files
        df = df.drop_duplicates(subset=["isin", "date"]).reset_index(drop=True)

        # Sort for per-ISIN time-series operations
        df = df.sort_values(["isin", "date"]).reset_index(drop=True)

        # ── Derived features (vectorized — one pass over all N ISINs) ─────────

        # return_pct: per-ISIN pct_change of Close (NaN for first row per ISIN)
        df["return_pct"] = (
            df.groupby("isin", sort=False)["close"]
            .pct_change()
        )

        # macd_hist: per-ISIN MACD histogram using config params
        # NaN for first (macd_slow - 1) rows per ISIN
        df["macd_hist"] = (
            df.groupby("isin", sort=False)["close"]
            .transform(_macd_hist_series)
        )

        # ── Build output DataFrames ────────────────────────────────────────────

        market_cols = ["isin", "date", "open", "high", "low", "close",
                       "volume", "rsi", "return_pct", "macd_hist"]
        esg_cols = ["isin", "date", "bloomberg_score", "lesg_score"]

        market_df = df[market_cols].copy()
        esg_df = df[[c for c in esg_cols if c in df.columns]].copy()

        # Deduplicated asset metadata
        assets = (
            df[["isin", "name", "sector"]]
            .drop_duplicates(subset=["isin"])
            .to_dict(orient="records")
        )

        n_assets = int(df["isin"].nunique())
        n_timesteps = int(df["date"].nunique())

        log.info(
            "xlsx_parsed",
            files=len(paths),
            n_assets=n_assets,
            n_timesteps=n_timesteps,
            total_rows=len(df),
        )

        return ParsedXLSX(
            assets=assets,
            market_df=market_df,
            esg_df=esg_df,
            n_assets=n_assets,
            n_timesteps=n_timesteps,
        )

    @staticmethod
    def derive_date_split(
        all_dates: np.ndarray,
    ) -> tuple[date, date, date, date]:
        """
        Auto-splits sorted unique trading dates 80/20 train/val.

        Returns (train_start, train_end, val_start, val_end) as date objects.
        T is dynamic — never hardcoded.
        """
        dates = np.sort(all_dates)
        T = len(dates)
        if T < 4:
            raise ValueError(f"Need at least 4 dates for a train/val split, got {T}")

        split_idx = int(T * 0.8)
        # Ensure at least 1 date in each split
        split_idx = max(1, min(split_idx, T - 1))

        train_dates = dates[:split_idx]
        val_dates = dates[split_idx:]

        def _to_date(ts: Any) -> date:
            return pd.Timestamp(ts).date()

        return (
            _to_date(train_dates[0]),
            _to_date(train_dates[-1]),
            _to_date(val_dates[0]),
            _to_date(val_dates[-1]),
        )
