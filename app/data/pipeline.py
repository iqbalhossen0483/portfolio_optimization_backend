"""
DataPipeline — orchestrates: fetch → validate → preprocess → assemble state tensors.
Enforces strict train/test split discipline (no look-ahead bias).
Warm-up period (first 26 trading days) consumed for MACD stabilisation.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
import numpy as np
import pandas as pd

from app.data.sources.market import MarketDataSource
from app.data.sources.esg import ESGDataSource
from app.data.preprocessing.normalizer import DataNormalizer
from app.data.preprocessing.indicators import (
    compute_rsi_matrix,
    compute_macd_histogram_matrix,
    compute_returns,
)
from app.config import get_settings


cfg = get_settings()


@dataclass
class ProcessedDataset:
    """Output of DataPipeline.prepare().  Ready for MarketEnvironment consumption."""
    n_assets: int
    n_timesteps: int                   # after warm-up removal
    state_vectors: np.ndarray          # (T, 10N)
    returns: np.ndarray                # (T, N) raw (not normalized) — used for reward
    esg_b_norm: np.ndarray             # (T, N) cross-sectionally normalized Bloomberg
    esg_l_norm: np.ndarray             # (T, N) cross-sectionally normalized LESG
    delta_esg: np.ndarray              # (T, N) per-stock disagreement
    mu_esg: np.ndarray                 # (T, N) per-stock consensus
    asset_isins: list[str]
    dates: list[date]
    normalizer: DataNormalizer         # frozen scalers (for test window reuse)


class DataPipeline:
    """
    Fetch → preprocess → validate data for a list of ISINs over a date range.

    Usage:
        pipeline = DataPipeline(market_source, esg_source)
        train_ds = pipeline.prepare(isins, train_start, train_end, fit=True)
        test_ds  = pipeline.prepare(isins, val_start, val_end, fit=False,
                                     normalizer=train_ds.normalizer)
    """

    def __init__(
        self,
        market_source: MarketDataSource,
        esg_source: ESGDataSource,
    ) -> None:
        self._market = market_source
        self._esg = esg_source

    async def prepare(
        self,
        isins: list[str],
        start: date,
        end: date,
        fit: bool = True,
        normalizer: DataNormalizer | None = None,
    ) -> ProcessedDataset:
        """
        fit=True  → training window: fit normalizer on this window
        fit=False → test/val window: reuse frozen normalizer (pass normalizer= arg)
        """
        if not fit and normalizer is None:
            raise ValueError("Pass normalizer= when fit=False to prevent look-ahead")

        n = len(isins)

        # ── 1. Fetch raw data ─────────────────────────────────────────────────
        ohlcv_dict = await self._market.fetch_ohlcv_batch(isins, start, end)
        esg_dict   = await self._esg.fetch_esg_batch(isins, start, end)

        # ── 2. Align on common business-day index ─────────────────────────────
        dates, ohlcv_arrays = self._align_ohlcv(isins, ohlcv_dict)
        bloomberg_raw, lesg_raw = self._align_esg(isins, dates, esg_dict)

        T = len(dates)
        open_   = ohlcv_arrays["open"]    # (T, N)
        high_   = ohlcv_arrays["high"]
        low_    = ohlcv_arrays["low"]
        close_  = ohlcv_arrays["close"]
        volume_ = ohlcv_arrays["volume"]

        # ── 3. Technical indicators ───────────────────────────────────────────
        raw_returns   = compute_returns(close_)                          # (T, N)
        rsi_raw       = compute_rsi_matrix(close_, cfg.rsi_period)       # (T, N)
        macd_hist_raw = compute_macd_histogram_matrix(                   # (T, N)
            close_, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal
        )

        # ── 4. Remove MACD warm-up rows (first 26 days, per spec) ────────────
        warmup = cfg.macd_warmup_days
        dates       = dates[warmup:]
        open_       = open_[warmup:]
        high_       = high_[warmup:]
        low_        = low_[warmup:]
        close_      = close_[warmup:]
        volume_     = volume_[warmup:]
        raw_returns = raw_returns[warmup:]
        rsi_raw     = rsi_raw[warmup:]
        macd_hist_raw = macd_hist_raw[warmup:]
        bloomberg_raw = bloomberg_raw[warmup:]
        lesg_raw      = lesg_raw[warmup:]
        T = len(dates)

        market_data = {
            "open": open_, "high": high_, "low": low_,
            "close": close_, "volume": volume_,
            "rsi": rsi_raw, "macd_hist": macd_hist_raw,
            "returns": raw_returns,
        }

        # ── 5. Normalization ──────────────────────────────────────────────────
        if fit:
            normalizer = DataNormalizer(n)
            normed = normalizer.fit_transform(market_data, bloomberg_raw, lesg_raw)
        else:
            assert normalizer is not None
            normed = normalizer.transform(market_data, bloomberg_raw, lesg_raw)

        # ── 6. Assemble per-timestep state vectors (10N each) ─────────────────
        state_vectors = np.stack(
            [DataNormalizer.build_state_vector(normed, t) for t in range(T)],
            axis=0,
        )   # (T, 10N)

        return ProcessedDataset(
            n_assets=n,
            n_timesteps=T,
            state_vectors=state_vectors,
            returns=raw_returns,
            esg_b_norm=normed["esg_b_norm"],
            esg_l_norm=normed["esg_l_norm"],
            delta_esg=normed["delta_esg"],
            mu_esg=normed["mu_esg"],
            asset_isins=isins,
            dates=[d.date() if hasattr(d, "date") else d for d in dates],
            normalizer=normalizer,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _align_ohlcv(
        self,
        isins: list[str],
        ohlcv_dict: dict[str, pd.DataFrame],
    ) -> tuple[list, dict[str, np.ndarray]]:
        """Inner-join all assets on common trading dates; return shared date index."""
        frames = [ohlcv_dict[isin].set_index("date") for isin in isins]
        common_idx = frames[0].index
        for f in frames[1:]:
            common_idx = common_idx.intersection(f.index)
        common_idx = common_idx.sort_values()

        result: dict[str, np.ndarray] = {col: np.zeros((len(common_idx), len(isins)))
                                          for col in ("open", "high", "low", "close", "volume")}
        for j, (isin, frame) in enumerate(zip(isins, frames)):
            aligned = frame.loc[common_idx]
            for col in result:
                result[col][:, j] = aligned[col].to_numpy(dtype=float)

        return list(common_idx), result

    def _align_esg(
        self,
        isins: list[str],
        dates: list,
        esg_dict: dict[str, pd.DataFrame],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward-fill ESG scores to match trading-day index.
        Bloomberg raw: 0-100; LESG raw: 0-10.
        Returns (bloomberg_raw (T,N), lesg_raw (T,N)).
        """
        T, N = len(dates), len(isins)
        bloomberg = np.zeros((T, N), dtype=float)
        lesg      = np.zeros((T, N), dtype=float)
        date_idx  = pd.DatetimeIndex(dates)

        for j, isin in enumerate(isins):
            df = esg_dict.get(isin, pd.DataFrame())
            if df.empty:
                bloomberg[:, j] = 50.0   # neutral fallback
                lesg[:, j]      = 5.0
                continue
            df = df.set_index("date").reindex(date_idx, method="ffill")
            bloomberg[:, j] = df["bloomberg_score"].fillna(50.0).to_numpy()
            lesg[:, j]      = df["lesg_score"].fillna(5.0).to_numpy()

        return bloomberg, lesg
