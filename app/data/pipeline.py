"""
DataPipeline — orchestrates: fetch → validate → preprocess → assemble state tensors.
Enforces strict train/test split discipline (no look-ahead bias).

When data comes from DatabaseMarketDataSource / DatabaseESGDataSource the
DataFrames already carry pre-computed columns (rsi, return_pct, macd_hist,
esg_b_norm, esg_l_norm, delta_esg, mu_esg).  The pipeline detects these and
skips recomputation — pure DB reads + in-memory normalization application.

Warmup = cfg.macd_slow (dynamic — controlled via config, not hardcoded).
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
import numpy as np
import pandas as pd

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

    def __init__(self, market_source, esg_source) -> None:
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
        esg_arrays = self._align_esg_full(isins, dates, esg_dict)

        T = len(dates)
        open_   = ohlcv_arrays["open"]    # (T, N)
        high_   = ohlcv_arrays["high"]
        low_    = ohlcv_arrays["low"]
        close_  = ohlcv_arrays["close"]
        volume_ = ohlcv_arrays["volume"]

        # ── 3. Technical indicators ───────────────────────────────────────────
        # DB source returns DataFrames with pre-computed columns → use directly
        has_precomputed = (
            "return_pct" in ohlcv_arrays
            and "rsi" in ohlcv_arrays
            and "macd_hist" in ohlcv_arrays
        )
        if has_precomputed:
            raw_returns   = ohlcv_arrays["return_pct"]   # (T, N) — already computed
            rsi_raw       = ohlcv_arrays["rsi"]           # (T, N) — already computed
            macd_hist_raw = ohlcv_arrays["macd_hist"]     # (T, N) — already computed
        else:
            raw_returns   = compute_returns(close_)
            rsi_raw       = compute_rsi_matrix(close_, cfg.rsi_period)
            macd_hist_raw = compute_macd_histogram_matrix(
                close_, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal
            )

        # ── 4. Remove MACD warm-up rows ───────────────────────────────────────
        # Warmup = cfg.macd_slow (dynamic — never hardcoded)
        warmup = cfg.macd_slow
        dates         = dates[warmup:]
        open_         = open_[warmup:]
        high_         = high_[warmup:]
        low_          = low_[warmup:]
        close_        = close_[warmup:]
        volume_       = volume_[warmup:]
        raw_returns   = raw_returns[warmup:]
        rsi_raw       = rsi_raw[warmup:]
        macd_hist_raw = macd_hist_raw[warmup:]
        T = len(dates)

        market_data = {
            "open": open_, "high": high_, "low": low_,
            "close": close_, "volume": volume_,
            "rsi": rsi_raw, "macd_hist": macd_hist_raw,
            "returns": raw_returns,
        }

        # ── 5. Normalization ──────────────────────────────────────────────────
        # DB source returns ESG with cross-sectional normalization already applied
        has_precomputed_esg = all(
            k in esg_arrays
            for k in ("esg_b_norm", "esg_l_norm", "delta_esg", "mu_esg")
        )

        if has_precomputed_esg:
            esg_b_norm  = esg_arrays["esg_b_norm"][warmup:]
            esg_l_norm  = esg_arrays["esg_l_norm"][warmup:]
            delta_esg_  = esg_arrays["delta_esg"][warmup:]
            mu_esg_     = esg_arrays["mu_esg"][warmup:]

            if fit:
                normalizer = DataNormalizer(n)
                normed = normalizer.fit_transform_market_only(market_data)
            else:
                assert normalizer is not None
                normed = normalizer.transform_market_only(market_data)

            normed["esg_b_norm"] = esg_b_norm
            normed["esg_l_norm"] = esg_l_norm
            normed["delta_esg"]  = delta_esg_
            normed["mu_esg"]     = mu_esg_
        else:
            bloomberg_raw = esg_arrays["bloomberg"][warmup:]
            lesg_raw      = esg_arrays["lesg"][warmup:]

            if fit:
                normalizer = DataNormalizer(n)
                normed = normalizer.fit_transform(market_data, bloomberg_raw, lesg_raw)
            else:
                assert normalizer is not None
                normed = normalizer.transform(market_data, bloomberg_raw, lesg_raw)

            esg_b_norm  = normed["esg_b_norm"]
            esg_l_norm  = normed["esg_l_norm"]
            delta_esg_  = normed["delta_esg"]
            mu_esg_     = normed["mu_esg"]

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
            esg_b_norm=esg_b_norm,
            esg_l_norm=esg_l_norm,
            delta_esg=delta_esg_,
            mu_esg=mu_esg_,
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
        """
        Inner-joins all assets on common trading dates.
        If the source DataFrames include pre-computed columns (rsi, return_pct,
        macd_hist) they are included in the output dict alongside OHLCV.
        """
        frames = [ohlcv_dict[isin].set_index("date") for isin in isins]
        common_idx = frames[0].index
        for f in frames[1:]:
            common_idx = common_idx.intersection(f.index)
        common_idx = common_idx.sort_values()

        # Always include OHLCV; include extras if present in the first frame
        base_cols = ["open", "high", "low", "close", "volume"]
        extra_cols = [c for c in ("return_pct", "rsi", "macd_hist")
                      if c in frames[0].columns]
        all_cols = base_cols + extra_cols

        result: dict[str, np.ndarray] = {
            col: np.zeros((len(common_idx), len(isins))) for col in all_cols
        }
        for j, (isin, frame) in enumerate(zip(isins, frames)):
            aligned = frame.loc[common_idx]
            for col in all_cols:
                result[col][:, j] = aligned[col].to_numpy(dtype=float)

        return list(common_idx), result

    def _align_esg_full(
        self,
        isins: list[str],
        dates: list,
        esg_dict: dict[str, pd.DataFrame],
    ) -> dict[str, np.ndarray]:
        """
        Aligns ESG data to the common date index.

        If the source DataFrames carry pre-computed Stage-2 columns
        (esg_b_norm, esg_l_norm, delta_esg, mu_esg) they are included in the
        output so the pipeline can detect and use them directly.

        Always includes raw bloomberg / lesg arrays as fallback for the
        non-DB path.
        """
        T, N = len(dates), len(isins)
        date_idx = pd.DatetimeIndex(dates)

        bloomberg = np.zeros((T, N), dtype=float)
        lesg      = np.zeros((T, N), dtype=float)

        # Pre-computed ESG norm arrays (only populated when source provides them)
        esg_b_norm  = np.full((T, N), np.nan)
        esg_l_norm  = np.full((T, N), np.nan)
        delta_esg   = np.full((T, N), np.nan)
        mu_esg      = np.full((T, N), np.nan)
        has_precomputed = False

        for j, isin in enumerate(isins):
            df = esg_dict.get(isin, pd.DataFrame())
            if df.empty:
                bloomberg[:, j] = 50.0
                lesg[:, j]      = 5.0
                continue

            df = df.set_index("date").reindex(date_idx, method="ffill")
            bloomberg[:, j] = df["bloomberg_score"].fillna(50.0).to_numpy()
            lesg[:, j]      = df["lesg_score"].fillna(5.0).to_numpy()

            if "esg_b_norm" in df.columns:
                has_precomputed = True
                esg_b_norm[:, j] = df["esg_b_norm"].to_numpy(dtype=float)
                esg_l_norm[:, j] = df["esg_l_norm"].to_numpy(dtype=float)
                delta_esg[:, j]  = df["delta_esg"].to_numpy(dtype=float)
                mu_esg[:, j]     = df["mu_esg"].to_numpy(dtype=float)

        result: dict[str, np.ndarray] = {
            "bloomberg": bloomberg,
            "lesg": lesg,
        }
        if has_precomputed:
            result["esg_b_norm"] = esg_b_norm
            result["esg_l_norm"] = esg_l_norm
            result["delta_esg"]  = delta_esg
            result["mu_esg"]     = mu_esg

        return result
