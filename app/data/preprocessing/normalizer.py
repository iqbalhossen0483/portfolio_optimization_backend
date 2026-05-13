"""
Min-Max normalization with two distinct axes per the workflow spec:
  - Cross-sectional: ESG scores normalized across N assets at each time t
  - Time-series: OHLCV, RSI, MACD, returns normalized per asset over training window W
    → min/max frozen on training window and reapplied to test window (no look-ahead)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class FrozenScaler:
    """Stores per-asset min/max from training window; applies to any window without leakage."""
    mins: np.ndarray   # shape (N,) or (N, F) — per asset [, per feature]
    maxs: np.ndarray

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        data shape: (..., N) or (..., N, F)
        Returns values in [0, 1]; clips to handle slight out-of-range test values.
        """
        denom = self.maxs - self.mins
        denom = np.where(denom == 0, 1.0, denom)   # avoid division by zero for constant series
        normed = (data - self.mins) / denom
        return np.clip(normed, 0.0, 1.0)


class TimeSeriesNormalizer:
    """
    Per-asset, time-series normalization.
    Fits on the training window; frozen before applying to test window.
    Handles shapes: (T, N) — T timesteps, N assets.
    """

    def __init__(self) -> None:
        self._scaler: FrozenScaler | None = None

    @property
    def is_fitted(self) -> bool:
        return self._scaler is not None

    def fit(self, data: np.ndarray) -> TimeSeriesNormalizer:
        """
        data: (T_train, N)
        Computes min/max per asset (axis=0 → across time for each asset).
        """
        assert data.ndim == 2, "Expected (T, N)"
        mins = data.min(axis=0)   # (N,)
        maxs = data.max(axis=0)   # (N,)
        self._scaler = FrozenScaler(mins=mins, maxs=maxs)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """data: (T, N) → (T, N) normalized."""
        assert self._scaler is not None, "Call fit() on training window first"
        return self._scaler.transform(data)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)


class CrossSectionalNormalizer:
    """
    Per-day normalization across N assets.
    ESG scores only — no fitting needed; computed on-the-fly per timestep.
    Introduces zero temporal look-ahead (uses only same-day peer values).
    """

    @staticmethod
    def transform(data: np.ndarray) -> np.ndarray:
        """
        data: (T, N) — e.g. Bloomberg ESG scores for N assets across T days
        Returns (T, N) normalized within each row (day).
        """
        assert data.ndim == 2
        row_min = data.min(axis=1, keepdims=True)   # (T, 1)
        row_max = data.max(axis=1, keepdims=True)   # (T, 1)
        denom = row_max - row_min
        denom = np.where(denom == 0, 1.0, denom)
        return (data - row_min) / denom


class DataNormalizer:
    """
    Top-level normalizer that enforces the full normalization pipeline from the spec:
      1. Cross-sectional ESG normalization (Bloomberg, LESG)
      2. Compute ΔESGᵢₜ and μESGᵢₜ (post-normalization)
      3. Time-series normalization for OHLCV, RSI, MACD histogram, individual returns
         — fitted on training window, frozen for test
    """

    def __init__(self, n_assets: int) -> None:
        self.n_assets = n_assets
        # One scaler per OHLCV component (5 scalers) + RSI + MACD + returns = 8
        self._ts_scalers: dict[str, TimeSeriesNormalizer] = {}
        self._cs_norm = CrossSectionalNormalizer()
        self._fitted = False

    # ── Training window fit ───────────────────────────────────────────────────

    def fit(self, train_data: dict[str, np.ndarray]) -> DataNormalizer:
        """
        train_data keys: "open", "high", "low", "close", "volume",
                         "rsi", "macd_hist", "returns"
        Each value shape: (T_train, N_assets)
        """
        for key in ("open", "high", "low", "close", "volume", "rsi", "macd_hist", "returns"):
            scaler = TimeSeriesNormalizer()
            scaler.fit(train_data[key])
            self._ts_scalers[key] = scaler
        self._fitted = True
        return self

    # ── Transform any window (train or test) ─────────────────────────────────

    def transform(
        self,
        market_data: dict[str, np.ndarray],
        bloomberg_esg: np.ndarray,    # (T, N) raw Bloomberg 0-100
        lesg_esg: np.ndarray,         # (T, N) raw LESG 0-10
    ) -> dict[str, np.ndarray]:
        """
        Returns normalized feature dict.  All arrays shape (T, N).

        ESG normalization runs cross-sectionally (per spec: same-day peer values only).
        Must complete before ΔESGᵢₜ / μESGᵢₜ computation.
        """
        assert self._fitted, "Call fit() on training window first"
        T, N = bloomberg_esg.shape

        # Step 1: Cross-sectional ESG normalization
        esg_b_norm = self._cs_norm.transform(bloomberg_esg)   # (T, N) ∈ [0,1]
        esg_l_norm = self._cs_norm.transform(lesg_esg)        # (T, N) ∈ [0,1]

        # Step 2: Per-stock ΔESGᵢₜ and μESGᵢₜ (post-normalization, per spec)
        delta_esg = np.abs(esg_b_norm - esg_l_norm)           # (T, N)
        mu_esg = (esg_b_norm + esg_l_norm) / 2.0              # (T, N)

        # Step 3: Time-series normalization (frozen scaler from training window)
        result: dict[str, np.ndarray] = {
            "esg_b_norm":  esg_b_norm,
            "esg_l_norm":  esg_l_norm,
            "delta_esg":   delta_esg,
            "mu_esg":      mu_esg,
        }
        for key in ("open", "high", "low", "close", "volume", "rsi", "macd_hist", "returns"):
            result[key + "_norm"] = self._ts_scalers[key].transform(market_data[key])

        return result

    def fit_transform(
        self,
        market_data: dict[str, np.ndarray],
        bloomberg_esg: np.ndarray,
        lesg_esg: np.ndarray,
    ) -> dict[str, np.ndarray]:
        self.fit(market_data)
        return self.transform(market_data, bloomberg_esg, lesg_esg)

    # ── State vector assembly ─────────────────────────────────────────────────

    @staticmethod
    def build_state_vector(normed: dict[str, np.ndarray], t: int) -> np.ndarray:
        """
        Assembles the 10N observation vector for timestep t:
          [open(N), high(N), low(N), close(N), volume(N),  ← 5N OHLCV
           rsi(N), macd(N), returns(N),                    ← 3N technical
           delta_esg(N), mu_esg(N)]                        ← 2N ESG
        = 10N total (matches spec)
        """
        keys = [
            "open_norm", "high_norm", "low_norm", "close_norm", "volume_norm",
            "rsi_norm", "macd_hist_norm", "returns_norm",
            "delta_esg", "mu_esg",
        ]
        return np.concatenate([normed[k][t] for k in keys], axis=0)   # (10N,)

    # ── Portfolio-level scalar aggregations (used in reward only) ────────────

    @staticmethod
    def portfolio_esg(weights: np.ndarray, esg_norm: np.ndarray, t: int) -> float:
        """ESGₜ^(B or L) = Σᵢ wᵢ · ESGᵢₜ_norm (weighted sum at time t)."""
        return float(np.dot(weights, esg_norm[t]))

    @staticmethod
    def portfolio_delta_esg(weights: np.ndarray, delta_esg: np.ndarray, t: int) -> float:
        """ΔESGₜ = Σᵢ wᵢ · ΔESGᵢₜ (portfolio-level disagreement scalar)."""
        return float(np.dot(weights, delta_esg[t]))
