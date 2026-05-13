"""
Technical indicators: RSI (period=14) and MACD histogram (12, 26, 9).
MACD warmup: first 26 bars are NaN — caller must drop warm-up period.
All functions operate on a 1-D price series per asset and are vectorized
across N assets in the pipeline layer.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ── RSI ───────────────────────────────────────────────────────────────────────

def compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """
    RSI using Wilder's smoothed moving average (standard finance convention).
    Input:  closes (T,) — 1D closing price array
    Output: rsi    (T,) — values in [0, 100]; NaN for first `period` bars
    """
    assert closes.ndim == 1
    delta = np.diff(closes, prepend=closes[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    rsi = np.full_like(closes, np.nan, dtype=float)
    if len(closes) <= period:
        return rsi

    # Seed with simple average for first window
    avg_gain = gain[1 : period + 1].mean()
    avg_loss = loss[1 : period + 1].mean()

    for i in range(period, len(closes)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
        rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def compute_rsi_matrix(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Vectorized RSI over (T, N) close prices.
    Returns (T, N) RSI values; NaN for warm-up bars.
    """
    T, N = closes.shape
    result = np.full((T, N), np.nan, dtype=float)
    for n in range(N):
        result[:, n] = compute_rsi(closes[:, n], period)
    return result


# ── MACD ──────────────────────────────────────────────────────────────────────

def _ema(series: np.ndarray, span: int) -> np.ndarray:
    """EMA with pandas for numerical stability (handles NaN seeds correctly)."""
    return pd.Series(series).ewm(span=span, adjust=False).mean().to_numpy()


def compute_macd_histogram(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> np.ndarray:
    """
    MACD histogram = MACD line − Signal line
    MACD line  = EMA(fast) − EMA(slow)
    Signal line = EMA(MACD, signal)

    Per spec: only the histogram enters the state vector.
    Minimum lookback to stabilise slow EMA: 26 bars.

    Input:  closes  (T,)
    Output: hist    (T,) — values are raw (not normalized here)
    """
    assert closes.ndim == 1
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    return macd_line - signal_line


def compute_macd_histogram_matrix(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> np.ndarray:
    """
    Vectorized MACD histogram over (T, N) close prices.
    Returns (T, N) histogram values.
    """
    T, N = closes.shape
    result = np.zeros((T, N), dtype=float)
    for n in range(N):
        result[:, n] = compute_macd_histogram(closes[:, n], fast, slow, signal)
    return result


# ── Returns ───────────────────────────────────────────────────────────────────

def compute_returns(closes: np.ndarray) -> np.ndarray:
    """
    Individual simple return: Rᵢₜ = (Closeₜ − Closeₜ₋₁) / Closeₜ₋₁
    Input:  closes  (T, N) or (T,)
    Output: returns same shape; first row set to 0.0 (no prior day)
    """
    if closes.ndim == 1:
        ret = np.zeros_like(closes, dtype=float)
        ret[1:] = (closes[1:] - closes[:-1]) / closes[:-1]
        return ret
    T, N = closes.shape
    ret = np.zeros((T, N), dtype=float)
    ret[1:] = (closes[1:] - closes[:-1]) / closes[:-1]
    return ret
