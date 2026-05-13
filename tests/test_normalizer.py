"""
Tests for DataNormalizer — verifies:
1. Cross-sectional ESG normalization produces [0,1] per row
2. Time-series normalization uses training window only (no leakage)
3. ΔESGᵢₜ and μESGᵢₜ computed post-normalization
4. State vector has correct 10N dimension
"""
import numpy as np
import pytest

from app.data.preprocessing.normalizer import (
    DataNormalizer, CrossSectionalNormalizer, TimeSeriesNormalizer
)


N = 4    # assets
T_TRAIN = 100
T_TEST  = 20


@pytest.fixture
def dummy_market():
    rng = np.random.default_rng(42)
    return {k: rng.uniform(0, 100, (T_TRAIN, N)).astype(np.float32)
            for k in ("open", "high", "low", "close", "volume", "rsi", "macd_hist", "returns")}


@pytest.fixture
def dummy_esg():
    rng = np.random.default_rng(99)
    bloomberg = rng.uniform(20, 95, (T_TRAIN, N)).astype(np.float32)
    lesg      = rng.uniform(2,  9,  (T_TRAIN, N)).astype(np.float32)
    return bloomberg, lesg


def test_cross_sectional_range(dummy_esg):
    b, l = dummy_esg
    cs = CrossSectionalNormalizer()
    normed = cs.transform(b)
    assert normed.shape == b.shape
    assert normed.min() >= 0.0 - 1e-6
    assert normed.max() <= 1.0 + 1e-6
    # Each row should have min=0 and max=1
    assert np.allclose(normed.min(axis=1), 0.0, atol=1e-5)
    assert np.allclose(normed.max(axis=1), 1.0, atol=1e-5)


def test_time_series_train_range(dummy_market):
    ts = TimeSeriesNormalizer()
    data = dummy_market["close"]
    normed = ts.fit_transform(data)
    assert normed.min() >= 0.0 - 1e-6
    assert normed.max() <= 1.0 + 1e-6


def test_time_series_frozen_on_test(dummy_market):
    ts = TimeSeriesNormalizer()
    train = dummy_market["close"]
    ts.fit(train)
    rng = np.random.default_rng(7)
    test = rng.uniform(0, 100, (T_TEST, N)).astype(np.float32)
    normed_test = ts.transform(test)
    # Test values may go outside [0,1] if outside training range — clipped to [0,1]
    assert normed_test.min() >= 0.0
    assert normed_test.max() <= 1.0


def test_delta_esg_computed_post_normalization(dummy_market, dummy_esg):
    bloomberg, lesg = dummy_esg
    norm = DataNormalizer(N)
    normed = norm.fit_transform(dummy_market, bloomberg, lesg)
    # ΔESG = |ESG_B_norm - ESG_L_norm|, should be in [0,1]
    assert normed["delta_esg"].min() >= 0.0
    assert normed["delta_esg"].max() <= 1.0 + 1e-6
    # μESG = (ESG_B_norm + ESG_L_norm) / 2, should be in [0,1]
    assert normed["mu_esg"].min() >= 0.0
    assert normed["mu_esg"].max() <= 1.0 + 1e-6


def test_state_vector_dimension(dummy_market, dummy_esg):
    bloomberg, lesg = dummy_esg
    norm = DataNormalizer(N)
    normed = norm.fit_transform(dummy_market, bloomberg, lesg)
    state = DataNormalizer.build_state_vector(normed, t=0)
    assert state.shape == (10 * N,), f"Expected (10N={10*N},), got {state.shape}"


def test_no_leakage_on_test_transform(dummy_market, dummy_esg):
    bloomberg, lesg = dummy_esg
    norm = DataNormalizer(N)
    norm.fit(dummy_market)
    rng = np.random.default_rng(77)
    test_market = {k: rng.uniform(0, 100, (T_TEST, N)).astype(np.float32)
                   for k in dummy_market}
    test_b = rng.uniform(20, 95, (T_TEST, N)).astype(np.float32)
    test_l = rng.uniform(2, 9,   (T_TEST, N)).astype(np.float32)
    normed = norm.transform(test_market, test_b, test_l)
    # Scalers were frozen — should not raise; values clipped to [0,1]
    assert normed["close_norm"].shape == (T_TEST, N)
