"""
Tests for MarketEnvironment — reward functions and topology behavior.
"""
import numpy as np
import pytest

from app.rl.environment import MarketEnvironment
from app.data.pipeline import ProcessedDataset
from app.data.preprocessing.normalizer import DataNormalizer


N = 4
T = 300


@pytest.fixture
def dummy_dataset():
    rng = np.random.default_rng(42)
    # Minimal ProcessedDataset for testing
    state_vectors = rng.uniform(0, 1, (T, 10 * N)).astype(np.float32)
    returns       = rng.uniform(-0.05, 0.05, (T, N)).astype(np.float32)
    esg_b_norm    = rng.uniform(0, 1, (T, N)).astype(np.float32)
    esg_l_norm    = rng.uniform(0, 1, (T, N)).astype(np.float32)
    delta_esg     = np.abs(esg_b_norm - esg_l_norm)
    mu_esg        = (esg_b_norm + esg_l_norm) / 2

    norm = DataNormalizer.__new__(DataNormalizer)
    norm.n_assets = N
    norm._fitted = True
    norm._ts_scalers = {}
    norm._cs_norm = None

    return ProcessedDataset(
        n_assets=N, n_timesteps=T,
        state_vectors=state_vectors,
        returns=returns,
        esg_b_norm=esg_b_norm,
        esg_l_norm=esg_l_norm,
        delta_esg=delta_esg,
        mu_esg=mu_esg,
        asset_isins=[f"ISIN{i}" for i in range(N)],
        dates=[None] * T,
        normalizer=norm,
    )


def _make_env(dataset, portfolio_model="C", topology="cooperative"):
    return MarketEnvironment(
        dataset=dataset,
        portfolio_model=portfolio_model,
        topology=topology,
        alpha_1=0.5, alpha_2=0.5, alpha_3=0.01,
        beta=0.3, lam=0.4,
    )


def test_env_reset_returns_obs(dummy_dataset):
    env = _make_env(dummy_dataset)
    obs = env.reset()
    assert obs.shape == (10 * N,)


def test_env_step_returns_valid_rewards(dummy_dataset):
    env = _make_env(dummy_dataset, topology="cooperative")
    env.reset()
    scores = np.random.randn(N).astype(np.float32)
    result = env.step(scores, scores, scores)
    assert "bloomberg" in result.rewards
    assert "lesg" in result.rewards
    assert "financial" in result.rewards
    assert isinstance(result.done, bool)


def test_cooperative_beta_applied(dummy_dataset):
    env_coop = _make_env(dummy_dataset, topology="cooperative", )
    env_comp = _make_env(dummy_dataset, topology="competitive")
    env_coop._t = env_comp._t = 10

    scores = np.ones(N, dtype=np.float32)
    env_coop._weights = np.ones(N) / N
    env_comp._weights = np.ones(N) / N

    # Build a state where delta_esg is large
    dummy_dataset.esg_b_norm[10] = np.ones(N)
    dummy_dataset.esg_l_norm[10] = np.zeros(N)
    dummy_dataset.delta_esg[10]  = np.ones(N)

    r_coop = env_coop._compute_rewards(0.02, 0.5, 0.5, 1.0)
    r_comp = env_comp._compute_rewards(0.02, 0.5, 0.5, 1.0)

    # Cooperative incurs penalty; competitive does not
    assert r_coop["bloomberg"] < r_comp["bloomberg"]


def test_portfolio_b_degenerate(dummy_dataset):
    env = _make_env(dummy_dataset, portfolio_model="B", topology="cooperative")
    env.reset()
    # When esg_b == esg_l at portfolio level → both signed terms = 0 → all agents get r_t
    rewards = env._compute_rewards(r_t=0.03, esg_b_t=0.5, esg_l_t=0.5, delta_esg_t=0.0)
    assert rewards["bloomberg"] == pytest.approx(0.03, abs=1e-5)
    assert rewards["lesg"]      == pytest.approx(0.03, abs=1e-5)
    assert rewards["financial"] == pytest.approx(0.03, abs=1e-5)


def test_mixed_topology_partial_beta(dummy_dataset):
    env_coop = _make_env(dummy_dataset, topology="cooperative")
    env_mix  = _make_env(dummy_dataset, topology="mixed")
    r_coop = env_coop._compute_rewards(0.02, 0.5, 0.5, 1.0)
    r_mix  = env_mix._compute_rewards(0.02, 0.5, 0.5, 1.0)
    # Mixed should have higher reward than cooperative (smaller penalty)
    assert r_mix["bloomberg"] > r_coop["bloomberg"]


def test_softmax_weights_non_negative_sum_one(dummy_dataset):
    env = _make_env(dummy_dataset)
    env.reset()
    scores = np.random.randn(N).astype(np.float32)
    weights = env._softmax(scores)
    assert abs(weights.sum() - 1.0) < 1e-5
    assert (weights >= 0).all()
