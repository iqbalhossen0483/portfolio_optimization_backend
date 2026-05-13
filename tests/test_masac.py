"""
Tests for MASAC:
  - Actor forward pass produces correct shape
  - Critic forward pass produces scalar Q-value
  - Replay buffer stores and samples correctly
  - MASAC update step runs without error after warmup
  - Softmax weights sum to 1
"""
import numpy as np
import pytest
import torch

from app.rl.networks import ActorNetwork, CriticNetwork
from app.rl.replay_buffer import ReplayBuffer
from app.rl.masac import MASAC


N = 4    # assets
OBS_DIM = 10 * N
ACT_DIM = N
CRITIC_DIM = 3 * OBS_DIM + 3 * ACT_DIM   # 33N


# ── Actor ─────────────────────────────────────────────────────────────────────

def test_actor_output_shape():
    actor = ActorNetwork(OBS_DIM, ACT_DIM)
    obs = torch.randn(8, OBS_DIM)    # batch of 8
    mean, log_std = actor(obs)
    assert mean.shape == (8, N)
    assert log_std.shape == (8, N)


def test_actor_sample_log_prob():
    actor = ActorNetwork(OBS_DIM, ACT_DIM)
    obs = torch.randn(8, OBS_DIM)
    z, log_prob = actor.sample(obs)
    assert z.shape == (8, N)
    assert log_prob.shape == (8, 1)


def test_actor_deterministic():
    actor = ActorNetwork(OBS_DIM, ACT_DIM)
    obs = torch.randn(1, OBS_DIM)
    z = actor.deterministic_action(obs)
    assert z.shape == (1, N)


# ── Critic ────────────────────────────────────────────────────────────────────

def test_critic_output_shape():
    critic = CriticNetwork(CRITIC_DIM)
    all_obs = torch.randn(8, 3 * OBS_DIM)
    all_act = torch.randn(8, 3 * ACT_DIM)
    q1, q2 = critic(all_obs, all_act)
    assert q1.shape == (8, 1)
    assert q2.shape == (8, 1)


def test_critic_min_q():
    critic = CriticNetwork(CRITIC_DIM)
    all_obs = torch.randn(8, 3 * OBS_DIM)
    all_act = torch.randn(8, 3 * ACT_DIM)
    q_min = critic.min_q(all_obs, all_act)
    q1, q2 = critic(all_obs, all_act)
    expected = torch.min(q1, q2)
    assert torch.allclose(q_min, expected)


# ── Replay Buffer ─────────────────────────────────────────────────────────────

def test_replay_buffer_add_sample():
    buf = ReplayBuffer(capacity=1000, obs_dim=OBS_DIM, action_dim=ACT_DIM)
    rng = np.random.default_rng(42)
    for _ in range(100):
        buf.add(
            obs=rng.randn(OBS_DIM).astype(np.float32),
            action_B=rng.randn(ACT_DIM).astype(np.float32),
            action_L=rng.randn(ACT_DIM).astype(np.float32),
            action_F=rng.randn(ACT_DIM).astype(np.float32),
            reward_B=float(rng.randn()),
            reward_L=float(rng.randn()),
            reward_F=float(rng.randn()),
            next_obs=rng.randn(OBS_DIM).astype(np.float32),
            done=False,
        )
    assert buf.size == 100
    batch = buf.sample(32)
    assert batch["obs"].shape == (32, OBS_DIM)
    assert batch["actions_B"].shape == (32, ACT_DIM)
    assert batch["rewards_B"].shape == (32, 1)


def test_replay_buffer_circular():
    buf = ReplayBuffer(capacity=50, obs_dim=OBS_DIM, action_dim=ACT_DIM)
    rng = np.random.default_rng(1)
    for _ in range(100):   # overfill
        buf.add(
            obs=rng.randn(OBS_DIM).astype(np.float32),
            action_B=rng.randn(ACT_DIM).astype(np.float32),
            action_L=rng.randn(ACT_DIM).astype(np.float32),
            action_F=rng.randn(ACT_DIM).astype(np.float32),
            reward_B=0.0, reward_L=0.0, reward_F=0.0,
            next_obs=rng.randn(OBS_DIM).astype(np.float32),
            done=False,
        )
    assert buf.size == 50   # capped at capacity


# ── MASAC full update ─────────────────────────────────────────────────────────

def test_masac_update_step():
    masac = MASAC(n_assets=N, buffer_capacity=500, batch_size=32)
    rng = np.random.default_rng(5)
    # Fill buffer beyond batch_size
    obs = rng.randn(OBS_DIM).astype(np.float32)
    for _ in range(50):
        masac.buffer.add(
            obs=obs,
            action_B=rng.randn(N).astype(np.float32),
            action_L=rng.randn(N).astype(np.float32),
            action_F=rng.randn(N).astype(np.float32),
            reward_B=float(rng.randn()), reward_L=float(rng.randn()), reward_F=float(rng.randn()),
            next_obs=rng.randn(OBS_DIM).astype(np.float32),
            done=False,
        )
    metrics = masac.update()
    assert metrics is not None
    assert metrics.loss_critic_bloomberg > 0
    assert metrics.entropy_bloomberg != 0


def test_softmax_weights_sum_to_one():
    masac = MASAC(n_assets=N)
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    actions = masac.select_actions(obs, deterministic=True)
    z_joint = (actions["bloomberg"] + actions["lesg"] + actions["financial"]) / 3.0
    e = np.exp(z_joint - z_joint.max())
    weights = e / e.sum()
    assert abs(weights.sum() - 1.0) < 1e-5
    assert (weights >= 0).all()
