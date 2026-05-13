"""
Uniform replay buffer for MASAC.
Stores joint transitions: (s, a^B, a^L, a^F, r^B, r^L, r^F, s')
Capacity: 1,000,000 transitions.  Minimum fill before training: 10,000.
"""
from __future__ import annotations
import numpy as np


class ReplayBuffer:
    """
    Circular buffer storing MASAC joint transitions.
    All arrays pre-allocated at initialization to avoid repeated allocation.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
    ) -> None:
        self.capacity   = capacity
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self._ptr  = 0
        self._size = 0

        # Joint transition storage
        # obs/next_obs shared across all agents (same global state s_t)
        self._obs      = np.zeros((capacity, obs_dim),      dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim),      dtype=np.float32)

        # Per-agent action vectors (3 agents)
        self._actions_B = np.zeros((capacity, action_dim), dtype=np.float32)
        self._actions_L = np.zeros((capacity, action_dim), dtype=np.float32)
        self._actions_F = np.zeros((capacity, action_dim), dtype=np.float32)

        # Per-agent scalar rewards
        self._rewards_B = np.zeros((capacity, 1), dtype=np.float32)
        self._rewards_L = np.zeros((capacity, 1), dtype=np.float32)
        self._rewards_F = np.zeros((capacity, 1), dtype=np.float32)

        self._dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action_B: np.ndarray,
        action_L: np.ndarray,
        action_F: np.ndarray,
        reward_B: float,
        reward_L: float,
        reward_F: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        i = self._ptr
        self._obs[i]       = obs
        self._next_obs[i]  = next_obs
        self._actions_B[i] = action_B
        self._actions_L[i] = action_L
        self._actions_F[i] = action_F
        self._rewards_B[i] = reward_B
        self._rewards_L[i] = reward_L
        self._rewards_F[i] = reward_F
        self._dones[i]     = float(done)

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs":       self._obs[idx],
            "next_obs":  self._next_obs[idx],
            "actions_B": self._actions_B[idx],
            "actions_L": self._actions_L[idx],
            "actions_F": self._actions_F[idx],
            "rewards_B": self._rewards_B[idx],
            "rewards_L": self._rewards_L[idx],
            "rewards_F": self._rewards_F[idx],
            "dones":     self._dones[idx],
        }

    @property
    def size(self) -> int:
        return self._size

    def is_ready(self, min_size: int) -> bool:
        return self._size >= min_size
