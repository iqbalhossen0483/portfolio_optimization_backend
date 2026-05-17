"""
Neural network architectures for MASAC:
  - ActorNetwork:  MLP (10N → 256 → 256) → twin heads (μ_π, log σ²) ∈ ℝᴺ
  - CriticNetwork: MLP (33N → 256 → 256) → scalar Q-value
No tanh squashing on actor output (Softmax accepts unbounded inputs directly).
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch.distributions import Normal


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class ActorNetwork(nn.Module):
    """
    Decentralized actor for one MASAC agent.
    Input:  obs (B, 10N)
    Output: (mean, log_std) each (B, N) — parameterize allocation score distribution
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, log_std) — shapes (B, N) each."""
        h = self.trunk(obs)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reparameterized sample z ~ N(μ, σ²).
        Returns (z, log_prob) where log_prob accounts for the Gaussian density.
        No tanh — z passes directly to Softmax in the orchestrator.
        """
        mean, log_std = self(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()
        log_prob = dist.log_prob(z).sum(dim=-1, keepdim=True)  # (B, 1)
        return z, log_prob

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Inference mode — return mean directly (no sampling)."""
        mean, _ = self(obs)
        return mean


class CriticNetwork(nn.Module):
    """
    Centralized twin critic for one MASAC agent (CTDE paradigm).
    Input:  (all_obs, all_actions) concatenated → (B, 33N)
            3 agents × 10N obs = 30N
            3 agents × N  act  = 3N
            total = 33N
    Output: scalar Q-value (B, 1)
    """

    def __init__(self, state_dim: int, hidden: int = 256) -> None:
        super().__init__()
        # Twin Q-networks
        self.q1 = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, all_obs: torch.Tensor, all_actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (Q1, Q2) each (B, 1)."""
        x = torch.cat([all_obs, all_actions], dim=-1)
        return self.q1(x), self.q2(x)

    def min_q(
        self, all_obs: torch.Tensor, all_actions: torch.Tensor
    ) -> torch.Tensor:
        """min(Q1, Q2) — used to reduce overestimation bias."""
        q1, q2 = self(all_obs, all_actions)
        return torch.min(q1, q2)
