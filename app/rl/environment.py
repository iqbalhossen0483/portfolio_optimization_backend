"""
MarketEnvironment — Gym-compatible trading environment for MASAC.

State:  10N-dimensional observation vector (shared across all agents).
Action: Each agent outputs N-dimensional allocation score z^(i) ∈ ℝᴺ.
        Scores are averaged and passed through Softmax → portfolio weights.
Reward: Per-agent, parameterized by portfolio model (A/B/C) and topology
        (cooperative/competitive/mixed).

Episode:
  - Length: 252 trading days (1 calendar year)
  - Reset: equal weights (1/N each)
  - No early termination on drawdown
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from app.data.pipeline import ProcessedDataset
from app.data.preprocessing.normalizer import DataNormalizer


@dataclass
class StepResult:
    obs: np.ndarray               # next state (10N,)
    rewards: dict[str, float]     # {"bloomberg": r_B, "lesg": r_L, "financial": r_F}
    done: bool
    info: dict                    # ancillary data (portfolio metrics, etc.)


class MarketEnvironment:
    """
    MASAC market environment with configurable portfolio model and topology.
    All state vectors are pre-computed by DataPipeline.

    portfolio_model: "A" | "B" | "C"
    topology:        "cooperative" | "competitive" | "mixed"
    """

    def __init__(
        self,
        dataset: ProcessedDataset,
        portfolio_model: str = "C",
        topology: str = "cooperative",
        alpha_1: float = 0.5,
        alpha_2: float = 0.5,
        alpha_3: float = 0.01,
        beta: float = 0.3,
        lam: float = 0.4,
    ) -> None:
        self.dataset = dataset
        self.N = dataset.n_assets
        self.portfolio_model = portfolio_model
        self.topology = topology

        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_3 = alpha_3
        self.beta    = beta
        self.lam     = lam

        self._t: int = 0
        self._episode_start: int = 0
        self._weights = np.ones(self.N) / self.N    # equal-weight initial

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Reset to episode start (equal weights)."""
        # Randomly sample episode start within dataset (training variety)
        max_start = max(0, self.dataset.n_timesteps - 252)
        self._episode_start = int(np.random.randint(0, max_start + 1)) if max_start > 0 else 0
        self._t = self._episode_start
        self._weights = np.ones(self.N) / self.N
        return self.dataset.state_vectors[self._t].copy()

    def step(
        self,
        score_B: np.ndarray,   # (N,) unnormalized scores from Bloomberg agent
        score_L: np.ndarray,   # (N,) from LESG agent
        score_F: np.ndarray,   # (N,) from Financial agent
    ) -> StepResult:
        # Aggregate scores → portfolio weights via Softmax
        z_joint = (score_B + score_L + score_F) / 3.0
        weights = self._softmax(z_joint)
        self._weights = weights

        # Compute per-step portfolio metrics
        r_t        = self._portfolio_return(weights)
        esg_b_t    = DataNormalizer.portfolio_esg(weights, self.dataset.esg_b_norm, self._t)
        esg_l_t    = DataNormalizer.portfolio_esg(weights, self.dataset.esg_l_norm, self._t)
        delta_esg_t = DataNormalizer.portfolio_delta_esg(weights, self.dataset.delta_esg, self._t)

        # Compute per-agent rewards
        rewards = self._compute_rewards(r_t, esg_b_t, esg_l_t, delta_esg_t)

        # Advance time
        self._t += 1
        episode_end = (self._t >= self._episode_start + 252) or (self._t >= self.dataset.n_timesteps - 1)

        next_obs = self.dataset.state_vectors[self._t].copy() if not episode_end else \
                   np.zeros(self.dataset.state_vectors.shape[1])

        info = {
            "t": self._t,
            "r_t": r_t,
            "esg_b_t": esg_b_t,
            "esg_l_t": esg_l_t,
            "delta_esg_t": delta_esg_t,
            "weights": weights.tolist(),
        }
        return StepResult(obs=next_obs, rewards=rewards, done=episode_end, info=info)

    # ── Reward functions ──────────────────────────────────────────────────────

    def _compute_rewards(
        self,
        r_t: float,
        esg_b_t: float,
        esg_l_t: float,
        delta_esg_t: float,
    ) -> dict[str, float]:
        effective_beta = self._effective_beta()

        if self.portfolio_model == "A":
            return {
                "bloomberg": r_t + self.alpha_1 * esg_b_t,
                "lesg":      r_t + self.alpha_2 * esg_l_t,
                "financial": r_t + self.alpha_3 * (esg_b_t + esg_l_t) / 2.0,
            }

        if self.portfolio_model == "B":
            # Signed disagreement; degenerate case when esg_b == esg_l → all get r_t
            signed_diff = esg_b_t - esg_l_t
            return {
                "bloomberg": r_t + self.lam * signed_diff,
                "lesg":      r_t + self.lam * (-signed_diff),
                "financial": r_t,
            }

        # Portfolio C — full model
        return {
            "bloomberg": r_t + self.alpha_1 * esg_b_t - effective_beta * delta_esg_t,
            "lesg":      r_t + self.alpha_2 * esg_l_t - effective_beta * delta_esg_t,
            "financial": r_t + self.alpha_3 * (esg_b_t + esg_l_t) / 2.0 - effective_beta * delta_esg_t,
        }

    def _effective_beta(self) -> float:
        if self.topology == "competitive":
            return 0.0
        if self.topology == "mixed":
            return self.beta * 0.5
        return self.beta   # cooperative

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _portfolio_return(self, weights: np.ndarray) -> float:
        """rₜ = Σᵢ wᵢ · Rᵢₜ  (raw simple return, sign-preserving, not normalized)."""
        return float(np.dot(weights, self.dataset.returns[self._t]))

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        e = np.exp(z - z.max())   # numerically stable
        return e / e.sum()

    @property
    def obs_dim(self) -> int:
        return self.dataset.state_vectors.shape[1]   # 10N

    @property
    def action_dim(self) -> int:
        return self.N
