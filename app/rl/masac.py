"""
MASAC — Multi-Agent Soft Actor-Critic.
Manages 3 agents, each with:
  - 1 Actor (decentralized execution)
  - 2 Critics + 2 Target critics (centralized training, CTDE)
  - 1 learnable temperature α_T

Target entropy: H̄ = −N  (negative number of assets)
Soft target update: τ = 0.005
"""
from __future__ import annotations
import copy
from typing import NamedTuple
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from app.rl.networks import ActorNetwork, CriticNetwork
from app.rl.replay_buffer import ReplayBuffer


AGENT_NAMES = ("bloomberg", "lesg", "financial")


class UpdateMetrics(NamedTuple):
    loss_critic_bloomberg: float
    loss_critic_lesg: float
    loss_critic_financial: float
    loss_actor_bloomberg: float
    loss_actor_lesg: float
    loss_actor_financial: float
    alpha_t_bloomberg: float
    alpha_t_lesg: float
    alpha_t_financial: float
    entropy_bloomberg: float
    entropy_lesg: float
    entropy_financial: float


class MASACAgent:
    """
    One MASAC agent: owns Actor, twin Critics, twin target Critics, temperature.
    Critic input dimension: 33N (3 agents × 10N obs + 3 agents × N actions).
    """

    def __init__(
        self,
        name: str,
        obs_dim: int,        # 10N
        action_dim: int,     # N
        n_agents: int = 3,
        hidden: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        initial_alpha_t: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        self.name = name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device or torch.device("cpu")

        # Centralized critic input: 3 × obs_dim + 3 × action_dim
        critic_in = n_agents * obs_dim + n_agents * action_dim

        self.actor = ActorNetwork(obs_dim, action_dim, hidden).to(self.device)
        self.critic = CriticNetwork(critic_in, hidden).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_opt  = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Auto-tuned temperature
        self.log_alpha_t = torch.tensor(
            np.log(initial_alpha_t), dtype=torch.float32,
            requires_grad=True, device=self.device
        )
        self.alpha_t_opt = optim.Adam([self.log_alpha_t], lr=lr_alpha)
        self.target_entropy = -float(action_dim)   # H̄ = −N


class MASAC:
    """
    Multi-agent coordinator: manages 3 MASACAgents and the shared ReplayBuffer.
    Implements the joint update step per the workflow spec.
    """

    def __init__(
        self,
        n_assets: int,
        gamma: float = 0.99,
        tau: float = 0.005,
        hidden: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        initial_alpha_t: float = 1.0,
        buffer_capacity: int = 1_000_000,
        batch_size: int = 256,
        device: str = "cpu",
    ) -> None:
        self.n_assets   = n_assets
        self.obs_dim    = 10 * n_assets    # 10N
        self.action_dim = n_assets          # N
        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size
        self.device     = torch.device(device)

        self.agents: dict[str, MASACAgent] = {}
        for name in AGENT_NAMES:
            self.agents[name] = MASACAgent(
                name=name,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden=hidden,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                lr_alpha=lr_alpha,
                initial_alpha_t=initial_alpha_t,
                device=self.device,
            )

        self.buffer = ReplayBuffer(
            capacity=buffer_capacity,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
        )

    # ── Action selection ──────────────────────────────────────────────────────

    def select_actions(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> dict[str, np.ndarray]:
        """
        obs: (10N,) — shared global observation
        Returns {"bloomberg": z, "lesg": z, "financial": z} each (N,)
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        actions: dict[str, np.ndarray] = {}
        with torch.no_grad():
            for name, agent in self.agents.items():
                if deterministic:
                    z = agent.actor.deterministic_action(obs_t)
                else:
                    z, _ = agent.actor.sample(obs_t)
                actions[name] = z.squeeze(0).cpu().numpy()
        return actions

    # ── Update step ───────────────────────────────────────────────────────────

    def update(self) -> UpdateMetrics | None:
        if not self.buffer.is_ready(self.batch_size):
            return None

        batch = self.buffer.sample(self.batch_size)
        obs      = torch.tensor(batch["obs"],      dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones    = torch.tensor(batch["dones"],    dtype=torch.float32, device=self.device)

        actions = {
            "bloomberg": torch.tensor(batch["actions_B"], dtype=torch.float32, device=self.device),
            "lesg":      torch.tensor(batch["actions_L"], dtype=torch.float32, device=self.device),
            "financial": torch.tensor(batch["actions_F"], dtype=torch.float32, device=self.device),
        }
        rewards = {
            "bloomberg": torch.tensor(batch["rewards_B"], dtype=torch.float32, device=self.device),
            "lesg":      torch.tensor(batch["rewards_L"], dtype=torch.float32, device=self.device),
            "financial": torch.tensor(batch["rewards_F"], dtype=torch.float32, device=self.device),
        }

        # Centralized inputs: concatenate all obs and all actions
        all_actions = torch.cat(list(actions.values()), dim=-1)   # (B, 3N)

        # Shared input tensors for critics
        joint_obs    = obs.repeat(1, 3).view(obs.shape[0], -1)       # naive: use same obs 3×
        joint_next   = next_obs.repeat(1, 3).view(next_obs.shape[0], -1)

        metrics_dict: dict = {}

        # ── Per-agent updates ──────────────────────────────────────────────────
        with torch.no_grad():
            # Sample next actions for TD target
            next_actions: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for name, agent in self.agents.items():
                z_next, log_prob_next = agent.actor.sample(next_obs)
                next_actions[name] = (z_next, log_prob_next)
            next_all_actions = torch.cat([na[0] for na in next_actions.values()], dim=-1)

        for name, agent in self.agents.items():
            alpha_t = agent.log_alpha_t.exp().detach()
            z_next, log_prob_next = next_actions[name]

            # ── Critic update ──────────────────────────────────────────────────
            with torch.no_grad():
                q_next = agent.critic_target.min_q(joint_next, next_all_actions)  # (B,1)
                td_target = rewards[name] + self.gamma * (1 - dones) * (q_next - alpha_t * log_prob_next)

            q1, q2 = agent.critic(joint_obs, all_actions)
            critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
            agent.critic_opt.zero_grad()
            critic_loss.backward()
            agent.critic_opt.step()

            # ── Actor update ───────────────────────────────────────────────────
            z_curr, log_prob_curr = agent.actor.sample(obs)
            # Rebuild all_actions with this agent's freshly sampled action
            curr_all_actions = self._replace_agent_action(all_actions, z_curr, name)
            q_curr = agent.critic.min_q(joint_obs, curr_all_actions)
            actor_loss = (alpha_t * log_prob_curr - q_curr).mean()
            agent.actor_opt.zero_grad()
            actor_loss.backward()
            agent.actor_opt.step()

            # ── Temperature update ─────────────────────────────────────────────
            alpha_loss = -(agent.log_alpha_t * (log_prob_curr + agent.target_entropy).detach()).mean()
            agent.alpha_t_opt.zero_grad()
            alpha_loss.backward()
            agent.alpha_t_opt.step()

            metrics_dict[name] = {
                "critic_loss": critic_loss.item(),
                "actor_loss":  actor_loss.item(),
                "alpha_t":     agent.log_alpha_t.exp().item(),
                "entropy":     -log_prob_curr.mean().item(),
            }

        # ── Soft target update ─────────────────────────────────────────────────
        for agent in self.agents.values():
            for p, p_tgt in zip(agent.critic.parameters(), agent.critic_target.parameters()):
                p_tgt.data.mul_(1 - self.tau).add_(p.data * self.tau)

        return UpdateMetrics(
            loss_critic_bloomberg=metrics_dict["bloomberg"]["critic_loss"],
            loss_critic_lesg     =metrics_dict["lesg"]["critic_loss"],
            loss_critic_financial=metrics_dict["financial"]["critic_loss"],
            loss_actor_bloomberg =metrics_dict["bloomberg"]["actor_loss"],
            loss_actor_lesg      =metrics_dict["lesg"]["actor_loss"],
            loss_actor_financial =metrics_dict["financial"]["actor_loss"],
            alpha_t_bloomberg    =metrics_dict["bloomberg"]["alpha_t"],
            alpha_t_lesg         =metrics_dict["lesg"]["alpha_t"],
            alpha_t_financial    =metrics_dict["financial"]["alpha_t"],
            entropy_bloomberg    =metrics_dict["bloomberg"]["entropy"],
            entropy_lesg         =metrics_dict["lesg"]["entropy"],
            entropy_financial    =metrics_dict["financial"]["entropy"],
        )

    def _replace_agent_action(
        self,
        all_actions: torch.Tensor,    # (B, 3N)
        new_action: torch.Tensor,     # (B, N)
        agent_name: str,
    ) -> torch.Tensor:
        """Swap one agent's slice in the joint action tensor."""
        idx = AGENT_NAMES.index(agent_name)
        updated = all_actions.clone()
        updated[:, idx * self.action_dim : (idx + 1) * self.action_dim] = new_action
        return updated

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)
        for name, agent in self.agents.items():
            torch.save({
                "actor": agent.actor.state_dict(),
                "critic": agent.critic.state_dict(),
                "critic_target": agent.critic_target.state_dict(),
                "log_alpha_t": agent.log_alpha_t.data,
            }, f"{path}/{name}.pt")

    def load(self, path: str) -> None:
        for name, agent in self.agents.items():
            ckpt = torch.load(f"{path}/{name}.pt", map_location=self.device)
            agent.actor.load_state_dict(ckpt["actor"])
            agent.critic.load_state_dict(ckpt["critic"])
            agent.critic_target.load_state_dict(ckpt["critic_target"])
            agent.log_alpha_t.data = ckpt["log_alpha_t"]
