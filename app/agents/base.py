"""
Base ADK agent for all portfolio agents.
Wraps a MASAC Actor with ADK tool definitions.
Each concrete agent registers its compute_scores tool in its subclass.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import structlog

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

log = structlog.get_logger(__name__)


class BasePortfolioAgent(Agent):
    """
    Base class providing shared tools available to all three portfolio agents:
      - log_decision: structured logging of agent allocations
    Subclasses add their own compute_scores tool and actor reference.
    """

    def __init__(self, name: str, masac_actor=None, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self._masac_actor = masac_actor   # loaded PyTorch ActorNetwork

    def set_actor(self, actor) -> None:
        """Inject a trained actor after loading from model store."""
        self._masac_actor = actor

    def _actor_forward(self, obs: np.ndarray) -> np.ndarray:
        """Run inference through the actor; returns N-dim score vector."""
        import torch
        if self._masac_actor is None:
            raise RuntimeError(f"Actor not loaded for agent {self.name}")
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            z = self._masac_actor.deterministic_action(obs_t)
        return z.squeeze(0).numpy()

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        e = np.exp(z - z.max())
        return e / e.sum()
