"""
LESGAgent — ADK agent wrapping the LESG-perspective MASAC actor.

Private reward signal: rₜ + α₂ · ESG_t^(L)  [− β · ΔESGₜ in cooperative/mixed]
Tool: compute_lesg_scores(state_vector) → allocation score vector z^(L)
"""
from __future__ import annotations
import numpy as np
from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from app.agents.base import BasePortfolioAgent


LESG_INSTRUCTIONS = """
You are the LESG ESG Portfolio Agent in a multi-agent reinforcement learning
system. Your mandate is to maximize portfolio returns weighted by LESG ESG scores.

You are biased toward assets with high LESG ESG ratings (scale 0-10, normalized
cross-sectionally to [0,1]). Your reward signal is:
  R = r_t + α₂ · ESG_t^(LESG)

In Cooperative mode, you also bear a shared penalty for ESG agency disagreement:
  R = r_t + α₂ · ESG_t^(LESG) − β · ΔESGₜ

Your output is a raw allocation score vector z^(L) ∈ ℝᴺ. You and the Bloomberg
agent often disagree — particularly on assets where the two rating agencies diverge.
In Portfolio B (signed disagreement), you directly oppose the Bloomberg agent:
  Your reward: rₜ + λ · (ESG_t^(L) − ESG_t^(B))
  Bloomberg's: rₜ + λ · (ESG_t^(B) − ESG_t^(L))

You reward stocks where LESG rates higher than Bloomberg; you penalize the reverse.
"""


class LESGAgent(BasePortfolioAgent):
    """
    ADK agent for the LESG ESG perspective.
    Registered tool: compute_lesg_scores
    """

    def __init__(self, masac_actor=None) -> None:
        def compute_lesg_scores(state_vector: list[float]) -> dict:
            """
            Compute allocation scores from the LESG ESG agent's perspective.

            Args:
                state_vector: 10N-dimensional normalized observation vector.

            Returns:
                dict with 'scores' (list[float], N-dim), 'agent' (str), 'rationale' (str)
            """
            obs = np.array(state_vector, dtype=np.float32)
            z = self._actor_forward(obs)
            return {
                "scores": z.tolist(),
                "agent": "lesg_esg",
                "rationale": "Scores reflect LESG ESG-weighted portfolio preference",
            }

        super().__init__(
            name="lesg_esg_agent",
            masac_actor=masac_actor,
            model="gemini-2.0-flash",
            instruction=LESG_INSTRUCTIONS,
            tools=[FunctionTool(compute_lesg_scores)],
        )
