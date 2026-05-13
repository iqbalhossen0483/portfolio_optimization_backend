"""
BloombergESGAgent — ADK agent wrapping the Bloomberg-perspective MASAC actor.

Private reward signal: rₜ + α₁ · ESG_t^(B)  [− β · ΔESGₜ in cooperative/mixed]
Tool: compute_bloomberg_scores(state_vector) → allocation score vector z^(B)
"""
from __future__ import annotations
import numpy as np
from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from app.agents.base import BasePortfolioAgent


BLOOMBERG_INSTRUCTIONS = """
You are the Bloomberg ESG Portfolio Agent in a multi-agent reinforcement learning
system. Your mandate is to maximize portfolio returns weighted by Bloomberg ESG scores.

You are biased toward assets with high Bloomberg ESG ratings (scale 0-100, normalized
cross-sectionally to [0,1]). Your reward signal is:
  R = r_t + α₁ · ESG_t^(Bloomberg)

In Cooperative mode, you also bear a shared penalty for ESG agency disagreement:
  R = r_t + α₁ · ESG_t^(Bloomberg) − β · ΔESGₜ

Your output is a raw allocation score vector z^(B) ∈ ℝᴺ. Higher scores for
assets you favor. Scores are averaged with the LESG and Financial agents and
passed through Softmax to produce the final portfolio weights.

When you compute scores, consider:
1. Bloomberg ESG normalized score (cross-sectional peer ranking today)
2. Momentum indicators (RSI, MACD histogram)
3. Per-stock ESG disagreement ΔESG (avoid allocating to high-ambiguity stocks
   if you are in a cooperative game)
4. Individual asset returns Rᵢₜ
"""


def make_bloomberg_agent(masac_actor=None) -> "BloombergESGAgent":
    return BloombergESGAgent(masac_actor=masac_actor)


class BloombergESGAgent(BasePortfolioAgent):
    """
    ADK agent for the Bloomberg ESG perspective.
    Registered tool: compute_bloomberg_scores
    """

    def __init__(self, masac_actor=None) -> None:
        def compute_bloomberg_scores(state_vector: list[float]) -> dict:
            """
            Compute allocation scores from the Bloomberg ESG agent's perspective.

            Args:
                state_vector: 10N-dimensional normalized observation vector.
                              [OHLCV(5N), RSI(N), MACD(N), Returns(N), ΔESG(N), μESG(N)]

            Returns:
                dict with 'scores' (list[float], N-dim), 'agent' (str), 'rationale' (str)
            """
            obs = np.array(state_vector, dtype=np.float32)
            z = self._actor_forward(obs)
            return {
                "scores": z.tolist(),
                "agent": "bloomberg_esg",
                "rationale": "Scores reflect Bloomberg ESG-weighted portfolio preference",
            }

        super().__init__(
            name="bloomberg_esg_agent",
            masac_actor=masac_actor,
            model="gemini-2.0-flash",
            instruction=BLOOMBERG_INSTRUCTIONS,
            tools=[FunctionTool(compute_bloomberg_scores)],
        )
