"""
FinancialAgent — ADK agent wrapping the pure-financial MASAC actor.

Private reward signal: rₜ  (α₃ ≈ 0, negligible ESG bias)
Acts as the pure financial performance anchor in multi-agent negotiation.
Tool: compute_financial_scores(state_vector) → allocation score vector z^(F)
"""
from __future__ import annotations
import numpy as np
from google.adk.tools import FunctionTool

from app.agents.base import BasePortfolioAgent


FINANCIAL_INSTRUCTIONS = """
You are the Financial Performance Agent in a multi-agent reinforcement learning
portfolio optimization system. Your sole mandate is to maximize raw portfolio
simple returns — you have negligible ESG bias (α₃ ≈ 0).

Your reward signal is approximately:
  R ≈ rₜ  (pure financial return)

You act as the financial anchor in the multi-agent negotiation. Your scores push
toward the highest-returning assets regardless of ESG profile. This creates
productive tension with the Bloomberg and LESG agents, who prioritize ESG ratings.

In Portfolio B, you are source-agnostic:
  R = rₜ  (the tie-breaking financial signal when ESG agents disagree perfectly)

In Cooperative Portfolio C, you still receive the shared ESG ambiguity penalty:
  R = rₜ − β · ΔESGₜ

Your influence in the joint score z_joint = (z^B + z^L + z^F) / 3 is the
primary driver when ESG signals cancel out.
"""


class FinancialAgent(BasePortfolioAgent):
    """
    ADK agent for pure financial return maximization.
    Registered tool: compute_financial_scores
    """

    def __init__(self, masac_actor=None) -> None:
        def compute_financial_scores(state_vector: list[float]) -> dict:
            """
            Compute allocation scores from the Financial agent's perspective.

            Args:
                state_vector: 10N-dimensional normalized observation vector.

            Returns:
                dict with 'scores' (list[float], N-dim), 'agent' (str), 'rationale' (str)
            """
            obs = np.array(state_vector, dtype=np.float32)
            z = self._actor_forward(obs)
            return {
                "scores": z.tolist(),
                "agent": "financial",
                "rationale": "Scores reflect pure financial return maximization",
            }

        super().__init__(
            name="financial_agent",
            masac_actor=masac_actor,
            model="gemini-2.0-flash",
            instruction=FINANCIAL_INSTRUCTIONS,
            tools=[FunctionTool(compute_financial_scores)],
        )
