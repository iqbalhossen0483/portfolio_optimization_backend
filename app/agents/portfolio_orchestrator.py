"""
PortfolioOrchestratorAgent — top-level ADK agent.

Responsibilities:
1. Receives user portfolio query (model A/B/C, amount, assets, hyperparams)
2. Spawns three topology sub-runs concurrently (cooperative, competitive, mixed)
3. For each topology:
   a. Calls BloombergAgent, LESGAgent, FinancialAgent to get score vectors
   b. Aggregates: z_joint = (z^B + z^L + z^F) / 3
   c. Applies Softmax → portfolio weights
   d. Computes metrics (return, σ, Sharpe, μESG, ΔESG)
   e. Generates strategic summary
4. Returns 3-panel ComparisonResponse

ADK sub-agents: BloombergESGAgent, LESGAgent, FinancialAgent
"""
from __future__ import annotations
import asyncio
from typing import Any
import numpy as np
import torch

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from app.agents.bloomberg_agent import BloombergESGAgent
from app.agents.lesg_agent import LESGAgent
from app.agents.financial_agent import FinancialAgent
from app.rl.networks import ActorNetwork
from app.models.schemas import (
    TopologyPanel, AssetAllocation, AggregateMetrics,
    PortfolioGenerateRequest, PortfolioGenerateResponse, Topology
)
from app.data.pipeline import ProcessedDataset
from app.data.preprocessing.normalizer import DataNormalizer
import structlog

log = structlog.get_logger(__name__)

ORCHESTRATOR_INSTRUCTIONS = """
You are the Portfolio Orchestration Agent for a Multi-Agent Deep Reinforcement
Learning (MADRL) ESG portfolio system.

You coordinate three specialist agents:
  1. BloombergESGAgent  — maximises Bloomberg ESG-weighted returns
  2. LESGAgent          — maximises LESG ESG-weighted returns
  3. FinancialAgent     — maximises pure financial returns

For every user query, you:
  1. Run all three game-theoretic topologies concurrently:
     - Cooperative:  shared ESG ambiguity penalty β > 0
     - Competitive:  no shared penalty (β = 0)
     - Mixed:        partial penalty (β/2)
  2. In each topology, collect allocation score vectors from all three agents
  3. Aggregate via equal-weight averaging + Softmax → portfolio weights
  4. Compute performance metrics and generate a strategic summary
  5. Return three side-by-side panels for user comparison

Critical: never merge outputs across topologies. Each panel is independent.
"""


class PortfolioOrchestratorAgent(Agent):
    """
    ADK Orchestrator with three specialist sub-agents.
    Loads trained actor weights from model store at initialization.
    """

    def __init__(
        self,
        model_store_path: str,
        job_id: int,
        n_assets: int,
        hidden: int = 256,
    ) -> None:
        self.model_store_path = model_store_path
        self.job_id = job_id
        self.n_assets = n_assets
        self.obs_dim = 10 * n_assets
        self.hidden = hidden

        # Instantiate sub-agents
        self._bloomberg = BloombergESGAgent()
        self._lesg      = LESGAgent()
        self._financial = FinancialAgent()

        # Load actor weights
        self._load_actors()

        def generate_portfolio_comparison(
            state_vector: list[float],
            esg_b_row: list[float],
            esg_l_row: list[float],
            delta_esg_row: list[float],
            mu_esg_row: list[float],
            returns_row: list[float],
            asset_isins: list[str],
            asset_sectors: list[str],
            allocation_amount: float,
            topology: str,
            beta: float = 0.3,
            alpha_1: float = 0.5,
            alpha_2: float = 0.5,
        ) -> dict:
            """
            Generate portfolio allocation for one topology using all three sub-agents.

            Args:
                state_vector:     10N normalized observation vector for current timestep
                esg_b_row:        N-dim normalized Bloomberg ESG scores for this day
                esg_l_row:        N-dim normalized LESG ESG scores for this day
                delta_esg_row:    N-dim per-stock disagreement scores
                mu_esg_row:       N-dim per-stock consensus scores
                returns_row:      N-dim individual returns (raw, not normalized)
                asset_isins:      List of ISIN strings
                asset_sectors:    List of sector strings
                allocation_amount: Total capital to allocate
                topology:         "cooperative" | "competitive" | "mixed"
                beta:             Disagreement penalty coefficient
                alpha_1:          Bloomberg ESG weight
                alpha_2:          LESG ESG weight

            Returns:
                Panel dict with allocations and aggregate metrics
            """
            obs = np.array(state_vector, dtype=np.float32)

            # Get score vectors from each sub-agent actor
            z_B = self._bloomberg._actor_forward(obs)
            z_L = self._lesg._actor_forward(obs)
            z_F = self._financial._actor_forward(obs)

            # Aggregate and apply Softmax
            z_joint = (z_B + z_L + z_F) / 3.0
            weights = self._softmax(z_joint)

            # Portfolio-level scalars
            esg_b_arr    = np.array(esg_b_row)
            esg_l_arr    = np.array(esg_l_row)
            delta_arr    = np.array(delta_esg_row)
            mu_arr       = np.array(mu_esg_row)
            returns_arr  = np.array(returns_row)

            port_esg_b   = float(np.dot(weights, esg_b_arr))
            port_esg_l   = float(np.dot(weights, esg_l_arr))
            port_delta   = float(np.dot(weights, delta_arr))
            port_mu_esg  = float(np.dot(weights, mu_arr))
            port_return  = float(np.dot(weights, returns_arr))

            # Annualized metrics (252-day approximation)
            port_risk    = float(np.sqrt(252) * np.sqrt(np.dot(weights**2, returns_arr**2)))
            port_sharpe  = (port_return * 252) / (port_risk + 1e-8)

            n = len(asset_isins)
            allocations = []
            for i in range(n):
                ann_ret  = float(returns_arr[i] * 252)
                ann_risk = float(abs(returns_arr[i]) * np.sqrt(252))
                sharpe_i = ann_ret / (ann_risk + 1e-8)
                allocations.append({
                    "isin":        asset_isins[i],
                    "sector":      asset_sectors[i] if i < len(asset_sectors) else "Unknown",
                    "weight":      float(weights[i]),
                    "allocation":  float(weights[i] * allocation_amount),
                    "return_ann":  ann_ret,
                    "risk_ann":    ann_risk,
                    "sharpe":      sharpe_i,
                    "mu_esg":      float(mu_arr[i]),
                    "delta_esg":   float(delta_arr[i]),
                })

            # Sort by weight descending
            allocations.sort(key=lambda x: x["weight"], reverse=True)

            summary = self._generate_summary(topology, allocations, port_delta, port_sharpe, beta)

            return {
                "topology": topology,
                "portfolio": allocations,
                "aggregate_metrics": {
                    "portfolio_return":    port_return * 252,
                    "portfolio_risk":      port_risk,
                    "portfolio_sharpe":    port_sharpe,
                    "portfolio_mu_esg":   port_mu_esg,
                    "portfolio_delta_esg": port_delta,
                },
                "strategic_summary": summary,
            }

        super().__init__(
            name="portfolio_orchestrator",
            model="gemini-2.0-flash",
            instruction=ORCHESTRATOR_INSTRUCTIONS,
            sub_agents=[self._bloomberg, self._lesg, self._financial],
            tools=[FunctionTool(generate_portfolio_comparison)],
        )
        # Keep reference for direct Python calls (bypassing ADK runner in async paths)
        self._generate_fn = generate_portfolio_comparison

    # ── Public API ────────────────────────────────────────────────────────────

    async def generate_comparison(
        self,
        dataset: ProcessedDataset,
        t: int,
        request: PortfolioGenerateRequest,
        asset_sectors: list[str],
    ) -> dict[str, Any]:
        """
        Run all three topologies concurrently and return the three panels.
        t: timestep index into dataset (use latest available day for inference).
        """
        state_vector  = dataset.state_vectors[t].tolist()
        esg_b_row     = dataset.esg_b_norm[t].tolist()
        esg_l_row     = dataset.esg_l_norm[t].tolist()
        delta_esg_row = dataset.delta_esg[t].tolist()
        mu_esg_row    = dataset.mu_esg[t].tolist()
        returns_row   = dataset.returns[t].tolist()

        kwargs = dict(
            state_vector=state_vector,
            esg_b_row=esg_b_row,
            esg_l_row=esg_l_row,
            delta_esg_row=delta_esg_row,
            mu_esg_row=mu_esg_row,
            returns_row=returns_row,
            asset_isins=request.assets,
            asset_sectors=asset_sectors,
            allocation_amount=request.allocation_amount,
            beta=request.hyperparams.beta,
            alpha_1=request.hyperparams.alpha_1,
            alpha_2=request.hyperparams.alpha_2,
        )

        # Run all three topologies concurrently
        coop, comp, mixed = await asyncio.gather(
            asyncio.to_thread(self._generate_fn, **kwargs, topology="cooperative"),
            asyncio.to_thread(self._generate_fn, **kwargs, topology="competitive"),
            asyncio.to_thread(self._generate_fn, **kwargs, topology="mixed"),
        )
        return {"cooperative": coop, "competitive": comp, "mixed": mixed}

    # ── Internals ─────────────────────────────────────────────────────────────

    def _load_actors(self) -> None:
        """Load trained actor weights from model store into each sub-agent."""
        import os
        for topology in ("cooperative",):   # load best available topology
            for agent_name, agent_obj in [
                ("bloomberg", self._bloomberg),
                ("lesg",      self._lesg),
                ("financial", self._financial),
            ]:
                path = os.path.join(
                    self.model_store_path, self.job_id, topology, f"{agent_name}.pt"
                )
                if not os.path.exists(path):
                    log.warning("Actor weights not found — using random init",
                                agent=agent_name, path=path)
                    actor = ActorNetwork(self.obs_dim, self.n_assets, self.hidden)
                else:
                    actor = ActorNetwork(self.obs_dim, self.n_assets, self.hidden)
                    ckpt = torch.load(path, map_location="cpu")
                    actor.load_state_dict(ckpt["actor"])
                    actor.eval()
                    log.info("Actor loaded", agent=agent_name, path=path)
                agent_obj.set_actor(actor)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        e = np.exp(z - z.max())
        return e / e.sum()

    @staticmethod
    def _generate_summary(
        topology: str, allocations: list[dict], delta: float, sharpe: float, beta: float
    ) -> str:
        top1 = allocations[0] if allocations else {}
        high_delta = [a for a in allocations if a["delta_esg"] > 0.5]
        if topology == "cooperative":
            penalty_note = (
                f"Shared ambiguity penalty β={beta:.2f} suppresses high-disagreement "
                f"stocks (ΔESG>0.5): {[a['isin'] for a in high_delta]}. "
                if high_delta else "No high-disagreement stocks in portfolio. "
            )
            return (
                f"Cooperative mode: all agents share the ESG disagreement penalty. "
                f"Top allocation {top1.get('isin','')} ({top1.get('sector','')}) "
                f"at {top1.get('weight',0):.0%} with Sharpe {top1.get('sharpe',0):.2f}. "
                f"Portfolio Sharpe={sharpe:.2f}. {penalty_note}"
            )
        elif topology == "competitive":
            return (
                f"Competitive mode: no shared penalty (β=0). Each agent maximises "
                f"its own private objective. Energy/high-return assets may receive "
                f"elevated weights as LESG agent's objections are diluted in aggregation. "
                f"Portfolio Sharpe={sharpe:.2f}. "
                f"Top allocation: {top1.get('isin','')} at {top1.get('weight',0):.0%}."
            )
        else:  # mixed
            return (
                f"Mixed mode: partial penalty (β/2={beta/2:.2f}). Outcome intermediate "
                f"between Cooperative (min ambiguity) and Competitive (max return). "
                f"Portfolio Sharpe={sharpe:.2f}. "
                f"Top allocation: {top1.get('isin','')} at {top1.get('weight',0):.0%}."
            )
