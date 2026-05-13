"""
TrainingOrchestrator — drives the MASAC training loop.

Training parameters (from spec):
  - Max steps: 500,000
  - Episode length: 252 trading days
  - Warmup: 10,000 steps (random actions, no gradient updates)
  - Batch size: 256
  - Convergence: rolling std of mean entropy over 100 steps < 0.01
  - Gradient updates: 1 per environment step (after warmup)
  - Emits step events to Redis PubSub for WebSocket streaming

For each topology, a separate Trainer + MASAC instance is created.
"""
from __future__ import annotations
import asyncio
import json
import os
import time
from collections import deque
from typing import AsyncIterator

import numpy as np
import structlog

from app.rl.environment import MarketEnvironment
from app.rl.masac import MASAC
from app.data.pipeline import ProcessedDataset
from app.config import get_settings

log = structlog.get_logger(__name__)
cfg = get_settings()


class TrainingOrchestrator:
    """
    Runs the MASAC training loop for a single (topology, portfolio_model) configuration.
    Publishes step metrics to Redis channel `pubsub:training:{job_id}`.
    """

    def __init__(
        self,
        job_id: str,
        dataset: ProcessedDataset,
        portfolio_model: str,
        topology: str,
        hyperparams: dict,
        model_store_path: str,
        redis_client=None,
    ) -> None:
        self.job_id  = job_id
        self.dataset = dataset
        self.topology = topology
        self.portfolio_model = portfolio_model
        self.hyperparams = hyperparams
        self.model_store_path = model_store_path
        self._redis = redis_client
        self._stop_requested = False

        self.env = MarketEnvironment(
            dataset=dataset,
            portfolio_model=portfolio_model,
            topology=topology,
            **{k: hyperparams[k] for k in ("alpha_1", "alpha_2", "alpha_3", "beta", "lam")
               if k in hyperparams},
        )

        self.masac = MASAC(
            n_assets=dataset.n_assets,
            gamma=cfg.masac_gamma,
            tau=cfg.masac_tau,
            hidden=cfg.masac_hidden_size,
            lr_actor=cfg.masac_lr_actor,
            lr_critic=cfg.masac_lr_critic,
            lr_alpha=cfg.masac_lr_alpha,
            initial_alpha_t=cfg.masac_initial_alpha_t,
            buffer_capacity=cfg.masac_buffer_capacity,
            batch_size=cfg.masac_batch_size,
        )

    async def run(self) -> dict:
        """
        Main training loop. Returns final metrics dict on completion.
        """
        entropy_window: deque[float] = deque(maxlen=cfg.masac_convergence_window)
        global_step = 0
        best_sharpe: float | None = None
        best_mu_esg: float | None = None
        start_time = time.time()

        obs = self.env.reset()

        log.info("Training started", job_id=self.job_id, topology=self.topology,
                 portfolio_model=self.portfolio_model, n_assets=self.dataset.n_assets)

        while global_step < cfg.masac_max_steps and not self._stop_requested:
            # ── Collect transition ─────────────────────────────────────────────
            warmup = global_step < cfg.masac_warmup_steps
            if warmup:
                # Random actions during warmup
                actions = {
                    "bloomberg": np.random.randn(self.dataset.n_assets).astype(np.float32),
                    "lesg":      np.random.randn(self.dataset.n_assets).astype(np.float32),
                    "financial": np.random.randn(self.dataset.n_assets).astype(np.float32),
                }
            else:
                actions = self.masac.select_actions(obs, deterministic=False)

            result = self.env.step(
                actions["bloomberg"], actions["lesg"], actions["financial"]
            )

            self.masac.buffer.add(
                obs=obs,
                action_B=actions["bloomberg"],
                action_L=actions["lesg"],
                action_F=actions["financial"],
                reward_B=result.rewards["bloomberg"],
                reward_L=result.rewards["lesg"],
                reward_F=result.rewards["financial"],
                next_obs=result.obs,
                done=result.done,
            )

            obs = result.obs if not result.done else self.env.reset()

            # ── Gradient update ────────────────────────────────────────────────
            if not warmup:
                metrics = self.masac.update()

                if metrics is not None:
                    mean_entropy = (
                        metrics.entropy_bloomberg
                        + metrics.entropy_lesg
                        + metrics.entropy_financial
                    ) / 3.0
                    entropy_window.append(mean_entropy)

                    # ── Convergence check ──────────────────────────────────────
                    if len(entropy_window) == cfg.masac_convergence_window:
                        rolling_std = float(np.std(entropy_window))
                        if rolling_std < cfg.masac_convergence_epsilon:
                            log.info("Converged", step=global_step, rolling_std=rolling_std)
                            await self._publish_converged(global_step, best_sharpe, best_mu_esg)
                            break

                    # ── Periodic publishing / checkpointing ───────────────────
                    if global_step % 500 == 0:
                        rolling_std = float(np.std(entropy_window)) if entropy_window else 0.0
                        await self._publish_step(global_step, metrics, rolling_std, result)

                    if global_step % 10_000 == 0:
                        sharpe, mu_esg = self._eval_validation()
                        if best_sharpe is None or sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_mu_esg = mu_esg
                            ckpt_path = os.path.join(
                                self.model_store_path, self.job_id, self.topology
                            )
                            self.masac.save(ckpt_path)
                            log.info("Checkpoint saved", step=global_step,
                                     sharpe=sharpe, mu_esg=mu_esg)

            global_step += 1

        elapsed = time.time() - start_time
        final = {
            "job_id": self.job_id,
            "topology": self.topology,
            "portfolio_model": self.portfolio_model,
            "steps_completed": global_step,
            "best_sharpe": best_sharpe,
            "best_mu_esg": best_mu_esg,
            "elapsed_seconds": elapsed,
        }
        log.info("Training complete", **final)
        return final

    def stop(self) -> None:
        self._stop_requested = True

    # ── Validation evaluation ─────────────────────────────────────────────────

    def _eval_validation(self) -> tuple[float, float]:
        """
        Quick 63-day rolling validation using deterministic actions.
        Returns (Sharpe, mean μESG) — display metrics only.
        """
        obs = self.env.dataset.state_vectors[0].copy()
        returns_list, mu_esg_list = [], []

        for t in range(min(cfg.validation_window_days, self.env.dataset.n_timesteps - 1)):
            actions = self.masac.select_actions(obs, deterministic=True)
            result = self.env.step(
                actions["bloomberg"], actions["lesg"], actions["financial"]
            )
            r_t = result.info["r_t"]
            returns_list.append(r_t)
            mu_esg_list.append(float(np.mean(result.info["weights"])))  # simplified
            obs = result.obs
            if result.done:
                break

        r_arr = np.array(returns_list)
        annualized_return = r_arr.mean() * 252
        annualized_std    = r_arr.std() * np.sqrt(252)
        sharpe = annualized_return / (annualized_std + 1e-8)
        mu_esg = float(np.mean(mu_esg_list))
        return float(sharpe), mu_esg

    # ── Redis publishing ──────────────────────────────────────────────────────

    async def _publish_step(self, step: int, metrics, rolling_std: float, result) -> None:
        if not self._redis:
            return
        payload = {
            "type": "step",
            "step": step,
            "entropy": (metrics.entropy_bloomberg + metrics.entropy_lesg + metrics.entropy_financial) / 3,
            "entropy_rolling_std": rolling_std,
            "reward_bloomberg": result.rewards["bloomberg"],
            "reward_lesg":      result.rewards["lesg"],
            "reward_financial": result.rewards["financial"],
            "loss_actor": (metrics.loss_actor_bloomberg + metrics.loss_actor_lesg + metrics.loss_actor_financial) / 3,
            "loss_critic": (metrics.loss_critic_bloomberg + metrics.loss_critic_lesg + metrics.loss_critic_financial) / 3,
            "alpha_t": metrics.alpha_t_bloomberg,
        }
        await self._redis.publish(f"pubsub:training:{self.job_id}", json.dumps(payload))

    async def _publish_converged(self, step: int, sharpe: float | None, mu_esg: float | None) -> None:
        if not self._redis:
            return
        payload = {
            "type": "converged",
            "step": step,
            "final_sharpe": sharpe or 0.0,
            "mu_esg": mu_esg or 0.0,
            "message": f"Training converged at step {step}",
        }
        await self._redis.publish(f"pubsub:training:{self.job_id}", json.dumps(payload))
