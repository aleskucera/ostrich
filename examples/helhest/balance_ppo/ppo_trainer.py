"""PPO clip-loss update step.

Standard PPO with clipped surrogate, clipped value loss, entropy bonus,
and global gradient clipping. Returns aggregated metrics for wandb.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

from examples.helhest.balance_ppo.actor_critic import ActorCritic
from examples.helhest.balance_ppo.rollout_buffer import RolloutBuffer


@dataclass
class PPOConfig:
    epochs: int = 10
    minibatch_size: int = 256
    clip_ratio: float = 0.2
    clip_value: float = 0.2     # set <=0 to disable value clipping
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02   # early stop when mean KL exceeds this; None = never


class PPOTrainer:
    def __init__(self, policy: ActorCritic, optimizer: torch.optim.Optimizer, config: PPOConfig):
        self.policy = policy
        self.optim = optimizer
        self.cfg = config

    def update(self, buffer: RolloutBuffer) -> dict:
        cfg = self.cfg
        accum = {
            "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
            "approx_kl": 0.0, "clip_fraction": 0.0, "grad_norm": 0.0,
            "n_updates": 0,
        }
        early_stop_epoch = cfg.epochs

        for epoch in range(cfg.epochs):
            epoch_kls = []
            for batch in buffer.minibatches(cfg.minibatch_size):
                logp_new, entropy, values_new = self.policy.evaluate(
                    batch["obs"], batch["actions"]
                )
                ratio = torch.exp(logp_new - batch["logp_old"])
                adv = batch["advantages"]

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                if cfg.clip_value > 0:
                    v_clipped = batch["values_old"] + torch.clamp(
                        values_new - batch["values_old"],
                        -cfg.clip_value, cfg.clip_value,
                    )
                    v_loss_unclipped = (values_new - batch["returns"]).pow(2)
                    v_loss_clipped = (v_clipped - batch["returns"]).pow(2)
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    value_loss = 0.5 * (values_new - batch["returns"]).pow(2).mean()

                entropy_loss = -entropy.mean()
                loss = policy_loss + cfg.vf_coef * value_loss + cfg.ent_coef * entropy_loss

                self.optim.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optim.step()

                with torch.no_grad():
                    approx_kl = (batch["logp_old"] - logp_new).mean().item()
                    clipped = ((ratio - 1.0).abs() > cfg.clip_ratio).float().mean().item()

                accum["policy_loss"]   += policy_loss.item()
                accum["value_loss"]    += value_loss.item()
                accum["entropy"]       += -entropy_loss.item()
                accum["approx_kl"]     += approx_kl
                accum["clip_fraction"] += clipped
                accum["grad_norm"]     += float(grad_norm)
                accum["n_updates"]     += 1
                epoch_kls.append(approx_kl)

            if cfg.target_kl is not None and len(epoch_kls):
                if (sum(epoch_kls) / len(epoch_kls)) > 1.5 * cfg.target_kl:
                    early_stop_epoch = epoch + 1
                    break

        n = max(accum["n_updates"], 1)
        return {
            "policy_loss":   accum["policy_loss"] / n,
            "value_loss":    accum["value_loss"] / n,
            "entropy":       accum["entropy"] / n,
            "approx_kl":     accum["approx_kl"] / n,
            "clip_fraction": accum["clip_fraction"] / n,
            "grad_norm":     accum["grad_norm"] / n,
            "epochs_run":    early_stop_epoch,
        }


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    """1 - Var[returns - values] / Var[returns]; 1.0 = perfect, 0.0 = predicting mean,
    <0 = worse than predicting mean."""
    var_returns = returns.var()
    if var_returns < 1e-8:
        return float("nan")
    return float(1.0 - (returns - values).var() / var_returns)
