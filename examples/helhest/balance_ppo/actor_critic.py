"""Actor-critic MLP for the Helhest balance PPO trainer.

Architecture follows the MuJoCo-PPO defaults: separate trunks for actor and
critic, tanh activations, state-independent log-std, orthogonal init with a
small final-layer gain so the initial policy is near-deterministic-zero.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal


def _orthogonal_init(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int = 64,
        v_max: float = 8.0,
        log_std_init: float = -0.5,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.v_max = v_max

        self.actor_trunk = nn.Sequential(
            _orthogonal_init(nn.Linear(obs_dim, hidden), gain=math.sqrt(2)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden, hidden), gain=math.sqrt(2)),
            nn.Tanh(),
        )
        self.actor_head = _orthogonal_init(nn.Linear(hidden, act_dim), gain=0.01)

        self.critic_trunk = nn.Sequential(
            _orthogonal_init(nn.Linear(obs_dim, hidden), gain=math.sqrt(2)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden, hidden), gain=math.sqrt(2)),
            nn.Tanh(),
        )
        self.critic_head = _orthogonal_init(nn.Linear(hidden, 1), gain=1.0)

        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def _dist(self, obs: torch.Tensor) -> Normal:
        mean = self.actor_head(self.actor_trunk(obs))
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.critic_trunk(obs)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        """Sample stochastic action; return raw sample (not yet squashed) + logp + value.

        The squash to wheel velocities (tanh × v_max) is applied by the trainer
        when writing into the sim, not here — this keeps the PPO ratio math
        operating on the raw Gaussian sample (no Jacobian correction needed).
        """
        dist = self._dist(obs)
        raw = dist.sample()
        logp = dist.log_prob(raw).sum(-1)
        value = self.value(obs)
        return raw, logp, value

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_head(self.actor_trunk(obs))

    def evaluate(self, obs: torch.Tensor, raw_action: torch.Tensor):
        """Score an existing raw action under the current policy (for PPO update)."""
        dist = self._dist(obs)
        logp = dist.log_prob(raw_action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.value(obs)
        return logp, entropy, value

    def squash(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Map raw Gaussian sample into wheel-velocity targets in [-v_max, v_max]."""
        return self.v_max * torch.tanh(raw_action)
