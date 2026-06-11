"""Per-iteration rollout buffer with GAE for PPO.

Stores transitions for T timesteps × N parallel worlds, computes advantages
via GAE-λ on the device, and yields shuffled minibatches for the update.

Episodes are fixed-length (always T): there is no done flag — every
trajectory terminates at the simulation horizon, so bootstrap value is the
critic's prediction at state[T].
"""
from __future__ import annotations

from typing import Iterator

import torch


class RolloutBuffer:
    def __init__(self, T: int, N: int, obs_dim: int, act_dim: int, device: torch.device):
        self.T = T
        self.N = N
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

        # All buffers are [T, N, ...] except advantages/returns which are filled later.
        self.obs       = torch.zeros(T, N, obs_dim, device=device, dtype=torch.float32)
        self.actions   = torch.zeros(T, N, act_dim, device=device, dtype=torch.float32)
        self.logp      = torch.zeros(T, N, device=device, dtype=torch.float32)
        self.values    = torch.zeros(T, N, device=device, dtype=torch.float32)
        self.rewards   = torch.zeros(T, N, device=device, dtype=torch.float32)
        self.advantages = torch.zeros(T, N, device=device, dtype=torch.float32)
        self.returns    = torch.zeros(T, N, device=device, dtype=torch.float32)

        # Helhest-specific scratch: chassis pose (x, y, z, pitch) per (T+1) per world,
        # used for video rendering and final-state diagnostics. Filled by the trainer.
        self.chassis_xyz_pitch = torch.zeros(T + 1, N, 4, device=device, dtype=torch.float32)

    def add(self, t: int, obs, action, logp, value, reward):
        self.obs[t]      = obs
        self.actions[t]  = action
        self.logp[t]     = logp
        self.values[t]   = value
        self.rewards[t]  = reward

    def compute_gae(self, last_value: torch.Tensor, gamma: float, lam: float) -> None:
        """GAE-λ on a fixed-horizon, no-done rollout. ``last_value`` is V(s_T)."""
        adv = torch.zeros(self.N, device=self.device, dtype=torch.float32)
        next_v = last_value
        for t in reversed(range(self.T)):
            delta = self.rewards[t] + gamma * next_v - self.values[t]
            adv = delta + gamma * lam * adv
            self.advantages[t] = adv
            next_v = self.values[t]
        self.returns = self.advantages + self.values

    def minibatches(self, batch_size: int, normalize_adv: bool = True) -> Iterator[dict]:
        TN = self.T * self.N

        obs   = self.obs.reshape(TN, self.obs_dim)
        act   = self.actions.reshape(TN, self.act_dim)
        logp  = self.logp.reshape(TN)
        vals  = self.values.reshape(TN)
        rets  = self.returns.reshape(TN)
        advs  = self.advantages.reshape(TN)

        if normalize_adv:
            advs = (advs - advs.mean()) / (advs.std().clamp_min(1e-8))

        idx = torch.randperm(TN, device=self.device)
        for start in range(0, TN, batch_size):
            sel = idx[start : start + batch_size]
            yield {
                "obs":        obs[sel],
                "actions":    act[sel],
                "logp_old":   logp[sel],
                "values_old": vals[sel],
                "returns":    rets[sel],
                "advantages": advs[sel],
            }
