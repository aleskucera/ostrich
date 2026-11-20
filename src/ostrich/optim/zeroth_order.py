"""Zeroth-order trajectory optimization with batched worlds.

The gradient of the Gaussian-smoothed loss F(theta) = E_eps[L(theta + sigma eps)]
is estimated from loss values alone (randomized smoothing, antithetic pairs),
so it needs no adjoint and includes the contribution of contact events, which
the exact gradient of L misses. On rough contact tasks (mesh terrain) this
optimizes far better than the hard adjoint; on smooth tasks the adjoint is as
good and much cheaper.

All 2 n_pairs + 1 perturbed plans run as parallel worlds of one model in a
single CUDA-graph rollout.

Two variance reductions (validated against a 128-pair reference on terrain,
~1.5-2x fewer samples for the same gradient quality):
  orthogonal directions  the n perturbation directions are mutually
                         orthogonal (same second moment as Gaussian ones)
  causal credit          a parameter only gets credit for loss terms it can
                         affect: with per-step losses l_t and the first step
                         s_p that parameter p influences, its estimate uses
                         sum_{t >= s_p} l_t ("reward-to-go")

Batched worlds must run fixed solver iteration counts: the engine's Newton and
linear loops stop only when ALL worlds have converged, which couples each
world's result to its neighbours (0.5-2.4% loss on the box task).
BatchedRollout enforces this via fixed_iteration_config.
"""
import dataclasses
from dataclasses import dataclass
from typing import Callable, Optional

import newton
import numpy as np
import warp as wp

from ..core.engine_config import OstrichEngineConfig
from ..core.logging_config import LoggingConfig


@dataclass
class ZeroOrderConfig:
    iterations: int = 100
    n_pairs: int = 16
    sigma: float = 0.3
    # geometric annealing sigma -> sigma_final over the iterations; None = fixed
    sigma_final: Optional[float] = None
    directions: str = "orth"  # "orth" | "iid"
    causal: bool = True
    lr: float = 0.1
    lr_min_ratio: float = 0.1  # cosine decay to lr * lr_min_ratio
    betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8
    seed: int = 0

    def __post_init__(self):
        if self.directions not in ("orth", "iid"):
            raise ValueError(f"directions must be 'orth' or 'iid', got {self.directions!r}")
        if self.n_pairs < 1 or self.iterations < 1:
            raise ValueError("n_pairs and iterations must be >= 1")
        if self.sigma <= 0 or (self.sigma_final is not None and self.sigma_final <= 0):
            raise ValueError("sigma and sigma_final must be > 0")

    @property
    def num_worlds(self) -> int:
        """Worlds a BatchedRollout needs: the plan itself plus 2 n_pairs perturbations."""
        return 2 * self.n_pairs + 1

    def sigma_at(self, it: int) -> float:
        if self.sigma_final is None:
            return self.sigma
        return self.sigma * (self.sigma_final / self.sigma) ** (it / max(self.iterations - 1, 1))


def sample_directions(rng: np.random.Generator, n: int, shape: tuple, kind: str = "orth") -> np.ndarray:
    """n perturbation directions of the given parameter shape, E[eps eps^T] = I.

    "orth": rows of random orthogonal matrices scaled to norm sqrt(d) (for
    n > d, independent orthogonal blocks). "iid": standard normal.
    """
    d = int(np.prod(shape))
    if kind == "iid":
        return rng.standard_normal((n, *shape))
    if kind != "orth":
        raise ValueError(kind)
    rows = []
    while len(rows) < n:
        q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        rows.extend(q.T[: n - len(rows)])
    return np.asarray(rows).reshape(n, *shape) * np.sqrt(d)


def estimate_gradient(lt_plus: np.ndarray, lt_minus: np.ndarray, eps: np.ndarray, sigma: float,
                      first_step: Optional[np.ndarray] = None) -> np.ndarray:
    """Antithetic zeroth-order gradient of the smoothed loss.

    lt_plus, lt_minus: per-step losses (S, n) of theta +- sigma * eps (they may
    also be totals of shape (n,), then no causal credit is possible).
    eps: (n, *P) directions. first_step: optional integer array broadcastable
    to P, the first index into the step-loss axis each parameter can affect;
    enables causal credit.
    """
    if lt_plus.ndim == 1:
        if first_step is not None:
            raise ValueError("causal credit needs per-step losses, got totals")
        lt_plus, lt_minus = lt_plus[None], lt_minus[None]
    n, pshape = eps.shape[0], eps.shape[1:]
    diff = lt_plus - lt_minus  # (S, n)
    if first_step is None:
        return np.einsum("j,j...->...", diff.sum(0), eps) / (2.0 * n * sigma)
    fs = np.broadcast_to(np.asarray(first_step, dtype=int), pshape).ravel()
    rtg = np.cumsum(diff[::-1], axis=0)[::-1]  # rtg[s] = sum_{t >= s}
    credit = rtg[np.clip(fs, 0, len(rtg) - 1)]  # (d, n)
    g = np.einsum("dj,jd->d", credit, eps.reshape(n, -1))
    return g.reshape(pshape) / (2.0 * n * sigma)


def spline_first_steps(interp: np.ndarray, state_offset: int = 1) -> np.ndarray:
    """First step-loss index each spline knot can affect, shape (K, 1).

    interp: (T, K) matrix mapping knots to per-step controls. Control t first
    changes state t + 1, so with step losses indexed by state (length T + 1)
    use state_offset=1.
    """
    starts = np.array([int(np.argmax(np.abs(interp[:, k]) > 0)) for k in range(interp.shape[1])])
    return (starts + state_offset)[:, None]


class Adam:
    """Adam on a numpy parameter array, cosine learning-rate decay."""

    def __init__(self, shape, lr, total_steps, lr_min_ratio=0.1, betas=(0.9, 0.999), eps=1e-8):
        self.m, self.v = np.zeros(shape), np.zeros(shape)
        self.lr, self.total, self.min_ratio = lr, total_steps, lr_min_ratio
        self.b1, self.b2, self.eps, self.t = betas[0], betas[1], eps, 0

    def step(self, params, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        frac = min(self.t / max(self.total, 1), 1.0)
        lr = self.lr * (self.min_ratio + (1 - self.min_ratio) * 0.5 * (1 + np.cos(np.pi * frac)))
        return params - lr * m_hat / (np.sqrt(v_hat) + self.eps)


def fixed_iteration_config(engine_config: OstrichEngineConfig) -> OstrichEngineConfig:
    """Copy of the config without early stopping, so every world runs the same
    number of Newton and linear iterations (worlds stay independent)."""
    return dataclasses.replace(
        engine_config,
        nr=dataclasses.replace(engine_config.nr, atol=0.0),
        linear=dataclasses.replace(engine_config.linear, tol=0.0, atol=0.0),
    )


class BatchedRollout:
    """Forward-only rollouts of all worlds of a replicated model, one plan per world.

    apply_params(params, controls): write the per-step controls of all worlds
        from params (W, *P); controls is a list of num_steps newton.Control.
    step_losses(body_q, body_qd, params): per-step losses (num_steps + 1, W)
        from body_q (num_steps + 1, W, bodies_per_world, 7) and body_qd (..., 6).
    initial_body_q / initial_body_qd: start state for one world (bodies_per_world,
        7/6) or for all worlds; default is the model's initial state.

    On CUDA the rollout is captured once as a graph and replayed.
    """

    def __init__(self, model: newton.Model, engine_config: OstrichEngineConfig, num_steps: int, dt: float,
                 apply_params: Callable, step_losses: Callable,
                 initial_body_q: Optional[np.ndarray] = None, initial_body_qd: Optional[np.ndarray] = None):
        self.model, self.num_steps, self.dt = model, num_steps, dt
        self.num_worlds = model.world_count
        self.bodies_per_world = model.body_count // self.num_worlds
        self.apply_params, self.step_losses = apply_params, step_losses
        self.engine = fixed_iteration_config(engine_config).create_engine(
            model, sim_steps=num_steps, logging_config=LoggingConfig(), differentiable_simulation=False)
        self.pipeline = newton.CollisionPipeline(model, requires_grad=False)
        self.states = [model.state() for _ in range(num_steps + 1)]
        self.controls = [model.control() for _ in range(num_steps)]
        self.q0 = self._tile(initial_body_q, self.states[0].body_q.numpy())
        self.qd0 = self._tile(initial_body_qd, self.states[0].body_qd.numpy())
        self.states[0].body_q.assign(self.q0)
        self.contacts = model.collide(self.states[0])
        self.graph = None
        self.body_q = self.body_qd = None  # last rollout's trajectories

    def _tile(self, value, default):
        if value is None:
            return default
        value = np.asarray(value, dtype=np.float32)
        if len(value) == self.bodies_per_world and self.num_worlds > 1:
            value = np.tile(value, (self.num_worlds, 1))
        if value.shape != default.shape:
            raise ValueError(f"initial state shape {value.shape} does not match {default.shape}")
        return value

    def _forward(self):
        for i in range(self.num_steps):
            self.pipeline.collide(self.states[i], self.contacts)
            self.engine.step(self.states[i], self.states[i + 1], self.controls[i], self.contacts, self.dt)

    def rollout(self, params: np.ndarray) -> np.ndarray:
        """params (W, *P) -> per-step losses (num_steps + 1, W)."""
        if len(params) != self.num_worlds:
            raise ValueError(f"need one plan per world ({self.num_worlds}), got {len(params)}")
        self.states[0].body_q.assign(self.q0)
        self.states[0].body_qd.assign(self.qd0)
        self.apply_params(params, self.controls)
        if self.model.device.is_cuda:
            if self.graph is None:
                with wp.ScopedCapture() as capture:
                    self._forward()
                self.graph = capture.graph
            wp.capture_launch(self.graph)
        else:
            self._forward()
        wp.synchronize()
        W, B = self.num_worlds, self.bodies_per_world
        self.body_q = np.array([s.body_q.numpy() for s in self.states]).reshape(self.num_steps + 1, W, B, 7)
        self.body_qd = np.array([s.body_qd.numpy() for s in self.states]).reshape(self.num_steps + 1, W, B, 6)
        return np.asarray(self.step_losses(self.body_q, self.body_qd, params), dtype=np.float64)


class ZeroOrderOptimizer:
    """Adam on the zeroth-order gradient of the smoothed loss.

    rollout: a BatchedRollout with config.num_worlds worlds, or any object with
        num_worlds and rollout(params (W, *P)) -> per-step losses (S, W).
    first_step: optional first step-loss index per parameter (see
        estimate_gradient); required for causal credit, e.g. spline_first_steps.
    """

    def __init__(self, rollout, config: ZeroOrderConfig = ZeroOrderConfig(), first_step: Optional[np.ndarray] = None):
        if rollout.num_worlds != config.num_worlds:
            raise ValueError(f"rollout has {rollout.num_worlds} worlds, config needs {config.num_worlds} "
                             f"(2 * n_pairs + 1)")
        if config.causal and first_step is None:
            raise ValueError("causal credit needs first_step (e.g. spline_first_steps(interp))")
        self.rollout, self.config = rollout, config
        self.first_step = first_step if config.causal else None
        self.rng = np.random.default_rng(config.seed)

    def estimate(self, theta: np.ndarray, sigma: float):
        """(loss at theta, gradient estimate, per-step losses of all worlds)."""
        n = self.config.n_pairs
        eps = sample_directions(self.rng, n, theta.shape, self.config.directions)
        lt = self.rollout.rollout(np.concatenate([theta[None], theta + sigma * eps, theta - sigma * eps]))
        g = estimate_gradient(lt[:, 1:n + 1], lt[:, n + 1:], eps, sigma, self.first_step)
        return float(lt[:, 0].sum()), g, lt

    def optimize(self, theta0: np.ndarray, callback: Optional[Callable] = None) -> dict:
        """Run config.iterations steps from theta0.

        callback(it, theta, loss, grad) is called every iteration; return True
        to stop early. The best iterate is by the (noisy) loss at theta.
        """
        c = self.config
        theta = np.array(theta0, dtype=np.float64)
        adam = Adam(theta.shape, c.lr, c.iterations, c.lr_min_ratio, c.betas, c.adam_eps)
        losses, best = [], (np.inf, theta.copy(), -1)
        for it in range(c.iterations):
            loss, g, _ = self.estimate(theta, c.sigma_at(it))
            losses.append(loss)
            if loss < best[0]:
                best = (loss, theta.copy(), it)
            if callback is not None and callback(it, theta, loss, g):
                break
            theta = adam.step(theta, g)
        return {"losses": losses, "best_loss": best[0], "best_params": best[1], "best_iter": best[2],
                "final_params": theta, "rollouts": len(losses) * c.num_worlds}
