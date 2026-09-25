"""Tests for ostrich.optim.zeroth_order."""
import sys
from pathlib import Path

import numpy as np
import pytest

from ostrich.optim.zeroth_order import (Adam, ZeroOrderConfig, ZeroOrderOptimizer, estimate_gradient,
                                        fixed_iteration_config, sample_directions, spline_first_steps)

# --- a time-structured quadratic toy: x_t = dt * sum_{s<t} (W theta)_s, loss_t = |x_t - r_t|^2 ---
T, K, D = 60, 6, 2
_rng = np.random.default_rng(0)
INTERP = np.clip(1 - np.abs(np.linspace(0, K - 1, T)[:, None] - np.arange(K)[None]), 0, None)  # hat functions
REF = np.cumsum(_rng.standard_normal((T + 1, D)), 0) * 0.05


def toy_step_losses(thetas):
    """thetas (n, K, D) -> per-step losses (T+1, n)."""
    u = np.einsum("tk,nkd->tnd", INTERP, thetas)
    x = np.concatenate([np.zeros((1, len(thetas), D)), np.cumsum(u, 0) * 0.1], 0)
    return np.sum((x - REF[:, None]) ** 2, -1)


def toy_gradient(theta):
    g = np.zeros_like(theta)
    for i in np.ndindex(theta.shape):
        e = np.zeros_like(theta); e[i] = 1e-6
        g[i] = (toy_step_losses((theta + e)[None]).sum() - toy_step_losses((theta - e)[None]).sum()) / 2e-6
    return g


class ToyRollout:
    def __init__(self, n_worlds):
        self.num_worlds = n_worlds

    def rollout(self, params):
        return toy_step_losses(params)


def test_orthogonal_directions():
    rng = np.random.default_rng(1)
    eps = sample_directions(rng, 5, (K, D), "orth").reshape(5, -1)
    gram = eps @ eps.T
    assert np.allclose(np.diag(gram), K * D)
    assert np.allclose(gram - np.diag(np.diag(gram)), 0, atol=1e-9)
    assert sample_directions(rng, 30, (K, D), "orth").shape == (30, K, D)  # n > d: stacked blocks


def test_spline_first_steps():
    fs = spline_first_steps(INTERP)
    assert fs.shape == (K, 1)
    assert fs[0, 0] == 1 and np.all(np.diff(fs[:, 0]) > 0)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kind", ["iid", "orth"])
def test_estimator_unbiased(kind, causal):
    rng = np.random.default_rng(2)
    theta = rng.standard_normal((K, D))
    truth = toy_gradient(theta)       # quadratic -> smoothed gradient equals the exact one
    fs = spline_first_steps(INTERP) if causal else None
    sigma, n, reps = 0.2, 8, 3000
    acc = np.zeros_like(theta)
    for _ in range(reps):
        eps = sample_directions(rng, n, theta.shape, kind)
        acc += estimate_gradient(toy_step_losses(theta + sigma * eps), toy_step_losses(theta - sigma * eps),
                                 eps, sigma, fs)
    assert np.linalg.norm(acc / reps - truth) / np.linalg.norm(truth) < 0.05


def test_causal_and_orthogonal_reduce_variance():
    rng = np.random.default_rng(3)
    theta = rng.standard_normal((K, D))
    truth = toy_gradient(theta)
    err = {}
    for kind, causal in [("iid", False), ("orth", True)]:
        fs = spline_first_steps(INTERP) if causal else None
        e = []
        for _ in range(400):
            eps = sample_directions(rng, 6, theta.shape, kind)
            g = estimate_gradient(toy_step_losses(theta + 0.2 * eps), toy_step_losses(theta - 0.2 * eps), eps, 0.2, fs)
            e.append(np.linalg.norm(g - truth))
        err[(kind, causal)] = np.mean(e)
    assert err[("orth", True)] < 0.9 * err[("iid", False)]


def test_totals_only_path():
    rng = np.random.default_rng(4)
    theta = rng.standard_normal((K, D))
    eps = sample_directions(rng, 4, theta.shape, "iid")
    lp, lm = toy_step_losses(theta + 0.1 * eps), toy_step_losses(theta - 0.1 * eps)
    assert np.allclose(estimate_gradient(lp.sum(0), lm.sum(0), eps, 0.1), estimate_gradient(lp, lm, eps, 0.1))
    with pytest.raises(ValueError):
        estimate_gradient(lp.sum(0), lm.sum(0), eps, 0.1, spline_first_steps(INTERP))


def test_optimizer_decreases_loss():
    cfg = ZeroOrderConfig(iterations=150, n_pairs=4, sigma=0.1, lr=0.05, seed=5)
    opt = ZeroOrderOptimizer(ToyRollout(cfg.num_worlds), cfg, first_step=spline_first_steps(INTERP))
    theta0 = np.zeros((K, D))
    res = opt.optimize(theta0)
    l0 = toy_step_losses(theta0[None]).sum()
    assert res["best_loss"] < 0.2 * l0
    assert res["rollouts"] == 150 * cfg.num_worlds


def test_optimizer_validates_world_count_and_causal():
    cfg = ZeroOrderConfig(n_pairs=4)
    with pytest.raises(ValueError):
        ZeroOrderOptimizer(ToyRollout(5), cfg, first_step=spline_first_steps(INTERP))
    with pytest.raises(ValueError):
        ZeroOrderOptimizer(ToyRollout(cfg.num_worlds), cfg)          # causal without first_step


def test_adam_cosine_schedule():
    a = Adam((1,), lr=1.0, total_steps=10, lr_min_ratio=0.1)
    steps = []
    p = np.zeros(1)
    for _ in range(10):
        q = a.step(p, np.ones(1)); steps.append(float((p - q)[0])); p = q
    assert steps[0] > steps[-1] > 0.05          # decays towards lr * lr_min_ratio


def test_fixed_iteration_config():
    from ostrich import LinearSolverConfig, NewtonRaphsonConfig, OstrichEngineConfig
    ec = OstrichEngineConfig(nr=NewtonRaphsonConfig(max_iters=12, atol=1e-3),
                             linear=LinearSolverConfig(max_iters=16, tol=1e-3, atol=1e-3))
    f = fixed_iteration_config(ec)
    assert f.nr.atol == 0 and f.linear.tol == 0 and f.linear.atol == 0
    assert f.nr.max_iters == 12 and f.linear.max_iters == 16
    assert ec.nr.atol == 1e-3                    # original untouched


def test_batched_rollout_worlds_independent_cpu():
    """On CPU (deterministic), a world's losses depend only on its own plan and
    match a single-world rollout, bit for bit."""
    wp = pytest.importorskip("warp")
    wp.set_device("cpu")
    try:
        import newton
        from ostrich import OstrichEngineConfig
        from ostrich.core.model_builder import OstrichModelBuilder
        from ostrich.optim.zeroth_order import BatchedRollout
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
        from helhest.common import create_helhest_model

        def build(n_worlds):
            b = OstrichModelBuilder()
            b.rigid_gap = 0.1
            b.add_ground_plane()
            create_helhest_model(b, xform=wp.transform(wp.vec3(0, 0, 0.45), wp.quat_identity()), is_visible=False,
                                 control_mode="velocity", k_p=150.0, k_d=0.0, friction_left_right=0.7,
                                 friction_rear=0.35)
            return b.finalize_replicated(num_worlds=n_worlds, gravity=-9.81)

        steps = 12

        def apply_params(params, controls):
            jtv = np.zeros((len(params), controls[0].joint_target_vel.size // len(params)), np.float32)
            jtv[:, 6:9] = params
            for c in controls:
                c.joint_target_vel.assign(jtv.reshape(c.joint_target_vel.shape))

        def step_losses(body_q, body_qd, params):
            return (body_q[:, :, 0, 0] - 1.0) ** 2        # chassis x towards x = 1

        def make(n):
            m = build(n)
            return BatchedRollout(m, OstrichEngineConfig(), steps, 0.05, apply_params, step_losses)

        p = np.array([[3.0, 3.0, 3.0]])
        three = make(3).rollout(np.concatenate([p, p + 2.0, p]))
        one = make(1).rollout(p)
        assert np.array_equal(three[:, 0], three[:, 2])
        assert not np.array_equal(three[:, 0], three[:, 1])
        assert np.array_equal(three[:, 0], one[:, 0])
    finally:
        wp.set_device("cuda:0" if wp.is_cuda_available() else "cpu")
