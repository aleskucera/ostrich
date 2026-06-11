"""Helhest_junior box trajectory optimization using Brax (jax.grad).

Brax counterpart of optimize_mjx.py / optimize_ostrich.py. Optimizes a K-knot
wheel-velocity spline so the junior matches a recorded real trajectory while
crossing the box. Runs one of Brax's three differentiable pipelines
(positional / generalized / spring) and logs the per-iteration loss curve and
gradient norms, so we can show *which* failure mode each pipeline exhibits:

  - positional (Position-Based Dynamics): does gradient descent through contact
    actually reduce the loss, or are the gradients uninformative?
  - generalized (QP) / spring (penalty): can the forward model even track the
    recorded box-crossing, i.e. does the best-found spline reach a low loss?

Brax has no cylinder collision geometry, so wheels are approximated as spheres
(radius 0.35, matching the cylinder radius). This is the same sphere
approximation noted in experiments/1_sim_to_real/sweep_brax.py.

Outputs results/brax_<pipeline>.json with the same per-trial schema as
optimize_mjx.py (losses, grad_norms, wall_s, best_loss).

Run inside the Brax venv:
    /home/kuceral4/projects/ostrich/.venv-brax/bin/python \
        experiments/3_gradient_quality_box/optimize_brax.py --pipeline positional
"""
import argparse
import json
import os
import pathlib
import time

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import numpy as np
import jax
import jax.numpy as jnp
import brax.io.mjcf as mjcf

RESULTS_DIR = pathlib.Path(__file__).parent / "results"

# Brax-friendly junior + box MJCF. Mirrors JUNIOR_BOX_XML geometry/masses from
# 1_sim_to_real_box/sweep_mujoco.py, but: (1) sphere wheels (Brax has no
# cylinder), (2) MuJoCo solver attributes (solref/solimp/condim/cone/impratio)
# stripped since Brax's MJCF loader ignores them.
BRAX_BOX_XML = """<mujoco model="helhest_junior_box_brax">
  <option gravity="0 0 -9.81" timestep="{dt}"/>
  <worldbody>
    <geom name="ground" type="plane" size="100 100 0.1" friction="{mu} 0.1 0.01"/>
    <geom name="box" type="box" pos="1.37 0 0.06" size="0.37 0.575 0.06"
          friction="{mu} 0.1 0.01"/>
    <body name="chassis" pos="0 0 0.35">
      <freejoint name="base_joint"/>
      <inertial mass="89.7" pos="-0.188 0 0" diaginertia="2.41 4.22 6.03"/>
      <geom type="box" pos="-0.13 0 0" size="0.24 0.28 0.10" contype="0" conaffinity="0"/>
      <geom type="box" pos="-0.61 0 0" size="0.24 0.12 0.10" contype="0" conaffinity="0"/>
      <body name="left_wheel" pos="0 0.365 0">
        <joint name="left_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.173021 0.336875 0.173021"/>
        <geom {wheel_geom} friction="{mu} 0.1 0.01"/>
      </body>
      <body name="right_wheel" pos="0 -0.365 0">
        <joint name="right_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.173021 0.336875 0.173021"/>
        <geom {wheel_geom} friction="{mu} 0.1 0.01"/>
      </body>
      <body name="rear_wheel" pos="-0.75 0 0">
        <joint name="rear_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.173021 0.336875 0.173021"/>
        <geom {wheel_geom} friction="{mu} 0.1 0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="left_act"  joint="left_wheel_j"  kv="{kv}"/>
    <velocity name="right_act" joint="right_wheel_j" kv="{kv}"/>
    <velocity name="rear_act"  joint="rear_wheel_j"  kv="{kv}"/>
  </actuator>
</mujoco>"""


def make_interp_matrix(T, K):
    W = np.zeros((T, K), dtype=np.float32)
    for t in range(T):
        x = t * (K - 1) / max(T - 1, 1)
        lo = int(x); hi = min(lo + 1, K - 1); a = x - lo
        W[t, lo] += 1.0 - a
        W[t, hi] += a
    return W


WHEEL_GEOM = {
    "sphere": 'type="sphere" size="0.35"',
    "capsule": 'type="capsule" fromto="0 -0.05 0 0 0.05 0" size="0.35"',
}


def build_sys(dt, kv, mu, wheel="sphere"):
    xml = BRAX_BOX_XML.format(dt=dt, kv=kv, mu=mu, wheel_geom=WHEEL_GEOM[wheel])
    return mjcf.loads(xml)


def make_rollout_loss(pipe, sys, W, target_xy):
    q0 = jnp.zeros(sys.q_size()).at[2].set(0.35).at[3].set(1.0)  # z + quat w
    qd0 = jnp.zeros(sys.qd_size())

    def loss_fn(params):
        ctrl_traj = W @ params  # [T,3]
        state0 = pipe.init(sys, q0, qd0)

        def step(state, ctrl):
            state = pipe.step(sys, state, ctrl)
            return state, state.x.pos[0, :2]

        _, xy = jax.lax.scan(step, state0, ctrl_traj)
        return jnp.mean(jnp.sum((xy - target_xy) ** 2, axis=1))

    return loss_fn


class SplineAdam:
    def __init__(self, K, num_dofs, lr=0.1, lr_min_ratio=0.2, total_steps=50,
                 betas=(0.9, 0.999), eps=1e-8):
        self.lr_init = lr; self.lr_min = lr * lr_min_ratio
        self.total_steps = total_steps
        self.b1, self.b2 = betas; self.eps = eps
        self.m = np.zeros((K, num_dofs), dtype=np.float64)
        self.v = np.zeros((K, num_dofs), dtype=np.float64)
        self.t = 0

    def _lr(self):
        p = min(self.t / max(1, self.total_steps), 1.0)
        return self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (1.0 + np.cos(np.pi * p))

    def step(self, params, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        mh = self.m / (1 - self.b1**self.t) if self.b1 > 0 else self.m
        vh = self.v / (1 - self.b2**self.t)
        return params - self._lr() * mh / (np.sqrt(vh) + self.eps)


def run_trial(loss_and_grad, K, lr, iterations, seed, clip_grad_norm=None):
    rng = np.random.default_rng(seed)
    params_np = (2.0 + 0.5 * rng.standard_normal((K, 3))).astype(np.float32)
    opt = SplineAdam(K=K, num_dofs=3, lr=lr, total_steps=iterations)
    losses, grad_norms, n_clipped = [], [], 0
    t0_total = time.perf_counter()
    for it in range(iterations):
        t0 = time.perf_counter()
        loss_val, grad = loss_and_grad(jnp.asarray(params_np))
        loss_val = float(loss_val); grad_np = np.asarray(grad).astype(np.float64)
        gnorm = float(np.linalg.norm(grad_np))
        if clip_grad_norm is not None and np.isfinite(gnorm) and gnorm > clip_grad_norm:
            grad_np = grad_np * (clip_grad_norm / gnorm); n_clipped += 1
        losses.append(loss_val); grad_norms.append(gnorm)
        if not np.all(np.isfinite(grad_np)):
            print(f"    iter {it:3d}: loss={loss_val:.4f}  |g|={gnorm}  (non-finite grad, stop)")
            break
        params_np = opt.step(params_np, grad_np).astype(np.float32)
        print(f"    iter {it:3d}: loss={loss_val:.4f}  |g|={gnorm:.4g}  "
              f"({time.perf_counter()-t0:.2f}s)", flush=True)
    return {"seed": int(seed), "losses": losses, "grad_norms": grad_norms,
            "n_clipped": int(n_clipped), "wall_s": time.perf_counter()-t0_total,
            "best_loss": float(np.nanmin(losses)) if losses else float("nan")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", choices=["positional", "generalized", "spring"],
                    default="positional")
    ap.add_argument("--gt", default=str(pathlib.Path(__file__).resolve().parents[1]
                                        / "1_sim_to_real_box" / "data"
                                        / "run_2026_05_20-18_10_33.json"))
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--num-trials", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--horizon-s", type=float, default=6.0)
    ap.add_argument("--dt", type=float, default=5e-3)
    ap.add_argument("--kv", type=float, default=150.0)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--wheel", choices=["sphere", "capsule"], default="sphere",
                    help="wheel collision geom. positional is unstable with capsule; "
                         "spring/generalized are stable with capsule (preferred, "
                         "matches the MJX capsule wheels).")
    ap.add_argument("--clip-grad-norm", type=float, default=1.0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    if args.pipeline == "positional":
        import brax.positional.pipeline as pipe
    elif args.pipeline == "generalized":
        import brax.generalized.pipeline as pipe
    else:
        import brax.spring.pipeline as pipe

    with open(args.gt) as f:
        gt = json.load(f)
    real_t = np.asarray(gt["real"]["t"])
    real_xy = np.column_stack([gt["real"]["x"], gt["real"]["y"]])
    m = (real_t >= 0) & (real_t <= args.horizon_s)
    real_t, real_xy = real_t[m], real_xy[m]

    T = int(round(args.horizon_s / args.dt))
    t_grid = np.arange(T) * args.dt
    target_xy = np.zeros((T, 2), dtype=np.float32)
    for c in range(2):
        target_xy[:, c] = np.interp(t_grid, real_t, real_xy[:, c])

    print(f"pipeline={args.pipeline}  wheel={args.wheel}  T={T}  dt={args.dt}  "
          f"kv={args.kv}  mu={args.mu}")
    print(f"JAX devices: {jax.devices()}")

    sys = build_sys(args.dt, args.kv, args.mu, wheel=args.wheel)
    W = make_interp_matrix(T, args.K)
    loss_fn = make_rollout_loss(pipe, sys, jnp.asarray(W), jnp.asarray(target_xy))
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    trials = []
    for k in range(args.num_trials):
        seed = args.seed_base + k
        print(f"\n--- trial {k+1}/{args.num_trials} (seed={seed}) ---")
        trials.append(run_trial(loss_and_grad, args.K, args.lr, args.iterations,
                                seed, clip_grad_norm=args.clip_grad_norm))

    out = {"simulator": f"Brax ({args.pipeline})",
           "gradient_method": "jax.grad (BPTT)",
           "pipeline": args.pipeline, "wheel": args.wheel, "gt": gt.get("run_id"),
           "K": args.K, "lr": args.lr, "iterations": args.iterations,
           "horizon_s": args.horizon_s, "dt": args.dt, "kv": args.kv, "mu": args.mu,
           "num_trials": args.num_trials, "trials": trials}
    save_path = args.save or str(RESULTS_DIR / f"brax_{args.pipeline}_{args.wheel}.json")
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(out, f)
    best = np.nanmin([t["best_loss"] for t in trials])
    print(f"\nBest loss across {args.num_trials} trials: {best:.4f}  -> {save_path}")


if __name__ == "__main__":
    main()
