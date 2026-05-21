"""Helhest trajectory optimization using Brax (JAX-based differentiable physics).

Optimizes K=10 spline control points (wheel velocity targets) for the 3-wheeled
helhest robot to match a target xy trajectory. Gradients are computed via JAX
reverse-mode AD through the chosen Brax pipeline.

Brax does not support cylinder collision geometry; wheels are approximated as
spheres.  Use brax_sweep.py to find stable (pipeline, dt, kv, ...) combinations,
then pass the best values here via CLI args.

Stability notes (from sweep):
  positional:   dt=0.01 kv=150  → stable, x≈0.1m/0.5s
  generalized:  dt=0.002 kv=any → barely stable, nearly zero motion
  spring:       run sweep to find best

Usage:
    python examples/comparison_gradient/helhest/brax_sim.py
    python examples/comparison_gradient/helhest/brax_sim.py \\
        --pipeline positional --dt 0.01 --kv 150
    python examples/comparison_gradient/helhest/brax_sim.py \\
        --pipeline generalized --dt 0.002 --kv 50 --baumgarte 0.1
    python examples/comparison_gradient/helhest/brax_sim.py --save results/brax.json
"""
import argparse
import json
import os
import pathlib
import time

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import numpy as np
import warnings

warnings.filterwarnings("ignore")

import brax.io.mjcf as mjcf
from config import DURATION, INIT_CTRL, TARGET_CTRL

K = 10   # spline control points
NU = 3   # wheels: left, right, rear
TRAJECTORY_WEIGHT = 10.0

BASE_XML = """<mujoco model="helhest">
  <option gravity="0 0 -9.81" timestep="{dt}"/>
  <worldbody>
    <geom name="ground" type="plane" size="100 100 0.1" friction="0.7 0.1 0.01"/>
    <body name="chassis" pos="0 0 0.37">
      <freejoint/>
      <inertial mass="85.0" pos="-0.047 0 0" diaginertia="0.6213 0.1583 0.677"/>
      <geom type="box" pos="-0.047 0 0" size="0.13 0.3 0.09"
            contype="0" conaffinity="0"/>
      <body name="left_wheel" pos="0 0.36 0">
        <joint name="left_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="sphere" size="0.36" friction="0.7 0.1 0.01"/>
      </body>
      <body name="right_wheel" pos="0 -0.36 0">
        <joint name="right_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="sphere" size="0.36" friction="0.7 0.1 0.01"/>
      </body>
      <body name="rear_wheel" pos="-0.697 0 0">
        <joint name="rear_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="sphere" size="0.36" friction="0.35 0.1 0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="left_vel"  joint="left_wheel_j"  kv="{kv}"/>
    <velocity name="right_vel" joint="right_wheel_j" kv="{kv}"/>
    <velocity name="rear_vel"  joint="rear_wheel_j"  kv="{kv}"/>
  </actuator>
</mujoco>"""


def patch_sys(sys, baumgarte=None, elasticity=None):
    """Apply runtime patches that don't change array shapes (no recompile)."""
    changes = {}
    if baumgarte is not None:
        changes["baumgarte_erp"] = jnp.array(baumgarte, jnp.float32)
    if elasticity is not None:
        changes["elasticity"] = jnp.full_like(sys.elasticity, elasticity)
    return sys.replace(**changes) if changes else sys


def make_interp_matrix(T: int, K: int) -> jnp.ndarray:
    W = np.zeros((T, K), dtype=np.float32)
    for t in range(T):
        k_float = t * (K - 1) / max(T - 1, 1)
        k_low = int(k_float)
        k_high = min(k_low + 1, K - 1)
        alpha = k_float - k_low
        W[t, k_low] += 1.0 - alpha
        W[t, k_high] += alpha
    return jnp.array(W)


def make_rollout(pipe, sys, W):
    """Return a JIT-able rollout function closed over pipeline, sys, W."""
    def rollout(params):
        expanded = W @ params  # (T, NU)
        q0 = jnp.zeros(sys.q_size()).at[2].set(0.37).at[3].set(1.0)
        state = pipe.init(sys, q0, jnp.zeros(sys.qd_size()))

        def step_fn(state, act):
            state = pipe.step(sys, state, act)
            return state, state.x.pos[0, :2]

        _, xy_traj = jax.lax.scan(step_fn, state, expanded)
        return xy_traj  # (T, 2)
    return rollout


def adam_step(grad, m, v, t, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    t += 1
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    return m_hat / (jnp.sqrt(v_hat) + eps), m, v, t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH")
    parser.add_argument("--pipeline", choices=["positional", "generalized", "spring"],
                        default="positional",
                        help="Brax physics pipeline (default: positional)")
    parser.add_argument("--dt", type=float, default=None,
                        help="Timestep override. Defaults: positional=0.01, "
                             "generalized=0.002, spring=0.01")
    parser.add_argument("--kv", type=float, default=150.0,
                        help="Motor gain kv (default: 150)")
    parser.add_argument("--baumgarte", type=float, default=None,
                        help="Baumgarte ERP contact correction rate (default: pipeline default)")
    parser.add_argument("--elasticity", type=float, default=None,
                        help="Contact elasticity 0–1 (default: pipeline default)")
    parser.add_argument("--duration", type=float, default=DURATION,
                        help=f"Simulated duration in seconds (default: {DURATION}). "
                             "Longer = more JIT time.")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Adam learning rate (default: 0.01)")
    parser.add_argument("--iters", type=int, default=50,
                        help="Number of optimization iterations (default: 50)")
    parser.add_argument("--target-only", action="store_true",
                        help="Only compute and save the target trajectory, skip optimization")
    args = parser.parse_args()

    # ── pipeline defaults ──
    if args.pipeline == "positional":
        import brax.positional.pipeline as pipe
        dt = args.dt or 0.01
    elif args.pipeline == "generalized":
        import brax.generalized.pipeline as pipe
        dt = args.dt or 0.002
    else:
        import brax.spring.pipeline as pipe
        dt = args.dt or 0.01

    T = int(args.duration / dt)
    kv = args.kv

    # ── build system ──
    xml = BASE_XML.format(dt=dt, kv=kv)
    sys = mjcf.loads(xml)
    sys = patch_sys(sys, baumgarte=args.baumgarte, elasticity=args.elasticity)

    W = make_interp_matrix(T, K)
    rollout = make_rollout(pipe, sys, W)

    # ── target trajectory ──
    target_params = jnp.tile(jnp.array(TARGET_CTRL, dtype=jnp.float32), (K, 1))
    target_xy = jax.jit(rollout)(target_params)
    jax.block_until_ready(target_xy)
    print(f"Target final xy: ({float(target_xy[-1, 0]):.3f}, {float(target_xy[-1, 1]):.3f})")

    if args.target_only:
        traj_result = {
            "simulator": f"Brax ({args.pipeline})",
            "problem": "helhest",
            "dt": dt,
            "T": T,
            "target_trajectory": np.array(target_xy).tolist(),
        }
        if args.save:
            pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(args.save).write_text(json.dumps(traj_result, indent=2))
            print(f"Saved to {args.save}")
        return

    # ── gradient function ──
    @jax.jit
    def loss_and_grad(params):
        def loss_fn(p):
            xy = rollout(p)
            return jnp.mean(jnp.sum((xy - target_xy) ** 2, axis=-1)) * TRAJECTORY_WEIGHT
        return jax.value_and_grad(loss_fn)(params)

    params = jnp.tile(jnp.array(INIT_CTRL, dtype=jnp.float32), (K, 1))
    m_adam = jnp.zeros_like(params)
    v_adam = jnp.zeros_like(params)
    t_adam = 0

    print(f"\nOptimizing: T={T}, dt={dt}, kv={kv}, K={K}, params={K*NU} "
          f"(Brax {args.pipeline} pipeline, JAX grad)")

    results = {
        "simulator": f"Brax ({args.pipeline})",
        "problem": "helhest",
        "pipeline": args.pipeline,
        "dt": dt,
        "kv": kv,
        "T": T,
        "K": K,
        "iterations": [],
        "loss": [],
        "time_ms": [],
    }

    for i in range(args.iters):
        t0 = time.perf_counter()
        loss, grad = loss_and_grad(params)
        jax.block_until_ready(grad)
        t_iter = (time.perf_counter() - t0) * 1000

        grad_norm = float(jnp.linalg.norm(grad))
        if grad_norm > 1.0:
            grad = grad / grad_norm
        update, m_adam, v_adam, t_adam = adam_step(grad, m_adam, v_adam, t_adam, lr=args.lr)
        params = params - update

        p0, pm, pN = params[0], params[K // 2], params[-1]
        print(
            f"Iter {i:3d}: loss={float(loss):.4f} | "
            f"cp[0]=({float(p0[0]):.2f},{float(p0[1]):.2f}) "
            f"cp[{K//2}]=({float(pm[0]):.2f},{float(pm[1]):.2f}) "
            f"cp[-1]=({float(pN[0]):.2f},{float(pN[1]):.2f}) | "
            f"t={t_iter:.1f}ms"
        )
        results["iterations"].append(i)
        results["loss"].append(float(loss))
        results["time_ms"].append(t_iter)

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
