"""Ball throw optimization using Brax (JAX-based differentiable physics).

Optimizes the initial 3D linear velocity of a ball to match a target trajectory.
Uses JAX reverse-mode AD (jax.grad) through Brax's spring pipeline.

Brax's spring pipeline uses spring/damper contact forces and is fully
differentiable via reverse-mode AD. First iteration includes JIT compilation and
is excluded from median timing.

Usage:
    python examples/comparison_gradient/ball_throw/brax.py
    python examples/comparison_gradient/ball_throw/brax.py --save results/brax.json
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

import brax.spring.pipeline as pipeline
import brax.io.mjcf as mjcf

from config import DT, DURATION, INIT_VEL, TARGET_VEL

T = int(DURATION / DT)  # 50 steps

LEARNING_RATE = 2e-2
MAX_GRAD = 100.0

BALL_XML = f"""
<mujoco model="ball_throw">
  <option gravity="0 0 -9.81" timestep="{DT}"/>
  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="100 100 0.1"
          friction="0.7 0.1 0.01"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <inertial mass="1.0" pos="0 0 0" diaginertia="0.016 0.016 0.016"/>
      <geom type="sphere" size="0.2" friction="0.7 0.1 0.01"/>
    </body>
  </worldbody>
</mujoco>
"""


def make_system():
    return mjcf.loads(BALL_XML)


def rollout(sys, vel):
    """Simulate T steps from rest with initial velocity vel (3,).
    Returns (T, 3) xyz positions of the ball.
    """
    # q: [x, y, z, qw, qx, qy, qz]; qd: [vx, vy, vz, wx, wy, wz]
    q0 = jnp.zeros(sys.q_size()).at[2].set(1.0).at[3].set(1.0)
    qd0 = jnp.zeros(sys.qd_size()).at[0].set(vel[0]).at[1].set(vel[1]).at[2].set(vel[2])
    state = pipeline.init(sys, q0, qd0)

    def step_fn(state, _):
        state = pipeline.step(sys, state, jnp.zeros(0))
        return state, state.x.pos[0, :]

    _, xyz_traj = jax.lax.scan(step_fn, state, None, length=T)
    return xyz_traj  # (T, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH")
    args = parser.parse_args()

    sys = make_system()

    # Compute target trajectory
    target_vel = jnp.array(TARGET_VEL, dtype=jnp.float32)
    target_xyz = jax.jit(lambda v: rollout(sys, v))(target_vel)
    jax.block_until_ready(target_xyz)
    print(f"Target final xyz: ({float(target_xyz[-1, 0]):.3f}, {float(target_xyz[-1, 1]):.3f}, {float(target_xyz[-1, 2]):.3f})")

    @jax.jit
    def loss_and_grad(vel):
        def loss_fn(v):
            xyz = rollout(sys, v)
            return jnp.mean(jnp.sum((xyz - target_xyz) ** 2, axis=-1))
        return jax.value_and_grad(loss_fn)(vel)

    vel = jnp.array(INIT_VEL, dtype=jnp.float32)
    print(f"\nOptimizing: T={T}, dt={DT}, params=3 (Brax, spring pipeline, JAX grad)")

    results = {
        "simulator": "Brax",
        "problem": "ball_throw",
        "dt": DT,
        "T": T,
        "iterations": [],
        "loss": [],
        "time_ms": [],
    }

    for i in range(50):
        t0 = time.perf_counter()
        loss, grad = loss_and_grad(vel)
        jax.block_until_ready(grad)
        t_iter = (time.perf_counter() - t0) * 1000

        grad = jnp.clip(grad, -MAX_GRAD, MAX_GRAD)
        vel = vel - LEARNING_RATE * grad

        print(
            f"Iter {i:3d}: loss={float(loss):.4f} | "
            f"vel=({float(vel[0]):.2f},{float(vel[1]):.2f},{float(vel[2]):.2f}) | "
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
