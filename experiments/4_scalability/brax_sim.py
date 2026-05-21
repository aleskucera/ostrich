"""Helhest scalability benchmark — Brax (JAX/GPU), variable number of worlds.

Brax uses jax.vmap for GPU-parallel batch simulation. Time scales sub-linearly
with num_worlds as the GPU parallelises across worlds. Peak GPU memory is tracked
via JAX device memory stats.

NOTE: Brax's positional pipeline requires dt=0.01 for stable simulation of
this robot (at dt=0.05 the robot sinks through the ground). Cylinder wheels
are approximated as spheres. T=300 steps (3s duration) vs T=60 for others.

Usage:
    python examples/comparison_scalability/helhest_scalability/brax.py --num-worlds 1
    python examples/comparison_scalability/helhest_scalability/brax.py --num-worlds 100 --save results/brax_100.json
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

import brax.positional.pipeline as pipeline
import brax.io.mjcf as mjcf

DT = 1e-2  # positional pipeline stable at dt=0.01; dt=0.05 sinks through ground
DURATION = 3.0
T = int(DURATION / DT)  # 300 steps
K = 10
NU = 3
ITERATIONS = 20

TARGET_CTRL = (1.0, 6.0, 0.0)
INIT_CTRL = (2.0, 5.0, 0.0)

TRAJECTORY_WEIGHT = 10.0

HELHEST_XML = f"""
<mujoco model="helhest">
  <option gravity="0 0 -9.81" timestep="{DT}"/>
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
    <velocity name="left_vel"  joint="left_wheel_j"  kv="150"/>
    <velocity name="right_vel" joint="right_wheel_j" kv="150"/>
    <velocity name="rear_vel"  joint="rear_wheel_j"  kv="150"/>
  </actuator>
</mujoco>
"""


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


def adam_step(grad, m, v, t, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    t += 1
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    return m_hat / (jnp.sqrt(v_hat) + eps), m, v, t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-worlds", type=int, default=1)
    parser.add_argument("--save", metavar="PATH")
    args = parser.parse_args()

    num_worlds = args.num_worlds
    sys = mjcf.loads(HELHEST_XML)
    W = make_interp_matrix(T, K)
    q0 = jnp.zeros(sys.q_size()).at[2].set(0.37).at[3].set(1.0)
    qd0 = jnp.zeros(sys.qd_size())

    def single_rollout(params):
        expanded = W @ params
        state = pipeline.init(sys, q0, qd0)

        def step_fn(state, act):
            state = pipeline.step(sys, state, act)
            return state, state.x.pos[0, :2]

        _, xy_traj = jax.lax.scan(step_fn, state, expanded)
        return xy_traj

    target_params = jnp.tile(jnp.array(TARGET_CTRL, dtype=jnp.float32), (K, 1))
    target_xy = jax.jit(single_rollout)(target_params)
    jax.block_until_ready(target_xy)

    def single_loss(params):
        xy = single_rollout(params)
        return jnp.mean(jnp.sum((xy - target_xy) ** 2, axis=-1)) * TRAJECTORY_WEIGHT

    batched_vg = jax.vmap(jax.value_and_grad(single_loss))

    @jax.jit
    def batch_step(batch_params):
        losses, grads = batched_vg(batch_params)
        return jnp.mean(losses), jnp.mean(grads, axis=0)

    params = jnp.tile(jnp.array(INIT_CTRL, dtype=jnp.float32), (K, 1))
    batch_params = jnp.stack([params] * num_worlds)
    m_adam = jnp.zeros_like(params)
    v_adam = jnp.zeros_like(params)
    t_adam = 0

    print(f"Optimizing: T={T}, dt={DT}, K={K}, num_worlds={num_worlds} (Brax, jax.vmap)")

    time_ms_list = []
    peak_gpu_mb = 0.0

    for i in range(ITERATIONS):
        t0 = time.perf_counter()
        avg_loss, avg_grad = batch_step(batch_params)
        jax.block_until_ready(avg_grad)
        t_iter = (time.perf_counter() - t0) * 1000

        try:
            mem_stats = jax.devices("gpu")[0].memory_stats()
            used_mb = mem_stats.get("bytes_in_use", 0) / 1024 ** 2
            peak_gpu_mb = max(peak_gpu_mb, used_mb)
        except Exception:
            pass

        grad_norm = float(jnp.linalg.norm(avg_grad))
        if grad_norm > 1.0:
            avg_grad = avg_grad / grad_norm
        update, m_adam, v_adam, t_adam = adam_step(avg_grad, m_adam, v_adam, t_adam, lr=0.01)
        params = params - update
        batch_params = jnp.stack([params] * num_worlds)

        print(f"  iter {i:3d}: loss={float(avg_loss):.4f} | t={t_iter:.0f}ms | gpu={peak_gpu_mb:.0f}MB")
        time_ms_list.append(t_iter)

    median_ms = float(np.median(time_ms_list[3:])) if len(time_ms_list) > 3 else float(np.median(time_ms_list))

    results = {
        "simulator": "Brax",
        "num_worlds": num_worlds,
        "median_time_ms": median_ms,
        "peak_gpu_mb": peak_gpu_mb if peak_gpu_mb > 0 else None,
        "time_ms": time_ms_list,
    }

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
