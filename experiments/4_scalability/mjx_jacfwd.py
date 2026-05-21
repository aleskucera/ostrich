"""Helhest scalability benchmark — MJX (grad), variable number of worlds via jax.vmap.

Usage:
    python examples/comparison/helhest_scalability/mjx_jacfwd.py --num-worlds 100 --save results/mjx_jacfwd_100.json
"""
import argparse
import json
import os
import pathlib
import time

os.environ.setdefault("DISPLAY", ":1")
os.environ.pop("WAYLAND_DISPLAY", None)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np
import optax

DT = 2e-3
DURATION = 3.0
T = int(DURATION / DT)
K = 30
TARGET_CTRL = np.array([1.0, 6.0, 0.0], dtype=np.float32)
INIT_CTRL = np.array([2.0, 5.0, 0.0], dtype=np.float32)
TRAJECTORY_WEIGHT = 10.0
SMOOTHNESS_WEIGHT = 1e-2
REGULARIZATION_WEIGHT = 1e-7
ITERATIONS = 5

HELHEST_XML = f"""
<mujoco model="helhest">
  <option gravity="0 0 -9.81" timestep="{DT}" iterations="10" ls_iterations="10"/>
  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="100 100 0.1" friction="0.7 0.1 0.01"/>
    <body name="chassis" pos="0 0 0.37">
      <freejoint name="base_joint"/>
      <inertial mass="85.0" pos="-0.047 0 0" diaginertia="0.6213 0.1583 0.6770"/>
      <geom type="box" pos="-0.047 0 0" size="0.13 0.3 0.09" rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
      <body name="battery" pos="-0.302 0.165 0">
        <inertial mass="2.0" pos="0 0 0" diaginertia="0.00768 0.0164 0.01208"/>
        <geom type="box" size="0.125 0.05 0.095" rgba="0.3 0.3 0.8 0.3" contype="0" conaffinity="0"/>
      </body>
      <body name="left_motor" pos="-0.09 0.14 0">
        <inertial mass="7.0" pos="0 0 0" diaginertia="0.0378 0.0084 0.0378"/>
        <geom type="box" size="0.0425 0.12 0.0425" rgba="0.8 0.3 0.3 0.3" contype="0" conaffinity="0"/>
      </body>
      <body name="right_motor" pos="-0.09 -0.14 0">
        <inertial mass="7.0" pos="0 0 0" diaginertia="0.0378 0.0084 0.0378"/>
        <geom type="box" size="0.0425 0.12 0.0425" rgba="0.8 0.3 0.3 0.3" contype="0" conaffinity="0"/>
      </body>
      <body name="rear_motor" pos="-0.22 -0.04 0">
        <inertial mass="7.0" pos="0 0 0" diaginertia="0.0378 0.0084 0.0378"/>
        <geom type="box" size="0.0425 0.12 0.0425" rgba="0.8 0.3 0.3 0.3" contype="0" conaffinity="0"/>
      </body>
      <body name="left_wheel_holder" pos="-0.477 0.095 0">
        <inertial mass="3.0" pos="0 0 0" diaginertia="0.0085 0.0981 0.0904"/>
        <geom type="box" size="0.3 0.02 0.09" rgba="0.6 0.6 0.6 0.3" contype="0" conaffinity="0"/>
      </body>
      <body name="right_wheel_holder" pos="-0.477 -0.095 0">
        <inertial mass="3.0" pos="0 0 0" diaginertia="0.0085 0.0981 0.0904"/>
        <geom type="box" size="0.3 0.02 0.09" rgba="0.6 0.6 0.6 0.3" contype="0" conaffinity="0"/>
      </body>
      <body name="left_wheel" pos="0 0.36 0">
        <joint name="left_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36" friction="0.7 0.1 0.01" rgba="0.15 0.15 0.15 1"/>
      </body>
      <body name="right_wheel" pos="0 -0.36 0">
        <joint name="right_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36" friction="0.7 0.1 0.01" rgba="0.15 0.15 0.15 1"/>
      </body>
      <body name="rear_wheel" pos="-0.697 0 0">
        <joint name="rear_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36" friction="0.35 0.1 0.01" rgba="0.15 0.15 0.15 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="left_act"  joint="left_wheel_j"  kv="100"/>
    <velocity name="right_act" joint="right_wheel_j" kv="100"/>
    <velocity name="rear_act"  joint="rear_wheel_j"  kv="100"/>
  </actuator>
</mujoco>
"""


def make_interp_matrix(T, K):
    W = np.zeros((T, K), dtype=np.float32)
    for t in range(T):
        k_float = t * (K - 1) / max(T - 1, 1)
        k_low = int(k_float)
        k_high = min(k_low + 1, K - 1)
        alpha = k_float - k_low
        W[t, k_low] += 1.0 - alpha
        W[t, k_high] += alpha
    return W


def rollout(mx, dx0, ctrl_traj):
    def step_fn(carry, ctrl_t):
        d = carry.replace(ctrl=ctrl_t)
        d = mjx.step(mx, d)
        return d, d.qpos[:2]
    _, xy_traj = jax.lax.scan(step_fn, dx0, ctrl_traj)
    return xy_traj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-worlds", type=int, default=1)
    parser.add_argument("--save", metavar="PATH")
    args = parser.parse_args()
    num_worlds = args.num_worlds

    mj_model = mujoco.MjModel.from_xml_string(HELHEST_XML)
    mx = mjx.put_model(mj_model)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    dx0 = mjx.put_data(mj_model, mj_data)

    print(f"T={T}, dt={DT}, K={K}, num_worlds={num_worlds}")

    W = jnp.array(make_interp_matrix(T, K))
    dx_batch = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_worlds), dx0)
    rollout_batch = jax.vmap(rollout, in_axes=(None, 0, None))

    target_ctrl_traj = jnp.tile(jnp.array(TARGET_CTRL), (T, 1))
    target_xy_single = jax.jit(rollout)(mx, dx0, target_ctrl_traj)
    target_xy_single.block_until_ready()
    target_xy_batch = jnp.stack([target_xy_single] * num_worlds)

    def trajectory_loss(params):
        ctrl_traj = W @ params
        xy_batch = rollout_batch(mx, dx_batch, ctrl_traj)
        delta = xy_batch - target_xy_batch
        traj = TRAJECTORY_WEIGHT / T * jnp.mean(jnp.sum(delta**2, axis=(1, 2)))
        smooth = SMOOTHNESS_WEIGHT * jnp.sum((ctrl_traj[1:] - ctrl_traj[:-1]) ** 2)
        reg = REGULARIZATION_WEIGHT * jnp.sum(ctrl_traj**2)
        return traj + smooth + reg

    value_fn = jax.jit(trajectory_loss)
    grad_fn = jax.jit(jax.jacfwd(trajectory_loss))
    params = jnp.tile(jnp.array(INIT_CTRL), (K, 1))

    print("Compiling...")
    loss = value_fn(params); loss.block_until_ready()
    grad = grad_fn(params); grad.block_until_ready()

    optimizer = optax.adam(learning_rate=0.1)
    opt_state = optimizer.init(params)
    device = jax.devices()[0]

    peak_mem_mb = 0.0
    time_ms_list = []
    for i in range(ITERATIONS):
        t0 = time.perf_counter()
        loss = value_fn(params)
        grad = grad_fn(params)
        loss.block_until_ready()
        grad.block_until_ready()
        t_iter = (time.perf_counter() - t0) * 1000

        mem_stats = device.memory_stats()
        used_mb = mem_stats.get("peak_bytes_in_use", 0) / 1024**2
        peak_mem_mb = max(peak_mem_mb, used_mb)

        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        print(f"  iter {i:3d}: loss={float(loss):.4f} | t={t_iter:.0f}ms | mem={used_mb:.0f}MB")
        time_ms_list.append(t_iter)

    results = {
        "simulator": "MJX-jacfwd",
        "num_worlds": num_worlds,
        "median_time_ms": float(np.median(time_ms_list[3:])) if len(time_ms_list) > 3 else float(np.median(time_ms_list)),
        "peak_gpu_mb": peak_mem_mb,
        "time_ms": time_ms_list,
    }
    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
