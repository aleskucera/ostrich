"""Helhest trajectory optimization using MuJoCo MJX — forward-mode AD (jacfwd).

Uses jax.jacfwd: K*nu=30 forward passes per gradient step.
The while_loop in the constraint solver is differentiable in forward mode.
"""
import argparse
import json
import os
import pathlib
import time

os.environ.setdefault("DISPLAY", ":1")
os.environ.pop("WAYLAND_DISPLAY", None)  # force GLFW to use X11 via XWayland

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np
import optax

DT = 2e-3
DURATION = 3.0
T = int(DURATION / DT)  # 250 steps
K = 10  # number of spline control points

TARGET_CTRL = np.array([1.0, 6.0, 0.0], dtype=np.float32)
INIT_CTRL = np.array([2.0, 5.0, 0.0], dtype=np.float32)

TRAJECTORY_WEIGHT = 10.0
SMOOTHNESS_WEIGHT = 1e-2
REGULARIZATION_WEIGHT = 1e-7

HELHEST_XML = f"""
<mujoco model="helhest">
  <option gravity="0 0 -9.81" timestep="{DT}" iterations="10" ls_iterations="10"/>

  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="100 100 0.1"
          friction="0.7 0.1 0.01"/>

    <body name="chassis" pos="0 0 0.37">
      <freejoint name="base_joint"/>
      <inertial mass="85.0" pos="-0.047 0 0"
                diaginertia="0.6213 0.1583 0.6770"/>
      <geom type="box" pos="-0.047 0 0" size="0.13 0.3 0.09"
            rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>

      <body name="battery" pos="-0.302 0.165 0">
        <inertial mass="2.0" pos="0 0 0" diaginertia="0.00768 0.0164 0.01208"/>
        <geom type="box" size="0.125 0.05 0.095" rgba="0.3 0.3 0.8 0.3"
              contype="0" conaffinity="0"/>
      </body>
      <body name="left_motor" pos="-0.09 0.14 0">
        <inertial mass="7.0" pos="0 0 0" diaginertia="0.0378 0.0084 0.0378"/>
        <geom type="box" size="0.0425 0.12 0.0425" rgba="0.8 0.3 0.3 0.3"
              contype="0" conaffinity="0"/>
      </body>
      <body name="right_motor" pos="-0.09 -0.14 0">
        <inertial mass="7.0" pos="0 0 0" diaginertia="0.0378 0.0084 0.0378"/>
        <geom type="box" size="0.0425 0.12 0.0425" rgba="0.8 0.3 0.3 0.3"
              contype="0" conaffinity="0"/>
      </body>
      <body name="rear_motor" pos="-0.22 -0.04 0">
        <inertial mass="7.0" pos="0 0 0" diaginertia="0.0378 0.0084 0.0378"/>
        <geom type="box" size="0.0425 0.12 0.0425" rgba="0.8 0.3 0.3 0.3"
              contype="0" conaffinity="0"/>
      </body>
      <body name="left_wheel_holder" pos="-0.477 0.095 0">
        <inertial mass="3.0" pos="0 0 0" diaginertia="0.0085 0.0981 0.0904"/>
        <geom type="box" size="0.3 0.02 0.09" rgba="0.6 0.6 0.6 0.3"
              contype="0" conaffinity="0"/>
      </body>
      <body name="right_wheel_holder" pos="-0.477 -0.095 0">
        <inertial mass="3.0" pos="0 0 0" diaginertia="0.0085 0.0981 0.0904"/>
        <geom type="box" size="0.3 0.02 0.09" rgba="0.6 0.6 0.6 0.3"
              contype="0" conaffinity="0"/>
      </body>

      <body name="left_wheel" pos="0 0.36 0">
        <joint name="left_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="0.7 0.1 0.01" rgba="0.15 0.15 0.15 1"/>
      </body>
      <body name="right_wheel" pos="0 -0.36 0">
        <joint name="right_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="0.7 0.1 0.01" rgba="0.15 0.15 0.15 1"/>
      </body>
      <body name="rear_wheel" pos="-0.697 0 0">
        <joint name="rear_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="0.35 0.1 0.01" rgba="0.15 0.15 0.15 1"/>
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


def make_interp_matrix(T: int, K: int) -> np.ndarray:
    W = np.zeros((T, K), dtype=np.float32)
    for t in range(T):
        k_float = t * (K - 1) / max(T - 1, 1)
        k_low = int(k_float)
        k_high = min(k_low + 1, K - 1)
        alpha = k_float - k_low
        W[t, k_low] += 1.0 - alpha
        W[t, k_high] += alpha
    return W


def make_init_data(mx, mj_model):
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    return mjx.put_data(mj_model, mj_data)


def rollout(mx, dx0, ctrl_traj):
    def step_fn(carry, ctrl_t):
        d = carry.replace(ctrl=ctrl_t)
        d = mjx.step(mx, d)
        return d, d.qpos[:2]

    _, xy_traj = jax.lax.scan(step_fn, dx0, ctrl_traj)
    return xy_traj


def trajectory_loss(mx, dx0, W, params, target_xy_traj):
    ctrl_traj = W @ params
    xy_traj = rollout(mx, dx0, ctrl_traj)
    delta = xy_traj - target_xy_traj
    traj = TRAJECTORY_WEIGHT / T * jnp.sum(delta**2)
    smooth = SMOOTHNESS_WEIGHT * jnp.sum((ctrl_traj[1:] - ctrl_traj[:-1]) ** 2)
    reg = REGULARIZATION_WEIGHT * jnp.sum(ctrl_traj**2)
    return traj + smooth + reg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON")
    parser.add_argument("--target-only", action="store_true", help="Only compute and save the target trajectory, skip optimization")
    args = parser.parse_args()

    mj_model = mujoco.MjModel.from_xml_string(HELHEST_XML)
    mx = mjx.put_model(mj_model)
    dx0 = make_init_data(mx, mj_model)

    print(f"T={T}, dt={DT}, K={K} control points, jacfwd passes={K * mj_model.nu}")

    target_ctrl_traj = jnp.tile(jnp.array(TARGET_CTRL), (T, 1))
    target_xy_traj = jax.jit(rollout)(mx, dx0, target_ctrl_traj)
    target_xy_traj.block_until_ready()
    print(f"Target final xy: ({target_xy_traj[-1, 0]:.3f}, {target_xy_traj[-1, 1]:.3f})")

    if args.target_only:
        traj_result = {
            "simulator": "MJX-jacfwd",
            "problem": "helhest",
            "dt": DT,
            "T": T,
            "target_trajectory": np.array(target_xy_traj).tolist(),
        }
        if args.save:
            pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(args.save).write_text(json.dumps(traj_result, indent=2))
            print(f"Saved to {args.save}")
        return

    W = jnp.array(make_interp_matrix(T, K))
    params = jnp.tile(jnp.array(INIT_CTRL), (K, 1))

    # jacfwd works through while_loop. Cost: K*nu forward passes per iteration.
    loss_fn = lambda p: trajectory_loss(mx, dx0, W, p, target_xy_traj)
    value_fn = jax.jit(loss_fn)
    grad_fn = jax.jit(jax.jacfwd(loss_fn))

    print(f"Compiling value_fn + grad_fn ({K * mj_model.nu} forward passes)...")
    t0 = time.perf_counter()
    loss = value_fn(params)
    loss.block_until_ready()
    grad = grad_fn(params)
    grad.block_until_ready()
    print(f"  compile: {time.perf_counter() - t0:.2f}s\n")

    optimizer = optax.adam(learning_rate=0.1)
    opt_state = optimizer.init(params)

    print(f"Optimizing: T={T}, dt={DT}, K={K}, lr=0.1 (Adam, forward-mode AD)")
    results = {
        "simulator": "MJX-jacfwd",
        "problem": "helhest",
        "dt": DT,
        "T": T,
        "K": K,
        "iterations": [],
        "loss": [],
        "time_ms": [],
    }
    for i in range(50):
        t0 = time.perf_counter()
        loss = value_fn(params)
        grad = grad_fn(params)
        loss.block_until_ready()
        grad.block_until_ready()
        t_iter = time.perf_counter() - t0

        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        p0, pm, pN = params[0], params[K // 2], params[-1]
        print(
            f"Iter {i:3d}: loss={loss:.4f} | "
            f"cp[0]=({p0[0]:.2f},{p0[1]:.2f}) "
            f"cp[{K//2}]=({pm[0]:.2f},{pm[1]:.2f}) "
            f"cp[-1]=({pN[0]:.2f},{pN[1]:.2f}) | "
            f"t={t_iter * 1000:.1f}ms"
        )
        results["iterations"].append(i)
        results["loss"].append(float(loss))
        results["time_ms"].append(t_iter * 1000)

        if loss < 1e-4:
            print("Converged!")
            break

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
