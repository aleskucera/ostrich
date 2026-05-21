"""Helhest trajectory optimization using MuJoCo CPU finite differences.

Uses centered finite differences on K*nu=30 spline control parameters.
Cost: 2*30=60 forward passes per gradient step.
Runs on CPU with standard mujoco (no JAX/GPU).
"""
import argparse
import json
import pathlib
import time

import mujoco
import numpy as np

DT = 2e-3
DURATION = 3.0
T = int(DURATION / DT)  # 1500 steps
K = 10  # spline control points

TARGET_CTRL = np.array([1.0, 6.0, 0.0], dtype=np.float64)
INIT_CTRL = np.array([2.0, 5.0, 0.0], dtype=np.float64)

TRAJECTORY_WEIGHT = 10.0
SMOOTHNESS_WEIGHT = 1e-2
REGULARIZATION_WEIGHT = 1e-7

FD_EPS = 1e-4

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
    W = np.zeros((T, K), dtype=np.float64)
    for t in range(T):
        k_float = t * (K - 1) / max(T - 1, 1)
        k_low = int(k_float)
        k_high = min(k_low + 1, K - 1)
        alpha = k_float - k_low
        W[t, k_low] += 1.0 - alpha
        W[t, k_high] += alpha
    return W


def rollout(mj_model, ctrl_traj):
    """Simulate T steps. Returns (T, 2) xy trajectory."""
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    xy = np.zeros((T, 2))
    for t in range(T):
        mj_data.ctrl[:] = ctrl_traj[t]
        mujoco.mj_step(mj_model, mj_data)
        xy[t] = mj_data.qpos[:2]
    return xy


def loss_fn(mj_model, params, W, target_xy):
    ctrl_traj = W @ params
    xy = rollout(mj_model, ctrl_traj)
    delta = xy - target_xy
    traj = TRAJECTORY_WEIGHT / T * np.sum(delta**2)
    smooth = SMOOTHNESS_WEIGHT * np.sum((ctrl_traj[1:] - ctrl_traj[:-1]) ** 2)
    reg = REGULARIZATION_WEIGHT * np.sum(ctrl_traj**2)
    return traj + smooth + reg


def grad_fd(mj_model, params, W, target_xy):
    """Centered FD over all K*nu parameters. Cost: 2*K*nu forward passes."""
    grad = np.zeros_like(params)
    for idx in np.ndindex(params.shape):
        p_plus = params.copy()
        p_plus[idx] += FD_EPS
        p_minus = params.copy()
        p_minus[idx] -= FD_EPS
        grad[idx] = (
            loss_fn(mj_model, p_plus, W, target_xy)
            - loss_fn(mj_model, p_minus, W, target_xy)
        ) / (2 * FD_EPS)
    return grad


def adam_step(grad, m, v, t, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    t += 1
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    return m_hat / (np.sqrt(v_hat) + eps), m, v, t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON")
    parser.add_argument("--target-only", action="store_true", help="Only compute and save the target trajectory, skip optimization")
    args = parser.parse_args()

    mj_model = mujoco.MjModel.from_xml_string(HELHEST_XML)
    W = make_interp_matrix(T, K)
    n_params = K * mj_model.nu

    print(f"T={T}, dt={DT}, K={K} control points, FD passes={2*n_params} (centered)")

    target_ctrl_traj = np.tile(TARGET_CTRL, (T, 1))
    target_xy = rollout(mj_model, target_ctrl_traj)
    print(f"Target final xy: ({target_xy[-1, 0]:.3f}, {target_xy[-1, 1]:.3f})")

    if args.target_only:
        traj_result = {
            "simulator": "MuJoCo-FD",
            "problem": "helhest",
            "dt": DT,
            "T": T,
            "target_trajectory": target_xy.tolist(),
        }
        if args.save:
            pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(args.save).write_text(json.dumps(traj_result, indent=2))
            print(f"Saved to {args.save}")
        return

    params = np.tile(INIT_CTRL, (K, 1))
    m_adam = np.zeros_like(params)
    v_adam = np.zeros_like(params)
    t_adam = 0

    print(f"\nOptimizing: T={T}, dt={DT}, K={K}, lr=0.01 (Adam, CPU FD)")

    results = {
        "simulator": "MuJoCo-FD",
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
        loss = loss_fn(mj_model, params, W, target_xy)
        grad = grad_fd(mj_model, params, W, target_xy)
        t_iter = (time.perf_counter() - t0) * 1000

        # Clip gradient norm before Adam to prevent divergence in contact-rich landscape
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1.0:
            grad = grad / grad_norm
        update, m_adam, v_adam, t_adam = adam_step(grad, m_adam, v_adam, t_adam, lr=0.01)
        params = params - update

        p0, pm, pN = params[0], params[K // 2], params[-1]
        print(
            f"Iter {i:3d}: loss={loss:.4f} | "
            f"cp[0]=({p0[0]:.2f},{p0[1]:.2f}) "
            f"cp[{K//2}]=({pm[0]:.2f},{pm[1]:.2f}) "
            f"cp[-1]=({pN[0]:.2f},{pN[1]:.2f}) | "
            f"t={t_iter:.1f}ms"
        )
        results["iterations"].append(i)
        results["loss"].append(float(loss))
        results["time_ms"].append(t_iter)

        if loss < 1e-4:
            print("Converged!")
            break

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
