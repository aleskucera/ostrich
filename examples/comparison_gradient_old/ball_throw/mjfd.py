"""Ball throw optimization using MuJoCo CPU finite differences.

Uses centered finite differences on the 3 initial velocity parameters.
Cost: 2*3=6 forward passes per gradient step (vs 3 for jacfwd, 1 for jax.grad).
Runs on CPU with standard mujoco (no JAX/GPU).
"""
import argparse
import json
import pathlib
import time

import mujoco
import numpy as np

DT = 3e-2
DURATION = 1.5
T = int(DURATION / DT)  # 50 steps

INIT_VEL = np.array([0.0, 2.0, 1.0], dtype=np.float64)
TARGET_VEL = np.array([0.0, 4.0, 7.0], dtype=np.float64)

LEARNING_RATE = 2e-2
MAX_GRAD = 100.0
FD_EPS = 1e-5

BALL_XML = f"""
<mujoco model="ball_throw">
  <option gravity="0 0 -9.81" timestep="{DT}" iterations="10" ls_iterations="10"/>
  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="100 100 0.1"
          friction="0.7 0.1 0.01"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <inertial mass="1.0" pos="0 0 0" diaginertia="0.016 0.016 0.016"/>
      <geom type="sphere" size="0.2" friction="0.7 0.1 0.01"
            solref="0.02 1" solimp="0.9 0.95 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""


def rollout(mj_model, vel):
    """Simulate T steps from initial linear velocity vel. Returns (T, 3) xyz."""
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    mj_data.qvel[:3] = vel
    xyz = np.zeros((T, 3))
    for t in range(T):
        mujoco.mj_step(mj_model, mj_data)
        xyz[t] = mj_data.qpos[:3]
    return xyz


def loss_fn(mj_model, vel, target_xyz):
    xyz = rollout(mj_model, vel)
    return np.sum((xyz - target_xyz) ** 2)


def grad_fd(mj_model, vel, target_xyz):
    """Centered finite differences: 2*n_params forward passes."""
    grad = np.zeros_like(vel)
    for i in range(len(vel)):
        v_plus = vel.copy()
        v_plus[i] += FD_EPS
        v_minus = vel.copy()
        v_minus[i] -= FD_EPS
        grad[i] = (
            loss_fn(mj_model, v_plus, target_xyz)
            - loss_fn(mj_model, v_minus, target_xyz)
        ) / (2 * FD_EPS)
    return grad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON")
    args = parser.parse_args()

    mj_model = mujoco.MjModel.from_xml_string(BALL_XML)

    print(f"T={T}, dt={DT}, params=3 (initial linear velocity), FD passes=6 (centered)")

    target_xyz = rollout(mj_model, TARGET_VEL)
    print(
        f"Target final xyz: ({target_xyz[-1, 0]:.3f}, {target_xyz[-1, 1]:.3f}, {target_xyz[-1, 2]:.3f})"
    )

    vel = INIT_VEL.copy()
    print(f"\nOptimizing: T={T}, dt={DT}, lr={LEARNING_RATE} (gradient descent, CPU FD)")

    results = {
        "simulator": "MuJoCo-FD",
        "problem": "ball_throw",
        "dt": DT,
        "T": T,
        "iterations": [],
        "loss": [],
        "time_ms": [],
    }

    for i in range(30):
        t0 = time.perf_counter()
        loss = loss_fn(mj_model, vel, target_xyz)
        grad = grad_fd(mj_model, vel, target_xyz)
        t_iter = (time.perf_counter() - t0) * 1000

        grad_clamped = np.clip(grad, -MAX_GRAD, MAX_GRAD)
        vel = vel - LEARNING_RATE * grad_clamped

        print(
            f"Iter {i:3d}: loss={loss:.4f} | "
            f"vel=({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}) | "
            f"grad=({grad[0]:.3f}, {grad[1]:.3f}, {grad[2]:.3f}) | "
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
