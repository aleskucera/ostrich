"""Ball throw optimization using MuJoCo MJX (JAX-based differentiable physics).

Comparable to examples/comparison/ball_throw/ball_throw_axion.py.

Optimizes the initial 3D linear velocity of a ball to match a target trajectory.
Uses gradient descent on the 3D initial velocity via jax.jacfwd (3 forward passes).
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

DT = 3e-2
DURATION = 1.5
T = int(DURATION / DT)  # 50 steps

# Initial guess and target for the ball's linear velocity [vx, vy, vz]
INIT_VEL = np.array([0.0, 2.0, 1.0], dtype=np.float32)
TARGET_VEL = np.array([0.0, 4.0, 7.0], dtype=np.float32)

LEARNING_RATE = 2e-2
MAX_GRAD = 100.0

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


def make_init_data(mx, mj_model):
    """Create MJX data with the ball at its starting position and zero velocity."""
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    return mjx.put_data(mj_model, mj_data)


def rollout(mx, dx0, vel):
    """Simulate T steps from dx0 with initial linear velocity vel (3,).
    Returns (T, 3) xyz positions of the ball.
    """
    dx = dx0.replace(qvel=dx0.qvel.at[:3].set(vel))

    def step_fn(carry, _):
        d = mjx.step(mx, carry)
        return d, d.qpos[:3]

    _, xyz_traj = jax.lax.scan(step_fn, dx, None, length=T)
    return xyz_traj  # (T, 3)


def ball_loss(mx, dx0, vel, target_xyz_traj):
    xyz_traj = rollout(mx, dx0, vel)
    delta = xyz_traj - target_xyz_traj
    return jnp.sum(delta**2)


def simulate_cpu(mj_model, init_vel_np, label=""):
    """Run on CPU with viewer. init_vel_np: (3,) initial linear velocity."""
    import time as _time

    import mujoco.viewer

    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    mj_data.qvel[:3] = init_vel_np

    print(f"  {label} {'step':>5}  {'x':>8}  {'y':>8}  {'z':>8}")
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.distance = 10.0
        viewer.cam.elevation = -20.0
        viewer.cam.azimuth = 90.0
        for step in range(T):
            step_start = _time.time()
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()
            if step % 10 == 0 or step == T - 1:
                x, y, z = mj_data.qpos[0], mj_data.qpos[1], mj_data.qpos[2]
                print(f"         {step:>5}  {x:>8.3f}  {y:>8.3f}  {z:>8.3f}")
            if not viewer.is_running():
                break
            elapsed = _time.time() - step_start
            _time.sleep(max(0.0, mj_model.opt.timestep - elapsed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON and skip CPU viewer")
    args = parser.parse_args()

    mj_model = mujoco.MjModel.from_xml_string(BALL_XML)
    mx = mjx.put_model(mj_model)
    dx0 = make_init_data(mx, mj_model)

    print(f"T={T}, dt={DT}, params=3 (initial linear velocity), reverse-mode AD (jax.grad)")

    # --- Visualize target episode on CPU ---
    if not args.save:
        print("Simulating target episode (CPU)...")
        simulate_cpu(mj_model, TARGET_VEL, label="target")

    # --- Compute target trajectory on GPU ---
    target_xyz_traj = jax.jit(rollout)(mx, dx0, jnp.array(TARGET_VEL))
    target_xyz_traj.block_until_ready()
    print(
        f"Target final xyz: ({target_xyz_traj[-1, 0]:.3f}, {target_xyz_traj[-1, 1]:.3f}, {target_xyz_traj[-1, 2]:.3f})"
    )

    # Solver uses _while_loop_scan (scan-based), so jax.grad works. 1 backward pass.
    loss_fn = lambda v: ball_loss(mx, dx0, v, target_xyz_traj)
    value_fn = jax.jit(loss_fn)
    grad_fn = jax.jit(jax.grad(loss_fn))

    print("Compiling value_fn + grad_fn (reverse-mode AD, 1 backward pass)...")
    t0 = time.perf_counter()
    vel = jnp.array(INIT_VEL)
    loss = value_fn(vel)
    loss.block_until_ready()
    grad = grad_fn(vel)
    grad.block_until_ready()
    print(f"  compile: {time.perf_counter() - t0:.2f}s\n")

    print(f"Optimizing: T={T}, dt={DT}, lr={LEARNING_RATE} (gradient descent, reverse-mode AD)")
    results = {
        "simulator": "MJX-grad",
        "problem": "ball_throw",
        "dt": DT,
        "T": T,
        "iterations": [],
        "loss": [],
        "time_ms": [],
    }
    for i in range(30):
        t0 = time.perf_counter()
        loss = value_fn(vel)
        grad = grad_fn(vel)
        loss.block_until_ready()
        grad.block_until_ready()
        t_iter = time.perf_counter() - t0

        grad_clamped = jnp.clip(grad, -MAX_GRAD, MAX_GRAD)
        vel = vel - LEARNING_RATE * grad_clamped

        print(
            f"Iter {i:3d}: loss={loss:.4f} | "
            f"vel=({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}) | "
            f"grad=({grad[0]:.3f}, {grad[1]:.3f}, {grad[2]:.3f}) | "
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
    else:
        print(f"\nOptimized velocity: ({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f})")
        print(
            f"Target velocity:    ({TARGET_VEL[0]:.3f}, {TARGET_VEL[1]:.3f}, {TARGET_VEL[2]:.3f})"
        )
        print("\nSimulating optimized episode (CPU)...")
        simulate_cpu(mj_model, np.array(vel), label="optimized")


if __name__ == "__main__":
    main()
