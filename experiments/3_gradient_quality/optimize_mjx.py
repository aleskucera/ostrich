"""Helhest trajectory optimization using MJX reverse-mode AD (jax.grad).

Optimizes K spline control points to match a real robot trajectory.
Uses calibrated physics params from Experiment 1.

Usage:
    python experiments/3_gradient_quality/optimize_mjx.py
    python experiments/3_gradient_quality/optimize_mjx.py \
        --ground-truth ../data/right_turn_b.json \
        --save results/mjx.json
"""
import argparse
import json
import math
import os
import pathlib
import time

os.environ.setdefault("DISPLAY", ":1")
os.environ.pop("WAYLAND_DISPLAY", None)

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np
import optax

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
DATA_DIR = pathlib.Path(__file__).parent.parent / "data"

# Calibrated params from Experiment 1 (sweep_mujoco.json), using
# dt = largest value satisfying both stability (Exp 2: 1.5 ms) and accuracy.
DT = 0.0015
KV = 4000.0
MU = 0.2

WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3

TRAJECTORY_WEIGHT = 10.0
YAW_WEIGHT = 5.0
REGULARIZATION_WEIGHT = 1e-7

HELHEST_XML = f"""
<mujoco model="helhest">
  <option gravity="0 0 -9.81" timestep="{DT}"
          solver="Newton" iterations="50" ls_iterations="50"
          cone="pyramidal" integrator="implicitfast"/>

  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="100 100 0.1"
          friction="{MU} 0.1 0.01"/>

    <body name="chassis" pos="0 0 0.37">
      <freejoint name="base_joint"/>
      <inertial mass="85.0" pos="-0.047 0 0"
                diaginertia="0.6213 0.1583 0.6770"/>
      <geom type="box" pos="-0.047 0 0" size="0.13 0.3 0.09"
            rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>

      <body name="battery" pos="-0.302 0.165 0">
        <inertial mass="2.0" pos="0 0 0" diaginertia="0.00768 0.0164 0.01208"/>
        <geom type="box" size="0.125 0.05 0.095" contype="0" conaffinity="0"/>
      </body>
      <body name="left_motor" pos="-0.09 0.14 0">
        <inertial mass="7.0" pos="0 0 0" diaginertia="0.0378 0.0084 0.0378"/>
        <geom type="box" size="0.0425 0.12 0.0425" contype="0" conaffinity="0"/>
      </body>
      <body name="right_motor" pos="-0.09 -0.14 0">
        <inertial mass="7.0" pos="0 0 0" diaginertia="0.0378 0.0084 0.0378"/>
        <geom type="box" size="0.0425 0.12 0.0425" contype="0" conaffinity="0"/>
      </body>
      <body name="rear_motor" pos="-0.22 -0.04 0">
        <inertial mass="7.0" pos="0 0 0" diaginertia="0.0378 0.0084 0.0378"/>
        <geom type="box" size="0.0425 0.12 0.0425" contype="0" conaffinity="0"/>
      </body>
      <body name="left_wheel_holder" pos="-0.477 0.095 0">
        <inertial mass="3.0" pos="0 0 0" diaginertia="0.0085 0.0981 0.0904"/>
        <geom type="box" size="0.3 0.02 0.09" contype="0" conaffinity="0"/>
      </body>
      <body name="right_wheel_holder" pos="-0.477 -0.095 0">
        <inertial mass="3.0" pos="0 0 0" diaginertia="0.0085 0.0981 0.0904"/>
        <geom type="box" size="0.3 0.02 0.09" contype="0" conaffinity="0"/>
      </body>

      <body name="left_wheel" pos="0 0.36 0">
        <joint name="left_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="{MU} 0.1 0.01"/>
      </body>
      <body name="right_wheel" pos="0 -0.36 0">
        <joint name="right_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="{MU} 0.1 0.01"/>
      </body>
      <body name="rear_wheel" pos="-0.697 0 0">
        <joint name="rear_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="{MU} 0.1 0.01"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <velocity name="left_act"  joint="left_wheel_j"  kv="{KV}"/>
    <velocity name="right_act" joint="right_wheel_j" kv="{KV}"/>
    <velocity name="rear_act"  joint="rear_wheel_j"  kv="{KV}"/>
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


def make_init_data(mx, mj_model):
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    return mjx.put_data(mj_model, mj_data)


def rollout(mx, dx0, ctrl_traj):
    """Simulate T steps. Returns (T, 2) xy and (T, 4) quaternions."""
    def step_fn(carry, ctrl_t):
        d = carry.replace(ctrl=ctrl_t)
        d = mjx.step(mx, d)
        return d, (d.qpos[:2], d.qpos[3:7])  # xy, quaternion

    _, (xy_traj, quat_traj) = jax.lax.scan(step_fn, dx0, ctrl_traj)
    return xy_traj, quat_traj


def quat_forward_xy(quat):
    """Extract forward direction (x,y) from quaternion [w,x,y,z]."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    # Rotate [1,0,0] by quaternion, take x,y
    fwd_x = 1.0 - 2.0 * (y * y + z * z)
    fwd_y = 2.0 * (x * y + w * z)
    return jnp.array([fwd_x, fwd_y])


def trajectory_loss(mx, dx0, W, params, target_xy):
    """Combined position + yaw loss."""
    ctrl_traj = W @ params  # (T, 3)
    xy_traj, quat_traj = rollout(mx, dx0, ctrl_traj)
    T = len(target_xy)

    # Position loss
    delta = xy_traj - target_xy
    pos_loss = TRAJECTORY_WEIGHT / T * jnp.sum(delta**2)

    # Yaw loss: compare robot heading with trajectory direction
    def yaw_penalty(t):
        fwd = quat_forward_xy(quat_traj[t])
        # Target direction from trajectory
        dx = target_xy[jnp.minimum(t + 1, T - 1), 0] - target_xy[t, 0]
        dy = target_xy[jnp.minimum(t + 1, T - 1), 1] - target_xy[t, 1]
        target_dir = jnp.array([dx, dy])
        target_dir = target_dir / (jnp.linalg.norm(target_dir) + 1e-8)
        dot = jnp.dot(fwd, target_dir)
        return 1.0 - dot * dot

    yaw_penalties = jax.vmap(yaw_penalty)(jnp.arange(T))
    yaw_loss = YAW_WEIGHT / T * jnp.sum(yaw_penalties)

    # Regularization
    reg = REGULARIZATION_WEIGHT / T * jnp.sum(ctrl_traj**2)

    return pos_loss + yaw_loss + reg


def load_ground_truth(path):
    with open(path) as f:
        gt = json.load(f)
    target_ctrl = gt["target_ctrl_rad_s"]
    duration = gt["trajectory"].get("constant_speed_duration_s", gt["trajectory"]["duration_s"])
    traj_xy = np.array(
        [[p["x"], p["y"]] for p in gt["trajectory"]["points"] if p["t"] <= duration],
        dtype=np.float32,
    )
    return target_ctrl, duration, traj_xy


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ground-truth", type=str,
                        default=str(DATA_DIR / "right_turn_b.json"))
    parser.add_argument("--save", metavar="PATH",
                        default=str(RESULTS_DIR / "mjx.json"))
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--init", choices=["perturbed", "zeros", "forward"],
                        default="perturbed")
    parser.add_argument("--horizon-s", type=float, default=None,
                        help="Truncate trajectory to first N seconds (default: full duration)")
    parser.add_argument("--num-trials", type=int, default=3,
                        help="Number of independent runs (different perturbed init guesses)")
    parser.add_argument("--seed-base", type=int, default=42,
                        help="First seed; trial k uses seed_base + k")
    args = parser.parse_args()

    target_ctrl, duration, traj_xy_np = load_ground_truth(args.ground_truth)
    if args.horizon_s is not None and args.horizon_s < duration:
        keep = max(2, int(args.horizon_s / duration * len(traj_xy_np)))
        traj_xy_np = traj_xy_np[:keep]
        duration = args.horizon_s
    T = int(duration / DT)

    # Resample target trajectory to match simulation steps
    real_t = np.linspace(0, 1, len(traj_xy_np))
    sim_t = np.linspace(0, 1, T)
    target_xy_np = np.zeros((T, 2), dtype=np.float32)
    target_xy_np[:, 0] = np.interp(sim_t, real_t, traj_xy_np[:, 0])
    target_xy_np[:, 1] = np.interp(sim_t, real_t, traj_xy_np[:, 1])
    target_xy = jnp.array(target_xy_np)

    print(f"Target: real robot trajectory ({len(traj_xy_np)} pts -> {T} sim steps)")
    print(f"Real robot ctrl: L={target_ctrl[0]:.3f} R={target_ctrl[1]:.3f} Rear={target_ctrl[2]:.3f}")
    print(f"T={T}, dt={DT}, K={args.K}, kv={KV}, mu={MU}, lr={args.lr}, "
          f"num_trials={args.num_trials}")

    mj_model = mujoco.MjModel.from_xml_string(HELHEST_XML)
    mx = mjx.put_model(mj_model)
    dx0 = make_init_data(mx, mj_model)

    # Spline setup (W shared across trials)
    W = jnp.array(make_interp_matrix(T, args.K))

    loss_fn = lambda p: trajectory_loss(mx, dx0, W, p, target_xy)
    value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    rollout_jit = jax.jit(rollout)

    print("Compiling value_and_grad_fn (jax.grad, reverse-mode AD)...")
    t0 = time.perf_counter()
    dummy = jnp.zeros((args.K, 3), dtype=jnp.float32)
    loss, grad = value_and_grad_fn(dummy)
    loss.block_until_ready()
    grad.block_until_ready()
    print(f"  compile: {time.perf_counter() - t0:.2f}s")

    def run_trial(init_ctrl):
        params = jnp.tile(jnp.array(init_ctrl, dtype=jnp.float32), (args.K, 1))
        schedule = optax.cosine_decay_schedule(
            init_value=args.lr, decay_steps=args.iterations, alpha=0.1
        )
        optimizer = optax.adam(learning_rate=schedule, b1=0.2, b2=0.999)
        opt_state = optimizer.init(params)

        trial = {
            "init_ctrl": list(init_ctrl),
            "iterations": [],
            "loss": [],
            "rmse_m": [],
            "time_ms": [],
            "best_iters": [],
        }
        best_loss = float("inf")

        for i in range(args.iterations):
            t0 = time.perf_counter()
            loss, grad = value_and_grad_fn(params)
            loss.block_until_ready()
            grad.block_until_ready()
            t_iter = time.perf_counter() - t0

            loss_val = float(loss)

            ctrl_traj = W @ params
            xy_traj, _ = rollout_jit(mx, dx0, ctrl_traj)
            xy_np = np.array(xy_traj)
            rmse_m = float(np.sqrt(np.mean(
                (xy_np[:, 0] - target_xy_np[:, 0])**2 +
                (xy_np[:, 1] - target_xy_np[:, 1])**2
            )))

            is_best = loss_val < best_loss
            if is_best:
                best_loss = loss_val
                trial["best_iters"].append(i)

            marker = " *" if is_best else ""
            print(f"  Iter {i:3d}: loss={loss_val:.4f} | RMSE={rmse_m:.3f}m | "
                  f"best={best_loss:.4f} | t={t_iter * 1000:.0f}ms{marker}")

            trial["iterations"].append(i)
            trial["loss"].append(loss_val)
            trial["rmse_m"].append(rmse_m)
            trial["time_ms"].append(t_iter * 1000)

            updates, opt_state = optimizer.update(grad, opt_state)
            params = optax.apply_updates(params, updates)

        trial["best_loss"] = float(best_loss)
        return trial

    trials = []
    for k in range(args.num_trials):
        seed = args.seed_base + k
        np.random.seed(seed)
        if args.init == "zeros":
            init_ctrl = [0.0, 0.0, 0.0]
        elif args.init == "forward":
            avg = float(np.mean(target_ctrl))
            init_ctrl = [avg, avg, avg]
        else:
            init_ctrl = [c + np.random.randn() * args.noise_std for c in target_ctrl]
        print(f"\n=== Trial {k + 1}/{args.num_trials} (seed={seed}) ===")
        print(f"Init ctrl ({args.init}): L={init_ctrl[0]:.3f} R={init_ctrl[1]:.3f} "
              f"Rear={init_ctrl[2]:.3f}")
        trial = run_trial(init_ctrl)
        trial["seed"] = seed
        trials.append(trial)

    aggregate = {
        "simulator": "MJX",
        "gradient_method": "jax.grad",
        "dt": DT,
        "T": T,
        "K": args.K,
        "num_trials": args.num_trials,
        "trials": trials,
    }
    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(aggregate, indent=2))
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
