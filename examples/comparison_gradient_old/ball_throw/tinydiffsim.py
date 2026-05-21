"""Ball throw optimization using tiny-differentiable-simulator (CppAD reverse-mode AD).

Optimizes the initial linear velocity of a sphere (3 parameters) to match a
target trajectory using gradient descent.  Gradients are computed via CppAD
reverse-mode AD recorded through the full forward simulation (including contact).

Contact is handled by TinyDiffSim's impulse-based LCP/PGS solver.  CppAD
differentiates through the piecewise-linear contact impulses using sub-gradients,
which is valid but may produce zero gradients at contact events.

Usage:
    python examples/comparison_gradient/ball_throw/tinydiffsim.py
    python examples/comparison_gradient/ball_throw/tinydiffsim.py --save results/tinydiffsim.json
"""
import argparse
import json
import pathlib
import time

import numpy as np
import pytinydiffsim_ad as pd

from config import DT, DURATION, INIT_VEL, TARGET_VEL

T = int(DURATION / DT)

# Optimization hyper-params (match MuJoCo-FD)
LEARNING_RATE = 2e-2
MAX_GRAD = 100.0


def rollout_ad(vx: pd.ADDouble, vy: pd.ADDouble, vz: pd.ADDouble):
    """Run T steps and return all ball xyz positions as list of T lists of 3 ADDouble."""
    world = pd.TinyWorld()
    world.gravity = pd.Vector3(0.0, 0.0, -9.81)

    # Keep geom objects alive for the duration of the rollout
    ball_geom = pd.TinySphere(pd.ADDouble(0.2))
    ball = pd.TinyRigidBody(pd.ADDouble(1.0), ball_geom)
    ball.world_pose.position = pd.Vector3(0.0, 0.0, 1.0)
    ball.world_pose.orientation = pd.Quaternion(0.0, 0.0, 0.0, 1.0)
    ball.linear_velocity = pd.Vector3(vx, vy, vz)
    ball.angular_velocity = pd.Vector3(pd.ADDouble(0.0), pd.ADDouble(0.0), pd.ADDouble(0.0))

    ground_geom = pd.TinyPlane()
    ground = pd.TinyRigidBody(pd.ADDouble(0.0), ground_geom)
    ground.world_pose.position = pd.Vector3(0.0, 0.0, 0.0)
    ground.world_pose.orientation = pd.Quaternion(0.0, 0.0, 0.0, 1.0)

    bodies = [ball, ground]

    dispatcher = world.get_collision_dispatcher()
    rb_solver = pd.TinyConstraintSolver()
    dt = pd.ADDouble(DT)

    positions = []
    for _ in range(T):
        for b in bodies:
            b.apply_gravity(world.gravity)
            b.apply_force_impulse(dt)
            b.clear_forces()
        contacts = world.compute_contacts_rigid_body(bodies, dispatcher)
        for _ in range(10):
            for c in contacts:
                rb_solver.resolve_collision(c, dt)
        for b in bodies:
            b.integrate(dt)
        pos = ball.world_pose.position
        positions.append([pos[0], pos[1], pos[2]])

    return positions  # list of T lists of 3 ADDouble


def rollout_float(vel):
    """Plain float rollout — returns (T, 3) array of xyz positions at each step."""
    vx, vy, vz = [pd.ADDouble(float(v)) for v in vel]
    positions = rollout_ad(vx, vy, vz)
    return np.array([[p[i].value() for i in range(3)] for p in positions])  # (T, 3)


def grad_and_loss_ad(vel, target_traj):
    """Compute loss + gradient via CppAD.  vel: (3,) float array, target_traj: (T, 3)."""
    v_ad = [pd.ADDouble(float(v)) for v in vel]
    v_ind = pd.independent(v_ad)

    positions = rollout_ad(v_ind[0], v_ind[1], v_ind[2])

    # L2 loss over full trajectory
    loss_ad = pd.ADDouble(0.0)
    for t in range(T):
        for i in range(3):
            diff = positions[t][i] - pd.ADDouble(float(target_traj[t, i]))
            loss_ad = loss_ad + diff * diff

    fn = pd.ADFun(v_ind, [loss_ad])
    fn.optimize()

    jac = fn.Jacobian(list(vel.astype(float)))  # Jacobian is (1 x 3) flat
    loss_val = float(loss_ad.value())
    return loss_val, np.array(jac)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH")
    args = parser.parse_args()

    # Compute target trajectory using float module
    target_traj = rollout_float(np.array(TARGET_VEL, dtype=float))  # (T, 3)
    print(f"Target final xyz: ({target_traj[-1, 0]:.3f}, {target_traj[-1, 1]:.3f}, {target_traj[-1, 2]:.3f})")

    vel = np.array(INIT_VEL, dtype=float)

    print(f"\nOptimizing: T={T}, dt={DT}, lr={LEARNING_RATE} (TinyDiffSim, CppAD reverse-mode)")

    results = {
        "simulator": "TinyDiffSim",
        "problem": "ball_throw",
        "dt": DT,
        "T": T,
        "iterations": [],
        "loss": [],
        "time_ms": [],
    }

    for i in range(30):
        t0 = time.perf_counter()
        loss, grad = grad_and_loss_ad(vel, target_traj)
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
