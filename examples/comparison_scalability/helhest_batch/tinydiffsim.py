"""Helhest batch benchmark — TinyDiffSim (CppAD), NUM_WORLDS sequential worlds.

TinyDiffSim supports batch simulation for its built-in robots (Ant, Laikago)
via VectorizedAntEnv/VectorizedLaikagoEnv, but has no generic batch API for
custom URDFs. Each world is therefore simulated sequentially on a single CPU
thread. This script runs NUM_WORLDS independent rollouts per iteration and
averages their gradients, mirroring the Ostrich multi-world setup.

Usage:
    python examples/comparison_scalability/helhest_batch/tinydiffsim.py
    python examples/comparison_scalability/helhest_batch/tinydiffsim.py --save results/tinydiffsim.json
"""
import argparse
import json
import pathlib
import tempfile
import time

import numpy as np
import pytinydiffsim_ad as pd

from config import NUM_WORLDS, TARGET_CTRL, INIT_CTRL, DURATION

# TinyDiffSim requires smaller timestep for stability with wheel contact.
DT = 5e-3
K = 10
T = int(DURATION / DT)  # 600 steps
NU = 3
KV = 10.0

TRAJECTORY_WEIGHT = 10.0

_HELHEST_URDF = """\
<?xml version="1.0"?>
<robot name="helhest_sphere_wheels">

  <link name="base_link">
    <collision>
      <origin xyz="-0.047 0 0"/>
      <geometry><box size="0.26 0.6 0.18"/></geometry>
    </collision>
    <inertial>
      <mass value="49"/>
      <origin xyz="-0.047 0 0"/>
      <inertia ixx="1.5" ixy="0" ixz="0" iyy="1.5" iyz="0" izz="1.5"/>
    </inertial>
  </link>

  <link name="left_wheel">
    <collision>
      <geometry><sphere radius="0.36"/></geometry>
    </collision>
    <inertial>
      <mass value="5.5"/>
      <inertia ixx="0.20045" ixy="0" ixz="0" iyy="0.20045" iyz="0" izz="0.3888"/>
    </inertial>
  </link>

  <link name="right_wheel">
    <collision>
      <geometry><sphere radius="0.36"/></geometry>
    </collision>
    <inertial>
      <mass value="5.5"/>
      <inertia ixx="0.20045" ixy="0" ixz="0" iyy="0.20045" iyz="0" izz="0.3888"/>
    </inertial>
  </link>

  <link name="rear_wheel">
    <collision>
      <geometry><sphere radius="0.36"/></geometry>
    </collision>
    <inertial>
      <mass value="5.5"/>
      <inertia ixx="0.20045" ixy="0" ixz="0" iyy="0.20045" iyz="0" izz="0.3888"/>
    </inertial>
  </link>

  <joint name="left_wheel_j" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.36 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <joint name="right_wheel_j" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.36 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <joint name="rear_wheel_j" type="continuous">
    <parent link="base_link"/>
    <child link="rear_wheel"/>
    <origin xyz="-0.697 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

</robot>
"""


def _load_robot(world):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        f.write(_HELHEST_URDF)
        urdf_path = f.name
    parser = pd.TinyUrdfParser()
    urdf_structs = parser.load_urdf(urdf_path)
    mb = pd.TinyMultiBody(True)
    pd.UrdfToMultiBody2().convert2(urdf_structs, world, mb)
    return mb


def _make_ground(world):
    urdf = pd.TinyUrdfStructures()
    bl = pd.TinyUrdfLink()
    bl.link_name = "ground"
    ine = pd.TinyUrdfInertial()
    ine.mass = pd.ADDouble(0.0)
    ine.inertia_xxyyzz = pd.Vector3(0.0, 0.0, 0.0)
    bl.urdf_inertial = ine
    col = pd.TinyUrdfCollision()
    col.geometry.geom_type = pd.PLANE_TYPE
    bl.urdf_collision_shapes = [col]
    urdf.base_links = [bl]
    mb = pd.TinyMultiBody(False)
    pd.UrdfToMultiBody2().convert2(urdf, world, mb)
    return mb


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


def rollout_ad(params_ad, W_ad):
    world = pd.TinyWorld()
    world.gravity = pd.Vector3(0.0, 0.0, -9.81)
    world.friction = pd.ADDouble(0.7)

    ground_mb = _make_ground(world)
    robot_mb = _load_robot(world)

    robot_mb.q[0] = pd.ADDouble(0.0)
    robot_mb.q[1] = pd.ADDouble(0.0)
    robot_mb.q[2] = pd.ADDouble(0.0)
    robot_mb.q[3] = pd.ADDouble(1.0)
    robot_mb.q[4] = pd.ADDouble(0.0)
    robot_mb.q[5] = pd.ADDouble(0.0)
    robot_mb.q[6] = pd.ADDouble(0.36)

    mbs = [ground_mb, robot_mb]
    dispatcher = world.get_collision_dispatcher()
    solver = pd.TinyMultiBodyConstraintSolver()
    dt = pd.ADDouble(DT)

    WHEEL_QD = [6, 7, 8]
    WHEEL_TAU = [0, 1, 2]

    xy_traj = []
    for step in range(T):
        ctrl = []
        for nu in range(NU):
            val = pd.ADDouble(0.0)
            for k in range(K):
                w = float(W_ad[step, k])
                if w != 0.0:
                    val = val + params_ad[k * NU + nu] * w
            ctrl.append(val)

        for i in range(NU):
            robot_mb.tau[WHEEL_TAU[i]] = (ctrl[i] - robot_mb.qd[WHEEL_QD[i]]) * pd.ADDouble(KV)

        pd.forward_kinematics(robot_mb, robot_mb.q, robot_mb.qd)
        pd.forward_dynamics(robot_mb, world.gravity)
        contacts = world.compute_contacts_multi_body(mbs, dispatcher)
        flat_contacts = [c for sublist in contacts for c in sublist]
        solver.resolve_collision(flat_contacts, dt)
        pd.integrate_euler(robot_mb, dt)

        xy_traj.append((robot_mb.q[4], robot_mb.q[5]))

    return xy_traj


def rollout_float(params_np, W):
    params_ad = [pd.ADDouble(float(v)) for v in params_np.flatten()]
    traj = rollout_ad(params_ad, W)
    return np.array([[x.value(), y.value()] for x, y in traj])


def grad_and_loss_ad(params_np, W, target_xy):
    params_flat = params_np.flatten()
    p_ad = [pd.ADDouble(float(v)) for v in params_flat]
    p_ind = pd.independent(p_ad)

    traj = rollout_ad(p_ind, W)

    loss_ad = pd.ADDouble(0.0)
    for step, (x, y) in enumerate(traj):
        dx = x - pd.ADDouble(float(target_xy[step, 0]))
        dy = y - pd.ADDouble(float(target_xy[step, 1]))
        loss_ad = loss_ad + (dx * dx + dy * dy) * pd.ADDouble(TRAJECTORY_WEIGHT / T)

    fn = pd.ADFun(p_ind, [loss_ad])
    fn.optimize()

    jac = fn.Jacobian(list(params_flat.astype(float)))
    return float(loss_ad.value()), np.array(jac).reshape(params_np.shape)


def adam_step(grad, m, v, t, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    t += 1
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    return m_hat / (np.sqrt(v_hat) + eps), m, v, t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH")
    # Fewer iterations because NUM_WORLDS sequential runs per iter is expensive
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    W = make_interp_matrix(T, K)

    target_params = np.tile(TARGET_CTRL, (K, 1))
    print("Computing target trajectory...")
    target_xy = rollout_float(target_params, W)
    print(f"Target final xy: ({target_xy[-1, 0]:.3f}, {target_xy[-1, 1]:.3f})")

    params = np.tile(INIT_CTRL, (K, 1)).astype(float)
    m_adam = np.zeros_like(params)
    v_adam = np.zeros_like(params)
    t_adam = 0

    print(
        f"\nOptimizing: T={T}, dt={DT}, K={K}, num_worlds={NUM_WORLDS} "
        f"(TinyDiffSim — sequential, no batch API for custom URDFs)"
    )

    results = {
        "simulator": "TinyDiffSim",
        "problem": "helhest_batch",
        "dt": DT,
        "T": T,
        "K": K,
        "num_worlds": NUM_WORLDS,
        "iterations": [],
        "loss": [],
        "time_ms": [],
        "peak_gpu_mb": None,
    }

    for i in range(args.iterations):
        t0 = time.perf_counter()

        # Run NUM_WORLDS sequential rollouts and average gradients/loss
        total_loss = 0.0
        total_grad = np.zeros_like(params)
        for _ in range(NUM_WORLDS):
            loss, grad = grad_and_loss_ad(params, W, target_xy)
            total_loss += loss
            total_grad += grad
        avg_loss = total_loss / NUM_WORLDS
        avg_grad = total_grad / NUM_WORLDS

        t_iter = (time.perf_counter() - t0) * 1000

        grad_norm = np.linalg.norm(avg_grad)
        if grad_norm > 1.0:
            avg_grad = avg_grad / grad_norm
        update, m_adam, v_adam, t_adam = adam_step(avg_grad, m_adam, v_adam, t_adam, lr=0.01)
        params = params - update

        print(
            f"Iter {i:3d}: loss={avg_loss:.4f} | num_worlds={NUM_WORLDS} | t={t_iter:.0f}ms"
        )
        results["iterations"].append(i)
        results["loss"].append(float(avg_loss))
        results["time_ms"].append(t_iter)

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
