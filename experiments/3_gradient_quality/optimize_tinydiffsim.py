"""Helhest trajectory optimization using TinyDiffSim (CppAD reverse-mode).

Optimizes K spline control points to match a real robot trajectory.
Uses calibrated physics params from Experiment 1.

Usage:
    python experiments/3_gradient_quality/optimize_tinydiffsim.py
    python experiments/3_gradient_quality/optimize_tinydiffsim.py \
        --ground-truth ../data/right_turn_b.json \
        --save results/tinydiffsim.json
"""
import argparse
import json
import math
import pathlib
import tempfile
import time

import numpy as np

try:
    import pytinydiffsim_ad as pd  # Google tiny-differentiable-simulator (AD build)
except ModuleNotFoundError:
    import pydiffarti as pd  # Qiao diffarticulated fork (same API)

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
DATA_DIR = pathlib.Path(__file__).parent.parent / "data"

# Calibrated params from Experiment 1 (sweep_tinydiffsim.json), using
# dt = largest accuracy-acceptable value (Exp 1 dt sweep: 2 ms, err 0.96 m).
# No obstacle stability test for TinyDiffSim (no box collision support).
DT = 0.002
KV = 36.0
FRICTION = 0.12

WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3
NU = 3
TRAJECTORY_WEIGHT = 10.0
YAW_WEIGHT = 5.0
REGULARIZATION_WEIGHT = 1e-7

WHEEL_QD = [6, 7, 8]
WHEEL_TAU = [0, 1, 2]

_HELHEST_URDF = """\
<?xml version="1.0"?>
<robot name="helhest_sphere_wheels">
  <link name="base_link">
    <collision><origin xyz="-0.047 0 0"/><geometry><box size="0.26 0.6 0.18"/></geometry></collision>
    <inertial><mass value="85"/><origin xyz="-0.047 0 0"/>
      <inertia ixx="0.6213" ixy="0" ixz="0" iyy="0.1583" iyz="0" izz="0.677"/></inertial>
  </link>
  <link name="left_wheel">
    <collision><geometry><sphere radius="0.36"/></geometry></collision>
    <inertial><mass value="5.5"/>
      <inertia ixx="0.20045" ixy="0" ixz="0" iyy="0.20045" iyz="0" izz="0.3888"/></inertial>
  </link>
  <link name="right_wheel">
    <collision><geometry><sphere radius="0.36"/></geometry></collision>
    <inertial><mass value="5.5"/>
      <inertia ixx="0.20045" ixy="0" ixz="0" iyy="0.20045" iyz="0" izz="0.3888"/></inertial>
  </link>
  <link name="rear_wheel">
    <collision><geometry><sphere radius="0.36"/></geometry></collision>
    <inertial><mass value="5.5"/>
      <inertia ixx="0.20045" ixy="0" ixz="0" iyy="0.20045" iyz="0" izz="0.3888"/></inertial>
  </link>
  <joint name="left_wheel_j" type="continuous">
    <parent link="base_link"/><child link="left_wheel"/>
    <origin xyz="0 0.36 0"/><axis xyz="0 1 0"/>
  </joint>
  <joint name="right_wheel_j" type="continuous">
    <parent link="base_link"/><child link="right_wheel"/>
    <origin xyz="0 -0.36 0"/><axis xyz="0 1 0"/>
  </joint>
  <joint name="rear_wheel_j" type="continuous">
    <parent link="base_link"/><child link="rear_wheel"/>
    <origin xyz="-0.697 0 0"/><axis xyz="0 1 0"/>
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


def make_interp_matrix(T, K):
    W = np.zeros((T, K), dtype=np.float64)
    for t in range(T):
        k_float = t * (K - 1) / max(T - 1, 1)
        k_low = int(k_float)
        k_high = min(k_low + 1, K - 1)
        alpha = k_float - k_low
        W[t, k_low] += 1.0 - alpha
        W[t, k_high] += alpha
    return W


class SplineAdam:
    def __init__(
        self,
        K,
        num_dofs,
        lr,
        total_steps=200,
        lr_min_ratio=0.05,
        betas=(0.2, 0.999),
        eps=1e-8,
        grad_clip=None,
    ):
        self.lr_init = lr
        self.lr_min = lr * lr_min_ratio
        self.total_steps = total_steps
        self.eps = eps
        self.beta1, self.beta2 = betas
        self.grad_clip = grad_clip
        self.m = np.zeros((K, num_dofs), dtype=np.float64)
        self.v = np.zeros((K, num_dofs), dtype=np.float64)
        self.t = 0

    def _cosine_lr(self):
        progress = min(self.t / self.total_steps, 1.0)
        return self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (1.0 + np.cos(np.pi * progress))

    def step(self, params, grad):
        self.t += 1
        if self.grad_clip is not None:
            grad_norm = np.linalg.norm(grad)
            if grad_norm > self.grad_clip:
                grad = grad * (self.grad_clip / grad_norm)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        lr = self._cosine_lr()
        return params - lr * m_hat / (np.sqrt(v_hat) + self.eps)


def rollout_ad(params_ad, W_ad, T):
    world = pd.TinyWorld()
    world.gravity = pd.Vector3(0.0, 0.0, -9.81)
    world.friction = pd.ADDouble(FRICTION)

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

    K = W_ad.shape[1]
    xy_traj = []
    quat_traj = []

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
        # quaternion: q[0..3] = (x,y,z,w) in TinyDiffSim
        quat_traj.append((robot_mb.q[0], robot_mb.q[1], robot_mb.q[2], robot_mb.q[3]))

    return xy_traj, quat_traj


def grad_and_loss_ad(params_np, W, target_xy, T):
    K = W.shape[1]
    params_flat = params_np.flatten()
    p_ad = [pd.ADDouble(float(v)) for v in params_flat]
    p_ind = pd.independent(p_ad)

    xy_traj, quat_traj = rollout_ad(p_ind, W, T)

    loss_ad = pd.ADDouble(0.0)

    # Position loss
    for step, (x, y) in enumerate(xy_traj):
        dx = x - pd.ADDouble(float(target_xy[step, 0]))
        dy = y - pd.ADDouble(float(target_xy[step, 1]))
        loss_ad = loss_ad + (dx * dx + dy * dy) * pd.ADDouble(TRAJECTORY_WEIGHT / T)

    # Yaw loss
    for step in range(len(xy_traj) - 1):
        # Robot forward from quaternion (simplified: use xy velocity direction)
        # TinyDiffSim quaternion is (x,y,z,w)
        qx, qy, qz, qw = quat_traj[step]
        # Forward direction x component: 1 - 2(qy^2 + qz^2)
        fwd_x = pd.ADDouble(1.0) - (qy * qy + qz * qz) * pd.ADDouble(2.0)
        # Forward direction y component: 2(qx*qy + qw*qz)
        fwd_y = (qx * qy + qw * qz) * pd.ADDouble(2.0)

        # Target direction
        tdx = float(target_xy[step + 1, 0] - target_xy[step, 0])
        tdy = float(target_xy[step + 1, 1] - target_xy[step, 1])
        tn = math.sqrt(tdx**2 + tdy**2) + 1e-8
        tdx /= tn
        tdy /= tn

        dot = fwd_x * pd.ADDouble(tdx) + fwd_y * pd.ADDouble(tdy)
        loss_ad = loss_ad + (pd.ADDouble(1.0) - dot * dot) * pd.ADDouble(YAW_WEIGHT / T)

    # Control regularization: diagonal approximation of sum_t (W @ params)[t]**2
    # = sum_k (W[:,k]**2).sum() * params[k,:]**2 (cross terms are small for linear-interp W)
    w_sq = (W * W).sum(axis=0)  # shape (K,)
    for k in range(W.shape[1]):
        coef = pd.ADDouble(float(w_sq[k]) * REGULARIZATION_WEIGHT / T)
        for j in range(NU):
            p = p_ind[k * NU + j]
            loss_ad = loss_ad + p * p * coef

    fn = pd.ADFun(p_ind, [loss_ad])
    fn.optimize()

    jac = fn.Jacobian(list(params_flat.astype(float)))
    loss_val = float(loss_ad.value())
    return loss_val, np.array(jac).reshape(params_np.shape)


def rollout_float(params_np, W, T):
    params_ad = [pd.ADDouble(float(v)) for v in params_np.flatten()]
    xy_traj, _ = rollout_ad(params_ad, W, T)
    return np.array([[x.value(), y.value()] for x, y in xy_traj])


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
    parser.add_argument("--ground-truth", type=str, default=str(DATA_DIR / "right_turn_b.json"))
    parser.add_argument("--save", metavar="PATH", default=str(RESULTS_DIR / "tinydiffsim.json"))
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--init", choices=["perturbed", "zeros", "forward"], default="perturbed")
    parser.add_argument(
        "--horizon-s",
        type=float,
        default=None,
        help="Truncate trajectory to first N seconds (default: full duration)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=3,
        help="Number of independent runs (different perturbed init guesses)",
    )
    parser.add_argument(
        "--seed-base", type=int, default=42, help="First seed; trial k uses seed_base + k"
    )
    args = parser.parse_args()

    target_ctrl, duration, traj_xy_np = load_ground_truth(args.ground_truth)
    if args.horizon_s is not None and args.horizon_s < duration:
        keep = max(2, int(args.horizon_s / duration * len(traj_xy_np)))
        traj_xy_np = traj_xy_np[:keep]
        duration = args.horizon_s
    T = int(duration / DT)

    # Resample target to sim steps
    real_t = np.linspace(0, 1, len(traj_xy_np))
    sim_t = np.linspace(0, 1, T)
    target_xy = np.zeros((T, 2), dtype=np.float64)
    target_xy[:, 0] = np.interp(sim_t, real_t, traj_xy_np[:, 0])
    target_xy[:, 1] = np.interp(sim_t, real_t, traj_xy_np[:, 1])

    print(f"Target: real robot trajectory ({len(traj_xy_np)} pts -> {T} sim steps)")
    print(
        f"Real robot ctrl: L={target_ctrl[0]:.3f} R={target_ctrl[1]:.3f} Rear={target_ctrl[2]:.3f}"
    )
    print(
        f"T={T}, dt={DT}, K={args.K}, kv={KV}, friction={FRICTION}, lr={args.lr}, "
        f"num_trials={args.num_trials}"
    )

    W = make_interp_matrix(T, args.K)

    def run_trial(init_ctrl):
        params = np.tile(init_ctrl, (args.K, 1)).astype(float)
        adam = SplineAdam(
            K=args.K, num_dofs=NU, lr=args.lr, lr_min_ratio=0.1, total_steps=args.iterations
        )
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
            loss, grad = grad_and_loss_ad(params, W, target_xy, T)
            t_iter = (time.perf_counter() - t0) * 1000

            sim_xy = rollout_float(params, W, T)
            rmse_m = float(
                np.sqrt(
                    np.mean(
                        (sim_xy[:, 0] - target_xy[:, 0]) ** 2
                        + (sim_xy[:, 1] - target_xy[:, 1]) ** 2
                    )
                )
            )

            is_best = loss < best_loss
            if is_best:
                best_loss = loss
                trial["best_iters"].append(i)

            marker = " *" if is_best else ""
            print(
                f"  Iter {i:3d}: loss={loss:.4f} | RMSE={rmse_m:.3f}m | "
                f"best={best_loss:.4f} | t={t_iter:.0f}ms{marker}"
            )

            trial["iterations"].append(i)
            trial["loss"].append(float(loss))
            trial["rmse_m"].append(rmse_m)
            trial["time_ms"].append(t_iter)

            params = adam.step(params, grad)
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
        print(
            f"Init ctrl ({args.init}): L={init_ctrl[0]:.3f} R={init_ctrl[1]:.3f} "
            f"Rear={init_ctrl[2]:.3f}"
        )
        trial = run_trial(init_ctrl)
        trial["seed"] = seed
        trials.append(trial)

    aggregate = {
        "simulator": "TinyDiffSim",
        "gradient_method": "CppAD",
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
