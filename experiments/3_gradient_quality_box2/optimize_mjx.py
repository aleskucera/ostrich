"""Helhest_junior box random-IC final-pose optimisation — MJX.

MJX counterpart of optimize_axion.py. Same task: random IC + random target +
weighted loss (final pos + final yaw + terminal velocity + smoothness + reg).
Cylinder wheels are swapped to capsules + wheel↔wheel collisions disabled
(see experiments/3_gradient_quality_box/optimize_mjx.py — MJX collision
matrix doesn't include cylinder↔box).

Usage:
    python experiments/3_gradient_quality_box2/optimize_mjx.py
"""
import argparse
import json
import os
import pathlib
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "1_sim_to_real_box"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "3_gradient_quality_box"))

os.environ.setdefault("DISPLAY", ":1")
os.environ.pop("WAYLAND_DISPLAY", None)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import mujoco
    import mujoco.mjx as mjx
    _HAVE_JAX = True
except ImportError as e:
    _HAVE_JAX = False
    _JAX_IMPORT_ERROR = e

from sweep_mujoco import BASE_PARAMS, JUNIOR_BOX_XML  # noqa: E402
from optimize_mjx import MJX_PARAMS, _patch_wheels_for_mjx  # noqa: E402

RESULTS_DIR = pathlib.Path(__file__).parent / "results"


def make_interp_matrix(T, K):
    W = np.zeros((T, K), dtype=np.float32)
    for t in range(T):
        x = t * (K - 1) / max(T - 1, 1)
        lo = int(x); hi = min(lo + 1, K - 1); a = x - lo
        W[t, lo] += 1.0 - a
        W[t, hi] += a
    return W


class SplineAdam:
    """Same hand-rolled Adam as optimize_axion.py (numpy)."""
    def __init__(self, K, num_dofs, lr=0.05, lr_min_ratio=0.2, total_steps=100,
                 betas=(0.9, 0.999), eps=1e-8):
        self.lr_init = lr; self.lr_min = lr * lr_min_ratio
        self.total_steps = total_steps; self.b1, self.b2 = betas; self.eps = eps
        self.m = np.zeros((K, num_dofs), dtype=np.float64)
        self.v = np.zeros((K, num_dofs), dtype=np.float64)
        self.t = 0

    def _cosine_lr(self):
        p = min(self.t / max(1, self.total_steps), 1.0)
        return self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (1.0 + np.cos(np.pi * p))

    def step(self, params, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        mh = self.m / (1 - self.b1**self.t) if self.b1 > 0 else self.m
        vh = self.v / (1 - self.b2**self.t)
        return params - self._cosine_lr() * mh / (np.sqrt(vh) + self.eps)


def build_mjx_model(box):
    fmt = {**BASE_PARAMS, **MJX_PARAMS, "dt": 5e-3,
           "ground_friction": MJX_PARAMS["mu"], "box_friction": MJX_PARAMS["mu"],
           "front_friction": MJX_PARAMS["mu"], "rear_friction": MJX_PARAMS["mu"],
           "ground_torsional": MJX_PARAMS["tor"], "front_torsional": MJX_PARAMS["tor"],
           "rear_torsional": MJX_PARAMS["tor"],
           "box_x": box["center"][0], "box_y": box["center"][1], "box_z": box["center"][2],
           "box_hx": box["half_extents"][0], "box_hy": box["half_extents"][1],
           "box_hz": box["half_extents"][2]}
    xml = JUNIOR_BOX_XML.format(**fmt)
    xml = _patch_wheels_for_mjx(xml)
    mj_model = mujoco.MjModel.from_xml_string(xml)
    return mjx.put_model(mj_model), mj_model


def make_init_dx(mx, mj_model, ic_xy, ic_yaw, dt):
    """Build initial mjx.Data with chassis at the randomised IC pose."""
    d = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, d)
    dx0 = mjx.put_data(mj_model, d)
    # qpos = [x, y, z, qw, qx, qy, qz, wheel_L, wheel_R, wheel_rear]
    # Override the chassis xy + yaw (z stays at the default spawn height).
    qw = float(np.cos(ic_yaw / 2.0))
    qz = float(np.sin(ic_yaw / 2.0))
    new_qpos = dx0.qpos.at[0].set(float(ic_xy[0]))
    new_qpos = new_qpos.at[1].set(float(ic_xy[1]))
    new_qpos = new_qpos.at[3].set(qw)
    new_qpos = new_qpos.at[4].set(0.0)
    new_qpos = new_qpos.at[5].set(0.0)
    new_qpos = new_qpos.at[6].set(qz)
    return dx0.replace(qpos=new_qpos), dt


def rollout(mx, dx0, ctrl_traj):
    """Run T mjx.step calls; return per-step (xy, yaw_quat) traces."""
    def step_fn(d, ctrl_t):
        d = d.replace(ctrl=ctrl_t)
        d = mjx.step(mx, d)
        return d, jnp.concatenate([d.qpos[:2], d.qpos[3:7]])   # [2 xy + 4 quat = 6]
    _, traj = jax.lax.scan(step_fn, dx0, ctrl_traj)
    return traj   # [T, 6]


def _quat_to_yaw(q):
    """JAX-friendly yaw from (qw, qx, qy, qz) — MuJoCo's wxyz convention."""
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return jnp.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def make_loss_fn(mx, dx0, W, ic_xy, target_xy, target_yaw, weights, dt, tail_frac):
    T = int(W.shape[0])
    tail_len = max(2, int(round(T * tail_frac)))
    tail_start = T - 1 - tail_len
    inv_dt = 1.0 / dt
    w_track = weights.get("track", 0.0)
    w_pos = weights["pos"]; w_yaw = weights["yaw"]; w_vel = weights["vel"]
    w_smooth = weights["smooth"]; w_reg = weights["reg"]

    # Linear waypoint reference: same shaping signal as the Axion script's
    # chassis_track_loss_kernel. Per-step xy distance to a straight line
    # from ic_xy to target_xy puts gradient signal at EVERY timestep instead
    # of requiring it to compound back from the terminal step.
    t_norm = jnp.linspace(0.0, 1.0, T)
    ref_xy = ic_xy[None, :] + t_norm[:, None] * (target_xy - ic_xy)[None, :]  # [T, 2]

    def loss_fn(params):
        ctrl_traj = W @ params                          # [T, 3]
        traj = rollout(mx, dx0, ctrl_traj)              # [T, 6]: xy + quat
        xy_traj = traj[:, :2]
        quat_traj = traj[:, 2:]

        # 0. Per-step waypoint tracking — main shaping term (see Axion analog)
        L_track = (w_track / T) * jnp.sum((xy_traj - ref_xy) ** 2)

        # 1. Final position
        final_xy = xy_traj[-1]
        L_pos = w_pos * jnp.sum((final_xy - target_xy) ** 2)

        # 2. Final yaw — wrap-safe via 1 - cos(Δ)
        final_yaw = _quat_to_yaw(quat_traj[-1])
        L_yaw = w_yaw * (1.0 - jnp.cos(final_yaw - target_yaw))

        # 3. Terminal velocity over tail (finite-diff on xy)
        tail_xy = xy_traj[tail_start:T]                    # [tail_len + 1, 2]
        v_xy = (tail_xy[1:] - tail_xy[:-1]) * inv_dt       # [tail_len, 2]
        L_vel = (w_vel / tail_len) * jnp.sum(v_xy ** 2)

        # 4. Smoothness Σ ‖u[t+1] - u[t]‖²
        du = ctrl_traj[1:] - ctrl_traj[:-1]                # [T-1, 3]
        L_smooth = (w_smooth / (T - 1)) * jnp.sum(du ** 2)

        # 5. Magnitude reg Σ ‖u[t]‖²
        L_reg = (w_reg / T) * jnp.sum(ctrl_traj ** 2)

        return L_track + L_pos + L_yaw + L_vel + L_smooth + L_reg

    return loss_fn


def _nvml_poller():
    try:
        import pynvml
    except ImportError:
        return None
    pynvml.nvmlInit()
    gpu_idx = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    h = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
    baseline = pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024**2
    peak = [baseline]; stop = threading.Event()

    def _poll():
        while not stop.is_set():
            used = pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024**2
            if used > peak[0]:
                peak[0] = used
            time.sleep(0.01)

    th = threading.Thread(target=_poll, daemon=True); th.start()

    def finalize():
        stop.set(); th.join(timeout=1.0)
        return float(peak[0]), float(peak[0] - baseline)

    return finalize


def sample_ic_target(rng, ic_perturb, target_perturb):
    ic = {
        "xy": [
            float(rng.uniform(-ic_perturb["xy"], ic_perturb["xy"])),
            float(rng.uniform(-ic_perturb["xy"], ic_perturb["xy"])),
        ],
        "yaw": float(rng.uniform(-ic_perturb["yaw_rad"], ic_perturb["yaw_rad"])),
    }
    target = {
        "xy": [
            float(target_perturb["xy_center"][0]
                  + rng.uniform(-target_perturb["xy_jitter"], target_perturb["xy_jitter"])),
            float(target_perturb["xy_center"][1]
                  + rng.uniform(-target_perturb["xy_jitter"], target_perturb["xy_jitter"])),
        ],
        "yaw": float(rng.uniform(-target_perturb["yaw_rad"], target_perturb["yaw_rad"])),
    }
    return ic, target


def run_trial(seed, K, lr, iterations, horizon, dt, ic, target, weights,
              clip_grad_norm, beta1, beta2, mx, mj_model, gt_box, tail_frac=0.10):
    T = int(horizon / dt)
    dx0, _ = make_init_dx(mx, mj_model, ic["xy"], ic["yaw"], dt)
    W = jnp.asarray(make_interp_matrix(T, K))
    ic_xy = jnp.asarray(ic["xy"], dtype=jnp.float32)
    target_xy = jnp.asarray(target["xy"], dtype=jnp.float32)
    target_yaw = float(target["yaw"])
    loss_fn = make_loss_fn(mx, dx0, W, ic_xy, target_xy, target_yaw,
                            weights, dt, tail_frac)
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    rng = np.random.default_rng(seed)
    params_np = (2.0 + 0.5 * rng.standard_normal((K, 3))).astype(np.float32)
    opt = SplineAdam(K=K, num_dofs=3, lr=lr, lr_min_ratio=0.2,
                     total_steps=iterations, betas=(beta1, beta2))

    losses, grad_norms, n_clipped = [], [], 0
    best_loss = float("inf")
    best_iter = 0
    best_params = params_np.copy()
    t0_total = time.perf_counter()
    for it in range(iterations):
        loss_val, grad = value_and_grad(jnp.asarray(params_np))
        loss_val = float(loss_val); grad_np = np.asarray(grad).astype(np.float64)
        gnorm = float(np.linalg.norm(grad_np))
        if clip_grad_norm is not None and gnorm > clip_grad_norm:
            grad_np = grad_np * (clip_grad_norm / gnorm)
            n_clipped += 1
        losses.append(loss_val); grad_norms.append(gnorm)
        # Best-iter snapshot (same justification as Axion run_trial)
        if loss_val < best_loss:
            best_loss = loss_val
            best_iter = it
            best_params = params_np.copy()
        params_np = opt.step(params_np, grad_np).astype(np.float32)
        if it % 10 == 0 or it == iterations - 1:
            star = " *" if (clip_grad_norm is not None and gnorm > clip_grad_norm) else ""
            print(f"    iter {it:3d}: loss={loss_val:.4f} |g|={gnorm:.2f}{star}", flush=True)
    wall = time.perf_counter() - t0_total

    # Final metrics computed on BEST-iter params, not iter N-1's drifted state.
    ctrl_traj_final = W @ jnp.asarray(best_params)
    final_traj = jax.jit(rollout)(mx, dx0, ctrl_traj_final)
    final_traj.block_until_ready()
    xy_T = np.asarray(final_traj[-1, :2])
    quat_T = np.asarray(final_traj[-1, 2:])
    xy_Tm1 = np.asarray(final_traj[-2, :2])
    yaw_T = float(np.arctan2(2.0 * (quat_T[0] * quat_T[3] + quat_T[1] * quat_T[2]),
                              1.0 - 2.0 * (quat_T[2] ** 2 + quat_T[3] ** 2)))
    v_xy = (xy_T - xy_Tm1) / dt
    terminal_speed = float(np.linalg.norm(v_xy))
    pos_error = float(np.linalg.norm(xy_T - np.asarray(target["xy"])))
    yaw_error = float(abs(np.arctan2(np.sin(yaw_T - target_yaw),
                                       np.cos(yaw_T - target_yaw))))
    u = np.asarray(ctrl_traj_final)
    du = np.diff(u, axis=0)
    control_jerk = float(np.linalg.norm(du, axis=1).sum())
    metrics = {
        "xy_final": [float(xy_T[0]), float(xy_T[1])],
        "yaw_final": yaw_T,
        "pos_error_m": pos_error,
        "yaw_error_rad": yaw_error,
        "terminal_speed_mps": terminal_speed,
        "control_jerk": control_jerk,
        "success": bool(pos_error < 0.2 and terminal_speed < 0.3),
    }
    print(f"    BEST-iter restored: iter {best_iter}, loss {best_loss:.4f}  "
          f"-> pos_err={pos_error:.3f}m  vel={terminal_speed:.3f}m/s  "
          f"{'OK' if metrics['success'] else 'FAIL'}", flush=True)
    return {"seed": int(seed), "ic": ic, "target": target,
            "losses": losses, "grad_norms": grad_norms,
            "n_clipped": int(n_clipped), "wall_s": wall,
            "best_loss": float(best_loss),
            "best_iter": int(best_iter),
            "final_loss": float(losses[-1]),
            "final_metrics": metrics}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--num-trials", type=int, default=25)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--horizon-s", type=float, default=6.0)
    ap.add_argument("--dt", type=float, default=5e-3)
    ap.add_argument("--ic-xy", type=float, default=0.1,
                    help="±range for IC xy perturbation (m). Default 0.1m — "
                    "matches Axion side; wider IC made many trials infeasible.")
    ap.add_argument("--ic-yaw-deg", type=float, default=5.0,
                    help="±range for IC yaw perturbation (deg). Default 5° — "
                    "matches Axion side.")
    ap.add_argument("--target-x", type=float, default=3.0)
    ap.add_argument("--target-y", type=float, default=0.0)
    ap.add_argument("--target-xy-jitter", type=float, default=0.3)
    ap.add_argument("--target-yaw-deg", type=float, default=15.0)
    ap.add_argument("--w-track", type=float, default=1.0,
                    help="Per-step chassis-xy tracking weight against a linear "
                    "IC→target reference (shaping signal; matches the Axion side).")
    ap.add_argument("--w-pos", type=float, default=1.0)
    ap.add_argument("--w-yaw", type=float, default=0.5)
    ap.add_argument("--w-vel", type=float, default=0.3)
    ap.add_argument("--w-smooth", type=float, default=1e-3)
    ap.add_argument("--w-reg", type=float, default=1e-5)
    ap.add_argument("--clip-grad-norm", type=float, default=1.0,
                    help="global-norm gradient clip; same as exp 3 box default")
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    # gt JSON only used for the box geometry (center + half_extents).
    ap.add_argument("--gt", default=str(pathlib.Path(__file__).resolve().parents[1]
                                         / "1_sim_to_real_box" / "data"
                                         / "run_2026_05_20-18_10_33.json"),
                    help="any GT JSON — only its box geometry is read")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    if not _HAVE_JAX:
        raise SystemExit(f"JAX/mujoco-mjx not installed: {_JAX_IMPORT_ERROR}")

    with open(args.gt) as f:
        gt = json.load(f)
    ic_perturb = {"xy": args.ic_xy, "yaw_rad": np.deg2rad(args.ic_yaw_deg)}
    target_perturb = {"xy_center": (args.target_x, args.target_y),
                       "xy_jitter": args.target_xy_jitter,
                       "yaw_rad": np.deg2rad(args.target_yaw_deg)}
    weights = {"track": args.w_track, "pos": args.w_pos, "yaw": args.w_yaw, "vel": args.w_vel,
               "smooth": args.w_smooth, "reg": args.w_reg}

    print(f"K={args.K} iters={args.iterations} lr={args.lr} "
          f"trials={args.num_trials} dt={args.dt} horizon={args.horizon_s}s")
    print(f"JAX devices: {jax.devices()}")
    print(f"IC perturb: xy±{args.ic_xy}m, yaw±{args.ic_yaw_deg}°")
    print(f"Target: ({args.target_x}, {args.target_y}) ± "
          f"{args.target_xy_jitter}m, yaw±{args.target_yaw_deg}°")
    print(f"Weights: {weights}  (clip={args.clip_grad_norm})")

    # Build MJX model ONCE — reused across trials. Only dx0 (initial state)
    # changes per trial.
    mx, mj_model = build_mjx_model(gt["box"])
    finalize_nvml = _nvml_poller()

    trials = []
    task_rng = np.random.default_rng(args.seed_base)
    for k in range(args.num_trials):
        ic, target = sample_ic_target(task_rng, ic_perturb, target_perturb)
        spline_seed = args.seed_base + k + 1000
        print(f"\n--- trial {k + 1}/{args.num_trials}  "
              f"IC=({ic['xy'][0]:+.2f},{ic['xy'][1]:+.2f},{np.rad2deg(ic['yaw']):+.1f}°)  "
              f"target=({target['xy'][0]:.2f},{target['xy'][1]:+.2f},{np.rad2deg(target['yaw']):+.1f}°) ---")
        t = run_trial(spline_seed, args.K, args.lr, args.iterations,
                      args.horizon_s, args.dt, ic, target, weights,
                      args.clip_grad_norm, args.beta1, args.beta2,
                      mx, mj_model, gt["box"])
        m = t["final_metrics"]
        print(f"  -> pos_err={m['pos_error_m']:.3f}m  vel={m['terminal_speed_mps']:.2f}m/s  "
              f"jerk={m['control_jerk']:.1f}  {'OK' if m['success'] else 'FAIL'}", flush=True)
        trials.append(t)

    nvml_abs, nvml_delta = (finalize_nvml() if finalize_nvml is not None else (None, None))

    success_rate = sum(t["final_metrics"]["success"] for t in trials) / len(trials)
    median_pos = float(np.median([t["final_metrics"]["pos_error_m"] for t in trials]))
    median_vel = float(np.median([t["final_metrics"]["terminal_speed_mps"] for t in trials]))

    out = {
        "simulator": "MJX",
        "task": "random-IC final-pose",
        "K": args.K, "lr": args.lr, "iterations": args.iterations,
        "horizon_s": args.horizon_s, "dt": args.dt,
        "clip_grad_norm": args.clip_grad_norm,
        "beta1": args.beta1, "beta2": args.beta2,
        "num_trials": args.num_trials, "seed_base": args.seed_base,
        "ic_perturbation": ic_perturb,
        "target_perturbation": {**target_perturb,
                                 "xy_center": list(target_perturb["xy_center"])},
        "loss_weights": weights,
        "params": MJX_PARAMS,
        "peak_gpu_mb_nvml_absolute": nvml_abs,
        "peak_gpu_mb_nvml": nvml_delta,
        "aggregate": {
            "success_rate": success_rate,
            "median_pos_error_m": median_pos,
            "median_terminal_speed_mps": median_vel,
        },
        "trials": trials,
    }
    save_path = args.save or str(RESULTS_DIR / "mjx.json")
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(out, f)
    print(f"\n=== Aggregate ===")
    print(f"  Success rate: {success_rate:.0%}")
    print(f"  Median pos error: {median_pos:.3f} m")
    print(f"  Median terminal speed: {median_vel:.3f} m/s")
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
