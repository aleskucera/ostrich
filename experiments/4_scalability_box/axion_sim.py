"""Helhest_junior box scalability benchmark — Axion, variable number of worlds.

Mirrors experiments/4_scalability/axion_sim.py but on the box scene + helhest
junior + GT-style loss from experiments/3_gradient_quality_box. Each world has
the SAME random spline init and the SAME real-trajectory target — the sweep
measures pure batch throughput, not optimization diversity.

Output JSON schema matches the flat-scene experiment so plot_results.py can
reuse the same shape.

Usage:
    python experiments/4_scalability_box/axion_sim.py --num-worlds 64 \
        --save experiments/4_scalability_box/results/axion_64.json
"""
import argparse
import json
import os
import pathlib
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import newton
import numpy as np
import warp as wp
from axion import (AxionDifferentiableSimulator, AxionEngineConfig, ComplianceConfig,
                   ContactsConfig, LinearSolverConfig, LinesearchConfig, LoggingConfig,
                   NewtonRaphsonConfig, RenderingConfig, SimulationConfig)

from examples.helhest_junior.common import create_helhest_junior_model
from examples.helhest_junior.replay_real import BOX_CENTER, BOX_HALF_EXTENTS

os.environ["PYOPENGL_PLATFORM"] = "glx"

WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3
K = 10                 # matches experiments/3_gradient_quality_box
DT = 0.10              # Axion's exp-3-box default; ~12x realtime
DURATION = 6.0         # matches experiments/3_gradient_quality_box horizon
ITERATIONS = 5         # matches experiments/4_scalability (throughput, not convergence)


def make_interp_matrix(T: int, K: int):
    W = np.zeros((T, K), dtype=np.float32)
    for t in range(T):
        x = t * (K - 1) / max(T - 1, 1)
        lo = int(x); hi = min(lo + 1, K - 1); a = x - lo
        W[t, lo] += 1.0 - a
        W[t, hi] += a
    return W, W.sum(axis=0)


class SplineAdam:
    def __init__(self, K, num_dofs, lr=0.1, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr; self.b1, self.b2 = betas; self.eps = eps
        self.m = np.zeros((K, num_dofs), dtype=np.float64)
        self.v = np.zeros((K, num_dofs), dtype=np.float64)
        self.t = 0

    def step(self, params, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        mh = self.m / (1 - self.b1**self.t)
        vh = self.v / (1 - self.b2**self.t)
        return params - self.lr * mh / (np.sqrt(vh) + self.eps)


@wp.kernel
def chassis_xy_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),    # [T, W, B]
    target_pose: wp.array(dtype=wp.transform, ndim=3),  # [T, W, B]
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """Mean squared XY distance, summed over (t, w) and weighted by 1/(T*W)."""
    t, w = wp.tid()
    p = wp.transform_get_translation(body_pose[t, w, 0])
    q = wp.transform_get_translation(target_pose[t, w, 0])
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    wp.atomic_add(loss, 0, weight * (dx * dx + dy * dy))


@wp.kernel
def fill_target_chassis_kernel(
    target_xyz: wp.array(dtype=wp.vec3),                # [T]
    target_pose: wp.array(dtype=wp.transform, ndim=3),  # [T, W, B]
):
    """Broadcast target across all worlds."""
    t, w = wp.tid()
    target_pose[t, w, 0] = wp.transform(target_xyz[t], wp.quat_identity())


class HelhestJuniorBoxScalabilityOptimizer(AxionDifferentiableSimulator):
    """Axion-side replicated-worlds optimizer. One spline broadcast to all worlds."""

    def __init__(self, sim_config, render_config, engine_config, logging_config,
                 target_xyz_rel, target_t,
                 mu_front=0.8, mu_rear=1.2, mu_rolling=0.7):
        self.K = K
        self.num_worlds = sim_config.num_worlds
        self.mu_front = mu_front
        self.mu_rear = mu_rear
        self.mu_rolling = mu_rolling
        super().__init__(sim_config, render_config, engine_config, logging_config)

        T = self.clock.total_sim_steps
        self.W, self.W_col_sums = make_interp_matrix(T, K)
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self._setup_target(target_xyz_rel, target_t)

    def build_model(self):
        self.builder.rigid_gap = 0.2
        ground_cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, ke=150.0, kd=150.0, kf=500.0)
        self.builder.add_ground_plane(cfg=ground_cfg)
        self.builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*BOX_CENTER), wp.quat_identity()),
            hx=BOX_HALF_EXTENTS[0], hy=BOX_HALF_EXTENTS[1], hz=BOX_HALF_EXTENTS[2],
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.8, ke=150.0, kd=150.0, kf=500.0))
        create_helhest_junior_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            control_mode="velocity",
            k_p=250.0, k_d=0.0,
            friction_left_right=self.mu_front, friction_rear=self.mu_rear,
            mu_rolling=self.mu_rolling)
        return self.builder.finalize_replicated(num_worlds=self.num_worlds,
                                                  requires_grad=True)

    def _setup_target(self, target_xyz_rel, target_t):
        T = self.clock.total_sim_steps
        dt = self.clock.dt
        t_sim = np.arange(T) * dt
        target = np.zeros((T, 3), dtype=np.float32)
        for c in range(3):
            target[:, c] = np.interp(t_sim, target_t, target_xyz_rel[:, c])
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])
        sim_origin = self.states[0].body_q.numpy()[0, :3]
        target_world = target + sim_origin.astype(np.float32)
        target_vec3 = wp.array(target_world, dtype=wp.vec3, device=self.model.device)
        wp.launch(fill_target_chassis_kernel, dim=(T, self.num_worlds),
                  inputs=[target_vec3], outputs=[self.trajectory.target_body_pose],
                  device=self.model.device)

    def _expand(self, params):
        return self.W @ params

    def _contract(self, grad_v):
        safe = np.where(self.W_col_sums > 0, self.W_col_sums, 1.0)
        return (self.W.T @ grad_v) / safe[:, None]

    def _apply_params(self, params):
        """Write same spline-expanded velocities to every world."""
        T = self.clock.total_sim_steps
        num_dofs = self.trajectory.joint_target_vel.shape[-1]
        expanded = self._expand(params)  # [T, 3]
        vel_np = np.zeros((T, self.num_worlds, num_dofs), dtype=np.float32)
        vel_np[:, :, WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = expanded[:, None, :]
        wp.copy(self.trajectory.joint_target_vel, wp.array(vel_np, dtype=wp.float32))
        for i in range(T):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    def compute_loss(self):
        T = self.trajectory.body_pose.shape[0]
        weight = 1.0 / (T * self.num_worlds)
        wp.launch(chassis_xy_loss_kernel, dim=(T, self.num_worlds),
                  inputs=[self.trajectory.body_pose, self.trajectory.target_body_pose,
                          weight],
                  outputs=[self.loss], device=self.solver.model.device)

    def update(self):
        pass  # spline_adam owned by trainer

    def opt_step(self):
        self.diff_step()
        wp.synchronize()
        loss_val = float(self.loss.numpy()[0])
        # Loss is normalized 1/(T*W), so per-world gradient is already scaled
        # down by W. Averaging across worlds yields a single per-world gradient
        # (matching the single-world regime); spline param update is therefore
        # batch-size independent. Same convention as experiments/4_scalability.
        grad_v = self.trajectory.joint_target_vel.grad.numpy()[
            :, :, WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS].mean(axis=1)
        grad_params = self._contract(grad_v)
        self.trajectory.joint_target_vel.grad.zero_()
        self.spline_params = self.spline_adam.step(self.spline_params, grad_params)
        self._apply_params(self.spline_params)
        self.tape.zero()
        self.loss.zero_()
        return loss_val


def _nvml_poller():
    """Start a daemon thread polling NVML used-mem at 100 Hz. Returns
    (stop_event, peak_holder, baseline_mb, finalize_fn). finalize_fn returns
    (peak_absolute_mb, peak_delta_mb)."""
    try:
        import pynvml
    except ImportError:
        return None
    pynvml.nvmlInit()
    gpu_idx = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    h = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
    baseline = pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024**2
    peak = [baseline]
    stop = threading.Event()

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-worlds", type=int, default=1)
    ap.add_argument("--gt", default=str(pathlib.Path(__file__).resolve().parents[1]
                                         / "1_sim_to_real_box" / "data"
                                         / "run_2026_05_20-18_10_33.json"))
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    with open(args.gt) as f:
        gt = json.load(f)
    real_t = np.asarray(gt["real"]["t"])
    real_xyz = np.column_stack([gt["real"]["x"], gt["real"]["y"], gt["real"]["z"]])
    m = (real_t >= 0) & (real_t <= DURATION)
    real_t = real_t[m]; real_xyz = real_xyz[m]

    sim_cfg = SimulationConfig(duration_seconds=DURATION, target_timestep_seconds=DT,
                                num_worlds=args.num_worlds, use_cuda_graph=True)
    rc = RenderingConfig(vis_type="null", target_fps=max(1, int(1 / DT)), start_paused=False)
    ec = AxionEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=16, backtrack_min_iter=12, atol=1e-3),
        linear=LinearSolverConfig(max_iters=16, tol=1e-3, atol=1e-3, regularization=1e-6),
        compliance=ComplianceConfig(joint=6e-8, contact=1e-7, friction=1e-6),
        linesearch=LinesearchConfig(enabled=False),
        contacts=ContactsConfig(max_per_world=256))

    finalize_nvml = _nvml_poller()

    sim = HelhestJuniorBoxScalabilityOptimizer(sim_cfg, rc, ec, LoggingConfig(),
                                                  real_xyz, real_t)
    rng = np.random.default_rng(42)
    init = 2.0 + 0.5 * rng.standard_normal((K, NUM_WHEEL_DOFS))
    sim.spline_params = init.astype(np.float64)
    sim.spline_adam = SplineAdam(K=K, num_dofs=NUM_WHEEL_DOFS, lr=0.1)
    sim._apply_params(sim.spline_params)

    print(f"Optimising: T={sim.clock.total_sim_steps}, dt={DT}, K={K}, "
          f"num_worlds={args.num_worlds}")
    peak_mem_mb = 0.0
    time_ms_list = []
    for i in range(ITERATIONS):
        t0 = time.perf_counter()
        loss = sim.opt_step()
        t_iter = (time.perf_counter() - t0) * 1000
        try:
            used_bytes = wp.get_mempool_used_bytes()
        except AttributeError:
            used_bytes = wp.get_mempool_used_mem_high()
        used_mb = used_bytes / 1024**2
        peak_mem_mb = max(peak_mem_mb, used_mb)
        print(f"  iter {i:3d}: loss={loss:.4f} | t={t_iter:.0f}ms | mem={used_mb:.0f}MB",
              flush=True)
        time_ms_list.append(t_iter)

    nvml_abs, nvml_delta = (finalize_nvml() if finalize_nvml is not None else (None, None))

    results = {
        "simulator": "Axion",
        "num_worlds": args.num_worlds,
        "median_time_ms": (float(np.median(time_ms_list[3:]))
                           if len(time_ms_list) > 3 else float(np.median(time_ms_list))),
        "peak_gpu_mb": peak_mem_mb,
        "peak_gpu_mb_nvml_absolute": nvml_abs,
        "peak_gpu_mb_nvml": nvml_delta,
        "time_ms": time_ms_list,
        "K": K, "dt": DT, "duration_s": DURATION, "iterations": ITERATIONS,
    }
    sim.close(); del sim
    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")
    if nvml_abs is not None:
        print(f"NVML peak: {nvml_abs:.0f} MB (delta {nvml_delta:.0f} MB)")


if __name__ == "__main__":
    main()
