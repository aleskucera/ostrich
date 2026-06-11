"""Helhest_junior box scalability benchmark — Semi-Implicit, variable num_worlds.

SI counterpart of ostrich_sim.py / mjx_sim.py for the box scene. Uses Newton's
SemiImplicit solver with replicated worlds via builder.finalize_replicated.
Same physics + spline + GT setup as experiments/3_gradient_quality_box.

Caveat: SI's CUDA-graph capture takes ~20 min per process (cold). The sweep
script therefore launches one process per num_worlds value — most of the
per-num_worlds wall-time is graph compilation, not the 5-iteration measurement.

Usage:
    python experiments/4_scalability_box/semi_implicit_sim.py --num-worlds 4 \
        --save experiments/4_scalability_box/results/semi_implicit_4.json
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
from ostrich import (LoggingConfig, RenderingConfig, SemiImplicitEngineConfig,
                   SimulationConfig)
from ostrich.simulation.differentiable_simulator import NewtonDifferentiableSimulator

from examples.helhest_junior.common import create_helhest_junior_model
from examples.helhest_junior.replay_real import BOX_CENTER, BOX_HALF_EXTENTS

os.environ["PYOPENGL_PLATFORM"] = "glx"

# SI tuned best from experiments/1_sim_to_real_box / 3_gradient_quality_box.
SI_MU = 0.05
SI_KE = 8e4
SI_KD = 2e3
SI_KF = 1500.0
SI_K_D_ACT = 200.0
SI_JOINT_ATTACH_KE = 1e6
SI_JOINT_ATTACH_KD = 1e2

WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3
K = 10                # matches experiments/3_gradient_quality_box
DT = 5e-4             # SI's exp-3-box default (stability edge)
DURATION = 6.0        # matches experiments/3_gradient_quality_box horizon
ITERATIONS = 5        # matches experiments/4_scalability


def make_interp_matrix(T, K):
    W = np.zeros((T, K), dtype=np.float32)
    for t in range(T):
        x = t * (K - 1) / max(T - 1, 1)
        lo = int(x); hi = min(lo + 1, K - 1); a = x - lo
        W[t, lo] += 1.0 - a
        W[t, hi] += a
    return W, W.sum(axis=0)


class SplineAdam:
    def __init__(self, K, num_dofs, lr=0.05, betas=(0.9, 0.999), eps=1e-8):
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
def chassis_xy_loss_multiworld_kernel(
    body_q: wp.array(dtype=wp.transform),    # [N*B] flat (replicated model)
    bodies_per_world: int,                   # B
    target_xy: wp.array(dtype=wp.vec2),      # [T+1]
    step_idx: int,
    weight: float,                           # 1 / (T * N)
    loss: wp.array(dtype=wp.float32),
):
    """Per-world chassis-XY distance at one timestep. Body 0 of each world = chassis."""
    w = wp.tid()  # 0..N-1
    p = wp.transform_get_translation(body_q[w * bodies_per_world])
    q = target_xy[step_idx]
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    wp.atomic_add(loss, 0, weight * (dx * dx + dy * dy))


class HelhestJuniorBoxSIScalability(NewtonDifferentiableSimulator):
    def __init__(self, sim_config, render_config, engine_config, logging_config,
                 target_xyz_rel, target_t):
        self.K = K
        self.num_worlds = sim_config.num_worlds
        super().__init__(sim_config, render_config, engine_config, logging_config)

        T = self.clock.total_sim_steps
        self.W, self.W_col_sums = make_interp_matrix(T, K)
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.bodies_per_world = self.model.body_count // self.num_worlds
        self.dofs_per_world = self.controls[0].joint_target_vel.shape[-1] // self.num_worlds
        self._setup_target(target_xyz_rel, target_t)

    def build_model(self):
        self.builder.rigid_gap = 0.2
        ground_cfg = newton.ModelBuilder.ShapeConfig(mu=SI_MU, ke=SI_KE, kd=SI_KD, kf=SI_KF)
        self.builder.add_ground_plane(cfg=ground_cfg)
        self.builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*BOX_CENTER), wp.quat_identity()),
            hx=BOX_HALF_EXTENTS[0], hy=BOX_HALF_EXTENTS[1], hz=BOX_HALF_EXTENTS[2],
            cfg=newton.ModelBuilder.ShapeConfig(mu=SI_MU, ke=SI_KE, kd=SI_KD, kf=SI_KF))
        create_helhest_junior_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            control_mode="velocity",
            k_p=0.0, k_d=SI_K_D_ACT,
            friction_left_right=SI_MU, friction_rear=SI_MU,
            mu_rolling=0.7,
            ke=SI_KE, kd=SI_KD, kf=SI_KF)
        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds, requires_grad=True)

    def _setup_target(self, target_xyz_rel, target_t):
        T = self.clock.total_sim_steps
        dt = self.clock.dt
        t_sim = np.arange(T + 1) * dt
        target_xy_rel = np.zeros((T + 1, 2), dtype=np.float32)
        for c in range(2):
            target_xy_rel[:, c] = np.interp(t_sim, target_t, target_xyz_rel[:, c])
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])
        sim_origin = self.states[0].body_q.numpy()[0, :2]  # world 0's chassis spawn
        target_xy_world = target_xy_rel + sim_origin.astype(np.float32)
        self.target_xy = wp.array(target_xy_world, dtype=wp.vec2, requires_grad=False,
                                   device=self.model.device)

    def _expand(self, params):
        return self.W @ params

    def _contract(self, grad_v):
        safe = np.where(self.W_col_sums > 0, self.W_col_sums, 1.0)
        return (self.W.T @ grad_v) / safe[:, None]

    def _apply_params(self, params):
        """Broadcast same spline-expanded controls across all worlds.

        joint_target_vel is a flat [N * dofs_per_world] array (Newton's
        replicated convention), so we tile the per-world wheel velocities
        across the N world chunks.
        """
        T = self.clock.total_sim_steps
        expanded = self._expand(params)  # [T, 3]
        for i in range(T):
            ctrl_np = np.zeros(self.num_worlds * self.dofs_per_world, dtype=np.float32)
            for w in range(self.num_worlds):
                base = w * self.dofs_per_world
                ctrl_np[base + WHEEL_DOF_OFFSET:base + WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = expanded[i]
            wp.copy(self.controls[i].joint_target_vel,
                    wp.array(ctrl_np, dtype=wp.float32, device=self.model.device))

    def compute_loss(self):
        T = self.clock.total_sim_steps
        weight = 1.0 / (T * self.num_worlds)
        for i in range(T):
            wp.launch(
                chassis_xy_loss_multiworld_kernel, dim=self.num_worlds,
                inputs=[self.states[i].body_q, self.bodies_per_world,
                        self.target_xy, i, weight],
                outputs=[self.loss],
                device=self.solver.model.device)

    def update(self):
        pass

    def opt_step(self, clip_grad_norm=1.0):
        self.diff_step()
        wp.synchronize()
        loss_val = float(self.loss.numpy()[0])
        # Collect per-world wheel grads and average — loss is already normalized
        # 1/(T*N), so the average across worlds is the right (batch-size
        # independent) spline gradient.
        T = self.clock.total_sim_steps
        grad_v = np.zeros((T, NUM_WHEEL_DOFS), dtype=np.float32)
        for i in range(T):
            g = self.controls[i].joint_target_vel.grad.numpy()
            acc = np.zeros(NUM_WHEEL_DOFS, dtype=np.float64)
            for w in range(self.num_worlds):
                base = w * self.dofs_per_world
                acc += g[base + WHEEL_DOF_OFFSET:base + WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS]
            grad_v[i] = (acc / self.num_worlds).astype(np.float32)
            self.controls[i].joint_target_vel.grad.zero_()
        grad_params = self._contract(grad_v)
        gnorm = float(np.linalg.norm(grad_params))
        if clip_grad_norm is not None and gnorm > clip_grad_norm:
            grad_params = grad_params * (clip_grad_norm / gnorm)
        self.spline_params = self.spline_adam.step(self.spline_params, grad_params)
        self._apply_params(self.spline_params)
        self.tape.zero()
        self.loss.zero_()
        return loss_val, gnorm


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
    ec = SemiImplicitEngineConfig(
        angular_damping=0.05, friction_smoothing=0.1,
        joint_attach_ke=SI_JOINT_ATTACH_KE, joint_attach_kd=SI_JOINT_ATTACH_KD)

    finalize_nvml = _nvml_poller()

    sim = HelhestJuniorBoxSIScalability(sim_cfg, rc, ec, LoggingConfig(),
                                          real_xyz, real_t)
    rng = np.random.default_rng(42)
    init = 2.0 + 0.5 * rng.standard_normal((K, NUM_WHEEL_DOFS))
    sim.spline_params = init.astype(np.float64)
    sim.spline_adam = SplineAdam(K=K, num_dofs=NUM_WHEEL_DOFS, lr=0.02)
    sim._apply_params(sim.spline_params)

    print(f"Optimising: T={sim.clock.total_sim_steps}, dt={DT}, K={K}, "
          f"num_worlds={args.num_worlds}, bodies_per_world={sim.bodies_per_world}, "
          f"dofs_per_world={sim.dofs_per_world}", flush=True)
    peak_mem_mb = 0.0
    time_ms_list = []
    for i in range(ITERATIONS):
        t0 = time.perf_counter()
        loss, gnorm = sim.opt_step()
        t_iter = (time.perf_counter() - t0) * 1000
        try:
            used_bytes = wp.get_mempool_used_bytes()
        except AttributeError:
            used_bytes = wp.get_mempool_used_mem_high()
        used_mb = used_bytes / 1024**2
        peak_mem_mb = max(peak_mem_mb, used_mb)
        print(f"  iter {i:3d}: loss={loss:.4f} |g|={gnorm:.3f} | t={t_iter:.0f}ms | mem={used_mb:.0f}MB",
              flush=True)
        time_ms_list.append(t_iter)

    nvml_abs, nvml_delta = (finalize_nvml() if finalize_nvml is not None else (None, None))

    results = {
        "simulator": "Semi-Implicit",
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
