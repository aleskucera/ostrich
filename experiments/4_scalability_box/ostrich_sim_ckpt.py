"""Ostrich box scalability benchmark WITH segment checkpointing.

Same task/model/loss as ostrich_sim.py. Memory anatomy of the baseline
TrajectoryBuffer (per step, per world): body pose/vel + joint targets are tiny;
the two constraint-force arrays (~6 KB) and eight contact arrays (~13 KB)
dominate. This variant keeps the light arrays at full horizon (the loss and the
adjoint's cross-step carry need them anyway) and windows ONLY the heavy arrays
to `window` steps. The forward pass stores no lambdas/contacts; the backward
pass re-simulates each segment from a boundary warm-start snapshot to refill
the window, then runs the unchanged per-step implicit adjoint.

Because the light buffers stay full-horizon, the adjoint carry
(data.body_*_prev.grad and the full-buffer grad slots) crosses segment
boundaries with no splicing — the inner backward loop is identical to the
baseline except that lambdas/contacts come from the window.

Recompute fidelity: the resim restores the Newton warm-start arrays
(_constr_force, _constr_force_prev_iter) snapshotted at each boundary. Any
residual deviation is at solver-tolerance level (exact on CPU, where execution
is deterministic).

Usage:
    python experiments/4_scalability_box/ostrich_sim_ckpt.py --num-worlds 64 \
        --ckpt-window 8 --save results/ostrich_ckpt_64.json
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import warp as wp
from ostrich import (LoggingConfig, OstrichEngineConfig, ComplianceConfig,
                   ContactsConfig, LinearSolverConfig, LinesearchConfig,
                   NewtonRaphsonConfig, RenderingConfig, SimulationConfig)
import ostrich.simulation.differentiable_simulator as ds_module
from ostrich.simulation.trajectory_buffer import TrajectoryBuffer

from ostrich_sim import (HelhestJuniorBoxScalabilityOptimizer, SplineAdam, K, DT,
                       DURATION, ITERATIONS, NUM_WHEEL_DOFS, WHEEL_DOF_OFFSET,
                       _nvml_poller)


class CkptTrajectoryBuffer(TrajectoryBuffer):
    """TrajectoryBuffer with heavy per-step arrays windowed to `window` steps.

    Light arrays (body pose/vel, targets, ext force, joint targets) stay at the
    full horizon. Heavy arrays (constraint forces, contacts) are allocated at
    `window` slots and indexed by step_idx % window — valid because backward
    segments are aligned to multiples of `window`.
    """

    def __init__(self, data, contacts, dims, num_steps, device, window):
        self.window = window
        # Allocate everything at window size first (cheap), then re-allocate
        # the light arrays at full horizon.
        super().__init__(data, contacts, dims, num_steps=window, device=device)
        self.num_steps = num_steps

        def _full(source, requires_grad=False, add_one_slot=False):
            shape = ((num_steps + 1,) if add_one_slot else (num_steps,)) + source.shape
            return wp.zeros(shape, dtype=source.dtype, device=device,
                            requires_grad=requires_grad)

        self.target_body_pose = _full(data.body_pose, add_one_slot=True)
        self.target_body_vel = _full(data.body_vel, add_one_slot=True)
        self.ext_force = _full(data.ext_force, True)
        self.body_pose = _full(data.body_pose, True, add_one_slot=True)
        self.body_vel = _full(data.body_vel, True, add_one_slot=True)
        self.joint_target_pos = _full(data.joint_target_pos, True)
        self.joint_target_vel = _full(data.joint_target_vel, True)

        # Boundary warm-start snapshots: one per segment.
        n_segs = (num_steps + window - 1) // window
        self.n_segs = n_segs
        self._boundary_cf = wp.zeros((n_segs,) + data._constr_force.shape,
                                     dtype=data._constr_force.dtype, device=device)
        self._boundary_cf_prev = wp.zeros((n_segs,) + data._constr_force.shape,
                                          dtype=data._constr_force.dtype, device=device)

    # -- forward pass: light arrays only + boundary snapshots ---------------
    def save_step_light(self, step_idx, data):
        if step_idx == 0:
            wp.copy(self.body_pose[0], data.body_pose_prev)
            wp.copy(self.body_vel[0], data.body_vel_prev)
        wp.copy(self.body_pose[step_idx + 1], data.body_pose)
        wp.copy(self.body_vel[step_idx + 1], data.body_vel)
        wp.copy(self.ext_force[step_idx], data.ext_force)
        wp.copy(self.joint_target_pos[step_idx], data.joint_target_pos)
        wp.copy(self.joint_target_vel[step_idx], data.joint_target_vel)

    def save_boundary(self, seg_idx, data):
        wp.copy(self._boundary_cf[seg_idx], data._constr_force)
        wp.copy(self._boundary_cf_prev[seg_idx], data._constr_force_prev_iter)

    def load_boundary(self, seg_idx, data):
        wp.copy(data._constr_force, self._boundary_cf[seg_idx])
        wp.copy(data._constr_force_prev_iter, self._boundary_cf_prev[seg_idx])

    # -- resim: heavy arrays into the window --------------------------------
    def save_step_heavy(self, step_idx, data, contacts):
        w = step_idx % self.window
        wp.copy(self._constr_force[w], data._constr_force)
        wp.copy(self._constr_force_prev_iter[w], data._constr_force_prev_iter)
        wp.copy(self.contact_count[w], contacts.contact_count)
        wp.copy(self.contact_point0[w], contacts.contact_point0)
        wp.copy(self.contact_point1[w], contacts.contact_point1)
        wp.copy(self.contact_normal[w], contacts.contact_normal)
        wp.copy(self.contact_shape0[w], contacts.contact_shape0)
        wp.copy(self.contact_shape1[w], contacts.contact_shape1)
        wp.copy(self.contact_thickness0[w], contacts.contact_thickness0)
        wp.copy(self.contact_thickness1[w], contacts.contact_thickness1)

    # -- backward: light from full buffers, heavy from window ---------------
    def load_step(self, step_idx, data, contacts):
        w = step_idx % self.window
        wp.copy(data.body_pose, self.body_pose[step_idx + 1])
        wp.copy(data.body_pose_prev, self.body_pose[step_idx])
        wp.copy(data.body_vel, self.body_vel[step_idx + 1])
        wp.copy(data.body_vel_prev, self.body_vel[step_idx])
        wp.copy(data.body_pose_grad, self.body_pose.grad[step_idx + 1])
        wp.copy(data.body_vel_grad, self.body_vel.grad[step_idx + 1])
        wp.copy(data.ext_force, self.ext_force[step_idx])
        wp.copy(data.joint_target_pos, self.joint_target_pos[step_idx])
        wp.copy(data.joint_target_vel, self.joint_target_vel[step_idx])
        wp.copy(data._constr_force, self._constr_force[w])
        wp.copy(data._constr_force_prev_iter, self._constr_force_prev_iter[w])
        wp.copy(contacts.contact_count, self.contact_count[w])
        wp.copy(contacts.contact_point0, self.contact_point0[w])
        wp.copy(contacts.contact_point1, self.contact_point1[w])
        wp.copy(contacts.contact_normal, self.contact_normal[w])
        wp.copy(contacts.contact_shape0, self.contact_shape0[w])
        wp.copy(contacts.contact_shape1, self.contact_shape1[w])
        wp.copy(contacts.contact_thickness0, self.contact_thickness0[w])
        wp.copy(contacts.contact_thickness1, self.contact_thickness1[w])


class HelhestJuniorBoxCkptOptimizer(HelhestJuniorBoxScalabilityOptimizer):
    """Checkpointed variant: windowed trajectory buffer + segment resim."""

    def __init__(self, *args, ckpt_window=8, **kwargs):
        self._ckpt_window = ckpt_window
        # Intercept the base class's TrajectoryBuffer construction so the full
        # heavy buffers are never allocated (a swap-after-init would leave the
        # full allocation in the mempool high-water mark).
        orig = ds_module.TrajectoryBuffer
        ds_module.TrajectoryBuffer = (
            lambda data, contacts, dims, num_steps, device: CkptTrajectoryBuffer(
                data, contacts, dims, num_steps, device, window=self._ckpt_window))
        try:
            super().__init__(*args, **kwargs)
        finally:
            ds_module.TrajectoryBuffer = orig

    def _forward_backward(self):
        traj = self.trajectory
        k = traj.window
        T = self.clock.total_sim_steps
        traj.zero_grad()

        # --- FORWARD PASS: light saves + boundary snapshots only ---
        for i in range(T):
            if i % k == 0:
                traj.save_boundary(i // k, self.solver.data)
            self.collision_pipeline.collide(self.states[i], self.contacts)
            self.solver.step(
                state_in=self.states[i], state_out=self.states[i + 1],
                control=self.controls[i], contacts=self.contacts,
                dt=self.clock.dt)
            traj.save_step_light(i, self.solver.data)

        self.tape.reset()
        with self.tape:
            self.compute_loss()
        self.tape.backward(self.loss)

        # --- BACKWARD: per segment (last to first): resim heavy window,
        # then the unchanged per-step implicit adjoint. ---
        n_segs = (T + k - 1) // k
        for s in range(n_segs - 1, -1, -1):
            lo = s * k
            hi = min(lo + k, T)
            # Re-simulate segment [lo, hi) to refill lambdas/contacts.
            traj.load_boundary(s, self.solver.data)
            for i in range(lo, hi):
                # Restore the exact stored pre-step state; resim only to
                # recover solver internals (lambdas) and contacts.
                wp.copy(self.states[i].body_q, traj.body_pose[i])
                wp.copy(self.states[i].body_qd, traj.body_vel[i])
                self.collision_pipeline.collide(self.states[i], self.contacts)
                self.solver.step(
                    state_in=self.states[i], state_out=self.states[i + 1],
                    control=self.controls[i], contacts=self.contacts,
                    dt=self.clock.dt)
                traj.save_step_heavy(i, self.solver.data, self.solver.ostrich_contacts)
            # Per-step adjoint, identical to baseline.
            for i in range(hi - 1, lo - 1, -1):
                traj.load_step(i, self.solver.data, self.solver.ostrich_contacts)
                self.solver.data.zero_gradients()
                self.solver.step_backward()
                traj.save_gradients(i, self.solver.data)
                traj.save_pose_gradients(i, self.solver.data)
                if self.solver.config.adjoint.gradient_normalization and i > 0:
                    traj.normalize_gradients(i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-worlds", type=int, default=1)
    ap.add_argument("--ckpt-window", type=int, default=8)
    ap.add_argument("--gt", default=str(pathlib.Path(__file__).resolve().parents[1]
                                         / "1_sim_to_real_box" / "data"
                                         / "run_2026_05_20-18_10_33.json"))
    ap.add_argument("--no-cuda-graph", action="store_true")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    with open(args.gt) as f:
        gt = json.load(f)
    real_t = np.asarray(gt["real"]["t"])
    real_xyz = np.column_stack([gt["real"]["x"], gt["real"]["y"], gt["real"]["z"]])
    m = (real_t >= 0) & (real_t <= DURATION)
    real_t = real_t[m]; real_xyz = real_xyz[m]

    sim_cfg = SimulationConfig(duration_seconds=DURATION, target_timestep_seconds=DT,
                                num_worlds=args.num_worlds,
                                use_cuda_graph=not args.no_cuda_graph)
    rc = RenderingConfig(vis_type="null", target_fps=max(1, int(1 / DT)),
                         start_paused=False)
    ec = OstrichEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=16, backtrack_min_iter=12, atol=1e-3),
        linear=LinearSolverConfig(max_iters=16, tol=1e-3, atol=1e-3, regularization=1e-6),
        compliance=ComplianceConfig(joint=6e-8, contact=1e-7, friction=1e-6),
        linesearch=LinesearchConfig(enabled=False),
        contacts=ContactsConfig(max_per_world=256))

    finalize_nvml = _nvml_poller()

    sim = HelhestJuniorBoxCkptOptimizer(sim_cfg, rc, ec, LoggingConfig(),
                                          real_xyz, real_t,
                                          ckpt_window=args.ckpt_window)
    rng = np.random.default_rng(42)
    init = 2.0 + 0.5 * rng.standard_normal((K, NUM_WHEEL_DOFS))
    sim.spline_params = init.astype(np.float64)
    sim.spline_adam = SplineAdam(K=K, num_dofs=NUM_WHEEL_DOFS, lr=0.1)
    sim._apply_params(sim.spline_params)

    print(f"Optimising (ckpt window={args.ckpt_window}): "
          f"T={sim.clock.total_sim_steps}, dt={DT}, K={K}, "
          f"num_worlds={args.num_worlds}", flush=True)
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
        "simulator": "Ostrich-ckpt",
        "ckpt_window": args.ckpt_window,
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
