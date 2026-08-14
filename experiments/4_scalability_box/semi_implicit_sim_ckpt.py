"""Semi-Implicit box scalability benchmark WITH segment checkpointing (BPTT).

Same task/model/loss as semi_implicit_sim.py. The baseline's memory is
dominated by the T+1 (=12001) State objects plus the Warp tape over the full
rollout. This variant never allocates that list:

  forward   : untaped ping-pong simulation, storing only body_q/qd snapshots at
              segment boundaries (every `window` steps).
  backward  : per segment (last to first), re-simulate the segment from its
              boundary snapshot on a fresh tape (window+1 reusable states),
              record the segment's per-step loss terms, seed the segment's
              final-state gradients with the adjoint carry from the previously
              processed (later-in-time) segment, tape.backward, then harvest
              the new carry from the segment's first state and the per-step
              control gradients.

Memory: O(T/window) boundary snapshots + O(window) live states, instead of
O(T) states + O(T) taped launches. Compute: one extra forward pass (~2x).

Controls are stored per-world-shaped ([T, dofs_per_world]) and tiled into the
replicated flat layout on the fly, so control storage is batch-size
independent (the baseline stores T full Control objects).

Usage:
    python experiments/4_scalability_box/semi_implicit_sim_ckpt.py \
        --num-worlds 64 --ckpt-window 110 --save results/si_ckpt_64.json
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import newton
import numpy as np
import warp as wp
from ostrich import (LoggingConfig, RenderingConfig, SemiImplicitEngineConfig,
                   SimulationConfig)
from ostrich.simulation.base_simulator import BaseSimulator

from semi_implicit_sim import (SI_MU, SI_KE, SI_KD, SI_KF, SI_K_D_ACT,
                               SI_JOINT_ATTACH_KE, SI_JOINT_ATTACH_KD,
                               WHEEL_DOF_OFFSET, NUM_WHEEL_DOFS, K, DT, DURATION,
                               ITERATIONS, make_interp_matrix, SplineAdam,
                               chassis_xy_loss_multiworld_kernel, _nvml_poller,
                               HelhestJuniorBoxSIScalability)


@wp.kernel
def tile_control_kernel(
    ctrl_per_world: wp.array(dtype=wp.float32),   # [dofs_per_world]
    dofs_per_world: int,
    ctrl_full: wp.array(dtype=wp.float32),        # [N * dofs_per_world]
):
    i = wp.tid()
    ctrl_full[i] = ctrl_per_world[i % dofs_per_world]


@wp.kernel
def reduce_control_grad_kernel(
    grad_full: wp.array(dtype=wp.float32),        # [N * dofs_per_world]
    dofs_per_world: int,
    num_worlds: int,
    grad_out: wp.array(dtype=wp.float32),         # [dofs_per_world] (summed)
):
    d = wp.tid()
    acc = float(0.0)
    for w in range(num_worlds):
        acc += grad_full[w * dofs_per_world + d]
    grad_out[d] = acc


@wp.kernel
def add_scalar_kernel(src: wp.array(dtype=wp.float32),
                      dst: wp.array(dtype=wp.float32)):
    dst[0] = dst[0] + src[0]


class HelhestJuniorBoxSICkpt(HelhestJuniorBoxSIScalability):
    """Checkpointed SI: bypasses the full states list of the base classes."""

    def __init__(self, sim_config, render_config, engine_config, logging_config,
                 target_xyz_rel, target_t, ckpt_window=110):
        self.K = K
        self.num_worlds = sim_config.num_worlds
        self.ckpt_window = ckpt_window
        # Skip DifferentiableSimulator/NewtonDifferentiableSimulator __init__
        # (they allocate T+1 State and T Control objects); go straight to
        # BaseSimulator, then build the window machinery ourselves.
        BaseSimulator.__init__(self, sim_config, render_config, engine_config,
                               logging_config)

        self.collision_pipeline = newton.CollisionPipeline(self.model,
                                                           requires_grad=False)
        self.collision_pipeline.collide(self.current_state, self.contacts)
        self.viewer = None
        self.cuda_graph = None
        self.tape = wp.Tape()

        T = self.clock.total_sim_steps
        kw = self.ckpt_window
        self.n_segs = (T + kw - 1) // kw

        self.W, self.W_col_sums = make_interp_matrix(T, K)
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=False)
        self.seg_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.bodies_per_world = self.model.body_count // self.num_worlds

        # Window states/controls (reused every segment) + one forward control.
        self.states_win = [self.model.state(requires_grad=True)
                           for _ in range(kw + 1)]
        self.controls_win = [self.model.control(requires_grad=True)
                             for _ in range(kw)]
        self.control_fwd = self.model.control(requires_grad=False)
        self.dofs_per_world = (self.control_fwd.joint_target_vel.shape[-1]
                               // self.num_worlds)

        # Boundary snapshots.
        s0 = self.states_win[0]
        self._bq = wp.zeros((self.n_segs,) + s0.body_q.shape, dtype=s0.body_q.dtype)
        self._bqd = wp.zeros((self.n_segs,) + s0.body_qd.shape, dtype=s0.body_qd.dtype)

        # Adjoint carry.
        self.carry_q = wp.zeros_like(s0.body_q.grad)
        self.carry_qd = wp.zeros_like(s0.body_qd.grad)

        # Per-step per-world controls + harvested gradients (batch-independent).
        self.ctrl_steps = wp.zeros((T, self.dofs_per_world), dtype=wp.float32)
        self.grad_steps = wp.zeros((T, self.dofs_per_world), dtype=wp.float32)

        self._setup_target(target_xyz_rel, target_t)

    def _setup_target(self, target_xyz_rel, target_t):
        T = self.clock.total_sim_steps
        dt = self.clock.dt
        t_sim = np.arange(T + 1) * dt
        target_xy_rel = np.zeros((T + 1, 2), dtype=np.float32)
        for c in range(2):
            target_xy_rel[:, c] = np.interp(t_sim, target_t, target_xyz_rel[:, c])
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd,
                       self.current_state)
        sim_origin = self.current_state.body_q.numpy()[0, :2]
        target_xy_world = target_xy_rel + sim_origin.astype(np.float32)
        self.target_xy = wp.array(target_xy_world, dtype=wp.vec2,
                                   requires_grad=False, device=self.model.device)

    def _apply_params(self, params):
        expanded = self._expand(params)  # [T, 3]
        T = self.clock.total_sim_steps
        ctrl_np = np.zeros((T, self.dofs_per_world), dtype=np.float32)
        ctrl_np[:, WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = expanded
        wp.copy(self.ctrl_steps, wp.array(ctrl_np, dtype=wp.float32,
                                          device=self.model.device))

    def _set_control(self, control, step_idx):
        wp.launch(tile_control_kernel,
                  dim=self.num_worlds * self.dofs_per_world,
                  inputs=[self.ctrl_steps[step_idx], self.dofs_per_world],
                  outputs=[control.joint_target_vel],
                  device=self.model.device)

    def _sim_step(self, state_in, state_out, control, taped):
        if taped:
            with self.tape:
                state_in.clear_forces()
        else:
            state_in.clear_forces()
        self.collision_pipeline.collide(state_in, self.contacts)
        if taped:
            with self.tape:
                self.solver.step(state_in=state_in, state_out=state_out,
                                 control=control, contacts=self.contacts,
                                 dt=self.clock.dt)
        else:
            self.solver.step(state_in=state_in, state_out=state_out,
                             control=control, contacts=self.contacts,
                             dt=self.clock.dt)

    def _forward_backward(self):
        T = self.clock.total_sim_steps
        kw = self.ckpt_window
        weight = 1.0 / (T * self.num_worlds)
        self.loss.zero_()
        self.carry_q.zero_()
        self.carry_qd.zero_()

        # --- FORWARD (untaped, ping-pong) ---
        a, b = self.states_win[0], self.states_win[1]
        wp.copy(a.body_q, self.current_state.body_q)
        wp.copy(a.body_qd, self.current_state.body_qd)
        for i in range(T):
            if i % kw == 0:
                s = i // kw
                wp.copy(self._bq[s], a.body_q)
                wp.copy(self._bqd[s], a.body_qd)
            self._set_control(self.control_fwd, i)
            self._sim_step(a, b, self.control_fwd, taped=False)
            a, b = b, a

        # --- BACKWARD: segments last -> first ---
        for s in range(self.n_segs - 1, -1, -1):
            lo = s * kw
            hi = min(lo + kw, T)
            L = hi - lo

            self.tape.reset()
            self.seg_loss.zero_()
            for st in self.states_win:
                st.body_q.grad.zero_()
                st.body_qd.grad.zero_()
            for c in self.controls_win:
                c.joint_target_vel.grad.zero_()

            wp.copy(self.states_win[0].body_q, self._bq[s])
            wp.copy(self.states_win[0].body_qd, self._bqd[s])
            for j in range(L):
                g = lo + j
                self._set_control(self.controls_win[j], g)
                self._sim_step(self.states_win[j], self.states_win[j + 1],
                               self.controls_win[j], taped=True)
                with self.tape:
                    wp.launch(chassis_xy_loss_multiworld_kernel,
                              dim=self.num_worlds,
                              inputs=[self.states_win[j].body_q,
                                      self.bodies_per_world, self.target_xy,
                                      g, weight],
                              outputs=[self.seg_loss],
                              device=self.model.device)

            # Seed the segment-final state with the carry, then backprop.
            wp.copy(self.states_win[L].body_q.grad, self.carry_q)
            wp.copy(self.states_win[L].body_qd.grad, self.carry_qd)
            self.tape.backward(self.seg_loss)

            wp.copy(self.carry_q, self.states_win[0].body_q.grad)
            wp.copy(self.carry_qd, self.states_win[0].body_qd.grad)
            for j in range(L):
                wp.launch(reduce_control_grad_kernel, dim=self.dofs_per_world,
                          inputs=[self.controls_win[j].joint_target_vel.grad,
                                  self.dofs_per_world, self.num_worlds],
                          outputs=[self.grad_steps[lo + j]],
                          device=self.model.device)
            wp.launch(add_scalar_kernel, dim=1, inputs=[self.seg_loss],
                      outputs=[self.loss], device=self.model.device)

    def diff_step(self):
        if self.use_cuda_graph and self.cuda_graph is None:
            with wp.ScopedCapture() as capture:
                self._forward_backward()
            self.cuda_graph = capture.graph
        if self.use_cuda_graph and self.cuda_graph:
            wp.capture_launch(self.cuda_graph)
        else:
            self._forward_backward()

    def compute_loss(self):
        pass  # folded into _forward_backward segments

    def opt_step(self, clip_grad_norm=1.0):
        self.diff_step()
        wp.synchronize()
        loss_val = float(self.loss.numpy()[0])
        # grad_steps holds per-step SUMS over worlds; average like the baseline.
        grad_v = (self.grad_steps.numpy()[
            :, WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS]
            / self.num_worlds)
        grad_params = self._contract(grad_v.astype(np.float32))
        gnorm = float(np.linalg.norm(grad_params))
        if clip_grad_norm is not None and gnorm > clip_grad_norm:
            grad_params = grad_params * (clip_grad_norm / gnorm)
        self.spline_params = self.spline_adam.step(self.spline_params, grad_params)
        self._apply_params(self.spline_params)
        return loss_val, gnorm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-worlds", type=int, default=1)
    ap.add_argument("--ckpt-window", type=int, default=110)
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
    ec = SemiImplicitEngineConfig(
        angular_damping=0.05, friction_smoothing=0.1,
        joint_attach_ke=SI_JOINT_ATTACH_KE, joint_attach_kd=SI_JOINT_ATTACH_KD)

    finalize_nvml = _nvml_poller()

    sim = HelhestJuniorBoxSICkpt(sim_cfg, rc, ec, LoggingConfig(),
                                   real_xyz, real_t,
                                   ckpt_window=args.ckpt_window)
    rng = np.random.default_rng(42)
    init = 2.0 + 0.5 * rng.standard_normal((K, NUM_WHEEL_DOFS))
    sim.spline_params = init.astype(np.float64)
    sim.spline_adam = SplineAdam(K=K, num_dofs=NUM_WHEEL_DOFS, lr=0.02)
    sim._apply_params(sim.spline_params)

    print(f"Optimising (ckpt window={args.ckpt_window}, segs={sim.n_segs}): "
          f"T={sim.clock.total_sim_steps}, dt={DT}, num_worlds={args.num_worlds}",
          flush=True)
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
        print(f"  iter {i:3d}: loss={loss:.4f} |g|={gnorm:.3f} | t={t_iter:.0f}ms | "
              f"mem={used_mb:.0f}MB", flush=True)
        time_ms_list.append(t_iter)

    nvml_abs, nvml_delta = (finalize_nvml() if finalize_nvml is not None else (None, None))

    results = {
        "simulator": "Semi-Implicit-ckpt",
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
