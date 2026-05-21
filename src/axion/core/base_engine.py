from typing import Optional

import numpy as np
import warp as wp
from axion.collision import build_reducer
from axion.collision.warm_start import ContactWarmStarter
from axion.optim import JacobiPreconditioner
from axion.optim import PCRSolver
from axion.optim import SystemLinearData
from axion.optim import SystemOperator
from newton import Contacts
from newton import Control
from newton import Model
from newton import State
from newton.solvers import SolverBase

from axion.adjoint import compute_adjoint_rhs_kernel
from axion.adjoint import compute_body_adjoint_init_kernel
from axion.adjoint import subtract_constraint_feedback_kernel
from axion.logging import AdjointHDF5Logger
from axion.logging import DatasetHDF5Logger
from axion.logging import SimulationHDF5Logger
from axion.profiling import EngineProfiler

from .backtracking import perform_backtracking
from .contacts import AxionContacts
from .engine_config import AxionEngineConfig
from .engine_data import EngineData
from .engine_dims import EngineDimensions
from .linear_system import compute_dbody_qd_from_dbody_lambda
from .linear_system import compute_linear_system
from .linesearch import perform_linesearch
from .logging_config import LoggingConfig
from .model import AxionModel
from .residual import compute_residual
from .residual import compute_residual_gradient
from .newton_step import apply_standard_newton_step
from .force_projection import project_contact_forces_kernel


@wp.kernel
def _check_newton_residuals_kernel(
    h_norm_sq: wp.array(dtype=float),
    atol_sq: float,
    keep_running: wp.array(dtype=int),
):
    tid = wp.tid()
    # Check if the residual for this specific world exceeds the tolerance
    if tid < h_norm_sq.shape[0]:
        if h_norm_sq[tid] > atol_sq:
            keep_running[0] = 1


@wp.kernel
def _update_newton_iter_kernel(
    iter_count: wp.array(dtype=int),
    max_iters: int,
    min_iter: int,
    keep_running: wp.array(dtype=int),
):
    # This kernel is explicitly launched with dim=1
    current_iter = iter_count[0] + 1
    iter_count[0] = current_iter

    # Force NR to keep running until we have cleared the backtracking
    # warmup window. The friction kernel early-exits at iter 0 because
    # ``constr_force_prev_iter`` is zeroed at engine.step start, which
    # makes the iter-0 residual "friction-blind" — it can pass the atol
    # convergence check without representing a physically-valid state
    # (zero friction force going into the next step). Holding the loop
    # until iter_count >= backtrack_min_iter ensures any iterate
    # backtracking can pick has the friction model fully engaged.
    # Overrides the residual check above when triggered.
    if current_iter < min_iter:
        keep_running[0] = 1

    # Force stop if we hit max iterations, overriding any residual checks
    # AND the min-iter override above.
    if current_iter >= max_iters:
        keep_running[0] = 0


@wp.kernel
def increment_timestep_kernel(step_count: wp.array(dtype=int)):
    step_count[0] = step_count[0] + 1


@wp.kernel
def decrement_timestep_kernel(step_count: wp.array(dtype=int)):
    step_count[0] = step_count[0] - 1


@wp.kernel
def init_backward_counter_kernel(
    forward_timestep: wp.array(dtype=int),
    backward_timestep: wp.array(dtype=int),
):
    backward_timestep[0] = forward_timestep[0] - 1


class AxionEngineBase(SolverBase):
    def __init__(
        self,
        model: Model,
        sim_steps: int,
        config: Optional[AxionEngineConfig] = AxionEngineConfig(),
        logging_config: Optional[LoggingConfig] = LoggingConfig(),
        differentiable_simulation: bool = False,
    ):
        super().__init__(model)
        self.config = config
        self.logging_config = logging_config

        # --- 2. Model & Data Setup ---
        self.axion_model = AxionModel(model)
        self.axion_contacts = AxionContacts(model, self.config.contacts.max_per_world)

        self.dims = EngineDimensions(
            num_worlds=self.axion_model.num_worlds,
            body_count=self.axion_model.body_count,
            contact_count=self.axion_contacts.max_contacts,
            joint_count=self.axion_model.joint_count,
            joint_dof_count=self.axion_model.joint_dof_count,
            linesearch_step_count=self.config.linesearch.num_steps,
            joint_constraint_count=self.axion_model.num_joint_constraints,
            control_constraint_count=self.axion_model.num_control_constraints,
        )

        self.data = EngineData(
            model=self.axion_model,
            dims=self.dims,
            config=self.config,
            device=self.device,
            alloc_history_arrays=self.logging_config.hdf5.enabled
            or self.logging_config.adjoint.enabled,
            alloc_grad_arrays=differentiable_simulation,
        )

        self.A_op = SystemOperator(
            data=SystemLinearData.from_engine(self),
            regularization=self.config.linear.regularization,
            device=self.device,
        )

        if self.config.linear.preconditioner_type == "per_body_pair":
            from axion.optim.per_body_pair_preconditioner import (
                PerBodyPairPreconditioner,
            )
            self.preconditioner = PerBodyPairPreconditioner(
                self, self.config.linear.regularization
            )
        else:  # "jacobi"
            self.preconditioner = JacobiPreconditioner(
                self, self.config.linear.regularization
            )

        self.cr_solver = PCRSolver(
            max_iters=self.config.linear.max_iters,
            batch_dim=self.dims.num_worlds,
            vec_dim=self.dims.N_c,
            device=self.device,
        )

        self.contact_reducer = build_reducer(
            self.config.contacts.reduction,
            self.axion_model,
            self.data,
            self.dims,
            self.device,
        )

        self.warm_starter = ContactWarmStarter(
            enabled=self.config.warm_start.enabled,
            axion_model=self.axion_model,
            data=self.data,
            dims=self.dims,
            device=self.device,
            cold_start_gravity=self.config.warm_start.cold_gravity,
            cold_start_impact=self.config.warm_start.cold_impact,
            cold_start_friction_v_threshold=(
                self.config.warm_start.cold_friction_v_threshold
            ),
            method=self.config.warm_start.method,
        )

        self.logger = None
        if self.logging_config.hdf5.enabled:
            self.logger = SimulationHDF5Logger(
                num_steps=sim_steps,
                data=self.data,
                config=self.config,
                dims=self.dims,
                device=self.device,
            )

        self.dataset_logger = None
        if self.logging_config.dataset.enabled:
            self.dataset_logger = DatasetHDF5Logger(
                num_steps=sim_steps,
                model=self.axion_model,
                data=self.data,
                contacts=self.axion_contacts,
                config=self.config,
                dims=self.dims,
                device=self.device,
            )

        self.adjoint_logger = None
        if self.logging_config.adjoint.enabled:
            assert (
                differentiable_simulation
            ), "logging.adjoint.enabled requires differentiable_simulation=True"
            self.adjoint_logger = AdjointHDF5Logger(
                num_steps=sim_steps,
                data=self.data,
                dims=self.dims,
                config=self.config,
                device=self.device,
            )

        self.timestep = wp.zeros(1, dtype=wp.int32, device=self.device)

        self.profiler = EngineProfiler(
            mode=config.profiling.mode,
            max_newton_iters=config.nr.max_iters,
            device=self.device,
        )

    def _save_iter_to_history(self):
        if not self.logging_config.hdf5.enabled:
            return

        wp.copy(dest=self.data.pcr_iter_count, src=self.cr_solver.iter_count)
        wp.copy(dest=self.data.pcr_final_res_norm_sq, src=self.cr_solver.r_sq)
        wp.copy(dest=self.data.pcr_res_norm_sq_history, src=self.cr_solver.history_r_sq)

        self.data.save_iter_to_history()

    def _check_convergence(self):
        # 1. Check residuals across all worlds (Sets keep_running = 1 if not converged)
        wp.launch(
            kernel=_check_newton_residuals_kernel,
            dim=(self.dims.num_worlds,),
            inputs=[
                self.data.res_norm_sq,
                self.config.nr.atol**2,
                self.data.keep_running,
            ],
            device=self.device,
        )

        # 2. Safely manage iter count with exactly 1 thread.
        #    Forces keep_running=1 while iter_count < backtrack_min_iter
        #    (the friction model's warmup window) and keep_running=0
        #    once iter_count >= max_newton_iters.
        wp.launch(
            kernel=_update_newton_iter_kernel,
            dim=(1,),
            inputs=[
                self.data.iter_count,
                self.config.nr.max_iters,
                self.config.nr.backtrack_min_iter,
                self.data.keep_running,
            ],
            device=self.device,
        )

    def reset_timestep_counter(self):
        self.timestep.zero_()

    def reset_backward_counter(self):
        wp.launch(
            kernel=init_backward_counter_kernel,
            dim=(1,),
            inputs=[self.timestep, self.backward_timestep],
            device=self.device,
        )

    def load_data(
        self,
        state_in: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ):
        self.data.dt = dt

        # =========================================================================
        # Load the data from the arguments
        # =========================================================================

        # Load the actuation data
        wp.copy(dest=self.data.ext_force, src=state_in.body_f)
        wp.copy(dest=self.data.joint_target_pos, src=control.joint_target_pos)
        wp.copy(dest=self.data.joint_target_vel, src=control.joint_target_vel)

        wp.copy(dest=self.data.body_pose_prev, src=state_in.body_q)
        wp.copy(dest=self.data.body_vel_prev, src=state_in.body_qd)

        self.axion_contacts.load_contact_data(
            contacts,
            self.axion_model,
            self.data,
            self.dims,
        )

        # Per-pair contact reduction. NoOpReducer is a Python-side return,
        # so this adds zero kernel overhead when policy="none" (the
        # default), preserving CUDA-graph capture behavior bit-for-bit.
        self.contact_reducer.apply(self.axion_contacts)

        # Cold reset of the friction-lag buffer must happen here, BEFORE
        # warm_starter.apply, so the warm starter's writes survive.
        # _constr_force itself is zeroed in engine.step() (the NR
        # initial iterate must remain λ=0 — empirically, FB-comp Newton
        # diverges from any non-zero starting λ near the touching-cone
        # boundary; tried and reverted in d4889f3 follow-up).
        self.data._constr_force_prev_iter.zero_()

        # Cross-step warm-start of contact normal/friction forces.
        # When disabled, this is a Python-side return (no kernel
        # launches). When enabled, populates _constr_force_prev_iter
        # from the previous step's converged state via
        # predicted-position matching against _prev_* buffers.
        self.warm_starter.apply(self.axion_contacts, self.data, dt)

    def compute_warm_start_forces(self):
        """Compute initial contact forces from the predicted body state.

        Assumes body_pose and body_vel have been set to the predicted state (q*, u*).
        Uses a two-pass solve to warm-start both normal and friction forces:

        Pass 1: Solve for normal + joint forces (friction inactive because
                constr_force_prev_iter is zero, so friction Jacobians are not assembled).
                Copy result to constr_force_prev_iter so friction sees nonzero normals.

        Pass 2: Re-solve with friction now active, giving a coupled warm-start.

        The result is stored in data._constr_force and data._constr_force_prev_iter,
        ready to warm-start the Newton-Raphson solver.
        """
        self.data._constr_force.zero_()
        self.data._constr_force_prev_iter.zero_()

        # --- Pass 1: normal + joint forces only (friction inactive) ---
        compute_linear_system(
            self.axion_model, self.axion_contacts, self.data, self.config, self.dims
        )
        self.data.C_values.zero_()
        self.preconditioner.update()
        self.cr_solver.solve(
            A=self.A_op,
            b=self.data.rhs,
            x=self.data._constr_force,
            preconditioner=self.preconditioner,
            iters=self.config.linear.max_iters,
            tol=self.config.linear.tol,
            atol=self.config.linear.atol,
            log=False,
        )

        # Expose pass-1 normal forces so the friction kernel activates in pass 2.
        # No projection needed: the friction kernel's early exit (mu * f_n <= 1e-6)
        # already treats negative normals as inactive contacts.
        wp.copy(dest=self.data._constr_force_prev_iter, src=self.data._constr_force)

        # --- Pass 2: re-solve with friction active ---
        self.data._constr_force.zero_()
        compute_linear_system(
            self.axion_model, self.axion_contacts, self.data, self.config, self.dims
        )
        self.data.C_values.zero_()
        self.preconditioner.update()
        self.cr_solver.solve(
            A=self.A_op,
            b=self.data.rhs,
            x=self.data._constr_force,
            preconditioner=self.preconditioner,
            iters=self.config.linear.max_iters,
            tol=self.config.linear.tol,
            atol=self.config.linear.atol,
            log=False,
        )

        wp.launch(
            kernel=project_contact_forces_kernel,
            dim=(self.dims.num_worlds, self.dims.contact_count),
            inputs=[
                self.data.constr_force.n,
                self.data.constr_force.f,
                self.axion_model.shape_material_mu,
                self.axion_model.shape_friction_axis_local,
                self.axion_model.shape_mu_perp,
                self.axion_model.shape_body,
                self.data.body_pose_prev,
                self.axion_contacts.contact_count,
                self.axion_contacts.contact_shape0,
                self.axion_contacts.contact_shape1,
                self.axion_contacts.contact_normal,
            ],
            device=self.device,
        )

        wp.copy(dest=self.data._constr_force_prev_iter, src=self.data._constr_force)

    def _solve(self):
        # =========================================================================
        # Solve non-linear system with Newton-Raphson (NR) method
        # =========================================================================
        prof = self.profiler
        per_component = prof.enabled and prof.mode == "per_component"
        end_to_end = prof.enabled and prof.mode == "end_to_end"

        def nr_loop_step(slot_idx: int = 0):
            """One NR iteration, optionally bracketed by per-phase events.

            ``slot_idx`` is the unrolled iteration index used to address
            the profiler event ring; pass 0 for the captured-while path.
            """
            self.data.keep_running.zero_()

            # Sync the friction-lag snapshot at the START of the iter so the
            # linear system and the residual evaluation within this iter both
            # see the same `prev_iter`. Previously this copy lived between the
            # solve and the step, which caused a one-iter mismatch: the linear
            # system was built using prev_iter from end of iter k-2 (friction
            # often inactive), while the residual used prev_iter from end of
            # iter k-1 (friction active). Linesearch could not reduce the
            # friction residual because the search direction had zero in
            # friction slots when the linear system thought friction was off.
            wp.copy(dest=self.data._constr_force_prev_iter, src=self.data._constr_force)

            if per_component:
                prof.record_boundary(0, slot_idx)
            compute_linear_system(
                self.axion_model, self.axion_contacts, self.data, self.config, self.dims
            )
            if per_component:
                prof.record_boundary(1, slot_idx)
            self.preconditioner.update()
            if per_component:
                prof.record_boundary(2, slot_idx)

            # Linear Solve
            self.data._dconstr_force.zero_()
            self.cr_solver.solve(
                A=self.A_op,
                b=self.data.rhs,
                x=self.data.dconstr_force.full,
                preconditioner=self.preconditioner,
                iters=self.config.linear.max_iters,
                tol=self.config.linear.tol,
                atol=self.config.linear.atol,
                log=self.logging_config.hdf5.enabled,
            )
            compute_dbody_qd_from_dbody_lambda(self.axion_model, self.data, self.config, self.dims)

            if per_component:
                prof.record_boundary(3, slot_idx)

            if self.config.linesearch.enabled:
                perform_linesearch(
                    self.axion_model, self.axion_contacts, self.data, self.config, self.dims
                )
            else:
                apply_standard_newton_step(self.axion_model, self.data, self.dims)
                compute_residual(
                    self.axion_model, self.axion_contacts, self.data, self.config, self.dims
                )
                self.data.tiled_sq_norm.compute(self.data.res.full, self.data.res_norm_sq)
            if per_component:
                prof.record_boundary(4, slot_idx)

            self.data.save_state_to_candidates()
            self._save_iter_to_history()
            self._check_convergence()
            if per_component:
                prof.record_boundary(5, slot_idx)

        # Run the NR loop
        self.data.keep_running.fill_(1)
        self.data.iter_count.zero_()
        if per_component:
            # Fixed unroll for per-iteration profiling. No early exit:
            # the loop always pays max_newton_iters iters regardless of
            # convergence. Convergence-check still runs and updates
            # keep_running for downstream code that inspects it.
            for k in range(self.config.nr.max_iters):
                nr_loop_step(slot_idx=k)
        elif self.device.is_capturing:
            wp.capture_while(self.data.keep_running, lambda: nr_loop_step(0))
        else:
            # Fallback for eager execution (no graph)
            while True:
                nr_loop_step(0)
                if self.data.keep_running.numpy()[0] == 0:
                    break

        if end_to_end:
            # boundary 4: end of NR loop, start of backtracking
            prof.record_boundary(4)
        perform_backtracking(self.axion_model, self.data, self.config, self.dims)

        # Snapshot the post-backtrack converged state so the next step
        # can warm-start its contacts. No-op when warm start is
        # disabled.
        self.warm_starter.snapshot(self.axion_contacts, self.data)

        if self.logger:
            self.logger.capture_step(self.timestep, self.data)
        if self.dataset_logger:
            self.dataset_logger.capture_step(self.timestep, self.data)

        wp.launch(
            kernel=increment_timestep_kernel, dim=(1,), inputs=[self.timestep], device=self.device
        )

    def step_backward(self):
        from axion.adjoint import adjoint_regularize_compliance_kernel
        from axion.adjoint import freeze_contact_mode_kernel
        from axion.adjoint import freeze_contact_mode_soft_kernel

        compute_linear_system(
            self.axion_model, self.axion_contacts, self.data, self.config, self.dims
        )

        # Freeze friction mode for the adjoint: replace FB-derived compliance
        # with linearized values based on the converged contact mode (sticking
        # vs sliding). Normal contacts are kept as-is (they converge well).
        # See docs/adjoint_warm_start_issue.md
        if self.config.adjoint.soft_blending:
            wp.launch(
                kernel=freeze_contact_mode_soft_kernel,
                dim=(self.dims.num_worlds, self.dims.contact_count),
                inputs=[
                    self.data.constr_active_mask.f,
                    self.data.C_values.f,
                    self.data.res.c.f,
                    self.data._constr_force[
                        :, self.dims.offset_f : self.dims.offset_f + 2 * self.dims.contact_count
                    ],
                    self.data._constr_force[:, self.dims.offset_n : self.dims.offset_f],
                    self.axion_contacts.contact_shape0,
                    self.axion_contacts.contact_shape1,
                    self.axion_contacts.contact_count,
                    self.axion_model.shape_material_mu,
                    self.config.compliance.joint,  # sticking: rigid like joints
                    100.0,  # sliding: very soft
                    self.config.adjoint.soft_blending_temperature,
                ],
                outputs=[],
                device=self.device,
            )
        else:
            wp.launch(
                kernel=freeze_contact_mode_kernel,
                dim=(self.dims.num_worlds, self.dims.contact_count),
                inputs=[
                    self.data.constr_active_mask.f,
                    self.data.C_values.f,
                    self.data.res.c.f,
                    self.data._constr_force[
                        :, self.dims.offset_f : self.dims.offset_f + 2 * self.dims.contact_count
                    ],
                    self.data._constr_force[:, self.dims.offset_n : self.dims.offset_f],
                    self.axion_contacts.contact_shape0,
                    self.axion_contacts.contact_shape1,
                    self.axion_contacts.contact_count,
                    self.axion_model.shape_material_mu,
                    self.config.compliance.joint,  # sticking: rigid like joints
                    100.0,  # sliding: very soft
                ],
                outputs=[],
                device=self.device,
            )

        # Adjoint regularization: add gamma to all constraint compliances
        # Turns [J M⁻¹ Jᵀ + C] into [J M⁻¹ Jᵀ + C + γI]
        if self.config.adjoint.regularization > 0.0:
            wp.launch(
                kernel=adjoint_regularize_compliance_kernel,
                dim=(self.dims.num_worlds, self.dims.num_constraints),
                inputs=[
                    self.data.C_values.full,
                    self.data.constr_active_mask.full,
                    self.config.adjoint.regularization,
                ],
                outputs=[],
                device=self.device,
            )

        wp.launch(
            kernel=compute_body_adjoint_init_kernel,
            dim=(self.dims.num_worlds, self.dims.body_count),
            inputs=[
                self.data.body_pose_grad,
                self.data.body_vel_grad,
                self.data.body_pose,
                self.axion_model.body_com,
                self.axion_model.body_inv_mass,
                self.axion_model.body_inv_inertia,
                self.data.dt,
            ],
            outputs=[
                self.data.w.d_spatial,
            ],
            device=self.device,
        )
        wp.launch(
            kernel=compute_adjoint_rhs_kernel,
            dim=(self.dims.num_worlds, self.dims.num_constraints),
            inputs=[
                self.data.J_values.full,
                self.data.constr_body_idx.full,
                self.data.constr_active_mask.full,
                self.data.w.d_spatial,
            ],
            outputs=[
                self.data.adjoint_rhs,
            ],
            device=self.device,
        )
        self.preconditioner.update()

        self.data.w.c.full.zero_()
        _ = self.cr_solver.solve(
            A=self.A_op,
            b=self.data.adjoint_rhs,
            x=self.data.w.c.full,
            preconditioner=self.preconditioner,
            iters=self.config.linear.max_iters,
            tol=self.config.linear.tol,
            atol=self.config.linear.atol,
            log=self.adjoint_logger is not None,
        )

        wp.launch(
            kernel=subtract_constraint_feedback_kernel,
            dim=(self.dims.num_worlds, self.dims.num_constraints),
            inputs=[
                self.data.w.c.full,
                self.data.J_values.full,
                self.data.constr_body_idx.full,
                self.data.constr_active_mask.full,
                self.data.body_pose,
                self.axion_model.body_inv_mass,
                self.axion_model.body_inv_inertia,
            ],
            outputs=[
                self.data.w.d_spatial,
            ],
            device=self.device,
        )

        self.data.w.sync_to_float()

        compute_residual_gradient(
            self.axion_model, self.axion_contacts, self.data, self.config, self.dims
        )

        if self.adjoint_logger:
            wp.copy(dest=self.data.pcr_iter_count, src=self.cr_solver.iter_count)
            wp.copy(dest=self.data.pcr_final_res_norm_sq, src=self.cr_solver.r_sq)
            wp.copy(dest=self.data.pcr_res_norm_sq_history, src=self.cr_solver.history_r_sq)
            wp.launch(
                kernel=decrement_timestep_kernel,
                dim=(1,),
                inputs=[self.timestep],
                device=self.device,
            )
            self.adjoint_logger.capture_step(self.timestep, self.data)

    def save_logs(self):
        if self.logger:
            self.logger.save_to_hdf5(self.logging_config.hdf5.file)
        if self.dataset_logger:
            self.dataset_logger.save_to_hdf5(self.logging_config.dataset.file)
        if self.adjoint_logger:
            self.adjoint_logger.save_to_hdf5(self.logging_config.adjoint.file)
