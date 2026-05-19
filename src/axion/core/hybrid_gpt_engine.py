from typing import Optional

from newton import Contacts
from newton import Control
from newton import Model
from newton import State
import warp as wp

from .base_engine import AxionEngineBase
from axion.core.engine_config import AxionEngineConfig
from axion.core.logging_config import LoggingConfig

# Neural network imports:
from pathlib import Path
import yaml
import torch
from newton import eval_fk
from axion.neural_solver.standalone.neural_predictor import (
    NeuralPredictor,
    control_active_mask_from_newton_model,
)
from axion.neural_solver.standalone.fast_neural_predictor import FastNeuralPredictor
from axion.neural_solver.standalone.neural_predictor_helpers import shift_body_qd_to_com_frame
from axion.neural_solver.utils.legacy_newton_contacts import create_axion_contacts_for_nn
from axion.neural_solver.utils.legacy_newton_contacts import restore_legacy_newton_contacts_if
from axion.neural_solver.utils.pendulum_lambda_layout import (
    PENDULUM_FULL_LAMBDA_DIM,
    PENDULUM_LAMBDA_N_CTRL,
    contract_pendulum_canonical_to_engine_lambdas_torch,
)

LOG_SPACE_MODEL = Path("mse")/"05-17-2026-22-58-43"
BEST_STATE_AND_JOINT_LAMBDA_MODEL_LATER = Path("mse") / "05-12-2026-17-30-11"  # best_valid_valid_model.pt
BEST_STATES_AND_JOINT_LAMBDA_MODEL = Path("mse") / "04-24-2026-17-02-15"  # best_valid_valid_model.pt

# Shared MSE checkpoint with .plan / .engine_meta.pt already built (see
# docs/torch_to_tensorrt_conversion.md for the rebuild recipe).
NN_BASE_PATH = Path.cwd() /"src"/"axion"/"neural_solver"/"train"/"trained_models"/BEST_STATES_AND_JOINT_LAMBDA_MODEL
NN_PENDULUM_PT_PATH = NN_BASE_PATH/"nn"/"best_valid_valid_model.pt"
NN_PENDULUM_CFG_PATH = NN_BASE_PATH/"cfg.yaml"

# Flip to True after running export_to_onnx.py + build_tensorrt_engine.py.
# Required for `execution: cuda_graph` — only the TRT path is capture-safe.
USE_TENSORRT_ENGINE = False
NN_PENDULUM_PLAN_PATH = NN_PENDULUM_PT_PATH.with_suffix(".plan")
NN_PENDULUM_META_PATH = NN_PENDULUM_PT_PATH.with_suffix(".engine_meta.pt")

# Pendulum-specific lambda warm-start cleanup applied before passing the
# neural prediction to the solver: zero entries with |λ| < HYBRID_LAMBDA_THRESH,
# and zero entries past index HYBRID_LAMBDA_ZERO_FROM (contact-feature lambdas
# the engine doesn't use for the pendulum). Mirrors the old eager-path
# `torch.where(... < 0.01, ...)` + `[..., 11:] = 0.0` logic, now done inside
# the predictor so it stays graph-capture-safe.
HYBRID_LAMBDA_THRESH = 0.01
HYBRID_LAMBDA_ZERO_FROM = 11

# See axion.neural_solver.utils.legacy_newton_contacts.
USE_LEGACY_NEWTON_CONTACT_CONVENTION = True


class HybridGPTEngine(AxionEngineBase):
    """
    Engine that uses GPT to predict initial guess for the Newton-Raphson solver.
    After that, it uses AxionEngine to solve the system of equations.
    """

    def __init__(
        self,
        model: Model,
        sim_steps: int,
        config: Optional[AxionEngineConfig] = AxionEngineConfig(),
        logging_config: Optional[LoggingConfig] = LoggingConfig(),
        differentiable_simulation: bool = False,
    ):
        super().__init__(model, sim_steps, config, logging_config, differentiable_simulation)

        #########################################
        #  Neural network initialization
        #########################################

        print("GPTEngine is using the device = ", self.device)
        nn_model_path = NN_PENDULUM_PT_PATH
        nn_cfg_path = NN_PENDULUM_CFG_PATH

        # Load the nn config file (always — needed for NeuralPredictor regardless of backend)
        print(f"Loading configuration from: {nn_cfg_path}")
        with open(nn_cfg_path, 'r') as f:
            loaded_nn_cfg = yaml.load(f, Loader=yaml.SafeLoader)

        # Load either the TensorRT engine wrapper (duck-types MSEModel) or the
        # torch .pt checkpoint, depending on the toggle above.
        if USE_TENSORRT_ENGINE:
            from axion.neural_solver.fast_inference.tensorrt_mse_engine import (
                TensorRTMSEEngine,
            )
            print(f"Loading TensorRT engine: {NN_PENDULUM_PLAN_PATH}")
            loaded_nn_model = TensorRTMSEEngine(
                plan_path=NN_PENDULUM_PLAN_PATH,
                meta_path=NN_PENDULUM_META_PATH,
                device=str(self.device),
            )
            # FastNeuralPredictor wraps the TRT engine in a capture-safe
            # process_inputs / predict cycle and handles the lambda
            # warm-start cleanup in-kernel (see HYBRID_LAMBDA_* above).
            self.nn_predictor = FastNeuralPredictor(
                newton_model=self.model,
                nn_model=loaded_nn_model,
                nn_cfg=loaded_nn_cfg,
                device=str(self.device),
                clip_small_lambdas=True,
                small_lambda_threshold=HYBRID_LAMBDA_THRESH,
                lambda_zero_from=HYBRID_LAMBDA_ZERO_FROM,
            )
        else:
            print(f"Loading model from: {nn_model_path}")
            loaded_nn_model, robot_name = torch.load(
                nn_model_path, map_location=str(self.device), weights_only=False
            )
            print(f"Loaded model for robot: {robot_name}")
            # Torch path: keep the original predictor (incompatible with
            # cuda_graph; lambda cleanup happens in `_neural_init_state_fn`).
            self.nn_predictor = NeuralPredictor(
                newton_model=self.model,
                nn_model=loaded_nn_model,
                nn_cfg=loaded_nn_cfg,
                device=str(self.device),
            )

        # Preallocated scratch for `_shift_body_qd_to_com_frame`. The kernel
        # reads the raw eval_fk velocity and writes the shifted one into
        # state.body_qd; we need a stable copy of the raw values, but a fresh
        # `wp.empty_like` per step would break CUDA-graph replay. Allocate
        # once here with shape `(body_count,)` (the same shape Newton uses for
        # `state.body_qd`).
        self._raw_body_qd = wp.empty(
            self.model.body_count,
            dtype=wp.spatial_vector,
            device=self.device,
        )

        self._sim_lambda_dim = int(self.dims.num_constraints)
        self._nn_lambda_dim = int(self.nn_predictor.lambda_dim)
        self._contract_nn_lambdas = (
            self._nn_lambda_dim == PENDULUM_FULL_LAMBDA_DIM
            and self._sim_lambda_dim == PENDULUM_FULL_LAMBDA_DIM - PENDULUM_LAMBDA_N_CTRL
        )
        if self._contract_nn_lambdas:
            print(
                "HybridGPTEngine: contracting NN λ "
                f"{self._nn_lambda_dim} → solver {self._sim_lambda_dim} "
                "(dropping canonical control slots 10–11)"
            )
            self._solver_lambdas = torch.zeros(
                (1, self._sim_lambda_dim),
                device=str(self.device),
                dtype=torch.float32,
            )
        else:
            self._solver_lambdas = None

        # Joint control modes are fixed for a built model; build the mask once so
        # cuda-graph capture never hits torch.zeros / CPU .item() in step().
        pred = self.nn_predictor
        self._control_active_mask = control_active_mask_from_newton_model(
            self.model,
            dof_q_per_env=pred.dof_q_per_env,
            num_worlds=pred.num_worlds,
            num_joints_per_env=pred.num_joints_per_env,
            joint_q_start=pred.joint_q_start,
            joint_q_end=pred.joint_q_end,
            device=pred.device,
        )

        # Exposed for external diagnostics capture (e.g., engine comparison scripts).
        # These are skipped during graph capture (see `_neural_init_state_fn`).
        self.last_predicted_next_lambdas = None
        self.last_predicted_next_body_pose = None
        self.last_predicted_next_body_vel = None

    def _neural_init_state_fn(
        self,
        state_in: State,
        state_out: State,
        axion_contacts: Contacts,
        dt: float,
        joint_target_pos: Optional[torch.Tensor] = None,
        control_active: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Perform neural network model inference to get an initial guess for the Newton method.
        For MSEModel: extracts both state and lambda predictions from the joint regression output.
        """
        # Process inputs: coordinate frame conversion, state embedding.
        self.nn_predictor.process_inputs(
            state_in,
            axion_contacts,
            dt,
            joint_target_pos=joint_target_pos,
            control_active=control_active,
        )

        # Trigger neural network inference:
        next_states, next_lambdas = self.nn_predictor.predict(dt)
        if next_lambdas is not None and not isinstance(
            self.nn_predictor, FastNeuralPredictor
        ):
            next_lambdas = torch.where(
                torch.abs(next_lambdas) < HYBRID_LAMBDA_THRESH,
                torch.zeros_like(next_lambdas),
                next_lambdas,
            )
            next_lambdas[..., HYBRID_LAMBDA_ZERO_FROM:] = 0.0

        dof_q = self.nn_predictor.dof_q_per_env
        dof_qd = self.nn_predictor.dof_qd_per_env
        # Note: rely on `next_states` being a contiguous preallocated buffer
        # (FastNeuralPredictor guarantees this). For the eager path the tensor
        # is also contiguous because it was constructed via torch.empty_like.
        wp.copy(
            dest=state_out.joint_q,
            src=wp.from_torch(next_states[0, :dof_q]),
        )
        wp.copy(
            dest=state_out.joint_qd,
            src=wp.from_torch(next_states[0, dof_q:dof_q + dof_qd]),
        )
        eval_fk(self.model, state_out.joint_q, state_out.joint_qd, state_out)

        # Newton's eval_fk produces body_qd in the parent-side joint-anchor frame,
        # but the Axion solver represents body_vel at the CoM. Shift in place so
        # both the diagnostic capture below and the warm-start copy into
        # self.data.body_vel are consistent with the engine's CoM-frame convention.
        shift_body_qd_to_com_frame(self.model, state_out, self._raw_body_qd, self.device)

        # Diagnostic capture (used by test_engines.py / comparison scripts).
        # Skipped during graph capture because `.numpy()` is a host transfer.
        if not self.device.is_capturing:
            self.last_predicted_next_body_pose = state_out.body_q.numpy().copy()
            self.last_predicted_next_body_vel = state_out.body_qd.numpy().copy()
            if next_lambdas is None:
                self.last_predicted_next_lambdas = None
            else:
                solver_lambdas_for_log = self._solver_lambdas_for_warm_start(next_lambdas)
                self.last_predicted_next_lambdas = (solver_lambdas_for_log.detach().cpu().numpy().copy())

        # Transfer neural prediction of states into solver's working arrays:
        wp.copy(dest=self.data.body_pose, src=state_out.body_q)
        wp.copy(dest=self.data.body_vel, src=state_out.body_qd)

        # Initial guess of lambda (constraint forces).
        if next_lambdas is not None and getattr(self.config, "use_neural_lambda_init", True):
            solver_lambdas = self._solver_lambdas_for_warm_start(next_lambdas)
            lambdas_wp = wp.from_torch(solver_lambdas[0])
            wp.copy(dest=self.data._constr_force, src=lambdas_wp)
            wp.copy(dest=self.data._constr_force_prev_iter, src=lambdas_wp)
        elif getattr(self.config, "use_warm_start_forces", False):
            self.compute_warm_start_forces()
        else:
            self.data._constr_force.zero_()
            self.data._constr_force_prev_iter.zero_()

    def _solver_lambdas_for_warm_start(self, next_lambdas: torch.Tensor) -> torch.Tensor:
        """Return λ in solver layout (contract 24→22 when control block is absent)."""
        if not self._contract_nn_lambdas:
            return next_lambdas

        # Torch copy_ must run on Warp's current stream during cuda-graph capture
        # (same constraint as FastNeuralPredictor's normalize / ring-buffer ops).
        pred = self.nn_predictor
        if isinstance(pred, FastNeuralPredictor):
            with torch.cuda.stream(pred._torch_stream_from_warp()):
                contract_pendulum_canonical_to_engine_lambdas_torch(
                    self._solver_lambdas, next_lambdas
                )
        else:
            contract_pendulum_canonical_to_engine_lambdas_torch(
                self._solver_lambdas, next_lambdas
            )
        return self._solver_lambdas

    def prewarm(
        self,
        state_in: State,
        contacts: Contacts,
        dt: float,
    ):
        """Seed the predictor's history buffer with the current state so the
        first captured step has a valid input. Called once eagerly by the
        InteractiveSimulator before `wp.ScopedCapture()` when the predictor
        supports it (FastNeuralPredictor only)."""
        prewarm_fn = getattr(self.nn_predictor, "prewarm", None)
        if prewarm_fn is None:
            return
        axion_contacts = create_axion_contacts_for_nn(
            self.nn_predictor,
            contacts,
            apply_legacy_convention=USE_LEGACY_NEWTON_CONTACT_CONVENTION,
        )
        prewarm_fn(state_in, axion_contacts, dt)

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ):
        prof = self.profiler
        end_to_end = prof.enabled and prof.mode == "end_to_end"
        # End-to-end phase boundaries (matches END_TO_END_PHASES / EngineProfiler):
        #   0: collide-start (simulator records before engine.step)
        #   1: load_data-start
        #   2: warm_start_copy-start
        #   3: nr_solve-start
        #   4: backtracking-start (inside _solve)
        #   5: output_copy-start
        #   6: output_copy-end
        #
        # HybridGPTEngine phase mapping (profiler summary rows):
        #   collide / load_data — same as AxionEngine (simulator + load_data)
        #   warm_start_copy — _neural_init_state_fn (NN initial guess for NR)
        #   nr_solve / backtracking / output_copy — same as AxionEngine (_solve + copy)
        # per_component — NR sub-phases inside _solve() (AxionEngineBase)

        if end_to_end:
            prof.record_boundary(1)
        self.load_data(state_in, control, contacts, dt)
        restore_legacy_newton_contacts_if(
            USE_LEGACY_NEWTON_CONTACT_CONVENTION, self.axion_contacts
        )
        if end_to_end:
            prof.record_boundary(2)

        # Extract joint target positions from control (shape: dof_q -> (1, dof_q)).
        joint_target_pos = wp.to_torch(control.joint_target_pos).unsqueeze(0)

        # Perform neural network model inference to get a initial guess for the Newton method.
        self._neural_init_state_fn(
            state_in,
            state_out,
            self.axion_contacts,
            dt,
            joint_target_pos=joint_target_pos,
            control_active=self._control_active_mask,
        )
        if end_to_end:
            prof.record_boundary(3)

        # Call Newton solver
        self._solve()
        if end_to_end:
            prof.record_boundary(5)

        wp.copy(dest=state_out.body_q, src=self.data.body_pose)
        wp.copy(dest=state_out.body_qd, src=self.data.body_vel)
        if end_to_end:
            prof.record_boundary(6)

