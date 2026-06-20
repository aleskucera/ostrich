"""
TeacherForcedMLPEngine — Axion solver always provides NN inputs; NN always drives simulation.

Single-state variant of TeacherForcedGPTEngine.  The engine maintains two
private Newton State buffers (_teacher_state_in, _teacher_state_out) that are
advanced by the full Axion Newton-Raphson solver every step, independently of
what the NN predicts.  Each step:

  1. Contacts are recomputed from _teacher_state_in via model.collide() so that
     both the state and the contact geometry fed to the NN always come from the
     Axion teacher trajectory, not the NN-driven outer trajectory.
  2. process_inputs() is called with the teacher state and teacher contacts so
     the SimpleMlpNeuralPredictor's single-step input bundle is filled with the
     true Axion trajectory.
  3. The Axion solver advances the teacher trajectory by one step (also using
     the teacher contacts, keeping the teacher trajectory self-consistent).
  4. The NN runs predict() and its output is written to state_out, which drives
     the actual simulation forward.

The difference from TeacherForcedGPTEngine is the network: the single-past-state
SimpleMlpModel (driven through SimpleMlpNeuralPredictor) is used, so there is no
history window / prewarm and no lambda head.

No TensorRT support — PyTorch path only.  Use execution: no_graph; the lazy
State allocation is incompatible with CUDA-graph replay.
"""

from pathlib import Path
from typing import Optional

import warp as wp
import newton
import yaml
import torch

from .base_engine import AxionEngineBase
from .engine_config import AxionEngineConfig
from .logging_config import LoggingConfig
from axion.neural_solver.standalone.simple_mlp_neural_predictor import SimpleMlpNeuralPredictor

# Trained simple-MLP checkpoint (state-only, single-step).
BEST_SIMPLE_MLP_MODEL = Path("simple_mlp") / "06-18-2026-23-38-58"

NN_BASE_PATH = Path.cwd() / "src" / "axion" / "neural_solver" / "train" / "trained_models" / BEST_SIMPLE_MLP_MODEL
NN_PENDULUM_PT_PATH = NN_BASE_PATH / "nn" / "best_valid_valid_model.pt"
NN_PENDULUM_CFG_PATH = NN_BASE_PATH / "cfg.yaml"


class TeacherForcedMLPEngine(AxionEngineBase):
    """
    Physics engine where the Axion Newton-Raphson solver always provides the
    NN's input context (teacher forcing), while the single-step SimpleMlpModel's
    predictions always drive the simulation output.

    An internal Axion trajectory is advanced in parallel each step, completely
    decoupled from state_out.  The NN therefore always receives ground-truth
    Axion states as inputs, regardless of its own prediction errors.

    WARNING: This physics solver is robot-model-dependent (double pendulum).
    TensorRT is not supported; use execution: no_graph.
    """

    def __init__(
        self,
        model: newton.Model,
        sim_steps: int,
        config: Optional[AxionEngineConfig] = None,
        logging_config: Optional[LoggingConfig] = None,
        differentiable_simulation: bool = False,
        nn_model_path: Path = NN_PENDULUM_PT_PATH,
        nn_cfg_path: Path = NN_PENDULUM_CFG_PATH,
    ):
        if config is None:
            config = AxionEngineConfig()
        if logging_config is None:
            logging_config = LoggingConfig()

        super().__init__(model, sim_steps, config, logging_config, differentiable_simulation)

        print("TeacherForcedMLPEngine is using the device =", self.device)

        print(f"Loading configuration from: {nn_cfg_path}")
        with open(nn_cfg_path, "r") as f:
            loaded_nn_cfg = yaml.load(f, Loader=yaml.SafeLoader)

        print(f"Loading model from: {nn_model_path}")
        loaded_nn_model, robot_name = torch.load(
            nn_model_path, map_location=str(self.device), weights_only=False
        )
        print(f"Loaded model for robot: {robot_name}")

        self.nn_predictor = SimpleMlpNeuralPredictor(
            newton_model=self.model,
            nn_model=loaded_nn_model,
            nn_cfg=loaded_nn_cfg,
            device=str(self.device),
        )

        # Internal teacher states — lazily allocated on the first step() call.
        self._teacher_state_in: Optional[newton.State] = None
        self._teacher_state_out: Optional[newton.State] = None

    def _init_teacher_states(self, state_in: newton.State) -> None:
        """Allocate and initialise the teacher state buffers from state_in."""
        self._teacher_state_in = self.model.state()
        self._teacher_state_out = self.model.state()
        wp.copy(dest=self._teacher_state_in.body_q, src=state_in.body_q)
        wp.copy(dest=self._teacher_state_in.body_qd, src=state_in.body_qd)
        wp.copy(dest=self._teacher_state_in.joint_q, src=state_in.joint_q)
        wp.copy(dest=self._teacher_state_in.joint_qd, src=state_in.joint_qd)

    def step(
        self,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control,
        contacts: newton.Contacts,
        dt: float,
    ):
        # Lazy initialisation: seed teacher trajectory from the first real state.
        if self._teacher_state_in is None:
            self._init_teacher_states(state_in)

        # ------------------------------------------------------------------
        # Recompute contacts from the Axion teacher state so that both the
        # joint state and the contact geometry in the NN's input come from the
        # same Axion trajectory (not the NN-driven outer trajectory).
        # ------------------------------------------------------------------
        teacher_newton_contacts = self.model.collide(self._teacher_state_in)
        teacher_axion_contacts = self.nn_predictor.create_axion_contacts(teacher_newton_contacts)

        # ------------------------------------------------------------------
        # Teacher forcing: feed the current Axion state to the NN's input bundle.
        # ------------------------------------------------------------------
        self.nn_predictor.process_inputs(self._teacher_state_in, teacher_axion_contacts, dt)

        # ------------------------------------------------------------------
        # Advance the internal Axion teacher trajectory by one step.
        # Use the teacher contacts so the teacher solver is self-consistent.
        # ------------------------------------------------------------------
        self.load_data(self._teacher_state_in, control, teacher_newton_contacts, dt)
        wp.copy(dest=self.data.body_pose, src=self._teacher_state_in.body_q)
        wp.copy(dest=self.data.body_vel, src=self._teacher_state_in.body_qd)
        self.data._constr_force.zero_()
        self.data._constr_force_prev_iter.zero_()
        self._solve()
        wp.copy(dest=self._teacher_state_out.body_q, src=self.data.body_pose)
        wp.copy(dest=self._teacher_state_out.body_qd, src=self.data.body_vel)
        newton.eval_ik(
            self.model,
            self._teacher_state_out,
            self._teacher_state_out.joint_q,
            self._teacher_state_out.joint_qd,
        )

        # ------------------------------------------------------------------
        # NN integration: NN drives the actual simulation output.
        # ------------------------------------------------------------------
        state_predicted, _ = self.nn_predictor.predict(dt=dt)

        dof_q = self.nn_predictor.dof_q_per_env
        wp.copy(
            dest=state_out.joint_q,
            src=wp.from_torch(state_predicted[0, :dof_q]),
        )
        wp.copy(
            dest=state_out.joint_qd,
            src=wp.from_torch(state_predicted[0, dof_q:]),
        )
        newton.eval_fk(self.model, state_out.joint_q, state_out.joint_qd, state_out)

        # Advance teacher state for next step.
        self._teacher_state_in, self._teacher_state_out = (
            self._teacher_state_out,
            self._teacher_state_in,
        )
