"""
Transformer-oriented neural model utilities provider.

Inherits all architecture-agnostic functionality from NeuralUtilsProvider and
adds a rolling state-history window (deque) to produce (B, T, dim) tensors
for transformer-based models.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Sequence

import torch

from axion.neural_solver.neural_model_utils_providers.neural_utils_provider import (
    NeuralUtilsProvider,
    NeuralModelUtilsProviderCfg,
    PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL,
    _ensure_bt,
)


class TransformerNeuralModelUtilsProvider(NeuralUtilsProvider):
    """
    Transformer-oriented neural model utilities provider.

    Keeps a rolling history of state snapshots and produces (B, T, dim)
    tensors without flattening the time dimension, matching what the
    transformer-based ModelMixedInput expects.
    """

    def __init__(
        self,
        robot_model,
        neural_model=None,
        *,
        num_states_history: int = 1,
        cfg: Optional[dict] = None,
        prediction_type: str = "relative",
        states_embedding_type: Optional[str] = "identical",
        angular_q_indices: Optional[Sequence[int]] = None,
        lambda_dim: Optional[int] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            robot_model=robot_model,
            neural_model=neural_model,
            cfg=cfg,
            prediction_type=prediction_type,
            states_embedding_type=states_embedding_type,
            angular_q_indices=angular_q_indices,
            lambda_dim=lambda_dim,
            device=device,
            **kwargs,
        )
        self.num_states_history = int(num_states_history)
        self.reset_states_history()

    def reset_states_history(self):
        # For transformer we do not pre-fill the history; it starts empty and
        # grows as the env is stepped.
        self.states_history: deque = deque(maxlen=self.num_states_history)

    def reset(self):
        super().reset()
        self.reset_states_history()

    def append_current_state_to_history(
        self,
        *,
        joint_acts=None,
        contacts: Optional[Dict[str, torch.Tensor]] = None,
        lambdas: Optional[torch.Tensor] = None,
        gravity_dir_body: Optional[torch.Tensor] = None,
    ):
        """
        Append a snapshot of the current env-related tensors to the history.

        Caller (e.g. env wrapper) is expected to keep self.states, self.root_body_q,
        and possibly joint_acts in sync with the simulator.
        Args:
            gravity_dir_body: gravity vector already converted to the body
                frame.  When provided it is stored instead of the world-frame
                constant ``self.gravity_dir`` so that the history matches what
                the training dataset contains.
        """
        entry: Dict[str, torch.Tensor] = {
            "root_body_q": self.root_body_q.clone(),
            "states": self.states.clone(),
            "lambdas": (
                lambdas.clone()
                if lambdas is not None
                else self.lambdas.clone()
            ),
            "gravity_dir": (
                gravity_dir_body.clone()
                if gravity_dir_body is not None
                else self.gravity_dir.clone()
            ),
        }

        if self.states_embedding_type in (None, "identical"):
            entry["states_embedding"] = entry["states"].clone()

        n = PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL
        if contacts is not None:
            for k in ("contact_normals", "contact_points_1", "contact_depths"):
                if k in contacts:
                    entry[k] = contacts[k].clone()
        else:
            entry["contact_normals"]  = torch.zeros((self.num_worlds, n * 3), device=self.torch_device)
            entry["contact_points_1"] = torch.zeros((self.num_worlds, n * 3), device=self.torch_device)
            entry["contact_depths"]   = torch.zeros((self.num_worlds, n),     device=self.torch_device)

        if joint_acts is not None:
            entry["joint_acts"] = joint_acts.clone()

        self.states_history.append(entry)

    def get_neural_model_inputs(self) -> Dict[str, torch.Tensor]:
        """
        Assemble model inputs for a transformer.

        If history is empty (e.g. dummy call for network construction),
        returns zero tensors with a singleton time dimension.
        """
        if len(self.states_history) == 0:
            processed_model_inputs: Dict[str, torch.Tensor] = {
                "root_body_q": torch.zeros_like(self.root_body_q).unsqueeze(1),
                "states": torch.zeros_like(self.states).unsqueeze(1),
                "lambdas": torch.zeros_like(self.lambdas).unsqueeze(1),
                "gravity_dir": torch.zeros_like(self.gravity_dir).unsqueeze(1),
                "contact_normals": torch.zeros(
                    (self.num_worlds, 1, 3 * PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL),
                    device=self.torch_device),
                "contact_points_1": torch.zeros(
                    (self.num_worlds, 1, 3 * PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL),
                    device=self.torch_device),
                "contact_depths": torch.zeros(
                    (self.num_worlds, 1, PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL),
                    device=self.torch_device),
            }
            if self.states_embedding_type in (None, "identical"):
                processed_model_inputs["states_embedding"] = processed_model_inputs["states"].clone()
            for name in self.expected_low_dim_keys:
                if name in ("control_active", "joint_position_control_error"):
                    processed_model_inputs[name] = torch.zeros(
                        (self.num_worlds, 1, self.dof_q_per_env),
                        dtype=torch.float32,
                        device=self.torch_device,
                    )
            return self.process_neural_model_inputs(processed_model_inputs)

        model_inputs: Dict[str, torch.Tensor] = torch.utils.data.default_collate(
            list(self.states_history)
        )
        for k in model_inputs:
            model_inputs[k] = model_inputs[k].permute(1, 0, 2)

        processed_model_inputs = self.process_neural_model_inputs(model_inputs)
        return processed_model_inputs
