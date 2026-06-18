"""
MLP neural model utilities provider.

Inherits all architecture-agnostic functionality from NeuralUtilsProvider.
Operates on a single time step (no history window): get_neural_model_inputs
returns (B, D) tensors using the current state and the most recently stored
contact snapshot.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

from axion.neural_solver.neural_model_utils_providers.neural_utils_provider import (
    NeuralUtilsProvider,
    PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL,
)


class MLPNeuralUtilsProvider(NeuralUtilsProvider):
    """
    Single-step neural model utilities provider for MLP-based models.

    Unlike the transformer provider there is no rolling history deque.
    A single contact snapshot (``_current_contacts``) is maintained and
    updated each step via ``append_current_state_to_history``, which is
    called by ``NnTrainingInterface._collide_and_append_to_history`` using
    the same interface as the transformer provider.
    """

    def __init__(
        self,
        robot_model,
        neural_model=None,
        *,
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
        self._init_current_contacts()

    def _init_current_contacts(self):
        """Allocate zero-filled contact buffers for the current time step."""
        n = PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL
        self._current_contacts: Dict[str, torch.Tensor] = {
            "contact_normals":  torch.zeros((self.num_worlds, n * 3), device=self.torch_device),
            "contact_points_1": torch.zeros((self.num_worlds, n * 3), device=self.torch_device),
            "contact_depths":   torch.zeros((self.num_worlds, n),     device=self.torch_device),
        }
        self._current_gravity_dir_body: Optional[torch.Tensor] = None

    def reset(self):
        super().reset()
        self._init_current_contacts()

    def append_current_state_to_history(
        self,
        *,
        joint_acts=None,
        contacts: Optional[Dict[str, torch.Tensor]] = None,
        lambdas: Optional[torch.Tensor] = None,
        gravity_dir_body: Optional[torch.Tensor] = None,
    ):
        """
        Store the current contact snapshot and gravity direction.

        Mirrors the signature of ``TransformerNeuralModelUtilsProvider.append_current_state_to_history``
        so ``NnTrainingInterface`` can call the same method regardless of provider type.
        Only contact data and gravity are stored; state is read directly from
        ``self.states`` at inference time.
        """
        n = PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL
        if contacts is not None:
            for k in ("contact_normals", "contact_points_1", "contact_depths"):
                if k in contacts:
                    self._current_contacts[k] = contacts[k].clone()
        else:
            self._current_contacts = {
                "contact_normals":  torch.zeros((self.num_worlds, n * 3), device=self.torch_device),
                "contact_points_1": torch.zeros((self.num_worlds, n * 3), device=self.torch_device),
                "contact_depths":   torch.zeros((self.num_worlds, n),     device=self.torch_device),
            }

        self._current_gravity_dir_body = (
            gravity_dir_body.clone() if gravity_dir_body is not None else None
        )

    def get_neural_model_inputs(self) -> Dict[str, torch.Tensor]:
        """
        Assemble model inputs for a single-step MLP.

        Returns (B, D) tensors — no time dimension — built from the current
        state buffers and the most recently stored contact snapshot.
        """
        gravity = (
            self._current_gravity_dir_body
            if self._current_gravity_dir_body is not None
            else self.gravity_dir
        )

        inputs: Dict[str, torch.Tensor] = {
            "root_body_q":      self.root_body_q.clone(),
            "states":           self.states.clone(),
            "lambdas":          self.lambdas.clone(),
            "gravity_dir":      gravity.clone(),
            "contact_normals":  self._current_contacts["contact_normals"].clone(),
            "contact_points_1": self._current_contacts["contact_points_1"].clone(),
            "contact_depths":   self._current_contacts["contact_depths"].clone(),
        }

        if self.states_embedding_type in (None, "identical"):
            inputs["states_embedding"] = inputs["states"].clone()

        return inputs
