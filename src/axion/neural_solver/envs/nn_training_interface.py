# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Environment wrapper that supports both ground-truth and neural model stepping.
Wraps the Axion simulation and exposes the API required by the trajectory sampler,
trainer, and evaluator.
"""

import sys
import os
import time
from pathlib import Path
from typing import Optional

import torch
import warp as wp

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.append(base_dir)

from axion.neural_solver.utils import warp_utils
from axion.neural_solver.utils.python_utils import print_ok
from axion.neural_solver.envs.axion_engine_wrapper import AxionEngineWrapper
from axion.neural_solver.neural_model_utils_providers.transformer_neural_utils_provider_new import (
    TransformerNeuralModelUtilsProvider,
)
from axion.neural_solver.neural_model_utils_providers.mlp_neural_utils_provider import (
    MLPNeuralUtilsProvider,
)

_PROVIDER_REGISTRY = {
    "TransformerNeuralModelUtilsProvider": TransformerNeuralModelUtilsProvider,
    "MLPNeuralUtilsProvider": MLPNeuralUtilsProvider,
}


def _build_utils_provider(robot_model, neural_model, utils_provider_cfg: dict, device_str: str):
    """Instantiate the correct utils provider based on ``utils_provider_cfg.name``."""
    name = utils_provider_cfg.get("name", "TransformerNeuralModelUtilsProvider")
    provider_cls = _PROVIDER_REGISTRY.get(name)
    kwargs = dict(utils_provider_cfg)
    kwargs.pop("name", None)
    # num_states_history is only meaningful for the transformer provider; pass it
    # through kwargs so MLPNeuralUtilsProvider can safely ignore it via **kwargs.
    return provider_cls(
        robot_model=robot_model,
        neural_model=neural_model,
        cfg=kwargs,
        num_states_history=utils_provider_cfg.get("num_states_history", 1),
        device=device_str,
    )


class NnTrainingInterface:
    """
    Simulation wrapper that supports both ground-truth (Axion engine) and neural
    model stepping.

    In ``ground-truth`` mode, ``step`` runs the Axion physics engine.
    In ``neural`` mode, ``step`` assembles model inputs from the state history,
    runs the neural model forward pass, and converts the prediction to next
    states -- matching the original ``NeuralEnvironment`` behaviour.
    """

    def __init__(
        self,
        env_name: str,
        num_envs: int,
        warp_env_cfg=None,
        utils_provider_cfg=None,
        neural_model=None,
        default_env_mode: str = "ground-truth",
        device: str = "cuda:0",
        render: bool = False,
        custom_articulation_builder=None,
        **kwargs,
    ):
        # Handle dict-like arguments similar to the original NeuralEnvironment.
        if utils_provider_cfg is None:
            utils_provider_cfg = {}
        if warp_env_cfg is None:
            warp_env_cfg = {}
        if custom_articulation_builder is not None:
            warp_env_cfg = {**warp_env_cfg, "custom_articulation_builder": custom_articulation_builder}

        # Resolve device so warp env and all state buffers use the requested device.
        # Pass device (string or warp.Device) to the wrapper so it resolves once and builds the model on the correct GPU.
        device_str = str(device) if isinstance(device, str) else str(wp.device_to_torch(device))

        # Use AxionEngineWrapper as backend, preserving (most of) the public API contract.
        self.simulator_wrapper = AxionEngineWrapper(
            env_name = env_name,
            num_worlds= num_envs,
            device = device,
            requires_grad= False, # Check if true
            warp_env_cfg=warp_env_cfg,
        )
        engine_lambda_dim = int(self.simulator_wrapper.next_lambdas.shape[-1])
        utils_provider_cfg = dict(utils_provider_cfg)
        utils_provider_cfg.setdefault("lambda_dim", engine_lambda_dim)

        self.utils_provider = _build_utils_provider(
            robot_model=self.simulator_wrapper.model,
            neural_model=neural_model,
            utils_provider_cfg=utils_provider_cfg,
            device_str=device_str,
        )

        assert default_env_mode in ("ground-truth", "neural")
        self.env_mode = default_env_mode

        # State buffers 
        self.states = torch.zeros(
            (self.num_envs, self.state_dim),
            device=self.torch_device,
        )
        self.lambdas = torch.zeros(
            (self.num_envs, self.utils_provider.lambda_dim),
            device=self.torch_device,
        )
        self.root_body_q = wp.to_torch(self.sim_states.body_q)[
            0 :: self.bodies_per_env, :
        ].view(self.num_envs, 7).to(self.torch_device)

    # ---- Properties used by trajectory sampler and simulation sampler ----

    @property
    def num_envs(self):
        return self.simulator_wrapper.num_envs

    @property
    def dof_q_per_env(self):
        return self.simulator_wrapper.dof_q_per_world

    @property
    def dof_qd_per_env(self):
        return self.simulator_wrapper.dof_qd_per_world

    @property
    def state_dim(self):
        return self.simulator_wrapper.dof_q_per_world + self.simulator_wrapper.dof_qd_per_world

    @property
    def lambda_dim(self):
        return int(self.simulator_wrapper.next_lambdas.shape[-1])

    @property
    def bodies_per_env(self):
        return self.simulator_wrapper.bodies_per_world

    @property
    def action_dim(self):
        return self.simulator_wrapper.control_dim

    @property
    def action_limits(self):
        return self.simulator_wrapper.control_limits

    @property
    def joint_types(self):
        return self.simulator_wrapper.joint_types

    @property
    def device(self):
        return self.simulator_wrapper.device

    @property
    def torch_device(self):
        return wp.device_to_torch(self.simulator_wrapper.device)

    @property
    def robot_name(self):
        return self.simulator_wrapper.robot_name

    @property
    def sim_states(self):
        return self.simulator_wrapper.state

    @property
    def model(self):
        return self.simulator_wrapper.model

    @property
    def eval_collisions(self):
        return self.simulator_wrapper.eval_collisions

    @property
    def num_contacts_per_env(self):
        return self.simulator_wrapper.num_contacts_per_env

    @property
    def frame_dt(self):
        return self.simulator_wrapper.frame_dt

    # ---- Mode and collisions (trajectory sampler sets these) ----

    def set_env_mode(self, env_mode: str):
        assert env_mode in ("ground-truth", "neural")
        self.env_mode = env_mode

    def set_eval_collisions(self, eval_collisions: bool):
        self.simulator_wrapper.set_eval_collisions(eval_collisions)

    def wrap2PI(self, states):
        self.utils_provider.wrap2PI(states)

    # ---- State sync (pure bookkeeping, no contacts / gravity / history) ----

    def _sync_states(self, states: Optional[torch.Tensor] = None):
        """Synchronise internal torch buffers with the simulator.

        *Ground-truth path* (``states is None``): reads joint coordinates
        produced by the engine back into torch tensors.
        *Reset / neural path* (``states`` given): pushes torch values into
        the simulator and runs forward-kinematics so body poses match.
        """
        if states is None:
            if not getattr(self.simulator_wrapper, "uses_generalized_coordinates", True):
                warp_utils.eval_ik(self.simulator_wrapper.model, self.simulator_wrapper.state)
            warp_utils.acquire_states_to_torch(self.simulator_wrapper, self.states)
        else:
            # Ensure both buffers are on the simulator device (e.g. cuda:1) to avoid cross-device kernel errors.
            states = states.to(self.torch_device)
            self.states = self.states.to(self.torch_device)
            self.states.copy_(states)
            warp_utils.assign_states_from_torch(self.simulator_wrapper, self.states)
            warp_utils.eval_fk(self.simulator_wrapper.model, self.simulator_wrapper.state)

        self.root_body_q.copy_(
            wp.to_torch(self.sim_states.body_q)[0 :: self.bodies_per_env, :].view(
                self.num_envs, 7
            )
        )
        self.utils_provider.states.copy_(self.states)
        self.utils_provider.root_body_q.copy_(self.root_body_q)
        self.utils_provider.wrap2PI(self.utils_provider.states)

    # ---- Neural-eval helpers (collision detection + history) ----

    def _collide_and_append_to_history(self):
        """Run collision detection, convert contacts & gravity, append to history.

        Must be called after ``_sync_states`` so that body poses are current.
        Used exclusively by the neural-eval path (``reset_for_eval`` /
        ``_step_neural``).
        """
        self.simulator_wrapper.contacts = self.simulator_wrapper.model.collide(
            self.simulator_wrapper.state
        )

        raw = self.convert_newton_contacts_to_contacts_for_nn_model()
        n = raw["contact_normals"].shape[1]
        contacts = {
            "contact_normals":  raw["contact_normals"].reshape(self.num_envs, n * 3),
            "contact_points_1": raw["contact_points_1"].reshape(self.num_envs, n * 3),
            "contact_depths":   raw["contact_depths"],
        }

        gravity_dir_body = self.get_gravity_dir()

        self.utils_provider.append_current_state_to_history(
            contacts=contacts,
            lambdas=self.lambdas,
            gravity_dir_body=gravity_dir_body,
        )

    # ---- Neural model step  ----

    @torch.no_grad()
    def _step_neural(self) -> torch.Tensor:
        """Step using the neural model instead of ground-truth physics.

        1. Read model inputs from the state history (contacts & gravity
           were placed there by the previous ``_collide_and_append_to_history``
           call).
        2. Run the neural-model forward pass.
        3. Convert the prediction to next states.
        4. Sync the simulator to those states, run collision detection, and
           store the fresh contacts / gravity in the history for the next step.
        """
        neural_model = self.utils_provider.neural_model
        assert neural_model is not None, (
            "Neural model must be set before stepping in neural mode. "
            "Call utils_provider.set_neural_model() first."
        )

        model_inputs = self.utils_provider.get_neural_model_inputs()
        prediction = neural_model(model_inputs)
        state_prediction = prediction['state']      # can be either full state or only velocities
        lambda_prediction = prediction['lambda']

        # Lambda-only models: advance state with ground-truth physics, lambdas from the network.
        if state_prediction is None:
            if lambda_prediction is None:
                raise RuntimeError(
                    "Neural step requires a state prediction and/or a lambda prediction."
                )
            self.simulator_wrapper.update()
            self._sync_states()
            current_lambdas = self.lambdas.unsqueeze(1)  # (B, 1, lambda_dim)
            if lambda_prediction.ndim == 3:
                lambda_prediction = lambda_prediction[:, -1:, :]
            next_lambdas = self.utils_provider.convert_prediction_to_next_lambdas(
                lambdas=current_lambdas,
                prediction=lambda_prediction,
            ).squeeze(1)
            self.lambdas.copy_(next_lambdas)
            self.utils_provider.lambdas.copy_(self.lambdas)
            self._collide_and_append_to_history()
            return self.states

        # State-only or mixed models:
        else:
            current_states = self.states.unsqueeze(1)  # (B, 1, state_dim)
            if state_prediction.ndim == 3:
                state_prediction = state_prediction[:, -1:, :]

            next_pred = self.utils_provider.convert_prediction_to_next_states(
                states=current_states,
                prediction=state_prediction,
                dt=self.frame_dt,
            ).squeeze(1)  # back to (B, pred_dim)

            # If the prediction is only the velocities, compute the next states from the velocities
            if next_pred.shape[-1] == 2:
                next_states = self.utils_provider.compute_next_state_from_qd(
                    states=current_states.squeeze(1),
                    qd_next=next_pred,
                    dt=self.frame_dt,
                )
            else:
                next_states = next_pred

            assert next_states.shape[-1] == self.state_dim
            # From this point on, we are sure that the next state has have full dim

            self._sync_states(next_states)

            if lambda_prediction is not None:
                current_lambdas = self.lambdas.unsqueeze(1)  # (B, 1, lambda_dim)
                if lambda_prediction.ndim == 3:
                    lambda_prediction = lambda_prediction[:, -1:, :]
                next_lambdas = self.utils_provider.convert_prediction_to_next_lambdas(
                    lambdas=current_lambdas,
                    prediction=lambda_prediction,
                ).squeeze(1)
                self.lambdas.copy_(next_lambdas)
                self.utils_provider.lambdas.copy_(self.lambdas)

            self._collide_and_append_to_history()
            return self.states

    # ---- Ground-truth stepping (dataset generation) ----

    def step(
        self,
        env_mode: Optional[str] = None,
    ) -> torch.Tensor:
        if env_mode is None:
            env_mode = self.env_mode
        self.set_env_mode(env_mode)

        if env_mode == "neural":
            return self._step_neural()
        else:
            self.simulator_wrapper.update()
            self._sync_states()
            self.lambdas.copy_(self.get_lambdas())
            if self.utils_provider.lambdas.shape[-1] == self.lambdas.shape[-1]:
                self.utils_provider.lambdas.copy_(self.lambdas)
            return self.states

    def step_with_joint_target_pos(
        self,
        joint_target_pos: torch.Tensor,
        env_mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Step with implicit PD position targets (``control.joint_target_pos``)."""
        if env_mode is None:
            env_mode = self.env_mode
        self.set_env_mode(env_mode)

        if self.dof_q_per_env > 0:
            joint_target_pos = joint_target_pos.to(self.torch_device)
            self.simulator_wrapper.control.joint_target_pos.assign(
                wp.from_torch(joint_target_pos.reshape(-1))
            )

        if env_mode == "neural":
            return self._step_neural()
        else:
            self.simulator_wrapper.update()
            self._sync_states()
            self.lambdas.copy_(self.get_lambdas())
            if self.utils_provider.lambdas.shape[-1] == self.lambdas.shape[-1]:
                self.utils_provider.lambdas.copy_(self.lambdas)
            return self.states

    def get_lambdas(self):
        """
        Call this function after simulation_wrapper.update() to get the lambdas
        after stepping the engine
        """
        return wp.to_torch(self.simulator_wrapper.next_lambdas).to(self.torch_device)

    # ---- Contact / gravity helpers (called explicitly by dataset sampler) ----

    def convert_newton_contacts_to_contacts_for_nn_model(self):
        """Process simulator contacts into the format expected by the NN."""
        return self.utils_provider.convert_newton_contacts_to_contacts_for_nn_model(
            self.simulator_wrapper.state,
            self.simulator_wrapper.contacts
        )

    def get_gravity_dir(self):
        """Return the gravity vector converted to the body's reference frame."""
        return self.utils_provider.convert_gravity_vec_w2b(self.simulator_wrapper.state)

    # ---- Reset (dataset generation) ----

    def reset(
        self,
        initial_states: torch.Tensor,
        plane_normals: Optional[torch.Tensor] = None,
        plane_d_coefficients: Optional[torch.Tensor] = None,
    ) -> None:
        """Reset the simulator to *initial_states*.

        Used by dataset-generation samplers.  Does **not** populate the state
        history or run collision detection — the sampler reads contacts and
        gravity explicitly after each ground-truth step.
        """
        assert initial_states.shape[0] == self.num_envs

        if plane_normals is not None:
            self.simulator_wrapper.reset_scene(
                plane_normals, plane_d_coefficients=plane_d_coefficients
            )

        initial_states = initial_states.to(self.torch_device)
        self._sync_states(initial_states)
        self.lambdas.zero_()
        self.utils_provider.lambdas.zero_()

    # ---- Reset (neural eval) ----

    def reset_for_eval(
        self,
        initial_states: torch.Tensor,
        plane_normals: Optional[torch.Tensor] = None,
        plane_d_coefficients: Optional[torch.Tensor] = None,
    ) -> None:
        """Reset the simulator **and** prepare the state history for neural eval.

        In addition to what ``reset`` does, this method:
        1. Clears the utils_provider state history.
        2. Runs collision detection for *initial_states*.
        3. Converts contacts and gravity to body frame.
        4. Appends the result as the first entry in the state history so that
           the very first ``_step_neural`` call sees correct inputs.
        """
        self.utils_provider.reset()

        assert initial_states.shape[0] == self.num_envs

        if plane_normals is not None:
            self.simulator_wrapper.reset_scene(
                plane_normals, plane_d_coefficients=plane_d_coefficients
            )

        initial_states = initial_states.to(self.torch_device)
        self._sync_states(initial_states)
        self.lambdas.zero_()
        self.utils_provider.lambdas.zero_()
        self._collide_and_append_to_history()

    def close(self):
        self.simulator_wrapper.close()
