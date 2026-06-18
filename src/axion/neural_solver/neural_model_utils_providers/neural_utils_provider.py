"""
Base neural model utilities provider.

Architecture-agnostic functionality shared by all provider subclasses
(TransformerNeuralModelUtilsProvider, MLPNeuralUtilsProvider, …).
Subclasses must implement ``get_neural_model_inputs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import warp as wp
import newton

from axion.types import reorder_ground_contacts_kernel, contact_penetration_depth_kernel
from axion.core.contacts import AxionContacts
from axion.neural_solver.standalone.neural_predictor_helpers import (
    get_contact_masks,
    convert_contacts_w2b_batched,
    apply_contact_mask,
    convert_gravity_w2b_batched,
)

PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL = 4
# Fallback only when ``lambda_dim`` is not passed; training uses engine ``num_constraints``
# (e.g. 24 for Pendulum with joint limits + 2 control constraint rows).
PENDULUM_NUM_OF_ALL_LAMBDAS = 24


def _ensure_bt(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is (B, T, D). Accepts (B, D) or (B, T, D)."""
    if x is None:
        return x
    if x.ndim == 2:
        return x.unsqueeze(1)
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected (B,D) or (B,T,D), got shape={tuple(x.shape)}")


def _wrap_to_pi_(angles: torch.Tensor) -> torch.Tensor:
    """In-place wrap of angles to [-pi, pi)."""
    two_pi = 2.0 * torch.pi
    angles.add_(torch.pi).remainder_(two_pi).sub_(torch.pi)
    return angles


def _resolve_state_and_lambda_prediction_types(
    cfg: Optional[dict],
    kwargs: dict,
    *,
    ctor_prediction_type: str = "relative",
) -> tuple[str, str]:
    """
    Legacy ``prediction_type`` sets the default for both when the specific keys are omitted.
    Precedence per key: kwargs override cfg; each key falls back to legacy then 'relative'.
    """
    legacy = ctor_prediction_type
    if cfg is not None and "prediction_type" in cfg:
        legacy = cfg["prediction_type"]
    if "prediction_type" in kwargs:
        legacy = kwargs["prediction_type"]
    fallback = legacy if legacy is not None else "relative"
    state_pt = fallback
    lambda_pt = fallback
    if cfg is not None:
        state_pt = cfg.get("state_prediction_type", state_pt)
        lambda_pt = cfg.get("lambda_prediction_type", lambda_pt)
    state_pt = kwargs.get("state_prediction_type", state_pt)
    lambda_pt = kwargs.get("lambda_prediction_type", lambda_pt)
    return state_pt, lambda_pt


@dataclass
class NeuralModelUtilsProviderCfg:
    prediction_type: str = "relative"  # legacy default for both state and lambda
    state_prediction_type: str = "relative"  # "relative" or "absolute"
    lambda_prediction_type: str = "relative"  # "relative" or "absolute"
    states_embedding_type: Optional[str] = "identical"  # None/"identical"
    angular_q_indices: Optional[Sequence[int]] = None  # indices in q to wrap; default: all q
    lambda_dim: Optional[int] = None  # optional override for per-world lambda dimension


class NeuralUtilsProvider:
    """
    Architecture-agnostic base class for neural model utilities providers.

    Handles device setup, dimension bookkeeping, contact processing, state
    wrapping, and prediction target / next-state conversions.

    Subclasses must implement :meth:`get_neural_model_inputs`.
    """

    def __init__(
        self,
        robot_model,
        neural_model: Optional[torch.nn.Module] = None,
        *,
        cfg: Optional[dict] = None,
        prediction_type: str = "relative",
        states_embedding_type: Optional[str] = "identical",
        angular_q_indices: Optional[Sequence[int]] = None,
        lambda_dim: Optional[int] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        self.robot_model = robot_model

        if device is not None:
            self.torch_device = torch.device(device) if isinstance(device, str) else device
        else:
            model_wp_device = getattr(self.robot_model, "device", None)
            self.torch_device = (
                wp.device_to_torch(model_wp_device) if model_wp_device is not None else torch.device("cpu")
            )

        prediction_quantity_type = "full_state"
        if cfg is not None:
            states_embedding_type = cfg.get("states_embedding_type", states_embedding_type)
            angular_q_indices = cfg.get("angular_q_indices", angular_q_indices)
            prediction_quantity_type = cfg.get("prediction_quantity_type", prediction_quantity_type)
            lambda_dim = cfg.get("lambda_dim", lambda_dim)

        state_prediction_type, lambda_prediction_type = _resolve_state_and_lambda_prediction_types(
            cfg, kwargs, ctor_prediction_type=prediction_type
        )
        if state_prediction_type not in ("relative", "absolute"):
            raise ValueError(
                f"state_prediction_type must be 'relative' or 'absolute', got {state_prediction_type!r}"
            )
        if lambda_prediction_type not in ("relative", "absolute"):
            raise ValueError(
                f"lambda_prediction_type must be 'relative' or 'absolute', got {lambda_prediction_type!r}"
            )
        if states_embedding_type not in (None, "identical"):
            raise ValueError(
                "states_embedding_type must be None or 'identical'. "
                f"Got {states_embedding_type!r}"
            )

        self.state_prediction_type = state_prediction_type
        self.lambda_prediction_type = lambda_prediction_type
        # Deprecated: mirrors state path; use state_prediction_type / lambda_prediction_type.
        self.prediction_type = state_prediction_type
        self.states_embedding_type = states_embedding_type
        self.prediction_quantity_type = prediction_quantity_type
        self.num_worlds = self.robot_model.world_count

        joint_coord_count = int(getattr(self.robot_model, "joint_coord_count", 0))
        joint_dof_count = int(getattr(self.robot_model, "joint_dof_count", 0))
        if joint_coord_count <= 0 or joint_dof_count <= 0:
            raise ValueError(
                "NeuralUtilsProvider expected a Newton/Axion model with "
                f"joint_coord_count/joint_dof_count, got {joint_coord_count}/{joint_dof_count}."
            )

        self.dof_q_per_env = joint_coord_count // self.num_worlds
        self.dof_qd_per_env = joint_dof_count // self.num_worlds
        self.state_dim = self.dof_q_per_env + self.dof_qd_per_env
        self.lambda_dim = int(lambda_dim) if lambda_dim is not None else PENDULUM_NUM_OF_ALL_LAMBDAS

        if prediction_quantity_type == "full_state":
            self.state_prediction_dim = self.state_dim
            self.state_embedding_dim = self.state_dim
        elif prediction_quantity_type == "velocities_only":
            self.state_prediction_dim = self.dof_qd_per_env
            self.state_embedding_dim = self.state_dim
        self.lambda_prediction_dim = self.lambda_dim

        if angular_q_indices is None:
            self.angular_q_indices = torch.arange(self.dof_q_per_env, device="cpu")
        else:
            self.angular_q_indices = torch.tensor(list(angular_q_indices), dtype=torch.long, device="cpu")

        self.axion_contacts = AxionContacts(
            model=self.robot_model,
            max_contacts_per_world=PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL,
        )
        self.bodies_per_world = self.robot_model.body_count // self.num_worlds

        # Root joint pivot in first-link body (COM) frame: from model joint child xform (index 0 = root)
        joint_X_c = self.robot_model.joint_X_c.numpy()
        root_joint_idx = 0
        pivot_in_body = joint_X_c[root_joint_idx, :3].astype("float32")
        self._com_to_pivot_offset = torch.as_tensor(pivot_in_body, dtype=torch.float32, device=self.torch_device)

        self.root_body_q = torch.zeros((self.num_worlds, 7), device=self.torch_device, dtype=torch.float32)
        self.gravity_dir = torch.zeros((self.num_worlds, 3), device=self.torch_device, dtype=torch.float32)
        up_axis = int(getattr(self.robot_model, "up_axis", 2))
        if 0 <= up_axis < 3:
            self.gravity_dir[:, up_axis] = -1.0
        self.states = torch.zeros((self.num_worlds, self.state_dim), device=self.torch_device, dtype=torch.float32)
        self.lambdas = torch.zeros((self.num_worlds, self.lambda_dim), device=self.torch_device, dtype=torch.float32)

        self.neural_model = None
        self.set_neural_model(neural_model)

        self._simulation_step = 0

        # Keys listed in YAML inputs.low_dim; used for dummy shapes in get_neural_model_inputs.
        self.expected_low_dim_keys: tuple[str, ...] = ()

    def set_expected_low_dim_keys(self, names: Sequence[str]) -> None:
        """Declare inputs.low_dim tensor names so empty-history input bundles match training."""
        self.expected_low_dim_keys = tuple(names)

    def set_neural_model(self, neural_model: Optional[torch.nn.Module]):
        self.neural_model = neural_model
        if self.neural_model is not None:
            self.neural_model.to(self.torch_device)

    def reset(self):
        self._simulation_step = 0

    def wrap2PI(self, states: torch.Tensor):
        """
        Wrap angular coordinates in-place.

        For pendulum states, we assume q occupies [:dof_q_per_env] and all q entries are angles.
        """
        if states is None:
            return
        if states.shape[-1] != self.state_dim:
            raise ValueError(f"wrap2PI expected last dim {self.state_dim}, got {states.shape[-1]}")
        if self.angular_q_indices.numel() == 0:
            return
        q = states[..., : self.dof_q_per_env]
        q_sel = q.index_select(-1, self.angular_q_indices.to(q.device))
        _wrap_to_pi_(q_sel)
        q.index_copy_(-1, self.angular_q_indices.to(q.device), q_sel)

    def convert_newton_contacts_to_contacts_for_nn_model(
        self,
        state_in: newton.State,
        newton_contacts: newton.Contacts,
    ):
        """
        1.  Batch the 1D newton contacts into 2D arrays (world, contact)
        2.  Reorder the contacts from newton such that points_0 are always on the robot body and
            points_1 are the corresponding points on the external object (the contact plane)
        3.  Calculate penetration depth (used for contact masking later)
        4.  Convert the contact data to torch tensors.
        5.  Calculate the contact mask (mask that defines active contacts)
        6.  Convert points_1 and contact normals to the body frame (robot body frame)
        7.  Apply the contact mask
        """
        self.axion_contacts.load_contact_data(newton_contacts, self.robot_model)

        shape = (self.num_worlds, PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL)
        device = str(self.torch_device)
        reordered_point0 = wp.zeros(shape, dtype=wp.vec3, device=device)
        reordered_point1 = wp.zeros(shape, dtype=wp.vec3, device=device)
        reordered_normal = wp.zeros(shape, dtype=wp.vec3, device=device)
        reordered_thickness0 = wp.zeros(shape, dtype=wp.float32, device=device)
        reordered_thickness1 = wp.zeros(shape, dtype=wp.float32, device=device)
        reordered_body_shape = wp.full(shape, -1, dtype=wp.int32, device=device)
        body_contact_count = wp.zeros((self.num_worlds, self.bodies_per_world), dtype=wp.int32, device=device)
        if len(self.robot_model.shape_body.shape) == 1:
            num_shapes_per_world = self.robot_model.shape_count // self.num_worlds
            shape_body_2d = self.robot_model.shape_body.reshape((self.num_worlds, num_shapes_per_world))
        else:
            raise NotImplementedError("This should not happen, Newton always returns 1d flat array... hopefully")

        wp.launch(
            kernel=reorder_ground_contacts_kernel,
            dim=(self.num_worlds, self.axion_contacts.max_contacts),
            inputs=[
                self.axion_contacts.contact_count,
                self.axion_contacts.contact_shape0,
                self.axion_contacts.contact_shape1,
                self.axion_contacts.contact_point0,
                self.axion_contacts.contact_point1,
                self.axion_contacts.contact_normal,
                self.axion_contacts.contact_thickness0,
                self.axion_contacts.contact_thickness1,
                shape_body_2d,
                self.bodies_per_world,
                body_contact_count,
            ],
            outputs=[
                reordered_point0,
                reordered_point1,
                reordered_normal,
                reordered_thickness0,
                reordered_thickness1,
                reordered_body_shape,
            ],
            device=str(self.torch_device)
        )

        contact_depths_wp_array = wp.zeros(
            (self.num_worlds, PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL),
            dtype=wp.float32, device=str(self.torch_device)
        )
        if len(state_in.body_q.shape) == 1:
            body_q_2d = state_in.body_q.reshape((self.num_worlds, self.bodies_per_world))
        else:
            raise NotImplementedError("This should not happen, Newton always returns 1d flat array... hopefully")

        wp.launch(
            kernel=contact_penetration_depth_kernel,
            dim=(self.num_worlds, PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL),
            inputs=[
                body_q_2d,
                shape_body_2d,
                self.bodies_per_world,
                reordered_point0,
                reordered_point1,
                reordered_normal,
                reordered_thickness0,
                reordered_thickness1,
                reordered_body_shape,
            ],
            outputs=[contact_depths_wp_array],
            device=str(self.torch_device)
        )

        contact_depths = wp.to_torch(contact_depths_wp_array)
        contact_normals = wp.to_torch(reordered_normal)
        contact_thickness = wp.to_torch(reordered_thickness0)
        contact_points_0 = wp.to_torch(reordered_point0)
        contact_points_1 = wp.to_torch(reordered_point1)
        contacts = {
            "contact_normals": contact_normals,
            "contact_depths": contact_depths,
            "contact_thicknesses": contact_thickness,
            "contact_points_0": contact_points_0,
            "contact_points_1": contact_points_1,
        }

        contact_masks = get_contact_masks(
            contacts["contact_depths"],
            contacts["contact_thicknesses"],
        )

        body_q_torch = wp.to_torch(body_q_2d)
        root_body_q = body_q_torch[:, 0, :]

        contact_points_1_body, contact_normals_body = convert_contacts_w2b_batched(
            root_body_q,
            contact_points_1,
            contact_normals,
            translation_only=False,
            com_to_pivot_offset=self._com_to_pivot_offset,
        )

        contacts["contact_points_1"] = contact_points_1_body
        contacts["contact_normals"] = contact_normals_body

        apply_contact_mask(contacts, contact_masks)

        return contacts

    def convert_gravity_vec_w2b(self, state_in: newton.State):
        """Convert gravity vector via the root_body_q + additional translation transform."""
        body_q_2d = state_in.body_q.reshape((self.num_worlds, self.bodies_per_world))
        body_q_torch = wp.to_torch(body_q_2d)
        root_body_q = body_q_torch[:, 0, :]
        return convert_gravity_w2b_batched(root_body_q, self.gravity_dir)

    def process_neural_model_inputs(self, model_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Normalize/complete the model_inputs dict in-place:
        - ensures (B,T,dim) shapes via _ensure_bt (safe for both (B,D) and (B,T,D))
        - provides states_embedding if requested
        - wraps angles for states / next_states
        """
        if "states" not in model_inputs:
            raise KeyError("model_inputs must contain key 'states'")

        model_inputs["states"] = _ensure_bt(model_inputs["states"])
        if "next_states" in model_inputs and model_inputs["next_states"] is not None:
            model_inputs["next_states"] = _ensure_bt(model_inputs["next_states"])

        if "root_body_q" in model_inputs and model_inputs["root_body_q"] is not None:
            model_inputs["root_body_q"] = _ensure_bt(model_inputs["root_body_q"])
        if "gravity_dir" in model_inputs and model_inputs["gravity_dir"] is not None:
            model_inputs["gravity_dir"] = _ensure_bt(model_inputs["gravity_dir"])
        if "lambdas" in model_inputs and model_inputs["lambdas"] is not None:
            model_inputs["lambdas"] = _ensure_bt(model_inputs["lambdas"])
        if "next_lambdas" in model_inputs and model_inputs["next_lambdas"] is not None:
            model_inputs["next_lambdas"] = _ensure_bt(model_inputs["next_lambdas"])

        if self.states_embedding_type in (None, "identical"):
            if "states_embedding" not in model_inputs or model_inputs["states_embedding"] is None:
                model_inputs["states_embedding"] = model_inputs["states"]
            else:
                model_inputs["states_embedding"] = _ensure_bt(model_inputs["states_embedding"])

        for contact_key in ("contact_normals", "contact_points_1", "contact_depths"):
            if contact_key in model_inputs and model_inputs[contact_key] is not None:
                model_inputs[contact_key] = _ensure_bt(model_inputs[contact_key])

        for ctrl_key in ("control_active", "joint_position_control_error"):
            if ctrl_key in model_inputs and model_inputs[ctrl_key] is not None:
                model_inputs[ctrl_key] = _ensure_bt(model_inputs[ctrl_key])

        self.wrap2PI(model_inputs["states"])
        if model_inputs.get("next_states", None) is not None:
            self.wrap2PI(model_inputs["next_states"])

        return model_inputs

    def get_neural_model_inputs(self) -> Dict[str, torch.Tensor]:
        """Assemble model inputs. Must be implemented by subclasses."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_neural_model_inputs()"
        )

    def convert_next_states_to_prediction(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        dt: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Convert dataset (states, next_states) into prediction target.
        Both tensors can be (B,T,D) or (B,D).
        """
        states_bt = _ensure_bt(states)
        next_bt = _ensure_bt(next_states)

        if states_bt.shape[-1] != self.state_dim or next_bt.shape[-1] != self.state_dim:
            raise ValueError("states/next_states last dim must equal state_dim")

        if self.state_prediction_type == "absolute":
            pred = next_bt.clone()
        else:
            pred = (next_bt - states_bt)
            if self.prediction_quantity_type == "full_state":
                q_delta = pred[..., : self.dof_q_per_env]
                q_sel = q_delta.index_select(-1, self.angular_q_indices.to(q_delta.device))
                _wrap_to_pi_(q_sel)
                q_delta.index_copy_(-1, self.angular_q_indices.to(q_delta.device), q_sel)
            elif self.prediction_quantity_type == "velocities_only":
                pred = pred[..., self.dof_q_per_env:]

        if pred.shape[-1] != self.state_prediction_dim:
            raise RuntimeError("Internal error: pred dim mismatch")
        return pred

    def convert_next_lambdas_to_prediction(
        self,
        lambdas: torch.Tensor,
        next_lambdas: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert dataset (lambdas, next_lambdas) into a lambda prediction target.
        Both tensors can be (B,T,D) or (B,D).
        """
        lambdas_bt = _ensure_bt(lambdas)
        next_lambdas_bt = _ensure_bt(next_lambdas)

        if (
            lambdas_bt.shape[-1] != self.lambda_dim or
            next_lambdas_bt.shape[-1] != self.lambda_dim
        ):
            raise ValueError("lambdas/next_lambdas last dim must equal lambda_dim")

        if self.lambda_prediction_type == "absolute":
            pred = next_lambdas_bt.clone()
        else:
            pred = next_lambdas_bt - lambdas_bt

        if pred.shape[-1] != self.lambda_prediction_dim:
            raise RuntimeError("Internal error: lambda pred dim mismatch")
        return pred

    def convert_prediction_to_next_states(
        self,
        states: torch.Tensor,
        prediction: torch.Tensor,
        dt: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Convert model output prediction back into next_states.
        Tensors can be (B,T,*) or (B,*).
        """
        states_bt = _ensure_bt(states)
        pred_bt = _ensure_bt(prediction)

        if pred_bt.shape[-1] < self.state_prediction_dim:
            raise ValueError(f"prediction last dim must be at least {self.state_prediction_dim}")

        pred_bt = pred_bt[..., : self.state_prediction_dim]

        if self.prediction_quantity_type == "full_state":
            if self.state_prediction_type == "absolute":
                next_states = pred_bt.clone()
            else:
                next_states = states_bt + pred_bt
                self.wrap2PI(next_states)
        elif self.prediction_quantity_type == "velocities_only":
            next_states = states_bt[..., self.dof_q_per_env:] + pred_bt

        return next_states

    def convert_prediction_to_next_lambdas(
        self,
        lambdas: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert lambda prediction back into next_lambdas.
        Tensors can be (B,T,*) or (B,*).
        """
        lambdas_bt = _ensure_bt(lambdas)
        pred_bt = _ensure_bt(prediction)

        if pred_bt.shape[-1] < self.lambda_prediction_dim:
            raise ValueError(
                f"prediction last dim must be at least {self.lambda_prediction_dim}"
            )

        pred_bt = pred_bt[..., : self.lambda_prediction_dim]

        if self.lambda_prediction_type == "absolute":
            next_lambdas = pred_bt.clone()
        else:
            next_lambdas = lambdas_bt + pred_bt

        return next_lambdas

    def compute_next_state_from_qd(
        self,
        states: torch.Tensor,
        qd_next: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute next state from qd via semi-implicit Euler integration."""
        q = states[..., :self.dof_q_per_env]
        q_next = q + qd_next * dt
        return torch.cat([q_next, qd_next], dim=-1)

    def calculate_total_energy(self, state_min_coords: torch.Tensor) -> torch.Tensor:
        """
        Calculate total energy (kinetic + potential) per sample.

        state_min_coords: (..., state_dim) with state_dim >= 4, layout [q0, q1, q0_dot, q1_dot].
        Returns: (...,) same shape as input without the last dimension (one energy per sample).
        """
        q0 = torch.pi / 2 - state_min_coords[..., 0]
        q1 = q0 - state_min_coords[..., 1]
        q0_dot = state_min_coords[..., 2]
        q1_dot = q0_dot + state_min_coords[..., 3]
        l = 1  # TODO: get from model
        m = 1  # TODO: get from model
        g = 9.81
        E_tot = (
            0.5 * m * g * l * (-3 * torch.cos(q0) - torch.cos(q1))
            + (1 / 6) * m * l**2 * (q1_dot**2 + 4 * q0_dot**2 + 3 * q0_dot * q1_dot * torch.cos(q0 - q1))
        )
        return E_tot
