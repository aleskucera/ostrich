"""
Standalone Simple-MLP Neural Predictor

A minimal black-box dynamics predictor for the single-step
``SimpleMlpModel``.  It mirrors the public surface of
``NeuralPredictor`` (``create_axion_contacts`` / ``process_inputs`` /
``predict``) so a teacher-forced engine can wrap it exactly like the GPT
engine wraps ``NeuralPredictor`` -- but it is stripped to the essentials the
simple MLP needs:

  * single past state (no rolling history deque / prewarm),
  * state-only prediction (no lambda / constraint-force head),
  * a plain ``forward(input_dict) -> tensor`` model (no ``.evaluate()`` and no
    MTL/MSE model-type detection).

The input dict assembled by ``process_inputs`` uses the same keys and frame
conventions as ``MLPNeuralUtilsProvider.get_neural_model_inputs`` (the path the
model was trained with), so predictions stay consistent with training.
"""

import numpy as np
import torch
import warp as wp
import newton
from typing import Optional

try:
    from src.axion.neural_solver.standalone.neural_predictor_helpers import (
        wrap2PI,
        get_contact_masks,
        convert_contacts_w2b_batched,
        apply_contact_mask,
        convert_gravity_w2b_batched,
    )
    from src.axion.types import reorder_ground_contacts_kernel, contact_penetration_depth_kernel
    from src.axion.core.contacts import AxionContacts
except ModuleNotFoundError:
    from axion.neural_solver.standalone.neural_predictor_helpers import (
        wrap2PI,
        get_contact_masks,
        convert_contacts_w2b_batched,
        apply_contact_mask,
        convert_gravity_w2b_batched,
    )
    from axion.types import reorder_ground_contacts_kernel, contact_penetration_depth_kernel
    from axion.core.contacts import AxionContacts


PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL = 4
DT_FROM_TRAINING = 0.01


class SimpleMlpNeuralPredictor:
    """
    Standalone single-step predictor for ``SimpleMlpModel``.

    Robot configuration (dofs, joint layout, q start/end, angular/continuous
    flags) is inferred from ``newton_model``.  The predictor holds a single
    current-state input bundle (``nn_model_inputs``) rather than a history
    window, runs the torch model's ``forward`` directly, and converts the
    relative/absolute prediction back to next states the same way the trainer's
    ``NeuralUtilsProvider`` does.
    """

    def __init__(
        self,
        newton_model: newton.Model,
        nn_model: torch.nn.Module,
        nn_cfg: dict,
        device: str = "cuda:0",
        # Robot-specific configuration that would be too cumbersome to infer
        joint_q_end=[1, 2],                            # Joint DOF end indices in q vector
        is_angular_dof=[True, True, True, True],       # Which DOFs are angular
        is_continuous_dof=[True, True, False, False],  # Position DOFs (angles) are continuous, velocities are not
    ):
        """
        Args:
            newton_model: Newton physics model (robot + scene); used to infer
                DOFs and joint layout.
            nn_model: Pretrained ``SimpleMlpModel`` (loaded via torch.load).
            nn_cfg: Configuration dictionary from the trainer's cfg.yaml.
            device: Device to run on ('cuda:0', 'cpu', etc.).
        """
        self.device = device

        # Robot model reference (used for contacts and body count)
        self.robot_model = newton_model
        self.num_worlds = 1
        self.dof_q_per_env = int(newton_model.joint_coord_count) // self.num_worlds
        self.dof_qd_per_env = int(newton_model.joint_dof_count) // self.num_worlds
        self.state_dim = self.dof_q_per_env + self.dof_qd_per_env
        self.num_joints_per_env = int(newton_model.joint_count) // self.num_worlds
        self.bodies_per_world = int(newton_model.body_count) // self.num_worlds
        joint_type_np = newton_model.joint_type.numpy()
        joint_q_start_global = newton_model.joint_q_start.numpy()
        self.joint_types = joint_type_np[:self.num_joints_per_env].copy()
        self.joint_q_start = (joint_q_start_global[:self.num_joints_per_env] % self.dof_q_per_env).tolist()
        self.joint_q_end = joint_q_end
        self.is_angular_dof = np.array(is_angular_dof)
        self.is_continuous_dof = np.array(is_continuous_dof)
        self.gravity_vector = torch.zeros((self.num_worlds, 3), device=str(self.device))
        self.gravity_vector[:, self.robot_model.up_axis] = -1.0  # gravity dir from model (should be along Z)
        # Root joint pivot in first-link body (COM) frame: from model joint child xform (index 0 = root)
        joint_X_c = self.robot_model.joint_X_c.numpy()
        root_joint_idx = 0
        pivot_in_body = joint_X_c[root_joint_idx, :3].astype("float32")
        self._com_to_pivot_offset = torch.as_tensor(pivot_in_body, dtype=torch.float32, device=self.device)

        # NN model
        self.nn_model = nn_model
        self.nn_model.to(device)
        self.nn_model.eval()

        # Load NN model configuration (prediction conventions)
        env_cfg = nn_cfg.get("env", {})
        self.neural_integrator_cfg = env_cfg.get(
            "utils_provider_cfg", env_cfg.get("neural_integrator_cfg", {})
        )
        self.state_prediction_type = self.neural_integrator_cfg.get("state_prediction_type", "relative")
        self.prediction_quantity_type = self.neural_integrator_cfg.get("prediction_quantity_type", "full_state")
        self.states_embedding_type = self.neural_integrator_cfg.get("states_embedding_type", "identical")

        if self.states_embedding_type not in (None, "identical"):
            raise NotImplementedError(f"Unknown states_embedding_type: {self.states_embedding_type}")
        if self.state_prediction_type not in ("relative", "absolute"):
            raise ValueError(
                f"state_prediction_type must be 'relative' or 'absolute', got {self.state_prediction_type!r}"
            )

        # Input bundle (input to torch model), filled by process_inputs.
        self.nn_model_inputs = {}
        self._cur_states: Optional[torch.Tensor] = None

    def reset(self):
        """Clear the cached input bundle (kept for API parity)."""
        self.nn_model_inputs = {}
        self._cur_states = None

    def create_axion_contacts(self, newton_contacts):
        """Create an AxionContacts object from Newton contacts."""
        axion_contacts = AxionContacts(
            model=self.robot_model,
            max_contacts_per_world=PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL,
        )
        axion_contacts.load_contact_data(newton_contacts, self.robot_model)
        return axion_contacts

    def _convert_newton_contacts_to_contacts_for_nn_model(
        self,
        state_in,
        axion_contacts,
        root_body_q: torch.Tensor,
    ):
        """
        1.  Reorder the contacts from newton such that points_0 are always on the
            robot body and points_1 are the corresponding points on the external
            object (the contact plane).
        2.  Calculate penetration depth (used for contact masking later).
        3.  Convert the contact data to torch tensors.
        4.  Calculate the contact mask (mask that defines active contacts).
        5.  Convert points_1 and contact normals to the body (pivot) frame.
        6.  Apply the contact mask.
        """
        num_shapes_per_world = self.robot_model.shape_count // self.num_worlds
        shape = (self.num_worlds, PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL)
        device = str(self.device)
        reordered_point0 = wp.zeros(shape, dtype=wp.vec3, device=device)
        reordered_point1 = wp.zeros(shape, dtype=wp.vec3, device=device)
        reordered_normal = wp.zeros(shape, dtype=wp.vec3, device=device)
        reordered_thickness0 = wp.zeros(shape, dtype=wp.float32, device=device)
        reordered_thickness1 = wp.zeros(shape, dtype=wp.float32, device=device)
        reordered_body_shape = wp.full(shape, -1, dtype=wp.int32, device=device)
        body_contact_count = wp.zeros((self.num_worlds, self.bodies_per_world), dtype=wp.int32, device=device)

        shape_body_2d = self.robot_model.shape_body.reshape((self.num_worlds, num_shapes_per_world))

        wp.launch(
            kernel=reorder_ground_contacts_kernel,
            dim=(self.num_worlds, axion_contacts.max_contacts),
            inputs=[
                axion_contacts.contact_count,
                axion_contacts.contact_shape0,
                axion_contacts.contact_shape1,
                axion_contacts.contact_point0,
                axion_contacts.contact_point1,
                axion_contacts.contact_normal,
                axion_contacts.contact_thickness0,
                axion_contacts.contact_thickness1,
                shape_body_2d,
                self.bodies_per_world,  # Newton uses global body indices; kernel converts to per-world
                body_contact_count,
            ],
            outputs=[
                reordered_point0,  # Always body
                reordered_point1,  # Always ground
                reordered_normal,
                reordered_thickness0,  # Always body
                reordered_thickness1,  # Always ground
                reordered_body_shape,  # Body shape index for each contact
            ],
            device=str(self.device),
        )

        # Calculate penetration depth using reordered contact data
        contact_depths_wp_array = wp.zeros(
            (self.num_worlds, PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL),
            dtype=wp.float32, device=str(self.device),
        )
        body_q_2d = state_in.body_q.reshape((self.num_worlds, self.bodies_per_world))

        wp.launch(
            kernel=contact_penetration_depth_kernel,
            dim=(self.num_worlds, PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL),
            inputs=[
                body_q_2d,
                shape_body_2d,
                self.bodies_per_world,  # Newton uses global body indices; kernel converts to per-world
                reordered_point0,  # Body points (reordered)
                reordered_point1,  # Ground points (reordered)
                reordered_normal,  # Normal from body to ground (reordered)
                reordered_thickness0,  # Body thickness (reordered)
                reordered_thickness1,  # Ground thickness (reordered)
                reordered_body_shape,  # Body shape indices
            ],
            outputs=[contact_depths_wp_array],
            device=str(self.device),
        )

        # Convert to torch -- shapes: (num_worlds, num_contacts, 3) for vec3,
        # (num_worlds, num_contacts) for scalars.
        contact_depths = wp.to_torch(contact_depths_wp_array)
        contact_normals = wp.to_torch(reordered_normal)
        contact_thickness = wp.to_torch(reordered_thickness0)  # Body thickness
        contact_points_0 = wp.to_torch(reordered_point0)  # Body points
        contact_points_1 = wp.to_torch(reordered_point1)  # Ground points
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

        # Convert contact points_1 and normals from world to body (pivot) frame
        contact_points_1_body, contact_normals_body = convert_contacts_w2b_batched(
            root_body_q,
            contact_points_1,
            contact_normals,
            translation_only=False,
            com_to_pivot_offset=self._com_to_pivot_offset,
        )

        contacts["contact_points_1"] = contact_points_1_body
        contacts["contact_normals"] = contact_normals_body

        # Zero out inactive contacts
        apply_contact_mask(contacts, contact_masks)

        return contacts  # processed contacts: converted to body reference frame and masked

    def _convert_gravity_vec_w2b(self, root_body_q: torch.Tensor):
        """Convert the gravity vector to the body frame. root_body_q: (num_worlds, 7)."""
        return convert_gravity_w2b_batched(root_body_q, self.gravity_vector)

    def process_inputs(
        self,
        state_in,        # newton.State
        axion_contacts,  # AxionContacts
        dt: float,
    ):
        """
        Assemble the single-step model input bundle: coordinate-frame conversion
        of contacts/gravity, state embedding, and continuous-DOF wrapping.

        Mirrors the keys/frames of ``MLPNeuralUtilsProvider.get_neural_model_inputs``.
        """
        # Axion engine integrates maximal coordinates (body_q/body_qd). Ensure
        # generalized coordinates are synchronized before reading joint_q/joint_qd.
        newton.eval_ik(self.robot_model, state_in, state_in.joint_q, state_in.joint_qd)

        # Minimal-coordinate representation from newton's state -> (1, state_dim)
        state_min_coords = torch.cat((wp.to_torch(state_in.joint_q), wp.to_torch(state_in.joint_qd)))
        states = state_min_coords.unsqueeze(0).to(self.device)

        # Root body q (q of first pendulum link); (num_worlds, 7)
        body_q_2d = state_in.body_q.reshape((self.num_worlds, self.bodies_per_world))
        body_q_torch = wp.to_torch(body_q_2d)  # (num_worlds, bodies_per_world, 7)
        root_body_q = body_q_torch[:, 0, :].to(self.device)  # (num_worlds, 7)

        # Wrap continuous DOFs (in place)
        wrap2PI(states, self.is_continuous_dof)

        # Process contacts and gravity
        processed_contacts = self._convert_newton_contacts_to_contacts_for_nn_model(
            state_in, axion_contacts, root_body_q
        )
        gravity_in_body = self._convert_gravity_vec_w2b(root_body_q)

        # State embedding (identical)
        states_embedding = states.clone()

        # Flatten contact tensors to match training format:
        # (num_worlds, num_contacts, 3) -> (num_worlds, num_contacts * 3)
        contact_normals = processed_contacts["contact_normals"].reshape(self.num_worlds, -1)
        contact_points_1 = processed_contacts["contact_points_1"].reshape(self.num_worlds, -1)
        contact_depths = processed_contacts["contact_depths"].reshape(self.num_worlds, -1)

        # Single-step inputs (B, D) -- no time dimension.
        self.nn_model_inputs = {
            "root_body_q": root_body_q,
            "states": states,
            "states_embedding": states_embedding,
            "gravity_dir": gravity_in_body,
            "contact_normals": contact_normals,
            "contact_points_1": contact_points_1,
            "contact_depths": contact_depths,
        }
        self._cur_states = states
        return self.nn_model_inputs

    def predict(self, dt: float) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict the next robot state.

        Returns:
            next_states: (num_worlds, state_dim) in generalized coordinates.
            next_lambdas: always ``None`` (state-only model; kept for API parity).
        """
        assert self._cur_states is not None, "process_inputs() must be called before predict()"

        with torch.no_grad():
            prediction = self.nn_model(self.nn_model_inputs)  # (num_worlds, pred_dim)
        # SimpleMlpModel.forward returns leading dims matching the input; ensure 2D.
        if prediction.ndim == 3:
            prediction = prediction[:, -1, :]

        next_states = self._convert_prediction_to_next_states(self._cur_states, prediction, dt)
        return next_states, None

    def compute_next_state_from_qd(
        self,
        states: torch.Tensor,
        qd_next: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute next state from qd via semi-implicit Euler integration."""
        assert dt == DT_FROM_TRAINING, "dt from Newton must be equal to DT_FROM_TRAINING"
        q = states[..., :self.dof_q_per_env]
        q_next = q + qd_next * dt
        return torch.cat([q_next, qd_next], dim=-1)

    def _convert_prediction_to_next_states(self, states, prediction, dt):
        """
        Convert the model prediction to next states, matching the trainer's
        ``NeuralUtilsProvider.convert_prediction_to_next_states``.

        Args:
            states: (num_worlds, state_dim)
            prediction: (num_worlds, pred_dim)
        """
        if self.prediction_quantity_type == "velocities_only":
            if self.state_prediction_type == "absolute":
                raise NotImplementedError
            qd_next = states[..., self.dof_q_per_env:] + prediction
            return self.compute_next_state_from_qd(states, qd_next, dt)

        # full_state
        if self.state_prediction_type == "absolute":
            next_states = prediction[..., :self.state_dim].clone()
        else:  # relative
            next_states = states + prediction[..., :self.state_dim]
            wrap2PI(next_states, self.is_continuous_dof)

        return next_states
