import os
import sys
from typing import Optional, Mapping, Any

import hydra.utils
import torch
import warp as wp
import newton
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Repo root so that "from examples.*" works when run from any entry point (train.py, generate script, etc.)
_env_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_env_dir, "..", "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from examples.double_pendulum.pendulum_articulation_definition import build_pendulum_model
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.types import JointMode
from axion.neural_solver.utils.warp_utils import device_to_torch

NUM_CONTACTS_PER_WORLD = 4 # hardcoded for double pendulum
FRAME_DT = 0.01
ENGINE_SUBSTEPS = 10
ENGINE_DT = FRAME_DT/ENGINE_SUBSTEPS
ENGINE_CONF_DIR = os.path.join(_repo_root, "examples", "conf")
ENGINE_CONFIG_GROUP = "hybrid"

class AxionEngineWrapper:
    """
    Wrapper around Axion. Currently it builds the double pendulum model and calls Axion as an integrator.
    Instance of AxionEngineWrapper will be created inside nn_training_interfaceclass.  
    """
    
    def __init__(
        self,
        env_name: str,
        num_worlds: int,
        device,
        requires_grad: bool,
        warp_env_cfg: Optional[Mapping[str, Any]] = None,
        ):
        # Resolve device so the model is built on the intended GPU (e.g. cuda:1 when given "cuda:1").
        self.device = wp.get_device(device) if isinstance(device, str) else device
        self.env_name = env_name
        self.num_worlds = num_worlds
        self.requires_grad = requires_grad
        self.robot_name: str = self.env_name    # for compatibility with nn_training_interface

        warp_env_cfg = warp_env_cfg or {}
        self.with_contacts: bool = bool(warp_env_cfg.get("with_contacts", True))
        joint_dof_mode = warp_env_cfg.get("joint_dof_mode", JointMode.NONE)

        # model (robot model built by AxionModelBuilder):
        if self.env_name in ("PendulumWithContact", "Pendulum", "pendulum"):
            self.model = build_pendulum_model(
                self.num_worlds,
                self.device,
                self.requires_grad,
                with_contacts=self.with_contacts,
                joint_dof_mode=joint_dof_mode,
            )
        else:
            raise NotImplementedError
        
        # value "inherited" from the robot model:
        self.state: newton.State = self.model.state()   #current state
        self.next_state: newton.State = self.model.state()
        self.control: newton.Control = self.model.control()
        self.contacts: newton.Contacts = self.model.collide(self.state)
        
        # Integrator (Axion engine): load via Hydra (examples/conf/engine/<group>.yaml).
        engine_group = warp_env_cfg.get("engine", ENGINE_CONFIG_GROUP)
        self.engine_cfg = self._load_engine_config(engine_group)
        self.engine = AxionEngine(
            model=self.model,
            sim_steps=1,
            config=self.engine_cfg,
        )

        lambda_count = int(self.engine.dims.num_constraints)
        self.next_lambdas: wp.array = wp.zeros((self.num_worlds, lambda_count), dtype=wp.float32, device=self.device)

        # number of DOF info
        self.dof_q_per_world= int(self.model.joint_coord_count / self.model.world_count)  # 2 for planar double pendulum
        self.dof_qd_per_world= int(self.model.joint_dof_count / self.model.world_count)   # 2 for planar double pendulum
        self.bodies_per_world= int(self.model.body_count / self.model.world_count)
        
        # joint type info
        self.num_joints_per_world= int(self.model.joint_count / self.model.world_count)
        self._torch_device = device_to_torch(self.model.device)
        self.joint_types = wp.to_torch(self.model.joint_type).to(
            self._torch_device
        )[:self.num_joints_per_world]

        # control info (logical action width for limits / samplers; position targets use dof_q_per_world)
        self.control_dim = self.dof_q_per_world  # 2 for planar double pendulum
        self.control_limits = torch.tensor(
            [[-1.0, 1.0]],
            dtype=torch.float32, device=self._torch_device
        ).expand(self.control_dim, 2).clone()

        # Batched contact layout (matches utils_provider AxionContacts max per world).
        self.num_contacts_per_env: int = NUM_CONTACTS_PER_WORLD
        self.eval_collisions: bool = True   # ?

        if self.with_contacts:
            # We group all static planes by shape_world and take the *last* plane
            # in each world (the one added after add_ground_plane in the source
            # builder) as the tilted plane.
            shape_types = wp.to_torch(self.model.shape_type).to(self._torch_device)
            shape_body = wp.to_torch(self.model.shape_body).to(self._torch_device)
            shape_world = wp.to_torch(self.model.shape_world).to(self._torch_device)

            is_static_plane = ((shape_types == int(newton.GeoType.PLANE)) & (shape_body == -1))
            plane_indices = torch.where(is_static_plane)[0]

            tilted_indices = []
            for w in range(self.num_worlds):
                world_planes = plane_indices[shape_world[plane_indices] == w]
                assert world_planes.numel() >= 2, (
                    f"Expected at least 2 plane shapes in world {w}, found {world_planes.numel()}"
                )
                tilted_indices.append(world_planes[-1].item())

            self._tilted_plane_shape_indices = torch.tensor(
                tilted_indices, dtype=torch.long, device=self._torch_device
            )
        else:
            self._tilted_plane_shape_indices = None

        # misc
        self.frame_dt = FRAME_DT
        self.uses_generalized_coordinates = False  # AxionEngine is a maximal-coordinate solver

    def set_eval_collisions(self, eval_collisions: bool) -> None:
        self.eval_collisions = eval_collisions

    def update(self) -> None:
        """
        Triggers (Axion's) collision detection and step Axion Engine
        """

        self.state.clear_forces()

        if self.eval_collisions:
            self.contacts = self.model.collide(self.state)
        else:
            raise NotImplementedError       

        # Engine (integrator) step
        self.engine.step(
            state_in = self.state,
            state_out = self.next_state,
            control= self.control,
            contacts=self.contacts, 
            dt= FRAME_DT
        )

        # Copy lambdas from engine data to self.next_lambdas
        wp.copy(dest=self.next_lambdas, src=self.engine.data._constr_force)         #   (num_worlds, dims.num_constraints)  -->  (num_worlds, lambda_count)
        # Swap state and next_state
        self.state, self.next_state = self.next_state, self.state

        # AxionEngine is a maximal-coordinate solver: it only writes body_q / body_qd.
        # Recover generalized coordinates so that joint_q / joint_qd are up-to-date.
        newton.eval_ik(self.model, self.state, self.state.joint_q, self.state.joint_qd)

    def assign_control(
        self,
        actions_wp: wp.array,
        control: newton.Control,
        state: newton.State,
    ) -> None:
        # Legacy hook: pendulum rollouts use joint_target_pos, not joint_act torques.
        # Kept for API compatibility with older envs that mapped actions into joint_act.
        return

    def reset(self) -> None:
        # Zero joint coordinates and velocities; run FK to update body_q.
        self.state.joint_q.zero_()
        self.state.joint_qd.zero_()
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

    def reset_scene(
        self,
        plane_normals: torch.Tensor,
        plane_d_coefficients: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update every world's tilted-plane normal and optional d (plane n·x + d = 0).
        The gravity-perpendicular ground plane is left unchanged.
        Only the second ("tilted") plane in each world is rotated and translated
        so that its collision normal matches *plane_normals* and it lies on n·x + d = 0.
        Args:
            plane_normals: (num_worlds, 3) tensor of unit normals, one per world.
            plane_d_coefficients: optional (num_worlds, 1) or (num_worlds,) tensor; plane offset d.
                If None, only rotation is updated (position unchanged).
        """
        if not self.with_contacts:
            raise RuntimeError(
                "reset_scene() called but simulator was built with with_contacts=False."
            )

        assert plane_normals.shape == (self.num_worlds, 3)

        # Ensure inputs are on the same device as model transforms
        plane_normals = plane_normals.to(self._torch_device)
        if plane_d_coefficients is not None:
            plane_d_coefficients = plane_d_coefficients.to(self._torch_device)

        transforms = wp.to_torch(self.model.shape_transform).to(
            self._torch_device
        )

        for world_idx in range(self.num_worlds):
            n = plane_normals[world_idx]
            rot = wp.quat_between_vectors(
                wp.vec3(0.0, 0.0, 1.0),
                wp.vec3(n[0].item(), n[1].item(), n[2].item()),
            )
            shape_idx = self._tilted_plane_shape_indices[world_idx].item()
            transforms[shape_idx, 3:7] = torch.tensor(
                [rot[0], rot[1], rot[2], rot[3]],
                device=transforms.device,
                dtype=transforms.dtype,
            )
            if plane_d_coefficients is not None:
                d = plane_d_coefficients[world_idx].item()
                # Plane n·x + d = 0: a point on the plane is -d*n
                transforms[shape_idx, 0:3] = (-d * n).to(
                    device=transforms.device, dtype=transforms.dtype
                )

        self.model.shape_transform.assign(
            wp.from_torch(transforms, dtype=wp.transform)
        )

    def close(self) -> None:
        # Nothing to clean up explicitly.
        return

    # "Private" methods:
    @staticmethod
    def _load_engine_config(
        engine_group: str = ENGINE_CONFIG_GROUP,
    ) -> AxionEngineConfig:
        """Load and instantiate an engine config via Hydra compose + instantiate."""
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=ENGINE_CONF_DIR, version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[f"engine={engine_group}"],
            )
            engine_cfg = hydra.utils.instantiate(cfg.engine)

        if not isinstance(engine_cfg, AxionEngineConfig):
            raise TypeError(
                f"Engine config group {engine_group!r} instantiated "
                f"{type(engine_cfg).__name__}, expected AxionEngineConfig subclass."
            )
        return engine_cfg

    
    def _engine_init_state_fn(
        self,
        state: newton.State,
        next_state: newton.State,
        contacts: newton.Contacts,
        dt: float,
    ) -> None:
        # Simple integrator hook: rely on AxionEngine to integrate bodies.
        self.engine.integrate_bodies(self.model, state, next_state, dt)

    # Properties:
    @property
    def num_envs(self) -> int:
        return self.num_worlds
    
