"""Rigid-body mechanics primitives — kinematics, dynamics, and contact-formulation math.

Holds the building blocks the engine composes into constraint kernels:
pose integration and kinematic Jacobian-vector products (kinematics);
spatial inertia, world-frame inertia, and momentum (dynamics); the
contact-frame basis (geometry); and the Fisher-Burmeister complementarity
function used to formulate non-smooth contact (contact mechanics).
"""
from .complementarity import scaled_fisher_burmeister
from .complementarity import scaled_fisher_burmeister_diff
from .geometry import orthogonal_basis
from .kinematic_mapping import G_matvec
from .kinematic_mapping import Gt_matvec
from .kinematic_mapping import mul_G
from .kinematic_mapping import mul_Gt
from .mass import SpatialInertia
from .mass import compute_spatial_momentum
from .mass import compute_world_inertia
from .mass import spatial_inertia_kernel
from .mass import world_spatial_inertia_kernel
from .pose_integration import integrate_batched_body_pose_kernel
from .pose_integration import integrate_body_pose_kernel

__all__ = [
    "orthogonal_basis",
    "scaled_fisher_burmeister",
    "scaled_fisher_burmeister_diff",
    "G_matvec",
    "Gt_matvec",
    "mul_G",
    "mul_Gt",
    "SpatialInertia",
    "compute_spatial_momentum",
    "compute_world_inertia",
    "spatial_inertia_kernel",
    "world_spatial_inertia_kernel",
    "integrate_body_pose_kernel",
    "integrate_batched_body_pose_kernel",
]
