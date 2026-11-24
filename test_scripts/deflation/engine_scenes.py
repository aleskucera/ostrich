"""Scenes for the in-engine DPCR experiments (engine_dpcr.py).

    terrain   Helhest driving over the surface.obj mesh (dump_linear_systems.py),
              k = 18 joint/control rows
    chain     12-link chain, masses alternating by 100, no contacts
              (adjoint_articulated.py), k = 72
    marv      MARV (21 bodies, 4 flippers + 16 wheels, position control) driving
              into the boulder field of examples/marv/control.py, k = 120

Each scene gives a model builder (with num_worlds), the step size, the number of
steps, the control field and per-step targets, and the control DOFs whose
gradients the adjoint experiments look at.
"""

import random
from dataclasses import dataclass
from typing import Callable

import newton
import numpy as np
import warp as wp

from examples.marv.common import create_marv_model
from ostrich.core.model_builder import OstrichModelBuilder
from test_scripts.deflation.adjoint_articulated import build_chain, targets_for
from test_scripts.dump_linear_systems import build_helhest_surface_model

CHAIN_LINKS, CHAIN_MASS_RATIO = 12, 100.0
MARV_WHEEL_SPEED = -8.0                      # rad/s, integrated into wheel position targets
MARV_FLIPPER_DOFS = (6, 11, 16, 21)          # after the 6 DOFs of the chassis free joint
MARV_WHEEL_DOFS = tuple(range(7, 11)) + tuple(range(12, 16)) + tuple(range(17, 21)) + tuple(range(22, 26))


@dataclass(frozen=True)
class Scene:
    name: str
    label: str
    build: Callable[[int], newton.Model]     # num_worlds -> model
    dt: float
    steps: int
    control_field: str                       # "joint_target_vel" | "joint_target_pos"
    targets: Callable[[int, int], np.ndarray]  # (step, num_dofs) -> targets of one world
    grad_dofs: tuple                         # control DOFs whose gradients are compared


def _terrain_targets(step, num_dofs):
    t = np.zeros(num_dofs, dtype=np.float32)
    t[-3:] = 6.0                             # the three wheels, as in dump_linear_systems.py
    return t


def _chain_targets(step, num_dofs):
    return targets_for(None, CHAIN_LINKS)


def _marv_targets(dt):
    def targets(step, num_dofs):
        t = np.zeros(num_dofs, dtype=np.float32)
        t[list(MARV_WHEEL_DOFS)] = MARV_WHEEL_SPEED * dt * (step + 1)
        return t
    return targets


def build_marv_model(num_worlds=1):
    """MARV in front of the boulder field and stairs of examples/marv/control.py (build_model)."""
    builder = OstrichModelBuilder()
    builder.rigid_gap = 0.2
    create_marv_model(builder, xform=wp.transform((2.0, 0.0, 0.5), wp.quat_identity()), is_visible=False)
    ground_cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e4, kd=1.0e3, kf=1.0e3, mu=1.0)
    builder.add_ground_plane(cfg=ground_cfg)
    step_depth, step_height, step_width, start_x, start_y = 0.6, 0.08, 4.0, 5.0, -4.0
    for i in range(6):
        h = (i + 1) * step_height
        builder.add_shape_box(-1, wp.transform(wp.vec3(start_x + i * step_depth, start_y, h / 2.0), wp.quat_identity()),
                              hx=step_depth / 2.0, hy=step_width / 2.0, hz=h / 2.0, cfg=ground_cfg)
    rng = random.Random(42)
    for _ in range(30):
        rx, ry = rng.uniform(3.0, 12.0), rng.uniform(-2.0, 2.0)
        builder.add_shape_box(
            -1, wp.transform(wp.vec3(rx, ry, 0.05), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), rng.uniform(0, 3.14))),
            hx=rng.uniform(0.15, 0.5), hy=rng.uniform(0.15, 0.5), hz=rng.uniform(0.05, 0.15), cfg=ground_cfg)
    return builder.finalize_replicated(num_worlds=num_worlds, gravity=-9.81, requires_grad=True)


SCENES = {
    "terrain": Scene("terrain", "Helhest, terrain", lambda w: build_helhest_surface_model(num_worlds=w),
                     dt=0.03, steps=25, control_field="joint_target_vel", targets=_terrain_targets,
                     grad_dofs=(-3, -2, -1)),
    "chain": Scene("chain", "12-link chain", lambda w: build_chain(CHAIN_LINKS, CHAIN_MASS_RATIO, num_worlds=w),
                   dt=0.01, steps=40, control_field="joint_target_vel", targets=_chain_targets,
                   grad_dofs=tuple(range(CHAIN_LINKS))),
    "marv": Scene("marv", "MARV, boulders", build_marv_model,
                  dt=0.01, steps=150, control_field="joint_target_pos", targets=_marv_targets(0.01),
                  grad_dofs=MARV_FLIPPER_DOFS + MARV_WHEEL_DOFS),
}
