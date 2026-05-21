from typing import Any

import warp as wp
from axion.mechanics import integrate_body_pose_kernel

from .engine_data import EngineData
from .engine_dims import EngineDimensions
from .model import AxionModel


@wp.kernel
def _update_kernel(
    dx: wp.array(dtype=Any, ndim=2),
    # Outputs
    x: wp.array(dtype=Any, ndim=2),
):
    world_idx, x_idx = wp.tid()

    x[world_idx, x_idx] = x[world_idx, x_idx] + dx[world_idx, x_idx]


def apply_standard_newton_step(
    model: AxionModel,
    data: EngineData,
    dims: EngineDimensions,
):
    wp.launch(
        kernel=_update_kernel,
        dim=(dims.num_worlds, dims.body_count),
        inputs=[data.dbody_vel],
        outputs=[data.body_vel],
        device=data.device,
    )

    wp.launch(
        kernel=_update_kernel,
        dim=(dims.num_worlds, dims.num_constraints),
        inputs=[data.dconstr_force.full],
        outputs=[data.constr_force.full],
        device=data.device,
    )

    wp.launch(
        kernel=integrate_body_pose_kernel,
        dim=(dims.num_worlds, dims.body_count),
        inputs=[
            data.body_vel,
            data.body_pose_prev,
            model.body_com,
            data.dt,
        ],
        outputs=[data.body_pose],
        device=data.device,
    )
