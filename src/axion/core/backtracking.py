import warp as wp

from .engine_config import EngineConfig
from .engine_data import EngineData
from .engine_dims import EngineDimensions
from .model import AxionModel
from .reductions import batched_argmin_dynamic_kernel
from .reductions import gather_1d_kernel
from .reductions import gather_2d_kernel


def perform_backtracking(
    model: AxionModel,
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):
    wp.launch(
        kernel=batched_argmin_dynamic_kernel,
        dim=(dims.num_worlds),
        inputs=[
            data.candidates_res_norm_sq,
            config.nr.backtrack_min_iter,
            data.iter_count,
        ],
        outputs=[
            data.candidates_best_idx,
        ],
        device=data.device,
    )

    wp.launch(
        kernel=gather_2d_kernel,
        dim=(dims.num_worlds, dims.body_count),
        inputs=[
            data.candidates_body_pose,
            data.candidates_best_idx,
        ],
        outputs=[
            data.body_pose,
        ],
        device=data.device,
    )

    wp.launch(
        kernel=gather_2d_kernel,
        dim=(dims.num_worlds, dims.body_count),
        inputs=[
            data.candidates_body_vel,
            data.candidates_best_idx,
        ],
        outputs=[
            data.body_vel,
        ],
        device=data.device,
    )

    wp.launch(
        kernel=gather_2d_kernel,
        dim=(dims.num_worlds, dims.num_constraints),
        inputs=[
            data._candidates_constr_force,
            data.candidates_best_idx,
        ],
        outputs=[
            data._constr_force,
        ],
        device=data.device,
    )

    wp.launch(
        kernel=gather_2d_kernel,
        dim=(dims.num_worlds, dims.num_constraints),
        inputs=[
            data._candidates_res,
            data.candidates_best_idx,
        ],
        outputs=[
            data._res,
        ],
        device=data.device,
    )

    wp.launch(
        kernel=gather_1d_kernel,
        dim=(dims.num_worlds),
        inputs=[
            data.candidates_res_norm_sq,
            data.candidates_best_idx,
        ],
        outputs=[
            data.res_norm_sq,
        ],
        device=data.device,
    )
