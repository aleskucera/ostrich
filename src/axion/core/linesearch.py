from typing import Any

import warp as wp
from axion.constraints import batch_contact_residual_kernel
from axion.constraints import batch_control_residual_kernel
from axion.constraints import batch_friction_residual_kernel
from axion.constraints import batch_joint_residual_kernel
from axion.constraints import batch_unconstrained_dynamics_kernel
from axion.constraints import fused_batch_contact_residual_kernel
from axion.constraints import fused_batch_control_residual_kernel
from axion.constraints import fused_batch_friction_residual_kernel
from axion.constraints import fused_batch_joint_residual_kernel
from axion.constraints import fused_batch_unconstrained_dynamics_kernel
from axion.mechanics import integrate_batched_body_pose_kernel

from .contacts import AxionContacts
from .engine_config import EngineConfig
from .engine_data import EngineData
from .engine_dims import EngineDimensions
from .model import AxionModel
from .reductions import batched_argmin_kernel
from .reductions import gather_1d_kernel
from .reductions import gather_2d_kernel


@wp.kernel
def linesearch_spread_kernel(
    x: wp.array(dtype=Any, ndim=2),
    dx: wp.array(dtype=Any, ndim=2),
    linesearch_steps: wp.array(dtype=wp.float32),
    # Outputs
    linesearch_x: wp.array(dtype=Any, ndim=3),
):
    batch_idx, world_idx, x_idx = wp.tid()

    spread_x = x[world_idx, x_idx] + linesearch_steps[batch_idx] * dx[world_idx, x_idx]
    linesearch_x[batch_idx, world_idx, x_idx] = spread_x


def _linesearch_spread(
    model: AxionModel,
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):
    wp.launch(
        kernel=linesearch_spread_kernel,
        dim=(
            dims.linesearch_step_count,
            dims.num_worlds,
            dims.body_count,
        ),
        inputs=[
            data.body_vel,
            data.dbody_vel,
            data.linesearch_step_size,
        ],
        outputs=[data.linesearch_body_vel],
        device=data.device,
    )

    wp.launch(
        kernel=integrate_batched_body_pose_kernel,
        dim=(
            dims.linesearch_step_count,
            dims.num_worlds,
            dims.body_count,
        ),
        inputs=[
            data.linesearch_body_vel,
            data.body_pose_prev,
            model.body_com,
            data.dt,
        ],
        outputs=[data.linesearch_body_pose],
        device=data.device,
    )

    wp.launch(
        kernel=linesearch_spread_kernel,
        dim=(
            dims.linesearch_step_count,
            dims.num_worlds,
            dims.num_constraints,
        ),
        inputs=[
            data.constr_force.full,
            data.dconstr_force.full,
            data.linesearch_step_size,
        ],
        outputs=[data.linesearch_constr_force.full],
        device=data.device,
    )


def _compute_batched_residual(
    model: AxionModel,
    contacts: AxionContacts,
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):
    wp.launch(
        kernel=fused_batch_unconstrained_dynamics_kernel,
        dim=(dims.num_worlds, dims.body_count),
        inputs=[
            data.linesearch_body_pose,
            data.linesearch_body_vel,
            data.body_vel_prev,
            data.ext_force,
            model.body_mass,
            model.body_inertia,
            data.dt,
            model.g_accel,
            dims.linesearch_step_count,
        ],
        outputs=[data.linesearch_res.d_spatial],
        device=data.device,
    )

    wp.launch(
        kernel=fused_batch_joint_residual_kernel,
        dim=(dims.num_worlds, dims.joint_count),
        inputs=[
            data.linesearch_body_pose,
            data.linesearch_constr_force.j,
            model.body_com,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_axis,
            model.joint_qd_start,
            model.joint_enabled,
            model.joint_constraint_offsets,
            data.dt,
            config.compliance.joint,
            dims.linesearch_step_count,
        ],
        outputs=[
            data.linesearch_res.d_spatial,
            data.linesearch_res.c.j,
        ],
        device=data.device,
    )

    wp.launch(
        kernel=fused_batch_control_residual_kernel,
        dim=(dims.num_worlds, dims.joint_count),
        inputs=[
            data.linesearch_body_pose,
            data.linesearch_body_vel,
            data.linesearch_constr_force.ctrl,
            model.body_com,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_axis,
            model.joint_qd_start,
            model.joint_enabled,
            model.joint_dof_mode,
            model.control_constraint_offsets,
            data.joint_target_pos,
            data.joint_target_vel,
            model.joint_target_ke,
            model.joint_target_kd,
            data.dt,
            dims.linesearch_step_count,
        ],
        outputs=[
            data.linesearch_res.d_spatial,
            data.linesearch_res.c.ctrl,
        ],
        device=data.device,
    )

    wp.launch(
        kernel=fused_batch_contact_residual_kernel,
        dim=(dims.num_worlds, dims.contact_count),
        inputs=[
            data.linesearch_body_pose,
            data.linesearch_body_vel,
            data.body_vel_prev,
            data.body_pose_prev,
            data.linesearch_constr_force.n,
            model.body_com,
            model.body_inv_mass,
            model.body_inv_inertia,
            model.shape_body,
            contacts.contact_count,
            contacts.contact_shape0,
            contacts.contact_shape1,
            contacts.contact_point0,
            contacts.contact_point1,
            contacts.contact_thickness0,
            contacts.contact_thickness1,
            contacts.contact_normal,
            data.dt,
            config.compliance.contact,
            config.compliance.contact_fb_smooth_eps_sq,
            dims.linesearch_step_count,
        ],
        outputs=[
            data.linesearch_res.d_spatial,
            data.linesearch_res.c.n,
        ],
        device=data.device,
    )

    wp.launch(
        kernel=fused_batch_friction_residual_kernel,
        dim=(dims.num_worlds, dims.contact_count),
        inputs=[
            data.linesearch_body_vel,
            data.body_pose_prev,
            data.linesearch_constr_force.f,
            data.constr_force_prev_iter.f,
            data.constr_force_prev_iter.n,
            model.body_com,
            model.body_inv_mass,
            model.body_inv_inertia,
            model.shape_body,
            model.shape_material_mu,
            model.shape_friction_axis_local,
            model.shape_mu_perp,
            contacts.contact_count,
            contacts.contact_shape0,
            contacts.contact_shape1,
            contacts.contact_point0,
            contacts.contact_point1,
            contacts.contact_thickness0,
            contacts.contact_thickness1,
            contacts.contact_normal,
            data.dt,
            config.compliance.friction,
            dims.linesearch_step_count,
        ],
        outputs=[
            data.linesearch_res.d_spatial,
            data.linesearch_res.c.f,
        ],
        device=data.device,
    )

    # FIX: Find out why is this bad?
    data.linesearch_res.sync_to_float()


def _linesearch_gather(
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):
    data.linesearch_tiled_res_sq_norm.compute(data.linesearch_res.full, data.linesearch_res_norm_sq)

    wp.launch(
        kernel=batched_argmin_kernel,
        dim=dims.num_worlds,
        inputs=[data.linesearch_res_norm_sq],
        outputs=[data.linesearch_minimal_index],
        device=data.device,
    )

    wp.launch(
        kernel=gather_2d_kernel,
        dim=(dims.num_worlds, dims.body_count),
        inputs=[
            data.linesearch_body_vel,
            data.linesearch_minimal_index,
        ],
        outputs=[data.body_vel],
        device=data.device,
    )

    wp.launch(
        kernel=gather_2d_kernel,
        dim=(dims.num_worlds, dims.body_count),
        inputs=[
            data.linesearch_body_pose,
            data.linesearch_minimal_index,
        ],
        outputs=[data.body_pose],
        device=data.device,
    )

    wp.launch(
        kernel=gather_2d_kernel,
        dim=(dims.num_worlds, dims.num_constraints),
        inputs=[
            data.linesearch_constr_force.full,
            data.linesearch_minimal_index,
        ],
        outputs=[data.constr_force.full],
        device=data.device,
    )

    wp.launch(
        kernel=gather_2d_kernel,
        dim=(dims.num_worlds, dims.N_u + dims.num_constraints),
        inputs=[
            data.linesearch_res.full,
            data.linesearch_minimal_index,
        ],
        outputs=[data.res.full],
        device=data.device,
    )

    wp.launch(
        kernel=gather_1d_kernel,
        dim=(dims.num_worlds),
        inputs=[
            data.linesearch_res_norm_sq,
            data.linesearch_minimal_index,
        ],
        outputs=[data.res_norm_sq],
        device=data.device,
    )


def perform_linesearch(
    model: AxionModel,
    contacts: AxionContacts,
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):
    _linesearch_spread(model, data, config, dims)
    _compute_batched_residual(model, contacts, data, config, dims)
    _linesearch_gather(data, config, dims)
