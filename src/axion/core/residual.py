import torch
import torch.nn as nn
import warp as wp
from axion.constraints import contact_residual_kernel
from axion.constraints import control_residual_kernel
from axion.constraints import friction_residual_kernel
from axion.constraints import joint_residual_kernel
from axion.constraints import unconstrained_dynamics_kernel
from axion.core.engine_config import EngineConfig
from axion.core.engine_data import EngineData
from axion.core.engine_dims import EngineDimensions
from axion.core.model import AxionModel
from axion.mechanics import compute_spatial_momentum
from axion.mechanics import compute_world_inertia

from .contacts import AxionContacts
from .types import JointMode


def compute_residual(
    model: AxionModel,
    contacts: AxionContacts,
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):
    wp.launch(
        kernel=unconstrained_dynamics_kernel,
        dim=(dims.num_worlds, dims.body_count),
        inputs=[
            data.body_pose,
            data.body_vel,
            data.body_vel_prev,
            data.ext_force,
            model.body_mass,
            model.body_inertia,
            data.dt,
            model.g_accel,
        ],
        outputs=[data.res.d_spatial],
        device=data.device,
    )

    wp.launch(
        kernel=joint_residual_kernel,
        dim=(dims.num_worlds, dims.joint_count),
        inputs=[
            data.body_pose,
            data.constr_force.j,
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
            model.joint_compliance,
            data.dt,
            config.compliance.joint,
        ],
        outputs=[
            data.res.d_spatial,
            data.res.c.j,
        ],
        device=data.device,
    )

    wp.launch(
        kernel=control_residual_kernel,
        dim=(dims.num_worlds, dims.joint_count),
        inputs=[
            data.body_pose,
            data.body_vel,
            data.constr_force.ctrl,
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
        ],
        outputs=[
            data.res.d_spatial,
            data.res.c.ctrl,
        ],
        device=data.device,
    )

    wp.launch(
        kernel=contact_residual_kernel,
        dim=(dims.num_worlds, dims.contact_count),
        inputs=[
            data.body_pose,
            data.body_vel,
            data.body_vel_prev,
            data.body_pose_prev,
            data.constr_force.n,
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
        ],
        outputs=[
            data.res.d_spatial,
            data.res.c.n,
        ],
        device=data.device,
    )

    wp.launch(
        kernel=friction_residual_kernel,
        dim=(dims.num_worlds, dims.contact_count),
        inputs=[
            data.body_vel,
            data.body_pose_prev,
            data.constr_force.f,
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
        ],
        outputs=[
            data.res.d_spatial,
            data.res.c.f,
        ],
        device=data.device,
    )

    data.res.sync_to_float()


@wp.kernel
def body_vel_prev_grad_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=2),
    body_mass: wp.array(dtype=wp.float32, ndim=2),
    body_inertia: wp.array(dtype=wp.mat33, ndim=2),
    adjoint_vector: wp.array(dtype=wp.spatial_vector, ndim=2),
    # --- Output ---
    body_vel_prev_grad: wp.array(dtype=wp.spatial_vector, ndim=2),
):
    world_idx, body_idx = wp.tid()
    if body_idx >= body_pose.shape[1]:
        return

    pose = body_pose[world_idx, body_idx]
    m = body_mass[world_idx, body_idx]
    I_body = body_inertia[world_idx, body_idx]

    # Compute World Inertia
    I_world = compute_world_inertia(pose, I_body)

    w = adjoint_vector[world_idx, body_idx]

    body_vel_prev_grad[world_idx, body_idx] += compute_spatial_momentum(-m, -I_world, w)


@wp.kernel
def control_target_grad_kernel(
    joint_type: wp.array(dtype=wp.int32, ndim=2),
    joint_qd_start: wp.array(dtype=wp.int32, ndim=2),
    joint_enabled: wp.array(dtype=wp.bool, ndim=2),
    joint_dof_mode: wp.array(dtype=wp.int32, ndim=2),
    control_constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    w_ctrl: wp.array(dtype=wp.float32, ndim=2),
    lambda_ctrl: wp.array(dtype=wp.float32, ndim=2),
    joint_target_ke: wp.array(dtype=wp.float32, ndim=2),
    joint_target_kd: wp.array(dtype=wp.float32, ndim=2),
    dt: wp.float32,
    # Outputs:
    target_pos_grad: wp.array(dtype=wp.float32, ndim=2),
    target_vel_grad: wp.array(dtype=wp.float32, ndim=2),
    target_ke_grad: wp.array(dtype=wp.float32, ndim=2),
    target_kd_grad: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, joint_idx = wp.tid()

    if joint_enabled[world_idx, joint_idx] == 0:
        return

    j_type = joint_type[world_idx, joint_idx]
    if j_type != 1 and j_type != 0:
        return

    qd_start_idx = joint_qd_start[world_idx, joint_idx]
    mode = joint_dof_mode[world_idx, qd_start_idx]
    if mode == 0:
        return

    ctrl_offset = control_constraint_offsets[world_idx, joint_idx]

    w_lambda = w_ctrl[world_idx, ctrl_offset]
    lam = lambda_ctrl[world_idx, ctrl_offset]
    ke = joint_target_ke[world_idx, qd_start_idx]
    kd = joint_target_kd[world_idx, qd_start_idx]

    if mode == JointMode.TARGET_POSITION:
        # R_c = (q - target)/h + α·λ·h,  α = 1/(h²·ke + h·kd)
        wp.atomic_add(target_pos_grad, world_idx, qd_start_idx, w_lambda * (-1.0 / dt))

        # dα/d(ke) = -h²/(h²·ke + h·kd)²,  dα/d(kd) = -h/(h²·ke + h·kd)²
        # dR_c/d(ke) = dα/d(ke) · λ · h,    dR_c/d(kd) = dα/d(kd) · λ · h
        denom = dt * dt * ke + dt * kd
        if denom > 1e-6:
            inv_denom_sq = 1.0 / (denom * denom)
            wp.atomic_add(target_ke_grad, world_idx, qd_start_idx,
                          w_lambda * (-dt * dt * inv_denom_sq) * lam * dt)
            wp.atomic_add(target_kd_grad, world_idx, qd_start_idx,
                          w_lambda * (-dt * inv_denom_sq) * lam * dt)

    elif mode == JointMode.TARGET_VELOCITY:
        # R_c = (qd - target_vel) + α·λ·h,  α = 1/(h·ke)
        wp.atomic_add(target_vel_grad, world_idx, qd_start_idx, w_lambda * (-1.0))

        # dα/d(ke) = -h/(h·ke)² = -1/(ke²·h)
        denom = dt * ke
        if denom > 1e-6:
            inv_denom_sq = 1.0 / (denom * denom)
            wp.atomic_add(target_ke_grad, world_idx, qd_start_idx,
                          w_lambda * (-dt * inv_denom_sq) * lam * dt)


@wp.kernel
def body_pose_prev_grad_kernel(
    body_pose_grad: wp.array(dtype=wp.transform, ndim=2),
    body_vel: wp.array(dtype=wp.spatial_vector, ndim=2),
    dt: wp.float32,
    # --- Output ---
    body_pose_prev_grad: wp.array(dtype=wp.transform, ndim=2),
):
    world_idx, body_idx = wp.tid()
    if body_idx >= body_pose_grad.shape[1]:
        return

    # Kinematic propagation: q+ = q- + dt * G(q-) * u
    # Full derivative: dq+/dq- = I + dt * (dG/dq-) * u
    #
    # Translation part: p+ = p- + dt*v, so dp+/dp- = I (exact, no correction needed)
    #
    # Rotation part: r+ = r- + (dt/2) * Q(r-) * w
    # dr+/dr- = I + (dt/2) * (dQ/dr-) * w
    # The correction term is [dQ(r)*w/dr]^T * grad_r

    g = body_pose_grad[world_idx, body_idx]
    grad_p = wp.transform_get_translation(g)
    grad_r = wp.transform_get_rotation(g)

    vel = body_vel[world_idx, body_idx]
    w1 = wp.spatial_bottom(vel)[0]  # angular x
    w2 = wp.spatial_bottom(vel)[1]  # angular y
    w3 = wp.spatial_bottom(vel)[2]  # angular z

    s = dt * 0.5

    # grad_r components (x=0, y=1, z=2, w=3 in quat storage)
    gr_x = grad_r[0]
    gr_y = grad_r[1]
    gr_z = grad_r[2]
    gr_w = grad_r[3]

    # [dQ(r)*w/dr]^T * grad_r
    # Q*w rows (from G_matvec):
    #   d_theta_x = theta_w*w1 + theta_z*w2 - theta_y*w3
    #   d_theta_y = -theta_z*w1 + theta_w*w2 + theta_x*w3
    #   d_theta_z = theta_y*w1 - theta_x*w2 + theta_w*w3
    #   d_theta_w = -theta_x*w1 - theta_y*w2 - theta_z*w3
    #
    # Differentiating each row w.r.t. each theta component and transposing:
    # d/d(theta_x): row_x gives 0, row_y gives w3, row_z gives -w2, row_w gives -w1
    #   correction_x = s * (0*gr_x + w3*gr_y + (-w2)*gr_z + (-w1)*gr_w)
    # d/d(theta_y): row_x gives -w3, row_y gives 0, row_z gives w1, row_w gives -w2
    #   correction_y = s * ((-w3)*gr_x + 0*gr_y + w1*gr_z + (-w2)*gr_w)
    # d/d(theta_z): row_x gives w2, row_y gives -w1, row_z gives 0, row_w gives -w3
    #   correction_z = s * (w2*gr_x + (-w1)*gr_y + 0*gr_z + (-w3)*gr_w)
    # d/d(theta_w): row_x gives w1, row_y gives w2, row_z gives w3, row_w gives 0
    #   correction_w = s * (w1*gr_x + w2*gr_y + w3*gr_z + 0*gr_w)

    corr_x = s * (w3 * gr_y - w2 * gr_z - w1 * gr_w)
    corr_y = s * (-w3 * gr_x + w1 * gr_z - w2 * gr_w)
    corr_z = s * (w2 * gr_x - w1 * gr_y - w3 * gr_w)
    corr_w = s * (w1 * gr_x + w2 * gr_y + w3 * gr_z)

    # Total: identity + correction
    out_r = wp.quat(
        grad_r[0] + corr_x,
        grad_r[1] + corr_y,
        grad_r[2] + corr_z,
        grad_r[3] + corr_w,
    )

    wp.atomic_add(
        body_pose_prev_grad,
        world_idx,
        body_idx,
        wp.transform(grad_p, out_r),
    )


def compute_residual_gradient(
    model: AxionModel,
    contacts: AxionContacts,
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):

    wp.launch(
        kernel=body_vel_prev_grad_kernel,
        dim=(dims.num_worlds, dims.body_count),
        inputs=[
            data.body_pose,
            model.body_mass,
            model.body_inertia,
            data.w.d_spatial,
        ],
        outputs=[data.body_vel_prev.grad],
        device=data.device,
    )

    wp.launch(
        kernel=body_pose_prev_grad_kernel,
        dim=(dims.num_worlds, dims.body_count),
        inputs=[
            data.body_pose_grad,
            data.body_vel,
            data.dt,
        ],
        outputs=[data.body_pose_prev.grad],
        device=data.device,
    )

    wp.launch(
        kernel=control_target_grad_kernel,
        dim=(dims.num_worlds, dims.joint_count),
        inputs=[
            model.joint_type,
            model.joint_qd_start,
            model.joint_enabled,
            model.joint_dof_mode,
            model.control_constraint_offsets,
            data.w.c.ctrl,
            data.constr_force.ctrl,
            model.joint_target_ke,
            model.joint_target_kd,
            data.dt,
        ],
        outputs=[
            data.joint_target_pos.grad,
            data.joint_target_vel.grad,
            model.joint_target_ke.grad,
            model.joint_target_kd.grad,
        ],
        device=data.device,
    )
