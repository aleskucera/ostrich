import warp as wp
from axion.constraints import contact_constraint_kernel
from axion.constraints import control_constraint_kernel
from axion.constraints import friction_constraint_kernel
from axion.constraints import joint_constraint_kernel
from axion.constraints import unconstrained_dynamics_kernel
from axion.constraints.utils import compute_spatial_momentum
from axion.constraints.utils import compute_world_inertia

from .contacts import AxionContacts
from .engine_config import EngineConfig
from .engine_data import EngineData
from .engine_dims import EngineDimensions
from .model import AxionModel


@wp.kernel
def compute_schur_complement_rhs_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    res_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    res_c: wp.array(dtype=wp.float32, ndim=2),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    dt: wp.float32,
    # Output array
    rhs: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, constraint_idx = wp.tid()

    is_active = constraint_active_mask[world_idx, constraint_idx]
    if is_active == 0.0:
        rhs[world_idx, constraint_idx] = 0.0
        return

    body_1 = constraint_body_idx[world_idx, constraint_idx, 0]
    body_2 = constraint_body_idx[world_idx, constraint_idx, 1]

    JHinvg = 0.0
    if body_1 >= 0:
        q_1 = body_pose[world_idx, body_1]
        m_inv_1 = body_m_inv[world_idx, body_1]
        I_inv_b_1 = body_I_inv[world_idx, body_1]
        I_inv_w_1 = compute_world_inertia(q_1, I_inv_b_1)

        J_1 = J_values[world_idx, constraint_idx, 0]
        JHinvg += wp.dot(
            J_1, compute_spatial_momentum(m_inv_1, I_inv_w_1, res_d[world_idx, body_1])
        )

    if body_2 >= 0:
        q_2 = body_pose[world_idx, body_2]
        m_inv_2 = body_m_inv[world_idx, body_2]
        I_inv_b_2 = body_I_inv[world_idx, body_2]
        I_inv_w_2 = compute_world_inertia(q_2, I_inv_b_2)

        J_2 = J_values[world_idx, constraint_idx, 1]
        JHinvg += wp.dot(
            J_2, compute_spatial_momentum(m_inv_2, I_inv_w_2, res_d[world_idx, body_2])
        )

    # b = (J * M^-1 * h_d - h_c) / dt
    rhs[world_idx, constraint_idx] = (JHinvg - res_c[world_idx, constraint_idx]) / dt


@wp.kernel
def compute_JT_dconstr_force_kernel(
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    dconstr_force: wp.array(dtype=wp.float32, ndim=2),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
    # Output array
    JT_dconstr_force: wp.array(dtype=wp.spatial_vector, ndim=2),
):
    world_idx, constraint_idx = wp.tid()

    body_1 = constraint_body_idx[world_idx, constraint_idx, 0]
    body_2 = constraint_body_idx[world_idx, constraint_idx, 1]

    J_1 = J_values[world_idx, constraint_idx, 0]
    J_2 = J_values[world_idx, constraint_idx, 1]
    dforce = dconstr_force[world_idx, constraint_idx]

    if body_1 >= 0:
        wp.atomic_add(JT_dconstr_force, world_idx, body_1, dforce * J_1)

    if body_2 >= 0:
        wp.atomic_add(JT_dconstr_force, world_idx, body_2, dforce * J_2)


@wp.kernel
def compute_dbody_vel_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    JT_dconstr_force: wp.array(dtype=wp.spatial_vector, ndim=2),
    res_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    dt: wp.float32,
    # Output array
    dbody_vel: wp.array(dtype=wp.spatial_vector, ndim=2),
):
    world_idx, body_idx = wp.tid()

    if body_idx >= body_pose.shape[1]:
        return

    pose = body_pose[world_idx, body_idx]
    m_inv = body_m_inv[world_idx, body_idx]
    I_inv_b = body_I_inv[world_idx, body_idx]
    I_inv_w = compute_world_inertia(pose, I_inv_b)

    dbody_vel[world_idx, body_idx] = compute_spatial_momentum(
        m_inv,
        I_inv_w,
        (JT_dconstr_force[world_idx, body_idx] * dt - res_d[world_idx, body_idx]),
    )


@wp.kernel
def compute_world_inv_inertia_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    # Outputs
    world_I_inv: wp.array(dtype=wp.mat33, ndim=2),
):
    world_idx, body_idx = wp.tid()

    # Boundary check (if needed, depending on how you launch)
    if body_idx >= body_pose.shape[1]:
        return

    pose = body_pose[world_idx, body_idx]
    I_inv_b = body_I_inv[world_idx, body_idx]

    # compute_world_inertia expects (transform, mat33) and returns mat33
    world_I_inv[world_idx, body_idx] = compute_world_inertia(pose, I_inv_b)


def compute_linear_system(
    model: AxionModel,
    contacts: AxionContacts,
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):

    wp.launch(
        kernel=compute_world_inv_inertia_kernel,
        dim=(model.num_worlds, model.body_count),
        inputs=[
            data.body_pose,
            model.body_inv_inertia,
        ],
        outputs=[
            data.world_inv_inertia,
        ],
        device=data.device,
    )

    wp.launch(
        kernel=unconstrained_dynamics_kernel,
        dim=(dims.N_w, dims.N_b),
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
        kernel=joint_constraint_kernel,
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
            data.constr_active_mask.j,
            data.res.d_spatial,
            data.res.c.j,
            data.J_values.j,
            data.C_values.j,
        ],
        device=data.device,
    )

    wp.launch(
        kernel=control_constraint_kernel,
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
            data.constr_active_mask.ctrl,
            data.res.d_spatial,
            data.res.c.ctrl,
            data.J_values.ctrl,
            data.C_values.ctrl,
        ],
        device=data.device,
    )

    wp.launch(
        kernel=contact_constraint_kernel,
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
            data.constr_active_mask.n,
            data.constr_body_idx.n,
            data.res.d_spatial,
            data.res.c.n,
            data.J_values.n,
            data.C_values.n,
        ],
        device=data.device,
    )

    wp.launch(
        kernel=friction_constraint_kernel,
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
            data.constr_active_mask.f,
            data.constr_body_idx.f,
            data.res.d_spatial,
            data.res.c.f,
            data.J_values.f,
            data.C_values.f,
        ],
        device=data.device,
    )

    data.res.sync_to_float()

    wp.launch(
        kernel=compute_schur_complement_rhs_kernel,
        dim=(dims.num_worlds, dims.num_constraints),
        inputs=[
            data.body_pose,
            model.body_inv_mass,
            model.body_inv_inertia,
            data.J_values.full,
            data.res.d_spatial,
            data.res.c.full,
            data.constr_body_idx.full,
            data.constr_active_mask.full,
            data.dt,
        ],
        outputs=[data.rhs],
        device=data.device,
    )


def compute_dbody_qd_from_dbody_lambda(
    model: AxionModel,
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):

    data.JT_dconstr_force.zero_()
    wp.launch(
        kernel=compute_JT_dconstr_force_kernel,
        dim=(dims.num_worlds, dims.num_constraints),
        inputs=[
            data.J_values.full,
            data.dconstr_force.full,
            data.constr_body_idx.full,
        ],
        outputs=[data.JT_dconstr_force],
        device=data.device,
    )

    wp.launch(
        kernel=compute_dbody_vel_kernel,
        dim=(dims.num_worlds, dims.body_count),
        inputs=[
            data.body_pose,
            model.body_inv_mass,
            model.body_inv_inertia,
            data.JT_dconstr_force,
            data.res.d_spatial,
            data.dt,
        ],
        outputs=[data.dbody_vel],
        device=data.device,
    )
