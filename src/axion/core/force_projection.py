import warp as wp

from axion.constraints.friction_constraint import resolve_friction_frame


@wp.kernel
def project_contact_forces_kernel(
    constr_force_n: wp.array(dtype=wp.float32, ndim=2),
    constr_force_f: wp.array(dtype=wp.float32, ndim=2),
    shape_material_mu: wp.array(dtype=wp.float32, ndim=2),
    shape_friction_axis_local: wp.array(dtype=wp.vec3, ndim=2),
    shape_mu_perp: wp.array(dtype=wp.float32, ndim=2),
    shape_body: wp.array(dtype=wp.int32, ndim=2),
    body_pose_prev: wp.array(dtype=wp.transform, ndim=2),
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    contact_shape0: wp.array(dtype=wp.int32, ndim=2),
    contact_shape1: wp.array(dtype=wp.int32, ndim=2),
    contact_normal: wp.array(dtype=wp.vec3, ndim=2),
):
    """Project contact forces onto the feasible set.

    Enforces:
      - Signorini condition: normal force >= 0
      - Elliptical Coulomb cone: sqrt((f.x/mu_x)^2 + (f.y/mu_y)^2) <= f_n
        (mu_x, mu_y are resolved per-contact via resolve_friction_frame; the
        isotropic case mu_x == mu_y == mu recovers the original circular cone.)
    """
    world_idx, contact_idx = wp.tid()
    friction_idx0 = 2 * contact_idx
    friction_idx1 = 2 * contact_idx + 1

    if contact_idx >= contact_count[world_idx]:
        constr_force_n[world_idx, contact_idx] = 0.0
        constr_force_f[world_idx, friction_idx0] = 0.0
        constr_force_f[world_idx, friction_idx1] = 0.0
        return

    shape0 = contact_shape0[world_idx, contact_idx]
    shape1 = contact_shape1[world_idx, contact_idx]

    if shape0 == shape1:
        constr_force_n[world_idx, contact_idx] = 0.0
        constr_force_f[world_idx, friction_idx0] = 0.0
        constr_force_f[world_idx, friction_idx1] = 0.0
        return

    mu0 = shape_material_mu[world_idx, shape0]
    mu1 = shape_material_mu[world_idx, shape1]

    # Signorini: clamp normal force to non-negative
    force_n = wp.max(constr_force_n[world_idx, contact_idx], 0.0)
    constr_force_n[world_idx, contact_idx] = force_n

    if force_n <= 0.0:
        constr_force_f[world_idx, friction_idx0] = 0.0
        constr_force_f[world_idx, friction_idx1] = 0.0
        return

    # Resolve the per-contact (mu_x, mu_y). The tangent vectors (t1, t2) returned
    # here aren't used directly — the friction force components stored in
    # constr_force_f are already in the same frame that the constraint solver
    # used, so we only need the cone shape (mu_x, mu_y) to project.
    body0 = wp.int32(-1)
    if shape0 >= 0:
        body0 = shape_body[world_idx, shape0]
    body1 = wp.int32(-1)
    if shape1 >= 0:
        body1 = shape_body[world_idx, shape1]

    pose0_prev = wp.transform_identity()
    if body0 >= 0:
        pose0_prev = body_pose_prev[world_idx, body0]
    pose1_prev = wp.transform_identity()
    if body1 >= 0:
        pose1_prev = body_pose_prev[world_idx, body1]

    axis0 = shape_friction_axis_local[world_idx, shape0]
    axis1 = shape_friction_axis_local[world_idx, shape1]
    mu_perp_0 = shape_mu_perp[world_idx, shape0]
    mu_perp_1 = shape_mu_perp[world_idx, shape1]
    n = contact_normal[world_idx, contact_idx]

    _, _, mu_x, mu_y = resolve_friction_frame(
        n, axis0, pose0_prev, body0, axis1, pose1_prev, body1,
        mu0, mu_perp_0, mu1, mu_perp_1,
    )

    force_f = wp.vec2(
        constr_force_f[world_idx, friction_idx0],
        constr_force_f[world_idx, friction_idx1],
    )

    # Elliptical cone: radial projection in tilde-space. Floor mu to keep 1/mu
    # finite; mu=0 directions collapse the cone in that axis and the projection
    # drives the corresponding f component toward zero.
    mu_floor = wp.float32(1e-6)
    inv_mux = 1.0 / wp.max(mu_x, mu_floor)
    inv_muy = 1.0 / wp.max(mu_y, mu_floor)
    f_x_tilde = force_f.x * inv_mux
    f_y_tilde = force_f.y * inv_muy
    e = wp.sqrt(f_x_tilde * f_x_tilde + f_y_tilde * f_y_tilde)
    if e > force_n:
        scale = force_n / e
        constr_force_f[world_idx, friction_idx0] = force_f.x * scale
        constr_force_f[world_idx, friction_idx1] = force_f.y * scale
