from dataclasses import dataclass

import warp as wp
from axion.mechanics import compute_spatial_momentum
from warp.optim.linear import LinearOperator


@dataclass
class SystemLinearData:
    N_w: int
    N_c: int
    N_b: int
    J_values: wp.array
    constraint_body_idx: wp.array
    constraint_active_mask: wp.array
    C_values: wp.array

    body_inv_mass: wp.array
    world_inv_inertia: wp.array

    @classmethod
    def from_engine(cls, engine):
        return cls(
            N_w=engine.dims.N_w,
            N_c=engine.dims.N_c,
            N_b=engine.dims.N_b,
            J_values=engine.data.J_values.full,
            constraint_body_idx=engine.data.constr_body_idx.full,
            constraint_active_mask=engine.data.constr_active_mask.full,
            C_values=engine.data.C_values.full,
            body_inv_mass=engine.axion_model.body_inv_mass,
            world_inv_inertia=engine.data.world_inv_inertia,
        )


@wp.kernel
def kernel_invM_Jt_matvec_direct(
    x: wp.array(dtype=wp.float32, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    body_inv_mass: wp.array(dtype=wp.float32, ndim=2),
    world_inv_inertia: wp.array(dtype=wp.mat33, ndim=2),
    # Output: Direct accumulation into body velocity vector
    out_dyn_vec: wp.array(dtype=wp.spatial_vector, ndim=2),
):
    world_idx, constraint_idx = wp.tid()

    is_active = constraint_active_mask[world_idx, constraint_idx]
    if is_active == 0.0:
        return

    body_1 = constraint_body_idx[world_idx, constraint_idx, 0]
    body_2 = constraint_body_idx[world_idx, constraint_idx, 1]

    J_1 = J_values[world_idx, constraint_idx, 0]
    J_2 = J_values[world_idx, constraint_idx, 1]
    x_i = x[world_idx, constraint_idx]

    # J^T * x -> Impulse
    # M^-1 * Impulse -> Delta Velocity

    # We use atomic_add here. Since N_b is small (~10), 'out_dyn_vec'
    # fits in L1 cache, making these atomics extremely fast.
    if body_1 >= 0:
        jt_x_1 = x_i * J_1
        m_inv_1 = body_inv_mass[world_idx, body_1]
        I_inv_1 = world_inv_inertia[world_idx, body_1]
        delta_v_1 = compute_spatial_momentum(m_inv_1, I_inv_1, jt_x_1)
        wp.atomic_add(out_dyn_vec, world_idx, body_1, delta_v_1)

    if body_2 >= 0:
        jt_x_2 = x_i * J_2
        m_inv_2 = body_inv_mass[world_idx, body_2]
        I_inv_2 = world_inv_inertia[world_idx, body_2]
        delta_v_2 = compute_spatial_momentum(m_inv_2, I_inv_2, jt_x_2)
        wp.atomic_add(out_dyn_vec, world_idx, body_2, delta_v_2)


@wp.kernel
def kernel_J_matvec_and_finalize(
    dyn_vec: wp.array(dtype=wp.spatial_vector, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    C_values: wp.array(dtype=wp.float32, ndim=2),
    vec_x: wp.array(dtype=wp.float32, ndim=2),
    vec_y: wp.array(dtype=wp.float32, ndim=2),
    alpha: float,
    beta: float,
    regularization: float,
    out_vec_z: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, constraint_idx = wp.tid()

    is_active = constraint_active_mask[world_idx, constraint_idx]

    # The result of A @ x
    Ax_i = 0.0

    if is_active > 0.0:
        body_1 = constraint_body_idx[world_idx, constraint_idx, 0]
        body_2 = constraint_body_idx[world_idx, constraint_idx, 1]
        J_1 = J_values[world_idx, constraint_idx, 0]
        J_2 = J_values[world_idx, constraint_idx, 1]

        j_v = 0.0
        # Standard sparse read (Gather) - No atomics needed here
        if body_1 >= 0:
            j_v += wp.dot(J_1, dyn_vec[world_idx, body_1])
        if body_2 >= 0:
            j_v += wp.dot(J_2, dyn_vec[world_idx, body_2])

        c_times_x = (C_values[world_idx, constraint_idx] + regularization) * vec_x[
            world_idx, constraint_idx
        ]
        Ax_i = j_v + c_times_x

    # Final composition: z = beta * y + alpha * Ax
    res = alpha * Ax_i
    if beta != 0.0:
        res += beta * vec_y[world_idx, constraint_idx]

    out_vec_z[world_idx, constraint_idx] = res


class SystemOperator(LinearOperator):
    def __init__(
        self,
        data: SystemLinearData,
        device: wp.Device,
        regularization: float = 1e-6,
    ):
        super().__init__(
            shape=(data.N_w, data.N_c, data.N_c),
            dtype=wp.float32,
            device=device,
            matvec=None,
        )
        self.data = data
        self.regularization = regularization

        self._tmp_dyn_vec = wp.zeros((data.N_w, data.N_b), dtype=wp.spatial_vector, device=device)

    def matvec(self, x, y, z, alpha, beta):
        # 1. Clear temp buffer
        self._tmp_dyn_vec.zero_()

        # 2. Compute v = M^-1 J^T x (Atomic Scatter)
        # This is optimal for small N_b because of L1 cache hits.
        wp.launch(
            kernel=kernel_invM_Jt_matvec_direct,
            dim=(self.data.N_w, self.data.N_c),
            inputs=[
                x,
                self.data.J_values,
                self.data.constraint_body_idx,
                self.data.constraint_active_mask,
                self.data.body_inv_mass,
                self.data.world_inv_inertia,
            ],
            outputs=[self._tmp_dyn_vec],
            device=self.device,
        )

        # 3. Compute z = beta*y + alpha * (J v + C x) (Gather)
        wp.launch(
            kernel=kernel_J_matvec_and_finalize,
            dim=(self.data.N_w, self.data.N_c),
            inputs=[
                self._tmp_dyn_vec,
                self.data.J_values,
                self.data.constraint_body_idx,
                self.data.constraint_active_mask,
                self.data.C_values,
                x,
                y,
                alpha,
                beta,
                self.regularization,
            ],
            outputs=[z],
            device=self.device,
        )
