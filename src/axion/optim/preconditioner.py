"""
Jacobi preconditioner for the system matrix A = J M⁻¹ Jᵀ + C.
"""
import warp as wp
from axion.mechanics import compute_spatial_momentum
from warp.optim.linear import LinearOperator


@wp.kernel
def compute_inv_diag_kernel(
    body_inv_mass: wp.array(dtype=wp.float32, ndim=2),
    world_inv_inertia: wp.array(dtype=wp.mat33, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    C_values: wp.array(dtype=wp.float32, ndim=2),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    regularization: wp.float32,
    # Output array
    P_inv_diag: wp.array(dtype=wp.float32, ndim=2),
):
    """
    Computes the inverse of the diagonal of the system matrix A = J M⁻¹ Jᵀ + C.
    The result P_inv_diag[i] = 1.0 / A[i,i] is stored.
    """
    world_idx, constraint_idx = wp.tid()

    is_active = constraint_active_mask[world_idx, constraint_idx]
    if is_active == 0.0:
        return

    body_1 = constraint_body_idx[world_idx, constraint_idx, 0]
    body_2 = constraint_body_idx[world_idx, constraint_idx, 1]

    result = 0.0
    if body_1 >= 0:
        m_inv_1 = body_inv_mass[world_idx, body_1]
        I_inv_1 = world_inv_inertia[world_idx, body_1]
        J_1 = J_values[world_idx, constraint_idx, 0]
        result += wp.dot(J_1, compute_spatial_momentum(m_inv_1, I_inv_1, J_1))
    if body_2 >= 0:
        m_inv_2 = body_inv_mass[world_idx, body_2]
        I_inv_2 = world_inv_inertia[world_idx, body_2]
        J_2 = J_values[world_idx, constraint_idx, 1]
        result += wp.dot(J_2, compute_spatial_momentum(m_inv_2, I_inv_2, J_2))

    # Add diagonal compliance term C[i,i]
    diag_A = result + C_values[world_idx, constraint_idx] + regularization

    # Compute and store inverse, with stabilization
    P_inv_diag[world_idx, constraint_idx] = 1.0 / (diag_A + 1e-6)


@wp.kernel
def apply_preconditioner_kernel(
    P_inv_diag: wp.array(dtype=wp.float32, ndim=2),
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    vec_x: wp.array(dtype=wp.float32, ndim=2),
    vec_y: wp.array(dtype=wp.float32, ndim=2),
    alpha: float,
    beta: float,
    out_vec_z: wp.array(dtype=wp.float32, ndim=2),
):
    """Applies the Jacobi preconditioner: z = beta*y + alpha * P⁻¹ * x"""
    world_idx, constraint_idx = wp.tid()

    is_active = constraint_active_mask[world_idx, constraint_idx]

    # Calculate the preconditioned value (M⁻¹ x)
    # If inactive, the result of the matrix operation is 0.0
    preconditioned_x = 0.0
    if is_active > 0.0:
        preconditioned_x = P_inv_diag[world_idx, constraint_idx] * vec_x[world_idx, constraint_idx]

    # Combine with beta * y and write to output.
    if beta == 0.0:
        out_vec_z[world_idx, constraint_idx] = alpha * preconditioned_x
    else:
        out_vec_z[world_idx, constraint_idx] = (
            beta * vec_y[world_idx, constraint_idx] + alpha * preconditioned_x
        )


class JacobiPreconditioner(LinearOperator):
    """
    A Jacobi (diagonal) preconditioner for the system matrix A = J M⁻¹ Jᵀ + C.

    This class provides a .matvec() method that applies the inverse of the
    diagonal of A, for use with Warp's iterative solvers.
    """

    def __init__(self, engine, regularization):
        super().__init__(
            shape=(engine.dims.N_w, engine.dims.N_c, engine.dims.N_c),
            dtype=wp.float32,
            device=engine.device,
            matvec=None,  # Will be set later
        )
        self.engine = engine
        self.regularization = regularization

        # Storage for the inverse diagonal elements
        self._P_inv_diag = wp.zeros(
            (engine.dims.N_w, engine.dims.N_c), dtype=wp.float32, device=self.device
        )

    def update(self):
        """
        Re-computes the preconditioner's data. This must be called each time
        the Jacobian (J) or compliance (C) values change.
        """
        wp.launch(
            kernel=compute_inv_diag_kernel,
            dim=(self.engine.dims.num_worlds, self.engine.dims.num_constraints),
            inputs=[
                self.engine.axion_model.body_inv_mass,
                self.engine.data.world_inv_inertia,
                self.engine.data.J_values.full,
                self.engine.data.C_values.full,
                self.engine.data.constr_body_idx.full,
                self.engine.data.constr_active_mask.full,
                self.regularization,
            ],
            outputs=[
                self._P_inv_diag,
            ],
            device=self.device,
        )

    def matvec(self, x, y, z, alpha, beta):
        """
        Performs the preconditioning operation z = beta*y + alpha*(M⁻¹@x),
        where M⁻¹ is the inverse diagonal matrix stored in `_P_inv_diag`.
        """
        wp.launch(
            kernel=apply_preconditioner_kernel,
            dim=(self.engine.dims.num_worlds, self.engine.dims.num_constraints),
            inputs=[
                self._P_inv_diag,
                self.engine.data.constr_active_mask.full,
                x,
                y,
                alpha,
                beta,
                z,
            ],
            device=self.device,
        )
