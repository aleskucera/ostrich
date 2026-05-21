from dataclasses import dataclass

import warp as wp
from axion.mechanics import compute_spatial_momentum
from axion.mechanics import compute_world_inertia


@dataclass
class FullSystemLinearData:
    N_w: int
    N_c: int
    N_b: int
    N_u: int
    J_values: wp.array
    constraint_body_idx: wp.array
    constraint_active_mask: wp.array
    C_values: wp.array

    body_mass: wp.array
    body_inertia: wp.array
    body_pose: wp.array
    engine_data: object  # EngineData reference, for reading dt dynamically

    @property
    def dt(self) -> float:
        return self.engine_data.dt

    @classmethod
    def from_engine(cls, engine):
        return cls(
            N_w=engine.dims.N_w,
            N_c=engine.dims.N_c,
            N_b=engine.dims.N_b,
            N_u=engine.dims.N_u,
            J_values=engine.data.J_values.full,
            constraint_body_idx=engine.data.constr_body_idx.full,
            constraint_active_mask=engine.data.constr_active_mask.full,
            C_values=engine.data.C_values.full,
            body_mass=engine.axion_model.body_mass,
            body_inertia=engine.axion_model.body_inertia,
            body_pose=engine.data.body_pose,
            engine_data=engine.data,
        )


@wp.kernel
def kernel_M_times_v(
    body_vel: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_pose: wp.array(dtype=wp.transform, ndim=2),
    body_mass: wp.array(dtype=wp.float32, ndim=2),
    body_inertia: wp.array(dtype=wp.mat33, ndim=2),
    # Output
    out_dyn_vec: wp.array(dtype=wp.spatial_vector, ndim=2),
):
    """Compute M * v for each body."""
    world_idx, body_idx = wp.tid()
    if body_idx >= body_vel.shape[1]:
        return

    m = body_mass[world_idx, body_idx]
    I_body = body_inertia[world_idx, body_idx]
    I_world = compute_world_inertia(body_pose[world_idx, body_idx], I_body)

    v = body_vel[world_idx, body_idx]
    out_dyn_vec[world_idx, body_idx] = compute_spatial_momentum(m, I_world, v)


@wp.kernel
def kernel_Jt_scatter(
    lam: wp.array(dtype=wp.float32, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    scale_factor: float,
    # Output: atomic accumulation into body vector
    out_dyn_vec: wp.array(dtype=wp.spatial_vector, ndim=2),
):
    """Compute scale_factor * J^T * lambda, scattered into body velocity space."""
    world_idx, constraint_idx = wp.tid()

    is_active = constraint_active_mask[world_idx, constraint_idx]
    if is_active == 0.0:
        return

    body_1 = constraint_body_idx[world_idx, constraint_idx, 0]
    body_2 = constraint_body_idx[world_idx, constraint_idx, 1]

    J_1 = J_values[world_idx, constraint_idx, 0]
    J_2 = J_values[world_idx, constraint_idx, 1]
    lam_i = lam[world_idx, constraint_idx]

    scale = scale_factor * lam_i

    if body_1 >= 0:
        wp.atomic_add(out_dyn_vec, world_idx, body_1, scale * J_1)

    if body_2 >= 0:
        wp.atomic_add(out_dyn_vec, world_idx, body_2, scale * J_2)


@wp.kernel
def kernel_J_gather_and_C(
    body_vel: wp.array(dtype=wp.spatial_vector, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    C_values: wp.array(dtype=wp.float32, ndim=2),
    lam: wp.array(dtype=wp.float32, ndim=2),
    j_scale: float,
    c_scale: float,
    # Output
    out_constraint_vec: wp.array(dtype=wp.float32, ndim=2),
):
    """Compute j_scale * J * v + c_scale * C * lambda."""
    world_idx, constraint_idx = wp.tid()

    is_active = constraint_active_mask[world_idx, constraint_idx]

    result = 0.0

    if is_active > 0.0:
        body_1 = constraint_body_idx[world_idx, constraint_idx, 0]
        body_2 = constraint_body_idx[world_idx, constraint_idx, 1]
        J_1 = J_values[world_idx, constraint_idx, 0]
        J_2 = J_values[world_idx, constraint_idx, 1]

        j_v = 0.0
        if body_1 >= 0:
            j_v += wp.dot(J_1, body_vel[world_idx, body_1])
        if body_2 >= 0:
            j_v += wp.dot(J_2, body_vel[world_idx, body_2])

        c_lam = C_values[world_idx, constraint_idx] * lam[world_idx, constraint_idx]

        result = j_scale * j_v + c_scale * c_lam

    out_constraint_vec[world_idx, constraint_idx] = result


class FullSystemOperator:
    """Full system operator computing:

        A_full @ [v; λ] = [ M·v - dt·Jᵀ·λ ]
                          [ J·v + dt·C·λ   ]

    Unlike SystemOperator (Schur complement in constraint space only),
    this operates on the combined (N_u + N_c) vector.
    """

    def __init__(self, data: FullSystemLinearData, device: wp.Device):
        self.data = data
        self.device = device

        self._tmp_dyn_vec = wp.zeros(
            (data.N_w, data.N_b), dtype=wp.spatial_vector, device=device
        )
        self._out_dyn_vec = wp.zeros(
            (data.N_w, data.N_b), dtype=wp.spatial_vector, device=device
        )
        self._out_constraint_vec = wp.zeros(
            (data.N_w, data.N_c), dtype=wp.float32, device=device
        )

    def _launch_matvec(self, x_vel, x_lam, out_vel, out_lam, jt_scale, j_scale, c_scale):
        """Shared implementation for matvec and matvec_transpose."""
        # out_vel = M * x_vel
        wp.launch(
            kernel=kernel_M_times_v,
            dim=(self.data.N_w, self.data.N_b),
            inputs=[x_vel, self.data.body_pose, self.data.body_mass, self.data.body_inertia],
            outputs=[out_vel],
            device=self.device,
        )

        # out_vel += jt_scale * J^T * x_lam  (atomic add)
        wp.launch(
            kernel=kernel_Jt_scatter,
            dim=(self.data.N_w, self.data.N_c),
            inputs=[
                x_lam,
                self.data.J_values,
                self.data.constraint_body_idx,
                self.data.constraint_active_mask,
                jt_scale,
            ],
            outputs=[out_vel],
            device=self.device,
        )

        # out_lam = j_scale * J * x_vel + c_scale * C * x_lam
        wp.launch(
            kernel=kernel_J_gather_and_C,
            dim=(self.data.N_w, self.data.N_c),
            inputs=[
                x_vel,
                self.data.J_values,
                self.data.constraint_body_idx,
                self.data.constraint_active_mask,
                self.data.C_values,
                x_lam,
                j_scale,
                c_scale,
            ],
            outputs=[out_lam],
            device=self.device,
        )

    def matvec(self, x_vel, x_lam, out_vel, out_lam):
        """Compute A_full @ [x_vel; x_lam] -> [out_vel; out_lam].

        A_full = [ M      -dt·Jᵀ ]
                 [ J       dt·C  ]
        """
        dt = self.data.dt
        self._launch_matvec(x_vel, x_lam, out_vel, out_lam,
                            jt_scale=-dt, j_scale=1.0, c_scale=dt)

    def matvec_transpose(self, x_vel, x_lam, out_vel, out_lam):
        """Compute A_fullᵀ @ [x_vel; x_lam] -> [out_vel; out_lam].

        A_fullᵀ = [ M        Jᵀ   ]
                   [-dt·J    dt·C  ]
        """
        dt = self.data.dt
        self._launch_matvec(x_vel, x_lam, out_vel, out_lam,
                            jt_scale=1.0, j_scale=-dt, c_scale=dt)
