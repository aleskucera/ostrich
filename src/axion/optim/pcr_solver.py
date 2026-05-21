import numpy as np
import warp as wp
from axion.tiled import TiledDot
from warp.optim.linear import LinearOperator


@wp.kernel
def _cr_update_xrz(
    # --- Inputs (Read Only) ---
    zAz: wp.array(dtype=wp.float32),
    yAp: wp.array(dtype=wp.float32),
    p: wp.array(dtype=wp.float32, ndim=2),
    Ap: wp.array(dtype=wp.float32, ndim=2),
    y: wp.array(dtype=wp.float32, ndim=2),
    # --- Modified (Read/Write) ---
    x: wp.array(dtype=wp.float32, ndim=2),
    r: wp.array(dtype=wp.float32, ndim=2),
    z: wp.array(dtype=wp.float32, ndim=2),
):
    """
    Updates x, r, and z.
    Convention: Inputs first, Modified arguments last.
    """
    world_idx, i = wp.tid()

    zAz_val = zAz[world_idx]
    yAp_val = yAp[world_idx]

    # Safe division
    alpha = float(0.0)
    if yAp_val > 0.0:
        alpha = zAz_val / yAp_val

    # Update state (Read/Write)
    x[world_idx, i] += alpha * p[world_idx, i]
    r[world_idx, i] -= alpha * Ap[world_idx, i]
    z[world_idx, i] -= alpha * y[world_idx, i]


@wp.kernel
def _cr_update_pAp(
    # --- Inputs ---
    zAz_old: wp.array(dtype=wp.float32),
    zAz_new: wp.array(dtype=wp.float32),
    z: wp.array(dtype=wp.float32, ndim=2),
    Az: wp.array(dtype=wp.float32, ndim=2),
    # --- Modified ---
    p: wp.array(dtype=wp.float32, ndim=2),
    Ap: wp.array(dtype=wp.float32, ndim=2),
):
    """Updates search direction p and Ap."""
    world_idx, i = wp.tid()

    zAz_old_val = zAz_old[world_idx]
    zAz_new_val = zAz_new[world_idx]

    # Safe division
    beta = float(0.0)
    if zAz_old_val > 0.0:
        beta = zAz_new_val / zAz_old_val

    # Update search vectors
    p[world_idx, i] = z[world_idx, i] + beta * p[world_idx, i]
    Ap[world_idx, i] = Az[world_idx, i] + beta * Ap[world_idx, i]


@wp.kernel
def _log_history_kernel(
    r_sq: wp.array(dtype=wp.float32),  # Current residual (B,)
    iter_ptr: wp.array(dtype=int),  # Current iter count (1,)
    history: wp.array(dtype=wp.float32, ndim=2),  # History buffer (max_iters, B)
):
    """
    Logs the current residual to the history buffer.
    """
    world_idx = wp.tid()
    iter_idx = iter_ptr[0]

    # Boundary check
    if iter_idx < history.shape[0]:
        history[iter_idx, world_idx] = r_sq[world_idx]


@wp.kernel
def _check_residuals_kernel(
    r_sq: wp.array(dtype=wp.float32),
    b_sq: wp.array(dtype=wp.float32),
    tol_sq: float,
    atol_sq: float,
    keep_running: wp.array(dtype=int),
):
    row = wp.tid()
    if row >= r_sq.shape[0]:
        return

    # Effective tolerance: max(atol^2, tol^2 * |b|^2)
    target = atol_sq
    rel_term = b_sq[row] * tol_sq
    if rel_term > target:
        target = rel_term

    # If ANY world is not converged, we must continue
    if r_sq[row] > target:
        keep_running[0] = 1


@wp.kernel
def _update_iter_kernel(
    iter_count: wp.array(dtype=int),
    max_iters: int,
    keep_running: wp.array(dtype=int),
):
    # Launched with dim=1
    current_iter = iter_count[0] + 1
    iter_count[0] = current_iter
    if current_iter >= max_iters:
        keep_running[0] = 0


class PCRSolver:
    def __init__(
        self,
        max_iters: int,
        batch_dim: int,
        vec_dim: int,
        device: wp.Device,
    ):
        self.max_iters = max_iters
        self.batch_dim = batch_dim
        self.vec_dim = vec_dim
        self.device = device

        with wp.ScopedDevice(device):
            # 1. Allocate Vectors (B, D)
            shape = (batch_dim, vec_dim)
            self.r = wp.zeros(shape, dtype=wp.float32)
            self.z = wp.zeros(shape, dtype=wp.float32)
            self.Az = wp.zeros(shape, dtype=wp.float32)
            self.p = wp.zeros(shape, dtype=wp.float32)
            self.Ap = wp.zeros(shape, dtype=wp.float32)
            self.y = wp.zeros(shape, dtype=wp.float32)

            # 2. Allocate Scalars (B,)
            self.zAz_old = wp.zeros((batch_dim,), dtype=wp.float32)
            self.zAz_new = wp.zeros((batch_dim,), dtype=wp.float32)
            self.y_Ap = wp.zeros((batch_dim,), dtype=wp.float32)

            # Convergence buffers (B,)
            self.b_sq = wp.zeros((batch_dim,), dtype=wp.float32)
            self.r_sq = wp.zeros((batch_dim,), dtype=wp.float32)

            # Control flow (scalars)
            self.keep_running = wp.zeros((1,), dtype=int)
            self.iter_count = wp.zeros((1,), dtype=int)

            # 3. Squared residual history for logging
            self.history_r_sq = wp.zeros(
                (max_iters + 1, batch_dim),
                dtype=wp.float32,
            )

        # Tiled Operations
        self.tiled_dot = TiledDot(
            shape=(batch_dim, vec_dim),
            dtype=wp.float32,
            tile_size=256,
            device=device,
        )

    def solve(
        self,
        A: LinearOperator,
        b: wp.array,
        x: wp.array,
        preconditioner: LinearOperator,
        iters: int,
        tol: float = 1e-5,
        atol: float = 1e-5,
        log: bool = False,
    ):
        """
        Solves Ax = b.

        Args:
            log (bool):
                If False (Fast Mode): Returns 'iter_count' as wp.array. No CPU sync.
                If True (Debug Mode): Returns dict with 'iterations' (int) and 'residuals' (numpy).
        """
        # --- Validation & Setup ---
        assert x.shape == (self.batch_dim, self.vec_dim)
        assert b.shape == (self.batch_dim, self.vec_dim)
        assert iters <= self.max_iters

        tol_sq = float(tol**2) if tol >= 0 else -1.0
        atol_sq = float(atol**2) if atol >= 0 else -1.0

        # Reset State
        self.keep_running.fill_(1)
        self.iter_count.zero_()

        if log:
            self.history_r_sq.zero_()

        # --- Initialization (Outside Graph) ---

        # 1. Compute initial residuals and search directions
        self.tiled_dot.compute(b, b, self.b_sq)  # |b|^2
        A.matvec(x, b, self.r, alpha=-1.0, beta=1.0)  # r = b - Ax
        preconditioner.matvec(self.r, self.z, self.z, alpha=1.0, beta=0.0)  # z = M^-1 r
        A.matvec(self.z, self.Az, self.Az, alpha=1.0, beta=0.0)  # Az = A z

        wp.copy(dest=self.p, src=self.z)
        wp.copy(dest=self.Ap, src=self.Az)

        # 2. Initial Scalars
        self.tiled_dot.compute(self.z, self.Az, self.zAz_new)

        # 3. Initial Convergence Check (for Iteration 0)
        self.tiled_dot.compute(self.r, self.r, self.r_sq)

        # If logging, record iteration 0
        if log:
            wp.launch(
                kernel=_log_history_kernel,
                dim=self.batch_dim,
                device=self.device,
                inputs=[self.r_sq, self.iter_count, self.history_r_sq],
            )

        # --- Graph Definition ---

        def solver_step():
            self.keep_running.zero_()  # Optimistic set to 0

            # 1. Prepare Scalars
            wp.copy(dest=self.zAz_old, src=self.zAz_new)

            # 2. Preconditioner Step: y = P^-1 Ap
            preconditioner.matvec(self.Ap, self.y, self.y, alpha=1.0, beta=0.0)

            # 3. Step Size Alpha: y_Ap = y . Ap
            self.tiled_dot.compute(self.y, self.Ap, self.y_Ap)

            # 4. Update Solution: x, r, z
            wp.launch(
                kernel=_cr_update_xrz,
                dim=(self.batch_dim, self.vec_dim),
                device=self.device,
                inputs=[self.zAz_old, self.y_Ap, self.p, self.Ap, self.y, x, self.r, self.z],
            )

            # 5. Update Gradients: Az = A z
            A.matvec(self.z, self.Az, self.Az, alpha=1.0, beta=0.0)

            # 6. Step Size Beta: zAz_new = z . Az
            self.tiled_dot.compute(self.z, self.Az, self.zAz_new)

            # 7. Update Search Dir: p, Ap
            wp.launch(
                kernel=_cr_update_pAp,
                dim=(self.batch_dim, self.vec_dim),
                device=self.device,
                inputs=[self.zAz_old, self.zAz_new, self.z, self.Az, self.p, self.Ap],
            )

            # 8. Convergence Check
            self.tiled_dot.compute(self.r, self.r, self.r_sq)

            # Check residuals across all worlds
            wp.launch(
                kernel=_check_residuals_kernel,
                dim=(self.batch_dim),
                device=self.device,
                inputs=[self.r_sq, self.b_sq, tol_sq, atol_sq, self.keep_running],
            )

            # Safely manage iter count with exactly 1 thread
            wp.launch(
                kernel=_update_iter_kernel,
                dim=1,
                device=self.device,
                inputs=[self.iter_count, iters, self.keep_running],
            )

            # 9. Logging (Optional, inside graph)
            if log:
                wp.launch(
                    kernel=_log_history_kernel,
                    dim=self.batch_dim,
                    device=self.device,
                    inputs=[self.r_sq, self.iter_count, self.history_r_sq],
                )

        # --- Capture and Run ---
        wp.capture_while(self.keep_running, solver_step)

        # if log:
        #     iters_cpu = int(self.iter_count.numpy()[0])
        #     # History buffer includes iter 0 + iters steps
        #     hist_np = self.history_r_sq.numpy()[: iters_cpu + 1]
        #     r_sq_np = self.r_sq.numpy()
        #
        #     return {
        #         "final_residual_squared": r_sq_np,
        #         "iterations": iters_cpu,
        #         "residual_squared_history": hist_np,
        #     }
        #
        # return {}
