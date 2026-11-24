"""
Deflated PCR (DPCR) with unit-vector deflation on a row set S.

S holds the joint and control rows (always the first `offset_n` rows) and,
optionally, contact rows picked per system by the AMG strength of connection.
With Z = unit vectors on S the Galerkin projection simplifies to

    AZ = A[:, S],   E = Zᵀ A Z = A_SS,   P = I − AQ,   Q = Z E⁻¹ Zᵀ,

and PCR on P A x̃ = P b iterates on the Schur complement of the rows outside S,
while the rows in S are solved exactly through a Cholesky factor of A_SS.
The solution is recovered as x = Q b + Pᵀ x̃.

Classes:
    RowDeflation              shared state: S, AZ, the Cholesky factor of A_SS
    DeflatedSystemOperator    P A, for the PCR iterations
    DeflationPreconditioner   updates the deflation, then Jacobi on diag(A_F) or diag(A)
    DeflatedPCRSolver         projects b, runs PCR, recovers x

The factor and S live in EngineData (`deflation_L`, `deflation_rows`) so the
TrajectoryBuffer can store them per step for the adjoint.
"""
import warp as wp
from ostrich.mechanics import compute_spatial_momentum
from warp.optim.linear import LinearOperator

from .pcr_solver import PCRSolver
from .preconditioner import JacobiPreconditioner
from .system_operator import SystemOperator


@wp.func
def _row_coupling(
    world_idx: int,
    row_i: int,
    row_j: int,
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
    body_inv_mass: wp.array(dtype=wp.float32, ndim=2),
    world_inv_inertia: wp.array(dtype=wp.mat33, ndim=2),
) -> float:
    """(J M⁻¹ Jᵀ)_ij: sum over the bodies the two rows share."""
    result = float(0.0)
    for s in range(2):
        body = constraint_body_idx[world_idx, row_i, s]
        if body >= 0:
            for t in range(2):
                if constraint_body_idx[world_idx, row_j, t] == body:
                    minv_J = compute_spatial_momentum(
                        body_inv_mass[world_idx, body],
                        world_inv_inertia[world_idx, body],
                        J_values[world_idx, row_j, t],
                    )
                    result += wp.dot(J_values[world_idx, row_i, s], minv_J)
    return result


@wp.kernel
def _diag_kernel(
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    C_values: wp.array(dtype=wp.float32, ndim=2),
    body_inv_mass: wp.array(dtype=wp.float32, ndim=2),
    world_inv_inertia: wp.array(dtype=wp.mat33, ndim=2),
    regularization: float,
    diag: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, i = wp.tid()
    d = float(0.0)
    if constraint_active_mask[world_idx, i] > 0.0:
        d = _row_coupling(
            world_idx, i, i, J_values, constraint_body_idx, body_inv_mass, world_inv_inertia
        ) + C_values[world_idx, i] + regularization
    diag[world_idx, i] = d


@wp.kernel
def _init_selection_kernel(
    first_candidate: int,
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    score: wp.array(dtype=wp.float32, ndim=2),
):
    """score = 0 on active candidate rows (raised by _strength_pair_kernel), −1 elsewhere."""
    world_idx, r = wp.tid()
    s = float(-1.0)
    if r >= first_candidate and constraint_active_mask[world_idx, r] > 0.0:
        s = 0.0
    score[world_idx, r] = s


@wp.kernel
def _strength_pair_kernel(
    first_candidate: int,
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    body_inv_mass: wp.array(dtype=wp.float32, ndim=2),
    world_inv_inertia: wp.array(dtype=wp.mat33, ndim=2),
    diag: wp.array(dtype=wp.float32, ndim=2),
    score: wp.array(dtype=wp.float32, ndim=2),
):
    """score_r = max_{j≠r} |A_rj| / sqrt(A_rr A_jj), one thread per pair (r, j)."""
    world_idx, r, j = wp.tid()
    if r < first_candidate or j == r:
        return
    if constraint_active_mask[world_idx, r] == 0.0 or constraint_active_mask[world_idx, j] == 0.0:
        return
    a_rj = _row_coupling(world_idx, r, j, J_values, constraint_body_idx, body_inv_mass, world_inv_inertia)
    if a_rj != 0.0:
        wp.atomic_max(score, world_idx, r, wp.abs(a_rj) / wp.sqrt(diag[world_idx, r] * diag[world_idx, j]))


@wp.kernel
def _init_rows_kernel(
    n_fixed: int,
    rows: wp.array(dtype=wp.int32, ndim=2),
):
    """S = the n_fixed joint/control rows; the strength slots start unused (−1)."""
    world_idx, a = wp.tid()
    r = int(-1)
    if a < n_fixed:
        r = a
    rows[world_idx, a] = r


@wp.kernel
def _select_rows_kernel(
    n_fixed: int,
    score: wp.array(dtype=wp.float32, ndim=2),
    rows: wp.array(dtype=wp.int32, ndim=2),
):
    """Put candidate r into strength slot `rank` if fewer than k_c rows score higher
    (ties broken by index); slots without a candidate keep −1."""
    world_idx, r = wp.tid()
    s = score[world_idx, r]
    if s <= 0.0:
        return
    rank = int(0)
    for j in range(score.shape[1]):
        sj = score[world_idx, j]
        if sj > s or (sj == s and j < r):
            rank += 1
    if n_fixed + rank < rows.shape[1]:
        rows[world_idx, n_fixed + rank] = r


@wp.kernel
def _mark_rows_kernel(
    rows: wp.array(dtype=wp.int32, ndim=2),
    in_S: wp.array(dtype=wp.int32, ndim=2),
):
    world_idx, a = wp.tid()
    r = rows[world_idx, a]
    if r >= 0:
        in_S[world_idx, r] = 1


@wp.kernel
def _az_kernel(
    rows: wp.array(dtype=wp.int32, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    C_values: wp.array(dtype=wp.float32, ndim=2),
    body_inv_mass: wp.array(dtype=wp.float32, ndim=2),
    world_inv_inertia: wp.array(dtype=wp.mat33, ndim=2),
    regularization: float,
    AZt: wp.array(dtype=wp.float32, ndim=3),
):
    """AZt[a, i] = A[i, S_a]; zero for an unused or inactive S_a and for inactive rows i."""
    world_idx, a, i = wp.tid()
    r = rows[world_idx, a]
    value = float(0.0)
    if r >= 0:
        if constraint_active_mask[world_idx, r] > 0.0 and constraint_active_mask[world_idx, i] > 0.0:
            value = _row_coupling(
                world_idx, i, r, J_values, constraint_body_idx, body_inv_mass, world_inv_inertia
            )
            if i == r:
                value += C_values[world_idx, r] + regularization
    AZt[world_idx, a, i] = value


@wp.func
def _row_used(
    world_idx: int,
    a: int,
    rows: wp.array(dtype=wp.int32, ndim=2),
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
) -> bool:
    r = rows[world_idx, a]
    if r < 0:
        return False
    return constraint_active_mask[world_idx, r] > 0.0


@wp.func
def _E_entry(
    world_idx: int,
    a: int,
    b: int,
    rows: wp.array(dtype=wp.int32, ndim=2),
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    AZt: wp.array(dtype=wp.float32, ndim=3),
) -> float:
    """E = A_SS, with an identity row/column on unused slots of S."""
    if _row_used(world_idx, a, rows, constraint_active_mask) and _row_used(
        world_idx, b, rows, constraint_active_mask
    ):
        return AZt[world_idx, b, rows[world_idx, a]]
    if a == b:
        return 1.0
    return 0.0


@wp.kernel
def _cholesky_column_kernel(
    j: int,
    rows: wp.array(dtype=wp.int32, ndim=2),
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    AZt: wp.array(dtype=wp.float32, ndim=3),
    L: wp.array(dtype=wp.float32, ndim=3),
    factor_failure: wp.array(dtype=wp.int32),
):
    """Column j of E = L Lᵀ, rows i >= j in parallel. Launched once per column, so
    the columns p < j it reads are final; every thread recomputes the pivot."""
    world_idx, i = wp.tid()
    if i < j:
        return
    d = _E_entry(world_idx, j, j, rows, constraint_active_mask, AZt)
    for p in range(j):
        d -= L[world_idx, j, p] * L[world_idx, j, p]
    if d <= 0.0:
        factor_failure[world_idx] = 1
        d = 1.0e-12
    d = wp.sqrt(d)
    if i == j:
        L[world_idx, j, j] = d
    else:
        v = _E_entry(world_idx, i, j, rows, constraint_active_mask, AZt)
        for p in range(j):
            v -= L[world_idx, i, p] * L[world_idx, j, p]
        L[world_idx, i, j] = v / d


@wp.kernel
def _lower_inverse_row_kernel(
    i: int,
    L: wp.array(dtype=wp.float32, ndim=3),
    Linv: wp.array(dtype=wp.float32, ndim=3),
):
    """Row i of L⁻¹ (all columns c in parallel), by forward substitution. Launched once per row."""
    world_idx, c = wp.tid()
    v = float(0.0)
    if c <= i:
        if c == i:
            v = 1.0
        for p in range(c, i):
            v -= L[world_idx, i, p] * Linv[world_idx, p, c]
        v = v / L[world_idx, i, i]
    Linv[world_idx, i, c] = v


@wp.kernel
def _einv_kernel(
    Linv: wp.array(dtype=wp.float32, ndim=3),
    Einv: wp.array(dtype=wp.float32, ndim=3),
):
    """E⁻¹ = L⁻ᵀ L⁻¹."""
    world_idx, a, b = wp.tid()
    v = float(0.0)
    for c in range(wp.max(a, b), Linv.shape[1]):
        v += Linv[world_idx, c, a] * Linv[world_idx, c, b]
    Einv[world_idx, a, b] = v


@wp.kernel
def _einv_apply_kernel(
    Einv: wp.array(dtype=wp.float32, ndim=3),
    t: wp.array(dtype=wp.float32, ndim=2),
    s: wp.array(dtype=wp.float32, ndim=2),
):
    """s = E⁻¹ t."""
    world_idx, a = wp.tid()
    v = float(0.0)
    for b in range(t.shape[1]):
        v += Einv[world_idx, a, b] * t[world_idx, b]
    s[world_idx, a] = v


@wp.kernel
def _gather_rows_kernel(
    rows: wp.array(dtype=wp.int32, ndim=2),
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    v: wp.array(dtype=wp.float32, ndim=2),
    t: wp.array(dtype=wp.float32, ndim=2),
):
    """t = v_S (Zᵀ v)."""
    world_idx, a = wp.tid()
    value = float(0.0)
    if _row_used(world_idx, a, rows, constraint_active_mask):
        value = v[world_idx, rows[world_idx, a]]
    t[world_idx, a] = value


@wp.kernel
def _residual_rows_kernel(
    tb: wp.array(dtype=wp.float32, ndim=2),
    AZt: wp.array(dtype=wp.float32, ndim=3),
    x: wp.array(dtype=wp.float32, ndim=2),
    t: wp.array(dtype=wp.float32, ndim=2),
):
    """t = Zᵀ b − (AZ)ᵀ x."""
    world_idx, a = wp.tid()
    value = tb[world_idx, a]
    for i in range(x.shape[1]):
        value -= AZt[world_idx, a, i] * x[world_idx, i]
    t[world_idx, a] = value


@wp.kernel
def _project_kernel(
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    AZt: wp.array(dtype=wp.float32, ndim=3),
    s: wp.array(dtype=wp.float32, ndim=2),
    v: wp.array(dtype=wp.float32, ndim=2),
    vec_y: wp.array(dtype=wp.float32, ndim=2),
    alpha: float,
    beta: float,
    out_vec_z: wp.array(dtype=wp.float32, ndim=2),
):
    """z = beta*y + alpha*(v − AZ s)."""
    world_idx, i = wp.tid()
    value = float(0.0)
    if constraint_active_mask[world_idx, i] > 0.0:
        value = v[world_idx, i]
        for a in range(s.shape[1]):
            value -= AZt[world_idx, a, i] * s[world_idx, a]
    res = alpha * value
    if beta != 0.0:
        res += beta * vec_y[world_idx, i]
    out_vec_z[world_idx, i] = res


@wp.kernel
def _scatter_add_rows_kernel(
    rows: wp.array(dtype=wp.int32, ndim=2),
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    s: wp.array(dtype=wp.float32, ndim=2),
    x: wp.array(dtype=wp.float32, ndim=2),
):
    """x += Z s."""
    world_idx, a = wp.tid()
    if _row_used(world_idx, a, rows, constraint_active_mask):
        x[world_idx, rows[world_idx, a]] += s[world_idx, a]


@wp.kernel
def _linv_az_kernel(
    in_S: wp.array(dtype=wp.int32, ndim=2),
    Linv: wp.array(dtype=wp.float32, ndim=3),
    AZt: wp.array(dtype=wp.float32, ndim=3),
    Y: wp.array(dtype=wp.float32, ndim=3),
):
    """Y[c, i] = (L⁻¹ A[S, i])_c for the rows i outside S."""
    world_idx, c, i = wp.tid()
    v = float(0.0)
    if in_S[world_idx, i] == 0:
        for p in range(c + 1):
            v += Linv[world_idx, c, p] * AZt[world_idx, p, i]
    Y[world_idx, c, i] = v


@wp.kernel
def _schur_inv_diag_kernel(
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    in_S: wp.array(dtype=wp.int32, ndim=2),
    diag: wp.array(dtype=wp.float32, ndim=2),
    Y: wp.array(dtype=wp.float32, ndim=3),
    P_inv_diag: wp.array(dtype=wp.float32, ndim=2),
):
    """1 / diag(A_F)_i with diag(A_F)_i = A_ii − ‖L⁻¹ A[S, i]‖²; rows in S keep diag(A)."""
    world_idx, i = wp.tid()
    if constraint_active_mask[world_idx, i] == 0.0:
        return
    d = diag[world_idx, i]
    if in_S[world_idx, i] == 0:
        sq = float(0.0)
        for c in range(Y.shape[1]):
            sq += Y[world_idx, c, i] * Y[world_idx, c, i]
        # Rounding can push the Schur diagonal below zero for nearly dependent rows.
        d = wp.max(d - sq, 1.0e-6 * d)
    P_inv_diag[world_idx, i] = 1.0 / (d + 1.0e-6)


class RowDeflation:
    """Shared state of unit-vector deflation on the row set S.

    S = the first `num_fixed_rows` rows (joint + control) followed by
    `strength_rows` contact rows chosen per system. A slot of S that is unused
    (no candidate left, or an inactive row) gets an identity column in E and a
    zero column in AZ, so E stays SPD and the slot has no effect.
    """

    def __init__(self, engine, num_fixed_rows: int, strength_rows: int, regularization: float):
        self.engine = engine
        self.num_fixed_rows = num_fixed_rows
        self.strength_rows = strength_rows
        self.regularization = regularization

        dims = engine.dims
        self.rows = engine.data.deflation_rows
        self.L = engine.data.deflation_L
        self.k = self.rows.shape[1]
        assert self.k == num_fixed_rows + strength_rows and self.k > 0

        shape_k = (dims.N_w, self.k)
        with wp.ScopedDevice(engine.device):
            self.AZt = wp.zeros((dims.N_w, self.k, dims.N_c), dtype=wp.float32)
            self.Y = wp.zeros((dims.N_w, self.k, dims.N_c), dtype=wp.float32)
            self.Linv = wp.zeros((dims.N_w, self.k, self.k), dtype=wp.float32)
            self.Einv = wp.zeros((dims.N_w, self.k, self.k), dtype=wp.float32)
            self.diag = wp.zeros((dims.N_w, dims.N_c), dtype=wp.float32)
            self.score = wp.zeros((dims.N_w, dims.N_c), dtype=wp.float32)
            self.in_S = wp.zeros((dims.N_w, dims.N_c), dtype=wp.int32)
            self.tb = wp.zeros(shape_k, dtype=wp.float32)
            self.t = wp.zeros(shape_k, dtype=wp.float32)
            self.s = wp.zeros(shape_k, dtype=wp.float32)
            self.factor_failure = wp.zeros((dims.N_w,), dtype=wp.int32)

    def _system_inputs(self):
        data, model = self.engine.data, self.engine.ostrich_model
        return [
            data.J_values.full,
            data.constr_body_idx.full,
            data.constr_active_mask.full,
            data.C_values.full,
            model.body_inv_mass,
            data.world_inv_inertia,
        ]

    def update(self, refactor: bool = True):
        """Rebuild S, AZ, the factor of A_SS and E⁻¹ for the current linear system.

        refactor=False keeps S and the factor currently in EngineData (e.g. loaded
        from the TrajectoryBuffer) and only rebuilds AZ and E⁻¹ from that factor.
        The Cholesky factor and L⁻¹ take one launch per column or row, each
        parallel over the rows of S; E⁻¹ is then applied as a dense product.
        """
        dims, dev = self.engine.dims, self.engine.device
        J, bidx, mask, C, minv, Iinv = self._system_inputs()

        wp.launch(
            _diag_kernel, dim=(dims.N_w, dims.N_c),
            inputs=[J, bidx, mask, C, minv, Iinv, self.regularization],
            outputs=[self.diag], device=dev,
        )
        if refactor:
            wp.launch(_init_rows_kernel, dim=(dims.N_w, self.k), inputs=[self.num_fixed_rows], outputs=[self.rows],
                      device=dev)
            if self.strength_rows > 0:
                wp.launch(_init_selection_kernel, dim=(dims.N_w, dims.N_c), inputs=[self.num_fixed_rows, mask],
                          outputs=[self.score], device=dev)
                wp.launch(
                    _strength_pair_kernel, dim=(dims.N_w, dims.N_c, dims.N_c),
                    inputs=[self.num_fixed_rows, J, bidx, mask, minv, Iinv, self.diag],
                    outputs=[self.score], device=dev,
                )
                wp.launch(_select_rows_kernel, dim=(dims.N_w, dims.N_c), inputs=[self.num_fixed_rows, self.score],
                          outputs=[self.rows], device=dev)

        self.in_S.zero_()
        wp.launch(_mark_rows_kernel, dim=(dims.N_w, self.k), inputs=[self.rows], outputs=[self.in_S], device=dev)
        wp.launch(
            _az_kernel, dim=(dims.N_w, self.k, dims.N_c),
            inputs=[self.rows, J, bidx, mask, C, minv, Iinv, self.regularization],
            outputs=[self.AZt], device=dev,
        )
        if refactor:
            self.factor_failure.zero_()
            for j in range(self.k):
                wp.launch(
                    _cholesky_column_kernel, dim=(dims.N_w, self.k),
                    inputs=[j, self.rows, mask, self.AZt], outputs=[self.L, self.factor_failure], device=dev,
                )
        for i in range(self.k):
            wp.launch(_lower_inverse_row_kernel, dim=(dims.N_w, self.k), inputs=[i, self.L], outputs=[self.Linv],
                      device=dev)
        wp.launch(_einv_kernel, dim=(dims.N_w, self.k, self.k), inputs=[self.Linv], outputs=[self.Einv], device=dev)

    def schur_inv_diag(self, P_inv_diag: wp.array):
        """Write 1/diag(A_F) (diag(A) on the rows of S) into P_inv_diag."""
        dims, dev = self.engine.dims, self.engine.device
        wp.launch(_linv_az_kernel, dim=(dims.N_w, self.k, dims.N_c), inputs=[self.in_S, self.Linv, self.AZt],
                  outputs=[self.Y], device=dev)
        wp.launch(
            _schur_inv_diag_kernel, dim=(dims.N_w, dims.N_c),
            inputs=[self.engine.data.constr_active_mask.full, self.in_S, self.diag, self.Y],
            outputs=[P_inv_diag], device=dev,
        )

    def _solve_rows(self, t: wp.array) -> wp.array:
        """Returns s = E⁻¹ t."""
        wp.launch(_einv_apply_kernel, dim=(self.engine.dims.N_w, self.k), inputs=[self.Einv, t], outputs=[self.s],
                  device=self.engine.device)
        return self.s

    def project(self, v: wp.array, y: wp.array, z: wp.array, alpha: float, beta: float):
        """z = beta*y + alpha*P v with P v = v − AZ E⁻¹ v_S."""
        dims, dev = self.engine.dims, self.engine.device
        mask = self.engine.data.constr_active_mask.full
        wp.launch(_gather_rows_kernel, dim=(dims.N_w, self.k), inputs=[self.rows, mask, v], outputs=[self.t], device=dev)
        s = self._solve_rows(self.t)
        wp.launch(
            _project_kernel, dim=(dims.N_w, dims.N_c),
            inputs=[mask, self.AZt, s, v, y, alpha, beta], outputs=[z], device=dev,
        )

    def project_rhs(self, b: wp.array, Pb: wp.array):
        """Pb = P b; keeps Zᵀ b for `recover`."""
        dims = self.engine.dims
        mask = self.engine.data.constr_active_mask.full
        wp.launch(_gather_rows_kernel, dim=(dims.N_w, self.k), inputs=[self.rows, mask, b], outputs=[self.tb],
                  device=self.engine.device)
        self.project(b, b, Pb, 1.0, 0.0)

    def recover(self, x: wp.array):
        """x ← Q b + Pᵀ x = x + Z E⁻¹ (Zᵀ b − (AZ)ᵀ x), in place."""
        dims, dev = self.engine.dims, self.engine.device
        wp.launch(_residual_rows_kernel, dim=(dims.N_w, self.k), inputs=[self.tb, self.AZt, x], outputs=[self.t],
                  device=dev)
        s = self._solve_rows(self.t)
        wp.launch(
            _scatter_add_rows_kernel, dim=(dims.N_w, self.k),
            inputs=[self.rows, self.engine.data.constr_active_mask.full, s], outputs=[x], device=dev,
        )


class DeflatedSystemOperator(SystemOperator):
    """P A = A − AZ E⁻¹ (AZ)ᵀ, applied as A x followed by the projection."""

    def __init__(self, base: SystemOperator, deflation: RowDeflation):
        super().__init__(data=base.data, device=base.device, regularization=base.regularization)
        self.deflation = deflation
        self._Ax = wp.zeros((base.data.N_w, base.data.N_c), dtype=wp.float32, device=base.device)

    def matvec(self, x, y, z, alpha, beta):
        super().matvec(x, x, self._Ax, 1.0, 0.0)
        self.deflation.project(self._Ax, y, z, alpha, beta)


class DeflationPreconditioner(JacobiPreconditioner):
    """Updates the deflation for the current system, then the Jacobi diagonal:
    diag(A_F) of the Schur complement (schur=True) or diag(A)."""

    def __init__(self, engine, regularization, deflation: RowDeflation, schur: bool):
        super().__init__(engine, regularization)
        self.deflation = deflation
        self.schur = schur

    def update(self, refactor: bool = True):
        self.deflation.update(refactor=refactor)
        if self.schur:
            self.deflation.schur_inv_diag(self._P_inv_diag)
        else:
            super().update()


class DeflatedPCRSolver(PCRSolver):
    """PCR on P A x̃ = P b, then x = Q b + Pᵀ x̃.

    Same call signature as PCRSolver.solve. The engine passes its plain
    SystemOperator; the deflated operator built on it is used for the
    iterations. The stopping test stays relative to ‖b‖, not ‖P b‖; PCR's own
    residual equals the true residual b − A x because P A = A Pᵀ.
    """

    def __init__(self, base_op: SystemOperator, deflation: RowDeflation, max_iters, batch_dim, vec_dim, device):
        super().__init__(max_iters=max_iters, batch_dim=batch_dim, vec_dim=vec_dim, device=device)
        self.base_op = base_op
        self.deflation = deflation
        self.op = DeflatedSystemOperator(base_op, deflation)
        with wp.ScopedDevice(device):
            self.Pb = wp.zeros((batch_dim, vec_dim), dtype=wp.float32)
            self.b_ref_sq = wp.zeros((batch_dim,), dtype=wp.float32)

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
        assert A is self.base_op, "DeflatedPCRSolver deflates the operator it was built on"
        self.tiled_dot.compute(b, b, self.b_ref_sq)
        self.deflation.project_rhs(b, self.Pb)
        super().solve(self.op, self.Pb, x, preconditioner, iters, tol, atol, log, b_ref_sq=self.b_ref_sq)
        self.deflation.recover(x)
