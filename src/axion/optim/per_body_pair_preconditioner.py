"""Per-body-pair block-Jacobi preconditioner for the Schur-complement
matrix A = J·M⁻¹·Jᵀ + C.

NOTATION
--------
Throughout this file:
  * `n_bodies` is the per-world body count.
  * The "ground sentinel" is the integer value `n_bodies` (used in
    place of -1 for static / world-frame bodies). It encodes "no
    movable body on this side of the constraint".
  * A *pair* is identified by a canonical (b_lo, b_hi) tuple where
    b_lo ≤ b_hi and either is in [0, n_bodies] (with `n_bodies`
    being the ground sentinel). Encoded as a single int `pair_id`.
  * Members of a pair are the active constraints whose two body
    slots match (b_lo, b_hi) (in either order). Member count per
    pair is small (typically 3–8 for Helhest).

Design rationale: see `docs/pcr_warm_start_options.md` (section
"Five block schemes considered"). The 3×3 per-contact block-Jacobi
fails because J_n ⊥ {J_t1, J_t2} by construction makes the within-
contact 3×3 block of A approximately diagonal. The off-diagonal
coupling that diagonal Jacobi misses lives between *different
contacts that share a body* — for a wheel with 4 ground contacts,
M_wheel⁻¹ couples those 4 contacts.

Per-body-pair grouping handles this cleanly:
  * each constraint has exactly one pair (b₁, b₂) — no overlap, no
    ownership-rule arbitrariness;
  * the per-pair block A_pair captures both bodies' contributions to
    A_ij within the pair;
  * the only coupling missed is cross-pair through a shared body
    (e.g., (chassis, wheel-1) <-> (chassis, wheel-2) joint
    constraints share chassis), which we measured to be small for
    Helhest's heavy chassis.

If per-body-pair turns out marginal, the natural escalation is
Schur-on-constraint-types (invert the joint block A_jj exactly,
Jacobi on the contact-block Schur complement) — see PCR doc.

This file is the new home for everything specific to this
preconditioner so it can be reverted cleanly if it doesn't pan out.
The existing JacobiPreconditioner in `preconditioner.py` is
untouched.

PHASE 1 (this file): pair-assignment only. Computes which pair each
active constraint belongs to and builds (pair_id → constraint_list)
member lists. No matvec yet; that's phase 4.
"""
import warp as wp
from axion.mechanics import compute_spatial_momentum
from warp.optim.linear import LinearOperator


# Maximum number of constraints we expect in any one pair group. With
# cluster contact reduction (max_per_pair=8) plus a few friction +
# joint slots, the largest realistic pair has ~24 constraints. Set
# the bound generously to absorb future growth; overflow is silently
# clipped in phase 1's atomic-insertion path and detected at phase 2.
MAX_MEMBERS_PER_PAIR = 64


@wp.kernel
def _compute_pair_ids_kernel(
    constr_body_idx: wp.array(dtype=wp.int32, ndim=3),     # (W, N_c, 2)
    constr_active_mask: wp.array(dtype=wp.float32, ndim=2),  # (W, N_c)
    n_bodies: int,
    # Output: per-constraint pair id, or -1 if inactive.
    constr_pair_id: wp.array(dtype=wp.int32, ndim=2),  # (W, N_c)
):
    """Compute a canonical pair id for each constraint.

    Encoding: pair_id = b_lo · (n_bodies + 1) + b_hi
    where (b_lo, b_hi) = sorted(b₁, b₂) with -1 mapped to n_bodies
    (the "ground sentinel"). Symmetric: pair_id is the same regardless
    of which body sits in slot 0 vs slot 1 of constr_body_idx.

    Inactive constraints get pair_id = -1.

    Range of pair_id for active constraints:
        0 ≤ pair_id < (n_bodies + 1)²
    For Helhest with n_bodies=16, this is 0..288.
    """
    w, c = wp.tid()
    if c >= constr_body_idx.shape[1]:
        return

    if constr_active_mask[w, c] == 0.0:
        constr_pair_id[w, c] = -1
        return

    b1 = constr_body_idx[w, c, 0]
    b2 = constr_body_idx[w, c, 1]

    # Map -1 (no body / static reference) to n_bodies (ground sentinel).
    e1 = b1
    if b1 < 0:
        e1 = n_bodies
    e2 = b2
    if b2 < 0:
        e2 = n_bodies

    # Canonicalize: sort so smaller body id is first.
    b_lo = e1
    b_hi = e2
    if e2 < e1:
        b_lo = e2
        b_hi = e1

    constr_pair_id[w, c] = b_lo * (n_bodies + 1) + b_hi


@wp.kernel
def _build_pair_member_lists_kernel(
    constr_pair_id: wp.array(dtype=wp.int32, ndim=2),  # (W, N_c)
    # Outputs:
    pair_member_count: wp.array(dtype=wp.int32, ndim=2),     # (W, n_pairs_max)
    pair_member_list: wp.array(dtype=wp.int32, ndim=3),       # (W, n_pairs_max, MAX_MEMBERS)
    pair_overflow_flag: wp.array(dtype=wp.int32, ndim=1),     # (W,)
):
    """Populate pair → constraint member lists by scanning every
    constraint and atomically inserting it into its pair's list.

    Overflow protection: if a pair has more constraints than
    MAX_MEMBERS_PER_PAIR, we silently clip the insertion and flag
    the world via pair_overflow_flag[w] = 1. Phase 2 callers should
    sync-and-check this flag before assuming the lists are complete.
    """
    w, c = wp.tid()
    if c >= constr_pair_id.shape[1]:
        return

    pid = constr_pair_id[w, c]
    if pid < 0:
        return

    # Atomically claim a slot in this pair's member list. The slot
    # value returned is the *previous* count, i.e. our slot index.
    slot = wp.atomic_add(pair_member_count, w, pid, 1)
    if slot < pair_member_list.shape[2]:
        pair_member_list[w, pid, slot] = c
    else:
        # Overflow — slot index is past our buffer. Flag the world.
        # (Multiple overflowing constraints will all set the flag to
        # 1; that's fine, the flag is just a "did anything overflow"
        # indicator.)
        pair_overflow_flag[w] = 1


@wp.func
def _jacobian_for_body(
    target_body: int,
    constr_idx: int,
    world_idx: int,
    constr_body_idx: wp.array(dtype=wp.int32, ndim=3),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    n_bodies: int,
):
    """Return the spatial-vector Jacobian of constraint `constr_idx`
    on `target_body`, looking up which slot (0 or 1) of J_values has
    the matching body.

    Convention: target_body = n_bodies means "ground sentinel" — the
    static reference, which has no Jacobian entry. We return
    a zero spatial vector in that case (the caller skips the ground
    sentinel before calling this anyway, but a defensive zero return
    avoids garbage if invoked).

    For real bodies (target_body ∈ [0, n_bodies)), exactly one of
    constr_body_idx[w, c, 0] / [w, c, 1] equals target_body for any
    constraint that's a member of the pair containing target_body.
    """
    if target_body >= n_bodies:
        return wp.spatial_vector()

    body_0 = constr_body_idx[world_idx, constr_idx, 0]
    body_1 = constr_body_idx[world_idx, constr_idx, 1]

    if body_0 == target_body:
        return J_values[world_idx, constr_idx, 0]
    if body_1 == target_body:
        return J_values[world_idx, constr_idx, 1]
    # Should not reach here for valid pair memberships, but return
    # zero defensively.
    return wp.spatial_vector()


@wp.kernel
def _extract_pair_blocks_kernel(
    pair_member_count: wp.array(dtype=wp.int32, ndim=2),       # (W, n_pairs_max)
    pair_member_list: wp.array(dtype=wp.int32, ndim=3),         # (W, n_pairs_max, MAX)
    constr_body_idx: wp.array(dtype=wp.int32, ndim=3),          # (W, N_c, 2)
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),        # (W, N_c, 2)
    body_inv_mass: wp.array(dtype=wp.float32, ndim=2),          # (W, N_b)
    world_inv_inertia: wp.array(dtype=wp.mat33, ndim=2),        # (W, N_b)
    C_values: wp.array(dtype=wp.float32, ndim=2),               # (W, N_c)
    n_bodies: int,
    regularization: float,
    # Output: dense (W, n_pairs_max, MAX, MAX) block storage. Empty
    # pair slots and entries past the pair's member count remain at
    # whatever was previously written (usually zero from the per-step
    # reset). Phase 3 factorization scans only the live (i, j) range.
    A_blocks: wp.array(dtype=wp.float32, ndim=4),
):
    """Build A_pair[i, j] = J_i^T · M⁻¹ · J_j (summed over both bodies
    in the pair) for every (pair, i, j) cell.

    Threads outside the live (i, j) range for their pair early-exit so
    A_blocks remains at the per-step reset value (0). Inside the live
    range we add C_ii on the diagonal plus a small `regularization`
    floor for numerical stability of the per-block factorization.
    """
    w, p, i, j = wp.tid()

    n_members = pair_member_count[w, p]
    if i >= n_members or j >= n_members:
        return

    # Decode pair_id back to (b_lo, b_hi). With pair_id = b_lo·(n_b+1) + b_hi,
    # the inverse is straightforward integer arithmetic.
    stride = n_bodies + 1
    b_hi = p % stride
    b_lo = p / stride

    c_i = pair_member_list[w, p, i]
    c_j = pair_member_list[w, p, j]

    val = float(0.0)

    # Body b_lo's contribution. By construction b_lo ≤ b_hi, and the
    # ground sentinel sorts to b_hi. So if b_lo equals n_bodies the
    # pair would be (ground, ground), which can't occur for an active
    # constraint. Skip is safe via the `< n_bodies` gate.
    if b_lo < n_bodies:
        J_i_lo = _jacobian_for_body(
            b_lo, c_i, w, constr_body_idx, J_values, n_bodies
        )
        J_j_lo = _jacobian_for_body(
            b_lo, c_j, w, constr_body_idx, J_values, n_bodies
        )
        m_inv = body_inv_mass[w, b_lo]
        I_inv = world_inv_inertia[w, b_lo]
        Mv = compute_spatial_momentum(m_inv, I_inv, J_j_lo)
        val += wp.dot(J_i_lo, Mv)

    # Body b_hi's contribution (skip the ground sentinel).
    if b_hi < n_bodies:
        J_i_hi = _jacobian_for_body(
            b_hi, c_i, w, constr_body_idx, J_values, n_bodies
        )
        J_j_hi = _jacobian_for_body(
            b_hi, c_j, w, constr_body_idx, J_values, n_bodies
        )
        m_inv = body_inv_mass[w, b_hi]
        I_inv = world_inv_inertia[w, b_hi]
        Mv = compute_spatial_momentum(m_inv, I_inv, J_j_hi)
        val += wp.dot(J_i_hi, Mv)

    # Diagonal: add the constraint's own compliance entry plus a small
    # numerical regularization to keep A_pair safely SPD for the
    # per-block factorization in phase 3.
    if i == j:
        val += C_values[w, c_i] + regularization

    A_blocks[w, p, i, j] = val


@wp.kernel
def _factor_pair_blocks_kernel(
    A_blocks: wp.array(dtype=wp.float32, ndim=4),         # (W, n_pairs_max, MAX, MAX)
    pair_member_count: wp.array(dtype=wp.int32, ndim=2),  # (W, n_pairs_max)
    # Outputs:
    L_blocks: wp.array(dtype=wp.float32, ndim=4),         # (W, n_pairs_max, MAX, MAX)
    factor_failure: wp.array(dtype=wp.int32, ndim=2),     # (W, n_pairs_max)
):
    """Per-pair lower-triangular Cholesky factorization, A_pair = L·Lᵀ.

    Threading: one thread per (world, pair_id). Each thread runs the
    serial column-Cholesky algorithm on its block. Block sizes are
    small in practice (3-15 for Helhest) so per-thread cost is
    bounded; the parallelism across (world × n_active_pairs) keeps
    the GPU busy. Threads on inactive pairs (member count 0) early-
    exit immediately.

    Algorithm (standard column Cholesky):
        for k = 0..n-1:
            L[k,k] = sqrt(A[k,k] - sum_{q<k} L[k,q]²)
            for i = k+1..n-1:
                L[i,k] = (A[i,k] - sum_{q<k} L[i,q]·L[k,q]) / L[k,k]
            L[k, k+1..n-1] = 0   (upper triangle stays at the per-step zero reset)

    On non-SPD input (diag ≤ 0 during factorization), set
    factor_failure[w, p] = 1 and return without writing further.
    Phase 4's apply must check this flag and either fall back to
    Jacobi for failed blocks or treat the failure as a fatal error.
    """
    w, p = wp.tid()
    n = pair_member_count[w, p]
    if n == 0:
        return

    for k in range(n):
        # Diagonal: L[k,k] = sqrt(A[k,k] - Σ_{q<k} L[k,q]²)
        diag_sum = float(0.0)
        for q in range(k):
            l_kq = L_blocks[w, p, k, q]
            diag_sum += l_kq * l_kq

        diag_val = A_blocks[w, p, k, k] - diag_sum
        if diag_val <= 0.0:
            factor_failure[w, p] = 1
            return
        L_kk = wp.sqrt(diag_val)
        L_blocks[w, p, k, k] = L_kk
        inv_L_kk = 1.0 / L_kk

        # Off-diagonal column k: L[i,k] = (A[i,k] - Σ_{q<k} L[i,q]·L[k,q]) / L[k,k]
        for i in range(k + 1, n):
            row_sum = float(0.0)
            for q in range(k):
                row_sum += L_blocks[w, p, i, q] * L_blocks[w, p, k, q]
            L_blocks[w, p, i, k] = (A_blocks[w, p, i, k] - row_sum) * inv_L_kk


@wp.kernel
def _apply_baseline_kernel(
    vec_y: wp.array(dtype=wp.float32, ndim=2),
    beta: float,
    out_z: wp.array(dtype=wp.float32, ndim=2),
):
    """First pass of matvec: out_z[w, c] = beta · y[w, c].

    Sets a clean baseline for every entry, including constraints not
    touched by any active pair. The per-pair pass then ADDS the
    alpha·M⁻¹·x contribution on top for active-pair entries.
    """
    w, c = wp.tid()
    if c >= vec_y.shape[1]:
        return
    out_z[w, c] = beta * vec_y[w, c]


@wp.kernel
def _apply_per_pair_kernel(
    L_blocks: wp.array(dtype=wp.float32, ndim=4),         # (W, n_pairs_max, MAX, MAX)
    pair_member_count: wp.array(dtype=wp.int32, ndim=2),  # (W, n_pairs_max)
    pair_member_list: wp.array(dtype=wp.int32, ndim=3),    # (W, n_pairs_max, MAX)
    factor_failure: wp.array(dtype=wp.int32, ndim=2),      # (W, n_pairs_max)
    P_inv_diag: wp.array(dtype=wp.float32, ndim=2),        # (W, N_c) - Jacobi fallback
    vec_x: wp.array(dtype=wp.float32, ndim=2),
    alpha: float,
    workspace: wp.array(dtype=wp.float32, ndim=3),         # (W, n_pairs_max, MAX)
    # Output (read-modify-write — caller filled with beta·y first):
    out_z: wp.array(dtype=wp.float32, ndim=2),
):
    """Per (world, pair_id), apply the per-pair preconditioner block:
    solve A_pair · z = x_pair via the cached Cholesky factor, then
    add alpha·z to out_z at the constraint indices in this pair.

    For pairs whose Cholesky failed (factor_failure[w, p] == 1), fall
    back to per-element Jacobi (using P_inv_diag from the existing
    JacobiPreconditioner) for that pair's constraints — graceful
    degradation when a block was numerically non-SPD.

    `workspace` is a per-(world, pair) scratch buffer used to hold
    x_pair and the intermediate y from the forward solve. Size MAX
    per pair; reused across calls (no clearing needed since the
    in-pair loop overwrites every used slot).
    """
    w, p = wp.tid()
    n = pair_member_count[w, p]
    if n == 0:
        return

    # ---- Failure path: Jacobi fallback for this block ----
    if factor_failure[w, p] != 0:
        for k in range(n):
            ci = pair_member_list[w, p, k]
            inv_diag = P_inv_diag[w, ci]
            out_z[w, ci] += alpha * inv_diag * vec_x[w, ci]
        return

    # ---- Normal path: triangular solve against cached L ----

    # Step 1: gather x_pair into workspace
    for k in range(n):
        ci = pair_member_list[w, p, k]
        workspace[w, p, k] = vec_x[w, ci]

    # Step 2: forward solve L·y = x_pair, in-place on workspace
    # L is lower-triangular: y[i] = (x[i] - Σ_{j<i} L[i,j]·y[j]) / L[i,i]
    for i in range(n):
        s = workspace[w, p, i]
        for j in range(i):
            s -= L_blocks[w, p, i, j] * workspace[w, p, j]
        workspace[w, p, i] = s / L_blocks[w, p, i, i]

    # Step 3: back solve Lᵀ·z = y, in-place on workspace
    # Lᵀ is upper-triangular: z[i] = (y[i] - Σ_{j>i} L[j,i]·z[j]) / L[i,i]
    for ii in range(n):
        i = n - 1 - ii  # iterate from n-1 down to 0
        s = workspace[w, p, i]
        for j in range(i + 1, n):
            s -= L_blocks[w, p, j, i] * workspace[w, p, j]
        workspace[w, p, i] = s / L_blocks[w, p, i, i]

    # Step 4: scatter z to out_z, scaled by alpha and added on top of
    # the beta·y baseline already there.
    for k in range(n):
        ci = pair_member_list[w, p, k]
        out_z[w, ci] += alpha * workspace[w, p, k]


class PerBodyPairPreconditioner(LinearOperator):
    """Block-Jacobi preconditioner with per-body-pair blocks.

    Phase 1: pair-assignment only. Subsequent phases will add block
    extraction (phase 2), per-pair dense factorization (phase 3),
    and matvec (phase 4) — at which point the class will subclass
    `warp.optim.linear.LinearOperator` and become a drop-in
    replacement for `JacobiPreconditioner`.

    Lifecycle:
        precond = PerBodyPairPreconditioner(engine)
        # Per simulation step (or when active constraint set changes):
        precond.update_pair_assignments()
        # Phases 2+ will add an .update() (block factorization) and
        # a .matvec() (apply preconditioner).
    """

    def __init__(self, engine, regularization: float = 1e-6):
        super().__init__(
            shape=(engine.dims.N_w, engine.dims.N_c, engine.dims.N_c),
            dtype=wp.float32,
            device=engine.device,
            matvec=None,  # bound to self.matvec below
        )
        self.engine = engine
        self.regularization = float(regularization)

        N_w = engine.dims.num_worlds
        N_c = engine.dims.num_constraints
        N_b = engine.dims.body_count

        # Pair encoding: ground sentinel = N_b, so max pair_id is
        # (N_b+1)² - 1. Storage cost grows quadratically with body
        # count; for N_b=16 → 289 pair slots, comfortably small.
        self.n_pairs_max = (N_b + 1) * (N_b + 1)
        self.n_bodies = N_b
        self.MAX_MEMBERS_PER_PAIR = MAX_MEMBERS_PER_PAIR

        with wp.ScopedDevice(self.device):
            # Per-constraint pair id (or -1 for inactive).
            self.constr_pair_id = wp.zeros((N_w, N_c), dtype=wp.int32)
            # Per-pair member count and member list.
            self.pair_member_count = wp.zeros(
                (N_w, self.n_pairs_max), dtype=wp.int32
            )
            self.pair_member_list = wp.zeros(
                (N_w, self.n_pairs_max, self.MAX_MEMBERS_PER_PAIR),
                dtype=wp.int32,
            )
            self.pair_overflow_flag = wp.zeros((N_w,), dtype=wp.int32)

            # Phase 2: dense (W, n_pairs_max, MAX, MAX) block storage.
            # Most slots are inactive (zero member count) and stay
            # zero; only ~7-20 pairs are live per step on Helhest.
            # Per-world cost: n_pairs_max · MAX² · 4 bytes ≈ 4.7 MB
            # for N_b=16, MAX=64. Tolerable; phase 3 may add a
            # compaction step if scaling to many worlds.
            self.A_blocks = wp.zeros(
                (N_w, self.n_pairs_max,
                 self.MAX_MEMBERS_PER_PAIR,
                 self.MAX_MEMBERS_PER_PAIR),
                dtype=wp.float32,
            )

            # Phase 3: lower-triangular Cholesky factor of each
            # active A_pair block. Same shape as A_blocks. Upper
            # triangle stays at the per-step zero reset; phase 4's
            # apply only reads the lower triangle anyway.
            self.L_blocks = wp.zeros(
                (N_w, self.n_pairs_max,
                 self.MAX_MEMBERS_PER_PAIR,
                 self.MAX_MEMBERS_PER_PAIR),
                dtype=wp.float32,
            )
            # Per-pair Cholesky failure flag. Set to 1 by the kernel
            # if a non-positive diagonal is hit during factorization
            # (i.e. A_pair was not SPD — likely from numerical issues
            # near impacts). Phase 4's apply must treat failed blocks
            # as a fall-back-to-Jacobi case.
            self.factor_failure = wp.zeros(
                (N_w, self.n_pairs_max), dtype=wp.int32
            )

            # Phase 4: triangular-solve workspace (one per pair).
            # Reused across solves; contents don't need clearing
            # because the per-pair kernel overwrites every used slot
            # before reading it.
            self._workspace = wp.zeros(
                (N_w, self.n_pairs_max, self.MAX_MEMBERS_PER_PAIR),
                dtype=wp.float32,
            )

            # Phase 4: Jacobi-fallback inv-diag, used when a Cholesky
            # block fails. Computed alongside the per-pair factor in
            # `update()`; reuses the existing Jacobi diag formula
            # from `axion.optim.preconditioner.compute_inv_diag_kernel`
            # without bringing in the full JacobiPreconditioner.
            self._P_inv_diag_fallback = wp.zeros(
                (N_w, N_c), dtype=wp.float32
            )

    def update_pair_assignments(self):
        """Recompute pair IDs and member lists.

        Should be called whenever the active constraint set changes —
        in practice once per simulation step (after collision detection
        + reduction). The per-NR-iter J_values do change inside a step,
        but the pair assignment itself only depends on which constraint
        indices are active and which bodies they touch, both of which
        are static within a step.
        """
        # Reset: zero member counts, clear overflow flag. The per-
        # constraint pair_id is fully overwritten by the kernel so no
        # need to clear it. The pair_member_list values are stale for
        # empty slots but won't be read (pair_member_count gates how
        # far we scan).
        self.pair_member_count.zero_()
        self.pair_overflow_flag.zero_()

        N_w = self.engine.dims.num_worlds
        N_c = self.engine.dims.num_constraints

        wp.launch(
            kernel=_compute_pair_ids_kernel,
            dim=(N_w, N_c),
            inputs=[
                self.engine.data._constr_body_idx,
                self.engine.data._constr_active_mask,
                self.n_bodies,
            ],
            outputs=[self.constr_pair_id],
            device=self.device,
        )

        wp.launch(
            kernel=_build_pair_member_lists_kernel,
            dim=(N_w, N_c),
            inputs=[self.constr_pair_id],
            outputs=[
                self.pair_member_count,
                self.pair_member_list,
                self.pair_overflow_flag,
            ],
            device=self.device,
        )

    def update_blocks(self):
        """Phase 2: extract A_pair blocks from the current J_values,
        body inertias, and C compliance entries.

        Must be called every NR iter (or whenever J_values changes),
        after `update_pair_assignments()` has populated the pair member
        lists for the current step.

        Resets `A_blocks` to zero before populating so empty pair slots
        and (i, j) entries past the live member range are deterministic.
        """
        # Reset so empty slots and out-of-range cells stay zero.
        # Phase 4's apply will scan only the live range, but a clean
        # slate here makes inspection / validation cleaner.
        self.A_blocks.zero_()

        N_w = self.engine.dims.num_worlds

        wp.launch(
            kernel=_extract_pair_blocks_kernel,
            dim=(
                N_w,
                self.n_pairs_max,
                self.MAX_MEMBERS_PER_PAIR,
                self.MAX_MEMBERS_PER_PAIR,
            ),
            inputs=[
                self.pair_member_count,
                self.pair_member_list,
                self.engine.data._constr_body_idx,
                self.engine.data._J_values,
                self.engine.axion_model.body_inv_mass,
                self.engine.data.world_inv_inertia,
                self.engine.data._C_values,
                self.n_bodies,
                self.regularization,
            ],
            outputs=[self.A_blocks],
            device=self.device,
        )

    def factor_blocks(self):
        """Phase 3: Cholesky-factor each active A_pair block.

        Must be called after `update_blocks()`. Together they form
        the full per-NR-iter setup pass; phase 4's apply then
        triangular-solves against `L_blocks` to apply the
        preconditioner.

        Resets `L_blocks` and `factor_failure` to zero before the
        kernel runs. After return, `factor_failure[w, p] == 1` if
        block (w, p)'s Cholesky hit a non-positive diagonal — typical
        signal of numerical SPD violation near impacts. Phase 4 must
        check this and fall back to plain Jacobi for failed blocks.
        """
        self.L_blocks.zero_()
        self.factor_failure.zero_()

        wp.launch(
            kernel=_factor_pair_blocks_kernel,
            dim=(self.engine.dims.num_worlds, self.n_pairs_max),
            inputs=[self.A_blocks, self.pair_member_count],
            outputs=[self.L_blocks, self.factor_failure],
            device=self.device,
        )

    def update(self):
        """Drop-in replacement for `JacobiPreconditioner.update()`.

        Performs the full per-NR-iter setup: pair assignment (cheap
        but only needs to run once per step in principle — see
        comment), block extraction, Cholesky factorization, and
        Jacobi-fallback inv-diag computation.

        For now this re-runs `update_pair_assignments()` every NR iter
        too. The pair structure is constant within a step (member
        constraints don't change between NR iters), so this is wasted
        work. Phase 5 (or a follow-up) can hoist the assignment to
        once-per-step from base_engine.load_data, leaving only
        update_blocks + factor_blocks here.
        """
        self.update_pair_assignments()
        self.update_blocks()
        self.factor_blocks()
        self._update_jacobi_fallback()

    def _update_jacobi_fallback(self):
        """Compute the Jacobi inv-diag used as fallback when a
        per-pair Cholesky fails. Reuses the existing
        `compute_inv_diag_kernel` from preconditioner.py rather than
        duplicating the formula here.
        """
        # Lazy import to keep cross-file deps tidy
        from axion.optim.preconditioner import compute_inv_diag_kernel

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
            outputs=[self._P_inv_diag_fallback],
            device=self.device,
        )

    def matvec(self, x, y, z, alpha, beta):
        """Apply the per-body-pair block-Jacobi preconditioner:
        `z = beta·y + alpha · M⁻¹·x`, where M is the block-diagonal
        approximation of A with blocks indexed by per-body-pair.

        Drop-in for `JacobiPreconditioner.matvec`. PCRSolver passes
        (x, y, z, alpha, beta) once per inner iter.
        """
        N_w = self.engine.dims.num_worlds
        N_c = self.engine.dims.num_constraints

        # Pass 1: out_z = beta · y, for every (world, constraint).
        # Sets baseline including constraints not in any active pair.
        wp.launch(
            kernel=_apply_baseline_kernel,
            dim=(N_w, N_c),
            inputs=[y, beta],
            outputs=[z],
            device=self.device,
        )

        # Pass 2: per-active-pair, accumulate alpha · M⁻¹·x onto z.
        wp.launch(
            kernel=_apply_per_pair_kernel,
            dim=(N_w, self.n_pairs_max),
            inputs=[
                self.L_blocks,
                self.pair_member_count,
                self.pair_member_list,
                self.factor_failure,
                self._P_inv_diag_fallback,
                x,
                alpha,
                self._workspace,
            ],
            outputs=[z],
            device=self.device,
        )

    # ---- Phase 1 introspection helpers (CPU-side, sync required) ----

    def num_active_pairs(self, world_idx: int = 0) -> int:
        """Number of distinct pairs with at least one active constraint
        in `world_idx`. Forces a sync — debug/diagnostic use only."""
        counts = self.pair_member_count.numpy()[world_idx]
        return int((counts > 0).sum())

    def active_pair_member_counts(self, world_idx: int = 0):
        """List of (pair_id, member_count) for each active pair."""
        counts = self.pair_member_count.numpy()[world_idx]
        nz = counts.nonzero()[0]
        return [(int(p), int(counts[p])) for p in nz]

    def overflowed(self, world_idx: int = 0) -> bool:
        """True iff any pair in `world_idx` exceeded MAX_MEMBERS_PER_PAIR."""
        return bool(self.pair_overflow_flag.numpy()[world_idx])

    def decode_pair_id(self, pair_id: int):
        """Decode a pair_id back into (b_lo, b_hi). The ground sentinel
        is `n_bodies`; -1 (no body) was mapped to that during encoding.
        Returns (-1, b_hi) if the pair includes the static reference."""
        b_lo = pair_id // (self.n_bodies + 1)
        b_hi = pair_id % (self.n_bodies + 1)
        if b_lo == self.n_bodies:
            b_lo = -1
        if b_hi == self.n_bodies:
            b_hi = -1
        return (b_lo, b_hi)
