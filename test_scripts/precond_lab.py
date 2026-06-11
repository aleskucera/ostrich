"""Offline PCR reference + baseline preconditioners for the learned-
preconditioner feasibility study.

Reconstructs the dense Schur-complement system A = J·M⁻¹·Jᵀ + (C+ε)·I
from the raw arrays dumped by `dump_linear_systems.py`, restricted to
each system's *active* constraint set (31–67 of 402 slots for Helhest —
dense is trivially cheap). Provides:

  * `load_systems` / `System`        — data access + dense reconstruction
  * `pcr`                            — PyTorch PCR, faithfully mirroring
                                       `ostrich/optim/pcr_solver.py`
  * `jacobi_apply`                   — 1/diag(A) preconditioner
  * `per_body_pair_apply`            — block-Cholesky per body-pair,
                                       mirroring per_body_pair_preconditioner.py

Run directly to (1) verify the reconstruction against the engine's own
PCR solution and (2) print iters-to-tol baselines:

    python test_scripts/precond_lab.py
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import torch

DATA = pathlib.Path(__file__).resolve().parent.parent / "data/baselines/helhest_systems.npz"

# PCR tolerances — match LinearSolverConfig defaults (engine_config.py:134).
TOL = 1e-5
ATOL = 1e-5
MAX_ITERS = 64  # generous cap so we see true iters-to-tol, not the engine's 26


@dataclass
class System:
    """One reconstructed linear system, reduced to its active set."""

    A: torch.Tensor          # (n, n) dense SPD
    b: torch.Tensor          # (n,)
    x_engine: torch.Tensor   # (n,) engine PCR solution (reference)
    pair_id: np.ndarray      # (n,) per-active-constraint body-pair id
    body_pair: np.ndarray    # (n, 2) the two body ids (ground sentinel = N_b)
    c_active: torch.Tensor   # (n,) C+reg diagonal contribution (graph feature)
    n: int
    Jm: torch.Tensor         # (n, 6*N_b) constraint→body wrench map (= Rᵀ)
    N_b: int                 # per-world body count


def _block_minv(m_inv: float, world_I_inv: np.ndarray) -> np.ndarray:
    """6x6 inverse spatial mass for one body. Layout [linear(3), angular(3)]
    per mass.py:compute_spatial_momentum (top=mass·v, bot=I·ω)."""
    M = np.zeros((6, 6), dtype=np.float64)
    M[0, 0] = M[1, 1] = M[2, 2] = m_inv
    M[3:6, 3:6] = world_I_inv
    return M


def load_systems(path=DATA, device="cpu", dtype=torch.float64) -> list[System]:
    d = np.load(path, allow_pickle=True)
    meta = d["meta"][0]
    N_b = int(meta["N_b"])
    ground = N_b  # sentinel: -1 (no/static body) maps to N_b

    J = d["J_values"]               # (S,1,Nc,2,6)
    bidx = d["constr_body_idx"]     # (S,1,Nc,2)
    mask = d["constr_active_mask"]  # (S,1,Nc)
    C = d["C_values"]               # (S,1,Nc)
    minv = d["body_inv_mass"]       # (S,1,Nb)
    Iinv = d["world_inv_inertia"]   # (S,1,Nb,3,3)
    b_all = d["b"]                  # (S,1,Nc)
    x_eng = d["x_engine"]           # (S,1,Nc)
    reg = d["regularization"]       # (S,)

    out: list[System] = []
    for s in range(J.shape[0]):
        act = np.nonzero(mask[s, 0] > 0.0)[0]
        n = len(act)
        if n == 0:
            continue

        # Per-body 6x6 inverse spatial mass blocks for this system.
        Mb = [_block_minv(float(minv[s, 0, b]), Iinv[s, 0, b]) for b in range(N_b)]

        # Dense J restricted to active rows, expanded over body DOFs:
        # Jm (n, 6*N_b). Each active constraint contributes its two
        # spatial Jacobian rows at its two body slots.
        Jm = np.zeros((n, 6 * N_b), dtype=np.float64)
        body_pair = np.full((n, 2), ground, dtype=np.int64)
        for r, c in enumerate(act):
            for slot in (0, 1):
                bdy = int(bidx[s, 0, c, slot])
                if bdy >= 0:
                    Jm[r, 6 * bdy:6 * bdy + 6] += J[s, 0, c, slot]
                    body_pair[r, slot] = bdy

        Minv = np.zeros((6 * N_b, 6 * N_b), dtype=np.float64)
        for b in range(N_b):
            Minv[6 * b:6 * b + 6, 6 * b:6 * b + 6] = Mb[b]

        cdiag = C[s, 0, act].astype(np.float64) + float(reg[s])
        A = Jm @ Minv @ Jm.T
        A[np.diag_indices(n)] += cdiag

        # Canonical body-pair id (mirrors _compute_pair_ids_kernel).
        lo = np.minimum(body_pair[:, 0], body_pair[:, 1])
        hi = np.maximum(body_pair[:, 0], body_pair[:, 1])
        pid = lo * (N_b + 1) + hi

        out.append(System(
            A=torch.tensor(A, device=device, dtype=dtype),
            b=torch.tensor(b_all[s, 0, act], device=device, dtype=dtype),
            x_engine=torch.tensor(x_eng[s, 0, act], device=device, dtype=dtype),
            pair_id=pid,
            body_pair=np.stack([lo, hi], axis=1),
            c_active=torch.tensor(cdiag, device=device, dtype=dtype),
            n=n,
            Jm=torch.tensor(Jm, device=device, dtype=dtype),
            N_b=N_b,
        ))
    return out


# --------------------------------------------------------------------------
# Preconditioners.  Each returns a closure r -> M⁻¹ r.
# --------------------------------------------------------------------------

def jacobi_apply(A: torch.Tensor):
    inv_diag = 1.0 / (torch.diagonal(A) + 1e-12)
    return lambda r: inv_diag * r


def per_body_pair_apply(A: torch.Tensor, pair_id: np.ndarray):
    """Block-Cholesky on the body-pair block-diagonal of A.

    The dense submatrix A[members][:,members] for a pair equals the Warp
    `_extract_pair_blocks` block exactly: within a pair (b_lo,b_hi) every
    member constraint's bodies are a subset of {b_lo,b_hi}, so A's full
    entry (a sum over shared bodies) reduces to the pair's two bodies.
    """
    groups: dict[int, list[int]] = {}
    for r, p in enumerate(pair_id):
        groups.setdefault(int(p), []).append(r)

    solvers = []
    for members in groups.values():
        idx = torch.tensor(members, device=A.device)
        blk = A[idx][:, idx]
        try:
            L = torch.linalg.cholesky(blk)
            solvers.append((idx, L))
        except torch.linalg.LinAlgError:
            # Non-SPD block: Jacobi fallback for these rows (mirrors the
            # factor_failure path in per_body_pair_preconditioner.py).
            solvers.append((idx, None))

    inv_diag = 1.0 / (torch.diagonal(A) + 1e-12)

    def apply(r: torch.Tensor) -> torch.Tensor:
        z = torch.zeros_like(r)
        for idx, L in solvers:
            if L is None:
                z[idx] = inv_diag[idx] * r[idx]
            else:
                z[idx] = torch.cholesky_solve(r[idx].unsqueeze(1), L).squeeze(1)
        return z

    return apply


# --------------------------------------------------------------------------
# PCR — faithful port of ostrich/optim/pcr_solver.py (the CR recurrence in
# solver_step + the init block).  Differentiable in the preconditioner.
# --------------------------------------------------------------------------

def pcr(A, b, M_apply, x0=None, max_iters=MAX_ITERS, tol=TOL, atol=ATOL,
        return_history=False):
    x = torch.zeros_like(b) if x0 is None else x0.clone()

    b_sq = torch.dot(b, b)
    target = max(atol * atol, float(tol * tol * b_sq))

    r = b - A @ x
    z = M_apply(r)
    Az = A @ z
    p, Ap = z.clone(), Az.clone()
    zAz = torch.dot(z, Az)

    r_sq = torch.dot(r, r)
    hist = [float(r_sq)]
    if float(r_sq) <= target:
        return (x, 0, hist) if return_history else (x, 0)

    it = 0
    for it in range(1, max_iters + 1):
        zAz_old = zAz
        y = M_apply(Ap)
        yAp = torch.dot(y, Ap)
        alpha = zAz_old / yAp if float(yAp) > 0.0 else zAz_old * 0.0

        x = x + alpha * p
        r = r - alpha * Ap
        z = z - alpha * y

        Az = A @ z
        zAz = torch.dot(z, Az)
        beta = zAz / zAz_old if float(zAz_old) > 0.0 else zAz * 0.0

        p = z + beta * p
        Ap = Az + beta * Ap

        r_sq = torch.dot(r, r)
        hist.append(float(r_sq))
        if float(r_sq) <= target:
            break

    return (x, it, hist) if return_history else (x, it)


# --------------------------------------------------------------------------
# Self-check + baseline report.
# --------------------------------------------------------------------------

def _summary(name, vals):
    a = np.asarray(vals, dtype=np.float64)
    print(f"  {name:18s} iters: mean {a.mean():5.1f}  p50 {np.percentile(a,50):5.1f}"
          f"  p95 {np.percentile(a,95):5.1f}  max {a.max():5.0f}"
          f"  | unconverged {int((a >= MAX_ITERS).sum())}/{len(a)}")


def main():
    syslist = load_systems()
    print(f"loaded {len(syslist)} systems  "
          f"(active n: min {min(s.n for s in syslist)} "
          f"max {max(s.n for s in syslist)})\n")

    # (1) Reconstruction fidelity: does the engine's PCR solution satisfy
    # our reconstructed A x ≈ b?  O(1) here would mean a wrong convention.
    rel = []
    for s in syslist:
        num = torch.linalg.norm(s.A @ s.x_engine - s.b)
        den = torch.linalg.norm(s.b) + 1e-30
        rel.append(float(num / den))
    rel = np.array(rel)
    print(f"reconstruction ||A x_engine - b|| / ||b||:")
    print(f"  p50 {np.percentile(rel,50):.2e}  p95 {np.percentile(rel,95):.2e}"
          f"  max {rel.max():.2e}  "
          f"(engine ran Jacobi-PCR capped at 26 iters, so ~1e-2..1e-5 expected)\n")

    # (2) Baselines: iters-to-tol with each preconditioner.
    jac, pbp = [], []
    for s in syslist:
        _, it_j = pcr(s.A, s.b, jacobi_apply(s.A))
        _, it_p = pcr(s.A, s.b, per_body_pair_apply(s.A, s.pair_id))
        jac.append(it_j)
        pbp.append(it_p)

    print("iters-to-tol over all systems:")
    _summary("jacobi", jac)
    _summary("per_body_pair", pbp)
    jac, pbp = np.array(jac), np.array(pbp)
    print(f"\nper_body_pair vs jacobi: mean {jac.mean():.1f} -> {pbp.mean():.1f} "
          f"({100*(1-pbp.mean()/jac.mean()):.0f}% fewer). "
          f"This per_body_pair curve is the bar the GNN must beat.")


if __name__ == "__main__":
    main()
