"""Galerkin (projection) deflated PCR — the method the deflation A/B tests.

Every earlier subspace probe (recycle_probe, recycle_approx, coarse_probe,
coarse_null_probe, ...) used the ADDITIVE two-level preconditioner
M⁻¹ = D⁻¹ + Z(ZᵀAZ)⁻¹Zᵀ. With an exact Z both forms reach the same
κ_eff, but with an imperfect Z the additive form can push eigenvalues
above the cluster (the "over-corrects" failure). Projection deflation
cannot: for ANY Z the spectrum of M⁻¹PA stays inside that of M⁻¹A.

    E = ZᵀAZ,  Q = Z E⁻¹ Zᵀ,  P = I − AQ
    solve  PA x̃ = Pb  with Jacobi(A)-PCR,  then  x = Qb + Pᵀx̃

b − Ax = Pb − PAx̃ exactly (PA = APᵀ), so PCR's own residual is the true
one; only the relative target must stay relative to ‖b‖, not ‖Pb‖.

Running this file does the step-1b sanity checks on both scenes:
Z empty ≡ Jacobi; exact bottom-k Z → deflation ≈ additive ≈ ceiling;
random Z → deflation not clearly worse than Jacobi (additive shown for
contrast); x vs dense solve (and vs x_engine on obstacle only — surface
x_engine is often unconverged at the engine's 16-iter cap).

Run:
    python test_scripts/deflation.py
"""

from __future__ import annotations

import numpy as np
import torch

from test_scripts.precond_lab import ATOL, DATA, MAX_ITERS, TOL, jacobi_apply, load_systems, pcr
from test_scripts.ideal_target_stability import bad_subspace
from test_scripts.recycle_probe import pct, two_level

DT = torch.float64
SCENES = {"obstacle": DATA, "surface": DATA.parent / "helhest_surface_systems.npz"}
RANDOM_SEEDS = (0, 1, 2)


def deflated_pcr(A, b, Z, M_apply, max_iters=MAX_ITERS, tol=TOL, atol=ATOL):
    """Projection-deflated PCR on the active-set system A x = b.

    Z (n×k) is any basis in the same active-set space as A (use
    recycle_probe.restrict for padded 402-slot bases); it is
    orthonormalised here and near-dependent columns are dropped.
    M_apply must precondition the ORIGINAL A (e.g. jacobi_apply(A)).

    Returns (x, iters, k): iters = PCR iterations (doc convention);
    k = columns kept = setup cost in A-products (to form AZ).
    """
    if Z.shape[1] == 0:
        x, it = pcr(A, b, M_apply, max_iters=max_iters, tol=tol, atol=atol)
        return x, it, 0

    U, s, _ = torch.linalg.svd(Z, full_matrices=False)
    Z = U[:, s > 1e-8 * s[0]]                       # orthonormal, rank-revealed

    AZ = A @ Z                                      # setup: k products with A
    L = torch.linalg.cholesky(Z.transpose(0, 1) @ AZ)

    def E_inv(Y):
        return torch.cholesky_solve(Y, L)

    PA = A - AZ @ E_inv(AZ.transpose(0, 1))
    Pb = b - AZ @ E_inv((Z.transpose(0, 1) @ b).unsqueeze(1)).squeeze(1)

    # pcr's target is tol·‖rhs‖; rescale so it means tol·‖b‖.
    b_norm, Pb_norm = torch.linalg.norm(b), torch.linalg.norm(Pb)
    tol_pb = float(tol * b_norm / Pb_norm) if float(Pb_norm) > 0.0 else tol
    xt, it = pcr(PA, Pb, M_apply, max_iters=max_iters, tol=tol_pb, atol=atol)

    # x = Qb + Pᵀx̃,  Pᵀx̃ = x̃ − Z E⁻¹ (AZ)ᵀ x̃   (A symmetric → no extra A-product)
    rhs = (Z.transpose(0, 1) @ b - AZ.transpose(0, 1) @ xt).unsqueeze(1)
    x = xt + Z @ E_inv(rhs).squeeze(1)
    return x, it, Z.shape[1]


def _row(name, its, jac_its, res_over_target, err):
    a, j = np.asarray(its), np.asarray(jac_its)
    print(f"  {name:24s} mean {a.mean():5.1f}  p50 {pct(a,50):4.0f}  p90 {pct(a,90):4.0f}"
          f"  max {a.max():3d}  unconv {int((a >= MAX_ITERS).sum()):3d}"
          f"  | vs Jacobi: wins {int((a < j).sum()):3d} losses {int((a > j).sum()):3d}"
          f"  | max res/target (converged) {max(r for r, i in zip(res_over_target, its) if i < MAX_ITERS):5.2f}"
          f"  err-vs-dense p50 {pct(err,50):.0e} max {max(err):.0e}")


def run_scene(label, path):
    S = load_systems(path=path, dtype=DT)
    print(f"\n===== {label}  ({len(S)} systems, active n {min(s.n for s in S)}–{max(s.n for s in S)}) =====")

    def solve_stats(s, x):
        r = float(torch.linalg.norm(s.b - s.A @ x))
        target = max(ATOL, TOL * float(torch.linalg.norm(s.b)))
        xd = torch.linalg.solve(s.A, s.b)
        return r / target, float(torch.linalg.norm(x - xd) / torch.linalg.norm(xd))

    jac_x, jac_its = [], []
    for s in S:
        x, it = pcr(s.A, s.b, jacobi_apply(s.A))
        jac_x.append(x)
        jac_its.append(it)

    # (1) Z empty must be bit-identical to the Jacobi baseline.
    same = sum(
        torch.equal(deflated_pcr(s.A, s.b, torch.zeros(s.n, 0, dtype=DT), jacobi_apply(s.A))[0], jac_x[i])
        for i, s in enumerate(S))
    print(f"(1) Z empty ≡ Jacobi: {same}/{len(S)} bit-identical")

    st = [solve_stats(s, jac_x[i]) for i, s in enumerate(S)]
    print("(2)-(4) iters-to-tol, both forms on the SAME Z:")
    _row("jacobi (bar)", jac_its, jac_its, [a for a, _ in st], [e for _, e in st])

    cases = [(f"exact k={k}", lambda s, _seed, k=k: bad_subspace(s.A, min(k, s.n)), (None,))
             for k in (8, 16)]
    cases.append(("random k=16 (3 seeds)",
                  lambda s, seed: torch.randn(s.n, 16, dtype=DT,
                                              generator=torch.Generator().manual_seed(seed)),
                  RANDOM_SEEDS))

    defl_x = None
    for name, make_Z, seeds in cases:
        res = {"deflation": ([], [], []), "additive": ([], [], [])}   # its, res/target, err
        jref = []
        for seed in seeds:
            for i, s in enumerate(S):
                Z = make_Z(s, seed)
                M = jacobi_apply(s.A)
                xd_, itd, _ = deflated_pcr(s.A, s.b, Z, M)
                xa, ita = pcr(s.A, s.b, two_level(s.A, Z))
                for key, x, it in (("deflation", xd_, itd), ("additive", xa, ita)):
                    a, e = solve_stats(s, x)
                    res[key][0].append(it)
                    res[key][1].append(a)
                    res[key][2].append(e)
                jref.append(jac_its[i])
                if name == "exact k=16":
                    defl_x = defl_x or []
                    defl_x.append(xd_)
        for key in ("deflation", "additive"):
            _row(f"{name} {key[:5]}", res[key][0], jref, res[key][1], res[key][2])

    # (4b) vs the engine's own stored solution — obstacle only (see docstring).
    if label == "obstacle":
        rel = lambda x, s: float(torch.linalg.norm(x - s.x_engine) / torch.linalg.norm(s.x_engine))
        dj = [rel(jac_x[i], s) for i, s in enumerate(S)]
        dd = [rel(defl_x[i], s) for i, s in enumerate(S)]
        print(f"(4b) ‖x − x_engine‖/‖x_engine‖ p50: jacobi {pct(dj,50):.1e}  "
              f"deflation(exact k=16) {pct(dd,50):.1e}  (x_engine is itself 16-iter-capped)")


def main():
    for label, path in SCENES.items():
        run_scene(label, path)


if __name__ == "__main__":
    main()
