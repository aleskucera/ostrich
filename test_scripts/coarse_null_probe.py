"""The one principled analytic coarse space we never tested:
deflate null(Jᵀ) — the self-equilibrated / constraint-redundancy
subspace where Finding 5 located the bad modes.

Key identity: with A = Jm·M⁻¹·Jmᵀ + C, for x with Jmᵀx = 0,
    A x = C x   (the J·M⁻¹·Jᵀ part annihilates).
So on null(Jᵀ), A IS just the tiny compliance C — that is the origin
of κ~1e5. The principled two-level is therefore:
    M⁻¹ = diag(A)⁻¹  +  Z (ZᵀAZ)⁻¹ Zᵀ ,   Z = orthonormal basis of null(Jᵀ)
computed from Jm ALONE (SVD) — no eigensolve of A, no learning, no
recycling. None of the failure modes that killed everything else apply.

Reports, on held-out val (and all):
  * effective null dim (cost ∝ this);
  * coverage: does null(Jᵀ) actually span the bad bottom-16 band?
    (also exposes whether the stiff-joint part of the band is MISSED,
     since joints produce body wrench → live in range(Jᵀ), not null);
  * the A|null(Jᵀ) = C mechanism check;
  * iters-to-tol vs Jacobi (bar) and exact-eigvec two-level (ceiling).

Run:
    python test_scripts/coarse_null_probe.py
"""

from __future__ import annotations

import numpy as np
import torch

from test_scripts.precond_lab import MAX_ITERS, load_systems, pcr, jacobi_apply
from test_scripts.ideal_target_stability import bad_subspace, overlap
from test_scripts.recycle_probe import two_level

DT = torch.float64


def null_jt_basis(Jm):
    """Orthonormal basis of {x : Jmᵀ x = 0} = left-null space of Jm."""
    U, sv, _ = torch.linalg.svd(Jm, full_matrices=True)   # Jm: (n, 6Nb)
    r = int((sv > sv.max() * 1e-10).sum()) if sv.numel() else 0
    return U[:, r:], r                                     # (n, n-r), rank r


def pct(a, p):
    return float(np.percentile(np.asarray(a, float), p)) if len(a) else float("nan")


def main():
    S = load_systems(dtype=DT)
    cut = int(0.8 * len(S))
    va = list(range(cut, len(S)))
    print(f"{len(S)} systems  val (held-out post-impact) {len(va)}\n")

    nulldim, cov16, ac_resid = [], [], []
    it_jac, it_null, it_ideal = [], [], []
    for i, s in enumerate(S):
        A, b = s.A, s.b
        Z, r = null_jt_basis(s.Jm)
        nd = Z.shape[1]
        nulldim.append(nd)

        # coverage of the bad bottom-16 eigenband by null(Jᵀ)
        Qbad = bad_subspace(A, min(16, s.n))
        k = min(16, s.n, nd)
        cov16.append(overlap(Z[:, :max(k, 1)] if nd >= k else Z,
                             Qbad[:, :k]) if nd > 0 else 0.0)

        # A|null = C  mechanism check: ||Zᵀ(A−C)Z|| / ||ZᵀAZ||
        if nd > 0:
            C = torch.diag(s.c_active)
            G = Z.T @ A @ Z
            Gc = Z.T @ C @ Z
            ac_resid.append(float(torch.linalg.norm(G - Gc) /
                                  (torch.linalg.norm(G) + 1e-30)))

        if i in va:
            _, ij = pcr(A, b, jacobi_apply(A)); it_jac.append(ij)
            _, ii = pcr(A, b, two_level(A, Qbad)); it_ideal.append(ii)
            _, ic = pcr(A, b, two_level(A, Z)) if nd > 0 \
                else pcr(A, b, jacobi_apply(A)); it_null.append(ic)

    print(f"null(Jᵀ) dim (cost ∝ this): p50 {pct(nulldim,50):.0f}  "
          f"min {min(nulldim)}  max {max(nulldim)}  (n is 31–67)")
    print(f"coverage of bad bottom-16 by null(Jᵀ): p50 {pct(cov16,50):.2f}  "
          f"p05 {pct(cov16,5):.2f}  "
          f"(<1 ⇒ stiff-joint part of the band is missed)")
    print(f"A|null = C check  ||Zᵀ(A−C)Z||/||ZᵀAZ||: p50 "
          f"{pct(ac_resid,50):.1e}  (≈0 confirms the mechanism)\n")

    def row(t, a):
        a = np.asarray(a, float)
        return (f"  {t:24s} mean {a.mean():5.1f}  p50 {pct(a,50):4.0f}  "
                f"p95 {pct(a,95):4.0f}  unconv {int((a>=MAX_ITERS).sum())}"
                f"/{len(a)}")

    ij = np.array(it_jac, float)
    print("VAL iters-to-tol:")
    print(row("jacobi (bar)", it_jac))
    print(row("null(Jᵀ) two-level", it_null))
    print(row("exact-eigvec (ceiling)", it_ideal))
    nm = np.array(it_null, float).mean()
    print(f"  → null(Jᵀ): {ij.mean():.1f}→{nm:.1f} "
          f"({100*(1-nm/ij.mean()):+.0f}%), beats Jacobi "
          f"{int((np.array(it_null,float)<ij).sum())}/{len(va)}")

    print("\nverdict:")
    cov = pct(cov16, 50)
    if nm < ij.mean() * 0.85 and cov > 0.8:
        print(f"  null(Jᵀ) spans the band (cov {cov:.2f}) and the analytic "
              f"two-level beats Jacobi ({ij.mean():.1f}→{nm:.1f}) with NO "
              f"eigensolve/learning/recycling. Mechanism confirmed. This is "
              f"the structured win — cost is the per-NR SVD of Jm (∝ null "
              f"dim p50 {pct(nulldim,50):.0f}); needs an in-engine cost A/B.")
    elif cov <= 0.8:
        print(f"  null(Jᵀ) covers only {cov:.2f} of the bad band — the "
              f"stiff-joint modes live in range(Jᵀ) and are MISSED. The "
              f"redundancy story (Finding 5) is only half of it; a pure "
              f"null(Jᵀ) coarse space is structurally insufficient.")
    else:
        print(f"  null(Jᵀ) spans the band but the two-level still doesn't "
              f"beat Jacobi (val {nm:.1f} vs {ij.mean():.1f}) — closes the "
              f"structured-analytic direction definitively.")


if __name__ == "__main__":
    main()
