"""Is the κ~1e11 of the Helhest PCR systems removable by a STATIC
(non-learned) diagonal rescaling, or is it genuinely non-local?

Decisive because:
  * Symmetric-Jacobi scaling  D = diag(A)^-1/2  →  D A D  has the same
    spectrum as the Jacobi-preconditioned CR we already measured
    (~20 iters). Sanity-checks the harness AND bounds "what diagonal
    preconditioning can ever do".
  * Ruiz equilibration uses full row scales, not just the diagonal. If
    Ruiz κ << Jacobi κ, a cheap classical static scaling beats Jacobi
    and the learned-operator need shrinks. If Ruiz ≈ Jacobi, the
    ill-conditioning is provably NOT diagonal-removable → it is
    structurally non-local → the low-rank / coarse learned operator is
    the right tool. Either way the answer changes the plan.

Run:
    python test_scripts/equilibration_diag.py
"""

from __future__ import annotations

import numpy as np
import torch

from test_scripts.precond_lab import MAX_ITERS, load_systems, pcr

DT = torch.float64


def sym_jacobi_scale(A):
    d = 1.0 / torch.sqrt(torch.diagonal(A).clamp_min(1e-30))
    return d[:, None] * A * d[None, :], d


def ruiz_equilibrate(A, iters=50):
    """Symmetric Ruiz equilibration with row 2-norms. Returns (S A S, S)
    where S is diagonal; keeps symmetry (same scale on rows and cols)."""
    n = A.shape[0]
    S = torch.ones(n, dtype=A.dtype, device=A.device)
    M = A.clone()
    for _ in range(iters):
        r = torch.sqrt(torch.linalg.norm(M, dim=1).clamp_min(1e-30))
        s = 1.0 / r
        M = s[:, None] * M * s[None, :]
        S = S * s
    return M, S


def identity_apply(_A):
    return lambda r: r


def pct(a, p):
    return np.percentile(np.asarray(a, float), p)


def cond_safe(A):
    try:
        return float(torch.linalg.cond(A))
    except Exception:
        return float("inf")


def main():
    S = load_systems(dtype=DT)
    print(f"loaded {len(S)} systems\n")

    k_raw, k_jac, k_ruiz = [], [], []
    it_jacprec, it_jacscale, it_ruiz = [], [], []

    for s in S:
        A, b = s.A, s.b

        As, d = sym_jacobi_scale(A)
        Ar, Sr = ruiz_equilibrate(A)

        k_raw.append(cond_safe(A))
        k_jac.append(cond_safe(As))
        k_ruiz.append(cond_safe(Ar))

        # Jacobi *preconditioned* PCR on raw A (what the engine does).
        inv = 1.0 / torch.diagonal(A).clamp_min(1e-30)
        _, ij = pcr(A, b, lambda r: inv * r)
        it_jacprec.append(ij)

        # Identity PCR on the symmetric-Jacobi-scaled system — should
        # match the line above (spectrum is identical). Sanity check.
        _, ijs = pcr(As, d * b, identity_apply(As))
        it_jacscale.append(ijs)

        # Identity PCR on the Ruiz-equilibrated system — does a better
        # static scaling beat Jacobi?
        _, ir = pcr(Ar, Sr * b, identity_apply(Ar))
        it_ruiz.append(ir)

    print("condition number κ(A):")
    for nm, k in (("raw", k_raw), ("sym-Jacobi scaled", k_jac),
                  ("Ruiz equilibrated", k_ruiz)):
        k = np.asarray(k, float)
        print(f"  {nm:20s} p50 {pct(k,50):.2e}  p95 {pct(k,95):.2e}  "
              f"max {k.max():.2e}")

    def itrow(nm, a):
        a = np.asarray(a, float)
        print(f"  {nm:24s} mean {a.mean():5.1f}  p50 {pct(a,50):5.1f}  "
              f"p95 {pct(a,95):5.1f}  unconv {int((a>=MAX_ITERS).sum())}/{len(a)}")

    print("\niters-to-tol:")
    itrow("Jacobi-precond (engine)", it_jacprec)
    itrow("sym-Jacobi-scaled (=^)", it_jacscale)
    itrow("Ruiz-equilibrated", it_ruiz)

    kr, kj, ku = np.array(k_raw), np.array(k_jac), np.array(k_ruiz)
    ij, ir = np.array(it_jacprec), np.array(it_ruiz)
    print("\nverdict:")
    print(f"  sym-Jacobi removes {np.log10(np.median(kr)/np.median(kj)):.1f} "
          f"orders of κ (median {np.median(kr):.1e} -> {np.median(kj):.1e})")
    print(f"  Ruiz removes       {np.log10(np.median(kr)/np.median(ku)):.1f} "
          f"orders of κ (median {np.median(kr):.1e} -> {np.median(ku):.1e})")
    gain = 100 * (1 - ir.mean() / ij.mean())
    if ku.mean() < kj.mean() / 10 and gain > 15:
        print(f"  -> Ruiz beats Jacobi by {gain:.0f}% iters: a CHEAP STATIC "
              f"scaling is most of the win. Reframe before learned operators.")
    else:
        print(f"  -> Ruiz ≈ Jacobi ({ir.mean():.1f} vs {ij.mean():.1f} iters): "
              f"the residual κ is NOT diagonal-removable → structurally "
              f"non-local → low-rank/coarse learned operator is the right tool.")


if __name__ == "__main__":
    main()
