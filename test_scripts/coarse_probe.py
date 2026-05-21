"""Coverage + stability probe for the per-body-wrench coarse space.

The whole coarse-space direction rests on two empirical claims:

  (A) COVERAGE — the per-body wrench space  range(Rᵀ), R = body-blocked
      Jᵀ, already spans the bad ~16-dim band of the Jacobi-scaled A.
  (B) STABILITY — that space is far more stable across the 85%-volatile
      A than A's own eigenvectors are (so it's a learnable/usable fixed
      object, unlike per-system eigenvectors).

And the punchline, with zero learning:

  (C) does the ANALYTIC additive two-level preconditioner
      M⁻¹ = diag(A)⁻¹ + Z (ZᵀAZ)⁻¹ Zᵀ ,  Z = Jm  (= Rᵀ)
      already beat Jacobi on iters-to-tol?

Run:
    python test_scripts/coarse_probe.py
"""

from __future__ import annotations

import numpy as np
import torch

from test_scripts.precond_lab import MAX_ITERS, load_systems, pcr, jacobi_apply

DT = torch.float64


def coarse_basis_scaled(s):
    """Orthonormal basis Qc (n×r_eff) of the per-body coarse space in the
    Jacobi-scaled coordinates, plus effective rank."""
    D = torch.diagonal(s.A).clamp_min(1e-30).rsqrt()      # diag(A)^-1/2
    Zs = D.unsqueeze(1) * s.Jm                              # (n, 6Nb) scaled
    U, sv, _ = torch.linalg.svd(Zs, full_matrices=False)
    tol = sv.max() * 1e-8
    r = int((sv > tol).sum())
    return U[:, :r], r, D


def two_level_apply(s, reg=1e-10):
    """M⁻¹ = diag(A)⁻¹ + Z (ZᵀAZ + εI)⁻¹ Zᵀ  (additive, SPD)."""
    A, Z = s.A, s.Jm
    jac = 1.0 / torch.diagonal(A).clamp_min(1e-30)
    G = Z.transpose(0, 1) @ (A @ Z)                         # (6Nb, 6Nb)
    G = G + reg * (torch.trace(G) / G.shape[0] + 1e-30) * torch.eye(
        G.shape[0], dtype=A.dtype)
    try:
        L = torch.linalg.cholesky(G)
        solve = lambda y: torch.cholesky_solve(y.unsqueeze(1), L).squeeze(1)
    except torch.linalg.LinAlgError:
        Gp = torch.linalg.pinv(G)
        solve = lambda y: Gp @ y
    return lambda r: jac * r + Z @ solve(Z.transpose(0, 1) @ r)


def pct(a, p):
    return np.percentile(np.asarray(a, float), p)


def main():
    S = load_systems(dtype=DT)
    print(f"loaded {len(S)} systems  (N_b={S[0].N_b}, "
          f"coarse dim = 6·N_b = {6*S[0].N_b})\n")

    r_eff, cov8, cov16 = [], [], []
    Gs, ns = [], []
    it_jac, it_2lvl = [], []

    for s in S:
        Qc, r, D = coarse_basis_scaled(s)
        r_eff.append(r)

        As = D.unsqueeze(1) * s.A * D.unsqueeze(0)
        w, V = torch.linalg.eigh(As)                        # ascending
        for kk, store in ((8, cov8), (16, cov16)):
            k = min(kk, s.n)
            Vk = V[:, :k]
            P = Qc.transpose(0, 1) @ Vk                     # (r, k)
            store.append(float((P * P).sum() / k))          # mean captured energy

        Gs.append((s.Jm.transpose(0, 1) @ (s.A @ s.Jm)))
        ns.append(s.n)

        _, ij = pcr(s.A, s.b, jacobi_apply(s.A))
        _, i2 = pcr(s.A, s.b, two_level_apply(s))
        it_jac.append(ij)
        it_2lvl.append(i2)

    # (A) coverage
    print("(A) COVERAGE — fraction of the bad eigen-band spanned by the "
          "per-body coarse space:")
    print(f"  effective coarse rank: p50 {pct(r_eff,50):.0f}  "
          f"min {min(r_eff)}  max {max(r_eff)}  (cap {6*S[0].N_b})")
    print(f"  bottom-8  eigenspace covered: p50 {pct(cov8,50):.2f}  "
          f"p05 {pct(cov8,5):.2f}")
    print(f"  bottom-16 eigenspace covered: p50 {pct(cov16,50):.2f}  "
          f"p05 {pct(cov16,5):.2f}")

    # (B) stability — relative Frobenius change between consecutive systems.
    # G is always (6Nb,6Nb) so comparable even when the active set (n)
    # changes; A is only compared on equal-n consecutive pairs.
    dG, dA = [], []
    for i in range(len(S) - 1):
        g0, g1 = Gs[i], Gs[i + 1]
        dG.append(float(torch.linalg.norm(g1 - g0) /
                        (torch.linalg.norm(g0) + 1e-30)))
        if ns[i] == ns[i + 1]:
            a0, a1 = S[i].A, S[i + 1].A
            dA.append(float(torch.linalg.norm(a1 - a0) /
                            (torch.linalg.norm(a0) + 1e-30)))
    print("\n(B) STABILITY — relative change between consecutive systems:")
    print(f"  coarse operator G=ZᵀAZ:  p50 {pct(dG,50):.2f}  "
          f"p95 {pct(dG,95):.2f}")
    print(f"  full A (equal-n pairs):  p50 {pct(dA,50):.2f}  "
          f"p95 {pct(dA,95):.2f}  ({len(dA)} pairs)  "
          f"[docs report ~0.85 within-step]")

    # (C) the punchline
    ij, i2 = np.array(it_jac, float), np.array(it_2lvl, float)
    print("\n(C) ANALYTIC two-level vs Jacobi — iters-to-tol (no learning):")
    print(f"  jacobi      mean {ij.mean():5.1f}  p50 {pct(ij,50):4.0f}  "
          f"p95 {pct(ij,95):4.0f}")
    print(f"  two-level   mean {i2.mean():5.1f}  p50 {pct(i2,50):4.0f}  "
          f"p95 {pct(i2,95):4.0f}  "
          f"unconv {int((i2>=MAX_ITERS).sum())}/{len(i2)}")
    win = int((i2 < ij).sum())
    print(f"  two-level beats Jacobi on {win}/{len(S)} systems; "
          f"mean {ij.mean():.1f} → {i2.mean():.1f} "
          f"({100*(1-i2.mean()/ij.mean()):+.0f}%)")

    print("\nverdict:")
    if pct(cov16, 50) > 0.8 and pct(dG, 50) < pct(dA, 50) / 2 and i2.mean() < ij.mean() * 0.85:
        print("  COVERAGE high, coarse op far more STABLE than A, and the "
              "analytic two-level already beats Jacobi → coarse-space "
              "direction validated; learned enrichment is the upside, not "
              "the necessity. Build the rig.")
    else:
        print("  Mixed/negative — read the three blocks: low coverage ⇒ "
              "body space misses the band (need enrichment / different "
              "coarse space); unstable G ⇒ premise wrong; two-level ≈ "
              "Jacobi ⇒ rigid-body modes are not the bottleneck.")


if __name__ == "__main__":
    main()
