"""Two cheap, decisive probes (Finding 6 follow-ups):

(a) CROSS-SUPPORT — embed every system's bottom-k deflation subspace in
    the fixed 402-slot constraint space and ask: is the bad subspace
    SHARED across different active supports, or support-specific? Plus:
    does a single GLOBAL rank-R basis (SVD over all systems) cover each
    system's bad subspace? If yes → a *fixed universal* deflation space
    works (no learning, no recycling).

(b) RECYCLING — non-learned baseline: precondition system i with the
    PREVIOUS system's bottom-k subspace (additive two-level). Variants:
    recycle-always vs recycle-only-if-same-support (Finding 6's clean
    regime), against Jacobi (bar) and each system's own exact subspace
    (ceiling). If recycling alone beats Jacobi, no GNN/rig is needed.

Run:
    python test_scripts/recycle_probe.py
"""

from __future__ import annotations

import numpy as np
import torch

from test_scripts.precond_lab import DATA, MAX_ITERS, load_systems, pcr, jacobi_apply
from test_scripts.ideal_target_stability import bad_subspace, overlap

DT = torch.float64
K = 16
NC = 402  # full constraint-slot count (meta N_c)


def embed(U, act, ncols=NC):
    """Scatter an (n×k) subspace basis into the fixed (ncols×k) space."""
    E = torch.zeros(ncols, U.shape[1], dtype=U.dtype)
    E[torch.tensor(act, dtype=torch.long)] = U
    return E


def two_level(A, Z, reg=1e-10):
    """M⁻¹ = diag(A)⁻¹ + Z (ZᵀAZ+εI)⁻¹ Zᵀ  (additive, SPD)."""
    jac = 1.0 / torch.diagonal(A).clamp_min(1e-30)
    if Z.shape[1] == 0:
        return lambda r: jac * r
    G = Z.transpose(0, 1) @ (A @ Z)
    G = G + reg * (torch.trace(G) / G.shape[0] + 1e-30) * torch.eye(
        G.shape[0], dtype=A.dtype)
    try:
        L = torch.linalg.cholesky(G)
        solve = lambda y: torch.cholesky_solve(y.unsqueeze(1), L).squeeze(1)
    except torch.linalg.LinAlgError:
        Gp = torch.linalg.pinv(G)
        solve = lambda y: Gp @ y
    return lambda r: jac * r + Z @ solve(Z.transpose(0, 1) @ r)


def restrict(E, act):
    """Take a (402×k) embedded basis to current support, re-orthonormalise.
    Drops columns that vanish on the new support."""
    sub = E[torch.tensor(act, dtype=torch.long)]          # (n, k)
    Q, R = torch.linalg.qr(sub)
    keep = torch.abs(torch.diagonal(R)) > 1e-8 * (torch.abs(R).max() + 1e-30)
    return Q[:, keep]


def pct(a, p):
    return float(np.percentile(np.asarray(a, float), p)) if len(a) else float("nan")


def main():
    S = load_systems(dtype=DT)
    d = np.load(DATA, allow_pickle=True)
    mask = d["constr_active_mask"]
    acts = [np.nonzero(mask[s, 0] > 0.0)[0] for s in range(mask.shape[0])]
    assert len(acts) == len(S)

    Q = [bad_subspace(S[i].A, min(K, S[i].n)) for i in range(len(S))]
    E = [embed(Q[i], acts[i]) for i in range(len(S))]      # 402×K, padded
    sig = [tuple(a.tolist()) for a in acts]

    # ---- (a) cross-support ------------------------------------------------
    rng = np.random.default_rng(0)
    same_ov, diff_ov = [], []
    for _ in range(4000):
        i, j = rng.choice(len(S), 2, replace=False)
        ov = overlap(E[i], E[j])  # padded → directly comparable
        (same_ov if sig[i] == sig[j] else diff_ov).append(ov)

    # single global rank-R basis via SVD over all padded subspaces
    M = torch.cat(E, dim=1)                                 # 402 × (S·K)
    Ug, sv, _ = torch.linalg.svd(M, full_matrices=False)
    glob_cov = {}
    for R in (16, 32, 64):
        GR = Ug[:, :R]
        c = [float((( GR.transpose(0,1) @ E[i] )**2).sum() / E[i].shape[1])
             for i in range(len(S))]
        glob_cov[R] = c

    print("(a) CROSS-SUPPORT — bottom-16 subspace overlap (402-embedded):")
    print(f"  same-support pairs : p50 {pct(same_ov,50):.3f}  "
          f"(sanity, ≈1 per Finding 6)")
    print(f"  DIFFERENT-support  : p50 {pct(diff_ov,50):.3f}  "
          f"p05 {pct(diff_ov,5):.3f}  p95 {pct(diff_ov,95):.3f}")
    print("  single GLOBAL rank-R basis — mean energy of each system's "
          "bottom-16 it captures:")
    for R, c in glob_cov.items():
        print(f"    R={R:2d}: p50 {pct(c,50):.2f}  p05 {pct(c,5):.2f}")

    # ---- (b) recycling ----------------------------------------------------
    it_jac, it_rec, it_recss, it_ideal = [], [], [], []
    for i in range(len(S)):
        A, b = S[i].A, S[i].b
        _, ij = pcr(A, b, jacobi_apply(A)); it_jac.append(ij)
        _, ii = pcr(A, b, two_level(A, Q[i])); it_ideal.append(ii)

        if i == 0:
            it_rec.append(ij); it_recss.append(ij); continue
        Zr = restrict(E[i - 1], acts[i])                    # prev subspace → now
        _, ir = pcr(A, b, two_level(A, Zr)); it_rec.append(ir)
        if sig[i] == sig[i - 1]:
            it_recss.append(ir)
        else:
            it_recss.append(ij)                             # fall back to Jacobi

    def row(tag, a):
        a = np.asarray(a, float)
        return (f"  {tag:22s} mean {a.mean():5.1f}  p50 {pct(a,50):4.0f}  "
                f"p95 {pct(a,95):4.0f}  unconv {int((a>=MAX_ITERS).sum())}/{len(a)}")

    ij = np.array(it_jac, float)
    print("\n(b) RECYCLING — iters-to-tol (no learning):")
    print(row("jacobi (bar)", it_jac))
    print(row("recycle-always", it_rec))
    print(row("recycle-if-same-supp", it_recss))
    print(row("ideal (own subspace)", it_ideal))
    for tag, a in (("recycle-always", it_rec), ("recycle-if-same-supp", it_recss)):
        a = np.array(a, float)
        w = int((a < ij).sum())
        print(f"  → {tag}: beats Jacobi on {w}/{len(S)}, "
              f"mean {ij.mean():.1f}→{a.mean():.1f} "
              f"({100*(1-a.mean()/ij.mean()):+.0f}%)")

    print("\nverdict:")
    gc = pct(glob_cov[32], 50)
    rss = np.array(it_recss, float).mean()
    if gc > 0.85:
        print(f"  a single global rank-32 basis covers p50 {gc:.2f} of every "
              f"system's bad band → a FIXED universal deflation space works; "
              f"no learning, no recycling needed. Cheapest possible win.")
    elif rss < ij.mean() * 0.85:
        print(f"  same-support recycling beats Jacobi "
              f"({ij.mean():.1f}→{rss:.1f}) with zero ML → a small recycling "
              f"change is the win; GNN/rig unnecessary.")
    elif pct(diff_ov, 50) > 0.6:
        print(f"  cross-support overlap p50 {pct(diff_ov,50):.2f}: subspace "
              f"partly shared but not enough for a fixed/recycled space → a "
              f"support-conditioned learned predictor (the rig) is the "
              f"remaining lever worth trying.")
    else:
        print(f"  cross-support overlap p50 {pct(diff_ov,50):.2f} and "
              f"recycling≈Jacobi: the stable subspace is support-specific "
              f"and support changes dominate → learned preconditioning is "
              f"capped. Stop; PCR-doc Option 5.")


if __name__ == "__main__":
    main()
