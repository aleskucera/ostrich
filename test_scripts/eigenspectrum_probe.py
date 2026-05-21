"""Size the low-rank head: spectrum of the Jacobi-scaled Helhest systems
and the CEILING of an ideal rank-k spectral correction.

After symmetric-Jacobi scaling the residual κ ≈ 1e5 is structurally
non-local (equilibration_diag.py). If that κ is set by a *few* small
outlier eigenvalues, an ideal rank-k deflation that lifts the k smallest
eigenvalues to 1 leaves effective κ = λ_max / λ_{k+1}. Running PCR with
that exact spectral preconditioner gives the BEST a "Jacobi + learned
rank-k correction" could ever do — and the k where iters collapse is the
rank the GNN head should emit.

  M⁻¹_k = I + Σ_{i<k} (1/λ_i − 1) vᵢ vᵢᵀ      (in the Jacobi-scaled space)

k=0 is plain Jacobi-scaled (the baseline). We report median ideal-iters
vs k, the spectral gap (is there a natural k?), and how concentrated κ
is in the bottom modes.

Run:
    python test_scripts/eigenspectrum_probe.py
"""

from __future__ import annotations

import numpy as np
import torch

from test_scripts.precond_lab import MAX_ITERS, load_systems, pcr
from test_scripts.equilibration_diag import sym_jacobi_scale

DT = torch.float64
K_LIST = [0, 1, 2, 4, 8, 16, 32]


def spectral_precond(w, V, k):
    """M⁻¹ = I + Σ_{i<k} (1/λ_i − 1) vᵢ vᵢᵀ  (w ascending)."""
    if k == 0:
        return lambda r: r
    Vk = V[:, :k]
    coef = (1.0 / w[:k]) - 1.0          # (k,)
    def apply(r):
        return r + Vk @ (coef * (Vk.transpose(0, 1) @ r))
    return apply


def pct(a, p):
    return np.percentile(np.asarray(a, float), p)


def main():
    S = load_systems(dtype=DT)
    print(f"loaded {len(S)} systems\n")

    lam_min, lam_max, kappa = [], [], []
    # fraction of "bad" eigenvalues: count λ < λ_max * 1e-3
    n_bad = []
    gap_at = {k: [] for k in K_LIST if k > 0}     # λ_{k}/λ_{k-1} ascending
    iters_k = {k: [] for k in K_LIST}
    # how much of the inverse-energy (Σ 1/λ) sits in the bottom 8 modes
    frac_invenergy_bottom8 = []

    for s in S:
        As, d = sym_jacobi_scale(s.A)
        bs = d * s.b
        w, V = torch.linalg.eigh(As)        # ascending
        w = w.clamp_min(1e-30)

        lam_min.append(float(w[0]))
        lam_max.append(float(w[-1]))
        kappa.append(float(w[-1] / w[0]))
        n_bad.append(int((w < w[-1] * 1e-3).sum()))

        inv = 1.0 / w
        frac_invenergy_bottom8.append(float(inv[:8].sum() / inv.sum()))

        for k in K_LIST:
            if k > 0 and k < len(w):
                gap_at[k].append(float(w[k] / w[k - 1]))
            M = spectral_precond(w, V, min(k, len(w) - 1))
            _, it = pcr(As, bs, M)
            iters_k[k].append(it)

    print("Jacobi-scaled spectrum:")
    print(f"  λ_min   p50 {pct(lam_min,50):.2e}  p05 {pct(lam_min,5):.2e}")
    print(f"  λ_max   p50 {pct(lam_max,50):.2e}")
    print(f"  κ       p50 {pct(kappa,50):.2e}  p95 {pct(kappa,95):.2e}")
    print(f"  #eig < λ_max·1e-3 (bad modes): p50 {pct(n_bad,50):.0f}  "
          f"p95 {pct(n_bad,95):.0f}  max {max(n_bad)}")
    print(f"  Σ(1/λ) fraction in bottom-8 modes: p50 "
          f"{pct(frac_invenergy_bottom8,50):.2f} "
          f"(→ how concentrated the ill-conditioning is)\n")

    print("IDEAL rank-k spectral correction — iters-to-tol (the ceiling):")
    base = np.mean(iters_k[0])
    for k in K_LIST:
        a = np.asarray(iters_k[k], float)
        tag = " (= Jacobi-scaled baseline)" if k == 0 else ""
        red = "" if k == 0 else f"  ({100*(1-a.mean()/base):+.0f}% vs k=0)"
        print(f"  k={k:2d}  mean {a.mean():5.1f}  p50 {pct(a,50):4.0f}  "
              f"p95 {pct(a,95):4.0f}  unconv {int((a>=MAX_ITERS).sum())}"
              f"/{len(a)}{red}{tag}")

    # natural rank: first k where median iters <= 8 (≈ single-digit solve)
    knat = None
    for k in K_LIST:
        if pct(iters_k[k], 50) <= 8:
            knat = k
            break
    print()
    if knat is not None:
        print(f"verdict: ideal rank-{knat} correction reaches median ≤8 iters "
              f"(from {base:.0f}). The bad κ lives in ≲{knat} non-local modes "
              f"→ a learned rank-{knat} head has real ceiling. Worth building.")
    else:
        print(f"verdict: even rank-32 does not collapse iters "
              f"(median {pct(iters_k[32],50):.0f}). κ is NOT low-rank — "
              f"low-rank correction has weak ceiling; pivot to coarse-space.")


if __name__ == "__main__":
    main()
