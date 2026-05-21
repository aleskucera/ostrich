"""The completion of the analytic coarse space (last offline lever).

Finding 13: null(Jᵀ) deflation gives +17% but covers only ~30% of the
bad band — the missing ~70% is the stiff joint/control coupling, which
lives in range(Jᵀ) (joints produce body wrench) and is the fixed,
always-active constraint block [0:offset_n).

Enriched analytic coarse space — still NO eigensolve / learning /
recycling, just an SVD of Jm + fixed standard-basis columns:

    Z = orthonormalise([ null(Jᵀ) basis  ‖  e_r for active joint/control r ])

Ablations: null-only, joint-only, enriched; vs Jacobi (bar) and the
exact-eigvec ideal (ceiling). Reports coverage of the bad bottom-16
band and the effective Z dim (cost ∝ this).

Run:
    python test_scripts/coarse_enriched_probe.py
"""

from __future__ import annotations

import numpy as np
import torch

from test_scripts.precond_lab import DATA, MAX_ITERS, load_systems, pcr, jacobi_apply
from test_scripts.ideal_target_stability import bad_subspace, overlap
from test_scripts.recycle_probe import two_level
from test_scripts.coarse_null_probe import null_jt_basis

DT = torch.float64


def orthonormal(cols):
    """QR of a possibly rank-deficient (n×k) stack → orthonormal basis of
    its column space (drops vanishing columns)."""
    if cols.shape[1] == 0:
        return cols
    Q, R = torch.linalg.qr(cols)
    keep = torch.abs(torch.diagonal(R)) > 1e-9 * (torch.abs(R).max() + 1e-30)
    return Q[:, keep]


def pct(a, p):
    return float(np.percentile(np.asarray(a, float), p)) if len(a) else float("nan")


def main():
    S = load_systems(dtype=DT)
    d = np.load(DATA, allow_pickle=True)
    meta = d["meta"][0]
    off_n = int(meta["offset_n"])           # joint+control occupy [0, off_n)
    mask = d["constr_active_mask"]
    acts = [np.nonzero(mask[s, 0] > 0.0)[0] for s in range(len(S))]
    cut = int(0.8 * len(S))
    va = list(range(cut, len(S)))
    print(f"{len(S)} systems  val {len(va)}  "
          f"(joint/control = global constraint idx < {off_n})\n")

    dims = {"null": [], "joint": [], "enriched": []}
    cov = {"null": [], "joint": [], "enriched": []}
    iters = {"jacobi": [], "null": [], "joint": [], "enriched": [], "ideal": []}

    for i, s in enumerate(S):
        A, b, n = s.A, s.b, s.n
        eye = torch.eye(n, dtype=DT)

        Znull, _ = null_jt_basis(s.Jm)                  # (n, n-rank)
        jc_rows = [r for r, g in enumerate(acts[i]) if g < off_n]
        Ijc = eye[:, jc_rows] if jc_rows else eye[:, :0]  # (n, n_jc)

        Zn = orthonormal(Znull)
        Zj = orthonormal(Ijc)
        Ze = orthonormal(torch.cat([Znull, Ijc], dim=1))

        Qbad = bad_subspace(A, min(16, n))
        for nm, Z in (("null", Zn), ("joint", Zj), ("enriched", Ze)):
            dims[nm].append(Z.shape[1])
            k = min(16, n)
            cov[nm].append(overlap(Z, Qbad[:, :k]) if Z.shape[1] >= k
                           else overlap(Qbad[:, :k][:, :Z.shape[1]], Z)
                           if Z.shape[1] > 0 else 0.0)

        if i in va:
            _, ij = pcr(A, b, jacobi_apply(A)); iters["jacobi"].append(ij)
            _, ii = pcr(A, b, two_level(A, Qbad)); iters["ideal"].append(ii)
            for nm, Z in (("null", Zn), ("joint", Zj), ("enriched", Ze)):
                _, it = pcr(A, b, two_level(A, Z)) if Z.shape[1] > 0 \
                    else pcr(A, b, jacobi_apply(A))
                iters[nm].append(it)

    print("coverage of bad bottom-16 band  |  effective Z dim (cost ∝):")
    for nm in ("null", "joint", "enriched"):
        print(f"  {nm:9s} cov p50 {pct(cov[nm],50):.2f} p05 {pct(cov[nm],5):.2f}"
              f"   dim p50 {pct(dims[nm],50):.0f} max {max(dims[nm])}")

    def row(t):
        a = np.asarray(iters[t], float)
        return (f"  {t:22s} mean {a.mean():5.1f}  p50 {pct(a,50):4.0f}  "
                f"p95 {pct(a,95):4.0f}  unconv {int((a>=MAX_ITERS).sum())}"
                f"/{len(a)}")

    jm = np.mean(iters["jacobi"])
    print("\nVAL iters-to-tol:")
    for t in ("jacobi", "null", "joint", "enriched", "ideal"):
        print(row(t))
    em = np.mean(iters["enriched"])
    print(f"\n  null   : {jm:.1f}→{np.mean(iters['null']):.1f} "
          f"({100*(1-np.mean(iters['null'])/jm):+.0f}%)")
    print(f"  enriched: {jm:.1f}→{em:.1f} ({100*(1-em/jm):+.0f}%)  "
          f"vs ideal {np.mean(iters['ideal']):.1f}")

    print("\nverdict:")
    ce = pct(cov["enriched"], 50)
    im = np.mean(iters["ideal"])
    if ce > 0.8 and em < jm * 0.7:
        print(f"  enrichment WORKS: coverage {pct(cov['null'],50):.2f}→{ce:.2f}, "
              f"iters {jm:.1f}→{em:.1f} (≈ ideal {im:.1f}), fully analytic "
              f"(SVD of Jm + fixed cols). This is the structured win — only "
              f"remaining question is the per-NR Jm-SVD cost vs real PCR "
              f"(in-engine A/B; same gate as everything else).")
    elif em < np.mean(iters["null"]) - 0.5:
        print(f"  enrichment helps beyond null-only "
              f"({np.mean(iters['null']):.1f}→{em:.1f}) but stays short of "
              f"the ideal {im:.1f} — joint block recovers part of the "
              f"missing band, not all. Modest analytic win, cost gate "
              f"still applies.")
    else:
        print(f"  enrichment does NOT improve on null-only "
              f"(enriched {em:.1f} vs null {np.mean(iters['null']):.1f}); the "
              f"joint-block directions are not the missing piece — analytic "
              f"coarse space caps at the null(Jᵀ) +17%.")


if __name__ == "__main__":
    main()
