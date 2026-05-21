"""Fair test of PROPER recycling (GCRO-DR style): the recycle subspace
is MAINTAINED and ACCUMULATED across the whole solve sequence, not
rebuilt from a single previous solve.

Per solve i (sequential, capture order, realistic online deployment):
  1. deflate with the carried recycle subspace U (restricted to support_i),
     additive two-level — measure iters-to-tol;
  2. harvest the m Krylov residual vectors the solve generates;
  3. update U  ← harmonic-Ritz_k over span([ U|_support_i , Krylov_i ]).
Step 3 is the accumulation: U integrates information across ALL prior
solves, so (since the bad subspace is stable, Findings 6/7) it should
converge toward the true subspace far better than the single-solve
harvest that failed in recycle_approx.py (m=16 → 2× worse than Jacobi).

Controlled comparison: identical m=16 Krylov budget as recycle_approx;
only difference = accumulation. Isolates whether accumulation rescues it.

Run:
    python test_scripts/recycle_gcrodr.py
"""

from __future__ import annotations

import numpy as np
import torch

from test_scripts.precond_lab import DATA, MAX_ITERS, load_systems, pcr, jacobi_apply
from test_scripts.ideal_target_stability import bad_subspace, overlap
from test_scripts.recycle_probe import embed, two_level, restrict
from test_scripts.recycle_approx import krylov_residual_basis, harmonic_ritz

DT = torch.float64
K = 16
M = 16  # Krylov harvest per solve — the realistic "free" budget


def pct(a, p):
    return float(np.percentile(np.asarray(a, float), p)) if len(a) else float("nan")


def main():
    S = load_systems(dtype=DT)
    d = np.load(DATA, allow_pickle=True)
    mask = d["constr_active_mask"]
    acts = [np.nonzero(mask[s, 0] > 0.0)[0] for s in range(mask.shape[0])]
    cut = int(0.8 * len(S))
    va = set(range(cut, len(S)))

    Qexact = [bad_subspace(S[i].A, min(K, S[i].n)) for i in range(len(S))]
    Eexact = [embed(Qexact[i], acts[i]) for i in range(len(S))]

    U = None  # carried recycle subspace, 402-padded
    it_gd, it_jac, it_rx, it_id = [], [], [], []
    uq = []   # accumulated-U quality: overlap vs exact bottom-16
    for i in range(len(S)):
        A, b = S[i].A, S[i].b
        D = torch.diagonal(A).clamp_min(1e-30).rsqrt()
        As = D.unsqueeze(1) * A * D.unsqueeze(0)

        # (1) deflated solve with carried subspace
        if U is None:
            Mfn = jacobi_apply(A)
        else:
            Zi = restrict(U, acts[i])
            Mfn = two_level(A, Zi)
        _, it = pcr(A, b, Mfn)

        # references (not part of the online method)
        _, ij = pcr(A, b, jacobi_apply(A))
        _, ii = pcr(A, b, two_level(A, Qexact[i]))
        _, ie = pcr(A, b, two_level(A, restrict(Eexact[i - 1], acts[i]))) if i > 0 \
            else pcr(A, b, jacobi_apply(A))
        it_gd.append(it); it_jac.append(ij); it_id.append(ii); it_rx.append(ie)

        # (2) harvest Krylov + (3) accumulate: harmonic-Ritz over [U , Vi]
        Vi = krylov_residual_basis(As, D * b, M)
        if U is not None:
            Zc = restrict(U, acts[i])                      # carried, in space i
            Cand, _ = torch.linalg.qr(torch.cat([Zc, Vi], dim=1))
        else:
            Cand = Vi
        Y = harmonic_ritz(As, Cand, K)
        Y, _ = torch.linalg.qr(Y)
        Yorig, _ = torch.linalg.qr(D.unsqueeze(1) * Y)     # → original space
        U = embed(Yorig, acts[i])
        kk = min(K, S[i].n, Yorig.shape[1])
        uq.append(overlap(Qexact[i][:, :kk], Yorig[:, :kk]))

    def seg(name, arr, idxs):
        a = np.array([arr[j] for j in idxs], float)
        return (f"  {name:22s} mean {a.mean():5.1f}  p50 {pct(a,50):4.0f}  "
                f"p95 {pct(a,95):4.0f}")

    vi = sorted(va)
    print(f"{len(S)} systems sequential; val = held-out post-impact "
          f"{len(vi)} (subspace has accumulated through the run)\n")
    print("VAL iters-to-tol:")
    print(seg("jacobi (bar)", it_jac, vi))
    print(seg("GCRO-DR accumulated", it_gd, vi))
    print(seg("recycle-exact (eigh)", it_rx, vi))
    print(seg("ideal (own exact)", it_id, vi))
    jm = np.array([it_jac[j] for j in vi], float).mean()
    gm = np.array([it_gd[j] for j in vi], float).mean()
    print(f"  → GCRO-DR: {jm:.1f}→{gm:.1f} ({100*(1-gm/jm):+.0f}%)")

    # accumulation working? subspace-quality + iters learning curve
    n = len(S)
    thirds = [range(0, n // 3), range(n // 3, 2 * n // 3), range(2 * n // 3, n)]
    print("\naccumulation curve (does U mature over the run?):")
    for lbl, rg in zip(("first⅓", "mid⅓", "last⅓"), thirds):
        q = np.array([uq[j] for j in rg]); itg = np.array([it_gd[j] for j in rg])
        print(f"  {lbl}: U⋂exact overlap p50 {pct(q,50):.2f}   "
              f"GCRO-DR iters mean {itg.mean():.1f}")

    print("\nverdict:")
    if gm < jm * 0.85:
        print(f"  accumulation RESCUES recycling: held-out {jm:.1f}→{gm:.1f} "
              f"with only m={M} free Krylov vecs/solve → proper GCRO-DR is "
              f"worth an in-engine prototype after all.")
    else:
        print(f"  accumulation does NOT rescue it (val {gm:.1f} vs Jacobi "
              f"{jm:.1f}). Even proper recycling can't track the subspace "
              f"from the free Krylov budget → recycling direction CLOSED. "
              f"Fall back to Eisenstat–Walker (PCR-doc Option 5).")


if __name__ == "__main__":
    main()
