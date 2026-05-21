"""DECISIVE probe: is the ideal target (the bad eigen-subspace) a stable
object across systems, or is it as volatile as A itself?

This bounds what ANY learned preconditioner can do, independent of
architecture/target. A learned predictor maps cheap features → the
deflation/coarse subspace. If that subspace rotates ~fully between
consecutive solves (as A does, ~62-85%/step), then no fixed-feature
predictor can track it — learned preconditioning is fundamentally
capped here. If it stays coherent while A moves a lot, a learnable
object exists and a rig is justified.

Method. The operationally meaningful target is the bottom-k deflation
subspace in *original* constraint space: span(D · v) where v are the
smallest eigenvectors of the Jacobi-scaled A and D = diag(A)^-1/2.
Subspaces live in ℝⁿ, so they're only comparable between systems with
the *identical active support*. We group systems by support and, within
each group, measure subspace overlap (mean cos²θ via principal angles)
for CONSECUTIVE pairs (the regime a warm/learned predictor operates in)
vs RANDOM pairs (temporal-coherence control), against the relative
change of A on the very same pairs.

overlap = ‖U1ᵀU2‖_F² / k  ∈ [0,1];  1 = identical subspace, 0 = orthogonal.

Run:
    python test_scripts/ideal_target_stability.py
"""

from __future__ import annotations

import numpy as np
import torch

from test_scripts.precond_lab import DATA, load_systems

DT = torch.float64
KS = (4, 8, 16)


def bad_subspace(A, k):
    """Orthonormal basis (n×k) of the bottom-k deflation subspace in
    ORIGINAL constraint space: span(D · v_scaled)."""
    D = torch.diagonal(A).clamp_min(1e-30).rsqrt()
    As = D.unsqueeze(1) * A * D.unsqueeze(0)
    _, V = torch.linalg.eigh(As)            # ascending
    Z = D.unsqueeze(1) * V[:, :k]           # map back to original space
    Q, _ = torch.linalg.qr(Z)               # re-orthonormalise
    return Q


def overlap(U1, U2):
    """mean cos²θ between equal-dim subspaces (1=same, 0=orthogonal)."""
    M = U1.transpose(0, 1) @ U2
    return float((M * M).sum() / U1.shape[1])


def pct(a, p):
    return float(np.percentile(np.asarray(a, float), p)) if len(a) else float("nan")


def main():
    S = load_systems(dtype=DT)
    d = np.load(DATA, allow_pickle=True)
    mask = d["constr_active_mask"]                       # (S,1,Nc)

    # Support signature per system (the tuple of active constraint ids).
    sig = []
    si = 0
    for s in range(mask.shape[0]):
        act = np.nonzero(mask[s, 0] > 0.0)[0]
        if len(act) == 0:
            continue
        sig.append(tuple(act.tolist()))
        si += 1
    assert len(sig) == len(S)

    # Group system-indices by identical support.
    groups: dict[tuple, list[int]] = {}
    for i, g in enumerate(sig):
        groups.setdefault(g, []).append(i)
    multi = [v for v in groups.values() if len(v) >= 2]
    print(f"{len(S)} systems, {len(groups)} distinct active supports; "
          f"{len(multi)} supports with ≥2 systems "
          f"({sum(len(v) for v in multi)} systems comparable)\n")

    rng = np.random.default_rng(0)
    for k in KS:
        cons_ov, rand_ov, dA = [], [], []
        for members in multi:
            Qs = [bad_subspace(S[i].A, min(k, S[i].n)) for i in members]
            # consecutive pairs (capture order ⇒ adjacent NR iters)
            for a in range(len(members) - 1):
                cons_ov.append(overlap(Qs[a], Qs[a + 1]))
                A0, A1 = S[members[a]].A, S[members[a + 1]].A
                dA.append(float(torch.linalg.norm(A1 - A0) /
                                (torch.linalg.norm(A0) + 1e-30)))
            # random non-adjacent pairs within the same support
            if len(members) >= 3:
                for _ in range(len(members)):
                    a, b = rng.choice(len(members), 2, replace=False)
                    rand_ov.append(overlap(Qs[a], Qs[b]))

        print(f"k={k:2d}:")
        print(f"  consecutive-pair subspace overlap : p50 {pct(cons_ov,50):.3f}"
              f"  p05 {pct(cons_ov,5):.3f}  (1=stable, 0=fully rotated)")
        print(f"  random-pair    subspace overlap   : p50 {pct(rand_ov,50):.3f}"
              f"   (temporal-coherence control)")
        print(f"  A relative change on same pairs   : p50 {pct(dA,50):.2f}"
              f"  p95 {pct(dA,95):.2f}\n")

    # Verdict keyed on k=16 (the band that sets iters per eigenspectrum probe).
    cons16 = []
    for members in multi:
        Qs = [bad_subspace(S[i].A, min(16, S[i].n)) for i in members]
        for a in range(len(members) - 1):
            cons16.append(overlap(Qs[a], Qs[a + 1]))
    m = pct(cons16, 50)
    print("verdict:")
    if m >= 0.85:
        print(f"  bottom-16 subspace overlap p50 {m:.2f} while A moves "
              f"~0.6+ → the bad subspace IS a stable object inside a "
              f"volatile A. A learnable target exists; a rig is justified "
              f"(target must be this subspace, not body-space).")
    elif m >= 0.6:
        print(f"  bottom-16 overlap p50 {m:.2f} — partially coherent. "
              f"Marginal: a predictor could track it only with strong "
              f"temporal conditioning. Borderline; weigh vs Option 5.")
    else:
        print(f"  bottom-16 overlap p50 {m:.2f} — the ideal subspace "
              f"rotates nearly as fast as A changes. NO fixed-feature "
              f"predictor can track it → learned preconditioning is "
              f"fundamentally capped here. Stop; use PCR-doc Option 5 "
              f"(Eisenstat–Walker).")


if __name__ == "__main__":
    main()
