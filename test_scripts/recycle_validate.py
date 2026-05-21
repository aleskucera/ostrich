"""Two confirmations before the no-ML recommendation.

(1) OUT-OF-SAMPLE global basis. Build the universal deflation basis from
    TRAIN systems only (early steps); evaluate coverage + iters-to-tol on
    held-out VAL systems (later / post-impact). Decides whether a single
    PRECOMPUTED fixed basis is real or in-sample luck. recycle-always
    (prev solve, not fit to data) is the robust comparator.

(2) BLOCK ENERGY. Where does the global subspace live? Energy per
    constraint-type block (joint / control / normal / friction). Confirms
    the Finding-5 contact-redundancy story vs the trivial "it's just the
    always-active joint rows" alternative.

Run:
    python test_scripts/recycle_validate.py
"""

from __future__ import annotations

import numpy as np
import torch

from test_scripts.precond_lab import DATA, MAX_ITERS, load_systems, pcr, jacobi_apply
from test_scripts.ideal_target_stability import bad_subspace
from test_scripts.recycle_probe import embed, two_level, restrict, K, NC

DT = torch.float64


def pct(a, p):
    return float(np.percentile(np.asarray(a, float), p)) if len(a) else float("nan")


def global_basis(Es, R):
    M = torch.cat(Es, dim=1)
    U, _, _ = torch.linalg.svd(M, full_matrices=False)
    return U[:, :R]


def main():
    S = load_systems(dtype=DT)
    d = np.load(DATA, allow_pickle=True)
    mask = d["constr_active_mask"]
    meta = d["meta"][0]
    acts = [np.nonzero(mask[s, 0] > 0.0)[0] for s in range(mask.shape[0])]

    Q = [bad_subspace(S[i].A, min(K, S[i].n)) for i in range(len(S))]
    E = [embed(Q[i], acts[i]) for i in range(len(S))]
    sig = [tuple(a.tolist()) for a in acts]

    cut = int(0.8 * len(S))
    tr, va = list(range(cut)), list(range(cut, len(S)))
    print(f"{len(S)} systems  train {len(tr)} (early)  "
          f"val {len(va)} (post-impact, held out)\n")

    # ---- (1) out-of-sample fixed basis -----------------------------------
    print("(1) OUT-OF-SAMPLE — basis built on TRAIN, evaluated on VAL:")
    for R in (16, 32, 64):
        GR = global_basis([E[i] for i in tr], R)
        cov = [float(((GR.transpose(0, 1) @ E[i]) ** 2).sum() / E[i].shape[1])
               for i in va]
        print(f"  R={R:2d}: val bottom-16 covered by TRAIN basis  "
              f"p50 {pct(cov,50):.2f}  p05 {pct(cov,5):.2f}")

    GR = global_basis([E[i] for i in tr], 32)
    it_jac, it_fix, it_rec, it_ideal = [], [], [], []
    for i in va:
        A, b = S[i].A, S[i].b
        _, ij = pcr(A, b, jacobi_apply(A)); it_jac.append(ij)
        Zf = restrict(GR, acts[i])
        _, ifx = pcr(A, b, two_level(A, Zf)); it_fix.append(ifx)
        _, ii = pcr(A, b, two_level(A, Q[i])); it_ideal.append(ii)
        Zr = restrict(E[i - 1], acts[i]) if i > 0 else Q[i]
        _, ir = pcr(A, b, two_level(A, Zr)); it_rec.append(ir)

    def row(t, a):
        a = np.asarray(a, float)
        return (f"  {t:24s} mean {a.mean():5.1f}  p50 {pct(a,50):4.0f}  "
                f"p95 {pct(a,95):4.0f}  unconv {int((a>=MAX_ITERS).sum())}/{len(a)}")

    ij = np.array(it_jac, float)
    print("\n  VAL iters-to-tol:")
    print(row("jacobi (bar)", it_jac))
    print(row("fixed-universal (R=32)", it_fix))
    print(row("recycle-always", it_rec))
    print(row("ideal (own subspace)", it_ideal))
    for t, a in (("fixed-universal", it_fix), ("recycle-always", it_rec)):
        a = np.array(a, float)
        print(f"  → {t}: {ij.mean():.1f}→{a.mean():.1f} "
              f"({100*(1-a.mean()/ij.mean()):+.0f}%), beats Jacobi "
              f"{int((a<ij).sum())}/{len(va)}")

    # ---- (2) block energy -------------------------------------------------
    oj, octrl = int(meta["offset_j"]), int(meta["offset_ctrl"])
    on, of, ncc = int(meta["offset_n"]), int(meta["offset_f"]), int(meta["N_c"])
    blocks = [("joint", oj, octrl), ("control", octrl, on),
              ("normal", on, of), ("friction", of, ncc)]
    Gall = global_basis(E, 32)                       # (402, 32)
    e = (Gall ** 2)                                   # energy per slot per vec
    tot = float(e.sum())
    print("\n(2) BLOCK ENERGY of the global rank-32 subspace:")
    for nm, a, b in blocks:
        frac = float(e[a:b].sum()) / tot
        print(f"  {nm:8s} slots[{a:3d}:{b:3d}] ({b-a:3d} slots): "
              f"{100*frac:5.1f}% of subspace energy")
    # always-active (joint+control) vs contact (normal+friction)
    aa = float(e[oj:on].sum()) / tot
    print(f"  always-active block [0:{on}] = {100*aa:.1f}%  | "
          f"contact/friction [{on}:{ncc}] = {100*(1-aa):.1f}%")

    print("\nverdict:")
    fix = np.array(it_fix, float).mean()
    rec = np.array(it_rec, float).mean()
    if fix < ij.mean() * 0.85:
        print(f"  PRECOMPUTED fixed basis generalises to held-out "
              f"post-impact ({ij.mean():.1f}→{fix:.1f}) → cheapest option "
              f"is a one-time SVD basis baked into the engine. No ML.")
    elif rec < ij.mean() * 0.85:
        print(f"  fixed basis does NOT generalise but recycling does "
              f"({ij.mean():.1f}→{rec:.1f}) → ship subspace-RECYCLING "
              f"(reuse previous solve's bottom-k). Still no ML.")
    else:
        print(f"  neither generalises out-of-sample → in-sample artifact; "
              f"reopen the learned (support-conditioned) rig.")


if __name__ == "__main__":
    main()
