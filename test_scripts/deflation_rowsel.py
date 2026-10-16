"""Row-selection deflation: Z = unit vectors on [18 joint/control rows ‖ k_c
contact rows], i.e. exact block elimination of those rows, with the contact
rows picked by a cheap rule instead of a network.

Selection rules (all on the Jacobi-scaled As = D A D, D = diag(A)^-1/2, so
|As_rj| = |A_rj| / sqrt(A_rr A_jj) is the classical AMG strength of
connection between rows r and j):
  strength A (all j)      s_r = max_{j≠r} |As_rj|, j over all rows
  strength A (contacts)   same, j over contact rows only (coupling to
                          joints is already removed by the joint elimination)
  strength S              same on the scaled Schur complement left after
                          joint/control elimination, Ss / sqrt(diag Ss ⊗ diag Ss)
References:
  oracle leverage         Σ_{i<16} V[r,i]² / λ_i on scaled Schur complement
                          (needs an eigendecomposition: not cheap, ceiling-ish)
  random                  uniform contact rows (seeded)

Iterations: float32 at the engine settings, held-out split; time per solve
at W = 1 from the dense Warp deflation kernels (gpu_costs of
deflation_hybrid.py — an upper bound, unit-vector Z is really a gather).

Run:
    python -m test_scripts.deflation_rowsel
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from test_scripts.precond_lab import load_systems
from test_scripts.recycle_probe import pct
from test_scripts.deflation import DT, SCENES
from test_scripts.deflation_cost import SETTINGS
from test_scripts.deflation_hybrid import N_JC, gpu_costs, run_iters

KCS = (4, 8, 12)
N_LEV = 16


def scaled(A):
    D = torch.diagonal(A).rsqrt()
    return D.unsqueeze(1) * A * D.unsqueeze(0)


def offdiag_max(M):
    M = M.abs().clone()
    M.fill_diagonal_(0.0)
    return M.max(1).values


def scores(A, jrows, crows):
    """Per-contact-row score for each rule (higher = eliminate first)."""
    As = scaled(A)
    J, C = torch.tensor(jrows), torch.tensor(crows)
    Ss = As[C][:, C] - As[C][:, J] @ torch.linalg.solve(As[J][:, J], As[J][:, C])
    Ss = 0.5 * (Ss + Ss.T)
    out = {
        "strength A (all j)": offdiag_max(As)[C],
        "strength A (contacts)": offdiag_max(As[C][:, C]),
        "strength S": offdiag_max(scaled(Ss)),
    }
    lam, V = torch.linalg.eigh(Ss)
    m = min(N_LEV, len(crows))
    out["oracle leverage"] = (V[:, :m] ** 2 / lam[:m]).sum(1)
    out["random"] = torch.from_numpy(np.random.default_rng(len(crows)).random(len(crows)))
    return out


def sources(S, d, va):
    off_n = int(d["meta"][0]["offset_n"])
    acts = [np.nonzero(d["constr_active_mask"][s, 0] > 0.0)[0] for s in range(len(S))]
    src = {"jacobi": {}, "joint/control": {}}
    for i in va:
        n = S[i].n
        jrows = [r for r, g in enumerate(acts[i]) if g < off_n]
        crows = [r for r in range(n) if r not in set(jrows)]
        eye = torch.eye(n, dtype=DT)
        src["joint/control"][i] = eye[:, jrows]
        for rule, s in scores(S[i].A, jrows, crows).items():
            order = torch.argsort(s, descending=True)
            for kc in KCS:
                kk = min(kc, len(crows) // 2)
                pick = [crows[int(j)] for j in order[:kk]]
                src.setdefault(f"{rule} {kc}", {})[i] = eye[:, jrows + pick]
    return src


def main():
    wp.init()
    print("(B) GPU cost, W = 1")
    cost = gpu_costs(np.load(SCENES["obstacle"], allow_pickle=True), sorted({N_JC} | {N_JC + k for k in KCS}))

    for label, path in SCENES.items():
        S = load_systems(path=path, dtype=DT)
        d = np.load(path, allow_pickle=True)
        va = list(range(int(0.8 * len(S)), len(S)))
        src = sources(S, d, va)
        print(f"\n===== {label}: held-out {len(va)} systems =====")
        for sname, (tol, cap) in SETTINGS.items():
            res = run_iters(S, va, src, torch.float32, tol, cap)
            tj = cost[0][0] + res["jacobi"][0].mean() * cost[0][1]
            print(f"\n  [{sname}] float32: mean median p90 #capped | rel-res p90 | W=1 ms/solve (Δ vs Jacobi)")
            for name, (its, rr, _, fail) in res.items():
                k = 0 if name == "jacobi" else N_JC if name == "joint/control" else N_JC + int(name.split()[-1])
                fs, per = cost[k]
                t = fs + its.mean() * per
                print(f"    {name:26s} {its.mean():5.1f} {pct(its,50):4.0f} {pct(its,90):4.0f}"
                      f" {int((its >= cap).sum()):3d} | {pct(rr,90):.0e} | {t:.3f} ({100 * (t / tj - 1):+4.0f}%)"
                      + (f"  [fail {fail}]" if fail else ""))


if __name__ == "__main__":
    main()
