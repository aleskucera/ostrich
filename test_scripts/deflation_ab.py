"""Step 1c — the A/B table: additive two-level vs projection deflation,
the SAME Z fed to both, for every Z source the doc tried additively.

Z sources (each built exactly as its original script builds it):
  exact k           bad_subspace of the system itself       (ceiling)
  cross-step        step N−1's last exact bottom-16, reused (cross_step_stability)
  recycle-exact     previous solve's exact bottom-16        (recycle_validate)
  harmonic m        harmonic Ritz from the prev solve's Krylov harvest (recycle_approx)
  gcrodr            harmonic Ritz accumulated over the sequence (recycle_gcrodr)
  null / joint / enriched   analytic coarse spaces          (coarse_*_probe)

Evaluated on the held-out split (last 20% of systems, as most scripts
do; cross_step_stability held out by step instead, so its additive row
differs slightly from the 14.4 quoted in PLAN.md). Sources that need a
previous solve still use the full sequence before the split.

Run:
    python test_scripts/deflation_ab.py
"""

from __future__ import annotations

import numpy as np
import torch

from test_scripts.precond_lab import ATOL, MAX_ITERS, TOL, jacobi_apply, load_systems, pcr
from test_scripts.ideal_target_stability import bad_subspace
from test_scripts.recycle_probe import embed, pct, restrict, two_level
from test_scripts.recycle_approx import approx_subspace, harmonic_ritz, krylov_residual_basis
from test_scripts.coarse_null_probe import null_jt_basis
from test_scripts.coarse_enriched_probe import orthonormal
from test_scripts.deflation import DT, SCENES, deflated_pcr

K = 16
GCRODR_M = 16


def z_sources(S, d, va):
    """{name: {i: Z_i}} for every held-out system i (Z in i's active space)."""
    n = len(S)
    acts = [np.nonzero(d["constr_active_mask"][s, 0] > 0.0)[0] for s in range(n)]
    off_n = int(d["meta"][0]["offset_n"])
    Q16 = [bad_subspace(S[i].A, min(K, S[i].n)) for i in range(n)]
    E16 = [embed(Q16[i], acts[i]) for i in range(n)]
    src = {}

    for k in (4, 8, 16, 32):
        src[f"exact k={k}"] = {i: bad_subspace(S[i].A, min(k, S[i].n)) for i in va}

    step_idx = d["step_idx"]
    last_of_step = {}
    for i in range(n):                         # systems are in capture order
        last_of_step[int(step_idx[i])] = i
    steps = sorted(last_of_step)
    prev_last = {s: last_of_step[steps[j - 1]] for j, s in enumerate(steps) if j > 0}
    src["cross-step (N-1)"] = {i: restrict(E16[prev_last[int(step_idx[i])]], acts[i])
                               for i in va if int(step_idx[i]) in prev_last}

    src["recycle-exact (prev)"] = {i: restrict(E16[i - 1], acts[i]) for i in va}

    torch.manual_seed(0)                       # approx_subspace draws a random RHS
    for m in (16, 24, 32):
        src[f"harmonic m={m}"] = {
            i: restrict(embed(approx_subspace(S[i - 1].A, m, K), acts[i - 1]), acts[i])
            for i in va}

    # GCRO-DR: U only depends on the Krylov harvests, not on how the solve
    # used it, so one pass gives the same Z sequence for both methods.
    U, g = None, {}
    for i in range(n):
        A = S[i].A
        D = torch.diagonal(A).clamp_min(1e-30).rsqrt()
        As = D.unsqueeze(1) * A * D.unsqueeze(0)
        if i in va:
            g[i] = restrict(U, acts[i]) if U is not None else torch.zeros(S[i].n, 0, dtype=DT)
        Vi = krylov_residual_basis(As, D * S[i].b, GCRODR_M)
        Cand = torch.linalg.qr(torch.cat([restrict(U, acts[i]), Vi], 1))[0] if U is not None else Vi
        Y = torch.linalg.qr(harmonic_ritz(As, Cand, K))[0]
        U = embed(torch.linalg.qr(D.unsqueeze(1) * Y)[0], acts[i])
    src["gcrodr accumulated"] = g

    src["null(J^T)"], src["joint/control"], src["enriched"] = {}, {}, {}
    for i in va:
        Zn, _ = null_jt_basis(S[i].Jm)
        eye = torch.eye(S[i].n, dtype=DT)
        Ijc = eye[:, [r for r, gi in enumerate(acts[i]) if gi < off_n]]
        src["null(J^T)"][i] = orthonormal(Zn)
        src["joint/control"][i] = orthonormal(Ijc)
        src["enriched"][i] = orthonormal(torch.cat([Zn, Ijc], 1))
    return src


def run_scene(label, path):
    S = load_systems(path=path, dtype=DT)
    d = np.load(path, allow_pickle=True)
    va = list(range(int(0.8 * len(S)), len(S)))
    jac = np.array([pcr(S[i].A, S[i].b, jacobi_apply(S[i].A))[1] for i in va])
    src = z_sources(S, d, va)

    print(f"\n===== {label}: held-out {len(va)} of {len(S)} systems =====")
    print(f"  {'Z source':22s} {'k p50':>5s} {'k/n':>4s} | {'additive':^31s} | {'deflation':^31s} | defl res/target")
    print(f"  {'':22s} {'':5s} {'p50':>4s} | {'mean  p50  p90  wins/loss unc':31s} | "
          f"{'mean  p50  p90  wins/loss unc':31s} | max (converged)")

    def cells(a):
        return (f"{a.mean():5.1f} {pct(a,50):4.0f} {pct(a,90):4.0f}  "
                f"{int((a < jac).sum()):3d}/{int((a > jac).sum()):<3d} {int((a >= MAX_ITERS).sum()):3d}")

    print(f"  {'jacobi (bar)':22s} {0:5d} {0:4.2f} | {cells(jac)} | {cells(jac)} |")
    for name, Zs in src.items():
        add, dfl, ks, kn, rt = [], [], [], [], []
        for i in va:
            A, b = S[i].A, S[i].b
            Z = Zs.get(i, torch.zeros(S[i].n, 0, dtype=DT))  # no source → Jacobi
            add.append(pcr(A, b, two_level(A, Z))[1])
            x, it, k = deflated_pcr(A, b, Z, jacobi_apply(A))
            dfl.append(it)
            ks.append(k)
            kn.append(k / S[i].n)
            if it < MAX_ITERS:
                rt.append(float(torch.linalg.norm(b - A @ x))
                          / max(ATOL, TOL * float(torch.linalg.norm(b))))
        add, dfl = np.array(add), np.array(dfl)
        print(f"  {name:22s} {pct(ks,50):5.0f} {pct(kn,50):4.2f} | {cells(add)} | {cells(dfl)} | {max(rt):5.2f}")


def main():
    print("iterations to tol (PCR, TOL=ATOL=1e-5, cap 64); wins/loss = systems "
          "faster/slower than Jacobi; unc = hit the cap")
    for label, path in SCENES.items():
        run_scene(label, path)


if __name__ == "__main__":
    main()
