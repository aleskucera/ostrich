"""Hybrid Z ceiling: Z = [joint/control ‖ exact bottom-k_c of the condensed
contact problem], i.e. the best a network predicting the contact part of
Z on top of the analytic joint/control part could ever do.

With Z_J = unit vectors on the 18 joint/control rows J, deflation leaves
    PA = [[0, 0], [0, S]],   S = A_CC − A_CJ A_JJ⁻¹ A_JC   (C = contact rows)
so Jacobi(A)-PCR sees D_C² S. Its slow directions are the bottom
eigenvectors v of the scaled Schur complement D_C S D_C
(= As_CC − As_CJ As_JJ⁻¹ As_JC, As = D A D), mapped back as y = D_C v and
embedded with zeros on J. Z = [Z_J ‖ Y] deflates exactly those modes.
k_c is capped at n_C // 2 so it stays a deflation, not a direct solve.

(A) iterations, held-out split: float64 tol 1e-5 / cap 64 (step 1c
    convention) and float32 at the engine settings (step 2).
    Comparison at the same k budget: exact bottom-(18 + k_c) of A.
(B) GPU, W = 1 (the only regime where joint/control won): fixed+setup and
    per-iteration cost of deflation at k = 18 + k_c (dense Warp kernels
    from deflation_cost.py — an upper bound, Z_J is really sparse).
(C) time per solve and the BUDGET left for producing Y (network
    inference): Jacobi total − hybrid total, and joint/control-alone total
    − hybrid total (the real alternative). Negative = cannot win even with
    a free, perfect network.

Run:
    python test_scripts/deflation_hybrid.py
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from ostrich.optim import PCRSolver
from test_scripts.precond_lab import jacobi_apply, load_systems
from test_scripts.recycle_probe import NC, pct
from test_scripts.ideal_target_stability import bad_subspace
from test_scripts.deflation import DT, SCENES, deflated_pcr
from test_scripts.deflation_cost import DEV, SETTINGS, Deflation, engine_objects, time_graph

N_JC = 18
KCS = (4, 8, 12, 20)


def hybrid_Z(A, jrows, kc):
    """[unit vectors on jrows ‖ D_C · bottom-kc eigvecs of the scaled Schur complement]."""
    n = A.shape[0]
    crows = [r for r in range(n) if r not in set(jrows)]
    kc = min(kc, len(crows) // 2)
    D = torch.diagonal(A).rsqrt()
    As = D.unsqueeze(1) * A * D.unsqueeze(0)
    J, C = torch.tensor(jrows), torch.tensor(crows)
    Ss = As[C][:, C] - As[C][:, J] @ torch.linalg.solve(As[J][:, J], As[J][:, C])
    _, V = torch.linalg.eigh(0.5 * (Ss + Ss.T))
    Z = torch.zeros(n, len(jrows) + kc, dtype=A.dtype)
    Z[J, torch.arange(len(jrows))] = 1.0
    Z[C, len(jrows):] = D[C].unsqueeze(1) * V[:, :kc]
    return Z, kc


def sources(S, d, va):
    off_n = int(d["meta"][0]["offset_n"])
    acts = [np.nonzero(d["constr_active_mask"][s, 0] > 0.0)[0] for s in range(len(S))]
    src = {"jacobi": {}, "joint/control": {}}
    kcs = {}
    for i in va:
        jrows = [r for r, g in enumerate(acts[i]) if g < off_n]
        src["joint/control"][i] = torch.eye(S[i].n, dtype=DT)[:, jrows]
        for kc in KCS:
            Z, kk = hybrid_Z(S[i].A, jrows, kc)
            src.setdefault(f"jc + S-bottom {kc}", {})[i] = Z
            kcs.setdefault(kc, []).append(kk)
        src.setdefault(f"A-bottom {N_JC + KCS[-1]}", {})[i] = bad_subspace(S[i].A, min(N_JC + KCS[-1], S[i].n // 2))
    return src, kcs


def run_iters(S, va, src, dtype, tol, cap):
    out = {}
    for name, Zs in src.items():
        its, rr, ks, fail = [], [], [], 0
        for i in va:
            A, b = S[i].A.to(dtype), S[i].b.to(dtype)
            Z = Zs.get(i, torch.zeros(S[i].n, 0, dtype=DT)).to(dtype)
            try:
                x, it, k = deflated_pcr(A, b, Z, jacobi_apply(A), max_iters=cap, tol=tol, atol=tol)
            except torch.linalg.LinAlgError:   # E = ZᵀAZ not SPD in this precision
                fail += 1
                continue
            its.append(it)
            ks.append(k / S[i].n)
            rr.append(float(torch.linalg.norm(S[i].b - S[i].A @ x.double()) / torch.linalg.norm(S[i].b)))
        out[name] = (np.array(its), np.array(rr), np.array(ks), fail)
    return out


def gpu_costs(d, ks):
    """W = 1: Jacobi (fixed, per-iter) and deflation (fixed+setup, per-iter) per k, ms."""
    mask = d["constr_active_mask"][:, 0] > 0.0
    i = int(mask.sum(1).argmax())      # largest active n: random Z must be independent on active rows
    A, M, b = engine_objects(d, np.array([i]))
    solver = PCRSolver(max_iters=32, batch_dim=1, vec_dim=NC, device=DEV)
    x = wp.zeros((1, NC), dtype=wp.float32, device=DEV)

    def slope(fn):
        t4, t32 = time_graph(lambda: fn(4)), time_graph(lambda: fn(32))
        per = (t32 - t4) / 28
        return t4 - 4 * per, per

    def plain(n):
        x.zero_()
        solver.solve(A, b, x, M, iters=n, tol=-1.0, atol=-1.0)
    out = {0: slope(plain)}
    rng = np.random.default_rng(0)
    for k in ks:
        Zt = np.linalg.qr(rng.standard_normal((1, NC, k)) * mask[i][None, :, None])[0].transpose(0, 2, 1)
        D = Deflation(A, wp.array(np.ascontiguousarray(Zt, dtype=np.float32), dtype=wp.float32, device=DEV))

        def deflated(n):
            x.zero_()
            D.setup(b)
            solver.solve(D.op, D.Pb, x, M, iters=n, tol=-1.0, atol=-1.0)
            D.finish(x)
        out[k] = slope(deflated)
        print(f"    k={k:2d}: fixed+setup {out[k][0]:.3f} ms, {1e3 * out[k][1]:5.1f} µs/iter", flush=True)
    return out


def main():
    wp.init()
    ks = sorted({N_JC} | {N_JC + kc for kc in KCS})
    print("(B) GPU cost, W = 1 (dense Warp deflation kernels, real engine PCR graph)")
    cost = gpu_costs(np.load(SCENES["obstacle"], allow_pickle=True), ks)
    print(f"    jacobi: fixed {cost[0][0]:.3f} ms, {1e3 * cost[0][1]:5.1f} µs/iter")

    for label, path in SCENES.items():
        S = load_systems(path=path, dtype=DT)
        d = np.load(path, allow_pickle=True)
        va = list(range(int(0.8 * len(S)), len(S)))
        src, kcs = sources(S, d, va)
        print(f"\n===== {label}: held-out {len(va)} systems, active n {min(S[i].n for i in va)}–"
              f"{max(S[i].n for i in va)}; k_c actually used (cap n_C//2) p50/min: "
              + "  ".join(f"{kc}→{pct(v,50):.0f}/{min(v)}" for kc, v in kcs.items()) + " =====")

        res = run_iters(S, va, src, torch.float64, 1e-5, 64)
        print("  (A) float64, tol 1e-5, cap 64:      median  p90  max | k/n p50")
        for name, (its, _, kn, _) in res.items():
            print(f"    {name:20s} {pct(its,50):5.0f} {pct(its,90):4.0f} {its.max():4d} | {pct(kn,50):.2f}")

        for sname, (tol, cap) in SETTINGS.items():
            res = run_iters(S, va, src, torch.float32, tol, cap)
            print(f"\n  [{sname}] float32: mean median p90 #capped | rel-res p90 | (C) W=1 ms/solve"
                  " = fixed+setup + mean iters × per-iter  (Δ vs Jacobi) | Y budget vs Jacobi / vs joint/control")
            tj = cost[0][0] + res["jacobi"][0].mean() * cost[0][1]
            tjc = cost[N_JC][0] + res["joint/control"][0].mean() * cost[N_JC][1]
            for name, (its, rr, _, fail) in res.items():
                if name.startswith("A-bottom"):
                    k = None
                elif name == "jacobi":
                    k = 0
                elif name == "joint/control":
                    k = N_JC
                else:
                    k = N_JC + int(name.split()[-1])
                line = (f"    {name:20s} {its.mean():5.1f} {pct(its,50):4.0f} {pct(its,90):4.0f}"
                        f" {int((its >= cap).sum()):3d} | {pct(rr,90):.0e}")
                if k is not None:
                    fs, per = cost[k]
                    t = fs + its.mean() * per
                    line += f" | {fs:.3f} + {its.mean():4.1f} × {per:.4f} = {t:.3f} ({100 * (t / tj - 1):+4.0f}%)"
                    if k > N_JC:
                        line += f" | {1e3 * (tj - t):+5.0f} / {1e3 * (tjc - t):+5.0f} µs"
                if fail:
                    line += f"  [E not SPD in float32: {fail}/{len(va)} systems, excluded]"
                print(line)


if __name__ == "__main__":
    main()
