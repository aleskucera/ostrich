"""Direct-solve baselines vs (deflated) PCR: is iterating worth it at all?

A = G M⁻¹ Gᵀ + diag(c),  G = constraint Jacobian (n × 6 N_b), here
N_b = 4 bodies → G has only 24 columns, so A is "rank-24 + diagonal".
Two direct solvers:

  dense Cholesky   assemble A (W × n × n), cholesky_ex + cholesky_solve.
                   "padded": all 402 slots, inactive rows = identity
                   (static shape, as a CUDA graph needs). "compact": only
                   n_max = max active rows of the scene (optimistic: needs
                   a gather and a static per-scene bound).
  body-space       Woodbury / static condensation onto the 6 N_b body DOFs:
                   (M + Gᵀ C⁻¹ G) v = Gᵀ C⁻¹ b,  x = C⁻¹ (b − G v),
                   a 24 × 24 Cholesky per world. Works because every row
                   has c > 0 (joint 1e-2, control 0.22, normal 0.1–1,
                   friction 1e-6 … 3e6, plus reg 1e-6).

(A) accuracy, float32 on the held-out systems: rel. residual ‖b − Ax‖/‖b‖
    (A, b in float64), with and without one step of iterative refinement
    (costs one A-matvec + one more solve with the same factor).
(B) GPU time per solve, torch CUDA graph, assembly from the raw engine
    arrays included, W = 1 / 64 / 1024.
(C) the PCR side from the same engine kernels: fixed + E[max iters over W
    systems] × per-iter, for Jacobi, joint/control (k=18) and joint/control
    + 8 strength rows (k=26, dense deflation kernels = upper bound).

Run:
    python -m test_scripts.deflation_direct
"""

from __future__ import annotations

import time

import numpy as np
import torch
import warp as wp

from ostrich.optim import PCRSolver
from test_scripts.precond_lab import load_systems
from test_scripts.recycle_probe import NC, pct
from test_scripts.deflation import DT, SCENES
from test_scripts.deflation_cost import DEV, REG, SETTINGS, Deflation, batch_iters, engine_objects, time_graph
from test_scripts.deflation_hybrid import run_iters
from test_scripts import deflation_rowsel

WORLDS = (1, 64, 1024)
torch.backends.cuda.preferred_linalg_library("cusolver")   # MAGMA batched potrs is not graph-capturable


def chol_solve(rhs, L):
    """cholesky_solve via two triangular solves (cuBLAS, capturable)."""
    y = torch.linalg.solve_triangular(L, rhs, upper=False)
    return torch.linalg.solve_triangular(L.transpose(-1, -2), y, upper=True)


PCR_K = {"jacobi": 0, "joint/control": 18, "strength A (all j) 8": 26}


# ---------------------------------------------------------------- solvers (batched torch)

def minv_blocks(minv, Iinv):
    """(W, Nb) inverse masses, (W, Nb, 3, 3) world inverse inertias → (W, Nb, 6, 6)."""
    W, Nb = minv.shape
    B = torch.zeros(W, Nb, 6, 6, dtype=minv.dtype, device=minv.device)
    B[:, :, :3, :3] = minv[..., None, None] * torch.eye(3, dtype=minv.dtype, device=minv.device)
    B[:, :, 3:, 3:] = Iinv
    return B


def jacobian(J, bidx, mask, Nb):
    """(W, n, 2, 6) spatial rows at body slots (−1 = ground) → G (W, n, 6 Nb), masked."""
    W, n = J.shape[:2]
    slot = torch.where(bidx < 0, Nb, bidx).long()                       # ground → dropped slot Nb
    G = torch.zeros(W, n, Nb + 1, 6, dtype=J.dtype, device=J.device)
    G.scatter_add_(2, slot[..., None].expand(W, n, 2, 6), J)
    return G[:, :, :Nb].reshape(W, n, 6 * Nb) * mask[..., None]


def dense_solve(G, B, c, mask, b, refine=False):
    W, n, m = G.shape
    GM = torch.einsum("wnbi,wbij->wnbj", G.view(W, n, -1, 6), B).reshape(W, n, m)
    A = GM @ G.transpose(1, 2)
    A.diagonal(dim1=1, dim2=2).add_(torch.where(mask > 0, c + REG, torch.ones_like(c)))
    L = torch.linalg.cholesky_ex(A)[0]
    x = chol_solve((b * mask).unsqueeze(2), L)
    if refine:
        x = x + chol_solve((b * mask).unsqueeze(2) - A @ x, L)
    return x.squeeze(2)


def body_solve(G, B, Minv_inv, c, mask, b, refine=False):
    """Woodbury on body DOFs. Minv_inv = block-diag mass matrix M (W, 6Nb, 6Nb)."""
    W, n, m = G.shape
    ci = mask / (c + REG)                                                # inactive rows: 0

    def solve(rhs, L):
        v = chol_solve((G.transpose(1, 2) @ (ci * rhs).unsqueeze(2)), L)
        return ci * (rhs - (G @ v).squeeze(2))

    K = Minv_inv + G.transpose(1, 2) @ (ci.unsqueeze(2) * G)
    L = torch.linalg.cholesky_ex(K)[0]
    x = solve(b * mask, L)
    if refine:
        GM = torch.einsum("wnbi,wbij->wnbj", G.view(W, n, -1, 6), B).reshape(W, n, m)
        Ax = (GM @ (G.transpose(1, 2) @ x.unsqueeze(2))).squeeze(2) + (c + REG) * x
        x = x + solve(b * mask - Ax * mask, L)
    return x


def mass_matrix(B):
    W, Nb = B.shape[:2]
    M = torch.zeros(W, 6 * Nb, 6 * Nb, dtype=B.dtype, device=B.device)
    for k in range(Nb):
        M[:, 6 * k:6 * k + 6, 6 * k:6 * k + 6] = torch.linalg.inv(B[:, k])
    return M


def world_tensors(d, idx, dtype, compact=None):
    """Raw engine arrays for worlds `idx` as torch tensors on the GPU."""
    t = lambda key: torch.from_numpy(np.ascontiguousarray(d[key][idx, 0])).to(DEV)
    J, bidx, mask, c, b = t("J_values").to(dtype), t("constr_body_idx"), t("constr_active_mask").to(dtype), \
        t("C_values").to(dtype), t("b").to(dtype)
    if compact is not None:                                             # active rows first, keep n_max
        order = torch.argsort(mask, dim=1, descending=True, stable=True)[:, :compact]
        g = lambda a: torch.gather(a, 1, order.view(*order.shape, *([1] * (a.dim() - 2))).expand(-1, -1, *a.shape[2:]))
        J, bidx, mask, c, b = g(J), g(bidx), g(mask), g(c), g(b)
    B = minv_blocks(t("body_inv_mass").to(dtype), t("world_inv_inertia").to(dtype))
    return J, bidx, mask, c, b, B


# ---------------------------------------------------------------- (A) accuracy

def accuracy(S, d, va):
    print("  (A) float32 rel. residual over held-out systems: p50 / p90 / max   [+1 refinement step]")
    Nb = int(d["meta"][0]["N_b"])
    acts = [np.nonzero(d["constr_active_mask"][s, 0] > 0.0)[0] for s in range(len(S))]
    J, bidx, mask, c, b, B = world_tensors(d, np.array(va), torch.float32)
    G = jacobian(J, bidx, mask, Nb)
    M = mass_matrix(B)
    for name, fn in {"dense Cholesky": lambda r: dense_solve(G, B, c, mask, b, r),
                     "body-space (24×24)": lambda r: body_solve(G, B, M, c, mask, b, r)}.items():
        line = f"    {name:20s}"
        for refine in (False, True):
            x = fn(refine).double().cpu()
            rr = np.array([float(torch.linalg.norm(S[i].b - S[i].A @ x[j, acts[i]]) / torch.linalg.norm(S[i].b))
                           for j, i in enumerate(va)])
            bad = int((~np.isfinite(rr)).sum())
            rr = rr[np.isfinite(rr)]
            line += f"  {pct(rr,50):.0e} / {pct(rr,90):.0e} / {rr.max():.0e}" + (f" [{bad} NaN]" if bad else "")
        print(line)


# ---------------------------------------------------------------- (B) direct timing

def time_cuda_graph(fn, warm_s=0.5, meas_s=1.0):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()

    def run(seconds):
        n, t0 = 0, time.perf_counter()
        while True:
            for _ in range(10):
                g.replay()
            torch.cuda.synchronize()
            n += 10
            if time.perf_counter() - t0 >= seconds:
                return (time.perf_counter() - t0) / n * 1e3
    run(warm_s)
    return run(meas_s)


def direct_times(d, nmax):
    Nb = int(d["meta"][0]["N_b"])
    out = {}
    for W in WORLDS:
        idx = np.arange(W) % len(d["b"])
        for name, compact in (("dense padded 402", None), (f"dense compact {nmax}", nmax), ("body-space 24", None)):
            J, bidx, mask, c, b, B = world_tensors(d, idx, torch.float32, compact)
            if name.startswith("body"):
                M = mass_matrix(B)   # the engine has M itself (mass, world inertia); inv is not capturable
                def fn():   # assembly (G) + solve inside the graph
                    G = jacobian(J, bidx, mask, Nb)
                    return body_solve(G, B, M, c, mask, b)
            else:
                def fn():
                    return dense_solve(jacobian(J, bidx, mask, Nb), B, c, mask, b)
            try:
                out[(W, name)] = time_cuda_graph(fn)
            except torch.OutOfMemoryError:
                out[(W, name)] = float("nan")
            torch.cuda.empty_cache()
    return out


def pcr_times(d):
    """(fixed, per-iter) ms of the engine PCR graph per W and k (dense deflation kernels)."""
    out = {}
    for W in WORLDS:
        idx = np.arange(W) % len(d["b"])
        A, M, b = engine_objects(d, idx)
        solver = PCRSolver(max_iters=32, batch_dim=W, vec_dim=NC, device=DEV)
        x = wp.zeros((W, NC), dtype=wp.float32, device=DEV)
        rng = np.random.default_rng(0)
        mask = d["constr_active_mask"][idx, 0] > 0.0
        for k in sorted(set(PCR_K.values())):
            if k:
                # random Z on the active rows of the largest-active world (E stays SPD for k ≤ active n)
                big = mask[int(mask.sum(1).argmax())]
                Zt = np.linalg.qr(rng.standard_normal((W, NC, k)) * big[None, :, None])[0].transpose(0, 2, 1)
                D = Deflation(A, wp.array(np.ascontiguousarray(Zt, dtype=np.float32), dtype=wp.float32, device=DEV))

            def run(n):
                x.zero_()
                if k:
                    D.setup(b)
                    solver.solve(D.op, D.Pb, x, M, iters=n, tol=-1.0, atol=-1.0)
                    D.finish(x)
                else:
                    solver.solve(A, b, x, M, iters=n, tol=-1.0, atol=-1.0)
            t4, t32 = time_graph(lambda: run(4)), time_graph(lambda: run(32))
            per = (t32 - t4) / 28
            out[(W, k)] = (t4 - 4 * per, per)
    return out


def main():
    wp.init()
    for label, path in SCENES.items():
        S = load_systems(path=path, dtype=DT)
        d = np.load(path, allow_pickle=True)
        va = list(range(int(0.8 * len(S)), len(S)))
        nmax = int((d["constr_active_mask"][:, 0] > 0).sum(1).max())
        print(f"\n===== {label}: held-out {len(va)} systems, n_max {nmax} =====")
        accuracy(S, d, va)

        print("  (B) direct, ms per solve (torch CUDA graph, assembly included)")
        dt = direct_times(d, nmax)
        for name in ("dense padded 402", f"dense compact {nmax}", "body-space 24"):
            print(f"    {name:20s}" + "".join(f"  W={W:<5d} {dt[(W, name)]:7.3f}" for W in WORLDS))

        print("  (C) PCR, ms per solve = fixed + E[max iters over W] × per-iter   (W = 1 / 64 / 1024)")
        pt = pcr_times(d)
        src = deflation_rowsel.sources(S, d, va)
        src = {k: v for k, v in src.items() if k in PCR_K}
        rng = np.random.default_rng(0)
        for sname, (tol, cap) in SETTINGS.items():
            res = run_iters(S, va, src, torch.float32, tol, cap)
            print(f"    [{sname}]")
            for name, k in PCR_K.items():
                its = res[name][0]
                cells = []
                for W in WORLDS:
                    it = batch_iters(its, W, rng)
                    fs, per = pt[(W, k)]
                    cells.append(f"W={W:<5d} {fs + it * per:7.3f} ({it:4.1f} it)")
                print(f"      {name:22s} rel-res p90 {pct(res[name][1],90):.0e}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
