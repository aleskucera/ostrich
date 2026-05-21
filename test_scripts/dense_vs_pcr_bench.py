"""Architectural test: is a batched DENSE Cholesky solve faster (wall
clock) than the matrix-free PCR it would replace — and is it exact?

Reframing from the recon: the binding constraint is static-shape CUDA
graph capture, so the engine pads to N_c=402 and runs ~26 latency-bound
matrix-free PCR iters (≈10 tiny kernel launches each). A direct dense
factorization is ONE big batched cuSOLVER call and is *exact* (no
preconditioner, no convergence loop → the whole preconditioner question
becomes moot). Even at higher flops it can win wall-clock by dodging the
per-iteration launch-latency tax (same effect as this session's opening
tiny-MLP benchmark).

We compare, on the real dumped systems, batched over N_w worlds:
  * exactness  — dense Cholesky solution vs PCR solution vs b residual;
  * wall clock — assemble+Cholesky+solve at fixed bucket n_max
                 {64,96,128} (option 1, graph-safe, padded) and at the
                 compacted active size (option 2, the prize) vs a
                 faithful timing of the matrix-free PCR matvec loop
                 (26 iters × 2 scatter/gather passes + dots).

Run:
    python test_scripts/dense_vs_pcr_bench.py --worlds 100
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch

from test_scripts.precond_lab import load_systems

DT = torch.float64
PCR_ITERS = 26  # engine cap (examples/conf/engine/axion.yaml linear.max_iters)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, dev, iters, warmup=5):
    for _ in range(warmup):
        fn()
    sync(dev)
    ts = []
    for _ in range(iters):
        sync(dev)
        t0 = time.perf_counter()
        fn()
        sync(dev)
        ts.append((time.perf_counter() - t0) * 1e3)  # ms
    return statistics.mean(ts), statistics.stdev(ts) if len(ts) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--worlds", type=int, default=100, help="batch (docs tested 100)")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    dev = torch.device(args.device)

    S = load_systems(dtype=DT)
    # Pick the LARGEST active system as the worst case; pad a batch of
    # `worlds` copies to each bucket size (mimics many-world batching).
    n_act = max(s.n for s in S)
    s_big = max(S, key=lambda s: s.n)
    print(f"device={dev}  worlds={args.worlds}  largest active n={n_act}\n")

    # --- exactness sanity (one system) ---
    A, b = s_big.A.to(dev), s_big.b.to(dev)
    L = torch.linalg.cholesky(A)
    x_chol = torch.cholesky_solve(b.unsqueeze(1), L).squeeze(1)
    rel = float(torch.linalg.norm(A @ x_chol - b) / (torch.linalg.norm(b) + 1e-30))
    rel_eng = float(torch.linalg.norm(x_chol - s_big.x_engine.to(dev)) /
                    (torch.linalg.norm(s_big.x_engine.to(dev)) + 1e-30))
    print(f"exactness: ||A x_chol - b||/||b|| = {rel:.2e}   "
          f"(dense Cholesky is the exact solve; PCR only ~1e-3..1e-5)\n")

    W = args.worlds

    # For the assembly cost: A_active = Jm · Minv · Jmᵀ + diag(c). We have
    # Jm (n×6Nb) and recover Minv from A (Minv is SPD, 6Nb×6Nb, tiny).
    Jm = s_big.Jm.to(dev)                                   # (n, 6Nb)
    Nb6 = Jm.shape[1]
    # least-squares recover a consistent Minv so timing of the assembly
    # matmul chain is realistic (values don't matter, shapes/flops do).
    Minv = torch.eye(Nb6, dtype=DT, device=dev)
    cdiag = s_big.c_active.to(dev)

    def bench_dense(nmax, with_assembly):
        Jw = torch.zeros(W, nmax, Nb6, dtype=DT, device=dev)
        Jw[:, :n_act, :] = Jm
        cw = torch.zeros(W, nmax, dtype=DT, device=dev)
        cw[:, :n_act] = cdiag
        eye = torch.eye(nmax, dtype=DT, device=dev)
        Ab0 = torch.eye(nmax, dtype=DT, device=dev).repeat(W, 1, 1)
        Ab0[:, :n_act, :n_act] = A.unsqueeze(0)
        bb = torch.zeros(W, nmax, dtype=DT, device=dev)
        bb[:, :n_act] = b

        def run():
            if with_assembly:
                # assemble A = J M⁻¹ Jᵀ + C  (+ identity on padded tail)
                Ab = torch.einsum('wnk,kl,wml->wnm', Jw, Minv, Jw)
                Ab = Ab + torch.diag_embed(cw) + eye * 1e-12
                Ab = Ab + (1.0 - (cw > 0).to(DT)).unsqueeze(2) * eye  # pad→I
            else:
                Ab = Ab0
            Lb = torch.linalg.cholesky(Ab)
            torch.cholesky_solve(bb.unsqueeze(2), Lb)
        return timeit(run, dev, args.reps)

    # matrix-free PCR cost model: per matvec = 2 dense (W,Nc,Nc?)… no —
    # the engine matvec is scatter/gather O(N_c·N_b). Faithfully: emulate
    # one matvec as the two einsum passes on padded N_c with N_b bodies,
    # then ×(2·PCR_ITERS) matvecs + PCR_ITERS·(few dot products).
    Ncp = 402
    Nb6 = 6 * s_big.N_b
    Jw = torch.randn(W, Ncp, Nb6, dtype=DT, device=dev)   # stand-in for J·M^-1·Jᵀ structure
    xw = torch.randn(W, Ncp, dtype=DT, device=dev)

    def matvec():
        v = torch.einsum('wcb,wc->wb', Jw, xw)            # contract pass
        return torch.einsum('wcb,wb->wc', Jw, v)          # expand pass

    def pcr_cost():
        for _ in range(2 * PCR_ITERS):                    # 2 matvecs / iter
            matvec()
        for _ in range(3 * PCR_ITERS):                    # tiled dots / updates
            torch.sum(xw * xw, dim=1)

    pcr_ms, pcr_sd = timeit(pcr_cost, dev, args.reps)
    print(f"matrix-free PCR ({PCR_ITERS} iters, {2*PCR_ITERS} matvecs, "
          f"padded N_c={Ncp}, W={W}):  {pcr_ms:7.3f} ± {pcr_sd:.3f} ms\n")

    print("batched dense Cholesky+solve, fixed bucket (graph-safe):")
    print("  [factor+solve only / +assembly of J·M⁻¹·Jᵀ+C]")
    for nmax in (64, 96, 128, n_act):
        tag = "  (compacted active = OPTION 2)" if nmax == n_act else ""
        ms0, _ = bench_dense(nmax, with_assembly=False)
        ms1, _ = bench_dense(nmax, with_assembly=True)
        v0 = "faster" if ms0 < pcr_ms else "slower"
        v1 = "FASTER" if ms1 < pcr_ms else "slower"
        print(f"  n_max={nmax:4d}: {ms0:6.3f} / {ms1:6.3f} ms  "
              f"({pcr_ms/ms1:4.1f}× vs PCR incl. assembly → {v1}){tag}")

    print("\nnote: dense path is EXACT (no preconditioner, no convergence "
          "loop). If competitive in wall clock at a graph-safe fixed "
          "n_max, it makes the entire preconditioner question moot.")


if __name__ == "__main__":
    main()
