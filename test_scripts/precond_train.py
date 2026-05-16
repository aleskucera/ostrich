"""Differentiable unrolled-PCR training for the three GNN heads, then
iters-to-tol evaluation against the Jacobi / per-body-pair baselines.

Loss = log10 relative residual after K unrolled PCR steps (no early
exit, pure-tensor so it is differentiable in the head's parameters):

  * diag  : M⁻¹ = learned positive diagonal (SPD)
  * block : M⁻¹ = learned per-body-pair SPD block solve
  * x0    : learned initial guess, PCR uses fixed Jacobi M

float64 throughout — cond(A) ~ 1e11, float32 PCR would be meaningless.
Train/val split is contiguous by capture order (val = last 20% =
later/post-impact systems → an honest generalisation test).

Run:
    python test_scripts/precond_train.py --epochs 400
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from test_scripts.precond_gnn import PrecondGNN, build_graph, body_features_from_npz
from test_scripts.precond_lab import (
    DATA, MAX_ITERS, load_systems, pcr, jacobi_apply, per_body_pair_apply,
)

DT = torch.float64
RANK = 16  # overridden by --rank in main()


def pcr_unroll(A, b, M_apply, x0, K):
    """Fixed-K PCR, no early exit / no .item() — autograd-safe.
    Returns ||b - A x_K|| / ||b|| (scalar tensor)."""
    x = torch.zeros_like(b) if x0 is None else x0
    r = b - A @ x
    z = M_apply(r)
    Az = A @ z
    p, Ap = z, Az
    zAz = torch.dot(z, Az)
    for _ in range(K):
        y = M_apply(Ap)
        yAp = torch.dot(y, Ap)
        alpha = zAz / (yAp + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        z = z - alpha * y
        Az = A @ z
        zAz_new = torch.dot(z, Az)
        beta = zAz_new / (zAz + 1e-30)
        p = z + beta * p
        Ap = Az + beta * Ap
        zAz = zAz_new
    return torch.linalg.norm(r) / (torch.linalg.norm(b) + 1e-30)


def build_all_graphs(systems, sysidx):
    d = np.load(DATA, allow_pickle=True)
    Jraw, mask = d["J_values"], d["constr_active_mask"]
    bf = body_features_from_npz(DATA, "cpu", DT)
    graphs = []
    for k, s in zip(sysidx, systems):
        act = np.nonzero(mask[k, 0] > 0.0)[0]
        graphs.append(build_graph(s, Jraw[k, 0, act], bf(k)))
    return graphs


def eval_iters(systems, graphs, mode, net):
    """Faithful early-exit PCR iters-to-tol with the learned operator."""
    its = []
    with torch.no_grad():
        for s, g in zip(systems, graphs):
            if mode == "diag":
                _, it = pcr(s.A, s.b, net.diag_apply(g))
            elif mode == "block":
                _, it = pcr(s.A, s.b, net.block_apply(g, s))
            elif mode == "x0":
                _, it = pcr(s.A, s.b, jacobi_apply(s.A), x0=net.x0(g))
            elif mode == "lowrank":
                _, it = pcr(s.A, s.b, net.lowrank_apply(g, s, RANK))
            its.append(it)
    return np.array(its)


def train_head(mode, tr, tr_g, va, va_g, epochs, K, lr):
    net = PrecondGNN().to(DT)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for ep in range(epochs):
        perm = np.random.permutation(len(tr))
        tot = 0.0
        for i in perm:
            s, g = tr[i], tr_g[i]
            if mode == "diag":
                loss = torch.log10(pcr_unroll(s.A, s.b, net.diag_apply(g), None, K) + 1e-30)
            elif mode == "block":
                loss = torch.log10(pcr_unroll(s.A, s.b, net.block_apply(g, s), None, K) + 1e-30)
            elif mode == "x0":
                loss = torch.log10(
                    pcr_unroll(s.A, s.b, jacobi_apply(s.A), net.x0(g), K) + 1e-30)
            elif mode == "lowrank":
                loss = torch.log10(
                    pcr_unroll(s.A, s.b, net.lowrank_apply(g, s, RANK), None, K) + 1e-30)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss)
        if (ep + 1) % max(1, epochs // 20) == 0:
            print(f"  [{mode}] epoch {ep+1:4d}/{epochs}  "
                  f"train log10-relres@{K} = {tot/len(tr):+.3f}", flush=True)
    return net


def summary(tag, a):
    a = np.asarray(a, float)
    return (f"{tag:16s} mean {a.mean():5.1f}  p50 {np.percentile(a,50):5.1f}  "
            f"p95 {np.percentile(a,95):5.1f}  max {a.max():4.0f}  "
            f"unconv {int((a>=MAX_ITERS).sum())}/{len(a)}")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--unroll-K", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=16, help="rank for the lowrank head")
    ap.add_argument("--only", type=str, default="",
                    help="comma-list of heads to train (default: all)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    global RANK
    RANK = args.rank

    S = load_systems(dtype=DT)
    n = len(S)
    cut = int(0.8 * n)
    idx = list(range(n))
    tr_i, va_i = idx[:cut], idx[cut:]
    tr, va = [S[i] for i in tr_i], [S[i] for i in va_i]
    tr_g = build_all_graphs(tr, tr_i)
    va_g = build_all_graphs(va, va_i)
    print(f"{n} systems  train {len(tr)}  val {len(va)} (val = later/post-impact)\n")

    # Baselines on the val split.
    jac = np.array([pcr(s.A, s.b, jacobi_apply(s.A))[1] for s in va])
    pbp = np.array([pcr(s.A, s.b, per_body_pair_apply(s.A, s.pair_id))[1] for s in va])
    print("VAL baselines:")
    print("  " + summary("jacobi", jac))
    print("  " + summary("per_body_pair", pbp))
    print()

    # lowrank & x0 first — they are the experiments that matter, so their
    # results land even if a later head is slow / interrupted.
    all_heads = ("lowrank", "x0", "diag", "block")
    heads = tuple(h for h in all_heads
                  if not args.only or h in args.only.split(","))

    results = {"jacobi": jac, "per_body_pair": pbp}
    for mode in heads:
        tag = f"'{mode}'" + (f" (rank {RANK})" if mode == "lowrank" else "")
        print(f"training head {tag} ...")
        net = train_head(mode, tr, tr_g, va, va_g, args.epochs, args.unroll_K, args.lr)
        va_it = eval_iters(va, va_g, mode, net)
        results[f"gnn_{mode}"] = va_it
        print("  " + summary(f"gnn_{mode}", va_it) + "\n")

    print("=" * 72)
    print("VAL iters-to-tol summary (bar to beat = jacobi):")
    for k, v in results.items():
        print("  " + summary(k, v))
    base = results["jacobi"].mean()
    print()
    for mode in heads:
        k = f"gnn_{mode}"
        m = results[k].mean()
        verdict = "BEATS" if m < base else "does NOT beat"
        print(f"  {k:12s} {m:5.1f} vs jacobi {base:5.1f}  -> {verdict} Jacobi")


if __name__ == "__main__":
    main()
