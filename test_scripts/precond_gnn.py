"""Tiny message-passing GNN over the constraint<->body bipartite graph,
with three heads probing the learned-preconditioner feasibility question
(see project_precond_baseline memory for why these three):

  * head "diag"  — per-constraint positive scale d>0; M⁻¹ r = d ⊙ r.
                   SPD by construction. Generalises Jacobi (which is the
                   special case d = 1/diag(A)). Tests: can a learned
                   *diagonal that sees graph+values* beat analytic 1/diag?

  * head "block" — per body-pair, a predicted SPD Cholesky factor L
                   (M⁻¹ via block solves, like per_body_pair but with a
                   LEARNED block). Directly motivated by the finding that
                   the *exact* A_pair block hurts: can a learned block do
                   better than both exact-block and diagonal?

  * head "x0"    — per-constraint initial guess Δλ₀; PCR (Jacobi M)
                   polishes. Tests: does a learned warm start cut iters?

Graph (per system, active set only, n≈31-67):
  constraint node c  -- edge (J spatial row, 6) --  body node b
Message passing: constr->body aggregate, body->constr aggregate, R rounds.
Everything is plain torch (index_add scatter); n is tiny so per-system
processing with no batching is fine.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from test_scripts.precond_lab import System


# --------------------------------------------------------------------------
# Per-system static graph: incidence + node/edge features (no learned params).
# Built once per System and cached on it.
# --------------------------------------------------------------------------

class Graph:
    def __init__(self, sys: System, J_rows: np.ndarray, body_feat: np.ndarray):
        # edges: arrays of (constraint_local_idx, body_idx, edge_feat[6])
        ce, be, ef = [], [], []
        for r in range(sys.n):
            for slot in (0, 1):
                b = int(sys.body_pair[r, slot])
                if b < body_feat.shape[0]:  # skip ground sentinel
                    ce.append(r)
                    be.append(b)
                    ef.append(J_rows[r, slot])
        self.ci = torch.tensor(ce, dtype=torch.long, device=sys.A.device)
        self.bi = torch.tensor(be, dtype=torch.long, device=sys.A.device)
        self.ef = torch.tensor(np.array(ef), dtype=sys.A.dtype, device=sys.A.device)
        self.n_c = sys.n
        self.n_b = int(body_feat.shape[0])

        diagA = torch.diagonal(sys.A)
        # constraint node features: C+reg, diag(A), |b|, log10|diag|
        self.cx = torch.stack([
            sys.c_active,
            diagA,
            sys.b.abs(),
            torch.log10(diagA.abs() + 1e-12),
        ], dim=1)
        self.bx = torch.tensor(body_feat, dtype=sys.A.dtype, device=sys.A.device)


def build_graph(sys: System, raw_J: np.ndarray, body_feat: np.ndarray) -> Graph:
    """raw_J: (n,2,6) active J rows for this system.
    body_feat: (n_b, k) per-body features (m_inv, mean diag of world Iinv)."""
    return Graph(sys, raw_J, body_feat)


# --------------------------------------------------------------------------
# The GNN.
# --------------------------------------------------------------------------

def mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class PrecondGNN(nn.Module):
    def __init__(self, hid=32, rounds=3, cx_dim=4, bx_dim=2, ef_dim=6):
        super().__init__()
        self.rounds = rounds
        self.hid = hid
        self.enc_c = mlp([cx_dim, hid, hid])
        self.enc_b = mlp([bx_dim, hid, hid])
        # message nets, shared across rounds (weight-tied -> tiny)
        self.msg_c2b = mlp([2 * hid + ef_dim, hid, hid])
        self.msg_b2c = mlp([2 * hid + ef_dim, hid, hid])
        self.upd_b = mlp([2 * hid, hid, hid])
        self.upd_c = mlp([2 * hid, hid, hid])
        # heads
        self.head_diag = mlp([hid, hid, 1])   # -> log-scale
        self.head_x0 = mlp([hid, hid, 1])     # -> Δλ0
        self.head_blk = mlp([2 * hid, hid, 1])  # per constraint-pair entry of L
        self.lr_k = 16                          # max rank the head can emit
        self.head_lr = mlp([hid, hid, self.lr_k])      # -> U columns (per constr)
        self.head_lrg = mlp([hid, hid, self.lr_k])     # -> log-gains (pooled)

    def trunk(self, g: Graph):
        c = self.enc_c(g.cx)
        b = self.enc_b(g.bx)
        for _ in range(self.rounds):
            # constraint -> body
            m = self.msg_c2b(torch.cat([c[g.ci], b[g.bi], g.ef], dim=1))
            agg_b = torch.zeros(g.n_b, self.hid, dtype=c.dtype, device=c.device)
            agg_b.index_add_(0, g.bi, m)
            b = self.upd_b(torch.cat([b, agg_b], dim=1))
            # body -> constraint
            m = self.msg_b2c(torch.cat([b[g.bi], c[g.ci], g.ef], dim=1))
            agg_c = torch.zeros(g.n_c, self.hid, dtype=c.dtype, device=c.device)
            agg_c.index_add_(0, g.ci, m)
            c = self.upd_c(torch.cat([c, agg_c], dim=1))
        return c

    # --- preconditioner / guess constructors -------------------------------

    def diag_apply(self, g: Graph):
        """Returns r -> d ⊙ r with d>0 (SPD)."""
        log_d = self.head_diag(self.trunk(g)).squeeze(1)
        d = torch.exp(log_d.clamp(-20, 20))
        return lambda r: d * r

    def x0(self, g: Graph):
        return self.head_x0(self.trunk(g)).squeeze(1)

    def lowrank_apply(self, g: Graph, sys: System, k: int = 8):
        """M⁻¹ = Jacobi + Q·diag(gain)·Qᵀ, built in the Jacobi-scaled
        space (the findings say that is the right space — don't relearn
        the units normalisation Jacobi already does).

        The GNN predicts a per-constraint rank-k basis U; we row-scale it
        by D = diag(A)^-1/2 (ties the correction to the Jacobi-scaled
        space), orthonormalise (QR, differentiable) so the gains are
        well-posed, and use softplus gains ≥ 0 → SPD by construction.
        Apply cost is O(n·k): cheap, matrix-free.
        """
        h = self.trunk(g)
        k = min(k, self.lr_k, sys.n)
        U = self.head_lr(h)[:, :k]                       # (n, k)
        gain = torch.nn.functional.softplus(
            self.head_lrg(h.mean(0))[:k])                # (k,) ≥ 0

        diagA = torch.diagonal(sys.A)
        Dm = diagA.clamp_min(1e-30).rsqrt()              # diag(A)^-1/2
        jac = 1.0 / diagA.clamp_min(1e-30)
        Ut = Dm.unsqueeze(1) * U                         # row-scale into scaled space
        Q, _ = torch.linalg.qr(Ut)                       # (n, k) orthonormal

        def apply(r):
            return jac * r + Q @ (gain * (Q.transpose(0, 1) @ r))

        return apply

    def block_apply(self, g: Graph, sys: System, eps=1e-6):
        """Per body-pair learned SPD block. For pair P with members idx,
        predict a lower-triangular L_P from the pair's constraint
        embeddings; M⁻¹ r |_P = (L L^T + eps I)^-1 r_P."""
        h = self.trunk(g)
        groups: dict[int, list[int]] = {}
        for r, p in enumerate(sys.pair_id):
            groups.setdefault(int(p), []).append(r)

        blocks = []
        for members in groups.values():
            idx = torch.tensor(members, device=h.device)
            hk = h[idx]                       # (k, hid)
            k = hk.shape[0]
            # pairwise features -> scalar entry of a (k,k) matrix
            hi = hk.unsqueeze(1).expand(k, k, -1)
            hj = hk.unsqueeze(0).expand(k, k, -1)
            M = self.head_blk(torch.cat([hi, hj], dim=-1)).squeeze(-1)  # (k,k)
            L = torch.tril(M)
            SPD = L @ L.transpose(0, 1) + eps * torch.eye(
                k, dtype=h.dtype, device=h.device)
            blocks.append((idx, torch.linalg.cholesky(SPD)))

        def apply(r):
            z = torch.zeros_like(r)
            for idx, Lc in blocks:
                z[idx] = torch.cholesky_solve(r[idx].unsqueeze(1), Lc).squeeze(1)
            return z

        return apply


def body_features_from_npz(path, device, dtype):
    """Per-body features shared by all systems of a model: [m_inv,
    mean diag(world_inv_inertia)]. world_inv_inertia varies per system
    (pose-dependent) so we use the per-system value at graph-build time;
    here we just return the count. Returns a function sys_idx->feat."""
    d = np.load(path, allow_pickle=True)
    minv = d["body_inv_mass"]       # (S,1,Nb)
    Iinv = d["world_inv_inertia"]   # (S,1,Nb,3,3)

    def feat(s):
        mi = minv[s, 0]                                   # (Nb,)
        di = np.trace(Iinv[s, 0], axis1=1, axis2=2) / 3.0  # (Nb,)
        return np.stack([mi, di], axis=1).astype(np.float64)

    return feat


if __name__ == "__main__":
    # Smoke: build graph for system 0, run a forward of each head.
    from test_scripts.precond_lab import load_systems, DATA

    S = load_systems(dtype=torch.float64)
    bf = body_features_from_npz(DATA, "cpu", torch.float64)
    d = np.load(DATA, allow_pickle=True)
    Jraw = d["J_values"]
    mask = d["constr_active_mask"]

    s0 = S[0]
    act = np.nonzero(mask[0, 0] > 0.0)[0]
    g = build_graph(s0, Jraw[0, 0, act], bf(0))

    net = PrecondGNN().double()
    r = torch.randn(s0.n, dtype=torch.float64)
    print("diag apply    ->", net.diag_apply(g)(r).shape)
    print("x0            ->", net.x0(g).shape)
    print("block apply   ->", net.block_apply(g, s0)(r).shape)
    print("lowrank k=8   ->", net.lowrank_apply(g, s0, 8)(r).shape)
    print("lowrank k=16  ->", net.lowrank_apply(g, s0, 16)(r).shape)
    print(f"params: {sum(p.numel() for p in net.parameters())}")
