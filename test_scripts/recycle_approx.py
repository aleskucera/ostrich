"""De-risking probe: does recycling survive when the subspace is the
APPROXIMATE one a real engine gets for free (harmonic Ritz from the
PCR Krylov vectors), not the exact dense `eigh`?

A deployable recycling solver does NOT eigendecompose A. It harvests the
m residual vectors the solver already generates on system i-1, extracts
the k smallest harmonic-Ritz pairs (an m×m generalised eigenproblem —
cheap), and recycles that approximate subspace into system i. This probe
reproduces exactly that, on the held-out post-impact split, against:
  jacobi (bar) · recycle-exact (eigh, idealised) · ideal (own exact).

If recycle-harmonic ≈ recycle-exact → engine work is de-risked and the
gain quantified. If recycle-harmonic ≈ jacobi → the approximation kills
it; do not build it.

Run:
    python test_scripts/recycle_approx.py
"""

from __future__ import annotations

import numpy as np
import torch

from test_scripts.precond_lab import DATA, MAX_ITERS, load_systems, pcr, jacobi_apply
from test_scripts.ideal_target_stability import bad_subspace, overlap
from test_scripts.recycle_probe import embed, two_level, restrict

DT = torch.float64
K = 16


def krylov_residual_basis(As, rhs, m):
    """m orthonormal vectors spanning the Krylov space the solver walks:
    the (preconditioned) CR residual iterates on the Jacobi-scaled As.
    This is what an engine has 'for free' after solving system i-1."""
    n = As.shape[0]
    m = min(m, n)
    x = torch.zeros(n, dtype=As.dtype)
    r = rhs - As @ x
    R = [r / (torch.linalg.norm(r) + 1e-30)]
    p = r.clone()
    Ap = As @ p
    rs = torch.dot(r, r)
    for _ in range(m - 1):
        a = rs / (torch.dot(Ap, Ap) + 1e-30)          # CR step
        x = x + a * p
        r = r - a * Ap
        R.append(r / (torch.linalg.norm(r) + 1e-30))
        Ar = As @ r
        b = torch.dot(Ar, Ap) / (torch.dot(Ap, Ap) + 1e-30)
        p = r - b * p
        Ap = Ar - b * Ap
        rs = torch.dot(r, r)
        if float(rs) < 1e-24:
            break
    V, _ = torch.linalg.qr(torch.stack(R, dim=1))      # (n, m') orthonormal
    return V


def harmonic_ritz(As, V, k):
    """k smallest harmonic-Ritz vectors of As in span(V).
    Solve (WᵀW) g = θ (VᵀAsV) g,  W = As V, both SPD; take smallest θ."""
    W = As @ V
    M1 = W.transpose(0, 1) @ W                          # SPD
    B = W.transpose(0, 1) @ V                           # = VᵀAsV, SPD
    B = 0.5 * (B + B.transpose(0, 1))
    L = torch.linalg.cholesky(B + 1e-14 * torch.eye(B.shape[0], dtype=B.dtype))
    Li = torch.linalg.inv(L)
    Csym = Li @ M1 @ Li.transpose(0, 1)
    Csym = 0.5 * (Csym + Csym.transpose(0, 1))
    w, U = torch.linalg.eigh(Csym)                      # ascending θ
    G = Li.transpose(0, 1) @ U[:, :min(k, U.shape[1])]   # back-transform
    Y = V @ G                                           # approx small-eig vecs
    return Y


def approx_subspace(A, m, k):
    """Recycled deflation basis from A's own Krylov harvest (orig space)."""
    D = torch.diagonal(A).clamp_min(1e-30).rsqrt()
    As = D.unsqueeze(1) * A * D.unsqueeze(0)
    rhs = torch.randn(A.shape[0], dtype=A.dtype)        # generic excitation
    V = krylov_residual_basis(As, rhs, m)
    Y = harmonic_ritz(As, V, k)
    Z = D.unsqueeze(1) * Y                              # → original space
    Q, _ = torch.linalg.qr(Z)
    return Q


def pct(a, p):
    return float(np.percentile(np.asarray(a, float), p)) if len(a) else float("nan")


def main():
    S = load_systems(dtype=DT)
    d = np.load(DATA, allow_pickle=True)
    mask = d["constr_active_mask"]
    acts = [np.nonzero(mask[s, 0] > 0.0)[0] for s in range(mask.shape[0])]
    cut = int(0.8 * len(S))
    va = list(range(cut, len(S)))
    print(f"{len(S)} systems  val (held-out post-impact) {len(va)}\n")

    Qexact = [bad_subspace(S[i].A, min(K, S[i].n)) for i in range(len(S))]
    Eexact = [embed(Qexact[i], acts[i]) for i in range(len(S))]

    for m in (16, 24, 32):
        Qapx = {}
        sub_ov = []
        for i in range(cut - 1, len(S)):                # only need val + its prev
            Qa = approx_subspace(S[i].A, m, K)
            Qapx[i] = Qa
            kk = min(K, S[i].n, Qexact[i].shape[1], Qa.shape[1])
            sub_ov.append(overlap(Qexact[i][:, :kk], Qa[:, :kk]))

        it_jac, it_rx, it_rh, it_id = [], [], [], []
        for i in va:
            A, b = S[i].A, S[i].b
            _, ij = pcr(A, b, jacobi_apply(A)); it_jac.append(ij)
            _, ii = pcr(A, b, two_level(A, Qexact[i])); it_id.append(ii)
            # recycle previous solve's subspace, restricted to current support
            Ze = restrict(Eexact[i - 1], acts[i])
            _, ie = pcr(A, b, two_level(A, Ze)); it_rx.append(ie)
            Zh = restrict(embed(Qapx[i - 1], acts[i - 1]), acts[i])
            _, ih = pcr(A, b, two_level(A, Zh)); it_rh.append(ih)

        jm = np.mean(it_jac)
        def line(t, a):
            a = np.asarray(a, float)
            return (f"    {t:22s} mean {a.mean():5.1f}  p50 {pct(a,50):4.0f}  "
                    f"p95 {pct(a,95):4.0f}  beats-J {int((a<np.array(it_jac)).sum())}"
                    f"/{len(a)}")
        print(f"m={m:2d} Krylov harvest  (approx⋂exact subspace overlap "
              f"p50 {pct(sub_ov,50):.2f} p05 {pct(sub_ov,5):.2f}):")
        print(line("jacobi (bar)", it_jac))
        print(line("recycle-exact (eigh)", it_rx))
        print(line("recycle-HARMONIC", it_rh))
        print(line("ideal (own exact)", it_id))
        rh = np.mean(it_rh)
        print(f"    → harmonic recycling: {jm:.1f}→{rh:.1f} "
              f"({100*(1-rh/jm):+.0f}%)\n")

    print("verdict: read the recycle-HARMONIC row at m≈24 (≈ what PCR runs "
          "anyway). If it is well below Jacobi and near recycle-exact, the "
          "in-engine spectral-recycling solver is de-risked and worth "
          "building; if it sits at Jacobi, the cheap approximation is "
          "insufficient and engine work is not warranted.")


if __name__ == "__main__":
    main()
