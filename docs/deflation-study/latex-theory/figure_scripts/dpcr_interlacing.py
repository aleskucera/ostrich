"""Figures for the theory document (docs/deflation-study/latex-theory).

Interlacing of the deflated, Jacobi-preconditioned spectrum (eq. dpcr-interlacing):
a synthetic SPD A with 4 soft outlier modes, P = diag(A), k = 4. Plots the
eigenvalues λ_j(P⁻¹A), the band [λ_j, λ_{j+k}], and the nonzero eigenvalues μ_j
of P⁻¹P̃A for Z = exact bottom-k eigenvectors of P⁻¹A and for a random Z.

Run:
    python docs/deflation-study/latex-theory/figure_scripts/dpcr_interlacing.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from _common import save_fig

DT = torch.float64
BLUE, ORANGE, INK, MUTED, BAND = "#2a78d6", "#eb6834", "#2b2b29", "#8a8980", "#e4e3dd"


def synthetic_A(n=40, n_soft=4, seed=0):
    """SPD A = Ds U Λ Uᵀ Ds: n_soft eigenvalues in [1e-4, 1e-3], the rest in [0.3, 3],
    with a diagonal scaling Ds over three orders of magnitude (so Jacobi matters)."""
    g = torch.Generator().manual_seed(seed)
    U = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=DT))[0]
    lam = torch.cat([torch.logspace(-4, -3, n_soft, dtype=DT),
                     torch.empty(n - n_soft, dtype=DT).uniform_(0.3, 3.0, generator=g)])
    Ds = torch.diag(torch.logspace(-1.5, 1.5, n, dtype=DT)[torch.randperm(n, generator=g)])
    return Ds @ U @ torch.diag(lam) @ U.T @ Ds


def deflated_spectrum(A, Z):
    """Nonzero eigenvalues of P⁻¹P̃A, P = diag(A), via the symmetric form S = P^-½ P̃A P^-½."""
    k = Z.shape[1]
    E = Z.T @ A @ Z
    PtA = A - A @ Z @ torch.linalg.solve(E, Z.T @ A)
    d = torch.diagonal(A).rsqrt()
    S = d[:, None] * PtA * d[None, :]
    return torch.linalg.eigvalsh(0.5 * (S + S.T))[k:]


def interlacing_figure(k=4):
    A = synthetic_A()
    n = A.shape[0]
    d = torch.diagonal(A).rsqrt()
    lam, V = torch.linalg.eigh(d[:, None] * A * d[None, :])          # spectrum of P⁻¹A
    Z_exact = d[:, None] * V[:, :k]                                    # eigvecs of P⁻¹A
    Z_rand = torch.randn(n, k, generator=torch.Generator().manual_seed(1), dtype=DT)
    mu_exact, mu_rand = deflated_spectrum(A, Z_exact), deflated_spectrum(A, Z_rand)
    lo, hi = lam[:n - k], lam[k:]
    assert bool((lo <= mu_rand * (1 + 1e-9)).all() and (mu_rand <= hi * (1 + 1e-9)).all())
    assert torch.allclose(mu_exact, hi, rtol=1e-8)

    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    j = torch.arange(1, n + 1)
    jd = torch.arange(1, n - k + 1)
    ax.fill_between(jd, lo, hi, step=None, color=BAND, lw=0,
                    label=r"interlacing band $[\lambda_j, \lambda_{j+k}]$")
    ax.plot(j, lam, "o", ms=5, mfc="none", mec=MUTED, mew=1.0,
            label=r"$\lambda_j(\mathbf{P}^{-1}\mathbf{A})$, no deflation")
    ax.plot(jd, mu_exact, "s", ms=4, color=BLUE, label=r"$\mu_j$, exact $\mathbf{Z}$ (bottom-$k$ eigenvectors)")
    ax.plot(jd, mu_rand, "^", ms=4.5, color=ORANGE, label=r"$\mu_j$, random $\mathbf{Z}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"index $j$ (ascending)")
    ax.set_ylabel("eigenvalue")
    ax.set_xlim(0, n + 1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK)
    ax.grid(axis="y", color=BAND, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_fig(fig, __file__)
    print(f"κ(P⁻¹A) {float(lam[-1] / lam[0]):.1e} | exact Z κ_eff {float(mu_exact[-1] / mu_exact[0]):.1e}"
          f" | random Z κ_eff {float(mu_rand[-1] / mu_rand[0]):.1e}")


if __name__ == "__main__":
    interlacing_figure()
