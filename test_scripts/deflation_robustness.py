"""Step 1d — robustness curve: iterations vs subspace error, additive vs
projection deflation.

Geometry: everything lives in the Jacobi-SCALED space As = D A D,
D = diag(A)^-½, where the preconditioned operator is symmetric and
bad_subspace's eigenvectors V are orthonormal. (Angles measured in the
original space are meaningless here: diag(A) spans orders of magnitude,
so a 5° tilt in original space is a huge tilt in the scaled space.)

Perturbation with a controlled angle θ, k = min(16, n // 2):

    Zs_θ = V_k cos θ + W sin θ,   Wᵀ V_k = 0,   Z = D Zs_θ

EVERY principal angle between span(Zs_θ) and span(V_k) equals θ. Two
kinds of W:
  random    isotropic random directions (3 seeds)
  spectral  W = V[:, k:2k]: each bad mode mixes with a NEIGHBOURING
            eigenvector — the kind of error Ritz/recycled vectors make

Real k = 16 Z sources (deflation_ab.py) are placed on the same axis by
their RMS angle θ_eq = arccos √(mean cos²), with their per-mode angles
printed — they turn out NOT to follow the uniform-angle curve.

Held-out split (last 20% of systems), both scenes.
Writes docs/deflation_robustness.png.

Run:
    python test_scripts/deflation_robustness.py
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from test_scripts.precond_lab import jacobi_apply, load_systems, pcr
from test_scripts.recycle_probe import pct, two_level
from test_scripts.deflation import DT, SCENES, deflated_pcr
from test_scripts.deflation_ab import z_sources

K = 16
ANGLES = (0, 0.1, 0.3, 1, 3, 10, 20, 30, 45, 60, 90)
SEEDS = (0, 1, 2)
K16_SOURCES = ("cross-step (N-1)", "recycle-exact (prev)", "harmonic m=16",
               "harmonic m=24", "harmonic m=32", "gcrodr accumulated")
MODES = (1, 2, 4, 8, 12, 16)
OUT = pathlib.Path(__file__).resolve().parent.parent / "docs/deflation_robustness.png"


def scaled_eig(A):
    D = torch.diagonal(A).clamp_min(1e-30).rsqrt()
    lam, V = torch.linalg.eigh(D.unsqueeze(1) * A * D.unsqueeze(0))
    return D, lam, V


def rotated(D, V, k, theta_deg, kind, gen):
    Vk = V[:, :k]
    if kind == "spectral":
        W = V[:, k:2 * k]
    else:
        W = torch.randn(V.shape[0], k, dtype=DT, generator=gen)
        W = torch.linalg.qr(W - Vk @ (Vk.transpose(0, 1) @ W))[0]
    t = np.deg2rad(theta_deg)
    return D.unsqueeze(1) * (Vk * np.cos(t) + W * np.sin(t))


def mode_angles(D, V, Z, k):
    """Angle (°) of each of the bottom-k scaled eigenvectors to span(D⁻¹Z)."""
    Zs = torch.linalg.qr(Z / D.unsqueeze(1))[0]
    c = torch.linalg.norm(Zs.transpose(0, 1) @ V[:, :k], dim=0).clamp(max=1.0)
    return np.degrees(np.arccos(c.numpy()))


def run_scene(label, path):
    S = load_systems(path=path, dtype=DT)
    va = list(range(int(0.8 * len(S)), len(S)))
    jac = np.array([pcr(S[i].A, S[i].b, jacobi_apply(S[i].A))[1] for i in va], float)
    eig = {i: scaled_eig(S[i].A) for i in va}

    print(f"\n===== {label}: held-out {len(va)} systems, Jacobi mean {jac.mean():.1f} "
          f"p50 {pct(jac,50):.0f} p90 {pct(jac,90):.0f} =====")
    print(f"  {'':5s} {'θ°':>5s} | {'additive':^20s} | {'deflation':^20s}")
    print(f"  {'':5s} {'':>5s} | {'mean  p50  p90':20s} | {'mean  p50  p90':20s}")
    curves = {}
    for kind in ("random", "spectral"):
        seeds = SEEDS if kind == "random" else (0,)
        for th in ANGLES:
            its = {"additive": [], "deflation": []}
            for seed in seeds:
                gen = torch.Generator().manual_seed(seed)
                for i in va:
                    A, b = S[i].A, S[i].b
                    D, _, V = eig[i]
                    Z = rotated(D, V, min(K, S[i].n // 2), th, kind, gen)
                    its["additive"].append(pcr(A, b, two_level(A, Z))[1])
                    its["deflation"].append(deflated_pcr(A, b, Z, jacobi_apply(A))[1])
            cells = []
            for key in ("additive", "deflation"):
                a = np.array(its[key], float)
                curves.setdefault((kind, key), []).append((pct(a, 50), pct(a, 90)))
                cells.append(f"{a.mean():5.1f} {pct(a,50):4.0f} {pct(a,90):4.0f}")
            print(f"  {kind[:5]:5s} {th:5g} | {cells[0]:20s} | {cells[1]:20s}")

    src = z_sources(S, np.load(path, allow_pickle=True), va)
    points = {}
    print("  real k=16 sources — per-mode angle (°) of scaled eigvec i to span(Z), p50;"
          " θ_eq = RMS; deflation iters p50:")
    print(f"    {'source':22s} " + " ".join(f"i={m:<3d}" for m in MODES) + "  θ_eq  iters")
    for name in K16_SOURCES:
        per, eq, its = [], [], []
        for i, Z in src[name].items():
            D, _, V = eig[i]
            k = min(K, S[i].n)
            ang = mode_angles(D, V, Z, k)
            eq.append(np.degrees(np.arccos(np.sqrt(np.mean(np.cos(np.radians(ang)) ** 2)))))
            if k == K:
                per.append(ang)
            its.append(deflated_pcr(S[i].A, S[i].b, Z, jacobi_apply(S[i].A))[1])
        per = np.array(per)
        points[name] = (pct(eq, 50), pct(its, 50))
        print(f"    {name:22s} " + " ".join(f"{np.median(per[:, m - 1]):5.1f}" for m in MODES)
              + f"  {pct(eq,50):4.0f}  {pct(its,50):5.0f}")
    gaps = [float(eig[i][1][0] / eig[i][1][min(K, S[i].n - 1)]) for i in va]
    print(f"  spectral gap λ1/λ17 p50 {pct(gaps,50):.0e}")
    return jac, curves, points


def main():
    fig, axes = plt.subplots(1, len(SCENES), figsize=(13, 5))
    x = np.arange(len(ANGLES))
    for ax, (label, path) in zip(axes, SCENES.items()):
        jac, curves, points = run_scene(label, path)
        style = {("random", "additive"): ("tab:red", "--"),
                 ("random", "deflation"): ("tab:blue", "--"),
                 ("spectral", "additive"): ("tab:red", "-"),
                 ("spectral", "deflation"): ("tab:blue", "-")}
        for (kind, key), arr in curves.items():
            c, ls = style[(kind, key)]
            ax.plot(x, np.array(arr)[:, 0], marker="o", ms=3, color=c, ls=ls,
                    label=f"{key}, {kind} error (median)")
        ax.axhline(pct(jac, 50), color="k", ls=":", lw=1, label="Jacobi median")
        ax.axhline(0.8 * pct(jac, 50), color="gray", ls="-.", lw=1, label="gate: −20%")
        for name, (th, it) in points.items():   # place real sources on the angle axis
            xp = np.interp(th, ANGLES, x)
            ax.plot(xp, it, "g*", ms=9)
            ax.annotate(name, (xp, it), fontsize=7, color="tab:green",
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xticks(x, [f"{a:g}" for a in ANGLES])
        ax.set_title(f"{label} (held-out, k=16, deflation of bottom-k Jacobi-scaled eigvecs)")
        ax.set_xlabel("principal angle θ to exact bottom-16, scaled space (°, non-linear axis)")
        ax.set_ylabel("PCR iterations (median)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
