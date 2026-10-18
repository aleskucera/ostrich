"""M3: the gradient of E[L](theta) from the mixture, vs the true one.

The corner drop's mixture (m2_corner_drop.run) is differentiable in theta:
splits and merges are closed-form torch, and the step backpropagates by the
implicit-function adjoint (mixture._Step). One forward and one backward pass
give E[L] and dE[L]/dtheta. Compared over a theta grid against the dense-grid
quadrature truth (corner_drop.npz: q_E, q_grad); the finite-difference slope
of the mixture's own E[L] curve shows whether the merges' discrete choices
make the curve kinked.

Run one theta: PYTHONPATH=. python m3_gradients.py run THETA
  (writes results/parts/m3_THETA.npz)
Then: PYTHONPATH=. python m3_gradients.py merge
  (writes results/m3_gradients.npz and results/m3_gradients.png)
"""

import sys
import time

import numpy as np
import torch

import ball_landscape as bl
import corner_drop as cd
import corner_drop_study as cs
import m2_corner_drop as m2

THETAS = np.round(np.linspace(-0.06, 0.08, 29), 4)
PARTS = bl.RESULTS / "parts"


def expected_loss(comps):
    """E[(x(T) - X_TARGET)^2] of the mixture, and its right-mode weight."""
    E = sum(p * ((m[0] - cs.X_TARGET) ** 2 + S[0, 0]) for _, p, m, S in comps)
    w_right, _, _ = m2.summarise([(h, p.detach(), m.detach(), S.detach()) for h, p, m, S in comps])
    return E, w_right


def run_one(theta):
    sim = cd.make_sim()
    th = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    t0 = time.time()
    comps, peak = m2.run(sim, th)
    E, w_right = expected_loss(comps)
    t1 = time.time()
    E.backward()
    t2 = time.time()
    PARTS.mkdir(exist_ok=True)
    np.savez(PARTS / f"m3_{theta}.npz", theta=theta, E=E.item(), grad=th.grad.item(), w_right=w_right, peak=peak,
             t_fwd=t1 - t0, t_bwd=t2 - t1)
    print(f"theta {theta:+.4f}: E[L] {E.item():.4f}, dE/dtheta {th.grad.item():+.3f}, right-mode weight {w_right:.3f}, "
          f"forward {t1 - t0:.0f}s, backward {t2 - t1:.0f}s", flush=True)


def merge():
    rows = [np.load(PARTS / f"m3_{th}.npz") for th in THETAS if (PARTS / f"m3_{th}.npz").exists()]
    out = {k: np.array([r[k] for r in rows]) for k in ("theta", "E", "grad", "w_right", "peak", "t_fwd", "t_bwd")}
    d = np.load(bl.RESULTS / "corner_drop.npz")
    out["E_true"] = np.interp(out["theta"], d["thetas"], d["q_E"])
    out["grad_true"] = np.interp(out["theta"], d["thetas"], d["q_grad"])
    out["w_true"] = np.interp(out["theta"], d["thetas"], d["w_right"])
    np.savez(bl.RESULTS / "m3_gradients.npz", **out)
    print(f"{'theta':>7} {'E mix':>7} {'true':>7} {'grad mix':>9} {'true':>8} {'w mix':>6} {'true':>6}")
    for i in range(len(out["theta"])):
        print(f"{out['theta'][i]:+7.3f} {out['E'][i]:7.3f} {out['E_true'][i]:7.3f} {out['grad'][i]:+9.2f} "
              f"{out['grad_true'][i]:+8.2f} {out['w_right'][i]:6.3f} {out['w_true'][i]:6.3f}")
    plot(out, d)


def plot(r, d):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, c_true, c_mix = "#0b0b0b", "#52514e", "#8a8f98", "#eb6834"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    th = r["theta"]
    mid = 0.5 * (th[1:] + th[:-1])
    fd = np.diff(r["E"]) / np.diff(th)
    panels = [
        (axes[0], "E[L](theta)", [(d["thetas"], d["q_E"], c_true, "-", "truth (dense grid)"), (th, r["E"], c_mix, "o-", "mixture")]),
        (axes[1], "dE[L]/dtheta", [(d["thetas"], d["q_grad"], c_true, "-", "truth (dense grid)"),
                                   (th, r["grad"], c_mix, "o-", "mixture, backprop"), (mid, fd, ink, "x", "mixture, slope of its E[L] curve")]),
        (axes[2], "right-mode weight", [(d["thetas"], d["w_right"], c_true, "-", "truth (dense grid)"), (th, r["w_right"], c_mix, "o-", "mixture")]),
    ]
    for ax, title, lines in panels:
        for x, y, c, style, label in lines:
            ax.plot(x, y, style, color=c, lw=1.6, ms=3.5, label=label)
        ax.set_xlim(th.min() - 0.005, th.max() + 0.005)
        ax.set_title(title, color=ink, fontsize=10, loc="left")
        ax.set_xlabel("theta (mean initial x) [m]", color=muted)
        ax.tick_params(colors=muted)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(bl.RESULTS / "m3_gradients.png", dpi=150)
    print(f"wrote {bl.RESULTS / 'm3_gradients.png'}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run_one(float(sys.argv[2]))
    else:
        merge()
