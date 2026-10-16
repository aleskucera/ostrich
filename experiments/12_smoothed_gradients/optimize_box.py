"""Box scenario: optimization per gradient estimator, with large annealed sigma.

Same loop as optimize_ball.optimize. The stuck region (push below the
friction limit) is a flat plateau L = 0.25 that sigma = 0.05 (15 N) cannot
see across, so the schedules include large annealed sigma: 0.3 is +-90 N
and 27 deg. Sample points are clamped into [0, 1]^2 (no negative push).
Besides the exact gradient, the Ostrich-style adjoint (friction weight held
fixed, contact compliance 1e-6) runs as a second gradient-only baseline.
Results are split by whether a start begins on the plateau.

Run: PYTHONPATH=. python optimize_box.py   (~40 min with 6 threads)
Writes results/optimize_box.npz and results/optimize_box.png.
"""

import numpy as np
import torch

import ball_landscape as bl
import box_scenario as bx
import optimize_ball as ob

SIGMAS = {"0.05": (0.05, 0.05), "0.15 -> 0.005": (0.15, 0.005), "0.3 -> 0.01": (0.3, 0.01)}
N_PAIRS, ITERS, STARTS = 4, 100, 12
PLATEAU = (bx.X_TARGET - 0.0) ** 2  # loss of a box that never moves


def main():
    sim = bx.make_sim()
    gen = torch.Generator().manual_seed(2)
    u0 = 0.05 + 0.9 * torch.rand(STARTS, 2, generator=gen, dtype=torch.float64)
    exact = lambda u: ob.batched(lambda x: bx.loss_and_grad(sim, x), u)
    ostrich = lambda u: ob.batched(lambda x: bx.loss_and_grad(sim, x, friction="frozen", contact_compliance=1e-6), u)
    out = ob.optimize(exact, u0, SIGMAS, gen, n_pairs=N_PAIRS, iters=ITERS,
                      project=lambda x: x.clamp(0.0, 1.0), extra_hard=[("Ostrich adjoint", ostrich)])
    np.savez(bl.RESULTS / "optimize_box.npz", **out)
    report()


def report():
    d = np.load(bl.RESULTS / "optimize_box.npz")
    on_plateau = np.abs(d["loss"][0, 0] - PLATEAU) < 1e-4  # starts are shared by all runs
    groups = (("all", np.ones_like(on_plateau)), ("plateau", on_plateau), ("elsewhere", ~on_plateau))
    print(f"\n{STARTS} starts, {on_plateau.sum()} on the plateau. Final loss: median [success L < 0.01 count] per group")
    print(f"{'method':>16} {'sigma':>14}  " + "  ".join(f"{g:>16}" for g, _ in groups))
    cell = lambda final, mask: f"{np.median(final[mask]):.4f} [{(final[mask] < 0.01).sum()}/{mask.sum()}]"
    for k, name in enumerate(d["hard_names"]):
        final = d["hard_loss"][k, -1]
        print(f"{name:>16} {'-':>14}  " + "  ".join(f"{cell(final, m):>16}" for _, m in groups))
    for k, cfg in enumerate(d["configs"]):
        m_, s_ = cfg.split("|")
        final = d["loss"][-1, k]
        print(f"{m_:>16} {s_:>14}  " + "  ".join(f"{cell(final, m):>16}" for _, m in groups))
    plot(d, on_plateau)


def plot(d, on_plateau):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#0b0b0b", "#52514e"
    styles = {"first": ("first-order", "#2a78d6", "o"), "zeroth": ("zeroth-order", "#eb6834", "s"),
              "alpha": ("alpha-order", "#1baf7a", "^")}
    configs = [c.split("|") for c in d["configs"]]
    fig, axes = plt.subplots(2, len(SIGMAS), figsize=(13, 7), sharex=True, sharey=True, constrained_layout=True)
    for row, (label, mask) in enumerate((("starts on the plateau", on_plateau), ("other starts", ~on_plateau))):
        for col, name in enumerate(SIGMAS):
            ax = axes[row, col]
            for k, (hname, ls) in enumerate(zip(d["hard_names"], ("--", ":"))):
                ax.plot(np.maximum(d["hard_rollouts"], 1), np.median(d["hard_loss"][k][:, mask], axis=1),
                        color=muted, lw=1.3, ls=ls, label=hname)
            for k, (m, n) in enumerate(configs):
                if n != name:
                    continue
                lab, color, marker = styles[m]
                ax.plot(np.maximum(d["rollouts"], 1), np.median(d["loss"][:, k][:, mask], axis=1), color=color, lw=2,
                        marker=marker, ms=4, markevery=4, label=lab)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"sigma = {name}, {label} ({mask.sum()})", color=ink, fontsize=10, loc="left")
            ax.grid(alpha=0.3, lw=0.5)
            ax.tick_params(colors=muted)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("forward rollouts", color=muted)
    for ax in axes[:, 0]:
        ax.set_ylabel("median deterministic loss", color=muted)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, loc="outside upper right", ncols=5, fontsize=9)
    fig.suptitle("Pushed box (Ostrich friction): Adam with each gradient estimator",
                 color=muted, fontsize=10, x=0.01, ha="left")
    fig.savefig(bl.RESULTS / "optimize_box.png", dpi=150)
    print(f"wrote {bl.RESULTS / 'optimize_box.png'}")


if __name__ == "__main__":
    main()
