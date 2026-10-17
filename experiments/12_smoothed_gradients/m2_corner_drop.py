"""M2: the corner drop propagated as a Gaussian mixture, vs the simulator.

Start: x0 ~ N(theta, SIGMA), the rest of the state certain (tiny covariance).
Each step every component is split by the wedge's branch tree:
  contact validity: left face (0 <= s_left <= face length), right face
  (likewise), apex vertex (between the faces); then impact: the ground's
  d_free and the d_free of the wedge feature that applies.
Pieces go through one batched step, linearised at their means with adjoint
Jacobians. Each component carries its contact history: the set of contact
features (apex, left face, right face, ground) it has hit, from its branch
labels. The mixture is reduced within each history only (Runnalls,
K_PER_HISTORY each), so different modes are never merged into one
meaningless mean. The set, not the ordered sequence: a ball rattling
between the apex and a face makes a new sequence every few steps (229
sequences after 51 steps), while at most 16 sets exist.

Compared at T against the dense grid of simulated drops (corner_drop.npz):
right-mode weight, E[L] and the x(T) density.

Run: PYTHONPATH=. python m2_corner_drop.py
Writes results/m2_corner_drop.npz and results/m2_corner_drop.png.
"""

import math
import time

import numpy as np
import torch

import ball_landscape as bl
import corner_drop as cd
import corner_drop_study as cs
import mixture as mx

SIGMA = cs.SIGMA
THETAS = (-0.02, 0.0, 0.01, 0.03)
K_PER_HISTORY = 8  # components kept per contact history
P_MIN = 1e-6


def branch_tree(sim, mirror=False):
    """label -> next boundary g(s) (< 0 means the first branch), or None.
    mirror: split the right face's validity first (for a symmetry check)."""
    f0 = torch.zeros(3, dtype=torch.float64)
    L = sim.face_len
    s_left = lambda s: (s[:2] - sim.apex) @ sim.dir_left
    s_right = lambda s: (s[:2] - sim.apex) @ sim.dir_right
    first, second = (s_right, s_left) if mirror else (s_left, s_right)
    validity = [
        lambda s: -first(s),  # < 0: first face's s > 0
        lambda s: first(s) - L,  # < 0: first face's s < L
        lambda s: -second(s),
        lambda s: second(s) - L,
    ]
    impact = lambda k: (lambda s: sim.free_gap(s[:3], s[3:], f0)[1][k])

    def feature(label):
        on_first = label[0] == -1 and label[1] == -1
        on_second = label[2] == -1 and label[3] == -1
        apex = label[0] == 1 and label[2] == 1
        if apex:
            return 3
        if on_first or on_second:
            return (2 if on_first else 1) if mirror else (1 if on_first else 2)
        return None

    tree_feature = feature

    def tree(label):
        if len(label) < 4:
            return validity[len(label)]
        if len(label) == 4:
            return impact(0)  # ground
        if len(label) == 5:
            k = feature(label)
            return impact(k) if k is not None else None
        return None

    def event(label):
        """The contact this step's branch makes: 'ground', a wedge feature, or None."""
        if len(label) >= 5 and label[4] == -1:
            return "ground"
        if len(label) >= 6 and label[5] == -1:
            return {1: "left", 2: "right", 3: "apex"}[tree_feature(label)]
        return None

    tree.event = event
    return tree


def run(sim, theta, steps=cd.T_STEPS, k_per_history=K_PER_HISTORY, mirror=False):
    mu = torch.tensor([theta, cd.Y0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    Sigma = torch.diag(torch.tensor([SIGMA**2, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10], dtype=torch.float64))
    comps = [((), torch.tensor(1.0, dtype=torch.float64), mu, Sigma)]
    tree = branch_tree(sim, mirror)
    peak = 1
    for _ in range(steps):
        pieces = []
        for history, p, m, S in comps:
            for lab, q, m2, S2 in mx.partition(m, S, tree, P_MIN):
                ev = tree.event(lab)
                new_history = tuple(sorted(set(history) | {ev})) if ev is not None else history
                pieces.append((new_history, p * q, m2, S2))
        total = sum(p for _, p, _, _ in pieces)
        pieces = [(h, p / total, m, S) for h, p, m, S in pieces]
        peak = max(peak, len(pieces))
        comps = mx.reduce_by_history(mx.propagate_batched(sim, pieces), k_per_history)
    return comps, peak


def summarise(comps):
    """Right-mode weight, E[L] and the x(T) marginal (weights, means, stds)."""
    w = torch.stack([p for _, p, _, _ in comps])
    mx_ = torch.stack([m[0] for _, _, m, _ in comps])
    sx = torch.stack([torch.sqrt(S[0, 0].clamp(min=1e-14)) for _, _, _, S in comps])
    w_right = (w * mx.norm_cdf(mx_ / sx)).sum()
    E = (w * ((mx_ - cs.X_TARGET) ** 2 + sx**2)).sum()
    return w_right.item(), E.item(), (w.numpy(), mx_.numpy(), sx.numpy())


def references(theta):
    d = np.load(bl.RESULTS / "corner_drop.npz")
    p = cs.gaussian(d["x"], theta, SIGMA) * cs.GRID_STEP
    return (p * (d["xT"] > 0)).sum(), (p * d["L"]).sum()


def main():
    sim = cd.make_sim()
    out = {}
    print(f"{'theta':>7} {'w_right pred':>12} {'true':>7} {'E[L] pred':>10} {'true':>7} {'peak pieces':>11} {'time':>6}")
    for th in THETAS:
        t0 = time.time()
        comps, peak = run(sim, th)
        w_r, E, marg = summarise(comps)
        w_true, E_true = references(th)
        print(f"{th:+7.3f} {w_r:12.3f} {w_true:7.3f} {E:10.3f} {E_true:7.3f} {peak:11d} {time.time() - t0:5.0f}s", flush=True)
        out[f"{th}"] = np.concatenate([marg[0][:, None], marg[1][:, None], marg[2][:, None]], axis=1)
        out[f"{th}_summary"] = np.array([w_r, w_true, E, E_true, peak])
    np.savez(bl.RESULTS / "m2_corner_drop.npz", **out)
    plot()


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = np.load(bl.RESULTS / "m2_corner_drop.npz")
    d = np.load(bl.RESULTS / "corner_drop.npz")
    ink, muted, c_mix = "#0b0b0b", "#52514e", "#eb6834"
    fig, axes = plt.subplots(1, len(THETAS), figsize=(16, 3.8), constrained_layout=True, sharey=True)
    rng = np.random.default_rng(1)
    xs = np.linspace(-2.3, 2.3, 1200)
    for ax, th in zip(axes, THETAS):
        idx = np.clip(np.rint((th + SIGMA * rng.standard_normal(20000) - d["x"][0]) / cs.GRID_STEP).astype(int), 0, len(d["x"]) - 1)
        ax.hist(d["xT"][idx], bins=120, range=(-2.3, 2.3), density=True, color="#c9ccd1", label="simulator, 20 000 drops")
        comps = r[f"{th}"]
        dens = sum(w * np.exp(-0.5 * ((xs - m) / s) ** 2) / (s * math.sqrt(2 * math.pi)) for w, m, s in comps)
        ax.plot(xs, dens, color=c_mix, lw=1.6, label="mixture prediction")
        w_r, w_true, E, E_true, peak = r[f"{th}_summary"]
        ax.set_title(f"theta = {th}: right-mode weight {w_r:.2f} (true {w_true:.2f})", color=ink, fontsize=10, loc="left")
        ax.set_xlabel("x(T) [m]", color=muted)
        ax.set_ylim(0, 6)
        ax.tick_params(colors=muted)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("density", color=muted)
    axes[0].legend(frameon=False, fontsize=9, loc="upper center")
    fig.suptitle(f"Corner drop: x(T) predicted by mixture propagation ({K_PER_HISTORY} components per contact history) vs the simulator",
                 color=muted, fontsize=10, x=0.01, ha="left")
    fig.savefig(bl.RESULTS / "m2_corner_drop.png", dpi=150)
    print(f"wrote {bl.RESULTS / 'm2_corner_drop.png'}")


if __name__ == "__main__":
    main()
