"""Compare recovered wheel-velocity splines against real joint setpoints.

Loads results/axion.json (which now includes per-trial init_params and
final_params after the rerun) and overlays the recovered spline against
the recorded joint setpoints from the GT JSON.

Output: results/recovered_control.png
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
GT_PATH = HERE.parent / "1_sim_to_real_box" / "data" / "run_2026_05_20-18_10_33.json"
RES = HERE / "results"
WHEEL_NAMES = ["left", "right", "rear"]
COLOR = "#2196F3"  # Axion blue

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def make_interp_matrix(T, K):
    W = np.zeros((T, K), dtype=np.float64)
    for t in range(T):
        k_float = t * (K - 1) / max(T - 1, 1)
        k_low = int(k_float)
        k_high = min(k_low + 1, K - 1)
        alpha = k_float - k_low
        W[t, k_low] += 1.0 - alpha
        W[t, k_high] += alpha
    return W


def main():
    axion = json.load(open(RES / "axion.json"))
    gt = json.load(open(GT_PATH))

    K = axion["K"]
    dt = axion["dt"]
    horizon_s = axion["horizon_s"]
    T = int(round(horizon_s / dt))
    W = make_interp_matrix(T, K)
    t_sim = np.arange(T) * dt

    real_t = np.array(gt["control"]["t"])
    real_lrr = np.array(gt["control"]["lrr"])
    mask = real_t <= horizon_s
    real_t = real_t[mask]
    real_lrr = real_lrr[mask]

    def lrr_from_params(params):
        """params: [K, 2] (left, right). Returns [T, 3] (left, right, rear=(L+R)/2)."""
        p = np.asarray(params)
        lr = W @ p                                    # [T, 2]
        rear = 0.5 * (lr[:, 0] + lr[:, 1])
        return np.column_stack([lr[:, 0], lr[:, 1], rear])

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.0), sharey=True)
    for w, (ax, name) in enumerate(zip(axes, WHEEL_NAMES)):
        ax.plot(real_t, real_lrr[:, w], "k--", lw=1.6, label="Real",
                zorder=10)
        for ti, trial in enumerate(axion["trials"]):
            init_lrr = lrr_from_params(trial["init_params"])
            final_lrr = lrr_from_params(trial["final_params"])
            ax.plot(t_sim, init_lrr[:, w], color="lightgray", lw=1.0,
                    alpha=0.7, zorder=3,
                    label="initial" if ti == 0 else None)
            ax.plot(t_sim, final_lrr[:, w], color=COLOR, lw=1.4, alpha=0.85,
                    zorder=5,
                    label="recovered (Axion)" if ti == 0 else None)
        ax.set_xlabel("time (s)")
        if w == 0:
            ax.set_ylabel("wheel velocity (rad/s)")
        ax.set_title(f"{name} wheel")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.04), fontsize=11)
    plt.subplots_adjust(bottom=0.25, top=0.88)
    out = RES / "recovered_control.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
