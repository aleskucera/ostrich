"""Plot the gradient difficulty-ramp: converged loss vs horizon, per engine.

Reads results/ramp_<engine>_<H>s.json (produced by run_ramp.sh) and draws best
converged loss against horizon on a log-y axis. Engines that diverge or return
non-finite gradients are drawn at a ceiling with an x marker, so the figure
shows the field collapsing to the survivors as the horizon (and thus the stiff
box contact) enters the problem.

Usage:
    python plot_ramp.py [--out ramp.png]
"""
import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# engine label -> (display name, color, marker).  Ostrich/MJX/Semi match the
# paper palette (2196F3 / E91E63 / FF9800).
STYLE = {
    "ostrich":          ("Ostrich",          "#2196F3", "o"),
    "mjx":              ("MJX",              "#E91E63", "s"),
    "semi_implicit":    ("Semi-Implicit",    "#FF9800", "^"),
    "xpbd":             ("XPBD",             "#9C27B0", "v"),
    "brax_positional":  ("Brax (positional)","#4CAF50", "D"),
    "brax_spring":      ("Brax (spring)",    "#795548", "P"),
    "brax_generalized": ("Brax (generalized)","#00BCD4","X"),
}
ORDER = list(STYLE.keys())


def load():
    """label -> {horizon: best_loss (nan if diverged/non-finite)}."""
    data = {}
    for path in glob.glob(os.path.join(RES, "ramp_*.json")):
        m = re.match(r"ramp_(.+)_(\d+)s\.json$", os.path.basename(path))
        if not m:
            continue
        label, H = m.group(1), int(m.group(2))
        if label not in STYLE:
            continue
        with open(path) as f:
            d = json.load(f)
        iters = d.get("iterations", 0)
        # A trial "converged" only if its gradients stayed finite for the whole
        # run. A NaN/inf gradient (or an early stop) means no real optimization
        # happened, even though iteration 0 has a finite loss -- so such runs are
        # marked diverged (ceiling), not plotted as a converged loss.
        best = np.inf
        for tr in d.get("trials", []):
            gnorms = tr.get("grad_norms", [])
            losses = tr.get("losses", [])
            grads_ok = len(gnorms) > 0 and all(np.isfinite(g) for g in gnorms)
            ran_full = iters == 0 or len(losses) >= iters
            finite_losses = [x for x in losses if np.isfinite(x)]
            # require actual descent: a dead/zero gradient (e.g. XPBD) leaves the
            # loss flat or drifting up, which is not a successful optimization.
            improved = bool(finite_losses) and min(finite_losses) < finite_losses[0]
            if grads_ok and ran_full and improved:
                best = min(best, min(finite_losses))
        data.setdefault(label, {})[H] = best if np.isfinite(best) else np.nan
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "ramp.png"))
    ap.add_argument("--ceiling", type=float, default=None,
                    help="y for diverged/NaN markers (default: 3x max finite loss)")
    args = ap.parse_args()

    data = load()
    if not data:
        raise SystemExit(f"no ramp_*.json found in {RES} (run run_ramp.sh first)")

    finite_vals = [v for hv in data.values() for v in hv.values() if np.isfinite(v)]
    ceiling = args.ceiling or (3.0 * max(finite_vals) if finite_vals else 1.0)

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for label in ORDER:
        if label not in data:
            continue
        name, color, marker = STYLE[label]
        hs = sorted(data[label])
        ys = [data[label][h] for h in hs]
        finite = [(h, y) for h, y in zip(hs, ys) if np.isfinite(y)]
        diverged = [h for h, y in zip(hs, ys) if not np.isfinite(y)]
        if finite:
            fx, fy = zip(*finite)
            ax.plot(fx, fy, marker=marker, color=color, lw=1.8, label=name, zorder=3)
        if diverged:
            ax.scatter(diverged, [ceiling] * len(diverged), marker="x",
                       color=color, s=70, linewidths=2.2, zorder=4)

    ax.axhspan(ceiling * 0.7, ceiling * 1.4, color="red", alpha=0.05, zorder=0)
    ax.set_yscale("log")
    ax.set_xlabel("Optimization horizon (s)")
    ax.set_ylabel(r"Best converged loss (m$^2$)")
    ax.set_title("Gradient quality vs horizon (× = diverged / non-finite)")
    ax.set_xticks(sorted({h for hv in data.values() for h in hv}))
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
