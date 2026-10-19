"""Gradient error with generic contact labels and margins on reference pools.

Pools from contact_margin.py (any scene; the box also joins its energy
margin from margin_ref_d*_a*.npz, same samples). Reference from pairs
1000.. (cross-fitted control variate), estimates from disjoint groups of
N = d pairs in 0..999:

  M2, linreg, gated   baselines, losses only
  labels    estimate_margin with m = +-1. Label source chosen per batch,
            no scene knowledge: the final set of loaded body pairs or the
            set of candidates ever loaded, whichever splits the batch's
            losses better (larger between-group variance); the batch's
            most common signature against the rest
  margin    calibrated candidates o (m - t) (threshold and orientation fitted
            to the labels); among those within AGREE_TOL of the best label
            agreement, the one a linear fit in z explains best (adjusted
            R^2). Its values are used if that R^2 >= R2_MIN and each side
            has at least MIN_SIDE samples, else the labels (a jumpy margin,
            or one calibrated on one or two samples, misleads the fit)
  energy    box only: the energy margin e_cross (scene-specific ablation)

  PYTHONPATH=. python contact_grad_err.py SCENE FILE_PREFIX   e.g. apex apex_ref
"""

import sys

import numpy as np

import method_c as mc

SIGMA = 0.05
AGREE_TOL, R2_MIN, MIN_SIDE = 0.05, 0.9, 3


def adj_r2(z, m):
    """Adjusted R^2 of the least-squares fit m ~ [1, z]."""
    X = np.hstack([np.ones((len(z), 1)), z])
    c, *_ = np.linalg.lstsq(X, m, rcond=None)
    n, p = X.shape
    rss, tss = ((m - X @ c) ** 2).sum(), ((m - m.mean()) ** 2).sum()
    return 1 - (rss / max(n - p, 1)) / max(tss / (n - 1), 1e-300)


def batch_labels(sigs, y):
    """+-1 labels from the signature type that best splits the losses y."""
    best, best_score = None, -1.0
    for s in sigs:
        vals, counts = np.unique(s, return_counts=True)
        lab = s != vals[counts.argmax()]
        if lab.all() or not lab.any():
            continue
        score = lab.mean() * (1 - lab.mean()) * (y[lab].mean() - y[~lab].mean()) ** 2
        if score > best_score:
            best, best_score = lab, score
    return best


def calibrated(m, lab):
    """Threshold and orientation of m that best match the labels:
    (agreement, o (m - t))."""
    order = np.argsort(m)
    ms, ls = m[order], lab[order]
    n = len(m)
    # predict True above the split: agreement for split after position i
    above_true = np.concatenate([np.cumsum(ls[::-1])[::-1], [0]])
    below_false = np.concatenate([[0], np.cumsum(~ls)])
    acc = (above_true + below_false) / n
    acc_rev = 1 - acc
    i, o = (acc.argmax(), 1.0) if acc.max() >= acc_rev.max() else (acc_rev.argmax(), -1.0)
    lo, hi = ms[max(i - 1, 0)], ms[min(i, n - 1)]
    t = 0.5 * (lo + hi) if 0 < i < n else (ms[0] - 1e-9 if i == 0 else ms[-1] + 1e-9)
    return max(acc.max(), acc_rev.max()), o * (m - t)


def main():
    scene, prefix = sys.argv[1], sys.argv[2]
    import glob
    import re

    for fn in sorted(glob.glob(f"results/night/{prefix}_d*_a*.npz"), key=lambda f: (int(re.search(r"_d(\d+)_", f)[1]), f)):
        K = np.load(fn)
        d = int(re.search(r"_d(\d+)_", fn)[1])
        Z, L = K["Z"], K["L"]
        P = len(Z) // 2
        eps, yp, ym = Z[:P], L[:P], L[P:]
        sigs = [K["sig"]] + ([K["sig_ever"]] if "sig_ever" in K.files else [])
        C = {k: K[k] for k in K.files if k.startswith(("min_", "max_", "end_", "sum_", "first_"))}
        C = {k: v for k, v in C.items() if np.isfinite(v).all() and v.std() > 1e-12}
        energy = None
        if scene == "box":
            energy = np.load(fn.replace("contact_ref", "margin_ref"))["e_cross"]
        g, se, _ = mc.cv_reference(eps[1000:], np.concatenate([yp[1000:], ym[1000:]]), SIGMA)
        th0, N = np.zeros(d), d
        err, picks = {}, {}
        add = lambda k, v: err.setdefault(k, []).append(np.linalg.norm(v - g) / np.linalg.norm(g))
        for s in range(0, 1000 - N + 1, N):
            idx = np.concatenate([np.arange(s, s + N), P + np.arange(s, s + N)])
            e, yy = eps[s : s + N], L[idx]
            add("M2", ((yp[s : s + N] - ym[s : s + N])[:, None] * e).mean(0) / (2 * SIGMA))
            add("linreg", mc.estimate_nd(lambda x: yy, th0, SIGMA, e, jumps=False)[0])
            add("gated", mc.estimate_gated(lambda x: yy, th0, SIGMA, e)[0])
            lab = batch_labels([sg[idx] for sg in sigs], yy)
            if lab is None:  # one side only in this batch: no jump to place
                for k in ("labels", "margin"):
                    add(k, mc.estimate_nd(lambda x: yy, th0, SIGMA, e, jumps=False)[0])
            else:
                add("labels", mc.estimate_margin(lambda x: (yy, np.where(lab, 1.0, -1.0)), th0, SIGMA, e)[0])
                cal = {k: calibrated(v[idx], lab) for k, v in C.items()}
                top_acc = max(a for a, _ in cal.values())
                zz = np.concatenate([e, -e])
                name, r2 = max(((k, adj_r2(zz, m)) for k, (a, m) in cal.items() if a >= top_acc - AGREE_TOL), key=lambda t: t[1])
                if r2 >= R2_MIN and min(lab.sum(), (~lab).sum()) >= MIN_SIDE:
                    picks[name] = picks.get(name, 0) + 1
                    add("margin", mc.estimate_margin(lambda x: (yy, cal[name][1]), th0, SIGMA, e)[0])
                else:
                    picks["(labels)"] = picks.get("(labels)", 0) + 1
                    add("margin", mc.estimate_margin(lambda x: (yy, np.where(lab, 1.0, -1.0)), th0, SIGMA, e)[0])
            if energy is not None:
                add("energy", mc.estimate_margin(lambda x: (yy, energy[idx]), th0, SIGMA, e)[0])
        row = "  ".join(f"{k} {np.median(v):.3f}" for k, v in err.items())
        top = sorted(picks.items(), key=lambda t: -t[1])[:3]
        a = re.search(r"_a([\d.]+)\.npz", fn)[1]
        print(f"d={d} a={a} N={N} outcome={K['tip'].mean():.2f} ref_se={se / np.linalg.norm(g):.3f} | {row} | picked {top}", flush=True)


if __name__ == "__main__":
    main()
