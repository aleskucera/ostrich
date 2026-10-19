"""Gradient error with generic contact margins on the reference pools.

Joins margin_ref_d*_a*.npz (losses, energy margin) with contact_ref_d*_a*.npz
(contact candidates and signatures from contact_margin.py, same samples).
Reference from pairs 1000.. (cross-fitted control variate), estimates from
disjoint groups of N = d pairs in 0..999:

  gated     baseline, losses only
  labels    estimate_margin with m = +-1: final contact signature, the
            batch's most common signature against the rest
  contact   the contact candidate whose sign best matches those labels in
            the batch (chosen per batch, no scene knowledge), its values
  energy    scene-specific energy margin e_cross (ablation)

  PYTHONPATH=. python contact_grad_err.py
"""

import numpy as np

import method_c as mc

SIGMA = 0.05


def contact_margin(C, sig, sl):
    """Labels and the best-matching candidate margin for the samples sl."""
    s = sig[sl]
    vals, counts = np.unique(s, return_counts=True)
    label = s != vals[counts.argmax()]
    best, best_acc = None, -1.0
    for name, m in C.items():
        acc = np.mean((m[sl] > 0) == label)
        o = 1.0 if acc >= 0.5 else -1.0
        acc = max(acc, 1 - acc)
        if acc > best_acc:
            best, best_acc = (name, o * m[sl]), acc
    return np.where(label, 1.0, -1.0), best


def main():
    for d in (10, 20):
        for a in (0.6, 0.7, 0.8):
            D = np.load(f"results/night/margin_ref_d{d}_a{a}.npz")
            K = np.load(f"results/night/contact_ref_d{d}_a{a}.npz")
            assert np.allclose(D["Z"], K["Z"])
            Z, L = D["Z"], D["L"]
            P = len(Z) // 2
            eps, yp, ym = Z[:P], L[:P], L[P:]
            sig = K["sig"]
            C = {k: K[k] for k in K.files if k.startswith(("min_s", "max_s", "end_s", "slack"))}
            g, se, _ = mc.cv_reference(eps[1000:], np.concatenate([yp[1000:], ym[1000:]]), SIGMA)
            th0, N = np.zeros(d), d
            err, picks = {}, {}
            add = lambda k, v: err.setdefault(k, []).append(np.linalg.norm(v - g) / np.linalg.norm(g))
            for s in range(0, 1000 - N + 1, N):
                idx = np.concatenate([np.arange(s, s + N), P + np.arange(s, s + N)])
                e, yy = eps[s : s + N], L[idx]
                lab, (name, m) = contact_margin(C, sig, idx)
                picks[name] = picks.get(name, 0) + 1
                add("gated", mc.estimate_gated(lambda x: yy, th0, SIGMA, e)[0])
                add("labels", mc.estimate_margin(lambda x: (yy, lab), th0, SIGMA, e)[0])
                add("contact", mc.estimate_margin(lambda x: (yy, m), th0, SIGMA, e)[0])
                add("energy", mc.estimate_margin(lambda x: (yy, D["e_cross"][idx]), th0, SIGMA, e)[0])
            row = "  ".join(f"{k} {np.median(v):.3f}" for k, v in err.items())
            top = sorted(picks.items(), key=lambda t: -t[1])[:3]
            print(f"d={d} a={a} N={N} ref_se={se / np.linalg.norm(g):.3f} | {row} | picked {top}", flush=True)


if __name__ == "__main__":
    main()
