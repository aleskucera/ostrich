"""Gradient error of the estimators on the margin reference pools
(launch_margin_ref.sh): reference from pairs 1000.. (cross-fitted control
variate), estimates from disjoint groups of N pairs in 0..999. Prints the
median relative error ||estimate - reference|| / ||reference||.

  PYTHONPATH=. python margin_grad_err.py
"""

import numpy as np

import method_c as mc

SIGMA = 0.05
MARGINS = ("e_max_all", "e_cross")


def main():
    for d in (10, 20):
        for a in (0.6, 0.7, 0.8):
            D = np.load(f"results/night/margin_ref_d{d}_a{a}.npz")
            Z, L = D["Z"], D["L"]
            P = len(Z) // 2
            eps, yp, ym = Z[:P], L[:P], L[P:]
            ref = slice(1000, P)
            g, se, _ = mc.cv_reference(eps[ref], np.concatenate([yp[ref], ym[ref]]), SIGMA)
            th0 = np.zeros(d)
            for N in (d, 2 * d):
                err = {}
                add = lambda k, v: err.setdefault(k, []).append(np.linalg.norm(v - g) / np.linalg.norm(g))
                for s in range(0, 1000 - N + 1, N):
                    sl = slice(s, s + N)
                    e, yy = eps[sl], np.concatenate([yp[sl], ym[sl]])
                    f = lambda x: yy
                    add("M2", ((yp[sl] - ym[sl])[:, None] * e).mean(0) / (2 * SIGMA))
                    add("linreg", mc.estimate_nd(f, th0, SIGMA, e, jumps=False)[0])
                    add("gated", mc.estimate_gated(f, th0, SIGMA, e)[0])
                    for k in MARGINS:
                        mm = np.concatenate([D[k][:P][sl], D[k][P:][sl]])
                        fm = lambda x: (yy, mm)
                        add(f"margin:{k}", mc.estimate_margin(fm, th0, SIGMA, e)[0])
                        add(f"mquad:{k}", mc.estimate_margin(fm, th0, SIGMA, e, quad=True)[0])
                row = "  ".join(f"{k} {np.median(v):.3f}" for k, v in err.items())
                print(f"d={d} a={a} N={N} ref_se={se / np.linalg.norm(g):.3f} | {row}", flush=True)


if __name__ == "__main__":
    main()
