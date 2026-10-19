"""Model margins for the d-dim box: tipping predicted from the push forces
alone, by a reduced model, no simulation.

  lift   largest tipping moment about the front corner over the push phases,
         box upright and sliding: F (h - HY) + m g (mu HY - HX)
  pivot  the box as one rigid body rotating about its front corner (tilt phi,
         clamped at phi >= 0), sliding forward under friction; the push acts
         as in the simulator (force F at the centre, torque -F (h - HY)).
         Moment about the corner, centre at (x_c, y_c) from the corner:
           sliding  F (h - HY) + mu m g y_c - m g x_c
           sticking F (y_c + h - HY) - m g x_c
         Margin: largest energy over the corner barrier,
         1/2 I_c w^2 + m g (y_c - r), frozen when phi passes the tipping
         angle (so >= 0 exactly when the model tips).

  PYTHONPATH=. python margin_model.py   (agreement and smoothness on the
                                         margin reference pools)
"""

import math

import numpy as np

import box_tip as bt

G, DT = 9.81, 0.01
R = math.hypot(bt.HX, bt.HY)
I_C = bt.MASS * (4 * bt.HX**2 + 4 * bt.HY**2) / 12 + bt.MASS * R**2
CRIT = math.atan(bt.HX / bt.HY)


def push_schedule(X):
    """X (B, d) -> F, h per push step (T_PUSH, B), as box_phases.forces."""
    k = X.shape[1] // 2
    phase = np.arange(bt.T_PUSH) * k // bt.T_PUSH
    F = bt.FORCE[0] + (bt.FORCE[1] - bt.FORCE[0]) * X[:, 0::2]
    h = bt.HEIGHT[0] + (bt.HEIGHT[1] - bt.HEIGHT[0]) * X[:, 1::2]
    return F.T[phase], h.T[phase]


def lift(X):
    F, h = push_schedule(X)
    return (F * (h - bt.HY) + bt.MASS * G * (bt.MU * bt.HY - bt.HX)).max(axis=0)


def pivot(X):
    F, h = push_schedule(X)
    B = X.shape[0]
    m, mg = bt.MASS, bt.MASS * G
    phi, w, v = np.zeros(B), np.zeros(B), np.zeros(B)
    best = np.full(B, -np.inf)
    done = np.zeros(B, bool)
    for t in range(bt.T_STEPS):
        Ft, ht = (F[t], h[t]) if t < bt.T_PUSH else (np.zeros(B), np.full(B, bt.HY))
        xc, yc = R * np.sin(CRIT - phi), R * np.cos(CRIT - phi)
        slide = (v > 0) | (Ft > bt.MU * mg)
        tau = np.where(slide, Ft * (ht - bt.HY) + bt.MU * mg * yc, Ft * (yc + ht - bt.HY)) - mg * xc
        v = np.where(slide, np.maximum(v + (Ft / m - bt.MU * G) * DT, 0), 0)
        w = np.where(done, w, w + tau / I_C * DT)
        phi = np.where(done, phi, phi + w * DT)
        rest = phi <= 0
        phi, w = np.where(rest, 0, phi), np.where(rest & (w < 0), 0, w)
        e = 0.5 * I_C * w**2 + mg * (R * np.cos(CRIT - phi) - R)
        best = np.where(done, best, np.maximum(best, e))
        done |= phi >= CRIT
    return best


def main():
    from margin_probe import heldout

    for d in (10, 20):
        for a in (0.6, 0.7, 0.8):
            D = np.load(f"results/night/margin_ref_d{d}_a{a}.npz")
            Z, tip = D["Z"], D["tip"]
            X = D["th"] + 0.05 * Z
            print(f"d={d} a={a} tipped {tip.mean():.3f}")
            for name, m in (("lift", lift(X)), ("pivot", pivot(X)), ("e_cross", D["e_cross"])):
                rl, cl = heldout(Z, m, tip, False)
                rq, cq = heldout(Z, m, tip, True)
                print(f"  {name:8s} sign agrees {np.mean((m > 0) == tip):.3f}  R2 lin {rl:.3f} quad {rq:.3f}"
                      f"  sign(fit)=tip lin {cl:.3f} quad {cq:.3f}")


if __name__ == "__main__":
    main()
