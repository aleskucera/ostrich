"""How accurate is the IFT adjoint when its linear solve is capped PCR?

The engine's backward step (base_engine.step_backward) solves, at the
converged state,
    d   = M⁻¹ f,   f = −∇_u L − h Gᵀ∇_q L        (compute_body_adjoint_init)
    A w = J d                                    (PCR, same tol/cap as forward, x0 = 0)
    w_u = d − M⁻¹ Jᵀ w                           (subtract_constraint_feedback)
and w (constraint rows) also seeds the pose VJP. With the default
friction_linearization = "true" and adjoint.regularization = 0, A is the
forward operator at the converged state. Offline stand-in: the LAST NR
system of each dumped step (C not zeroed), i.e. the operator the final
NR iteration used.

Loss gradients f (per body, spatial [linear, angular]):
  random     Gaussian on all bodies
  chassis    Gaussian on body 0 only (chassis, 106 kg) — e.g. a loss on
             the final chassis pose/velocity

Solvers, float32 except the reference:
  exact            float64 dense solve (reference)
  PCR Jacobi       engine settings (1e-5/cap16 and yaml 1e-3/cap26)
  PCR joint/ctrl   joint/control deflation, same settings
  body-space       24×24 Woodbury Cholesky for w (+1 refinement step),
                   then w_u = d − M⁻¹Jᵀw as the engine does
  body-space native  w_u = (M + Jᵀ C⁻¹ J)⁻¹ f directly, w = C⁻¹ J w_u

Metrics: relative error of w_u (body-space adjoint = the gradient
carried to the previous step, up to M) and of w (constraint adjoint),
and the cosine between computed and exact w_u. Single step only: over a
horizon these errors compound through the chain of step Jacobians.

Run:
    python -m test_scripts.adjoint_accuracy
"""

from __future__ import annotations

import numpy as np
import torch

from test_scripts.precond_lab import _block_minv, jacobi_apply, load_systems
from test_scripts.recycle_probe import pct
from test_scripts.deflation import DT, SCENES, deflated_pcr
from test_scripts.deflation_cost import SETTINGS

N_DRAWS = 8


def last_nr_systems(d, nS):
    """Index of the last NR system of each step (C not zeroed)."""
    step, it, cz = d["step_idx"][:nS], d["iter_in_step"][:nS], d["c_all_zero"][:nS]
    out = []
    for s in np.unique(step):
        idx = [i for i in np.nonzero(step == s)[0] if not cz[i]]
        if idx:
            out.append(max(idx, key=lambda i: it[i]))
    return out


def minv_dense(d, i, Nb):
    Mi = np.zeros((6 * Nb, 6 * Nb))
    for b in range(Nb):
        Mi[6 * b:6 * b + 6, 6 * b:6 * b + 6] = _block_minv(float(d["body_inv_mass"][i, 0, b]),
                                                           d["world_inv_inertia"][i, 0, b])
    return torch.tensor(Mi, dtype=DT)


def body_space(Jm, Minv, c, rhs, refine):
    """Woodbury solve of (Jm Minv Jmᵀ + diag c) w = rhs in float32."""
    Jm, Minv, c, rhs = Jm.float(), Minv.float(), c.float(), rhs.float()
    M = torch.linalg.inv(Minv)
    ci = 1.0 / c
    L = torch.linalg.cholesky(M + Jm.T @ (ci.unsqueeze(1) * Jm))

    def solve(r):
        v = torch.cholesky_solve((Jm.T @ (ci * r)).unsqueeze(1), L).squeeze(1)
        return ci * (r - Jm @ v)
    w = solve(rhs)
    if refine:
        w = w + solve(rhs - (Jm @ (Minv @ (Jm.T @ w)) + c * w))
    return w


def body_space_native(Jm, Minv, c, f):
    """Adjoint directly in body space, float32: by Woodbury
    w_u = d − M⁻¹Jᵀ A⁻¹ J d = (M + Jᵀ C⁻¹ J)⁻¹ f, and w = C⁻¹ J w_u
    (from A w = J d ⇒ J w_u = C w). No cancellation in w_u."""
    Jm, Minv, c, f = Jm.float(), Minv.float(), c.float(), f.float()
    K = torch.linalg.inv(Minv) + Jm.T @ (Jm / c.unsqueeze(1))
    wu = torch.cholesky_solve(f.unsqueeze(1), torch.linalg.cholesky(K)).squeeze(1)
    return (Jm @ wu) / c, wu


def main():
    rng = np.random.default_rng(0)
    for label, path in SCENES.items():
        S = load_systems(path=path, dtype=DT)
        d = np.load(path, allow_pickle=True)
        Nb = int(d["meta"][0]["N_b"])
        off_n = int(d["meta"][0]["offset_n"])
        idx = last_nr_systems(d, len(S))
        print(f"\n===== {label}: {len(idx)} converged-state systems (last NR iter per step), "
              f"{N_DRAWS} loss draws each =====")
        for loss in ("random", "chassis"):
            errs = {}
            for i in idx:
                A, Jm, n = S[i].A, S[i].Jm, S[i].n
                c = torch.diagonal(A - Jm @ minv_dense(d, i, Nb) @ Jm.T)       # C + reg as assembled
                Minv = minv_dense(d, i, Nb)
                act = np.nonzero(d["constr_active_mask"][i, 0] > 0.0)[0]
                jrows = [r for r, g in enumerate(act) if g < off_n]
                Zjc = torch.eye(n, dtype=DT)[:, jrows]
                for _ in range(N_DRAWS):
                    f = torch.zeros(6 * Nb, dtype=DT)
                    if loss == "random":
                        f[:] = torch.from_numpy(rng.standard_normal(6 * Nb))
                    else:
                        f[:6] = torch.from_numpy(rng.standard_normal(6))
                    dvec = Minv @ f
                    rhs = Jm @ dvec
                    w_ex = torch.linalg.solve(A, rhs)
                    wu_ex = dvec - Minv @ (Jm.T @ w_ex)

                    cands = {}
                    for sname, (tol, cap) in SETTINGS.items():
                        Af, bf = A.float(), rhs.float()
                        w, _, _ = deflated_pcr(Af, bf, torch.zeros(n, 0), jacobi_apply(Af),
                                               max_iters=cap, tol=tol, atol=tol)
                        cands[f"PCR Jacobi {sname}"] = (w, None)
                        w, _, _ = deflated_pcr(Af, bf, Zjc.float(), jacobi_apply(Af),
                                               max_iters=cap, tol=tol, atol=tol)
                        cands[f"PCR joint/ctrl {sname}"] = (w, None)
                    cands["body-space (w, then w_u)"] = (body_space(Jm, Minv, c, rhs, False), None)
                    cands["body-space +1 ref. (w, then w_u)"] = (body_space(Jm, Minv, c, rhs, True), None)
                    cands["body-space native w_u = K⁻¹f"] = body_space_native(Jm, Minv, c, f)
                    for name, (w, wu) in cands.items():
                        w = w.double()
                        wu = dvec - Minv @ (Jm.T @ w) if wu is None else wu.double()
                        e_u = float(torch.linalg.norm(wu - wu_ex) / torch.linalg.norm(wu_ex))
                        e_w = float(torch.linalg.norm(w - w_ex) / torch.linalg.norm(w_ex))
                        cos = float(wu @ wu_ex / (torch.linalg.norm(wu) * torch.linalg.norm(wu_ex)))
                        errs.setdefault(name, []).append((e_u, e_w, cos))
            print(f"  loss gradient: {loss}")
            print(f"    {'solver':36s} rel.err w_u p50 / p90 / max   | rel.err w p50 / p90 | cos(w_u) min  | >10% err")
            for name, v in errs.items():
                v = np.array(v)
                eu, ew, cs = v[:, 0], v[:, 1], v[:, 2]
                print(f"    {name:36s} {pct(eu,50):7.1e} / {pct(eu,90):7.1e} / {eu.max():7.1e}"
                      f" | {pct(ew,50):7.1e} / {pct(ew,90):7.1e} | {cs.min():+.4f}"
                      f"  | {100 * (eu > 0.1).mean():4.1f}%")


if __name__ == "__main__":
    main()
