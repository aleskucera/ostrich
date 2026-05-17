"""Cross-scene generalization check: do the flat-ground (obstacle)
findings hold on the Helhest *surface* (mesh-terrain) scene?

Runs the load-bearing measurements on any dumped systems npz so two
scenes can be compared side by side:
  * reconstruction fidelity (convention sanity for this scene)
  * κ: raw vs sym-Jacobi-scaled (Finding 2 — mostly removable units?)
  * iters-to-tol: jacobi / per-body-pair / null(Jᵀ) / exact-bottom16 /
    cross-step pipeline   (Findings 1,3,13,14)
  * A|null(Jᵀ) = C mechanism check (Finding 13)

Run:
    python test_scripts/generalization_report.py \
        --data data/baselines/helhest_systems.npz --label flat
    python test_scripts/generalization_report.py \
        --data data/baselines/helhest_surface_systems.npz --label surface
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from test_scripts.precond_lab import load_systems, pcr, jacobi_apply, per_body_pair_apply
from test_scripts.equilibration_diag import sym_jacobi_scale
from test_scripts.ideal_target_stability import bad_subspace, overlap
from test_scripts.recycle_probe import two_level, embed, restrict
from test_scripts.coarse_null_probe import null_jt_basis

DT = torch.float64
K = 16


def pct(a, p):
    return float(np.percentile(np.asarray(a, float), p)) if len(a) else float("nan")


def report(path, label):
    S = load_systems(path=path, dtype=DT)
    d = np.load(path, allow_pickle=True)
    mask = d["constr_active_mask"]
    acts = [np.nonzero(mask[s, 0] > 0.0)[0] for s in range(len(S))]
    step_idx = d["step_idx"] if "step_idx" in d.files else np.zeros(len(S), int)
    iter_in_step = d["iter_in_step"] if "iter_in_step" in d.files else np.arange(len(S))

    recon, kraw, kjac, acres = [], [], [], []
    Q, E = [], []
    for i, s in enumerate(S):
        recon.append(float(torch.linalg.norm(s.A @ s.x_engine - s.b) /
                           (torch.linalg.norm(s.b) + 1e-30)))
        As, _ = sym_jacobi_scale(s.A)
        kraw.append(float(torch.linalg.cond(s.A)))
        kjac.append(float(torch.linalg.cond(As)))
        Qb = bad_subspace(s.A, min(K, s.n)); Q.append(Qb); E.append(embed(Qb, acts[i]))
        Zn, _ = null_jt_basis(s.Jm)
        if Zn.shape[1]:
            C = torch.diag(s.c_active); G = Zn.T @ s.A @ Zn
            acres.append(float(torch.linalg.norm(G - Zn.T @ C @ Zn) /
                               (torch.linalg.norm(G) + 1e-30)))

    # held-out late steps for the operational comparisons
    steps = {}
    for i in range(len(S)):
        steps.setdefault(int(step_idx[i]), []).append(i)
    for k in steps:
        steps[k] = sorted(steps[k], key=lambda i: int(iter_in_step[i]))
    sids = sorted(steps)
    vcut = sids[int(0.8 * len(sids))] if len(sids) > 1 else sids[0]

    ij, ipbp, inull, iideal, ipipe = [], [], [], [], []
    for n_i, sid in enumerate(sids):
        if sid < vcut:
            continue
        Zprev = (E[steps[sids[n_i - 1]][-1]]
                 if n_i > 0 and sids[n_i - 1] >= 0 else None)
        for i in steps[sid]:
            A, b = S[i].A, S[i].b
            ij.append(pcr(A, b, jacobi_apply(A))[1])
            ipbp.append(pcr(A, b, per_body_pair_apply(A, S[i].pair_id))[1])
            iideal.append(pcr(A, b, two_level(A, Q[i]))[1])
            Zn, _ = null_jt_basis(S[i].Jm)
            inull.append(pcr(A, b, two_level(A, Zn))[1] if Zn.shape[1]
                         else pcr(A, b, jacobi_apply(A))[1])
            if Zprev is not None:
                Zr = restrict(Zprev, acts[i])
                ipipe.append(pcr(A, b, two_level(A, Zr))[1] if Zr.shape[1]
                             else pcr(A, b, jacobi_apply(A))[1])

    print(f"\n===== {label}  ({len(S)} systems, {len(sids)} steps, "
          f"active n {min(s.n for s in S)}–{max(s.n for s in S)}) =====")
    print(f"  reconstruction ‖Ax_eng−b‖/‖b‖ : p50 {pct(recon,50):.1e} "
          f"(faithful if ≪1)")
    print(f"  κ(A) raw p50 {pct(kraw,50):.1e}  → sym-Jacobi p50 "
          f"{pct(kjac,50):.1e}  ({np.log10(pct(kraw,50)/pct(kjac,50)):.1f} "
          f"orders removed = units)")
    print(f"  A|null(Jᵀ)=C check p50 {pct(acres,50):.1e} (≈0 confirms)")
    jm = np.mean(ij)
    def line(t, a):
        a = np.asarray(a, float)
        return (f"  {t:20s} mean {a.mean():5.1f}  "
                f"({100*(1-a.mean()/jm):+.0f}% vs jacobi)")
    print("  VAL iters-to-tol (held-out late steps):")
    print(line("jacobi (bar)", ij))
    print(line("per-body-pair", ipbp))
    print(line("null(Jᵀ)", inull))
    print(line("cross-step pipeline", ipipe) if ipipe else
          "  cross-step pipeline   (n/a — single step)")
    print(line("ideal (exact b16)", iideal))
    return dict(jac=np.mean(ij), pbp=np.mean(ipbp), null=np.mean(inull),
                pipe=np.mean(ipipe) if ipipe else float("nan"),
                ideal=np.mean(iideal), kraw=pct(kraw, 50), kjac=pct(kjac, 50))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--label", default="scene")
    report(ap.parse_args().data, ap.parse_args().label)


if __name__ == "__main__":
    main()
