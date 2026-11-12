"""Does a body-space / constraint-space hybrid improve the one-step adjoint?
And does iterative refinement of the engine's adjoint do the same?

Companion of adjoint_accuracy.py (same systems, loss draws and reference).
The adjoint unknowns (w_u, w) satisfy the KKT system
    M w_u + Jᵀ w = f,      J w_u − C w = 0,
i.e. w_u = K⁻¹ f with K = M + Jᵀ C⁻¹ J. The engine eliminates w_u:
    d = M⁻¹ f,   A w = J d,   w_u = d − M⁻¹ Jᵀ w,
which is the Woodbury formula for K⁻¹ f; the subtraction amplifies the
relative error of w by
    amp = ‖M⁻¹ Jᵀ w‖ / ‖w_u‖.

Hybrid: split the active rows into S (eliminated in body space) and F
(kept in constraint space). For a general right-hand side (r1, r2):
    M̂ = M + J_Sᵀ C_S⁻¹ J_S                       (Cholesky)
    r̂ = r1 + J_Sᵀ C_S⁻¹ r2_S
    A_F w_F = J_F M̂⁻¹ r̂ − r2_F,   A_F = J_F M̂⁻¹ J_Fᵀ + C_F
    w_u = M̂⁻¹ (r̂ − J_Fᵀ w_F),   w_S = C_S⁻¹ (J_S w_u − r2_S)
S = ∅ is the engine's formula, S = all rows the body-space solve.
amp_F = ‖M̂⁻¹ J_Fᵀ w_F‖ / ‖w_u‖.

Iterative refinement (fixed precision, float32): from the current
(w_u, w) form the KKT residual
    r1 = f − M w_u − Jᵀ w,   r2 = −(J w_u − C w)
with multiplications only (no C⁻¹, so it also works for c = 0 rows), solve
the same KKT system for the correction with the same split and solver, and
add it. One step costs one more A_F solve.

Everything is float32 except the float64 reference w_u = K⁻¹ f. A_F is
solved with Jacobi PCR at the engine setting (tol 1e-5, cap 16), with the
same PCR given the iteration budget of one or two refinement steps (caps
32 and 48), or with Cholesky.

Splits (row types from the dump's offsets):
  engine (S = none)   all rows in constraint space
  joint/ctrl          joints and controls in body space
  all contacts        normals + friction in body space
  rho > 1e2           rows with ρ_i = (J_i M⁻¹ J_iᵀ) / c_i > 100
  all (body space)    everything in body space

Run:
    python -m test_scripts.deflation.adjoint_hybrid
"""

from __future__ import annotations

import numpy as np
import torch

from test_scripts.precond_lab import jacobi_apply, load_systems
from test_scripts.recycle_probe import pct
from test_scripts.deflation.deflation import DT, SCENES, deflated_pcr
from test_scripts.deflation.adjoint_accuracy import last_nr_systems, minv_dense

N_DRAWS = 8
# (label, A_F solver, refinement steps); a solver is (tol, cap) for PCR or "chol"
SOLVERS = (
    ("PCR cap16", (1e-5, 16), 0),
    ("PCR cap16 +1 IR", (1e-5, 16), 1),
    ("PCR cap16 +2 IR", (1e-5, 16), 2),
    ("PCR cap32", (1e-5, 32), 0),
    ("PCR cap48", (1e-5, 48), 0),
    ("Cholesky", "chol", 0),
    ("Cholesky +1 IR", "chol", 1),
)


class KKTSolver:
    """Float32 solver of  M u + Jᵀ w = r1,  J u − C w = r2  for one split."""

    def __init__(self, Jm, Minv, c, S, solver):
        Jm, Minv, c = Jm.float(), Minv.float(), c.float()
        n = Jm.shape[0]
        self.S, self.F = S, np.setdiff1d(np.arange(n), S)
        self.Js, self.Jf, self.cs, self.cf = Jm[self.S], Jm[self.F], c[self.S], c[self.F]
        self.Jm, self.M, self.c, self.n, self.solver = Jm, torch.linalg.inv(Minv), c, n, solver
        Mh = self.M + self.Js.T @ (self.Js / self.cs.unsqueeze(1))
        self.L = torch.linalg.cholesky(Mh)
        if len(self.F):
            self.Af = self.Jf @ self.mh_solve(self.Jf.T) + torch.diag(self.cf)
            if solver == "chol":
                self.Lf = torch.linalg.cholesky(self.Af)

    def mh_solve(self, X):
        """M̂⁻¹ X for a vector or a matrix X."""
        if X.dim() == 1:
            return torch.cholesky_solve(X.unsqueeze(1), self.L).squeeze(1)
        return torch.cholesky_solve(X, self.L)

    def solve(self, r1, r2):
        rh = r1 + self.Js.T @ (r2[self.S] / self.cs)
        it = 0
        if len(self.F):
            bf = self.Jf @ self.mh_solve(rh) - r2[self.F]
            if self.solver == "chol":
                wf = torch.cholesky_solve(bf.unsqueeze(1), self.Lf).squeeze(1)
            else:
                tol, cap = self.solver
                wf, it, _ = deflated_pcr(self.Af, bf, torch.zeros(len(self.F), 0), jacobi_apply(self.Af),
                                         max_iters=cap, tol=tol, atol=tol)
            u = self.mh_solve(rh - self.Jf.T @ wf)
        else:
            wf, u = torch.zeros(0), self.mh_solve(rh)
        w = torch.zeros(self.n)
        w[self.F] = wf
        w[self.S] = (self.Js @ u - r2[self.S]) / self.cs
        return u, w, it

    def adjoint(self, f, refine):
        f = f.float()
        u, w, it = self.solve(f, torch.zeros(self.n))
        for _ in range(refine):
            r1 = f - self.M @ u - self.Jm.T @ w
            r2 = -(self.Jm @ u - self.c * w)
            du, dw, it2 = self.solve(r1, r2)
            u, w, it = u + du, w + dw, it + it2
        return u, w, it


def amplification(Jm, Minv, c, f, S):
    """Exact (float64) amp_F = ‖M̂⁻¹ J_Fᵀ w_F‖ / ‖w_u‖ of a split."""
    n = Jm.shape[0]
    F = np.setdiff1d(np.arange(n), S)
    if not len(F):
        return 0.0
    Js, Jf = Jm[S], Jm[F]
    Mh = torch.linalg.inv(Minv) + Js.T @ (Js / c[S].unsqueeze(1))
    dh = torch.linalg.solve(Mh, f)
    wf = torch.linalg.solve(Jf @ torch.linalg.solve(Mh, Jf.T) + torch.diag(c[F]), Jf @ dh)
    back = torch.linalg.solve(Mh, Jf.T @ wf)
    return float(torch.linalg.norm(back) / torch.linalg.norm(dh - back))


def main():
    rng = np.random.default_rng(0)
    for label, path in SCENES.items():
        S_all = load_systems(path=path, dtype=DT)
        d = np.load(path, allow_pickle=True)
        meta = d["meta"][0]
        Nb, off_n = int(meta["N_b"]), int(meta["offset_n"])
        idx = last_nr_systems(d, len(S_all))
        res, amps, its, sizes = {}, {}, {}, {}
        for i in idx:
            A, Jm, n = S_all[i].A, S_all[i].Jm, S_all[i].n
            Minv = minv_dense(d, i, Nb)
            c = torch.diagonal(A - Jm @ Minv @ Jm.T)                        # C + reg as assembled
            act = np.nonzero(d["constr_active_mask"][i, 0] > 0.0)[0]
            rho = ((torch.diagonal(A) - c) / c).numpy()
            splits = {
                "engine (S = none)": np.array([], dtype=int),
                "S = joint/ctrl": np.array([r for r, g in enumerate(act) if g < off_n]),
                "S = all contacts": np.array([r for r, g in enumerate(act) if g >= off_n]),
                "S = rho > 1e2": np.nonzero(rho > 1e2)[0],
                "S = all (body space)": np.arange(n),
            }
            K = torch.linalg.inv(Minv) + Jm.T @ (Jm / c.unsqueeze(1))
            for _ in range(N_DRAWS):
                f = torch.from_numpy(rng.standard_normal(6 * Nb))
                wu_ex = torch.linalg.solve(K, f)
                for sname, Sidx in splits.items():
                    sizes.setdefault(sname, []).append(len(Sidx))
                    amps.setdefault(sname, []).append(amplification(Jm, Minv, c, f, Sidx))
                    for tag, spec, nref in SOLVERS:
                        if sname == "S = all (body space)" and spec != "chol":
                            continue
                        u, _, it = KKTSolver(Jm, Minv, c, Sidx, spec).adjoint(f, nref)
                        e = float(torch.linalg.norm(u.double() - wu_ex) / torch.linalg.norm(wu_ex))
                        res.setdefault((sname, tag), []).append(e)
                        its.setdefault((sname, tag), []).append(it)

        print(f"\n===== {label}: {len(idx)} systems x {N_DRAWS} random loss draws, float32 =====")
        print(f"  {'split':22s} {'|S|':>4s} {'amp p50 / p90':>16s} | {'A_F solver':16s} {'its':>4s} | "
              f"rel.err w_u p50 / p90 / max   | >10%  >1%")
        for sname in amps:
            a = np.array(amps[sname])
            first = True
            for tag, _, _ in SOLVERS:
                if (sname, tag) not in res:
                    continue
                e, it = np.array(res[(sname, tag)]), np.array(its[(sname, tag)])
                head = (f"  {sname:22s} {np.median(sizes[sname]):4.0f} {pct(a,50):7.1e} / {pct(a,90):6.1e} |"
                        if first else f"  {'':22s} {'':4s} {'':16s} |")
                print(f"{head} {tag:16s} {np.median(it):4.0f} | {pct(e,50):7.1e} / {pct(e,90):7.1e} / "
                      f"{e.max():7.1e} | {100 * (e > 0.1).mean():4.1f}% {100 * (e > 0.01).mean():4.1f}%")
                first = False


if __name__ == "__main__":
    main()
