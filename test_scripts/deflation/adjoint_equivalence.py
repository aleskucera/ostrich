"""Is the hybrid adjoint (S = joint/ctrl in body space) the same thing as
joint/ctrl-deflated PCR + the engine's formula?

Claim tested: for any approximate w with residual r = J d - A w, the gradient
error is  dw_u = K^-1 J^T C^-1 r  (push-through identity). Deflation with unit
vectors on S gives r_S = 0; the hybrid gives r_S = 0 too. Both iterate on the
same Schur complement with the same right-hand side, so they can differ only by
the preconditioner (diag(A)_F vs diag(A_F)) and by rounding.

Also: CG instead of CR for the adjoint (CG minimises ||dw||_A >= ||dw_u||_M).

Run:
    python test_scripts/deflation/adjoint_equivalence.py
"""
import sys
import numpy as np
import torch

from test_scripts.precond_lab import jacobi_apply, load_systems, pcr
from test_scripts.recycle_probe import pct
from test_scripts.deflation.deflation import DT, SCENES
from test_scripts.deflation.adjoint_accuracy import last_nr_systems, minv_dense
from test_scripts.deflation.adjoint_hybrid import KKTSolver

N_DRAWS = 8
TOL = 1e-5


def pcg(A, b, M_apply, max_iters, tol, atol):
    x = torch.zeros_like(b)
    r = b.clone()
    target = max(atol * atol, float(tol * tol * torch.dot(b, b)))
    if float(torch.dot(r, r)) <= target:
        return x, 0
    z = M_apply(r)
    p = z.clone()
    rz = torch.dot(r, z)
    it = 0
    for it in range(1, max_iters + 1):
        Ap = A @ p
        pAp = torch.dot(p, Ap)
        if float(pAp) <= 0.0:
            break
        a = rz / pAp
        x = x + a * p
        r = r - a * Ap
        if float(torch.dot(r, r)) <= target:
            break
        z = M_apply(r)
        rz_new = torch.dot(r, z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, it


def deflated(A, b, S, krylov, cap, schur_diag):
    """Unit-vector deflation on rows S, float32. krylov in {pcr, pcg}.
    schur_diag: precondition with diag of the Schur complement on F rows."""
    n = A.shape[0]
    if len(S) == 0:
        M = jacobi_apply(A)
        x, it = krylov(A, b, M, max_iters=cap, tol=TOL, atol=TOL)
        return x, it
    Z = torch.eye(n, dtype=A.dtype)[:, S]
    AZ = A @ Z
    L = torch.linalg.cholesky(Z.T @ AZ)
    Einv = lambda Y: torch.cholesky_solve(Y, L)
    PA = A - AZ @ Einv(AZ.T)
    Pb = b - AZ @ Einv((Z.T @ b).unsqueeze(1)).squeeze(1)
    if schur_diag:
        dg = torch.diagonal(PA).clone()
        dg[S] = torch.diagonal(A)[S]
        M = lambda r, inv=1.0 / dg: inv * r
    else:
        M = jacobi_apply(A)
    bn, pbn = torch.linalg.norm(b), torch.linalg.norm(Pb)
    tol_pb = float(TOL * bn / pbn) if float(pbn) > 0 else TOL
    xt, it = krylov(PA, Pb, M, max_iters=cap, tol=tol_pb, atol=TOL)
    x = xt + Z @ Einv((Z.T @ b - AZ.T @ xt).unsqueeze(1)).squeeze(1)
    return x, it


def main():
    rng = np.random.default_rng(0)
    for label, path in SCENES.items():
        S_all = load_systems(path=path, dtype=DT)
        d = np.load(path, allow_pickle=True)
        meta = d["meta"][0]
        Nb, off_n = int(meta["N_b"]), int(meta["offset_n"])
        idx = last_nr_systems(d, len(S_all))
        res, its, ident = {}, {}, []
        for i in idx:
            A, Jm, n = S_all[i].A, S_all[i].Jm, S_all[i].n
            Minv = minv_dense(d, i, Nb)
            Mm = torch.linalg.inv(Minv)
            c = torch.diagonal(A - Jm @ Minv @ Jm.T)
            act = np.nonzero(d["constr_active_mask"][i, 0] > 0.0)[0]
            jc = np.array([r for r, g in enumerate(act) if g < off_n])
            none = np.array([], dtype=int)
            K = Mm + Jm.T @ (Jm / c.unsqueeze(1))
            for _ in range(N_DRAWS):
                f = torch.from_numpy(rng.standard_normal(6 * Nb))
                wu_ex = torch.linalg.solve(K, f)
                dvec = Minv @ f
                rhs = Jm @ dvec
                Af, bf = A.float(), rhs.float()

                def engine_formula(w):
                    w = w.double()
                    return dvec - Minv @ (Jm.T @ w), w

                cands = {}
                for cap in (16, 32):
                    for kname, kry in (("PCR", pcr), ("PCG", pcg)):
                        cands[f"{kname} Jacobi, engine formula, cap{cap}"] = deflated(Af, bf, none, kry, cap, False)
                        cands[f"{kname} defl jc, P=diag(A), engine formula, cap{cap}"] = deflated(Af, bf, jc, kry, cap, False)
                        cands[f"{kname} defl jc, P=diag(Schur), engine formula, cap{cap}"] = deflated(Af, bf, jc, kry, cap, True)
                for name, (w, it) in cands.items():
                    u, w64 = engine_formula(w)
                    e = float(torch.linalg.norm(u - wu_ex) / torch.linalg.norm(wu_ex))
                    eM = float(torch.sqrt((u - wu_ex) @ (Mm @ (u - wu_ex)) / (wu_ex @ (Mm @ wu_ex))))
                    res.setdefault(name, []).append((e, eM))
                    its.setdefault(name, []).append(it)
                    if name == "PCR Jacobi, engine formula, cap16":
                        r = rhs - A @ w64
                        pred = torch.linalg.solve(K, Jm.T @ (r / c))
                        ident.append(float(torch.linalg.norm(pred - (u - wu_ex)) / torch.linalg.norm(u - wu_ex)))
                for cap in (16, 32):
                    u, _, it = KKTSolver(Jm, Minv, c, jc, (TOL, cap)).adjoint(f, 0)
                    u = u.double()
                    name = f"hybrid S=jc (body space), PCR cap{cap}"
                    e = float(torch.linalg.norm(u - wu_ex) / torch.linalg.norm(wu_ex))
                    eM = float(torch.sqrt((u - wu_ex) @ (Mm @ (u - wu_ex)) / (wu_ex @ (Mm @ wu_ex))))
                    res.setdefault(name, []).append((e, eM))
                    its.setdefault(name, []).append(it)

        print(f"\n===== {label}: {len(idx)} systems x {N_DRAWS} draws, float32 =====")
        print(f"  identity dw_u = K^-1 J^T C^-1 r : rel. mismatch p50 {pct(ident,50):.1e}  max {max(ident):.1e}")
        print(f"  {'variant':52s} its | rel.err w_u p50 / p90 | M-norm p50 | >10%   >1%")
        for name in sorted(res, key=lambda s: (s[-2:], s)):
            v = np.array(res[name])
            e, eM = v[:, 0], v[:, 1]
            print(f"  {name:52s} {np.median(its[name]):3.0f} | {pct(e,50):7.1e} / {pct(e,90):7.1e} | {pct(eM,50):7.1e}"
                  f"   | {100*(e>0.1).mean():4.1f}% {100*(e>0.01).mean():5.1f}%")


if __name__ == "__main__":
    sys.exit(main())
