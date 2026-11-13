"""Forward iteration counts, held-out 20%, float64, tol 1e-5, cap 64.
Jacobi PCR vs joint/ctrl deflation with P = diag(A) vs P = diag(Schur complement).

Run:
    python test_scripts/deflation/deflation_forward_precond.py
"""
import numpy as np

from test_scripts.precond_lab import load_systems, pcr
from test_scripts.deflation.adjoint_equivalence import deflated
from test_scripts.deflation.deflation import DT, SCENES

for label, path in SCENES.items():
    S = load_systems(path=path, dtype=DT)
    d = np.load(path, allow_pickle=True)
    off_n = int(d["meta"][0]["offset_n"])
    held = range(int(0.8 * len(S)), len(S))
    out = {"Jacobi": [], "defl jc, P=diag(A)": [], "defl jc, P=diag(Schur)": []}
    for i in held:
        A, b, n = S[i].A, S[i].b, S[i].n
        act = np.nonzero(d["constr_active_mask"][i, 0] > 0.0)[0]
        jc = np.array([r for r, g in enumerate(act) if g < off_n])
        out["Jacobi"].append(deflated(A, b, np.array([], dtype=int), pcr, 64, False)[1])
        out["defl jc, P=diag(A)"].append(deflated(A, b, jc, pcr, 64, False)[1])
        out["defl jc, P=diag(Schur)"].append(deflated(A, b, jc, pcr, 64, True)[1])
    print(label, {k: (float(np.median(v)), float(np.percentile(v, 90))) for k, v in out.items()})
