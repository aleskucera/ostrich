"""DECISIVE gate for the async cross-step pipeline idea.

The pipeline: compute step N's deflation subspace from step N−1's data
(off the critical path), consume it for ALL of step N's NR solves, fall
back to Jacobi if not ready. It lives or dies on one unmeasured fact:

  does step N−1's bad subspace still cover step N's bad band, ACROSS the
  contact-set change at the step boundary?

We have within-step ≈1.0 (Finding 6) but never the clean cross-step
number. This probe (needs step_idx / iter_in_step from the re-dump):

  1. within-step consecutive overlap            (sanity vs Finding 6)
  2. CROSS-STEP: subspace(last iter of N−1) vs
       (a) first iter of N, (b) ALL iters of N  (pipeline reuses it all step)
  3. random-pair overlap                        (coherence control)
  4. OPERATIONAL: run PCR on every step-N system deflated with step
     N−1's last exact subspace → iters vs Jacobi (bar) and ideal.
     = "if the eigensolve were free/hidden, does the pipeline speed PCR?"

Run:
    python test_scripts/cross_step_stability.py
"""

from __future__ import annotations

import numpy as np
import torch

from test_scripts.precond_lab import DATA, MAX_ITERS, load_systems, pcr, jacobi_apply
from test_scripts.ideal_target_stability import bad_subspace, overlap
from test_scripts.recycle_probe import embed, two_level, restrict

DT = torch.float64
K = 16


def pct(a, p):
    return float(np.percentile(np.asarray(a, float), p)) if len(a) else float("nan")


def main():
    S = load_systems(dtype=DT)
    d = np.load(DATA, allow_pickle=True)
    if "step_idx" not in d.files:
        raise SystemExit("npz lacks step_idx — re-dump with the updated "
                         "dump_linear_systems.py first.")
    step_idx = d["step_idx"]
    iter_in_step = d["iter_in_step"]
    mask = d["constr_active_mask"]
    acts = [np.nonzero(mask[s, 0] > 0.0)[0] for s in range(len(S))]

    Q = [bad_subspace(S[i].A, min(K, S[i].n)) for i in range(len(S))]
    E = [embed(Q[i], acts[i]) for i in range(len(S))]

    # group system indices by step, ordered by iter_in_step
    steps = {}
    for i in range(len(S)):
        steps.setdefault(int(step_idx[i]), []).append(i)
    for k in steps:
        steps[k] = sorted(steps[k], key=lambda i: int(iter_in_step[i]))
    step_ids = sorted(steps)
    print(f"{len(S)} systems, {len(step_ids)} steps "
          f"(NR iters/step: p50 {pct([len(v) for v in steps.values()],50):.0f})\n")

    # 1. within-step consecutive
    wi = []
    for k in step_ids:
        g = steps[k]
        for a in range(len(g) - 1):
            wi.append(overlap(E[g[a]], E[g[a + 1]]))

    # 2. cross-step: last iter of N-1 vs first / all of N
    xs_first, xs_all, dA_xs = [], [], []
    for n in range(1, len(step_ids)):
        prev_last = steps[step_ids[n - 1]][-1]
        cur = steps[step_ids[n]]
        xs_first.append(overlap(E[prev_last], E[cur[0]]))
        for j in cur:
            xs_all.append(overlap(E[prev_last], E[j]))
        A0, A1 = S[prev_last].A, S[cur[0]].A
        if A0.shape == A1.shape:
            dA_xs.append(float(torch.linalg.norm(A1 - A0) /
                               (torch.linalg.norm(A0) + 1e-30)))

    # 3. random-pair control
    rng = np.random.default_rng(0)
    rnd = [overlap(E[i], E[j]) for i, j in
           rng.integers(0, len(S), (2000, 2))]

    print("subspace overlap (bottom-16, 402-embedded; 1=same, 0=orthogonal):")
    print(f"  within-step consecutive : p50 {pct(wi,50):.3f}  p05 {pct(wi,5):.3f}")
    print(f"  CROSS-STEP last(N-1)→first(N): p50 {pct(xs_first,50):.3f}  "
          f"p05 {pct(xs_first,5):.3f}")
    print(f"  CROSS-STEP last(N-1)→ALL of N : p50 {pct(xs_all,50):.3f}  "
          f"p05 {pct(xs_all,5):.3f}  (pipeline reuses it for the whole step)")
    print(f"  random-pair control     : p50 {pct(rnd,50):.3f}")
    print(f"  A change across boundary: p50 {pct(dA_xs,50):.2f}  "
          f"p95 {pct(dA_xs,95):.2f}\n")

    # 4. operational: pipeline simulation, evaluated on held-out late steps
    val_cut = step_ids[int(0.8 * len(step_ids))]
    it_jac, it_pipe, it_ideal = [], [], []
    for n in range(1, len(step_ids)):
        if step_ids[n] < val_cut:
            continue
        Zprev = E[steps[step_ids[n - 1]][-1]]          # step N-1 last subspace
        for i in steps[step_ids[n]]:
            A, b = S[i].A, S[i].b
            _, ij = pcr(A, b, jacobi_apply(A)); it_jac.append(ij)
            _, ii = pcr(A, b, two_level(A, Q[i])); it_ideal.append(ii)
            Zr = restrict(Zprev, acts[i])
            _, ip = pcr(A, b, two_level(A, Zr)) if Zr.shape[1] > 0 \
                else pcr(A, b, jacobi_apply(A))
            it_pipe.append(ip)

    def row(t, a):
        a = np.asarray(a, float)
        return (f"  {t:24s} mean {a.mean():5.1f}  p50 {pct(a,50):4.0f}  "
                f"p95 {pct(a,95):4.0f}")

    ij = np.array(it_jac, float)
    print(f"OPERATIONAL — held-out late steps (≥ step {val_cut}), pipeline "
          f"= step N−1's last exact subspace for all of step N:")
    print(row("jacobi (bar)", it_jac))
    print(row("cross-step pipeline", it_pipe))
    print(row("ideal (own subspace)", it_ideal))
    pm = np.array(it_pipe, float).mean()
    print(f"  → pipeline: {ij.mean():.1f}→{pm:.1f} "
          f"({100*(1-pm/ij.mean()):+.0f}%), beats Jacobi "
          f"{int((np.array(it_pipe,float)<ij).sum())}/{len(it_pipe)}")

    print("\nverdict:")
    xa = pct(xs_all, 50)
    if xa > 0.85 and pm < ij.mean() * 0.85:
        print(f"  cross-step subspace holds (overlap {xa:.2f} across the "
              f"contact-set change) AND reusing step N−1's subspace for all "
              f"of step N beats Jacobi ({ij.mean():.1f}→{pm:.1f}). The async "
              f"pipeline's core assumption is VALID — viability now reduces "
              f"to the in-engine question of hiding the eigensolve.")
    elif xa <= 0.6:
        print(f"  cross-step overlap p50 {xa:.2f} ≈ random ({pct(rnd,50):.2f}) "
              f"— the subspace does NOT survive the step boundary. A stale "
              f"step-N−1 subspace is useless for step N → async pipeline "
              f"dead regardless of how well the eigensolve is hidden.")
    else:
        print(f"  partial: cross-step overlap {xa:.2f}, pipeline "
              f"{ij.mean():.1f}→{pm:.1f}. Marginal — the boundary degrades "
              f"the subspace enough that the bounded payoff likely doesn't "
              f"justify the pipeline's engineering complexity.")


if __name__ == "__main__":
    main()
