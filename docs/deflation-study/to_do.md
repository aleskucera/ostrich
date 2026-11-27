# To do

Open issues found on 2026-11-27, when the in-engine figures (Figures 4, 5
and 7 of `latex-theory/`) were compared with the offline results and the
text.

## 1. The speed claim of §7.1 does not hold in the engine

§7.1 ("Evidence so far") says DPCR lowers the GPU time per solve at W = 1
from about 0.63 to 0.35 ms (obstacle) and from 0.84 to 0.51 ms (terrain),
taken from the offline Figure 3 (`fig:deflation-z-sources-time`). In the
engine (Figure 5, `fig:engine-time`), DPCR is slower per Newton iteration
than Jacobi PCR in 5 of 6 cases. Preconditioner setup + PCR solve, ms:

| scene | W | Jacobi PCR | DPCR | DPCR + 8 rows |
|---|---|---|---|---|
| terrain | 1 | 0.60 | 0.73 | 0.56 |
| chain | 1 | 0.61 | 0.91 | 1.06 |
| marv | 1 | 0.63 | 1.82 | 2.08 |
| terrain | 64 | 1.65 | 2.94 | 5.03 |
| chain | 64 | 1.85 | 4.48 | 6.86 |
| marv | 64 | 2.61 | 13.5 | 17.7 |

The setup is what is slow (0.2–15 ms; Jacobi about 0.01 ms). Offline,
Figure 3 treated the joint/control setup as nearly free. The solve itself
also gains less than the iteration counts suggest: on the chain, DPCR needs
1 iteration against 16, but its solve is only 2–3× faster.

To do:
- Profile the setup phase in `src/ostrich/optim/deflation.py` (row
  selection, A Z, Cholesky and E⁻¹ of A_SS, diag(A_F)) and find which part
  dominates and why it grows so steeply with W and with k (marv k = 120).
- Check whether the PCR solve has a fixed cost per solve that hides the
  saved iterations (e.g. it runs to the cap inside the CUDA graph instead
  of stopping early).
- Decide whether the setup can be done less often (once per time step
  instead of per Newton iteration, or reused as in
  `adjoint.reuse_deflation_factor`).
- Then rewrite the evidence and weaknesses of §7.1 from the engine numbers,
  not the offline estimate.

Data: `figure_scripts/engine_time.py` → `test_scripts/deflation/engine_dpcr.py`
(`forward_time`), GTX 1650 on AC power, commit d41c7f3. The run printed
four "Contact buffer overflowed" warnings (3340–3386 > 3328 contacts); the
contact buffer should be enlarged before the numbers are final.

## 2. On the chain, the engine's DPCR is a direct solve

The chain has no contacts, so "joint/control" covers every row (k = 72)
and the engine's DPCR solves the whole system directly. That is why it
reports 1 iteration in Figure 4 (PCR always runs at least one) and why its
gradient in Figure 7 matches the exact adjoint. Figure 6
(`fig:adjoint-chain-fd`) eliminates only the joint rows and iterates on the
control rows (the `joints_*` variant of `adjoint_articulated.py`), which is
the non-degenerate case. Figure 7's caption calls it the engine counterpart
of Figure 6, but on the chain the two compare different solvers.

To do:
- Add a joints-only option to the engine's deflation (e.g.
  `deflation="joint"`), so that the control rows stay in F, and rerun the
  chain in `engine_iterations.py`, `engine_time.py` and
  `engine_adjoint.py` with it.
- Until then, say in the text (and in the captions of Figures 4 and 7)
  that on the chain DPCR is a direct solve.

## 3. Offline and engine DPCR use different preconditioners

The offline DPCR of Figure 2 (`fig:deflation-z-sources`) and of §3.2 keeps
P = diag(A). The engine's DPCR uses P = diag(A_F)
(`LinearSolverConfig.deflation_jacobi = "schur"`, the default). So Figure 4
(`fig:engine-iterations`) is not a like-for-like counterpart of Figure 2.
The two preconditioners differ in the forward solve: §7.2 reports that
diag(A_F) keeps the obstacle median at 7 iterations but raises its 90th
percentile from 7 to 14. The shortened caption of Figure 4 no longer
mentions diag(A_F).

To do:
- Rerun the engine figures with `deflation_jacobi="full"` as well (or
  instead), so that the forward comparison with Figure 2 uses the same
  preconditioner.
- Decide which preconditioner the engine should default to: diag(A_F) is
  needed for the adjoint (§7.2), but may cost forward iterations.
- State the preconditioner in the captions of Figures 4, 5 and 7.
