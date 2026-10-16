8. Immediate next steps

8.1 Where we are (summary of the discussion so far)

- NeurKItt, the paper, works in two parts. A network predicts the "bad" subspace: the eigenvectors with the smallest eigenvalues. GCRO-DR then deflates that subspace. The authors need a network because they solve independent, non-symmetric systems with nothing to recycle. They use the GCRO form, which sets C = orth(AY) and makes the residual orthogonal to AY. That is the natural choice for non-symmetric, indefinite systems, but it amplifies prediction errors by about λ_good/λ_bad. That makes it a poor fit for ostrich.
- Ostrich's A = J M⁻¹ Jᵀ + C is SPD. The textbook method for that case is Galerkin deflated (P)CG/PCR. Choose an n×k matrix Z, set E = ZᵀAZ (k×k SPD), Q = Z E⁻¹ Zᵀ and P = I − AQ. Solve PA x̃ = Pb with the existing Jacobi-preconditioned PCR, then recover x = Qb + Pᵀx̃. Interlacing guarantees that, for ANY Z, every eigenvalue of the deflated operator stays inside [λ_1, λ_n] of the Jacobi-preconditioned spectrum. The method cannot be worse-conditioned than Jacobi, and it cannot create new outliers.
- Today the ostrich solver (src/ostrich/optim/pcr_solver.py) does preconditioning only: Jacobi, or per-body-pair block Jacobi. It does no deflation.
- Every subspace experiment in test_scripts used a different method, the ADDITIVE two-level preconditioner M⁻¹ = D⁻¹ + Z(ZᵀAZ)⁻¹Zᵀ (`two_level` in recycle_probe.py, `two_level_apply` in coarse_probe.py). eigenspectrum_probe.py used the spectral correction I + Σ(1/λᵢ − 1)vᵢvᵢᵀ. docs/learned_preconditioner_feasibility.md calls these "deflation", but they are not projection deflation. With an exact Z, both forms give the same κ_eff = λ_n/λ_{k+1}, so the "ideal rank-k" ceiling numbers are valid. With an imperfect Z, the additive form adds +1 to the spectrum on span(A^{1/2}Z). That can push up to k eigenvalues ABOVE the cluster, creating new outliers, which is the "over-corrects" failure. True deflation cannot do this.
- Consequence: the doc closed several directions on additive-form results only. These were harvested-Ritz recycling (2× worse), accumulated GCRO-DR-style recycling (−1%) and a fixed universal basis. None has been rerun with true deflation. Also, no network is needed to start. Ostrich solves a long sequence of similar systems, so Z can come from earlier solves: an exact eigensolve on step N−1, or harvested Ritz vectors. Other sources are analytic, e.g. null(Jᵀ).

8.2 Suggested order

General rules for all steps:
- Work offline in test_scripts (PyTorch, float64). Do not modify src/ at all. Step 2 ends with a proposal for the engine change, not an implementation.
- Keep the harness settings from precond_lab.py (TOL = ATOL = 1e-5, MAX_ITERS = 64) so numbers compare directly with the doc.
- Write results to a new doc, docs/deflation-study/deflation_ab.md. Work happens in this repo on branch deflation-ab (the study started in the NeurKItt checkout, where ostrich is a submodule).

Step 0 — Prerequisite: regenerate the dataset. DONE 2026-10-12 (ostrich fbc3efe, newton 102e812).
- data/baselines/ is gitignored. Regenerate both scenes (from the repo root, with `uv run --no-sync` or `.venv/bin/python`; never plain `uv sync` / `uv run`, which reverts torch to cu128 and breaks it on the GPU):
    python test_scripts/dump_linear_systems.py --scene obstacle --out data/baselines/helhest_systems.npz
    python test_scripts/dump_linear_systems.py --scene surface  --out data/baselines/helhest_surface_systems.npz
  (`helhest_surface_systems.npz` is the script's default name and the name generalization_report.py expects.)
- This needs the CUDA device and the newton submodule environment (see README).
- The surface scene matters. The doc's Postscript 3 shows that conclusions from the obstacle scene alone did not generalize.
- Result: obstacle 335 systems, active n 31–67; surface 322 systems, active n 50–170. Both match the doc (it had 50–172 for surface).
- Re-measured baselines (the engine changed a lot after the doc's 2026-05-17 numbers, so ALWAYS compare against these, not the doc):
  - Obstacle, Jacobi PCR: mean 19.9 over all systems (doc 20.3); 1/335 doesn't converge within 64 iterations. Held-out late/post-impact Jacobi bar is now 18.4 in every script (the doc's scripts reported 18.2–18.8).
  - Surface, Jacobi PCR: held-out mean 28.1.
  - Reconstruction ‖A·x_engine − b‖/‖b‖ p50: obstacle 1.2e-4, surface 9.1e-3. On surface the engine's 16-iteration cap often stops before convergence, so x_engine is NOT a valid reference there; use a dense solve.
  - Every other doc number reproduced within a few percent, EXCEPT the cross-step pipeline on obstacle: now 18.4 → 14.4 (+22%, p95 25 vs Jacobi 22), doc 18.5 → 11.9 (+36%). The subspace shift at the step boundary is unchanged (overlap last(N−1)→first(N) p50 0.44), so the additive form now over-corrects more. That makes this row a good A/B candidate.

Step 1 — Offline, iteration counts only: implement Galerkin-deflated PCR and A/B it against the additive form on the same Z.

1a. DONE 2026-10-12 (ostrich cd6852a). Implement `deflated_pcr(A, b, Z, M_apply)` in a new file, test_scripts/deflation.py. The simplest correct version reuses the existing `pcr` from precond_lab.py:
- Orthonormalize Z (QR) and drop near-dependent columns using rank-revealing QR or the SVD. Harvested Ritz bases can be nearly dependent.
- Compute AZ, E = Zᵀ(AZ), and the Cholesky factor of E.
- Build the dense PA = A − AZ E⁻¹ (AZ)ᵀ and the deflated right-hand side Pb = b − AZ E⁻¹ Zᵀb. The systems are small, so dense is fine offline.
- Run `pcr(PA, Pb, M_apply)`. M_apply must be the Jacobi of the ORIGINAL A, i.e. jacobi_apply(A), not of PA.
- Recover x = Q b + x̃ − Q(A x̃), using Pᵀ = I − QA.
- Stopping criterion: the true residual b − Ax equals the deflated residual Pb − PAx̃ exactly. So the inner residual is the right quantity, but the relative tolerance must be relative to ‖b‖, not ‖Pb‖. Scale tol by ‖b‖/‖Pb‖, or pass an equivalent target.
- Follow the existing scripts' handling of the active set and padding (`restrict` and `embed` in recycle_probe.py).
- Iteration-count convention: report PCR iterations, as the doc does. Record the setup cost separately (k products with A to form AZ), because step 2 needs it.

1b. DONE 2026-10-12 (ostrich cd6852a; results in docs/deflation-study/deflation_ab.md). Sanity checks. Run these before trusting any comparison:
- Z empty → identical to the Jacobi baseline.
- Z = bad_subspace(A, k) from ideal_target_stability.py → about the eigenspectrum_probe ceiling (k = 8 → ~15 iterations, k = 16 → ~8; re-measured means 15.4 / 8.3). That probe uses the symmetrically scaled system (k=0 mean 23.2, vs 19.9 for left-Jacobi PCR), so expect the right ballpark, not exact equality. A closer reference: additive `two_level` with the system's own exact bottom-16 gives held-out mean 11.4.
- Z random → expected median iterations ≤ Jacobi, up to noise. RESULT: obstacle is clearly worse (27.5 vs 19.9; additive 32.8), surface is equal (39.4 vs 39.5). This is NOT a bug. The theory bounds κ_eff, not the iteration count: projecting out random directions moves RHS energy into the soft modes (obstacle ~7×, surface unchanged). Interlacing holds, and rounding, cluster splitting and a larger initial residual were ruled out. So "theory forbids clearly worse" was too strong. Deflation is still better than additive with the same bad Z.
- x matches a dense solve to within tolerance. Also check it against the engine solution stored in the npz, but only on obstacle: surface x_engine is often unconverged (see step 0).

1c. DONE 2026-10-13 (test_scripts/deflation_ab.py; full tables in docs/deflation-study/deflation_ab.md). The A/B table. For each Z source, run additive (`two_level`) and deflation with the SAME Z, on both scenes. Report median and p90 iterations, and wins vs Jacobi (count).
- Exact current subspace, bad_subspace on the system itself, k ∈ {4, 8, 16, 32}. This is the ceiling.
(Additive numbers below were re-measured 2026-10-12 on the obstacle held-out split, Jacobi bar 18.4.)
- Exact subspace from step N−1, reused for all of step N (logic of cross_step_stability.py). Additive gives 18.4 → 14.4 (+22%; doc had 18.5 → 11.9). For reference: the exact bottom-16 of the previous SOLVE (recycle_validate.py "recycle-always") gives 18.4 → 11.9.
- Harmonic Ritz harvested from the previous solve, m ∈ {16, 24, 32} (`approx_subspace` and `harmonic_ritz` in recycle_approx.py). Additive gives 37.9 / 24.9 / 15.4 (m = 16 is 2× worse than Jacobi). This is the most important row.
- Accumulated recycling across the sequence (logic of recycle_gcrodr.py). Additive gives 18.6 (−1%).
- Analytic null(Jᵀ) (`null_jt_basis` in coarse_null_probe.py). Additive gives 15.0 (+18%). Note that dim null(Jᵀ) is p50 31 (10–43) with n 31–67, so Z is large; that matters for step 2. The completion Z = [null(Jᵀ) ‖ joint/control block] is NOT untested, as the doc says: coarse_enriched_probe.py tested it additively, and it gave 16.4 (+11%), worse than null-only. Joint-only gave 14.7. Run all three with deflation.

1d. DONE 2026-10-13 (test_scripts/deflation_robustness.py, docs/deflation-study/deflation_robustness.png; details in docs/deflation-study/deflation_ab.md). Robustness curve. Take Z = exact current subspace, k = 16, and perturb it with controlled noise to sweep the principal angle to the true subspace from 0° to about 60°. Plot iterations vs angle for additive vs deflation. This shows how accurate any Z source (or a future network) must be.

Decision gate for Step 1 → Step 2:
- A CHEAP Z source qualifies if, with deflation, it cuts median iterations by ≥ ~20% vs Jacobi on BOTH scenes and does not lose to Jacobi in p90. Cheap means no per-step eigensolve: harvested Ritz, accumulated recycling, or analytic.
- The exact-from-N−1 source also qualifies, but carries a per-step eigensolve cost that step 2 must include.
- If nothing qualifies, report that and skip to step 3.
- RESULT (step 1c, held-out last 20% of systems; Jacobi median 19 obstacle / 29 surface; deflation median, change vs Jacobi):
  - joint/control (fixed 18 rows, Z costs nothing): 7 / 12 (−63% / −59%), 0 losses. QUALIFIES, the cheapest. Deflating these rows = exact solve of the stiff joint block + PCR on the Schur complement. Additive with the same Z gave only 15 / 29.
  - GCRO-DR accumulated (m = 16): 14 / 11 (−26% / −62%). QUALIFIES.
  - harmonic m = 32: 12 / 11. Qualifies on numbers, but m = 32 exceeds the engine's 16-iteration budget. m = 24 misses on obstacle (−16%); m = 16 fails (obstacle 20, 50/67 losses).
  - null(Jᵀ): 12 / 19. Qualifies, but k/n is 0.59 / 0.82, so the coarse solve is costly. enriched gives 4 / 6, but k/n is 0.90 / 0.95: effectively a direct solve, not a fair iteration count.
  - cross-step N−1 (exact, needs a per-step eigensolve): 11 / 20, 0 losses on surface (additive had 28).
  - Across all sources, deflation ≈ additive with exact Z and is much better with approximate Z. The old negative verdicts on recycling and the cross-step pipeline were mostly verdicts on the additive form.
  - Step-2 candidates: joint/control, GCRO-DR; cross-step as the expensive reference.
- RESULT (step 1d). Angles must be measured in the Jacobi-scaled space. Deflation ≤ additive at every angle.
  - Surface (gap λ1/λ17 ~0.08) gives a monotone curve: a uniform error ≲ 20° passes the gate, ≲ 5° stays near the ceiling.
  - Obstacle (gap ~7e-6) is non-monotone. A uniform random error of 1° already loses to Jacobi, while 90° beats 10–60°. Partially capturing the ultra-soft, RHS-free modes is the worst case: the leftover eigenvalue is ≈ λ/sin²θ, and ‖Pb‖ grows. The working sources (GCRO-DR, harmonic m=32) ignore modes 1–4 and capture modes ~12–16.
  - So a single principal angle is NOT a valid accuracy spec, and for step 3 a uniform projection loss is the wrong target (see the doc).

Step 2 — Estimate wall time before touching the engine. DONE 2026-10-13 (test_scripts/deflation_cost.py; details in docs/deflation-study/deflation_ab.md).
- RESULT: no Z source gives a clearly positive net gain on both scenes, so no engine change is proposed.
  - Engine facts: the dump config caps PCR at 16, and Jacobi hits the cap on ~90% of held-out solves (residual p90 8e-4 / 2e-2). All worlds iterate until the slowest converges, so the batch cost follows the MAX iteration count.
  - Cross-step breaks in float32 (surface) and was dropped. Block-Jacobi on the joint block is no substitute for joint/control deflation.
  - joint/control, single world (optimized kernels, measured): obstacle −29% (default config) / −31% (tol 1e-3), surface −23% (tol 1e-3), surface +6% at the default config. Deflation costs +80 µs setup per solve and +24% per iteration (one extra launch). Batched it loses (+24% to +147%): setup and AZ traffic scale with W·k, and surface's batch maximum stays at the cap.
  - GCRO-DR / any dense Z (incl. a network): at the default config even zero overhead gives 0% in batch. At tol 1e-3 the ceiling is −25% to −41%, eaten by the dense projection and the Z update.
  - Open lever: accuracy. Joint/control deflation converges (6e-6 / 2e-4) where Jacobi stops unconverged at the cap. Whether that cuts Newton iterations needs an in-engine run (a src/ change; not done).

Step 2c — Direct-solve baseline. DONE 2026-10-13 (test_scripts/deflation_direct.py; tables in docs/deflation-study/deflation_ab.md, "Direct-solve baseline").
- Question: since the systems are small (n ≤ 170), is iterating worth it at all?
- Body-space solve: A = G M⁻¹ Gᵀ + C, where G has only 6·N_b = 24 columns (Helhest has 4 bodies), and every row has c > 0. So by Woodbury, (M + Gᵀ C⁻¹ G) v = Gᵀ C⁻¹ b and x = C⁻¹(b − G v). That is a 24×24 Cholesky per world, independent of the number of active contacts. The ultra-soft modes (null(Gᵀ), held only by c) are handled exactly.
- RESULT, float32 GPU, assembly included, ms per solve at W = 1 / 64 / 1024:

  | | obstacle | surface |
  |---|---|---|
  | body-space | 0.115 / 0.79 / 11.0 | 0.108 / 0.79 / 11.1 |
  | dense compact | 0.16 / 0.72 / 9.1 (n 67) | 0.28 / 2.4 / 33.6 (n 170) |
  | dense padded 402 | 0.66 / 13.7 / 207 | 0.64 / 13.8 / 207 |
  | best PCR (any variant, either tol setting) | 0.45 / 1.87 / 18.2 | 0.61 / 2.11 / 21.8 |

  - Body-space is 3.9–5.6× faster than the best PCR variant (Jacobi, joint/control, joint/control + 8 strength rows) at W = 1, 2.4–2.7× at W = 64 and 1.7–2.0× at W = 1024.
  - Accuracy, p90 relative residual: body-space 3e-4 / 1e-5, or 2e-5 / 3e-7 with one refinement step (not timed); dense ~1e-7; Jacobi PCR at the cap 8e-4 / 2e-2.
- Consequence: for these scenes the direct solve supersedes deflation, and with it the network-predicted-Z idea (step 3). Deflation gains ≤ ~30% at W = 1 and loses in batch; the direct solve is several times faster everywhere.
- PCR should still win with many bodies (6·N_b in the hundreds+), deformables, or c → 0. The crossover is not measured. The natural engine design is a size-based dispatch.
- Context: ostrich's PCR comes from Macklin et al. 2019 (paper ref [11]), which targets deformables with thousands of DOFs. MuJoCo's Newton solver does the body-space analogue (Cholesky of M + JᵀDJ).
- Next, if pursued (src/ change, user's decision):
  - a Warp body-space solve in the NR loop in place of PCR;
  - measure time per step and the Newton iteration count;
  - reuse the same factor for the adjoint (eq. 18; A is symmetric).

Step 2d — Adjoint accuracy, offline. DONE 2026-10-13 (test_scripts/adjoint_accuracy.py; tables in docs/deflation-study/deflation_ab.md, "Adjoint accuracy").
- Engine backward step (base_engine.step_backward): d = M⁻¹f, then A w = J d by PCR (forward tol/cap, x0 = 0), then w_u = d − M⁻¹Jᵀw. With the defaults, A is the forward operator.
- Offline stand-in: the last NR system per step, 25 steps per scene × 8 loss gradients (random on all bodies / chassis only), compared against a float64 exact solve.
- RESULT, relative error of w_u (the gradient carried to the previous step), float32:
  - PCR Jacobi at yaml config, obstacle: p50 4e-2, p90 0.46, 39% of draws > 10% error, worst cosine 0.68.
  - PCR Jacobi, surface: 3–6% of draws > 10% at yaml, 21% at 1e-5/cap16.
  - Joint/control deflation helps only partly (obstacle yaml: 42% random / 9% chassis > 10%).
  - Body-space native, w_u = (M + Jᵀ C⁻¹ J)⁻¹ f and w = C⁻¹ J w_u, using the same 24×24 factor as the forward solve: ≤ 0.5% error on every draw, cosine 1.0000.
- Cause: cancellation. w_u = d − M⁻¹Jᵀw subtracts two nearly equal vectors when bodies are strongly constrained, so relative error in w is amplified. Even an accurate w (3e-3) gives 14% p50 error in w_u through this formula on obstacle. The body-space form has no subtraction.
- Consequence: the body-space formulation gives a 2–6× faster forward solve AND an essentially exact, almost free adjoint (two triangular solves with the forward factor). This is the paper hook: "exact implicit gradients at lower cost".
- Open: these are single-step errors (they compound over a horizon), and the offline operator is the last NR system, not the exact converged state. The decisive test is in-engine (src/ change, user's decision): an FD gradient check and a trajectory-optimization run, PCR adjoint vs body-space adjoint.

Step 3 — Only if no cheap Z is good enough: consider a network.
- A network is justified only if step 1 shows that an accurate Z helps (the ceiling and robustness curve) but no cheap source reaches the required accuracy.
- Target: the bad subspace of the Jacobi-scaled A (bad_subspace).
- Input: the engine state or system data available before the solve.
- Loss: NeurKItt's basis-invariant projection loss, Σᵢ‖sᵢ − QQᵀsᵢ‖ with QR on the outputs. Never MSE on the vectors.
- Success metric: iterations under DEFLATION, plus the step-2 wall-time accounting, not subspace overlap. The doc notes that the padded overlap metric is saturated and misleading.
- The step 1d curve gives the accuracy the network must hit.
