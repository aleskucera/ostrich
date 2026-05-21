"""Measure how much the Schur-complement matrix A = J·M⁻¹·Jᵀ + C changes
between NR iterations and across simulation steps. Used to assess
whether Krylov subspace recycling is viable: recycling pays off when
A's slowly-changing eigenvectors persist across solves.

Two proxies:
  1. ||diag(A_k) - diag(A_{k-1})||₂ / ||diag(A_k)||₂   — cheap, only
     captures diagonal entries, but those drive the Jacobi preconditioner.
  2. ||A_k·v - A_{k-1}·v||₂ / ||A_k·v||₂  for a fixed random vector v
     — captures full A change including off-diagonals.

Across-NR-iter changes (within a single sim step) tell us about
within-step recycling potential. Across-step changes (between
successive steps' first NR iter) tell us about cross-step recycling.
"""
import os
import sys

import hydra
import numpy as np
import warp as wp
from omegaconf import DictConfig

CONFIG_PATH = "../examples/conf"


@hydra.main(
    config_path=CONFIG_PATH,
    config_name="helhest_obstacle_benchmark",
    version_base=None,
)
def main(cfg: DictConfig):
    from axion import (
        EngineConfig,
        LoggingConfig,
        RenderingConfig,
        SimulationConfig,
    )
    from axion.optim.preconditioner import JacobiPreconditioner

    sys.path.insert(
        0, os.path.join(os.path.dirname(__file__), "..", "examples", "helhest")
    )
    from obstacle_benchmark import HelhestObstacleBenchmark  # noqa: E402

    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    sim = HelhestObstacleBenchmark(
        sim_config, render_config, engine_config, logging_config,
        control_mode=cfg.control.mode, k_p=cfg.control.k_p, k_d=cfg.control.k_d,
        friction=cfg.friction_coeff, drive_velocity=cfg.drive_velocity,
    )

    # Records: list of dicts per NR iter — step_idx, iter_idx, diag, Av
    records = []

    # Fixed random probe vector (re-used across iters and steps).
    # Shape matches dconstr_force.full == (N_w, N_c).
    rng = np.random.default_rng(0)
    v_np = rng.standard_normal((1, sim.solver.dims.N_c)).astype(np.float32)
    v = wp.array(v_np, dtype=wp.float32, device=sim.solver.device)
    Av_buf = wp.zeros_like(v)

    orig_update = JacobiPreconditioner.update
    step_iter_state = {"step_idx": -1, "iter_idx": 0}

    def patched_update(self):
        orig_update(self)
        # Sync to read CPU values
        wp.synchronize()
        # Sample diagonal
        diag_inv = self._P_inv_diag.numpy()[0].copy()
        # Compute A·v via the engine's matvec operator
        sim.solver.A_op.matvec(v, Av_buf, Av_buf, alpha=1.0, beta=0.0)
        wp.synchronize()
        Av = Av_buf.numpy()[0].copy()
        records.append({
            "step": step_iter_state["step_idx"],
            "iter": step_iter_state["iter_idx"],
            "diag_inv": diag_inv,
            "Av": Av,
        })
        step_iter_state["iter_idx"] += 1

    JacobiPreconditioner.update = patched_update

    # Hook into engine.step to track step_idx
    orig_step = sim.solver.step

    def patched_step(state_in, state_out, control, contacts, dt):
        step_iter_state["step_idx"] += 1
        step_iter_state["iter_idx"] = 0
        orig_step(state_in, state_out, control, contacts, dt)

    sim.solver.step = patched_step

    # Run a short slice — eager mode (no graph) so the patches fire.
    # Force fewer steps; we don't need 200.
    sim.run()

    # ---- Analysis ----
    if len(records) < 2:
        print("Not enough records collected.")
        return

    R = records
    N = len(R)
    print(f"\nCollected {N} NR-iter records across "
          f"{R[-1]['step'] - R[0]['step'] + 1} sim steps.")

    # Compute diag from inv_diag (diag(A) = 1/diag_inv where non-zero)
    # We'll work with diag_inv directly since the relative change is the same
    # up to the inverse — but for clarity convert.
    def diag_of(rec):
        di = rec["diag_inv"]
        out = np.zeros_like(di)
        mask = np.abs(di) > 1e-30
        out[mask] = 1.0 / di[mask]
        return out

    # --- Within-step: how does A change between successive NR iters? ---
    within_diag = []
    within_Av = []
    across_diag = []
    across_Av = []

    for k in range(1, N):
        prev = R[k - 1]
        curr = R[k]
        d_prev = diag_of(prev)
        d_curr = diag_of(curr)
        # Mask zeros in either to avoid noise from inactive constraint slots
        m = (np.abs(d_prev) > 1e-30) & (np.abs(d_curr) > 1e-30)
        if m.sum() < 2:
            continue
        diag_rel = (
            np.linalg.norm((d_curr - d_prev)[m])
            / max(np.linalg.norm(d_curr[m]), 1e-30)
        )
        Av_rel = (
            np.linalg.norm(curr["Av"] - prev["Av"])
            / max(np.linalg.norm(curr["Av"]), 1e-30)
        )
        if curr["step"] == prev["step"]:
            within_diag.append(diag_rel)
            within_Av.append(Av_rel)
        else:
            across_diag.append(diag_rel)
            across_Av.append(Av_rel)

    def stats(name, arr):
        if not arr:
            print(f"  {name}: (no samples)")
            return
        a = np.array(arr)
        print(f"  {name}: n={len(a)} "
              f"p50={np.percentile(a,50):.3e} "
              f"p95={np.percentile(a,95):.3e} "
              f"max={a.max():.3e}")

    print("\n=== Within-step (NR iter k vs NR iter k-1) ===")
    stats("rel ||Δdiag||/||diag||", within_diag)
    stats("rel ||A·v - A_prev·v||/||A·v||", within_Av)

    print("\n=== Across-step (first iter of step n vs last iter of step n-1) ===")
    stats("rel ||Δdiag||/||diag||", across_diag)
    stats("rel ||A·v - A_prev·v||/||A·v||", across_Av)

    print("\n--- Interpretation ---")
    print("For Krylov recycling viability, lower is better.")
    print("Rough rule of thumb (ad hoc, not literature):")
    print("  rel_change < 1e-2  → recycling likely effective")
    print("  rel_change in [1e-2, 1e-1] → marginal, depends on spectrum")
    print("  rel_change > 1e-1  → recycling unlikely to help")


if __name__ == "__main__":
    main()
