"""Phase 3 smoke test for PerBodyPairPreconditioner.

Validates the per-pair Cholesky factorization by reconstructing
A_pair from the GPU-computed L and comparing against the original
A_pair. For each active pair:
  * No factor_failure flagged (block was SPD)
  * L is lower-triangular (upper triangle is zero)
  * L · Lᵀ ≈ A_pair to float32 precision
  * Diagonal of L is positive

Captures at the same mid-run step as phase 2 for direct comparison.
"""
import os
import sys

import hydra
import numpy as np
from omegaconf import DictConfig


CONFIG_PATH = "../examples/conf"


@hydra.main(
    config_path=CONFIG_PATH,
    config_name="helhest_obstacle_benchmark",
    version_base=None,
)
def main(cfg: DictConfig):
    import warp as wp
    from axion import (
        EngineConfig,
        LoggingConfig,
        RenderingConfig,
        SimulationConfig,
    )
    from axion.optim.per_body_pair_preconditioner import (
        PerBodyPairPreconditioner,
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

    engine = sim.solver
    precond = PerBodyPairPreconditioner(
        engine, regularization=engine.config.linear.regularization
    )

    state = {"step": -1, "iter": 0, "captured": False, "result": None}
    target_step = 30  # mid-run rolling regime

    orig_update = JacobiPreconditioner.update

    def patched_update(self):
        orig_update(self)
        if state["captured"]:
            return
        if state["step"] != target_step or state["iter"] != 0:
            state["iter"] += 1
            return

        precond.update_pair_assignments()
        precond.update_blocks()
        precond.factor_blocks()
        wp.synchronize()

        state["result"] = {
            "A_blocks": precond.A_blocks.numpy(),
            "L_blocks": precond.L_blocks.numpy(),
            "member_count": precond.pair_member_count.numpy(),
            "factor_failure": precond.factor_failure.numpy(),
        }
        state["captured"] = True
        state["iter"] += 1

    JacobiPreconditioner.update = patched_update

    orig_step = engine.step

    def patched_step(state_in, state_out, control, contacts, dt):
        state["step"] += 1
        state["iter"] = 0
        orig_step(state_in, state_out, control, contacts, dt)

    engine.step = patched_step

    sim.run()

    if not state["captured"]:
        print(f"ERROR: never captured at step {target_step}.")
        return

    R = state["result"]
    w = 0
    A_blocks = R["A_blocks"][w]
    L_blocks = R["L_blocks"][w]
    member_count = R["member_count"][w]
    factor_failure = R["factor_failure"][w]

    active_pairs = np.where(member_count > 0)[0]

    print(f"Validating Cholesky on {len(active_pairs)} active pairs...")

    n_failed = int(factor_failure.sum())
    print(f"  factor_failure flags raised : {n_failed}")
    if n_failed:
        print("  (FAILED — block was non-SPD)")

    max_recon_err = 0.0
    max_upper_err = 0.0
    min_diag = np.inf
    for pid in active_pairs:
        n = int(member_count[pid])
        L = L_blocks[pid, :n, :n].astype(np.float64)
        A = A_blocks[pid, :n, :n].astype(np.float64)

        # Reconstruction: L · Lᵀ should equal A
        recon = L @ L.T
        err = np.abs(recon - A).max()
        max_recon_err = max(max_recon_err, err)

        # Upper triangle should be zero
        upper = np.triu(L, k=1)
        max_upper_err = max(max_upper_err, np.abs(upper).max())

        # Diagonal should be positive
        diag = np.diag(L)
        min_diag = min(min_diag, diag.min())

    print(f"  max |L·Lᵀ − A_pair|         : {max_recon_err:.2e}")
    print(f"  max |upper-triangle entry|  : {max_upper_err:.2e}")
    print(f"  min L diagonal              : {min_diag:.2e}")

    print()
    if (
        n_failed == 0
        and max_recon_err < 1e-3
        and max_upper_err == 0.0
        and min_diag > 0.0
    ):
        print("✓ Phase 3 Cholesky validated.")
    else:
        print("✗ Phase 3 validation FAILED.")


if __name__ == "__main__":
    main()
