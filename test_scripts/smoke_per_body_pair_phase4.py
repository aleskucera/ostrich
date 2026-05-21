"""Phase 4 smoke test for PerBodyPairPreconditioner.

Verifies that the matvec correctly computes z = beta·y + alpha·M⁻¹·x
where M is the block-diagonal preconditioner with blocks indexed by
per-body-pair.

Validation against a numpy reference:
  * For each active pair, gather x at the pair's constraint indices,
    solve A_pair · z_pair = x_pair using numpy's direct solver,
    scatter z_pair to the corresponding output indices.
  * For inactive constraints, z[c] = beta·y[c] (no preconditioner
    contribution).
  * Compare to the GPU matvec output.

Tests four (alpha, beta) combinations to exercise the linear-combo
logic that PCRSolver actually uses.
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
    target_step = 30

    orig_update = JacobiPreconditioner.update
    rng = np.random.default_rng(42)

    def patched_update(self):
        orig_update(self)
        if state["captured"]:
            return
        if state["step"] != target_step or state["iter"] != 0:
            state["iter"] += 1
            return

        # Run the full setup pass
        precond.update()

        N_c = engine.dims.num_constraints
        # Build random x and y
        x_np = rng.standard_normal((1, N_c)).astype(np.float32)
        y_np = rng.standard_normal((1, N_c)).astype(np.float32)
        x = wp.array(x_np, dtype=wp.float32, device=engine.device)
        y = wp.array(y_np, dtype=wp.float32, device=engine.device)

        # Run matvec under several (alpha, beta) combos
        cases = [
            (1.0, 0.0),     # pure preconditioner-apply
            (0.0, 1.0),     # pure y pass-through
            (1.0, 1.0),     # the typical PCR call
            (0.7, -0.3),    # arbitrary scaling
        ]
        gpu_results = {}
        for alpha, beta in cases:
            z = wp.zeros_like(x)
            precond.matvec(x, y, z, float(alpha), float(beta))
            wp.synchronize()
            gpu_results[(alpha, beta)] = z.numpy()[0].copy()

        state["result"] = {
            "x": x_np[0],
            "y": y_np[0],
            "gpu": gpu_results,
            "A_blocks": precond.A_blocks.numpy(),
            "member_count": precond.pair_member_count.numpy(),
            "member_list": precond.pair_member_list.numpy(),
            "factor_failure": precond.factor_failure.numpy(),
            "constr_pair_id": precond.constr_pair_id.numpy(),
            "active_mask": engine.data._constr_active_mask.numpy(),
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
    x = R["x"]
    y = R["y"]
    A_blocks = R["A_blocks"][w]
    member_count = R["member_count"][w]
    member_list = R["member_list"][w]
    factor_failure = R["factor_failure"][w]
    active_pair_ids = np.where(member_count > 0)[0]

    if int(factor_failure.sum()) > 0:
        print(f"WARN: {factor_failure.sum()} pairs flagged factor_failure; "
              f"M⁻¹ test still proceeds (Jacobi fallback path).")

    # Build numpy reference for M⁻¹·x
    Minv_x_ref = np.zeros_like(x)
    for pid in active_pair_ids:
        n = int(member_count[pid])
        constraints = [int(member_list[pid, k]) for k in range(n)]
        A = A_blocks[pid, :n, :n].astype(np.float64)
        x_pair = x[constraints].astype(np.float64)
        z_pair = np.linalg.solve(A, x_pair)
        for k, c in enumerate(constraints):
            Minv_x_ref[c] = z_pair[k]

    print(f"Validating matvec on {len(active_pair_ids)} active pairs...")

    all_pass = True
    for (alpha, beta), z_gpu in R["gpu"].items():
        z_ref = beta * y + alpha * Minv_x_ref
        diff = np.abs(z_gpu - z_ref)
        max_err = diff.max()
        # Relative error vs the larger of |alpha|·||M⁻¹·x||_∞ and |beta|·||y||_∞
        scale = max(
            abs(alpha) * np.abs(Minv_x_ref).max(),
            abs(beta) * np.abs(y).max(),
            1e-10,
        )
        rel_err = max_err / scale
        passed = rel_err < 1e-4
        marker = "✓" if passed else "✗"
        print(f"  {marker} α={alpha:+.2f} β={beta:+.2f}: "
              f"max_abs_err={max_err:.2e} rel_err={rel_err:.2e}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("✓ Phase 4 matvec validated.")
    else:
        print("✗ Phase 4 validation FAILED.")


if __name__ == "__main__":
    main()
