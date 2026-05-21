"""Phase-1 verification: run a few obstacle-benchmark steps with warm
start enabled, then inspect the ContactWarmStarter._prev_* buffers to
confirm the snapshot kernel wrote sensible values.

Expected post-run state (after the robot has dropped into ground contact):
  _prev_count[0] > 0           — at least one active contact stored
  any |_prev_lambda_n[0, :]| > 0   — converged normal forces non-zero
  _prev_b0 / _prev_b1 plausible    — robot/ground body indices
  _prev_p_world plausible          — z near 0 (ground level)
"""
import os
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


CONFIG_PATH = "../examples/conf"


@hydra.main(config_path=CONFIG_PATH, config_name="helhest_obstacle_benchmark", version_base=None)
def main(cfg: DictConfig):
    from axion import (
        EngineConfig,
        LoggingConfig,
        RenderingConfig,
        SimulationConfig,
    )

    # Force warm start on for this verification
    OmegaConf.update(cfg, "engine.enable_contact_warm_start", True, force_add=True)

    # Limit to a short run (10 steps is plenty)
    OmegaConf.update(cfg, "simulation.duration_seconds", 0.3)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "helhest"))
    from obstacle_benchmark import HelhestObstacleBenchmark  # noqa: E402

    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    sim = HelhestObstacleBenchmark(
        sim_config, render_config, engine_config, logging_config,
        control_mode=cfg.control.mode,
        k_p=cfg.control.k_p, k_d=cfg.control.k_d,
        friction=cfg.friction_coeff, drive_velocity=cfg.drive_velocity,
    )
    sim.run()

    ws = sim.solver.warm_starter
    print("\n========== ContactWarmStarter._prev_* inspection ==========")
    prev_count = ws._prev_count.numpy()
    prev_b0 = ws._prev_b0.numpy()
    prev_b1 = ws._prev_b1.numpy()
    prev_p_world = ws._prev_p_world.numpy()
    prev_normal = ws._prev_normal.numpy()
    prev_lambda_n = ws._prev_lambda_n.numpy()
    prev_lambda_t = ws._prev_lambda_t.numpy()

    print(f"prev_count (per world):  {prev_count}")
    n_active = int(prev_count[0])
    if n_active == 0:
        print("⚠ no contacts in last snapshot — robot might be airborne in this run")
        return

    print(f"\nworld 0, active slots [0, {n_active}):")
    print(f"{'idx':>4} {'b0':>3} {'b1':>3} | {'p_world':>30} | "
          f"{'normal':>26} | {'λ_n':>9} | {'λ_t':>22}")
    for i in range(min(n_active, 12)):
        print(f"{i:>4d} {prev_b0[0,i]:>3d} {prev_b1[0,i]:>3d} | "
              f"{prev_p_world[0,i]} | "
              f"{prev_normal[0,i]} | "
              f"{prev_lambda_n[0,i]:>9.2f} | "
              f"{prev_lambda_t[0,i]}")

    # Sanity checks
    print()
    nonzero_lambda_n = int(np.sum(np.abs(prev_lambda_n[0, :n_active]) > 1e-6))
    nonzero_lambda_t = int(
        np.sum(np.linalg.norm(prev_lambda_t[0, :n_active, :], axis=-1) > 1e-6)
    )
    print(f"slots with |λ_n| > 1e-6:   {nonzero_lambda_n}/{n_active}")
    print(f"slots with |λ_t| > 1e-6:   {nonzero_lambda_t}/{n_active}")

    z_min = float(np.min(prev_p_world[0, :n_active, 2]))
    z_max = float(np.max(prev_p_world[0, :n_active, 2]))
    print(f"contact z range: [{z_min:.4f}, {z_max:.4f}]  "
          f"(should be near 0 if rolling on ground)")

    n_z = prev_normal[0, :n_active, 2]
    print(f"normal z component: min={float(n_z.min()):+.3f}  "
          f"max={float(n_z.max()):+.3f}  "
          f"(should be ~+1 for ground contacts)")

    print()
    print("---- last-step match diagnostics (from .apply at start of step) ----")
    diag_attempts = ws._diag_attempts.numpy()[0]
    diag_matched = ws._diag_matched.numpy()[0]
    diag_max_dist = ws._diag_max_dist.numpy()[0]
    rate = (diag_matched / diag_attempts) if diag_attempts else 0.0
    print(f"attempts={int(diag_attempts)}  matched={int(diag_matched)}  "
          f"rate={rate:.1%}  max_match_dist={float(diag_max_dist)*1000:.2f} mm")


if __name__ == "__main__":
    main()
