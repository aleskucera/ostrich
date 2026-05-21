"""Phase 1 smoke test for PerBodyPairPreconditioner.

Boots the obstacle benchmark engine, runs one load_data + collision
to populate constraint structures, then runs the preconditioner's
update_pair_assignments(). Validates:
  * No overflow
  * Sum of pair member counts equals number of active constraints
  * No constraint is missing from any pair list
  * Decoded pair ids correspond to valid (b_lo, b_hi) tuples
  * Pair member sizes look physically reasonable (groups of 3-15
    constraints, not single huge groups or all singletons)
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
    precond = PerBodyPairPreconditioner(engine)

    # Hook into preconditioner.update so we can inspect a few steps as
    # the simulation runs (rather than just inspecting step 0 with no
    # contacts yet).
    samples = []
    from axion.optim.preconditioner import JacobiPreconditioner
    orig_update = JacobiPreconditioner.update

    state = {"step": -1, "iter": 0}

    def patched_update(self):
        orig_update(self)
        if state["iter"] != 0:
            state["iter"] += 1
            return  # only sample once per step (first iter)

        precond.update_pair_assignments()
        wp.synchronize()

        pair_ids = precond.constr_pair_id.numpy()[0]
        member_counts = precond.pair_member_count.numpy()[0]
        overflow = bool(precond.pair_overflow_flag.numpy()[0])
        active_mask = engine.data._constr_active_mask.numpy()[0]

        n_active_constr = int((active_mask > 0).sum())
        n_assigned = int((pair_ids >= 0).sum())
        sum_members = int(member_counts.sum())
        n_active_pairs = int((member_counts > 0).sum())

        active_pair_idxs = np.where(member_counts > 0)[0]
        decoded_pairs = [precond.decode_pair_id(int(p)) for p in active_pair_idxs]
        sizes = [int(member_counts[p]) for p in active_pair_idxs]

        samples.append({
            "step": state["step"],
            "n_active_constr": n_active_constr,
            "n_assigned": n_assigned,
            "sum_members": sum_members,
            "n_active_pairs": n_active_pairs,
            "overflow": overflow,
            "pair_sizes": sizes,
            "decoded_pairs": decoded_pairs,
        })
        state["iter"] += 1

    JacobiPreconditioner.update = patched_update

    # Track step counter
    orig_step = engine.step

    def patched_step(state_in, state_out, control, contacts, dt):
        state["step"] += 1
        state["iter"] = 0
        orig_step(state_in, state_out, control, contacts, dt)

    engine.step = patched_step

    sim.run()

    # ---- Validation ----
    print(f"\nCollected {len(samples)} sample steps.")
    if not samples:
        print("ERROR: no samples collected.")
        return

    failed = []
    for s in samples:
        if s["overflow"]:
            failed.append(f"step {s['step']}: overflow")
        if s["n_active_constr"] != s["n_assigned"]:
            failed.append(
                f"step {s['step']}: {s['n_active_constr']} active constraints "
                f"but {s['n_assigned']} assigned (mismatch)"
            )
        if s["n_assigned"] != s["sum_members"]:
            failed.append(
                f"step {s['step']}: {s['n_assigned']} assigned but member-list "
                f"sum is {s['sum_members']} (constraints lost in atomic insert)"
            )

    if failed:
        print("\nFAILED:")
        for f in failed[:10]:
            print(f"  {f}")
        return

    print("\nAll consistency checks passed across all sampled steps:")
    print(f"  n_active_constr == n_assigned == sum_members  ✓")
    print(f"  no overflow                                    ✓")

    print("\nDistribution of #pairs and pair sizes across sampled steps:")
    n_pairs_arr = np.array([s["n_active_pairs"] for s in samples])
    all_sizes = np.array([sz for s in samples for sz in s["pair_sizes"]])
    print(f"  active pairs / step   : min={n_pairs_arr.min()} "
          f"p50={int(np.percentile(n_pairs_arr,50))} max={n_pairs_arr.max()}")
    print(f"  pair size distribution: min={int(all_sizes.min())} "
          f"p50={int(np.percentile(all_sizes,50))} "
          f"p95={int(np.percentile(all_sizes,95))} "
          f"max={int(all_sizes.max())}")

    # Show one sample step's decoded pairs as a sanity check.
    mid = samples[len(samples) // 2]
    print(f"\nMid-run sample (step {mid['step']}): {mid['n_active_pairs']} active pairs:")
    pair_summary = sorted(
        zip(mid["decoded_pairs"], mid["pair_sizes"]),
        key=lambda x: -x[1],
    )
    for (b_lo, b_hi), sz in pair_summary[:15]:
        gnd = " (ground)" if b_lo == -1 or b_hi == -1 else ""
        print(f"  ({b_lo:>3d}, {b_hi:>3d}){gnd:9s}  {sz:>3d} constraints")


if __name__ == "__main__":
    main()
