"""Step-by-step warm-start diagnostics: monkey-patch
ContactWarmStarter.apply to read its diag counters AFTER each call,
record per-step (matched, unmatched, cold_normal, cold_friction).
"""
import os
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

CONFIG_PATH = "../examples/conf"


@hydra.main(config_path=CONFIG_PATH, config_name="helhest_obstacle_benchmark", version_base=None)
def main(cfg: DictConfig):
    import warp as wp
    from axion import (
        EngineConfig,
        LoggingConfig,
        RenderingConfig,
        SimulationConfig,
    )
    from axion.collision.warm_start import ContactWarmStarter

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

    # Monkey-patch: log diag counters after each apply().
    records = []
    orig_apply = ContactWarmStarter.apply
    def patched(self, contacts, data, dt):
        orig_apply(self, contacts, data, dt)
        if not self.enabled:
            return
        wp.synchronize()
        cd_count = int(self._diag_closest_d_count.numpy()[0])
        cd_sum = float(self._diag_closest_d_sum.numpy()[0])
        cd_raw_sum = float(self._diag_closest_d_raw_sum.numpy()[0])
        cd_mean = (cd_sum / cd_count) if cd_count else 0.0
        cd_raw_mean = (cd_raw_sum / cd_count) if cd_count else 0.0
        records.append((
            int(self._diag_attempts.numpy()[0]),
            int(self._diag_matched.numpy()[0]),
            int(self._diag_cold_normal.numpy()[0]),
            int(self._diag_cold_friction.numpy()[0]),
            int(self._diag_no_pair.numpy()[0]),
            int(self._diag_over_thresh.numpy()[0]),
            cd_mean,
            cd_raw_mean,
        ))
    ContactWarmStarter.apply = patched

    sim.run()

    R = np.array(records, dtype=object)
    if len(R) == 0:
        print("No records collected — warm start disabled?")
        return
    attempts = np.array([r[0] for r in R], dtype=int)
    matched = np.array([r[1] for r in R], dtype=int)
    cold_n = np.array([r[2] for r in R], dtype=int)
    cold_f = np.array([r[3] for r in R], dtype=int)
    no_pair = np.array([r[4] for r in R], dtype=int)
    over_thresh = np.array([r[5] for r in R], dtype=int)
    cd_mean = np.array([r[6] for r in R], dtype=float)
    cd_raw_mean = np.array([r[7] for r in R], dtype=float)
    unmatched = attempts - matched
    # Sanity: unmatched should equal no_pair + over_thresh.
    # Steps where attempts==0 (e.g., before contact) are fine — both counts 0.
    print(f"\n=== Warm-start per-step stats over {len(R)} steps ===")
    print(f"attempts       p50={int(np.percentile(attempts,50))} p95={int(np.percentile(attempts,95))} max={int(attempts.max())}")
    print(f"matched        p50={int(np.percentile(matched,50))} p95={int(np.percentile(matched,95))} max={int(matched.max())}")
    print(f"unmatched      p50={int(np.percentile(unmatched,50))} p95={int(np.percentile(unmatched,95))} max={int(unmatched.max())}")
    print(f"cold_normal    p50={int(np.percentile(cold_n,50))} p95={int(np.percentile(cold_n,95))} max={int(cold_n.max())} sum={int(cold_n.sum())}")
    print(f"cold_friction  p50={int(np.percentile(cold_f,50))} p95={int(np.percentile(cold_f,95))} max={int(cold_f.max())} sum={int(cold_f.sum())}")
    print(f"\nFailure-mode breakdown (sum across all steps):")
    total_attempts = int(attempts.sum())
    total_matched = int(matched.sum())
    total_no_pair = int(no_pair.sum())
    total_over = int(over_thresh.sum())
    print(f"  total attempts:   {total_attempts}")
    print(f"  total matched:    {total_matched} ({100*total_matched/max(1,total_attempts):.1f}%)")
    print(f"  total no_pair:    {total_no_pair} ({100*total_no_pair/max(1,total_attempts):.1f}%)")
    print(f"  total over_thresh: {total_over} ({100*total_over/max(1,total_attempts):.1f}%)")
    nz = cd_mean[cd_mean > 0]
    if len(nz):
        print(f"\nMean closest-in-pair distance (m), across steps with pair_seen:")
        print(f"  p25={np.percentile(nz,25)*1000:.2f} mm  "
              f"p50={np.percentile(nz,50)*1000:.2f} mm  "
              f"p75={np.percentile(nz,75)*1000:.2f} mm  "
              f"p95={np.percentile(nz,95)*1000:.2f} mm  "
              f"max={nz.max()*1000:.2f} mm")
    print(f"\nsteps with cold_normal>0:   {int((cold_n>0).sum())}/{len(R)}")
    print(f"steps with cold_friction>0: {int((cold_f>0).sum())}/{len(R)}")
    if (cold_n > 0).any():
        idx_first = int(np.argmax(cold_n > 0))
        print(f"\nFirst step with cold-start: step {idx_first}, "
              f"unmatched={unmatched[idx_first]}, cold_normal={cold_n[idx_first]}")

    def show(start, end, title):
        end = min(end, len(R))
        if end <= start: return
        print(f"\n--- {title} (steps {start}..{end-1}) ---")
        print(f"{'step':>4} {'att':>4} {'match':>5} {'no_pair':>7} {'over':>5} "
              f"{'cd_mm':>6} {'cd_raw':>6} {'cold_n':>6} {'cold_f':>6}")
        for i in range(start, end):
            print(f"{i:>4d} {attempts[i]:>4d} {matched[i]:>5d} {no_pair[i]:>7d} "
                  f"{over_thresh[i]:>5d} {cd_mean[i]*1000:>6.2f} {cd_raw_mean[i]*1000:>6.2f} "
                  f"{cold_n[i]:>6d} {cold_f[i]:>6d}")
    show(0, 20, "head")
    show(60, 70, "mid")
    show(150, 160, "late")


if __name__ == "__main__":
    main()
