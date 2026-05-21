"""Run surface_drive with monkey-patched apply diag dump."""
import os, sys
sys.path.insert(0, "test_scripts")
import warp as wp
import hydra
import numpy as np
from omegaconf import DictConfig

@hydra.main(config_path="../examples/conf", config_name="helhest", version_base=None)
def main(cfg):
    from axion import EngineConfig, LoggingConfig, RenderingConfig, SimulationConfig
    from axion.collision.warm_start import ContactWarmStarter
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "helhest"))
    from surface_drive import HelhestSurfaceSimulator as HelhestSurfaceDrive

    sim_config = hydra.utils.instantiate(cfg.simulation)
    render_config = hydra.utils.instantiate(cfg.rendering)
    engine_config = hydra.utils.instantiate(cfg.engine)
    logging_config = hydra.utils.instantiate(cfg.logging)
    sim = HelhestSurfaceDrive(sim_config, render_config, engine_config, logging_config,
        control_mode=cfg.control.mode, k_p=cfg.control.k_p, k_d=cfg.control.k_d,
        friction=cfg.friction_coeff)
    records = []
    orig = ContactWarmStarter.apply
    def patched(self, contacts, data, dt):
        orig(self, contacts, data, dt)
        if not self.enabled: return
        wp.synchronize()
        c = int(self._diag_closest_d_count.numpy()[0])
        records.append((
            int(self._diag_attempts.numpy()[0]),
            int(self._diag_matched.numpy()[0]),
            int(self._diag_over_thresh.numpy()[0]),
            (float(self._diag_closest_d_sum.numpy()[0]) / c) if c else 0.0,
            (float(self._diag_closest_d_raw_sum.numpy()[0]) / c) if c else 0.0,
        ))
    ContactWarmStarter.apply = patched
    sim.run()
    R = np.array(records)
    if not len(R): return
    a, m, o, cd, cd_raw = R.T
    print(f"\nsurface_drive: total attempts={int(a.sum())} matched={int(m.sum())} ({100*m.sum()/max(1,a.sum()):.1f}%)")
    print(f"\n--- mid (steps 100..120) ---")
    print(f"{'step':>4} {'att':>4} {'match':>5} {'over':>5} {'cd_mm':>6} {'cd_raw':>6}")
    for i in range(100, min(120, len(R))):
        print(f"{int(i):>4d} {int(a[i]):>4d} {int(m[i]):>5d} {int(o[i]):>5d} {cd[i]*1000:>6.2f} {cd_raw[i]*1000:>6.2f}")

main()
