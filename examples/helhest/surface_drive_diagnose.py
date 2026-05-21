"""Diagnostic run: collect PCR convergence traces over many Helhest-on-surface
steps and dump them to NPZ for pattern analysis.

Subclasses the benchmark scene; the only difference is that after every step
we copy the engine's per-NR-iter PCR history (iter count + residual decay) to
host memory, then aggregate and print summary statistics at the end.
"""
import os
import pathlib
from typing import override

import hydra
import numpy as np
import warp as wp
from omegaconf import DictConfig

try:
    from examples.helhest.surface_drive_benchmark import HelhestSurfaceBenchmark
except ImportError:
    from surface_drive_benchmark import HelhestSurfaceBenchmark

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")


class HelhestSurfaceDiagnose(HelhestSurfaceBenchmark):
    """Captures PCR per-iter residual histories per step and dumps them."""

    def __init__(self, *args, npz_output: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.npz_output = npz_output

        engine = self.solver
        self.max_newton_iters = engine.config.nr.max_iters
        self.max_linear_iters = engine.config.linear.max_iters
        self.num_worlds = engine.dims.num_worlds
        self.linesearch_enabled = bool(engine.config.linesearch.enabled)

        # Linesearch step size grid is fixed at construction (a single 1D
        # array shared across all NR iters). Capture it once.
        if self.linesearch_enabled:
            self._ls_step_sizes = engine.data.linesearch_step_size.numpy().copy()
        else:
            self._ls_step_sizes = None

        self._iter_counts_per_step = []   # [n_steps, max_newton_iters]
        self._residuals_per_step = []     # [n_steps, max_newton_iters, max_linear_iters+1, num_worlds]
        self._nr_iters_per_step = []      # [n_steps]
        self._nr_res_norm_sq_per_step = []  # [n_steps, max_newton_iters, num_worlds]
        self._ls_min_idx_per_step = []    # [n_steps, max_newton_iters, num_worlds] (only if linesearch on)
        self._ls_res_per_step = []        # [n_steps, max_newton_iters, step_count, num_worlds]
        # Trajectories so we can diff w/ vs w/o linesearch.
        self._body_pose_per_step = []     # [n_steps, num_worlds, body_count, 7]
        self._body_vel_per_step = []      # [n_steps, num_worlds, body_count, 6]

    @override
    def _run_segment_with_graph(self, segment_num: int):
        super()._run_segment_with_graph(segment_num)
        self._capture_step_history()

    @override
    def _run_segment_without_graph(self, segment_num: int):
        super()._run_segment_without_graph(segment_num)
        self._capture_step_history()

    def _capture_step_history(self):
        engine = self.solver
        data = engine.data

        # Force the device to finish before we read the history buffers.
        wp.synchronize_device(self.model.device)

        nr_iters = int(data.iter_count.numpy()[0])
        iter_counts = data.pcr_history_iter_count.numpy().reshape(self.max_newton_iters)
        residuals = data.pcr_history_res_norm_sq_history.numpy().copy()
        # Newton residual norm at each NR-iter (post-step). Shape (K, W).
        nr_res = data.candidates_res_norm_sq.numpy().copy()

        self._nr_iters_per_step.append(nr_iters)
        self._iter_counts_per_step.append(iter_counts)
        self._residuals_per_step.append(residuals)
        self._nr_res_norm_sq_per_step.append(nr_res)

        if self.linesearch_enabled:
            # Per-NR-iter chosen-α index. Shape (K, W).
            ls_idx = data.ls_history_minimal_index.numpy().copy()
            self._ls_min_idx_per_step.append(ls_idx)
            # Per-NR-iter, per-candidate residual squared. Shape (K, A, W).
            ls_res = data.ls_history_res_norm_sq.numpy().copy()
            self._ls_res_per_step.append(ls_res)

        # Snapshot post-step body state (world 0). Used to diff trajectories
        # between runs with and without linesearch.
        self._body_pose_per_step.append(data.body_pose.numpy().copy())
        self._body_vel_per_step.append(data.body_vel.numpy().copy())

    def run(self):
        try:
            super().run()
        finally:
            self._dump_and_summarize()

    def _dump_and_summarize(self):
        if not self._iter_counts_per_step:
            print("[diagnose] no steps captured; nothing to dump.")
            return

        nr_iters = np.asarray(self._nr_iters_per_step, dtype=np.int32)
        iter_counts = np.stack(self._iter_counts_per_step)         # (S, K)
        residuals = np.stack(self._residuals_per_step)             # (S, K, L+1, W)
        nr_res = np.stack(self._nr_res_norm_sq_per_step)           # (S, K, W)

        ls_min_idx = None
        if self.linesearch_enabled and self._ls_min_idx_per_step:
            ls_min_idx = np.stack(self._ls_min_idx_per_step)       # (S, K, W)

        os.makedirs(os.path.dirname(self.npz_output) or ".", exist_ok=True)
        save_kwargs = dict(
            nr_iters=nr_iters,
            pcr_iter_counts=iter_counts,
            pcr_residual_sq_history=residuals,
            nr_res_norm_sq=nr_res,
            max_newton_iters=self.max_newton_iters,
            max_linear_iters=self.max_linear_iters,
            num_worlds=self.num_worlds,
        )
        if ls_min_idx is not None:
            save_kwargs["ls_min_idx"] = ls_min_idx
            save_kwargs["ls_step_sizes"] = self._ls_step_sizes
            if self._ls_res_per_step:
                save_kwargs["ls_res_norm_sq"] = np.stack(self._ls_res_per_step)
        save_kwargs["body_pose"] = np.stack(self._body_pose_per_step)
        save_kwargs["body_vel"] = np.stack(self._body_vel_per_step)
        np.savez(self.npz_output, **save_kwargs)
        print(f"\n[diagnose] dumped trace to {self.npz_output}")
        self._print_summary(nr_iters, iter_counts, residuals, nr_res, ls_min_idx)

    def _print_summary(self, nr_iters, iter_counts, residuals, nr_res, ls_min_idx=None):
        max_K = self.max_newton_iters
        max_L = self.max_linear_iters
        n_steps = iter_counts.shape[0]

        # Build a (step, nr_iter) mask that excludes NR iters that didn't run.
        idx = np.arange(max_K)[None, :]
        ran_mask = idx < nr_iters[:, None]    # (S, K)

        flat_iters = iter_counts[ran_mask]
        total_pcr = int(flat_iters.sum())
        n_pairs = int(ran_mask.sum())

        print("=" * 72)
        print(f"[diagnose] {n_steps} steps, {n_pairs} (step, NR-iter) pairs")
        print("-" * 72)

        # 1. Newton iter count distribution
        nr_max = int(nr_iters.max())
        nr_min = int(nr_iters.min())
        nr_mean = float(nr_iters.mean())
        print(f"NR iters per step    min={nr_min}  mean={nr_mean:.2f}  max={nr_max}")
        nr_hit_max = int((nr_iters >= max_K).sum())
        print(f"NR-iters hit max ({max_K}): {nr_hit_max}/{n_steps} steps  "
              f"({100*nr_hit_max/n_steps:.1f}%)")

        # 2. PCR iter count distribution across all (step, NR-iter) pairs
        pcr_min = int(flat_iters.min())
        pcr_max = int(flat_iters.max())
        pcr_mean = float(flat_iters.mean())
        pcr_p50 = float(np.percentile(flat_iters, 50))
        pcr_p95 = float(np.percentile(flat_iters, 95))
        print(f"PCR iters per NR-it  min={pcr_min}  mean={pcr_mean:.2f}  "
              f"p50={pcr_p50:.0f}  p95={pcr_p95:.0f}  max={pcr_max}")

        pcr_hit_max = int((flat_iters >= max_L).sum())
        print(f"PCR hit max ({max_L}): {pcr_hit_max}/{n_pairs} solves  "
              f"({100*pcr_hit_max/n_pairs:.1f}%)")

        bins = np.bincount(flat_iters.astype(np.int64), minlength=max_L + 2)
        print("PCR iter histogram   |", end="")
        for k, c in enumerate(bins):
            if c == 0:
                continue
            print(f" {k}:{c}", end="")
        print()

        # 3. Per-NR-iter median PCR iter count (does early NR cost more?)
        per_nr_med = []
        per_nr_mean = []
        for k in range(max_K):
            col = iter_counts[ran_mask[:, k], k]
            if col.size:
                per_nr_med.append(int(np.median(col)))
                per_nr_mean.append(float(col.mean()))
            else:
                per_nr_med.append(-1)
                per_nr_mean.append(float("nan"))
        print(f"PCR iters by NR-iter (median): {per_nr_med}")
        # Show first few and last few means with 2 decimals
        head = ", ".join(f"{v:.2f}" for v in per_nr_mean[:6])
        tail = ", ".join(f"{v:.2f}" for v in per_nr_mean[-6:])
        print(f"PCR iters by NR-iter (mean):   [{head}, ..., {tail}]")

        # 4. Convergence quality: log-residual drop per iter, averaged over solves
        # residuals[s, k, ell, w] = r²; iter count for (s,k) is iter_counts[s,k].
        # We compute, for each (s,k) with ran_mask, the log10(r²[ell] / r²[0])
        # at ell = 1, 4, 8, max_L.
        eps = 1e-30
        log_drops = {1: [], 4: [], 8: [], max_L: []}
        plateau_count = 0
        plateau_threshold = 0.05  # less than ~10% drop in last 4 iters → plateau

        for s in range(n_steps):
            for k in range(int(nr_iters[s])):
                hist = residuals[s, k, :, 0]  # only world 0
                count = int(iter_counts[s, k])
                if count <= 1:
                    continue
                r0 = hist[0] + eps
                for ell in log_drops:
                    if ell <= count:
                        log_drops[ell].append(np.log10((hist[ell] + eps) / r0))
                # Plateau detection: in the last 4 iters before stop, did r²
                # drop by less than `plateau_threshold` decades?
                if count >= 5:
                    last_drop = np.log10((hist[count] + eps) / (hist[count - 4] + eps))
                    if abs(last_drop) < plateau_threshold:
                        plateau_count += 1

        print("-" * 72)
        print("Residual decay (log10 r²[ell] / r²[0], world 0):")
        for ell in sorted(log_drops):
            arr = np.asarray(log_drops[ell]) if log_drops[ell] else np.zeros(0)
            if arr.size:
                print(f"  iter={ell:>3}: mean={arr.mean():+.2f}  "
                      f"p50={np.percentile(arr, 50):+.2f}  "
                      f"p95={np.percentile(arr, 95):+.2f}  n={arr.size}")
            else:
                print(f"  iter={ell:>3}: (no solves ran this many iters)")
        print(f"Plateau (last 4 iters drop < {plateau_threshold} decades): "
              f"{plateau_count}/{n_pairs} solves "
              f"({100*plateau_count/max(n_pairs,1):.1f}%)")

        # 5. Newton residual progression: are we close to atol or far?
        # nr_res[s, k, w] = ||r||² at the candidate stored after NR iter k.
        # An "always max iters" symptom only matters if the final residual
        # is well above atol². Show the final residual of each step (world 0).
        eps_nr = 1e-30
        print("-" * 72)
        final_nr = np.array(
            [nr_res[s, int(nr_iters[s]) - 1, 0] for s in range(n_steps)]
        )
        log10_final = np.log10(final_nr + eps_nr)
        print(f"NR final ||r||² (world 0):  "
              f"min={final_nr.min():.2e}  "
              f"p50={np.percentile(final_nr, 50):.2e}  "
              f"p95={np.percentile(final_nr, 95):.2e}  "
              f"max={final_nr.max():.2e}")
        print(f"NR final log10||r||²:       "
              f"min={log10_final.min():+.2f}  "
              f"p50={np.percentile(log10_final, 50):+.2f}  "
              f"p95={np.percentile(log10_final, 95):+.2f}  "
              f"max={log10_final.max():+.2f}")
        # Per-NR-iter median of log10 ||r||² across steps (world 0).
        per_nr_log = []
        for k in range(max_K):
            col = nr_res[ran_mask[:, k], k, 0]
            if col.size:
                per_nr_log.append(float(np.log10(np.median(col) + eps_nr)))
            else:
                per_nr_log.append(float("nan"))
        head = ", ".join(f"{v:+.2f}" for v in per_nr_log[:8])
        tail = ", ".join(f"{v:+.2f}" for v in per_nr_log[-8:])
        print(f"NR log10||r||² by iter median: [{head}, ..., {tail}]")

        # 6. Linesearch chosen-α distribution (only when linesearch is on).
        if ls_min_idx is not None and self._ls_step_sizes is not None:
            print("-" * 72)
            grid = self._ls_step_sizes        # shape (step_count,)
            # Map per-(step, NR-iter, world) chosen index -> α value.
            # ls_min_idx shape (S, K, W); apply ran_mask over (S, K).
            world_alpha = grid[ls_min_idx[:, :, 0]]  # shape (S, K), world 0
            chosen_alpha = world_alpha[ran_mask]
            print(f"Linesearch step-size grid: {grid.size} candidates from "
                  f"{grid.min():.3e} to {grid.max():.3e}; median {np.median(grid):.3f}")
            print(f"Chosen α (world 0): "
                  f"min={chosen_alpha.min():.3e}  "
                  f"p25={np.percentile(chosen_alpha, 25):.3f}  "
                  f"p50={np.percentile(chosen_alpha, 50):.3f}  "
                  f"p75={np.percentile(chosen_alpha, 75):.3f}  "
                  f"max={chosen_alpha.max():.3f}")
            # Bucket into rough regimes
            n_tiny = int((chosen_alpha < 0.05).sum())
            n_small = int(((chosen_alpha >= 0.05) & (chosen_alpha < 0.5)).sum())
            n_mid = int(((chosen_alpha >= 0.5) & (chosen_alpha < 0.9)).sum())
            n_unit = int(((chosen_alpha >= 0.9) & (chosen_alpha <= 1.1)).sum())
            n_over = int((chosen_alpha > 1.1).sum())
            print(f"Chosen-α buckets:  "
                  f"<0.05: {n_tiny}  "
                  f"0.05–0.5: {n_small}  "
                  f"0.5–0.9: {n_mid}  "
                  f"~1.0: {n_unit}  "
                  f">1.1: {n_over}  "
                  f"(total: {chosen_alpha.size})")
            # Per-NR-iter median α across steps (world 0).
            per_nr_alpha = []
            for k in range(max_K):
                col = world_alpha[ran_mask[:, k], k]
                if col.size:
                    per_nr_alpha.append(float(np.median(col)))
                else:
                    per_nr_alpha.append(float("nan"))
            head_a = ", ".join(f"{v:.3f}" for v in per_nr_alpha[:8])
            tail_a = ", ".join(f"{v:.3f}" for v in per_nr_alpha[-8:])
            print(f"Chosen α by NR-iter (median):  [{head_a}, ..., {tail_a}]")
        print("=" * 72)


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest_diagnose", version_base=None)
def helhest_surface_diagnose(cfg: DictConfig):
    from axion import EngineConfig
    from axion import LoggingConfig
    from axion import RenderingConfig
    from axion import SimulationConfig

    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    simulator = HelhestSurfaceDiagnose(
        sim_config,
        render_config,
        engine_config,
        logging_config,
        control_mode=cfg.control.mode,
        k_p=cfg.control.k_p,
        k_d=cfg.control.k_d,
        friction=cfg.friction_coeff,
        drive_velocity=cfg.drive_velocity,
        npz_output=cfg.npz_output,
    )
    simulator.run()


if __name__ == "__main__":
    helhest_surface_diagnose()
