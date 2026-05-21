"""Measure how much off-diagonal coupling exists within per-body
constraint groups of A. Determines whether per-body block-Jacobi has
real headroom over plain (diagonal) Jacobi.

For each body B with > 1 constraint touching it, build A_BB by direct
computation from J_values and M⁻¹, then report the ratio
||off-diag(A_BB)||_F / ||diag(A_BB)||_2.

Interpretation:
  ratio < 0.1   →  A_BB is mostly diagonal; per-body block-Jacobi
                   would barely help. Diagonal Jacobi already
                   captures the relevant structure.
  0.1 < ratio < 0.5  →  modest off-diagonal coupling; per-body block
                   might give 10–20% PCR-iter reduction.
  ratio > 0.5   →  large off-diagonal coupling that diagonal Jacobi
                   misses; per-body block should pay off.

Per-body samples are aggregated across NR iters and sim steps so we
get a distribution, not a single value.
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

    samples_per_body = []  # list of dicts: {step, iter, body, k, ratio, diag_norm, offdiag_norm}
    state = {"step": -1, "iter": 0}

    orig_update = JacobiPreconditioner.update

    def patched_update(self):
        orig_update(self)
        wp.synchronize()
        # Sample at a subset of iters per step to keep cost down.
        if state["iter"] not in (0, 4, 8, 12):
            state["iter"] += 1
            return

        # Gather all the data we need (CPU side)
        J_values = self.engine.data._J_values.numpy()[0]  # (N_c, 2, 6)
        body_idx = self.engine.data._constr_body_idx.numpy()[0]  # (N_c, 2)
        active = self.engine.data._constr_active_mask.numpy()[0]  # (N_c,)
        C_vals = self.engine.data._C_values.numpy()[0]  # (N_c,)
        body_inv_mass = self.engine.axion_model.body_inv_mass.numpy()[0]  # (N_b,)
        body_inv_inertia = self.engine.data.world_inv_inertia.numpy()[0]  # (N_b, 3, 3)

        N_c = J_values.shape[0]
        N_b = body_inv_mass.shape[0]

        # Group constraints by each body that touches them. A constraint
        # appears in 1 or 2 body groups (one per non-(-1) body slot).
        body_groups = [[] for _ in range(N_b)]
        for c in range(N_c):
            if active[c] == 0.0:
                continue
            for slot in (0, 1):
                b = body_idx[c, slot]
                if b >= 0:
                    body_groups[b].append((c, slot))

        # For each body with >1 constraint, build A_BB and measure.
        for b in range(N_b):
            group = body_groups[b]
            if len(group) <= 1:
                continue
            k = len(group)

            m_inv = body_inv_mass[b]
            I_inv = body_inv_inertia[b]

            # Build A_BB[i, j] = sum over SHARED bodies of J_i^T · M⁻¹ · J_j.
            # Restricted to body B's contribution, A_BB_partial[i, j] is
            # exactly J_i_B^T · M_B⁻¹ · J_j_B. The full A_ij also has
            # contributions from other shared bodies, but for measuring
            # body-B-internal coupling the partial is the right quantity.
            A_BB = np.zeros((k, k), dtype=np.float64)
            for i_idx, (ci, si) in enumerate(group):
                Ji = J_values[ci, si]  # (6,)
                Ji_lin, Ji_ang = Ji[:3], Ji[3:]
                # M⁻¹·J = (m_inv*lin, I_inv·ang)
                MinvJi_lin = m_inv * Ji_lin
                MinvJi_ang = I_inv @ Ji_ang
                for j_idx, (cj, sj) in enumerate(group):
                    Jj = J_values[cj, sj]
                    Jj_lin, Jj_ang = Jj[:3], Jj[3:]
                    A_BB[i_idx, j_idx] = (
                        Jj_lin @ MinvJi_lin + Jj_ang @ MinvJi_ang
                    )

            # Add diagonal compliance ONLY on the diagonal entries (C is
            # constraint-diagonal; per body group it shows up only on
            # diagonal of A_BB).
            for i_idx, (ci, _) in enumerate(group):
                A_BB[i_idx, i_idx] += float(C_vals[ci])

            # Off-diag Frobenius norm vs diag 2-norm
            diag = np.diag(A_BB).copy()
            off = A_BB.copy()
            np.fill_diagonal(off, 0.0)

            diag_norm = np.linalg.norm(diag)
            off_norm = np.linalg.norm(off, ord="fro")
            ratio = off_norm / max(diag_norm, 1e-30)

            samples_per_body.append({
                "step": state["step"],
                "iter": state["iter"],
                "body": int(b),
                "k": k,
                "diag_norm": diag_norm,
                "off_norm": off_norm,
                "ratio": ratio,
            })

        state["iter"] += 1

    JacobiPreconditioner.update = patched_update

    # Track step counter
    orig_step = sim.solver.step

    def patched_step(state_in, state_out, control, contacts, dt):
        state["step"] += 1
        state["iter"] = 0
        orig_step(state_in, state_out, control, contacts, dt)

    sim.solver.step = patched_step

    sim.run()

    if not samples_per_body:
        print("No samples collected.")
        return

    print(f"\nCollected {len(samples_per_body)} per-body samples "
          f"(across {state['step'] + 1} steps).")

    ratios = np.array([s["ratio"] for s in samples_per_body])
    ks = np.array([s["k"] for s in samples_per_body])
    print()
    print("Distribution of per-body ||off-diag||_F / ||diag||_2:")
    print(f"  count:   {len(ratios)}")
    print(f"  p25:     {np.percentile(ratios, 25):.3f}")
    print(f"  p50:     {np.percentile(ratios, 50):.3f}")
    print(f"  p75:     {np.percentile(ratios, 75):.3f}")
    print(f"  p95:     {np.percentile(ratios, 95):.3f}")
    print(f"  max:     {ratios.max():.3f}")

    print()
    print("By body-group size:")
    for k_size in sorted(set(ks)):
        sel = ratios[ks == k_size]
        print(f"  k = {k_size:>2d}: n={len(sel):>5d}  "
              f"p50={np.percentile(sel,50):.3f}  "
              f"p95={np.percentile(sel,95):.3f}  "
              f"max={sel.max():.3f}")

    print()
    print("Verdict (rough rule of thumb):")
    p50 = np.percentile(ratios, 50)
    if p50 < 0.1:
        print(f"  p50 ratio = {p50:.3f} < 0.1 → A_BB is mostly diagonal.")
        print("  Per-body block-Jacobi would NOT pay off. Plain Jacobi already")
        print("  captures the structure that matters.")
    elif p50 < 0.5:
        print(f"  p50 ratio = {p50:.3f} in [0.1, 0.5) → modest off-diagonal.")
        print("  Per-body block-Jacobi MIGHT give 10–20% PCR-iter reduction.")
        print("  Worth a 1–2 week prototype if speed matters; may not.")
    else:
        print(f"  p50 ratio = {p50:.3f} ≥ 0.5 → significant off-diagonal coupling.")
        print("  Per-body block-Jacobi SHOULD pay off. Worth implementing.")


if __name__ == "__main__":
    main()
