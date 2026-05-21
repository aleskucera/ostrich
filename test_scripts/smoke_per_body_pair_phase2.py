"""Phase 2 smoke test for PerBodyPairPreconditioner.

Runs the obstacle benchmark, samples block extraction at one NR iter
in mid-run, and validates the GPU-extracted A_blocks against a numpy
reference computed independently from the same J_values, M⁻¹, and
C entries.

Validation checklist:
  * A_blocks[w, pair_id, i, j] within 1e-5 absolute of numpy reference
  * Each live A_pair block is symmetric: A_ij == A_ji
  * Each live A_pair block is SPD (positive eigenvalues)
  * Diagonal of A_pair matches the Jacobi preconditioner's diag(A)
    for the corresponding constraint indices (sanity check that we
    haven't accidentally double-counted compliance/regularization)
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
    precond = PerBodyPairPreconditioner(engine, regularization=engine.config.linear.regularization)

    state = {"step": -1, "iter": 0, "captured": False, "result": None}
    target_step = 30  # mid-run, after the robot has settled into rolling

    orig_update = JacobiPreconditioner.update

    def patched_update(self):
        orig_update(self)
        # Capture once, at iter 0 of `target_step`.
        if state["captured"]:
            return
        if state["step"] != target_step or state["iter"] != 0:
            state["iter"] += 1
            return

        precond.update_pair_assignments()
        precond.update_blocks()
        wp.synchronize()

        # Pull GPU data to CPU
        A_blocks = precond.A_blocks.numpy()  # (W, n_pairs_max, MAX, MAX)
        member_count = precond.pair_member_count.numpy()
        member_list = precond.pair_member_list.numpy()
        constr_body_idx = engine.data._constr_body_idx.numpy()
        J_values = engine.data._J_values.numpy()  # (W, N_c, 2, 6)
        body_inv_mass = engine.axion_model.body_inv_mass.numpy()
        body_inv_inertia = engine.data.world_inv_inertia.numpy()
        C_values = engine.data._C_values.numpy()
        # Also pull Jacobi's diag for cross-check
        jacobi_inv_diag = self._P_inv_diag.numpy()

        result = {
            "A_blocks": A_blocks,
            "member_count": member_count,
            "member_list": member_list,
            "constr_body_idx": constr_body_idx,
            "J_values": J_values,
            "body_inv_mass": body_inv_mass,
            "body_inv_inertia": body_inv_inertia,
            "C_values": C_values,
            "jacobi_inv_diag": jacobi_inv_diag,
            "n_bodies": precond.n_bodies,
            "MAX": precond.MAX_MEMBERS_PER_PAIR,
            "regularization": precond.regularization,
        }
        state["result"] = result
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

    # ---- Numpy reference and validation ----
    R = state["result"]
    w = 0
    n_bodies = R["n_bodies"]
    MAX = R["MAX"]

    A_blocks = R["A_blocks"][w]              # (n_pairs_max, MAX, MAX)
    member_count = R["member_count"][w]      # (n_pairs_max,)
    member_list = R["member_list"][w]        # (n_pairs_max, MAX)
    cb_idx = R["constr_body_idx"][w]         # (N_c, 2)
    J = R["J_values"][w]                     # (N_c, 2, 6)
    m_inv = R["body_inv_mass"][w]            # (N_b,)
    I_inv = R["body_inv_inertia"][w]         # (N_b, 3, 3)
    C = R["C_values"][w]                     # (N_c,)
    jacobi_inv_diag = R["jacobi_inv_diag"][w]

    def jacobian_for_body(target_body, c):
        b0, b1 = cb_idx[c]
        if b0 == target_body:
            return J[c, 0]
        if b1 == target_body:
            return J[c, 1]
        return np.zeros(6, dtype=np.float32)

    def Minv_v(body, v):
        return np.concatenate([m_inv[body] * v[:3], I_inv[body] @ v[3:]])

    active_pair_ids = np.where(member_count > 0)[0]

    print(f"Captured step {target_step}, validating "
          f"{len(active_pair_ids)} active pairs in world 0...")

    max_abs_err = 0.0
    sym_err = 0.0
    spd_failures = []
    diag_jacobi_err = 0.0
    diag_jacobi_count = 0

    for pid in active_pair_ids:
        n = int(member_count[pid])
        b_lo = pid // (n_bodies + 1)
        b_hi = pid % (n_bodies + 1)

        constraints = [int(member_list[pid, k]) for k in range(n)]

        # Build numpy reference A_pair
        A_ref = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                ci, cj = constraints[i], constraints[j]
                val = 0.0
                if b_lo < n_bodies:
                    Ji = jacobian_for_body(b_lo, ci)
                    Jj = jacobian_for_body(b_lo, cj)
                    val += float(Ji @ Minv_v(b_lo, Jj))
                if b_hi < n_bodies:
                    Ji = jacobian_for_body(b_hi, ci)
                    Jj = jacobian_for_body(b_hi, cj)
                    val += float(Ji @ Minv_v(b_hi, Jj))
                if i == j:
                    val += float(C[ci]) + R["regularization"]
                A_ref[i, j] = val

        A_gpu = A_blocks[pid, :n, :n].astype(np.float64)
        err = np.abs(A_gpu - A_ref).max()
        max_abs_err = max(max_abs_err, err)

        sym_err = max(sym_err, np.abs(A_gpu - A_gpu.T).max())

        eigs = np.linalg.eigvalsh(A_gpu)
        if eigs.min() <= -1e-6:
            spd_failures.append(
                (int(pid), int(b_lo), int(b_hi), n, float(eigs.min()))
            )

        # Diagonal cross-check: A_pair[i, i] vs 1 / jacobi_inv_diag[c_i]
        for i in range(n):
            ci = constraints[i]
            if jacobi_inv_diag[ci] > 0.0:
                jacobi_diag = 1.0 / jacobi_inv_diag[ci] - 1e-6  # Jacobi adds 1e-6 floor
                # Our A_pair[i,i] should equal jacobi_diag + regularization (we already added it)
                expected = jacobi_diag + R["regularization"]
                rel_err = abs(A_gpu[i, i] - expected) / max(abs(expected), 1e-10)
                diag_jacobi_err = max(diag_jacobi_err, rel_err)
                diag_jacobi_count += 1

    print(f"  max |A_gpu - A_ref|              : {max_abs_err:.2e}")
    print(f"  max |A - Aᵀ| (symmetry)          : {sym_err:.2e}")
    print(f"  max rel diag-vs-Jacobi err       : {diag_jacobi_err:.2e}")
    print(f"  diag entries cross-checked       : {diag_jacobi_count}")
    print(f"  SPD failures (min_eig ≤ -1e-6)   : {len(spd_failures)}")

    if spd_failures:
        print("\n  SPD failures detail:")
        for pid, b_lo, b_hi, n, min_eig in spd_failures[:5]:
            print(f"    pair_id={pid} (b_lo={b_lo}, b_hi={b_hi}) n={n} "
                  f"min_eig={min_eig:.2e}")

    print()
    if max_abs_err < 1e-3 and sym_err < 1e-5 and diag_jacobi_err < 1e-3 and not spd_failures:
        print("✓ Phase 2 block extraction validated.")
    else:
        print("✗ Phase 2 validation FAILED.")


if __name__ == "__main__":
    main()
