"""Control stability benchmark — Axion (implicit servo).

Scene: single pendulum, pivot fixed at (0,0,2), 1m link, 1 kg.
       Displaced to θ₀ = π/3 rad at t=0. Target: θ = 0 (hanging at rest).

Uses Axion's implicit servo constraint (JointMode.TARGET_POSITION, ke=kp, kd=kd).

Three experiments:
  dt_sweep      — fix kp=500, kd=50; sweep Δt ∈ {0.001, 0.005, 0.01, 0.05, 0.1}
  gain_sweep    — fix Δt=0.05; sweep kp ∈ {10, 50, 100, 200, 500, 1000, 5000}
                  kd is set to 2*sqrt(kp * I) (critical damping heuristic)
  binary_search — fix kp=500, kd=50; binary-search for the largest stable Δt

Usage:
    python examples/comparison/control_stability/axion.py \
        --experiment dt_sweep \
        --save examples/comparison/control_stability/results/axion_implicit_dt.json

    python examples/comparison/control_stability/axion.py \
        --experiment binary_search \
        --save examples/comparison/control_stability/results/axion_threshold.json
"""
import argparse
import json
import math
import os
import pathlib

from config import (DURATION, LINK_MASS, Q_INIT, Q_TARGET, STABILITY_TOL,
                    DT_SWEEP_KP, DT_SWEEP_KD, DT_VALUES, GAIN_SWEEP_DT, KP_VALUES,
                    BSEARCH_KP, BSEARCH_KD, BSEARCH_MAX, BSEARCH_TOL, BSEARCH_DIVERGE)
import newton
import numpy as np
import warp as wp
from axion import AxionEngineConfig
from axion import InteractiveSimulator
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion.core.types import JointMode
from axion.simulation.sim_config import SyncMode
from axion import ComplianceConfig
from axion import ContactsConfig
from axion import LinearSolverConfig
from axion import LinesearchConfig
from axion import NewtonRaphsonConfig
from newton import Model

os.environ["PYOPENGL_PLATFORM"] = "glx"

# --- Scene constants (axion-specific geometry) ---
LINK_HX = 0.5  # half-length (full link = 1 m)
LINK_HY = 0.05
LINK_HZ = 0.05
LINK_DENSITY = LINK_MASS / (2 * LINK_HX * 2 * LINK_HY * 2 * LINK_HZ)
PIVOT_POS = wp.vec3(0.0, 0.0, 2.0)


def kd_from_kp(kp: float) -> float:
    """Critical-damping heuristic: kd = 2*sqrt(kp * I), I ≈ m*L²/3."""
    I = LINK_MASS * (2 * LINK_HX) ** 2 / 3.0
    return 2.0 * math.sqrt(kp * I)


# ---------------------------------------------------------------------------
# Simulator class
# ---------------------------------------------------------------------------
class PendulumSim(InteractiveSimulator):
    """Single-pendulum stability simulator using Axion implicit servo."""

    def __init__(
        self,
        sim_config,
        render_config,
        engine_config,
        logging_config,
        kp: float,
        kd: float,
    ):
        self.kp = kp
        self.kd = kd
        super().__init__(sim_config, render_config, engine_config, logging_config)
        self._set_initial_displacement()

    def build_model(self) -> Model:
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=LINK_DENSITY)
        link = self.builder.add_link()
        self.builder.add_shape_box(link, hx=LINK_HX, hy=LINK_HY, hz=LINK_HZ, cfg=shape_cfg)

        pivot_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi * 0.5)
        j = self.builder.add_joint_revolute(
            parent=-1,
            child=link,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform(PIVOT_POS, pivot_rot),
            child_xform=wp.transform(wp.vec3(-LINK_HX, 0.0, 0.0), wp.quat_identity()),
        )
        self.builder.add_articulation([j])
        model = self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=False,
        )
        # Set joint mode and gains on the model BEFORE the engine is created
        dev = model.device
        wp.copy(
            model.joint_dof_mode,
            wp.array(
                np.array([int(JointMode.TARGET_POSITION)], dtype=np.int32),
                dtype=wp.int32,
                device=dev,
            ),
        )
        wp.copy(
            model.joint_target_ke,
            wp.array(np.array([self.kp], dtype=np.float32), dtype=wp.float32, device=dev),
        )
        wp.copy(
            model.joint_target_kd,
            wp.array(np.array([self.kd], dtype=np.float32), dtype=wp.float32, device=dev),
        )
        return model

    def _set_initial_displacement(self):
        """Set initial joint position and recompute forward kinematics."""
        q = self.current_state.joint_q.numpy()
        q[0] = Q_INIT  # start at rest (0 = hanging)
        self.current_state.joint_q.assign(wp.array(q, dtype=wp.float32, device=self.model.device))
        newton.eval_fk(
            self.model,
            self.current_state.joint_q,
            self.current_state.joint_qd,
            self.current_state,
        )

    def control_policy(self, state):
        wp.copy(
            self.control.joint_target_pos,
            wp.array(
                np.array([Q_TARGET], dtype=np.float32), dtype=wp.float32, device=self.model.device
            ),
        )

    def run_headless(self) -> dict:
        T = self.clock.total_sim_steps
        times = []
        angles = []
        stable = True

        for step in range(T):
            self._single_physics_step(step)
            wp.synchronize()

            newton.eval_ik(
                self.model,
                self.current_state,
                self.current_state.joint_q,
                self.current_state.joint_qd,
            )
            q = float(self.current_state.joint_q.numpy()[0])
            t = float((step + 1) * self.clock.dt)

            if not math.isfinite(q):
                stable = False
                times.append(t)
                angles.append(None)
                break

            if abs(q) > STABILITY_TOL:
                stable = False  # mark but keep recording to show divergence

            times.append(t)
            angles.append(q)

        return {
            "dt": self.clock.dt,
            "T": T,
            "kp": self.kp,
            "kd": self.kd,
            "time": times,
            "joint_angle": angles,
            "stable": stable,
        }


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------
def _make_engine_config(kp: float, kd: float) -> AxionEngineConfig:
    return AxionEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=16, backtrack_min_iter=8, atol=0.0001),
        linear=LinearSolverConfig(max_iters=16, atol=0.0001, tol=0.0001, regularization=1e-06),
        compliance=ComplianceConfig(joint=1e-06, contact=1e-06, friction=1e-06),
        linesearch=LinesearchConfig(enabled=False, conservative_step_count=16, conservative_upper_bound=0.05, min_step=1e-06, optimistic_step_count=48, optimistic_window=0.4),
        contacts=ContactsConfig(max_per_world=64),
    )


def run_one(dt: float, kp: float, kd: float) -> dict:
    T = max(1, int(DURATION / dt))
    sim_config = SimulationConfig(
        duration_seconds=DURATION,
        target_timestep_seconds=dt,
        num_worlds=1,
        sync_mode=SyncMode.ALIGN_FPS_TO_DT,
        use_cuda_graph=False,
    )
    render_config = RenderingConfig(
        vis_type="null",
        target_fps=30,
        usd_file=None,
        world_offset_x=5.0,
        world_offset_y=5.0,
        start_paused=False,
    )
    logging_config = LoggingConfig()

    sim = PendulumSim(
        sim_config,
        render_config,
        _make_engine_config(kp, kd),
        logging_config,
        kp=kp,
        kd=kd,
    )
    return sim.run_headless()


# ---------------------------------------------------------------------------
# Binary search helpers
# ---------------------------------------------------------------------------
def _is_stable_threshold(dt: float, kp: float, kd: float) -> bool:
    """Stability check for binary search: unstable if |q| ever exceeds π."""
    run = run_one(dt, kp, kd)
    angles = [a for a in run["joint_angle"] if a is not None]
    if not angles:
        return False
    return max(abs(a) for a in angles) < BSEARCH_DIVERGE


def find_threshold(kp: float, kd: float) -> dict:
    """Binary-search for the largest stable timestep."""
    lo = 0.001
    hi = None

    probe = lo * 2.0
    n_evals = 0
    while probe <= BSEARCH_MAX:
        n_evals += 1
        stable = _is_stable_threshold(probe, kp, kd)
        print(f"  probe dt={probe:.4f}s → {'STABLE' if stable else 'UNSTABLE'}")
        if not stable:
            hi = probe
            break
        lo = probe
        probe *= 2.0

    if hi is None:
        return {"max_stable_dt": lo, "n_evals": n_evals, "hit_max": True}

    while hi - lo > BSEARCH_TOL:
        mid = (lo + hi) / 2.0
        n_evals += 1
        stable = _is_stable_threshold(mid, kp, kd)
        print(f"  bisect dt={mid:.4f}s → {'STABLE' if stable else 'UNSTABLE'}")
        if stable:
            lo = mid
        else:
            hi = mid

    return {"max_stable_dt": lo, "n_evals": n_evals, "hit_max": False}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", choices=["dt_sweep", "gain_sweep", "binary_search"], default="binary_search"
    )
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON")
    args = parser.parse_args()

    results = {
        "simulator": "Axion-Implicit",
        "experiment": args.experiment,
        "runs": [],
    }

    if args.experiment == "dt_sweep":
        kp, kd = DT_SWEEP_KP, DT_SWEEP_KD
        print(f"Axion-Implicit — dt_sweep (kp={kp}, kd={kd}):")
        for dt in DT_VALUES:
            print(f"  dt={dt:.3f}s ...", end=" ", flush=True)
            run = run_one(dt, kp, kd)
            results["runs"].append(run)
            print(
                "STABLE"
                if run["stable"]
                else f"UNSTABLE (max|θ|={max((abs(a) for a in run['joint_angle'] if a is not None), default=0):.2f} rad)"
            )

    elif args.experiment == "gain_sweep":
        dt = GAIN_SWEEP_DT
        print(f"Axion-Implicit — gain_sweep (dt={dt}s):")
        for kp in KP_VALUES:
            kd = kd_from_kp(kp)
            print(f"  kp={kp:5d}  kd={kd:.1f} ...", end=" ", flush=True)
            run = run_one(dt, kp, kd)
            results["runs"].append(run)
            max_angle = max((abs(a) for a in run["joint_angle"] if a is not None), default=0)
            print("STABLE" if run["stable"] else f"UNSTABLE (max|θ|={max_angle:.2f} rad)")

    else:  # binary_search
        kp, kd = BSEARCH_KP, BSEARCH_KD
        print(f"Axion-Implicit — binary_search (kp={kp}, kd={kd}):")
        threshold = find_threshold(kp, kd)
        results["max_stable_dt"] = threshold["max_stable_dt"]
        results["n_evals"] = threshold["n_evals"]
        results["hit_max"] = threshold["hit_max"]
        results["kp"] = kp
        results["kd"] = kd
        suffix = "hit BSEARCH_MAX" if threshold["hit_max"] else f"{threshold['n_evals']} evals"
        print(f"  => max_stable_dt = {threshold['max_stable_dt']:.4f}s ({suffix})")

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
