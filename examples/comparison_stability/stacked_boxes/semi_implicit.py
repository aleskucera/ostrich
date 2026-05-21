"""Stacked-boxes stability benchmark — Newton Semi-Implicit Euler solver.

Scene: 3-box inverted pyramid matching examples/box_stack.py.
  Box 1 (bottom): hx=0.2 m, density=1500 kg/m³  →   96 kg
  Box 2 (middle): hx=0.8 m, density=1500 kg/m³  → 6144 kg
  Box 3 (top):    hx=1.6 m, density=1500 kg/m³  → 49152 kg
  Mass ratio top/bottom ≈ 512:1.

A simulation is deemed STABLE if it completes without NaN and the top-box
height stays within STABILITY_TOL of its initial value throughout.

Usage:
    python examples/comparison_stability/stacked_boxes/semi_implicit.py \
        --save examples/comparison_stability/stacked_boxes/results/semi_implicit.json
"""
import argparse
import json
import os
import pathlib

from config import DURATION, DENSITY, HX1, HX2, HX3, Z1, Z2, Z3, STABILITY_TOL, KE, KD, KF, MU, BSEARCH_TOL, BSEARCH_MAX
import numpy as np
import warp as wp
from axion import InteractiveSimulator
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion import SemiImplicitEngineConfig
from axion.simulation.sim_config import SyncMode
from newton import Model
from newton import ModelBuilder as NewtonModelBuilder

os.environ["PYOPENGL_PLATFORM"] = "glx"


def find_threshold(engine_config) -> dict:
    lo = 0.001
    hi = None
    probe = lo * 2.0
    n_evals = 0
    while probe <= BSEARCH_MAX:
        n_evals += 1
        print(f"  probe dt={probe:.4f}s ...", end=" ", flush=True)
        r = run_one(probe, engine_config)
        if r["stable"]:
            print("stable")
            lo = probe
            probe *= 2.0
        else:
            print("UNSTABLE")
            hi = probe
            break

    if hi is None:
        print(f"  Stable up to {lo:.4f}s (BSEARCH_MAX reached)")
        return {"max_stable_dt": lo, "n_evals": n_evals, "hit_max": True}

    while hi - lo > BSEARCH_TOL:
        mid = (lo + hi) / 2.0
        n_evals += 1
        print(f"  bisect dt={mid:.4f}s ...", end=" ", flush=True)
        r = run_one(mid, engine_config)
        if r["stable"]:
            print("stable")
            lo = mid
        else:
            print("UNSTABLE")
            hi = mid

    print(f"  → max stable dt = {lo:.4f}s  (n_evals={n_evals})")
    return {"max_stable_dt": lo, "n_evals": n_evals, "hit_max": False}


class StackedBoxesSim(InteractiveSimulator):
    def build_model(self) -> Model:
        contact_cfg = dict(ke=KE, kd=KD, kf=KF, mu=MU)
        self.builder.rigid_gap = 1.0

        b0 = self.builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, Z1), wp.quat_identity()),
        )
        self.builder.add_shape_box(
            body=b0,
            hx=HX1,
            hy=HX1,
            hz=HX1,
            cfg=NewtonModelBuilder.ShapeConfig(density=DENSITY, **contact_cfg),
        )

        b1 = self.builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, Z2), wp.quat_identity()),
        )
        self.builder.add_shape_box(
            body=b1,
            hx=HX2,
            hy=HX2,
            hz=HX2,
            cfg=NewtonModelBuilder.ShapeConfig(density=DENSITY, **contact_cfg),
        )

        b2 = self.builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, Z3), wp.quat_identity()),
        )
        self.builder.add_shape_box(
            body=b2,
            hx=HX3,
            hy=HX3,
            hz=HX3,
            cfg=NewtonModelBuilder.ShapeConfig(density=DENSITY, **contact_cfg),
        )

        self.builder.add_ground_plane(
            cfg=NewtonModelBuilder.ShapeConfig(density=0.0, **contact_cfg)
        )
        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=False,
        )

    def run_headless(self) -> dict:
        T = self.clock.total_sim_steps
        times = []
        heights = []
        stable = True

        for step in range(T):
            self._single_physics_step(step)
            wp.synchronize()

            bq = self.current_state.body_q.numpy()
            h = float(bq[2, 2])
            t = float((step + 1) * self.clock.dt)

            if not np.isfinite(h):
                stable = False
                times.append(t)
                heights.append(None)
                break

            if abs(h - Z3) > STABILITY_TOL:
                stable = False

            times.append(t)
            heights.append(h)

        return {
            "dt": self.clock.dt,
            "T": T,
            "time": times,
            "top_box_height": heights,
            "stable": stable,
        }


def run_one(dt: float, engine_config: SemiImplicitEngineConfig) -> dict:
    T = max(1, int(DURATION / dt))
    sim_config = SimulationConfig(
        duration_seconds=DURATION,
        target_timestep_seconds=dt,
        num_worlds=1,
        sync_mode=SyncMode.ALIGN_FPS_TO_DT,
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
    sim = StackedBoxesSim(sim_config, render_config, engine_config, logging_config)
    return sim.run_headless()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON")
    args = parser.parse_args()

    engine_config = SemiImplicitEngineConfig(
        angular_damping=0.0,
    )

    base = {
        "simulator": "Semi-Implicit",
        "problem": "stacked_boxes",
        "density": DENSITY,
        "hx1": HX1,
        "hx2": HX2,
        "hx3": HX3,
        "z_top_initial": Z3,
        "stability_tol": STABILITY_TOL,
    }

    print("Semi-Implicit — binary_search:")
    threshold = find_threshold(engine_config)
    results = {**base, **threshold}

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
