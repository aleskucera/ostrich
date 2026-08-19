"""helhest_junior open-loop replay in Ostrich (in-process, same venv).

Presents the same job/result interface as the out-of-process engines
(bridge.run_chrono / run_agx) so sweeps and scoring treat all engines uniformly:
`run_ostrich(jobs) -> results`. Reuses the validated replay simulator from
examples/helhest_junior/replay_real.py, overriding the scene: flat ground plus
the job's generic body list instead of the hard-coded box (the new GT drives
roam tens of meters, where the old box would sit mid-path).

Engine params (job["params"]) default to the tuned baseline of the old box
sweep (experiments/1_sim_to_real_box/results/sweep_ostrich.json): dt 0.05,
mu_front 0.8, mu_rear 1.0, compliance_contact 1e-6.
"""

import pathlib
import sys
import time

import numpy as np
import warp as wp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))  # repo root

import newton  # noqa: E402
from ostrich import (ComplianceConfig, ContactsConfig, LinearSolverConfig,  # noqa: E402
                     LinesearchConfig, LoggingConfig, NewtonRaphsonConfig,
                     OstrichEngineConfig, RenderingConfig, SimulationConfig)

from examples.helhest_junior.replay_real import HelhestJuniorReplaySimulator  # noqa: E402

DEFAULT_PARAMS = {
    "mu_front": 0.8,
    "mu_rear": 1.0,
    "mu_rolling": 0.7,
    "compliance_contact": 1e-6,
    "nr_max_iters": 16,
    "linear_max_iters": 16,
    "k_p": 250.0,
    # Anisotropic wheel friction: None = isotropic. When set, mu_front/mu_rear
    # become the LATERAL coefficients and these the longitudinal ones (the
    # friction axis is the wheel spin axis; see _add_wheel in common.py).
    "mu_long_front": None,
    "mu_long_rear": None,
}


class FlatSceneReplaySimulator(HelhestJuniorReplaySimulator):
    """Replay simulator with flat ground + a generic scene list (no box)."""

    def __init__(self, *args, scene=None, ground_mu=0.8,
                 mu_long_front=None, mu_long_rear=None, **kwargs):
        self._scene = scene or []
        self._ground_mu = ground_mu
        self._mu_long_front = mu_long_front
        self._mu_long_rear = mu_long_rear
        super().__init__(*args, **kwargs)

    def build_model(self) -> newton.Model:
        self.builder.rigid_gap = 0.2
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            mu=self._ground_mu, **self.ground_cfg_kwargs)
        self.builder.add_ground_plane(cfg=ground_cfg)

        # The robot is built FIRST so the chassis stays body 0 — replay() logs
        # body_q[0] as the chassis pose. (Dynamic scene bodies added before the
        # robot once silently shifted the indexing and the "trajectory" was a
        # motionless rock.)
        from examples.helhest_junior.common import create_helhest_junior_model
        create_helhest_junior_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            control_mode=self.control_mode,
            k_p=self.k_p, k_d=self.k_d,
            friction_left_right=self.mu_front,
            friction_rear=self.mu_rear,
            mu_rolling=self.mu_rolling,
            friction_long_left_right=self._mu_long_front,
            friction_long_rear=self._mu_long_rear,
        )

        for body in self._scene:
            he = body["half_extents"]
            cfg = newton.ModelBuilder.ShapeConfig(
                mu=body.get("mu", 0.5), **self.box_cfg_kwargs)
            if body["type"] == "static_box":
                self.builder.add_shape_box(
                    body=-1,
                    xform=wp.transform(wp.vec3(*body["pos"]), wp.quat_identity()),
                    hx=he[0], hy=he[1], hz=he[2], cfg=cfg)
            else:
                b = self.builder.add_body(
                    xform=wp.transform(wp.vec3(*body["pos"]), wp.quat_identity()))
                cfg.density = body.get("density", 400.0)
                self.builder.add_shape_box(body=b, hx=he[0], hy=he[1], hz=he[2],
                                           cfg=cfg)

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds)


def _make_sim(job, params):
    dt = job["dt"]
    engine_cfg = OstrichEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=params["nr_max_iters"],
                               backtrack_min_iter=12, atol=1e-3),
        linear=LinearSolverConfig(max_iters=params["linear_max_iters"],
                                  tol=1e-3, atol=1e-3, regularization=1e-6),
        compliance=ComplianceConfig(joint=6e-8,
                                    contact=params["compliance_contact"],
                                    friction=1e-6),
        linesearch=LinesearchConfig(enabled=False),
        contacts=ContactsConfig(max_per_world=256),
    )
    return FlatSceneReplaySimulator(
        SimulationConfig(duration_seconds=job["duration_s"],
                         target_timestep_seconds=dt,
                         num_worlds=1, use_cuda_graph=False),
        RenderingConfig(vis_type="null", target_fps=int(round(1 / dt)),
                        start_paused=False),
        engine_cfg, LoggingConfig(),
        control_mode="velocity",
        mu_front=params["mu_front"], mu_rear=params["mu_rear"],
        mu_rolling=params["mu_rolling"], k_p=params["k_p"],
        mu_long_front=params["mu_long_front"], mu_long_rear=params["mu_long_rear"],
        scene=job.get("scene"), ground_mu=job.get("ground", {}).get("mu", 0.8),
    )


def run_ostrich(jobs: list[dict], use_graph: bool | None = None) -> list[dict]:
    if use_graph is None:
        use_graph = wp.get_device().is_cuda
    results = []
    for job in jobs:
        params = {**DEFAULT_PARAMS, **job.get("params", {})}
        dt = job["dt"]
        if job.get("ground", {}).get("tilt_deg"):
            raise NotImplementedError("tilt probe not supported for ostrich runner")
        sim = _make_sim(job, params)
        setpoints = np.asarray(job["control"]["lrr"], dtype=np.float32)
        settle_steps = max(60, int(round(job.get("settle_time_s", 1.0) / dt)))
        t0 = time.perf_counter()
        if use_graph:
            pose, _ = sim.replay_graph(setpoints, settle_steps=settle_steps)
        else:
            pose, _ = sim.replay(setpoints, settle_steps=settle_steps)
        wall = time.perf_counter() - t0
        pose = np.asarray(pose, dtype=np.float64)
        finite = np.all(np.isfinite(pose), axis=1) & (np.abs(pose[:, 2]) < 5.0)
        stable = bool(finite.all())
        n_ok = int(finite.argmin()) if not stable else pose.shape[0]
        results.append({
            "id": job["id"],
            "dt": dt,
            "pose": pose[:n_ok].tolist(),
            "wheel_omega": None,
            "wall_clock_s": wall,
            "n_steps": n_ok,
            "threads": 0,  # GPU
            "stable": stable,
            "diverged_at_s": None if stable else n_ok * dt,
            "params_effective": params,
        })
        del sim
    return results
