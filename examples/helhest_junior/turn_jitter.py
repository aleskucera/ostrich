"""Watch Helhest Junior jitter while it turns on flat triangulated ground.

No keyboard: the robot drives itself through a fixed loop -- settle, spin left,
spin right, forward, back -- so you can just leave it running and watch. The
turns are the point; the straight runs are there as a contrast, because the same
robot on the same ground is visibly smoother going forward than rotating.

A three-wheel skid-steer cannot rotate without sliding its wheels sideways, so a
turn drives every contact into the sliding regime of the friction solve. When
this example was written that produced violent stick-slip (spin yaw_rate -0.66,
rate_std 0.67, jerk_rms ~23 at dt=0.03) whose cause was then unknown. It is now
explained and fixed -- see docs/jitter_investigation_findings.md: the friction
row never bounded |f_t| by mu*f_n at sliding (wheels welded to the ground; the
turn only advanced when a wheel unloaded), amplified by cluster contact
reduction reshuffling the flat-mesh support points. With the cone fix,
nr.w_relaxation 0.5 and the fps reducer (all defaults now), the same loop
measures spin yaw_rate ~ -2.15 (75% of the no-slip rate) with rate_std/|mean|
~ 0.10 and near-zero chassis roll.

Ground is a triangulated flat MESH rather than newton's analytic plane because
the two are not the same surface to the solver: a plane has one contact normal
everywhere, a mesh hands the wheel a new pair of triangles every few
centimetres. The remaining smoothness gap on this ground is the cylinder wheel's
rim-edge contacts against triangles (see the trade-off note in common.py).

    python examples/helhest_junior/turn_jitter.py
    python examples/helhest_junior/turn_jitter.py engine.compliance.contact=1e-10
    python examples/helhest_junior/turn_jitter.py ground.cell=0.25 control.k_p=150

The viewer plots yaw rate and yaw acceleration live (the "Plots" panel), and each
phase prints its own numbers when it ends, so the effect of an override is
visible without re-reading the window:

  rate_std  std of yaw rate; a smooth rotation holds it near zero
  jerk_rms  rms yaw ACCELERATION -- the stick-slip signature
"""

import os
import pathlib
from typing import override

import hydra
import newton
import numpy as np
import warp as wp
from omegaconf import DictConfig

from ostrich import EngineConfig
from ostrich import LoggingConfig
from ostrich import RenderingConfig
from ostrich import SimulationConfig

try:
    from examples.helhest_junior.common import create_helhest_junior_model
    from examples.helhest_junior.common import HelhestJuniorConfig
    from examples.helhest_junior.control import HelhestJuniorControlSimulator
except ImportError:
    from common import create_helhest_junior_model
    from common import HelhestJuniorConfig
    from control import HelhestJuniorControlSimulator

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")


def flat_mesh(extent: float, cell: float) -> newton.Mesh:
    """A flat square patch as a triangle mesh, one vertex per cell centre.

    Deliberately not `add_ground_plane`: the whole point is the triangulation.
    Winding is CCW seen from +z, so face normals point up.
    """
    n = int(round(extent / cell))
    xs = -0.5 * extent + (np.arange(n) + 0.5) * cell
    XX, YY = np.meshgrid(xs, xs)
    vertices = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)]).astype(np.float32)

    ii, jj = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    v00, v01 = ii * n + jj, ii * n + (jj + 1)
    v10, v11 = (ii + 1) * n + jj, (ii + 1) * n + (jj + 1)
    faces = np.column_stack([v00, v01, v11, v00, v11, v10]).reshape(-1, 3).astype(np.int32)
    return newton.Mesh(vertices, faces.flatten())


class TurnJitterSimulator(HelhestJuniorControlSimulator):
    """control.py's robot on flat mesh ground, driving itself in a loop."""

    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
        extent: float = 20.0,
        cell: float = 0.06,
        terrain_mu: float = 0.8,
        friction_lat_left_right: float = 0.5,
        friction_lat_rear: float = 0.2,
        friction_long_left_right: float = 0.9,
        friction_long_rear: float = 0.6,
        turn_speed: float = 3.0,
        drive_speed: float = 5.0,
        **kwargs,
    ):
        self.extent, self.cell, self.terrain_mu = extent, cell, terrain_mu
        self.friction_lat_left_right = friction_lat_left_right
        self.friction_lat_rear = friction_lat_rear
        self.friction_long_left_right = friction_long_left_right
        self.friction_long_rear = friction_long_rear

        t, d = turn_speed, drive_speed
        # (label, seconds, [left, right, rear] wheel rad/s), looped forever.
        self.script = (
            ("settle", 1.5, (0.0, 0.0, 0.0)),
            ("SPIN LEFT", 5.0, (t, -t, 0.0)),
            ("stop", 1.0, (0.0, 0.0, 0.0)),
            ("SPIN RIGHT", 5.0, (-t, t, 0.0)),
            ("stop", 1.0, (0.0, 0.0, 0.0)),
            ("forward", 3.0, (d, d, d)),
            ("stop", 1.0, (0.0, 0.0, 0.0)),
            ("back", 3.0, (-d, -d, -d)),
            ("stop", 1.0, (0.0, 0.0, 0.0)),
        )
        self.loop_seconds = sum(p[1] for p in self.script)
        self._phase = None
        self._yaw_prev = None
        self._rate_prev = None
        self._rates: list[float] = []
        self._jerks: list[float] = []
        super().__init__(sim_config, render_config, engine_config, logging_config, **kwargs)

    @override
    def build_model(self) -> newton.Model:
        self.builder.rigid_gap = 0.01
        self.builder.add_shape_mesh(
            body=-1,
            mesh=flat_mesh(self.extent, self.cell),
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0, has_shape_collision=True, mu=self.terrain_mu
            ),
        )
        create_helhest_junior_model(
            self.builder,
            xform=wp.transform(
                wp.vec3(0.0, 0.0, HelhestJuniorConfig.WHEEL_RADIUS + 0.02), wp.quat_identity()
            ),
            control_mode=self.control_mode,
            k_p=self.k_p,
            k_d=self.k_d,
            friction_left_right=self.friction_lat_left_right,
            friction_rear=self.friction_lat_rear,
            friction_long_left_right=self.friction_long_left_right,
            friction_long_rear=self.friction_long_rear,
        )
        n = int(round(self.extent / self.cell))
        print(
            f"flat MESH ground: {self.extent} m at {self.cell} m cells "
            f"({n}x{n} vertices, {2 * (n - 1) ** 2} triangles)\n"
            f"loop: {' -> '.join(p[0] for p in self.script)}  ({self.loop_seconds:.1f} s)"
        )
        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)

    def _current_phase(self):
        t = self.clock.time % self.loop_seconds
        for label, seconds, cmd in self.script:
            if t < seconds:
                return label, cmd
            t -= seconds
        return self.script[-1][0], self.script[-1][2]

    @override
    def _update_input(self):
        """Scripted, not keyboard. Written straight through with no rate limit.

        The abrupt command step is deliberate: releasing the wheels into a turn
        is exactly when stick-slip is easiest to see.
        """
        label, cmd = self._current_phase()
        if label != self._phase:
            self._report(self._phase)
            self._phase = label
            self._rates.clear()
            self._jerks.clear()
        self._cmd = np.asarray(cmd, dtype=np.float32)
        wp.copy(self.target_velocities, wp.array(self._cmd, device=self.model.device))

    def _report(self, label):
        if label is None or len(self._rates) < 5:
            return
        r, j = np.asarray(self._rates), np.asarray(self._jerks)
        print(
            f"  {label:11} yaw_rate {r.mean():7.3f} rad/s   "
            f"rate_std {r.std():6.3f}   jerk_rms {np.sqrt((j**2).mean()):7.2f} rad/s^2"
        )

    @override
    def _run_simulation_segment(self, segment_num: int):
        super()._run_simulation_segment(segment_num)
        self._measure()

    def _measure(self):
        """Yaw rate and acceleration from the pose physics just produced.

        Read here rather than after rendering: once the display frames start
        blending, ``current_state.body_q`` points at the interpolated buffer.
        """
        q = self.current_state.body_q.numpy()[0]
        x, y, z, w = (float(v) for v in q[3:7])
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        dt = self.steps_per_segment * self.clock.dt

        if self._yaw_prev is not None:
            rate = np.arctan2(np.sin(yaw - self._yaw_prev), np.cos(yaw - self._yaw_prev)) / dt
            self.viewer.log_scalar("yaw_rate (rad/s)", rate)
            self._rates.append(rate)
            if self._rate_prev is not None:
                jerk = (rate - self._rate_prev) / dt
                self.viewer.log_scalar("yaw_accel (rad/s^2)", jerk)
                self._jerks.append(jerk)
            self._rate_prev = rate
        self._yaw_prev = yaw


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest_jitter", version_base=None)
def turn_jitter_example(cfg: DictConfig):
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    render_config.vis_type = "gl"
    render_config.start_paused = False  # watch-only demo: never wait for a keypress

    TurnJitterSimulator(
        hydra.utils.instantiate(cfg.simulation),
        render_config,
        hydra.utils.instantiate(cfg.engine),
        hydra.utils.instantiate(cfg.logging),
        extent=cfg.ground.extent,
        cell=cfg.ground.cell,
        terrain_mu=cfg.terrain_mu,
        friction_lat_left_right=cfg.friction.lateral_left_right,
        friction_lat_rear=cfg.friction.lateral_rear,
        friction_long_left_right=cfg.friction.long_left_right,
        friction_long_rear=cfg.friction.long_rear,
        turn_speed=cfg.turn_speed,
        drive_speed=cfg.drive_speed,
        control_mode=cfg.control.mode,
        k_p=cfg.control.k_p,
        k_d=cfg.control.k_d,
    ).run()


if __name__ == "__main__":
    turn_jitter_example()
