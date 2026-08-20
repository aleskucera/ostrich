"""Drive Helhest interactively and see what its lidar sees.

The obstacle course + IJKL keyboard drive of control.py, plus a simulated 3D sensor rigidly
mounted on the chassis. Two sensor models:

  +sensor=ouster (default)  front-facing OSDome: front hemisphere of polar ray rings, blind
                            behind; datasheet range noise (1-sigma = 0.01 + 2.25e-4 r^2, cap
                            0.10 m) + 3% dropout.
  +sensor=odin              Manifold Odin-1 dTOF depth camera, reproduced from the REAL
                            robot's recordings: the 256x192 per-pixel ray table measured from
                            bags/in_speed_odin0 (~155 x 93 deg FOV, fixed rays, the sensor's
                            dead edge pixels stay dead), 0.2-30 m window, +-3 cm @ 1-sigma.

Every rendered frame the sensor ray-casts the whole scene -- including the robot's OWN wheels
and chassis, which return exactly like they do on the real sensor (the perception stack's
self-filter box exists to cut them); only hits inside the sensor's minimum range are dropped.
The cloud is drawn in the same window, colored by range (orange = near -> blue = far).

Runs REAL TIME by default (each frame sleeps out its sim-time budget; the stock loop instead
renders at monitor refresh, i.e. 4.3x reality at 144 Hz). `+realtime=false` unlocks it.

Controls (same as control.py): I/K forward/back, J/L turn, SPACE pause, mouse orbits, ESC quits.

    uv run python examples/helhest/lidar_teleop.py
    uv run python examples/helhest/lidar_teleop.py +sensor=odin friction_coeff=0.3
    uv run python examples/helhest/lidar_teleop.py +realtime=false

Headless self-test (no window; drives a script and prints per-scan hit stats):

    uv run python examples/helhest/lidar_teleop.py rendering=headless +selftest_steps=60 +sensor=odin
"""

import pathlib
from typing import override

import hydra
import newton
import numpy as np
import warp as wp
from newton._src.geometry.raycast import ray_intersect_geom
from newton._src.geometry.types import GeoType
from newton._src.utils.heightfield import HeightfieldData
from omegaconf import DictConfig
from ostrich import EngineConfig
from ostrich import LoggingConfig
from ostrich import RenderingConfig
from ostrich import SimulationConfig

try:
    from examples.helhest.control import HelhestControlSimulator
except ImportError:
    from control import HelhestControlSimulator

CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")

# --- OSDome, mounted axis-FORWARD: rays cover the front hemisphere (polar angle from the
# +x body axis), so the robot is blind behind -- matching the real front-facing mount.
LIDAR_CHANNELS = 48  # polar rings from near-axis to the dome rim
LIDAR_COLS = 720  # around the dome axis
# Sensor parameter sets: (min_range, max_range, noise base/quad/cap [1-sigma(r) = base +
# quad*r^2, capped], dropout). Ouster values follow helhest_stack's perception.sim.ouster
# datasheet model; Odin-1 is +-3 cm @ 1-sigma (constant) over a 0.2-30 m window, and its
# per-frame invalid fraction beyond the dead pixels measured ~0.6% on the real bags.
SENSORS = {
    "ouster": dict(min_range=0.3, max_range=25.0, base=0.01, quad=2.25e-4, cap=0.10, dropout=0.03),
    "odin": dict(min_range=0.2, max_range=30.0, base=0.03, quad=0.0, cap=0.03, dropout=0.006),
}
LIDAR_MOUNT = wp.vec3(0.25, 0.0, 0.25)  # chassis-local, front of the body, above the deck
ODIN_DIRS_FILE = pathlib.Path(__file__).parent.joinpath("odin_dirs.npy")
_MISS = 1.0e10


@wp.kernel
def _lidar_scan_kernel(
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_transform: wp.array(dtype=wp.transform),
    geom_type: wp.array(dtype=int),
    geom_size: wp.array(dtype=wp.vec3),
    shape_source_ptr: wp.array(dtype=wp.uint64),
    shape_heightfield_index: wp.array(dtype=wp.int32),
    heightfield_data: wp.array(dtype=HeightfieldData),
    heightfield_elevations: wp.array(dtype=wp.float32),
    chassis: int,
    mount: wp.vec3,
    dirs_local: wp.array(dtype=wp.vec3),
    min_range: float,
    max_range: float,
    dists: wp.array(dtype=float),
):
    """Closest hit per ray over ALL shapes -- the robot's own wheels/chassis return like on the
    real sensor; only hits inside the minimum range (the mount's immediate surroundings) drop."""
    ray, shape_idx = wp.tid()
    X_wc = body_q[chassis]
    origin = wp.transform_point(X_wc, mount)
    direction = wp.transform_vector(X_wc, dirs_local[ray])
    b = shape_body[shape_idx]
    X_wb = wp.transform_identity()
    if b >= 0:
        X_wb = body_q[b]
    geom_to_world = wp.mul(X_wb, shape_transform[shape_idx])
    geomtype = geom_type[shape_idx]
    mesh_id = wp.uint64(0)
    if geomtype == int(GeoType.MESH) or geomtype == int(GeoType.CONVEX_MESH):
        mesh_id = shape_source_ptr[shape_idx]
    t, _normal = ray_intersect_geom(
        geom_to_world,
        geom_size[shape_idx],
        geomtype,
        origin,
        direction,
        mesh_id,
        shape_idx,
        shape_heightfield_index,
        heightfield_data,
        heightfield_elevations,
    )
    if t >= min_range and t <= max_range:
        wp.atomic_min(dists, ray, t)


@wp.kernel
def _lidar_points_kernel(
    body_q: wp.array(dtype=wp.transform),
    chassis: int,
    mount: wp.vec3,
    dirs_local: wp.array(dtype=wp.vec3),
    dists: wp.array(dtype=float),
    max_range: float,
    noise_base: float,
    noise_quad: float,
    noise_cap: float,
    dropout: float,
    seed: int,
    points: wp.array(dtype=wp.vec3),
    colors: wp.array(dtype=wp.vec3),
):
    """Hit distances -> world points + range-colored dots (orange near -> blue far), with the
    sensor's range noise (1-sigma = base + quad*r^2, capped) and dropout applied to the
    CLOSEST hit -- the sensor perturbs its measurement, not the geometry test."""
    ray = wp.tid()
    X_wc = body_q[chassis]
    origin = wp.transform_point(X_wc, mount)
    direction = wp.transform_vector(X_wc, dirs_local[ray])
    t = dists[ray]
    rng = wp.rand_init(seed, ray)
    if t < _MISS and wp.randf(rng) >= dropout:
        sigma = wp.min(noise_base + noise_quad * t * t, noise_cap)
        t += sigma * wp.randn(rng)
        frac = t / max_range
        points[ray] = origin + t * direction
        colors[ray] = (1.0 - frac) * wp.vec3(1.0, 0.45, 0.1) + frac * wp.vec3(0.15, 0.45, 1.0)
    else:
        points[ray] = wp.vec3(0.0, 0.0, -1.0e6)  # park misses far below the world
        colors[ray] = wp.vec3(0.0, 0.0, 0.0)


def _dome_directions() -> np.ndarray:
    """Chassis-local unit ray directions [channels * cols, 3]: the front hemisphere, as polar
    rings about the +x (forward) dome axis -- theta ~0 looks straight ahead, theta = 90 deg is
    the dome rim (straight up / down / sideways). Nothing points backward."""
    theta = np.radians(np.linspace(2.0, 90.0, LIDAR_CHANNELS))  # polar angle from +x
    psi = np.radians(np.linspace(0.0, 360.0, LIDAR_COLS, endpoint=False))  # around +x
    th, ps = np.meshgrid(theta, psi, indexing="ij")
    dirs = np.stack(
        [np.cos(th), np.sin(th) * np.cos(ps), np.sin(th) * np.sin(ps)], axis=-1
    ).reshape(-1, 3)
    return np.ascontiguousarray(dirs, np.float32)


def _odin_directions() -> np.ndarray:
    """The Odin-1's per-pixel ray table MEASURED from the real robot's bags (mean unit
    direction per pixel over 120 frames of bags/in_speed_odin0 -- the directions are fixed to
    |mean| = 1.0000, i.e. a true depth camera). Base-frame, so the real mount orientation is
    baked in; the sensor's permanently dead edge pixels (never a return in the recording) are
    dropped, exactly like the real device."""
    table = np.load(ODIN_DIRS_FILE).reshape(-1, 3)
    live = np.linalg.norm(table, axis=1) > 0.5
    return np.ascontiguousarray(table[live], np.float32)


class HelhestLidarTeleop(HelhestControlSimulator):
    def __init__(self, *args, sensor: str = "ouster", realtime: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.simulation_config.num_worlds == 1, "single-world demo"
        self._chassis = 0  # first body added by create_helhest_model
        self._sensor = SENSORS[sensor]
        # Real-time pacing now lives in InteractiveSimulator._pace_real_time
        self.rendering_config.real_time = realtime
        dev = self.model.device
        dirs = _odin_directions() if sensor == "odin" else _dome_directions()
        self._n_rays = len(dirs)
        self._dirs = wp.array(dirs, dtype=wp.vec3, device=dev)
        self._dists = wp.zeros(self._n_rays, dtype=wp.float32, device=dev)
        self._points = wp.zeros(self._n_rays, dtype=wp.vec3, device=dev)
        self._colors = wp.zeros(self._n_rays, dtype=wp.vec3, device=dev)
        # ViewerGL's point kernel requires per-point radii (the scalar overload doesn't reach it)
        self._radii = wp.full(self._n_rays, 0.025, dtype=wp.float32, device=dev)
        self._scan_seed = 0  # bumped per scan so the noise decorrelates frame to frame
        self.script_vels: np.ndarray | None = None  # selftest hook (bypasses the keyboard)

    @override
    def _update_input(self):
        if self.script_vels is not None:
            wp.copy(
                self.target_velocities,
                wp.array(self.script_vels.astype(np.float32), device=self.model.device),
            )
            return
        super()._update_input()

    def _scan(self):
        self._dists.fill_(_MISS)
        m = self.model
        wp.launch(
            _lidar_scan_kernel,
            dim=(self._n_rays, m.shape_count),
            inputs=[
                self.current_state.body_q,
                m.shape_body,
                m.shape_transform,
                m.shape_type,
                m.shape_scale,
                m.shape_source_ptr,
                m.shape_heightfield_index,
                m.heightfield_data,
                m.heightfield_elevations,
                self._chassis,
                LIDAR_MOUNT,
                self._dirs,
                self._sensor["min_range"],
                self._sensor["max_range"],
            ],
            outputs=[self._dists],
            device=m.device,
        )
        self._scan_seed += 1
        wp.launch(
            _lidar_points_kernel,
            dim=self._n_rays,
            inputs=[
                self.current_state.body_q,
                self._chassis,
                LIDAR_MOUNT,
                self._dirs,
                self._dists,
                self._sensor["max_range"],
                self._sensor["base"],
                self._sensor["quad"],
                self._sensor["cap"],
                self._sensor["dropout"],
                self._scan_seed,
            ],
            outputs=[self._points, self._colors],
            device=m.device,
        )

    @override
    def _render(self, segment_num: int):
        sim_time = segment_num * self.steps_per_segment * self.clock.dt
        self._scan()
        self.viewer.begin_frame(sim_time)
        self.viewer.log_state(self.current_state)
        self.viewer.log_contacts(self.contacts, self.current_state)
        self.viewer.log_points("lidar", self._points, radii=self._radii, colors=self._colors)
        self.viewer.end_frame()

    def selftest(self, n_segments: int):
        """Headless: scripted drive (straight, then turning) + per-scan hit statistics."""
        for seg in range(n_segments):
            self.script_vels = (
                np.array([5.0, 5.0, 5.0]) if seg < n_segments // 2 else np.array([5.0, 1.0, 3.0])
            )
            self._run_simulation_segment(seg)
            if seg % 10 == 0:
                self._scan()
                d = self._dists.numpy()
                hits = d[d < _MISS]
                kept = int((self._points.numpy()[:, 2] > -1.0e5).sum())  # survived the dropout
                bq = self.current_state.body_q.numpy()[self._chassis]
                print(
                    f"seg {seg:3d} pos=({bq[0]:+6.2f},{bq[1]:+6.2f},{bq[2]:+5.2f}) "
                    f"hits={len(hits)}/{self._n_rays} kept={kept} "
                    f"range=[{hits.min():.2f}, {hits.max():.2f}] m"
                    if len(hits)
                    else f"seg {seg:3d} NO HITS"
                )
        assert len(hits) > self._n_rays * 0.3, "lidar sees too little of the obstacle course"
        drop = 1.0 - kept / len(hits)
        expected_drop = self._sensor["dropout"]
        assert expected_drop * 0.3 < drop < expected_drop * 2.0 + 0.01, (
            f"dropout {drop:.3f} vs configured {expected_drop}"
        )
        # same pose, two back-to-back scans -> the noise must decorrelate the clouds at ~sigma scale
        self._scan()
        p1 = self._points.numpy()
        self._scan()
        p2 = self._points.numpy()
        both = (p1[:, 2] > -1.0e5) & (p2[:, 2] > -1.0e5)
        jitter = float(np.linalg.norm(p1[both] - p2[both], axis=1).mean())
        assert 0.005 < jitter < 0.2, f"per-scan noise jitter {jitter:.4f} m implausible"
        print(f"lidar selftest OK (dropout={drop:.3f}, scan-to-scan jitter={jitter*100:.1f} cm)")


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def helhest_lidar_teleop(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    simulator = HelhestLidarTeleop(
        sim_config,
        render_config,
        engine_config,
        logging_config,
        sensor=cfg.get("sensor", "ouster"),
        realtime=bool(cfg.get("realtime", True)),
        control_mode=cfg.control.mode,
        k_p=cfg.control.k_p,
        k_d=cfg.control.k_d,
        friction=cfg.friction_coeff,
    )
    if cfg.get("selftest_steps"):
        simulator.selftest(int(cfg.selftest_steps))
    else:
        simulator.run()


if __name__ == "__main__":
    helhest_lidar_teleop()
