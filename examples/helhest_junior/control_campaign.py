"""Drive Helhest Junior through one of the clark_paper campaign worlds.

Same teleop as control_world.py, but the terrain comes from the npz files that
clark_paper/sim_worlds builds (0.10 m cells, arena x -2..16, y -5..5) instead of
helhest_stack's worlds.

    python examples/helhest_junior/control_campaign.py
    python examples/helhest_junior/control_campaign.py world=curb_or_mound variant=box30
    python examples/helhest_junior/control_campaign.py world=shadowed_trench variant=hidden
    python examples/helhest_junior/control_campaign.py world=curb_or_mound variant=box30 pose=decision

Worlds: shadowed_trench (hidden, bump, control), two_gaps, blind_crest,
open_field, drifting_berm (hidden, control), remembered_box (hidden, no_box),
curb_or_mound (box25, box30, box35, no_box). Drive with I/J/K/L.

The npz is consumed the way control_world.py consumes helhest's builders: the
cells under every solid (`boxes`, the obstacles the route must miss) are
flattened back to ground and the solids are re-added as boxes, so a wheel
pressing on a wall finds an analytic face rather than sliver triangles; the
ground, including steps the vehicle may ride such as curb_or_mound's 0.15 m
curb, is newton's heightfield. The arena `perimeter` walls are added as solids
too, which is what the campaign's scan generator does.

`pose=` spawns the robot at one of the spec's named poses (start, or a role
such as decision / out_of_view / abeam) so a scene can be entered where the
planner would decide. Route waypoints and the hazard footprint are printed at
start-up; they are not drawn.
"""

import json
import os
import pathlib
import sys
from dataclasses import dataclass
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
    from examples.helhest_junior.control_world import HelhestJuniorWorldSimulator
except ImportError:
    from common import create_helhest_junior_model
    from common import HelhestJuniorConfig
    from control_world import HelhestJuniorWorldSimulator

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")


def _campaign_results_dir() -> pathlib.Path:
    """clark_paper is a sibling checkout, not a dependency of ostrich."""
    env = os.environ.get("CLARK_PAPER")
    repo = pathlib.Path(__file__).resolve().parents[2]
    roots = [pathlib.Path(env)] if env else [repo.parent / "clark_paper", repo / "clark_paper"]
    for r in roots:
        d = r / "sim_worlds" / "results"
        if d.is_dir():
            return d
    sys.exit(
        "ERROR: cannot find clark_paper/sim_worlds/results. Looked in "
        + ", ".join(str(r) for r in roots)
        + ".\nSet CLARK_PAPER=/path/to/clark_paper."
    )


@dataclass(frozen=True)
class Box:
    """One axis-aligned solid, the (cx, cy, hx, hy, h, yaw) row of the npz."""

    cx: float
    cy: float
    hx: float
    hy: float
    h: float
    yaw: float


class Heightmap:
    """The npz grid with helhest.heightmap.Heightmap's interface (H, x0, y0, nx, ny, cell, sample)."""

    def __init__(self, heights, origin, cell):
        self.H = np.asarray(heights, dtype=np.float64)  # [ny, nx]
        self.ny, self.nx = self.H.shape
        self.x0, self.y0 = float(origin[0]), float(origin[1])
        self.cell = float(cell)

    def sample(self, x, y):
        """Bilinear height at (x, y); cell centers sit at x0 + (j + 0.5) cell."""
        fx = (float(x) - self.x0) / self.cell - 0.5
        fy = (float(y) - self.y0) / self.cell - 0.5
        ix = int(np.clip(np.floor(fx), 0, self.nx - 2))
        iy = int(np.clip(np.floor(fy), 0, self.ny - 2))
        tx = float(np.clip(fx - ix, 0.0, 1.0))
        ty = float(np.clip(fy - iy, 0.0, 1.0))
        h00, h10 = self.H[iy, ix], self.H[iy, ix + 1]
        h01, h11 = self.H[iy + 1, ix], self.H[iy + 1, ix + 1]
        return (1 - ty) * ((1 - tx) * h00 + tx * h10) + ty * ((1 - tx) * h01 + tx * h11)


def load_campaign_world(world: str, variant: str):
    """Return (heightmap with solids flattened, solids, spec) for one npz."""
    results = _campaign_results_dir()
    npz_path = results / f"{world}_{variant}.npz"
    spec_path = results / f"{world}_{variant}_spec.json"
    if not npz_path.exists():
        have = sorted(p.stem for p in results.glob("*.npz"))
        sys.exit(f"ERROR: no {npz_path.name}; have {', '.join(have)}")
    z = np.load(npz_path)
    spec = json.load(open(spec_path)) if spec_path.exists() else {}
    hm = Heightmap(z["H"], z["origin"], float(z["cell"]))

    solids = [Box(*map(float, row)) for row in z["boxes"]]
    perimeter = [Box(*map(float, row)) for row in z["perimeter"]]

    # flatten the solids' footprints back to the surrounding ground (the
    # builders stamp obstacles by overwriting, so the ring around a footprint is
    # the ground it replaced; on the flat worlds this is 0.0)
    xs = hm.x0 + (np.arange(hm.nx) + 0.5) * hm.cell
    ys = hm.y0 + (np.arange(hm.ny) + 0.5) * hm.cell
    X, Y = np.meshgrid(xs, ys)
    H = hm.H.copy()
    for b in solids:
        c, s = np.cos(b.yaw), np.sin(b.yaw)
        u = c * (X - b.cx) + s * (Y - b.cy)
        v = -s * (X - b.cx) + c * (Y - b.cy)
        inside = (np.abs(u) <= b.hx) & (np.abs(v) <= b.hy)
        ring = (np.abs(u) <= b.hx + 2 * hm.cell) & (np.abs(v) <= b.hy + 2 * hm.cell) & ~inside
        H[inside] = float(np.median(hm.H[ring])) if ring.any() else 0.0
    return Heightmap(H, (hm.x0, hm.y0), hm.cell), solids + perimeter, spec, z


class HelhestJuniorCampaignSimulator(HelhestJuniorWorldSimulator):
    """control_world.py's teleop robot, dropped into a clark_paper campaign world."""

    def __init__(self, *args, world: str = "curb_or_mound", variant: str = "box30",
                 pose: str = "start", **kwargs):
        self.campaign_world = world
        self.campaign_variant = variant
        self.spawn_pose = pose
        # HelhestJuniorWorldSimulator validates its `world` against helhest's
        # WORLDS; pass one of its names so the check passes, then override build_model.
        super().__init__(*args, world="pillars", **kwargs)

    def _spawn(self, spec, z) -> tuple[float, float, float]:
        if self.spawn_pose == "start":
            sx, sy, syaw = (float(v) for v in z["start"])
            return sx, sy, syaw
        poses = spec.get("shadow_preview", {}).get("poses", [])
        for p in poses:
            if p.get("role") == self.spawn_pose or p.get("name") == self.spawn_pose:
                return float(p["x"]), float(p["y"]), float(p["yaw"])
        names = [f"{p.get('name')} ({p.get('role')})" for p in poses]
        sys.exit(f"ERROR: unknown pose {self.spawn_pose!r}; have start, " + ", ".join(names))

    @override
    def build_model(self) -> newton.Model:
        self.builder.rigid_gap = 0.01

        hm, solids, spec, z = load_campaign_world(self.campaign_world, self.campaign_variant)
        self.heightmap = hm
        self.goal = tuple(float(v) for v in z["goal"])
        self._add_terrain(hm)
        self._add_obstacles(solids)

        sx, sy, syaw = self._spawn(spec, z)
        self.start = (sx, sy, syaw)
        spawn_z = float(hm.sample(sx, sy)) + HelhestJuniorConfig.WHEEL_RADIUS + 0.02
        create_helhest_junior_model(
            self.builder,
            xform=wp.transform(
                wp.vec3(sx, sy, spawn_z),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), syaw),
            ),
            control_mode=self.control_mode,
            k_p=self.k_p,
            k_d=self.k_d,
            friction_left_right=self.friction_lat_left_right,
            friction_rear=self.friction_lat_rear,
            friction_long_left_right=self.friction_long_left_right,
            friction_long_rear=self.friction_long_rear,
        )

        print(
            f"campaign world '{self.campaign_world}/{self.campaign_variant}': "
            f"{hm.nx}x{hm.ny} cells @ {hm.cell} m, {len(solids)} solid(s) incl. perimeter, "
            f"spawn ({sx:.2f}, {sy:.2f}, yaw {np.degrees(syaw):.0f} deg) -> goal {self.goal}"
        )
        for name in [str(n) for n in np.atleast_1d(z["route_names"])] if "route_names" in z else []:
            if name in z:
                pts = ", ".join(f"({x:.1f}, {y:.1f})" for x, y in np.asarray(z[name]))
                print(f"  route {name}: {pts}")
        if "hazard" in z and z["hazard"].any():
            iy, ix = np.nonzero(z["hazard"])
            print(
                f"  hazard cells: x {hm.x0 + ix.min() * hm.cell:.1f}..{hm.x0 + (ix.max() + 1) * hm.cell:.1f}, "
                f"y {hm.y0 + iy.min() * hm.cell:.1f}..{hm.y0 + (iy.max() + 1) * hm.cell:.1f}"
            )
        for p in spec.get("shadow_preview", {}).get("poses", []):
            print(f"  pose {p.get('name')} [{p.get('role')}]: ({p['x']:.2f}, {p['y']:.2f}, {np.degrees(p['yaw']):.0f} deg)")
        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest_campaign", version_base=None)
def helhest_junior_campaign_example(cfg: DictConfig):
    # rendering=gl (default) opens the viewer; rendering=headless runs the
    # configured duration without a window, which is how the world is smoke-tested.
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)

    simulator = HelhestJuniorCampaignSimulator(
        hydra.utils.instantiate(cfg.simulation),
        render_config,
        hydra.utils.instantiate(cfg.engine),
        hydra.utils.instantiate(cfg.logging),
        world=cfg.world,
        variant=cfg.variant,
        pose=cfg.pose,
        terrain_mu=cfg.terrain_mu,
        friction_lat_left_right=cfg.friction.lateral_left_right,
        friction_lat_rear=cfg.friction.lateral_rear,
        friction_long_left_right=cfg.friction.long_left_right,
        friction_long_rear=cfg.friction.long_rear,
        control_mode=cfg.control.mode,
        k_p=cfg.control.k_p,
        k_d=cfg.control.k_d,
    )
    simulator.run()


if __name__ == "__main__":
    helhest_junior_campaign_example()
