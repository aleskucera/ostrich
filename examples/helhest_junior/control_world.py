"""Drive Helhest Junior around one of helhest_stack's worlds.

Same keyboard teleop as control.py -- this only swaps the hand-built stairs and
boulders for the worlds the dasenka demo runs, so a drive here is directly
comparable to what the stack's planner sees on the same terrain.

    python examples/helhest_junior/control_world.py
    python examples/helhest_junior/control_world.py world=slalom
    python examples/helhest_junior/control_world.py world=bumpy rendering.target_fps=15

Worlds: gap, slalom, pillars, pocket, ridge, bumpy.  Drive with I/J/K/L.

Two things are deliberately not the naive translation of a heightmap:

*Obstacles are solid boxes, not terrain.* The stack's builders rasterise walls
into the grid, so a wall face -- a 0.06 m cell rising 1.0 m -- would become one
quad with a 16:1 aspect ratio. A wheel pressing on a pillar finds a handful of
sliver triangles where a box gives a dense analytic manifold. So the obstacle
cells are flattened back to ground and re-added as `add_shape_box`.

*The ground is a heightfield, not a triangle mesh.* Contact quality is better
(contact_probe measured 8.7 mm vs 18.5 mm peak-to-peak vertical bounce at
dt=0.05) and it costs a DDA walk over cells rather than a BVH descent.

Wheel friction is anisotropic and matches the tuned dasenka values: `mu` acts
along the projected wheel spin axis (lateral) and `mu_long` perpendicular to it
(rolling), so a three-wheel skid-steer can rotate without its own lateral grip
fighting the turn.
"""

import os
import pathlib
import sys
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


def _import_helhest():
    """helhest_stack is a sibling checkout, not a dependency of ostrich.

    Try the installed package first, then the sibling working copy, so the demo
    runs with a bare `python examples/...` the way control.py does.
    """
    try:
        from helhest import worlds
    except ModuleNotFoundError:
        repo = pathlib.Path(__file__).resolve().parents[2]
        env = os.environ.get("HELHEST_STACK")
        # sibling checkout first (../helhest_stack), then inside the repo
        roots = [pathlib.Path(env)] if env else [repo.parent / "helhest_stack", repo / "helhest_stack"]
        for src in (r / "src" for r in roots):
            if (src / "helhest").is_dir():
                sys.path.insert(0, str(src))
                break
        else:
            sys.exit(
                "ERROR: cannot find helhest_stack. Looked in "
                + ", ".join(str(r / "src") for r in roots)
                + ".\nSet HELHEST_STACK=/path/to/helhest_stack, or put its src on PYTHONPATH."
            )
        from helhest import worlds
    return worlds


class HelhestJuniorWorldSimulator(HelhestJuniorControlSimulator):
    """control.py's teleop robot, dropped into a helhest_stack world."""

    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
        world: str = "pillars",
        terrain_mu: float = 0.8,
        friction_lat_left_right: float = 0.5,
        friction_lat_rear: float = 0.2,
        friction_long_left_right: float = 0.9,
        friction_long_rear: float = 0.6,
        **kwargs,
    ):
        self.worlds = _import_helhest()
        if world not in self.worlds.WORLDS:
            sys.exit(f"ERROR: unknown world {world!r}; have {', '.join(self.worlds.WORLDS)}")
        self.world_name = world
        self.terrain_mu = terrain_mu
        self.friction_lat_left_right = friction_lat_left_right
        self.friction_lat_rear = friction_lat_rear
        self.friction_long_left_right = friction_long_left_right
        self.friction_long_rear = friction_long_rear
        super().__init__(sim_config, render_config, engine_config, logging_config, **kwargs)

    def _ground_only(self, hm, boxes):
        """`hm` with the obstacle cells flattened back to ground.

        The builders stamp obstacles with `H[mask] = height`, overwriting the
        ground rather than adding to it, so clearing the mask to 0.0 recovers the
        ground exactly: the five obstacle worlds start from zeros and only draw
        walls, and bumpy has no obstacles to clear.

        Stamped on this heightmap's OWN cell centres -- rebuilding the grid from
        reconstructed limits rounds to a different cell count.
        """
        if not boxes:
            return hm
        xs = hm.x0 + (np.arange(hm.nx) + 0.5) * hm.cell
        ys = hm.y0 + (np.arange(hm.ny) + 0.5) * hm.cell
        mask = self.worlds.stamp(boxes, *np.meshgrid(xs, ys))
        H = hm.H.copy()
        H[mask > 0.0] = 0.0
        return type(hm)(H, (hm.x0, hm.y0), hm.cell)

    def _add_terrain(self, hm):
        """The ground as newton's native heightfield collider.

        Newton lays a heightfield out with cols along x and rows along y, which
        is exactly this Heightmap's [ny, nx] layout, so the data goes across
        untransposed. The field is centred on its own origin, hence the xform.
        """
        z_min, z_max = float(hm.H.min()), float(hm.H.max())
        if z_max - z_min < 1e-6:
            z_max = z_min + 1.0  # a zero z-range is degenerate; flat data normalises to 0 anyway

        self.builder.add_shape_heightfield(
            xform=wp.transform(
                wp.vec3(
                    hm.x0 + hm.nx * hm.cell / 2.0,
                    hm.y0 + hm.ny * hm.cell / 2.0,
                    0.0,
                ),
                wp.quat_identity(),
            ),
            heightfield=newton.Heightfield(
                data=np.ascontiguousarray(hm.H, np.float32),
                nrow=hm.ny,
                ncol=hm.nx,
                hx=(hm.nx - 1) * hm.cell / 2.0,
                hy=(hm.ny - 1) * hm.cell / 2.0,
                min_z=z_min,
                max_z=z_max,
            ),
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0, has_shape_collision=True, mu=self.terrain_mu
            ),
        )

    def _add_obstacles(self, boxes):
        for box in boxes:
            self.builder.add_shape_box(
                body=-1,
                xform=wp.transform(
                    wp.vec3(box.cx, box.cy, 0.5 * box.h),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), box.yaw),
                ),
                hx=box.hx,
                hy=box.hy,
                hz=0.5 * box.h,
                cfg=newton.ModelBuilder.ShapeConfig(
                    density=0.0, has_shape_collision=True, mu=self.terrain_mu
                ),
            )

    @override
    def build_model(self) -> newton.Model:
        # Contact-generation margin. control.py uses 1.0 against a handful of
        # boxes; that is far too wide for a full terrain, so use the odin_sim value.
        self.builder.rigid_gap = 0.01

        builder_fn, self.start, self.goal = self.worlds.WORLDS[self.world_name]
        self.heightmap = builder_fn()
        boxes = self.worlds.OBSTACLES[self.world_name]

        self._add_terrain(self._ground_only(self.heightmap, boxes))
        self._add_obstacles(boxes)

        sx, sy, syaw = self.start
        spawn_z = float(self.heightmap.sample(sx, sy)) + HelhestJuniorConfig.WHEEL_RADIUS + 0.02
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
            f"world '{self.world_name}': {self.heightmap.nx}x{self.heightmap.ny} cells "
            f"@ {self.heightmap.cell} m, {len(boxes)} solid obstacle(s), "
            f"start ({sx}, {sy}) -> goal {self.goal}"
        )
        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest_world", version_base=None)
def helhest_junior_world_example(cfg: DictConfig):
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    render_config.vis_type = "gl"

    simulator = HelhestJuniorWorldSimulator(
        hydra.utils.instantiate(cfg.simulation),
        render_config,
        hydra.utils.instantiate(cfg.engine),
        hydra.utils.instantiate(cfg.logging),
        world=cfg.world,
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
    helhest_junior_world_example()
