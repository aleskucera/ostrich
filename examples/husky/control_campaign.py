"""Drive a Clearpath Husky A200 through one of the clark_paper campaign worlds.

The Husky counterpart of examples/helhest_junior/control_campaign.py: same CLI,
same terrain treatment, same I/J/K/L teleop, four wheels instead of three.

    python examples/husky/control_campaign.py
    python examples/husky/control_campaign.py world=curb_or_mound variant=box30
    python examples/husky/control_campaign.py world=shadowed_trench variant=hidden
    python examples/husky/control_campaign.py world=curb_or_mound variant=box30 pose=decision

Worlds: shadowed_trench (hidden, bump, control), two_gaps, blind_crest,
open_field, drifting_berm (hidden, control), remembered_box (hidden, no_box),
curb_or_mound (box25, box30, box35, no_box). Drive with I/J/K/L.

The npz is consumed exactly as the Helhest campaign consumes it -- in fact by
the same `load_campaign_world`: the cells under every solid are flattened back
to ground and the solids are re-added as boxes, so a wheel pressing on a wall
finds an analytic face rather than sliver triangles; the ground, including steps
the vehicle may ride, is newton's heightfield.

`pose=` spawns the robot at one of the spec's named poses (start, or a role such
as decision / out_of_view / abeam).
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
from ostrich import InteractiveSimulator
from ostrich import LoggingConfig
from ostrich import RenderingConfig
from ostrich import SimulationConfig

# `examples` is importable from the editable install; the insert keeps a bare
# `python examples/husky/control_campaign.py` working without one.
_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from examples.helhest_junior.control_campaign import load_campaign_world  # noqa: E402
from examples.husky.common import create_husky_model  # noqa: E402
from examples.husky.common import HuskyConfig  # noqa: E402

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")


@wp.kernel
def integrate_wheel_position_kernel(
    current_wheel_angles: wp.array(dtype=wp.float32),
    target_velocities: wp.array(dtype=wp.float32),
    dt: float,
    joint_target_pos: wp.array(dtype=wp.float32),
    fl_idx: int,
    fr_idx: int,
    rl_idx: int,
    rr_idx: int,
):
    # Integrate: angle = angle + velocity * dt, per wheel.
    a_fl = current_wheel_angles[0] + target_velocities[0] * dt
    a_fr = current_wheel_angles[1] + target_velocities[1] * dt
    a_rl = current_wheel_angles[2] + target_velocities[2] * dt
    a_rr = current_wheel_angles[3] + target_velocities[3] * dt

    current_wheel_angles[0] = a_fl
    current_wheel_angles[1] = a_fr
    current_wheel_angles[2] = a_rl
    current_wheel_angles[3] = a_rr

    joint_target_pos[fl_idx] = a_fl
    joint_target_pos[fr_idx] = a_fr
    joint_target_pos[rl_idx] = a_rl
    joint_target_pos[rr_idx] = a_rr


@wp.kernel
def apply_wheel_velocity_kernel(
    target_velocities: wp.array(dtype=wp.float32),
    joint_target_vel: wp.array(dtype=wp.float32),
    fl_idx: int,
    fr_idx: int,
    rl_idx: int,
    rr_idx: int,
):
    joint_target_vel[fl_idx] = target_velocities[0]
    joint_target_vel[fr_idx] = target_velocities[1]
    joint_target_vel[rl_idx] = target_velocities[2]
    joint_target_vel[rr_idx] = target_velocities[3]


class HuskyCampaignSimulator(InteractiveSimulator):
    """Keyboard-driven Husky in a clark_paper campaign world."""

    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
        world: str = "curb_or_mound",
        variant: str = "box30",
        pose: str = "start",
        terrain_mu: float = 0.8,
        friction_lateral: float = 0.5,
        friction_long: float = 0.9,
        control_mode: str = "velocity",
        k_p: float = 250.0,
        k_d: float = 0.0,
    ):
        self.campaign_world = world
        self.campaign_variant = variant
        self.spawn_pose = pose
        self.terrain_mu = terrain_mu
        self.friction_lateral = friction_lateral
        self.friction_long = friction_long
        self.control_mode = control_mode
        self.k_p = k_p
        self.k_d = k_d
        super().__init__(sim_config, render_config, engine_config, logging_config)

        # Commanded wheel speeds [fl, fr, rl, rr] (rad/s) and, in position mode,
        # the integrated wheel angles they drive.
        self.target_velocities = wp.zeros(4, dtype=wp.float32, device=self.model.device)
        self.wheel_angles = wp.zeros(4, dtype=wp.float32, device=self.model.device)
        self.joint_target_buffer = wp.zeros_like(self.model.joint_target_pos)

        # Rate-limited keyboard command: the target ramps toward the key-commanded
        # speed instead of snapping to it. Same limits as the Helhest teleop.
        self._cmd = np.zeros(4, dtype=np.float32)
        self.max_wheel_accel = 10.0  # rad/s**2
        self.max_wheel_decel = 20.0  # rad/s**2

    # ------------------------------------------------------------------ world

    def _spawn(self, spec, z) -> tuple[float, float, float]:
        if self.spawn_pose == "start":
            return tuple(float(v) for v in z["start"])
        poses = spec.get("shadow_preview", {}).get("poses", [])
        for p in poses:
            if p.get("role") == self.spawn_pose or p.get("name") == self.spawn_pose:
                return float(p["x"]), float(p["y"]), float(p["yaw"])
        names = [f"{p.get('name')} ({p.get('role')})" for p in poses]
        sys.exit(f"ERROR: unknown pose {self.spawn_pose!r}; have start, " + ", ".join(names))

    def _add_terrain(self, hm):
        """The ground as newton's native heightfield collider.

        Newton lays a heightfield out with cols along x and rows along y, which
        is exactly this Heightmap's [ny, nx] layout, so the data goes across
        untransposed. The field is centred on its own origin, hence the xform.
        """
        z_min, z_max = float(hm.H.min()), float(hm.H.max())
        if z_max - z_min < 1e-6:
            z_max = z_min + 1.0  # a zero z-range is degenerate

        self.builder.add_shape_heightfield(
            xform=wp.transform(
                wp.vec3(hm.x0 + hm.nx * hm.cell / 2.0, hm.y0 + hm.ny * hm.cell / 2.0, 0.0),
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
        self.builder.rigid_gap = 0.01

        hm, solids, spec, z = load_campaign_world(self.campaign_world, self.campaign_variant)
        self.heightmap = hm
        self.goal = tuple(float(v) for v in z["goal"])
        self._add_terrain(hm)
        self._add_obstacles(solids)

        sx, sy, syaw = self._spawn(spec, z)
        self.start = (sx, sy, syaw)
        spawn_z = float(hm.sample(sx, sy)) + HuskyConfig.WHEEL_RADIUS + 0.02
        create_husky_model(
            self.builder,
            xform=wp.transform(
                wp.vec3(sx, sy, spawn_z),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), syaw),
            ),
            control_mode=self.control_mode,
            k_p=self.k_p,
            k_d=self.k_d,
            friction=self.friction_lateral,
            friction_long=self.friction_long,
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
            print(
                f"  pose {p.get('name')} [{p.get('role')}]: "
                f"({p['x']:.2f}, {p['y']:.2f}, {np.degrees(p['yaw']):.0f} deg)"
            )
        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)

    # ---------------------------------------------------------------- control

    @override
    def _run_simulation_segment(self, segment_num: int):
        self._update_input()
        super()._run_simulation_segment(segment_num)

    def _update_input(self):
        """Read I/J/K/L and update the commanded wheel speeds (skid-steer)."""
        base_speed = 5.0
        turn_speed = 3.0

        left_v = 0.0
        right_v = 0.0

        if self.viewer and hasattr(self.viewer, "is_key_down"):
            if self.viewer.is_key_down("i"):  # forward
                left_v += base_speed
                right_v += base_speed
            if self.viewer.is_key_down("k"):  # backward
                left_v -= base_speed
                right_v -= base_speed
            if self.viewer.is_key_down("j"):  # turn left
                left_v -= turn_speed
                right_v += turn_speed
            if self.viewer.is_key_down("l"):  # turn right
                left_v += turn_speed
                right_v -= turn_speed

        # Both wheels of a side share the command; that is the skid-steer.
        target = np.array([left_v, right_v, left_v, right_v], dtype=np.float32)
        seg_dt = self.steps_per_segment * self.clock.dt
        accelerating = np.abs(target) >= np.abs(self._cmd)
        max_delta = np.where(
            accelerating,
            self.max_wheel_accel * seg_dt,
            self.max_wheel_decel * seg_dt,
        ).astype(np.float32)
        self._cmd = (self._cmd + np.clip(target - self._cmd, -max_delta, max_delta)).astype(
            np.float32
        )

        wp.copy(self.target_velocities, wp.array(self._cmd, device=self.model.device))

    @override
    def init_state_fn(
        self,
        current_state: newton.State,
        next_state: newton.State,
        contacts: newton.Contacts,
        dt: float,
    ):
        self.solver.integrate_bodies(self.model, current_state, next_state, dt)

    @override
    def control_policy(self, current_state: newton.State):
        dofs = [
            HuskyConfig.FRONT_LEFT_DOF,
            HuskyConfig.FRONT_RIGHT_DOF,
            HuskyConfig.REAR_LEFT_DOF,
            HuskyConfig.REAR_RIGHT_DOF,
        ]
        if self.control_mode == "velocity":
            wp.launch(
                kernel=apply_wheel_velocity_kernel,
                dim=1,
                inputs=[self.target_velocities, self.joint_target_buffer, *dofs],
                device=self.model.device,
            )
            wp.copy(self.control.joint_target_vel, self.joint_target_buffer)
        else:
            wp.launch(
                kernel=integrate_wheel_position_kernel,
                dim=1,
                inputs=[
                    self.wheel_angles,
                    self.target_velocities,
                    self.clock.dt,
                    self.joint_target_buffer,
                    *dofs,
                ],
                device=self.model.device,
            )
            wp.copy(self.control.joint_target_pos, self.joint_target_buffer)


@hydra.main(config_path=str(CONFIG_PATH), config_name="husky_campaign", version_base=None)
def husky_campaign_example(cfg: DictConfig):
    # rendering=gl (default) opens the viewer; rendering=headless runs the
    # configured duration without a window, which is how the world is smoke-tested.
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)

    simulator = HuskyCampaignSimulator(
        hydra.utils.instantiate(cfg.simulation),
        render_config,
        hydra.utils.instantiate(cfg.engine),
        hydra.utils.instantiate(cfg.logging),
        world=cfg.world,
        variant=cfg.variant,
        pose=cfg.pose,
        terrain_mu=cfg.terrain_mu,
        friction_lateral=cfg.friction.lateral,
        friction_long=cfg.friction.long,
        control_mode=cfg.control.mode,
        k_p=cfg.control.k_p,
        k_d=cfg.control.k_d,
    )
    simulator.run()


if __name__ == "__main__":
    husky_campaign_example()
