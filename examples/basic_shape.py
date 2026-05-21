import os
import pathlib

import hydra
import newton.examples
import newton.usd
import warp as wp
from axion import EngineConfig
from axion import InteractiveSimulator
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from omegaconf import DictConfig
from pxr import Usd

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.joinpath("conf")


class Simulator(InteractiveSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
    ):
        super().__init__(
            sim_config,
            render_config,
            engine_config,
            logging_config,
        )

    def build_model(self) -> newton.Model:

        self.builder.rigid_gap = 1.0

        rigid_cfg = self.builder.ShapeConfig()
        rigid_cfg.restitution = 0.0
        rigid_cfg.has_shape_collision = True
        rigid_cfg.mu = 1.0
        rigid_cfg.density = 1000.0

        # add ground plane
        self.builder.add_ground_plane()

        # z height to drop shapes from
        drop_z = 2.0

        # SPHERE
        self.sphere_pos = wp.vec3(0.0, -2.0, drop_z)
        body_sphere = self.builder.add_body(
            xform=wp.transform(p=self.sphere_pos, q=wp.quat_identity()), label="sphere"
        )
        self.builder.add_shape_sphere(body_sphere, radius=0.5)

        # ELLIPSOID (flat disk shape: a=b > c for stability when resting on ground)
        self.ellipsoid_pos = wp.vec3(0.0, -6.0, drop_z)
        body_ellipsoid = self.builder.add_body(
            xform=wp.transform(p=self.ellipsoid_pos, q=wp.quat_identity()), label="ellipsoid"
        )
        self.builder.add_shape_ellipsoid(body_ellipsoid, a=0.5, b=0.5, c=0.25)

        # CAPSULE
        self.capsule_pos = wp.vec3(0.0, 0.0, drop_z)
        body_capsule = self.builder.add_body(
            xform=wp.transform(p=self.capsule_pos, q=wp.quat_identity()), label="capsule"
        )
        self.builder.add_shape_capsule(body_capsule, radius=0.3, half_height=0.7)

        # CYLINDER
        self.cylinder_pos = wp.vec3(0.0, -4.0, drop_z)
        body_cylinder = self.builder.add_body(
            xform=wp.transform(p=self.cylinder_pos, q=wp.quat_identity()), label="cylinder"
        )
        self.builder.add_shape_cylinder(body_cylinder, radius=0.4, half_height=0.6)

        # BOX
        self.box_pos = wp.vec3(0.0, 2.0, drop_z)
        body_box = self.builder.add_body(
            xform=wp.transform(p=self.box_pos, q=wp.quat_identity()), label="box"
        )
        self.builder.add_shape_box(body_box, hx=0.5, hy=0.35, hz=0.25)

        # MESH (bunny)
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
        demo_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        self.mesh_pos = wp.vec3(0.0, 4.0, drop_z - 0.5)
        body_mesh = self.builder.add_body(
            xform=wp.transform(p=self.mesh_pos, q=wp.quat(0.5, 0.5, 0.5, 0.5)), label="mesh"
        )
        self.builder.add_shape_mesh(body_mesh, mesh=demo_mesh)

        # CONE (no collision support in the standard collision pipeline)
        self.cone_pos = wp.vec3(0.0, 6.0, drop_z)
        body_cone = self.builder.add_body(
            xform=wp.transform(p=self.cone_pos, q=wp.quat_identity()), label="cone"
        )
        self.builder.add_shape_cone(body_cone, radius=0.45, half_height=0.6)

        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
def basic_shape_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    simulator = Simulator(
        sim_config=sim_config,
        render_config=render_config,
        engine_config=engine_config,
        logging_config=logging_config,
    )

    simulator.run()


if __name__ == "__main__":
    basic_shape_example()
