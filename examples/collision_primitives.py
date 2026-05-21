import os
import pathlib

import hydra
import newton
import warp as wp
from axion import EngineConfig
from axion import InteractiveSimulator
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from omegaconf import DictConfig

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
        FRICTION = 0.4
        RESTITUTION = 0.0
        DENSITY = 1000.0
        KE = 200.0
        KD = 50.0
        KF = 200.0

        ball1 = self.builder.add_body(
            xform=wp.transform((0.0, 0.0, 2.0), wp.quat_identity()), label="ball1"
        )
        self.builder.add_shape_sphere(
            body=ball1,
            radius=0.5,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=DENSITY,
                ke=KE,
                kd=KD,
                kf=KF,
                mu=FRICTION,
                restitution=RESTITUTION,
            ),
        )

        ball2 = self.builder.add_body(
            xform=wp.transform((0.0, 0.3, 4.5), wp.quat_identity()), label="ball2"
        )

        self.builder.add_shape_sphere(
            body=ball2,
            radius=0.5,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=DENSITY,
                ke=KE,
                kd=KD,
                kf=KF,
                mu=FRICTION,
                restitution=RESTITUTION,
            ),
        )

        ball3 = self.builder.add_body(
            xform=wp.transform((0.0, -0.6, 6.5), wp.quat_identity()), label="ball3"
        )

        self.builder.add_shape_sphere(
            body=ball3,
            radius=0.4,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=DENSITY,
                ke=KE,
                kd=KD,
                kf=KF,
                mu=FRICTION,
                restitution=RESTITUTION,
            ),
        )

        ball4 = self.builder.add_body(
            xform=wp.transform((0.0, -0.6, 10.5), wp.quat_identity()), label="ball4"
        )

        self.builder.add_shape_sphere(
            body=ball4,
            radius=0.4,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=DENSITY,
                ke=KE,
                kd=KD,
                kf=KF,
                mu=FRICTION,
                restitution=RESTITUTION,
            ),
        )

        box1 = self.builder.add_body(
            xform=wp.transform((0.0, 0.0, 9.0), wp.quat_identity()), label="box1"
        )

        self.builder.add_shape_box(
            body=box1,
            hx=0.4,
            hy=0.4,
            hz=0.4,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=DENSITY,
                ke=KE,
                kd=KD,
                kf=KF,
                mu=FRICTION,
                restitution=RESTITUTION,
            ),
        )

        self.builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=10,
                kd=10,
                kf=0.0,
                mu=FRICTION,
                restitution=RESTITUTION,
            )
        )

        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
def collision_primitives_example(cfg: DictConfig):
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
    collision_primitives_example()
