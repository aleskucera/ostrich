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

        self.builder.rigid_gap = 1.0

        ball = self.builder.add_body(
            xform=wp.transform((0.0, 0.0, 2.0), wp.quat_identity()), label="ball"
        )
        initial_velocity = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        self.builder.add_shape_sphere(
            body=ball,
            radius=1.0,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=10.0,
                ke=6000.0,
                kd=1000.0,
                kf=200.0,
                mu=FRICTION,
                restitution=RESTITUTION,
            ),
        )

        self.builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=6000.0,
                kd=1000.0,
                kf=200.0,
                mu=FRICTION,
                restitution=RESTITUTION,
            )
        )

        self.builder.body_qd[0] = initial_velocity
        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
def ball_bounce_example(cfg: DictConfig):
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
    ball_bounce_example()
