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
# wp.config.print_launches = True
# wp.config.verbose = True
# wp.config.verbose_warnings = True
# wp.config.mode = "debug"
# wp.config.verify_cuda = True


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
        FRICTION = 0.1
        DENSITY = 1500.0
        KE = 10000000.0
        KD = 200000.0
        KF = 50000.0

        box1_hx = 0.2
        box2_hx = 0.8
        box3_hx = 1.6

        self.builder.rigid_gap = 1.0

        box1 = self.builder.add_body(
            xform=wp.transform((0.0, 0.0, box1_hx), wp.quat_identity()), label="box1"
        )
        self.builder.add_shape_box(
            body=box1,
            hx=box1_hx,
            hy=box1_hx,
            hz=box1_hx,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=DENSITY,
                ke=KE,
                kd=KD,
                kf=KF,
                mu=FRICTION,
            ),
        )

        box2 = self.builder.add_body(
            xform=wp.transform((0.0, 0.0, 2 * box1_hx + box2_hx), wp.quat_identity()), label="box2"
        )
        self.builder.add_shape_box(
            body=box2,
            hx=box2_hx,
            hy=box2_hx,
            hz=box2_hx,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=DENSITY,
                ke=KE,
                kd=KD,
                kf=KF,
                mu=FRICTION,
            ),
        )

        box3 = self.builder.add_body(
            xform=wp.transform((0.0, 0.0, 2 * box1_hx + 2 * box2_hx + box3_hx), wp.quat_identity()),
            label="box3",
        )
        self.builder.add_shape_box(
            body=box3,
            hx=box3_hx,
            hy=box3_hx,
            hz=box3_hx,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=DENSITY,
                ke=KE,
                kd=KD,
                kf=KF,
                mu=FRICTION,
            ),
        )

        self.builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=KE,
                kd=KD,
                kf=KF,
                mu=FRICTION,
            )
        )

        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
def box_stack_example(cfg: DictConfig):
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
    box_stack_example()
