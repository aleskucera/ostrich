import os
import pathlib

import hydra
import newton
import numpy as np
import warp as wp
from axion import DatasetSimulator
from axion import EngineConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion.generation import RandomSceneGenerator
from omegaconf import DictConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")


class RandomSimulator(DatasetSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
    ):
        self.pos_min = wp.vec3(-1.0, -1.0, 0.0)
        self.pos_max = wp.vec3(1.0, 1.0, 3.0)
        self.lin_vel_min, self.lin_vel_max = -5.0, 5.0
        self.ang_vel_min, self.ang_vel_max = -3.14, 3.14
        self.joint_target_lower_bound = -10.0
        self.joint_target_upper_bound = 10.0
        self.seed = 42

        super().__init__(
            sim_config,
            render_config,
            engine_config,
            logging_config,
        )

    def build_model(self) -> newton.Model:
        self.builder.rigid_gap = 0.2
        # So we should add ground plane to the builder here.
        self.builder.add_ground_plane()

        # 2. Initialize SceneGenerator with our builder
        gen = RandomSceneGenerator(self.builder, seed=self.seed)

        num_objects = np.random.randint(3, 15)
        for i in range(num_objects):  # Pause again to show post-resolve state.
            gen.generate_random_object(
                pos_bounds=((-1, -1, 0), (1, 1, 3)),
                density_bounds=(10.0, 100.0),
                size_bounds=(0.1, 0.3),
            )

        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
def random_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    simulator = RandomSimulator(
        sim_config=sim_config,
        render_config=render_config,
        engine_config=engine_config,
        logging_config=logging_config,
    )

    simulator.run()


if __name__ == "__main__":
    random_example()
