import os
import pathlib

import hydra
import newton
import warp as wp
from axion import InteractiveSimulator
from axion import EngineConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion import LoggingConfig
from axion.generation import PlacementSceneGenerator
from omegaconf import DictConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")


class JointsSimulator(InteractiveSimulator):
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
        # Add ground plane
        self.builder.add_ground_plane()

        # Initialize SceneGenerator
        gen = PlacementSceneGenerator(self.builder, seed=123)

        # 1. Revolute Chain (Snake-like)
        gen.generate_chain(length=2, start_pos=(-3, -3, 1), shape_type="box", joint_type="revolute")

        # 2. Ball Joint Chain (Rope-like)
        print("Generating Ball Joint Chain...")
        gen.generate_chain(length=2, start_pos=(3, 3, 1), shape_type="capsule", joint_type="ball")

        # 3. Fixed Chain (Structure)
        print("Generating Fixed Structure...")
        gen.generate_chain(length=2, start_pos=(-3, 3, 1), shape_type="box", joint_type="fixed")

        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
def joints_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    simulator = JointsSimulator(
        sim_config=sim_config,
        render_config=render_config,
        engine_config=engine_config,
        logging_config=logging_config,
    )

    simulator.run()


if __name__ == "__main__":
    joints_example()
