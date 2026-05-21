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


class RandomSimulator(InteractiveSimulator):
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
        # So we should add ground plane to the builder here.
        self.builder.add_ground_plane()

        # 2. Initialize SceneGenerator with our builder
        gen = PlacementSceneGenerator(self.builder, seed=42)

        print("Generating grounded objects...")
        ground_ids = []
        for i in range(4):
            idx = gen.generate_random_ground_touching()
            if idx is not None:
                ground_ids.append(idx)

        print("Generating touching objects...")
        for gid in ground_ids:
            prev = gid
            for _ in range(2):
                new_idx = gen.generate_random_touching(prev)
                if new_idx is not None:
                    prev = new_idx

        print("Generating free objects...")
        for i in range(4):
            gen.generate_random_free()

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
