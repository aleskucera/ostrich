# Configuration System

Ostrich's examples use the [Hydra](https://hydra.cc/) library to manage simulation parameters. This allows for easy experimentation and configuration changes directly from the command line.

!!! info "This Configuration is for the Examples"
    The Hydra configuration files discussed here are located within the `ostrich/examples/conf` directory. They are designed to make running the pre-built examples easy and flexible.

    You are **not required** to use Hydra in your own projects. You can instantiate Ostrich's configuration classes directly in your Python code. This guide explains how the examples are set up so you can understand them and optionally adapt the pattern for your own use.

---

## How the Examples Use Hydra

The core idea is to separate configuration from code. Instead of hardcoding values like simulation duration or solver iterations, these values are defined in `.yaml` files.

### The Configuration Directory Structure

All configuration files for the examples live in `src/ostrich/examples/conf/`:

```
conf/
├── config.yaml              # Main configuration entry point
├── engine/                  # Physics engine settings (e.g., base.yaml)
├── simulation/              # Time parameters (e.g., base.yaml)
├── rendering/               # USD output settings (e.g., 30_fps.yaml, headless.yaml)
├── execution/               # Performance settings (e.g., cuda_graph.yaml)
└── profiling/               # Debugging/timing settings (e.g., disabled.yaml)
```

The main `config.yaml` file defines the default configuration by composing files from each subdirectory, which Hydra calls **Configuration Groups**.

```yaml title="conf/config.yaml"
defaults:
  - simulation: base
  - rendering: 30_fps 
  - execution: cuda_graph
  - profiling: disabled
  - engine: base
  - _self_

project_name: ${hydra:job.name}
```

This tells Hydra: "By default, use `base.yaml` from the `simulation/` group, `30_fps.yaml` from the `rendering/` group, and so on."

---

## Command-Line Usage for Examples

This system makes it easy to modify any parameter without changing the source code.

### Viewing the Configuration

To see all available parameters and their default values, run any example with the `--help` (or `-h`) flag:

```bash
uv run ball_bounce_example --help
```

### Overriding Parameters

You can override any parameter from the command line using the `key=value` syntax. For nested parameters, use a dot (`.`) path.

```bash title="Overriding Specific Parameters"
# Run for 10 seconds instead of the default 4.0s
uv run ball_bounce_example simulation.duration_seconds=10.0

# Increase physics solver accuracy
uv run ball_bounce_example engine.newton_iters=16
```

### Switching Configuration Groups

A more powerful feature is swapping out entire groups of settings.

```bash title="Switching Groups"
# Run in "headless" mode (disables USD rendering, useful for training)
uv run ball_bounce_example rendering=headless

# Use a different set of physics engine parameters
uv run helhest_example engine=helhest
```

---

## How It's Implemented

Let's look at `ball_bounce_example.py` to see how code and configuration are connected.

```python title="ball_bounce_example.py" hl_lines="11-12 18-20 22-27"
from importlib.resources import files

import hydra
import warp as wp
from ostrich import (
    InteractiveSimulator, EngineConfig, ExecutionConfig, 
    ProfilingConfig, RenderingConfig, SimulationConfig
)
from omegaconf import DictConfig

# (1) Define the path to the example configuration files
CONFIG_PATH = files("ostrich").joinpath("examples").joinpath("conf")

class Simulator(InteractiveSimulator):
    # ... (build_model implementation)
    ...

# (2) Decorate the main function to enable Hydra
@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
def ball_bounce_example(cfg: DictConfig): # (3)
    
    # (4) Instantiate Python objects from the configuration
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    profile_config: ProfilingConfig = hydra.utils.instantiate(cfg.profiling)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)

    simulator = Simulator(
        sim_config=sim_config,
        render_config=render_config,
        exec_config=exec_config,
        profile_config=profile_config,
        engine_config=engine_config,
    )

    simulator.run()

if __name__ == "__main__":
    ball_bounce_example()

```

1. **Locating Config Files**: The script uses `importlib.resources` to find the absolute path to the `conf` directory inside the installed `ostrich` package.
2. **The `@hydra.main` Decorator**: This is the magic that hooks your function into Hydra. It tells Hydra where to find the configuration files (`config_path` and `config_name`).
3. **Receiving the Configuration**: Hydra parses the YAML files and command-line overrides, then passes the final, merged configuration into your function as a special dictionary-like object called a `DictConfig`.
4. **Instantiating Objects**: The `_target_` key in the YAML files (e.g., `_target_: ostrich.SimulationConfig`) tells Hydra which Python class to create. The `hydra.utils.instantiate()` function reads this key and uses the rest of the config values as constructor arguments to create the actual `SimulationConfig`, `EngineConfig`, etc. objects.

---

## Using Configuration in Your Own Project

You have two primary options for managing configuration in your own Ostrich-based project.

=== "Option 1: Without Hydra"

    You don't need Hydra at all. You can simply import the configuration dataclasses and instantiate them yourself. This is the most straightforward approach for simple projects.

    ```python
    from ostrich import (
        Simulator, SimulationConfig, EngineConfig, RenderingConfig, 
        ExecutionConfig, ProfilingConfig
    )

    # Manually create the configuration objects
    sim_config = SimulationConfig(
        duration_seconds=3.0,
        target_timestep_seconds=1e-3,
    )

    render_config = RenderingConfig(
        enable=True,
        fps=30,
        scaling=100.0,
        usd_file="sim.usd",
    )

    exec_config = ExecutionConfig(
        use_cuda_graph=True,
        headless_steps_per_segment=10,
    )

    profile_config = ProfilingConfig(
        enable_timing=False,
        enable_hdf5_logging=False,
        hdf5_log_file="simulation.h5",
    )

    engine_config = EngineConfig(
        newton_iters=8,
        linear_iters=4,
        joint_stabilization_factor=0.01,
        contact_stabilization_factor=0.1,
        contact_compliance=1e-4,
        friction_compliance=1e-6,
        contact_fb_alpha=0.25,
        contact_fb_beta=0.25,
        friction_fb_alpha=0.25,
        friction_fb_beta=0.25,
        linesearch_steps=2,
        matrixfree_representation=True,
    )

    simulator = MySimulator(
        sim_config=sim_config,
        render_config=render_config,
        exec_config=exec_config,
        profile_config=profile_config,
        engine_config=engine_config,
    )

    my_sim.run()
    ```

=== "Option 2: With Hydra"

    If you like the flexibility of the examples, you can adopt the same pattern.

    1. **Copy the `conf` directory** from `src/ostrich/examples/conf` into your own project's root.
    2. **Create your main script** and use the `@hydra.main` decorator, pointing it to your new `conf` directory.

    ```python
    import hydra
    from omegaconf import DictConfig
    from ostrich import ...

    @hydra.main(config_path="str(CONFIG_PATH)", config_name="config", version_base=None)
    def my_main_function(cfg: DictConfig):
        # The rest of the logic is the same as the examples...
        sim_config = hydra.utils.instantiate(cfg.simulation)
        engine_config = hydra.utils.instantiate(cfg.engine)
        # ...
    ```

    This gives you a powerful, battle-tested configuration system for your project with minimal setup.

