# Quick Start Guide

Get up and running with Ostrich in minutes. This guide assumes you have already [installed Ostrich](installation.md) using `uv`.

## Running Your First Simulation

Ostrich comes with several pre-built examples. Let's run the simplest one.

```bash
uv run ball_bounce_example
```

This command will:

1. Execute the physics simulation for a ball bouncing on a ground plane.
2. Generate a `ball_bounce_example.usd` file in your directory containing the full motion history.

!!! tip "Viewing USD Files"
    Universal Scene Description (`.usd`) is a 3D scene format developed by Pixar. You can view the output files using any USD-compatible viewer, such as [Blender](https://www.blender.org/) or Apple's [Reality Converter](https://developer.apple.com/augmented-reality/tools/).

## Exploring the Examples

You can run several other examples using the same `uv run` command. Each demonstrates different features of the Ostrich physics engine.

| Command | Description |
| :--- | :--- |
| `uv run ball_bounce_example` | A simple ball bouncing, demonstrating basic rigid body dynamics and collision. |
| `uv run collision_primitives_example` | Multiple primitive shapes (spheres, boxes, capsules) interacting with each other. |
| `uv run helhest_example` | A simulation of an articulated quadruped robot, demonstrating joints, actuators, and more complex dynamics. |

---

## Configuration with Hydra

All examples are configured using a powerful library called [Hydra](https://hydra.cc/). Hydra allows you to easily override any part of the simulation's configuration directly from the command line, without ever touching the code.

The configuration files for the examples are located in `src/ostrich/examples/conf/`.

### Viewing All Configuration Options

To see every available parameter, its default value, and the available configuration groups, run any example with the `-h` or `--help` flag. [hydra.cc](https://hydra.cc/docs/1.0/advanced/hydra-command-line-flags/)

```bash
uv run ball_bounce_example -h
```

This will print a detailed help message showing the entire default configuration tree.

### Overriding Specific Parameters

You can change any setting using the `key=value` syntax. For nested parameters, use a dot (`.`) path.

```bash title="Command-Line Overrides"
# Change the simulation duration from 4.0s to 10.0s
uv run ball_bounce_example simulation.duration_seconds=10.0

# Increase the physics solver accuracy by using more iterations
uv run ball_bounce_example engine.newton_iters=20

# Run a simulation for 2 seconds and save the output to a different file
uv run ball_bounce_example simulation.duration_seconds=2.0 rendering.usd_file=my_test.usd
```

### Using Configuration Groups

The help output also shows `Configuration groups`. These are pre-defined bundles of settings that let you swap out entire sections of the configuration with a single command.

For example, the `rendering` group has two options: `30_fps` (the default) and `headless`.

```bash title="Using Configuration Groups"
# Run the simulation without rendering to a USD file (useful for headless servers)
uv run ball_bounce_example rendering=headless
```

By specifying `rendering=headless`, you are selecting the `headless.yaml` configuration file from the `src/ostrich/examples/conf/rendering/` directory. This single command is a shortcut for setting `rendering.enable=false`.

!!! info "Combining Overrides"
    You can combine configuration groups and specific overrides in a single command. The most specific override (the `key=value` pair) always wins.

    ```bash
    # Use the 'helhest' simulation settings but override the duration
    uv run helhest_example simulation=helhest simulation.duration_seconds=5.0
    ```

## Next Steps

You now have the basics of running and configuring Ostrich simulations.

- [**Create Your First Simulation**](first-simulation.md) to learn how to build a scene from scratch.
- Dive into the [**User Guide**](../user-guide/concepts.md) to understand the core physics concepts behind Ostrich.
- Explore the [**Configuration System Guide**](../user-guide/configuration.md) for a deeper look into how Hydra is used in Ostrich.
