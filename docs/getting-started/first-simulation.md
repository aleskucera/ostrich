# Your First Simulation

This tutorial will guide you through creating a custom physics simulation with Axion.

## Basic Structure

Every Axion simulation follows the same pattern:

!!! info "The Three Core Steps"

    1.  **Inherit from `InteractiveSimulator`** - This base class handles the simulation loop, USD export, and configuration.
    2.  **Override `build_model()`** - This is where you define your physics scene (bodies, joints, constraints).
    3.  **Configure and run** - You set physics parameters and execute the simulation from the main entry point.

The simplest possible simulator looks like this:

```python
from axion import InteractiveSimulator
import warp as wp

class MySimulator(InteractiveSimulator):
    def build_model(self) -> wp.sim.Model:
        # Build your physics model here
        pass
```

---

## Step 1: Create a Falling Rod

Let's begin by creating a simple simulation of a single rod falling under gravity. This will introduce the core concepts of rigid bodies and shapes.

```python hl_lines="30-35 37-46 48-49"
import warp as wp
from axion import InteractiveSimulator
from axion import EngineConfig
from axion import RenderingConfig
from axion import SimulationConfig


class Simulator(InteractiveSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: EngineConfig,
    ):
        super().__init__(sim_config, render_config, engine_config)

    def build_model(self) -> wp.sim.Model:
        # Create model builder with Z-axis up (gravity points down in -Z)
        builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))

        # Define initial rotation: 15 degrees tilt around X-axis
        # This makes the rod fall instead of standing perfectly upright
        angle = 0.2618  # radians (approx. 15 degrees)
        rot_a_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle)

        # (1) Create a rigid body "rod A" at position (0, 0, 3) meters
        # The rod starts 3 meters above ground with a slight tilt
        rod_a = builder.add_body(
            origin=wp.transform((0, 0, 3), rot_a_quat),
            name="rod A"
        )
        
        # (2) Attach a box shape to the body (this defines its collision geometry)
        builder.add_shape_box(
            body=rod_a,
            hx=0.2,  # half-width: 0.4m total width
            hy=0.2,  # half-depth: 0.4m total depth  
            hz=1.0,  # half-height: 2.0m total height (tall rod)
            density=1000.0,      # kg/m³ (like water)
            mu=0.8,              # friction coefficient (fairly grippy)
            restitution=0.3,     # bounce factor (0=no bounce, 1=perfect bounce)
        )
        
        # (3) This creates an infinite horizontal surface at Z=0
        builder.set_ground_plane(mu=0.8, restitution=0.3)

        # Finalize the model (required to prepare for simulation)
        return builder.finalize()

if __name__ == "__main__":
    # Create simulator instance with configuration
    sim = Simulator(
        sim_config=SimulationConfig(duration_seconds=5.0),  # Run for 5 seconds
        render_config=RenderingConfig(),                    # Default rendering settings
        engine_config=EngineConfig(),                       # Default physics settings
    )
    sim.run()  # Execute the simulation and generate USD file
```

1. **Create a Rigid Body**: The `origin` transform sets the body's initial state. It combines a position (`(0, 0, 3)`) with a rotation (`rot_a_quat`) to place the body in the world. A body itself has no physical presence; it's just a point in space.
2. **Define its Shape**: We give the body physical form by attaching a `shape`. `add_shape_box` creates a box collider. The `h` prefix in `hx`, `hy`, `hz` stands for **half-extents**. So, a `hz` of `1.0` creates a rod that is 2.0 meters tall. The shape also defines the physical material properties.
3. **Add a Ground Plane**: This is a convenient helper to create an infinite, static collision surface at `Z=0` so objects have something to collide with.

!!! success "Result"
    When you run `uv run python first_simulation.py`:

    1.  **Initialization**: The rod spawns 3 meters above the ground with a 15° tilt.
    2.  **Gravity**: The rod falls under gravity (9.81 m/s² downward).
    3.  **Impact**: When it hits the ground plane, the physics engine computes contact forces.
    4.  **Friction & Bounce**: The rod tumbles and slides to a rest, governed by the `mu` and `restitution` values. A USD file is saved with the animation.

---

## Step 2: Add Multiple Bodies

Now let's add a second rod. This demonstrates how the physics engine automatically handles body-body collisions in addition to body-ground collisions.

```python hl_lines="22-33"
# ... (imports and class definition are the same) ...

    def build_model(self) -> wp.sim.Model:
        builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))

        # Create two rods with opposite tilts - they'll fall toward each other
        angle = 0.2618  # 15 degrees in radians
        
        # Rod A: tilted forward (+15°)
        rot_a_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle)
        rod_a = builder.add_body(
            origin=wp.transform((0, 0, 3), rot_a_quat), 
            name="rod A"
        )
        builder.add_shape_box(
            body=rod_a,
            hx=0.2, hy=0.2, hz=1.0,  # Same dimensions as before
            density=1000.0,
            mu=0.8, restitution=0.3,
        )

        # Rod B: tilted backward (-15°) and offset 1m in Y direction
        rot_b_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -angle) # (1)
        rod_b = builder.add_body(
            origin=wp.transform((0, 1, 3), rot_b_quat),  # (2)
            name="rod B"
        )
        builder.add_shape_box(
            body=rod_b,
            hx=0.2, hy=0.2, hz=1.0,
            density=1000.0,  # Same material properties
            mu=0.8, restitution=0.3,
        )

        # Add ground plane for both rods to land on
        builder.set_ground_plane(mu=0.8, restitution=0.3)

        return builder.finalize()

# ... (if __name__ == "__main__" block is the same) ...
```

1. **Opposite Tilt**: We create a new rotation for Rod B using `-angle`. This makes it tilt in the opposite direction from Rod A, causing them to fall towards each other.
2. **Different Position**: We change the `x, y, z` position in the `transform` to `(0, 1, 3)`. This spawns Rod B one meter away from Rod A along the Y-axis.

!!! note "What's New in Step 2?"
    - **Multiple Bodies**: We now have two dynamic bodies, `rod A` and `rod B`.
    - **Body-Body Collision**: The physics engine will now solve for contacts between the two rods in addition to contacts with the ground.
    - **Different Initial Conditions**: By giving the rods opposite tilts and slightly different starting positions, you can create more complex and interesting interactions.

---

## Step 3: Add Joints

Constraints are used to connect bodies and restrict their motion. Here, we'll connect the two rods with a **revolute joint** to create a hinged mechanism.

```python hl_lines="28-44"
# ... (imports and class definition are the same) ...

    def build_model(self) -> wp.sim.Model:
        # Create model builder with Z-axis up
        builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))

        angle = 0.2618
        rot_a_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle)

        # Add rod A with initial rotation
        rod_a = builder.add_body(origin=wp.transform((0, 0, 3), rot_a_quat), name="rod A")
        builder.add_shape_box(
            body=rod_a,
            hx=0.2, hy=0.2, hz=1.0,
            density=1000.0, mu=0.8, restitution=0.3,
        )

        rot_b_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -angle)

        # Add rod B with initial rotation
        rod_b = builder.add_body(origin=wp.transform((0, 1, 3), rot_b_quat), name="rod B")
        builder.add_shape_box(
            body=rod_b,
            hx=0.2, hy=0.2, hz=1.0,
            density=1000.0, mu=0.8, restitution=0.3,
        )

        # Add revolute joint connecting the two rods
        # This creates a hinge that allows rotation around the X-axis
        builder.add_joint_revolute( # (1)
            parent=rod_a,                                    # First rod acts as parent
            child=rod_b,                                     # Second rod is the child
            axis=wp.vec3(1.0, 0.0, 0.0),                     # Rotation axis (X-axis)
            # Connection points are computed to be at the bottom corners of each rod
            # These coordinates account for the initial rotations and positions
            parent_xform=wp.transform( # (2)
                wp.vec3(0.0, 0.2329623, -1.06242221),
                wp.quat_identity()
            ),
            child_xform=wp.transform( # (3)
                wp.vec3(0.0, -0.2329623, -1.06242221),
                wp.quat_identity()
            ),
        )

        builder.set_ground_plane(mu=0.8, restitution=0.3)
        return builder.finalize()

# ... (if __name__ == "__main__" block is the same) ...
```

### Understanding Joints

1. **`add_joint_revolute`**: This function creates a hinge joint. It constrains the motion between a `parent` and `child` body, forcing them to pivot around a common point along a specified `axis`.
2. **`parent_xform`**: This defines the joint's anchor point **in the local coordinate system of the parent body (`rod_a`)**. The joint frame is attached to `rod_a` at this local transform.
3. **`child_xform`**: This defines the joint's anchor point **in the local coordinate system of the child body (`rod_b`)**. The physics engine will ensure these two local frames are always coincident, creating the joint.

---

## Understanding the Physics Parameters

Let's break down the key parameters you've been using.

### Material Properties

These are set on `add_shape_*` calls and define how a body reacts to contact.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `density` | `float` | Mass per unit volume (kg/m³). In combination with the shape's volume, this determines the final mass and inertia of the body. |
| `mu` | `float` | Coulomb friction coefficient. `0.0` is frictionless (like ice), while `1.0` is very high friction. |
| `restitution` | `float` | Coefficient of restitution, or "bounciness". `0.0` means no bounce at all, while `1.0` would be a perfectly elastic bounce with no energy loss. |

### Transform and Rotation

A transform defines an object's position and orientation in space.

```python
# A transform combines a position vector and a rotation quaternion
wp.transform(wp.vec3(x, y, z), quaternion)

# A quaternion is a way to represent 3D rotation.
# The easiest way to create one is from an axis and an angle.
wp.quat_from_axis_angle(axis_vector, angle_in_radians)

# Common Axes:
# X-axis: wp.vec3(1, 0, 0)
# Y-axis: wp.vec3(0, 1, 0)
# Z-axis: wp.vec3(0, 0, 1)  (This is our 'up' vector)
```

## Configuration Options

The `__main__` block in the script passes several configuration objects to the simulator. You can customize them to change the simulation's behavior.

=== "SimulationConfig"
    ```python
    @dataclass
    class SimulationConfig:
        """Parameters defining the simulation's timeline and execution
        strategy."""

        duration_seconds: float = 3.0
        target_timestep_seconds: float = 1e-3
        num_worlds: int = 1
        sync_mode: SyncMode = SyncMode.ALIGN_FPS_TO_DT
        use_cuda_graph: bool = True
    ```

=== "EngineConfig"
    ```python
    @dataclass(frozen=True)
    class EngineConfig:
        """
        Configuration parameters for the AxionEngine solver.

        This object centralizes all tunable parameters for the physics simulation,
        including solver iterations, stabilization factors, and compliance values.
        Making it a frozen dataclass ensures that configuration is immutable
        during a simulation run.
        """

        newton_iters: int = 8
        linear_iters: int = 4

        joint_stabilization_factor: float = 0.01
        contact_stabilization_factor: float = 0.1
        contact_compliance: float = 1e-4
        friction_compliance: float = 1e-6

        contact_fb_alpha: float = 0.25
        contact_fb_beta: float = 0.25
        friction_fb_alpha: float = 0.25
        friction_fb_beta: float = 0.25

        linesearch_steps: int = 2

        matrixfree_representation: bool = True
    ```

=== "RenderingConfig"
    ```python
    @dataclass
    class RenderingConfig:
        """Parameters for rendering the simulation to a USD file."""

        enable: bool = True
        fps: int = 30
        scaling: float = 100.0
        usd_file: str = "sim.usd"
    ```

=== "LoggingConfig"
    ```python
    # Three independent sub-configs, one per logging subsystem.
    # Each sub-config has its own ``enabled`` flag and ``file`` path.
    @dataclass
    class LoggingConfig:
        hdf5: HDF5LoggingConfig = field(default_factory=HDF5LoggingConfig)
        dataset: DatasetLoggingConfig = field(default_factory=DatasetLoggingConfig)
        adjoint: AdjointLoggingConfig = field(default_factory=AdjointLoggingConfig)
    ```

=== "ProfilingConfig"
    ```python
    # Lives on AxionEngineConfig as ``profiling``. Drives the CUDA-event
    # profiler (``axion.profiling.EngineProfiler``).
    @dataclass
    class ProfilingConfig:
        mode: Literal["off", "end_to_end", "per_component"] = "off"
        segment_timing: bool = False
    ```

## Next Steps

- Explore the [User Guide](../user-guide/concepts.md) to understand core physics concepts in more depth.
- Learn about other [constraints](../user-guide/constraints.md) like prismatic (slider) or ball joints.
- See all available [configuration options](../getting-started/configuration.md) for fine-tuning your simulation.
