"""Shared helpers for differentiable simulator gradient tests."""
import numpy as np
import warp as wp
import newton
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig, LinearSolverConfig, NewtonRaphsonConfig
from axion.core.logging_config import LoggingConfig
from axion.core.model_builder import AxionModelBuilder
from axion.simulation.trajectory_buffer import TrajectoryBuffer


# =============================================================================
# Scene builders
# =============================================================================


def build_box_on_ground(num_worlds=1, height=0.6):
    """Single box near a ground plane (contacts active)."""
    builder = AxionModelBuilder()
    builder.rigid_gap = 0.05
    builder.add_ground_plane()
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, height), wp.quat_identity()),
    )
    builder.add_shape_box(
        body=body,
        hx=0.5,
        hy=0.5,
        hz=0.5,
        cfg=newton.ModelBuilder.ShapeConfig(density=100.0, mu=0.5),
    )
    return builder.finalize_replicated(num_worlds=num_worlds, gravity=-9.81)


def build_free_box(num_worlds=1):
    """Single box high above a ground plane (no active contacts)."""
    builder = AxionModelBuilder()
    builder.rigid_gap = 0.05
    builder.add_ground_plane()
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 5.0), wp.quat_identity()),
    )
    builder.add_shape_box(
        body=body,
        hx=0.5,
        hy=0.5,
        hz=0.5,
        cfg=newton.ModelBuilder.ShapeConfig(density=100.0, mu=0.5),
    )
    return builder.finalize_replicated(num_worlds=num_worlds, gravity=-9.81)


def build_revolute_pendulum(num_worlds=1):
    """Single-link pendulum with revolute joint (for control gradient tests)."""
    builder = AxionModelBuilder()

    builder.rigid_gap = 0.05
    builder.add_ground_plane()

    link = builder.add_link()
    builder.add_shape_box(
        link,
        hx=0.5,
        hy=0.1,
        hz=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(density=100.0),
    )

    j0 = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 0.0, 1.0),
        # Joint raised above ground to avoid spurious pendulum-ground contacts
        # which create hard-to-converge friction and corrupt the adjoint
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j0], label="arm")

    # Add a box on the ground to ensure contacts exist (engine requires active contacts)
    box = builder.add_body(
        xform=wp.transform(wp.vec3(5.0, 0.0, 0.6), wp.quat_identity()),
    )
    builder.add_shape_box(
        box, hx=0.5, hy=0.5, hz=0.5,
        cfg=newton.ModelBuilder.ShapeConfig(density=100.0, mu=0.5),
    )

    return builder.finalize_replicated(num_worlds=num_worlds, gravity=0.0)


def build_two_boxes_symmetric(num_worlds=1):
    """Two identical boxes placed symmetrically about x=0 on a ground plane."""
    builder = AxionModelBuilder()
    builder.rigid_gap = 0.05
    builder.add_ground_plane()

    for x_sign in [-1.0, 1.0]:
        body = builder.add_body(
            xform=wp.transform(wp.vec3(x_sign * 2.0, 0.0, 0.6), wp.quat_identity()),
        )
        builder.add_shape_box(
            body=body,
            hx=0.5,
            hy=0.5,
            hz=0.5,
            cfg=newton.ModelBuilder.ShapeConfig(density=100.0, mu=0.5),
        )
    return builder.finalize_replicated(num_worlds=num_worlds, gravity=-9.81)


# =============================================================================
# Engine / simulation helpers
# =============================================================================


def make_engine(model, config=None, sim_steps=5):
    if config is None:
        config = AxionEngineConfig(
            nr=NewtonRaphsonConfig(max_iters=20),
            linear=LinearSolverConfig(max_iters=200, tol=1e-8, atol=1e-8),
        )
    return AxionEngine(
        model=model,
        sim_steps=sim_steps,
        config=config,
        logging_config=LoggingConfig(),
        differentiable_simulation=True,
    )


def forward_one_step(engine, model, state_in, control, dt):
    """Run one forward step and return state_out."""
    state_out = model.state()
    contacts = model.collide(state_in)
    engine.step(state_in, state_out, control, contacts, dt)
    return state_out


def scalar_loss_vel(state, weights):
    """L = w^T @ flatten(body_qd). Returns scalar."""
    qd = state.body_qd.numpy().flatten()
    return np.dot(weights, qd)
