"""Test anisotropic friction: a body with a body-local friction axis should
slide differently along vs perpendicular to that axis."""

import newton
import numpy as np
import pytest
import warp as wp
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.logging_config import LoggingConfig
from axion.core.model_builder import AxionModelBuilder

wp.init()


@wp.kernel
def apply_force_kernel(
    body_f: wp.array(dtype=wp.spatial_vector, ndim=1),
    force: wp.spatial_vector,
):
    body_idx = wp.tid()
    body_f[body_idx] = force


def _build_box_on_ground(mu_lat: float, mu_long: float, ground_mu: float, box_mass: float = 1.0):
    """Box with anisotropic friction. Friction axis is body-local Y, so
    mu_lat resists motion along world-Y (when box is upright) and mu_long
    resists motion along world-X.

    At each contact the combine rule averages the box's per-axis coefficients
    with the ground's scalar mu, so resolved mu_x = (mu_lat + ground_mu)/2
    and mu_y = (mu_long + ground_mu)/2.
    """
    builder = AxionModelBuilder()
    builder.rigid_gap = 0.01
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=ground_mu))
    body = builder.add_body(
        mass=box_mass,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.11), wp.quat_identity()),
    )
    builder.add_shape_box(
        body=body,
        hx=0.1,
        hy=0.1,
        hz=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(mu=mu_lat),
        custom_attributes={
            "friction_axis_local": wp.vec3(0.0, 1.0, 0.0),
            "mu_perp": mu_long,
        },
    )
    return builder.finalize_replicated(num_worlds=1, gravity=-9.81)


def _simulate_under_force(model, force_world: wp.vec3, dt: float, n_steps: int):
    """Run the engine for n_steps with a constant body force. Returns final body xy."""
    engine = AxionEngine(
        model=model, sim_steps=n_steps,
        config=AxionEngineConfig(), logging_config=LoggingConfig(),
    )
    state_in = model.state()
    state_out = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    ctrl = model.control()

    # Settle on the ground for a few steps first (no horizontal force)
    settle_steps = 30
    for _ in range(settle_steps):
        contacts = model.collide(state_in)
        engine.step(state_in, state_out, ctrl, contacts, dt)
        state_in, state_out = state_out, state_in

    q_settled = state_in.body_q.numpy().reshape(-1, 7)
    x0, y0 = float(q_settled[0, 0]), float(q_settled[0, 1])

    # Apply horizontal force for n_steps. spatial_vector layout is
    # (force_xyz, torque_xyz) — force goes in the first three components.
    f = wp.spatial_vector(force_world.x, force_world.y, force_world.z, 0.0, 0.0, 0.0)
    for _ in range(n_steps):
        wp.launch(
            kernel=apply_force_kernel,
            dim=model.body_count,
            inputs=[state_in.body_f, f],
            device=state_in.body_f.device,
        )
        contacts = model.collide(state_in)
        engine.step(state_in, state_out, ctrl, contacts, dt)
        state_in, state_out = state_out, state_in

    q_final = state_in.body_q.numpy().reshape(-1, 7)
    return x0, y0, float(q_final[0, 0]), float(q_final[0, 1])


def test_anisotropic_box_slides_along_low_friction_axis():
    """Apply equal horizontal force along X and along Y to two otherwise-
    identical anisotropic boxes (axis=(0,1,0), mu_lat=2.0, mu_long=0.0)
    on a frictionless ground (mu=0). The combine rule averages, so the
    resolved cone has mu_x=1.0 (lateral) and mu_y=0.0 (rolling).

    Force F=1 N (below lateral threshold mu_x*m*g=9.81 N, above the
    longitudinal-axis threshold mu_y*m*g=0). Expected:
      - Force along X (longitudinal): box slides freely.
      - Force along Y (lateral): box does not slide.
    """
    mu_lat, mu_long, ground_mu = 2.0, 0.02, 0.0
    m = 1.0
    F = 5.0
    dt = 0.01
    n_steps = 50  # 0.5s of forcing

    # Push along X (longitudinal = mu_y resolved to 0)
    model_x = _build_box_on_ground(mu_lat, mu_long, ground_mu, box_mass=m)
    x0, y0, x1, y1 = _simulate_under_force(model_x, wp.vec3(F, 0.0, 0.0), dt, n_steps)
    dx_long = x1 - x0
    dy_long = y1 - y0

    # Push along Y (lateral = mu_x resolved to 1.0, threshold 9.81 N >> F=1 N)
    model_y = _build_box_on_ground(mu_lat, mu_long, ground_mu, box_mass=m)
    x0, y0, x1, y1 = _simulate_under_force(model_y, wp.vec3(0.0, F, 0.0), dt, n_steps)
    dx_lat = x1 - x0
    dy_lat = y1 - y0

    print(f"\n  Force along X (longitudinal, resolved mu_y~0.01): dx={dx_long:.4f}, dy={dy_long:.4f}")
    print(f"  Force along Y (lateral,      resolved mu_x~1.0): dx={dx_lat:.4f},  dy={dy_lat:.4f}")

    # Longitudinal push: must move noticeably. Integration damping makes the
    # actual displacement smaller than the ideal a*t^2/2; we only assert that
    # the box slides at a level well above noise.
    assert dx_long > 0.01, f"Box should slide along low-friction axis: dx_long={dx_long}"
    assert abs(dy_long) < 0.01, f"Off-axis drift too large: dy_long={dy_long}"

    # Lateral push: F << mu_x*m*g, box sticks.
    assert abs(dy_lat) < 0.005, f"Box should not slide along high-friction axis: dy_lat={dy_lat}"
    assert abs(dx_lat) < 0.005, f"Off-axis drift too large: dx_lat={dx_lat}"

    # The key invariant: large asymmetry between low- and high-friction directions.
    assert dx_long > 10.0 * max(abs(dy_lat), abs(dx_lat), 1e-6), (
        f"Expected strong direction asymmetry; got dx_long={dx_long}, dy_lat={dy_lat}, dx_lat={dx_lat}"
    )


def test_isotropic_fallback_matches_default():
    """A shape with no custom_attributes should still simulate cleanly through
    the isotropic fallback path of resolve_friction_frame."""
    builder = AxionModelBuilder()
    builder.rigid_gap = 0.01
    builder.add_ground_plane()
    body = builder.add_body(mass=1.0, xform=wp.transform(wp.vec3(0, 0, 0.11), wp.quat_identity()))
    builder.add_shape_box(
        body=body, hx=0.1, hy=0.1, hz=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(mu=0.5),
        # No custom_attributes — should default to zero axis + sentinel mu_perp.
    )
    model = builder.finalize_replicated(num_worlds=1, gravity=-9.81)

    x0, y0, x1, y1 = _simulate_under_force(model, wp.vec3(2.0, 0.0, 0.0), 0.01, 50)
    # mu*m*g = 4.905 N; F=2 N below threshold → should stick.
    print(f"  Isotropic mu=0.5, F=2 (< mu*m*g=4.9): dx={x1-x0:.4f}, dy={y1-y0:.4f}")
    assert abs(x1 - x0) < 0.02, f"Box under sub-threshold force should stick: dx={x1-x0}"
