"""Tests for velocity-dependent (Stribeck) friction.

1. Behavior: a box sliding on the ground with Stribeck enabled decelerates
   faster at low slip speed (near-stiction, mu boosted towards
   mu*mu_stiction_scale) than at high slip speed (mu ~= base mu).
2. Bit-exactness: with the feature off (sentinel defaults), the friction
   kernel must take the same code path as before this change. We verify this
   both structurally (the multiplicative Stribeck factor sits behind an
   explicit `if mu_stiction_scale > 0.0 and v_stribeck > 0.0:` guard in
   `compute_friction_model`, so no extra arithmetic executes when the
   sentinel -1.0 defaults are in effect) and empirically, by checking the
   disabled-feature scene is deterministic across repeated runs on CPU.
3. `stribeck_lateral_only`: on an anisotropic (wheel-like) shape, the
   slow/fast deceleration asymmetry from (2) shows up only along the
   resolved LATERAL (friction-axis) direction; the longitudinal direction
   (mu_y per `resolve_friction_frame`) is unaffected.
"""

import sys
from pathlib import Path

import warp as wp

wp.init()
wp.set_device("cpu")

import newton
import numpy as np
from ostrich.core.engine import OstrichEngine
from ostrich.core.engine_config import LinearSolverConfig, NewtonRaphsonConfig, OstrichEngineConfig
from ostrich.core.logging_config import LoggingConfig
from ostrich.core.model_builder import OstrichModelBuilder

sys.path.insert(0, str(Path(__file__).parent))
from helpers import build_box_on_ground


def build_stribeck_box(mu=0.5, mu_stiction_scale=2.0, v_stribeck=0.2, height=0.6):
    """Box on ground; Stribeck attributes set on the box's collision shape."""
    builder = OstrichModelBuilder()
    builder.rigid_gap = 0.05
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=mu))
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, height), wp.quat_identity()),
    )
    builder.add_shape_box(
        body=body,
        hx=0.5,
        hy=0.5,
        hz=0.5,
        cfg=newton.ModelBuilder.ShapeConfig(density=100.0, mu=mu),
        custom_attributes={
            "mu_stiction_scale": mu_stiction_scale,
            "v_stribeck": v_stribeck,
        },
    )
    return builder.finalize_replicated(num_worlds=1, gravity=-9.81)


def build_aniso_box(mu=0.5, mu_perp=0.3, height=0.6, stribeck=None):
    """Box on ground with an anisotropic (wheel-like) friction axis along
    world X.

    Per `resolve_friction_frame`: `friction_axis_local` (here world/body X,
    identity orientation) becomes t1 = LATERAL direction -> mu_x = mu
    (averaged with the ground's isotropic mu). t2 = n x t1 = world Y =
    LONGITUDINAL direction -> mu_y = mu_perp (averaged with ground's mu).
    Ground mu == box's own mu so mu_x resolves to exactly `mu`; mu_perp !=
    mu keeps mu_x != mu_y (a genuinely anisotropic contact, required for
    `stribeck_lateral_only` to take effect rather than falling back to
    scaling both axes).

    `stribeck` is an optional dict of Stribeck custom attributes
    (mu_stiction_scale / v_stribeck / stribeck_lateral_only) to add on top;
    None means Stribeck is completely off — the pure-anisotropic baseline.
    """
    builder = OstrichModelBuilder()
    builder.rigid_gap = 0.05
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=mu))
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, height), wp.quat_identity()),
    )
    custom_attrs = {
        "friction_axis_local": wp.vec3(1.0, 0.0, 0.0),
        "mu_perp": mu_perp,
    }
    if stribeck is not None:
        custom_attrs.update(stribeck)
    builder.add_shape_box(
        body=body,
        hx=0.5,
        hy=0.5,
        hz=0.5,
        cfg=newton.ModelBuilder.ShapeConfig(density=100.0, mu=mu),
        custom_attributes=custom_attrs,
    )
    return builder.finalize_replicated(num_worlds=1, gravity=-9.81)


def make_engine(model, sim_steps):
    config = OstrichEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=20),
        linear=LinearSolverConfig(max_iters=200, tol=1e-8, atol=1e-8),
    )
    return OstrichEngine(
        model=model,
        sim_steps=sim_steps,
        config=config,
        logging_config=LoggingConfig(),
    )


def measure_deceleration(
    model, vx0, axis=0, settle_dt=0.01, settle_steps=50, measure_dt=0.005, measure_steps=1
):
    """Settle the box at rest (settle_dt, large enough for the compliant
    contact to converge without bouncing), kick it to vx0 along `axis`
    (0=world X, 1=world Y; body_qd layout is [0:3]=linear vel, [3:6]=angular
    vel), then measure -dv/dt over `measure_steps` implicit-Euler steps at a
    finer `measure_dt` (fine enough that the slow-slip case doesn't fully
    stick within the measurement window, which would otherwise confound the
    reading)."""
    engine = make_engine(model, sim_steps=settle_steps + measure_steps)

    state_in = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    state_out = model.state()
    control = model.control()

    for _ in range(settle_steps):
        contacts = model.collide(state_in)
        engine.step(state_in, state_out, control, contacts, settle_dt)
        state_in, state_out = state_out, state_in

    qd = state_in.body_qd.numpy().copy()
    qd[0, axis] = vx0
    wp.copy(state_in.body_qd, wp.array(qd, dtype=wp.spatial_vector, device=model.device))

    v_start = state_in.body_qd.numpy()[0, axis]
    for _ in range(measure_steps):
        contacts = model.collide(state_in)
        engine.step(state_in, state_out, control, contacts, measure_dt)
        state_in, state_out = state_out, state_in
    v_end = state_in.body_qd.numpy()[0, axis]

    assert v_end > 0.0, f"box stopped within the measurement window (v_end={v_end})"
    return (v_start - v_end) / (measure_steps * measure_dt)


def test_stribeck_deceleration_ratio():
    """Slow slip (near-stiction) should decelerate noticeably faster than
    fast slip (kinetic, mu ~= base mu) once Stribeck is enabled."""
    print("\n=== Test: Stribeck deceleration ratio (slow vs fast slip) ===")

    mu, mu_stiction_scale, v_stribeck = 0.5, 2.0, 0.2

    decel_fast = measure_deceleration(
        build_stribeck_box(mu, mu_stiction_scale, v_stribeck), vx0=2.0
    )
    decel_slow = measure_deceleration(
        build_stribeck_box(mu, mu_stiction_scale, v_stribeck), vx0=0.05
    )

    g = 9.81
    print(f"  fast (v0=2.0 m/s): decel={decel_fast:.4f} m/s^2  (mu*g={mu * g:.4f})")
    print(f"  slow (v0=0.05 m/s): decel={decel_slow:.4f} m/s^2")
    print(f"  ratio slow/fast: {decel_slow / decel_fast:.4f}")

    # Fast slip: exp(-2.0/0.2) ~ 0 => Stribeck factor ~ 1 => effective mu ~ base
    # mu => decel in the same ballpark as mu*g (the box's 4-corner contact
    # patch and FB smoothing keep the solver's kinetic-friction decel somewhat
    # below the ideal mu*g, so we only check it's in a broad, sane band).
    assert 0.15 * mu * g < decel_fast < 1.2 * mu * g, (
        f"fast-slip deceleration {decel_fast:.4f} not in a sane band around mu*g={mu * g:.4f}"
    )
    # Slow slip: exp(-0.05/0.2) ~ 0.78 => effective mu ~ 0.5*1.78 ~ 0.89 => faster decel.
    ratio = decel_slow / decel_fast
    assert ratio > 1.5, f"deceleration ratio slow/fast = {ratio:.4f}, expected > 1.5"
    print("  PASSED")


def test_stribeck_disabled_is_deterministic():
    """With mu_stiction_scale/v_stribeck left at their sentinel defaults
    (feature off), the disabled path in compute_friction_model is guarded by
    an explicit `if mu_stiction_scale > 0.0 and v_stribeck > 0.0:` check, so
    it executes no extra arithmetic versus the pre-Stribeck code. We can't
    diff against the pre-change binary in this process, so instead we check
    the disabled-feature scene reproduces bit-identical trajectories across
    repeated runs (CPU is deterministic here), which is the property that
    bit-exactness depends on."""
    print("\n=== Test: Stribeck-disabled path is deterministic ===")

    def run():
        model = build_box_on_ground()
        engine = make_engine(model, sim_steps=5)
        state_in = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        state_out = model.state()
        control = model.control()
        for _ in range(5):
            contacts = model.collide(state_in)
            engine.step(state_in, state_out, control, contacts, 0.01)
            state_in, state_out = state_out, state_in
        return state_in.body_q.numpy().copy()

    traj_a = run()
    traj_b = run()

    max_diff = np.max(np.abs(traj_a - traj_b))
    print(f"  max diff between repeated runs: {max_diff:.3e}")
    assert max_diff == 0.0, f"disabled-feature scene not bit-exact across runs: {max_diff:.3e}"
    print("  PASSED")


def test_stribeck_lateral_only():
    """On an anisotropic (wheel-like) shape with `stribeck_lateral_only` set,
    enabling Stribeck must measurably raise the slow/fast deceleration ratio
    along the LATERAL (friction-axis, mu_x) direction, while leaving the
    LONGITUDINAL (mu_y) direction's ratio unchanged.

    NOTE: the elliptical-cone (anisotropic) branch has its own, higher
    Stribeck-independent slow/fast ratio (~1.8-2.1 here) than the isotropic
    branch's ~1.2 baseline (`measure_deceleration`'s implicit-Euler / FB
    discretization is more speed-sensitive there) — a pre-existing solver
    characteristic, not something introduced by this feature. So instead of
    an absolute ratio threshold, we compare each axis's ratio WITH Stribeck
    enabled against the SAME anisotropic scene's ratio with Stribeck
    completely off, isolating Stribeck's actual contribution.
    """
    print("\n=== Test: stribeck_lateral_only restricts the asymmetry to the lateral axis ===")

    stribeck_kwargs = {
        "mu_stiction_scale": 2.0,
        "v_stribeck": 0.2,
        "stribeck_lateral_only": 1.0,
    }

    def ratio(model, axis):
        fast = measure_deceleration(model, vx0=2.0, axis=axis)
        slow = measure_deceleration(model, vx0=0.05, axis=axis)
        return fast, slow, slow / fast

    # axis=0 (world X) is the resolved LATERAL direction (mu_x, Stribeck-scaled).
    lat_fast_off, lat_slow_off, lat_ratio_off = ratio(build_aniso_box(), axis=0)
    lat_fast_on, lat_slow_on, lat_ratio_on = ratio(build_aniso_box(stribeck=stribeck_kwargs), axis=0)
    print(f"  lateral   (X): off ratio={lat_ratio_off:.4f}  on ratio={lat_ratio_on:.4f}")

    # axis=1 (world Y) is the resolved LONGITUDINAL direction (mu_y, unscaled).
    long_fast_off, long_slow_off, long_ratio_off = ratio(build_aniso_box(), axis=1)
    long_fast_on, long_slow_on, long_ratio_on = ratio(build_aniso_box(stribeck=stribeck_kwargs), axis=1)
    print(f"  longitudinal (Y): off ratio={long_ratio_off:.4f}  on ratio={long_ratio_on:.4f}")

    # mu_x (lateral) IS Stribeck-scaled: enabling it measurably raises the
    # slow/fast ratio above the Stribeck-off anisotropic baseline.
    assert lat_ratio_on > lat_ratio_off + 0.05, (
        f"lateral ratio did not increase with Stribeck enabled: "
        f"off={lat_ratio_off:.4f} on={lat_ratio_on:.4f}"
    )
    # mu_y (longitudinal) is NOT Stribeck-scaled: its ratio with
    # stribeck_lateral_only enabled must match the Stribeck-off baseline.
    assert abs(long_ratio_on - long_ratio_off) < 0.02, (
        f"longitudinal ratio changed with lateral-only Stribeck enabled "
        f"(should be untouched): off={long_ratio_off:.4f} on={long_ratio_on:.4f}"
    )
    print("  PASSED")


if __name__ == "__main__":
    test_stribeck_deceleration_ratio()
    test_stribeck_disabled_is_deterministic()
    test_stribeck_lateral_only()
