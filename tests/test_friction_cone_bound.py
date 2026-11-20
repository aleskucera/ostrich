"""Regression tests for the Coulomb cone bound.

The unbounded-friction bug: `compute_friction_model` normalised the friction
weight w by the previous iterate's own force (`raw_imp_norm`) instead of the
cone limit (`clamped_imp_norm`). Once a contact saturates (raw >= limit) the
FB gap is pinned at 0, mu drops out of the row, and the fixed point of the
NR iteration is scale-free in lambda_f — friction supplies whatever
tangential load is applied, with no upper bound ("welded" contacts; see
docs/jitter_investigation_findings.md).

`test_fixed_point_respects_coulomb_bound` is the discriminating regression:
it drives the *actual* production `compute_friction_model` through the
single-contact fixed-point iteration the solver performs, with a tangential
load 3x above the cone. With the fix the force converges to mu*f_n and a
residual slip velocity remains; with the bug it converges to the full
applied load and the slip velocity is annihilated.

`test_sliding_box_coulomb_bound` is the integration-level sanity check: a
box kicked into sliding on a ground plane decelerates at ~mu*g and the
solver's tangential force saturates the cone exactly. NOTE: a full-engine
scene cannot discriminate this specific bug — a box at rest sits at the
contact-FB corner (signed_dist=0), where lambda_n is only weakly pinned and
drifts with the applied load, inflating the static budget mu*lambda_n for
any push/torque (a separate, pre-existing quirk); and a freely sliding
contact sits exactly ON the cone, where raw == limit and the buggy and
fixed expressions coincide. The regime where they differ (raw > limit under
sustained forced slip) is only reached in-sim by multi-contact wheel
scenarios such as the Helhest spin-in-place.
"""

import newton
import numpy as np
import warp as wp
from ostrich.constraints.friction_constraint import FRICTION_CONE, compute_friction_model
from ostrich.core.engine import OstrichEngine
from ostrich.core.engine_config import OstrichEngineConfig
from ostrich.core.logging_config import LoggingConfig
from ostrich.core.model_builder import OstrichModelBuilder

wp.init()


# ---------------------------------------------------------------------------
# 1. Kernel-level fixed point of the production friction model
# ---------------------------------------------------------------------------


@wp.kernel
def _coulomb_fixed_point_kernel(
    mu_x: wp.float32,
    mu_y: wp.float32,
    f_n: wp.float32,
    F_ext: wp.float32,
    m: wp.float32,
    dt: wp.float32,
    n_iters: wp.int32,
    lam0: wp.float32,
    # out[0] = converged lambda_t1, out[1] = residual slip velocity
    out: wp.array(dtype=wp.float32),
):
    """Single contact, mass m, normal load f_n, constant tangential push
    F_ext along t1. Iterates exactly the update the engine performs: build w
    from the PREVIOUS iterate's (lambda_f, f_n), then solve the row
    v_t + w * lambda = 0 with v_t = u_free + (dt/m) * lambda.

    `lam0` seeds the iteration. A beyond-limit seed models the warm-start /
    prior-iterate lock-in state: with the bug, `w = |v_t|/|lambda_prev|` is
    scale-free there and the iteration stays at the full applied load; with
    the fix it recovers to the cone."""
    J_t1_0 = wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    J_t2_0 = wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    J_zero = wp.spatial_vector()
    vel_zero = wp.spatial_vector()
    r = 1.0 / m  # effective mass of a translating point contact
    u_free = dt * F_ext / m
    lam = float(lam0)
    for _ in range(n_iters):
        v_slip = u_free + dt * r * lam
        vel0 = wp.spatial_vector(v_slip, 0.0, 0.0, 0.0, 0.0, 0.0)
        v_t, w_x, w_y = compute_friction_model(
            mu_x, mu_y, -1.0, -1.0, -1.0,
            J_t1_0, J_t2_0, J_zero, J_zero,
            vel0, vel_zero,
            wp.vec2(lam, 0.0), f_n, dt, r,
            FRICTION_CONE,
        )
        lam = -u_free / (dt * r + w_x)
    out[0] = lam
    out[1] = u_free + dt * r * lam


def _solve_fixed_point(mu_x, mu_y, f_n=1000.0, load_factor=3.0, m=100.0, dt=0.03,
                       lam0=None):
    F_ext = load_factor * mu_x * f_n  # push along t1, well above the cone
    if lam0 is None:
        # Seed at the applied load: the lock-in state a warm start / prior
        # NR iterate leaves behind once the contact has been overloaded.
        lam0 = -F_ext
    out = wp.zeros(2, dtype=wp.float32)
    wp.launch(
        kernel=_coulomb_fixed_point_kernel,
        dim=1,
        inputs=[mu_x, mu_y, f_n, F_ext, m, dt, 60, lam0],
        outputs=[out],
    )
    lam, v_slip = out.numpy()
    return float(lam), float(v_slip), F_ext


def test_fixed_point_respects_coulomb_bound():
    """Tangential load 3x above the cone: the converged friction force must
    be bounded by mu * f_n (x1.35 tolerance), and a residual slip velocity
    must remain (the mass slides). With the unbounded-friction bug the force
    converges to the full applied load and the slip velocity vanishes."""
    for mu_x, mu_y, label in ((0.8, 0.8, "isotropic"), (0.8, 0.4, "elliptical")):
        lam, v_slip, F_ext = _solve_fixed_point(mu_x, mu_y)
        limit = mu_x * 1000.0
        print(f"\n  {label}: f_t={lam:.1f} N (limit {limit:.0f}, load {F_ext:.0f}), "
              f"residual slip {v_slip:.4f} m/s")
        assert abs(lam) <= 1.35 * limit, (
            f"{label}: friction force {abs(lam):.1f} N exceeds the Coulomb cone "
            f"mu*f_n = {limit:.1f} N (applied load {F_ext:.1f} N)"
        )
        # The load exceeds the cone, so the mass must keep sliding.
        assert v_slip > 0.1, (
            f"{label}: no residual slip under a load 3x the Coulomb limit "
            f"(v_slip={v_slip:.5f} m/s) — friction is unbounded"
        )


# ---------------------------------------------------------------------------
# 2. Integration-level sanity: sliding box decelerates at ~mu*g
# ---------------------------------------------------------------------------

MU = 0.5  # both box and ground -> resolved contact mu = 0.5
BOX_MASS = 1.0
G = 9.81
DT = 0.01


def _build_box_on_ground():
    builder = OstrichModelBuilder()
    builder.rigid_gap = 0.01
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=MU))
    body = builder.add_body(
        mass=BOX_MASS,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.06), wp.quat_identity()),
    )
    # Flat box (low COM) so sliding cannot tip it.
    builder.add_shape_box(
        body=body,
        hx=0.2,
        hy=0.2,
        hz=0.05,
        cfg=newton.ModelBuilder.ShapeConfig(mu=MU),
    )
    return builder.finalize_replicated(num_worlds=1, gravity=-G)


def test_sliding_box_coulomb_bound():
    """Kick a box to 5 m/s on a mu=0.5 plane and slide for 50 steps: the
    deceleration must be ~mu*g (bounded kinetic friction) and the solver's
    tangential force must respect |lam_f| <= mu * lam_n (x1.35 tolerance
    for partial NR convergence)."""
    model = _build_box_on_ground()
    n_steps = 50
    tail = 20  # steps whose forces enter the cone assertion
    v_kick = 5.0

    engine = OstrichEngine(
        model=model, sim_steps=n_steps,
        config=OstrichEngineConfig(), logging_config=LoggingConfig(),
    )
    state_in = model.state()
    state_out = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    ctrl = model.control()

    # Settle on the ground first.
    for _ in range(30):
        contacts = model.collide(state_in)
        engine.step(state_in, state_out, ctrl, contacts, DT)
        state_in, state_out = state_out, state_in

    # Kick into sliding along +x.
    qd = state_in.body_qd.numpy()
    qd[...] = 0.0
    qd.reshape(-1, 6)[0, 0] = v_kick
    wp.copy(
        state_in.body_qd,
        wp.array(qd, dtype=wp.spatial_vector).reshape(state_in.body_qd.shape),
    )

    tangential_sum = 0.0
    normal_sum = 0.0
    for step in range(n_steps):
        contacts = model.collide(state_in)
        engine.step(state_in, state_out, ctrl, contacts, DT)
        state_in, state_out = state_out, state_in

        if step >= n_steps - tail:
            lam_n = engine.data.constr_force.n.numpy()[0]  # (contact_count,)
            lam_f = engine.data.constr_force.f.numpy()[0].reshape(-1, 2)
            tangential_sum += float(np.linalg.norm(lam_f, axis=1).sum())
            normal_sum += float(lam_n.sum())

    v_x = float(state_in.body_qd.numpy().reshape(-1, 6)[0, 0])
    decel = (v_kick - v_x) / (n_steps * DT)

    assert normal_sum > 0.0, "Box lost ground contact"
    ratio = tangential_sum / (MU * normal_sum)
    print(
        f"\n  |lam_f|/(mu*lam_n) over last {tail} steps: {ratio:.3f}, "
        f"final v_x={v_x:.3f} m/s, mean decel={decel:.2f} m/s^2 (mu*g={MU*G:.2f})"
    )

    # Solver-level Coulomb bound (tolerance for partial NR convergence).
    assert ratio <= 1.35, (
        f"Tangential force exceeds the Coulomb cone: |lam_f| = {ratio:.3f} * mu*lam_n"
    )
    # Physical Coulomb bound: kinetic friction cannot exceed ~mu*m*g, so the
    # box must still be sliding after 0.5 s (ideal: 5 - 4.9*0.5 = 2.55 m/s).
    assert decel <= 1.35 * MU * G, (
        f"Sliding deceleration {decel:.2f} m/s^2 exceeds 1.35*mu*g = {1.35*MU*G:.2f}"
    )
    assert v_x > 1.0, f"Box should still be sliding after 0.5 s, got v_x={v_x}"
