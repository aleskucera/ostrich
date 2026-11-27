"""Deflated PCR (linear.deflation="joint_control") against dense solves.

Scene: a velocity-controlled 3-link chain hanging from a fixed pivot next to a box
resting on the ground, so the system has joint, control and contact rows.
"""
import newton
import numpy as np
import pytest
import warp as wp
from ostrich import JointMode
from ostrich.core.engine_config import AdjointConfig, LinearSolverConfig, OstrichEngineConfig
from ostrich.core.logging_config import LoggingConfig
from ostrich.core.model_builder import OstrichModelBuilder
from ostrich.simulation.trajectory_buffer import TrajectoryBuffer

wp.init()

DT = 0.01
LINKS = 3
TIGHT = dict(max_iters=64, tol=1e-6, atol=1e-8)


def build_chain_and_box():
    builder = OstrichModelBuilder()
    builder.add_ground_plane()
    no_collision = newton.ModelBuilder.ShapeConfig(has_shape_collision=False)
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi * 0.5)
    joints, parent = [], -1
    for k in range(LINKS):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.25, hy=0.05, hz=0.05, cfg=no_collision)
        pxf = (wp.transform(p=wp.vec3(0.0, 0.0, 2.0), q=rot) if parent < 0
               else wp.transform(p=wp.vec3(0.25, 0.0, 0.0), q=wp.quat_identity()))
        joints.append(builder.add_joint_revolute(
            parent=parent, child=link, axis=wp.vec3(0.0, 1.0, 0.0), parent_xform=pxf,
            child_xform=wp.transform(p=wp.vec3(-0.25, 0.0, 0.0), q=wp.quat_identity()),
            target_ke=100.0, target_kd=0.0,
            custom_attributes={"joint_dof_mode": [JointMode.TARGET_VELOCITY]}))
        parent = link
    builder.add_articulation(joints)
    box = builder.add_body(xform=wp.transform(wp.vec3(3.0, 0.0, 0.2), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.2, hy=0.2, hz=0.2)
    return builder.finalize_replicated(num_worlds=1, gravity=-9.81, requires_grad=True)


def make_engine(model, steps, linear, adjoint=AdjointConfig()):
    config = OstrichEngineConfig(linear=linear, adjoint=adjoint)
    return config.create_engine(model=model, sim_steps=steps, logging_config=LoggingConfig(),
                                differentiable_simulation=True)


def rollout(model, engine, steps, buffer=None):
    states = [model.state() for _ in range(steps + 1)]
    newton.eval_fk(model, model.joint_q, model.joint_qd, states[0])
    control = model.control()
    targets = np.zeros(control.joint_target_vel.shape, dtype=np.float32)
    targets[:LINKS] = (1.0, -0.5, 0.5)                          # the chain; the box's free joint has none
    wp.copy(control.joint_target_vel, wp.array(targets,
                                               dtype=wp.float32, device=model.device))
    for i in range(steps):
        states[i].clear_forces()
        engine.step(states[i], states[i + 1], control, model.collide(states[i]), DT)
        if buffer is not None:
            buffer.save_step(i, engine.data, engine.ostrich_contacts)
    return states


def dense_system(A, b):
    """Dense float64 A and b of world 0 on the active rows, and the active row indices."""
    d = A.data
    J, bidx = d.J_values.numpy()[0], d.constraint_body_idx.numpy()[0]
    C, minv, Iinv = d.C_values.numpy()[0], d.body_inv_mass.numpy()[0], d.world_inv_inertia.numpy()[0]
    act = np.nonzero(d.constraint_active_mask.numpy()[0] > 0.0)[0]
    Jm = np.zeros((len(act), 6 * d.N_b))
    for r, c in enumerate(act):
        for slot in (0, 1):
            body = int(bidx[c, slot])
            if body >= 0:
                Jm[r, 6 * body:6 * body + 6] += J[c, slot]
    Minv = np.zeros((6 * d.N_b, 6 * d.N_b))
    for k in range(d.N_b):
        Minv[6 * k:6 * k + 3, 6 * k:6 * k + 3] = minv[k] * np.eye(3)
        Minv[6 * k + 3:6 * k + 6, 6 * k + 3:6 * k + 6] = Iinv[k]
    Ad = Jm @ Minv @ Jm.T + np.diag(C[act].astype(np.float64) + A.regularization)
    return Ad, b.numpy()[0, act].astype(np.float64), act


class SolveRecorder:
    """Wraps engine.cr_solver.solve and checks every solve against a dense solve."""

    def __init__(self, engine):
        self.engine, self.orig, self.records = engine, engine.cr_solver.solve, []
        engine.cr_solver.solve = self

    def __call__(self, A, b, x, preconditioner, iters, tol, atol, log=False):
        self.orig(A=A, b=b, x=x, preconditioner=preconditioner, iters=iters, tol=tol, atol=atol, log=log)
        Ad, bd, act = dense_system(A, b)
        xr = np.linalg.solve(Ad, bd)
        xs = x.numpy()[0, act].astype(np.float64)
        n_contact = int(np.sum(act >= self.engine.dims.offset_n))
        self.records.append((np.linalg.norm(xs - xr) / max(np.linalg.norm(xr), 1e-30), n_contact))


@pytest.fixture(scope="module")
def model():
    return build_chain_and_box()


@pytest.mark.parametrize("strength_rows", [0, 4])
def test_dpcr_matches_dense_solve(model, strength_rows):
    engine = make_engine(model, 10, LinearSolverConfig(deflation="joint_control",
                                                       deflation_strength_rows=strength_rows, **TIGHT))
    rec = SolveRecorder(engine)
    rollout(model, engine, 10)
    errors = np.array([e for e, _ in rec.records])
    assert any(n > 0 for _, n in rec.records), "scene must produce contact rows"
    assert errors.max() < 1e-3, f"max rel. error vs dense solve {errors.max():.1e}"
    assert int(engine.deflation.factor_failure.numpy().max()) == 0


def test_strength_rows_are_active_contact_rows(model):
    engine = make_engine(model, 10, LinearSolverConfig(deflation="joint_control", deflation_strength_rows=4))
    rollout(model, engine, 10)
    n_fixed = engine.dims.offset_n
    rows = engine.data.deflation_rows.numpy()[0]
    mask = engine.data.constr_active_mask.full.numpy()[0]
    assert np.array_equal(rows[:n_fixed], np.arange(n_fixed))
    picked = rows[n_fixed:][rows[n_fixed:] >= 0]
    assert len(picked) > 0 and len(set(picked)) == len(picked)
    assert np.all(picked >= n_fixed) and np.all(mask[picked] > 0)


def test_schur_jacobi_diagonal(model):
    engine = make_engine(model, 10, LinearSolverConfig(deflation="joint_control"))
    rollout(model, engine, 10)
    engine.preconditioner.update()
    Ad, _, act = dense_system(engine.A_op, engine.data.rhs)
    S = np.nonzero(act < engine.dims.offset_n)[0]
    F = np.nonzero(act >= engine.dims.offset_n)[0]
    assert len(F) > 0
    schur = np.diag(Ad)[F] - np.einsum("ij,ji->i", Ad[np.ix_(F, S)], np.linalg.solve(Ad[np.ix_(S, S)], Ad[np.ix_(S, F)]))
    p_inv = engine.preconditioner._P_inv_diag.numpy()[0, act[F]]
    np.testing.assert_allclose(1.0 / p_inv - 1e-6, schur, rtol=1e-3)


@pytest.mark.parametrize("strength_rows", [0, 4])
def test_tile_factorization_matches_loop(model, strength_rows):
    engine = make_engine(model, 10, LinearSolverConfig(deflation="joint_control",
                                                       deflation_strength_rows=strength_rows))
    rollout(model, engine, 10)
    df = engine.deflation
    if df.factor_mode == "loop":
        pytest.skip("no tile factorization on this device (CPU, no libmathdx, or too little shared memory)")
    results = {}
    for mode in (df.factor_mode, "loop"):
        df.factor_mode = mode
        df.update()
        results[mode] = {n: getattr(df, n).numpy().copy() for n in ("L", "Linv", "Einv")}
        assert int(df.factor_failure.numpy().max()) == 0
    loop = results.pop("loop")
    tile = results.popitem()[1]
    for n in ("L", "Linv", "Einv"):
        np.testing.assert_allclose(np.tril(tile[n]) if n == "L" else tile[n],
                                   np.tril(loop[n]) if n == "L" else loop[n], rtol=1e-4, atol=1e-5, err_msg=n)


@pytest.mark.skipif(not wp.get_device("cuda:0").is_cuda if wp.get_cuda_device_count() else True,
                    reason="needs a CUDA device")
def test_step_under_graph_capture(model):
    engine = make_engine(model, 2, LinearSolverConfig(deflation="joint_control", deflation_strength_rows=4))
    state_in, state_out = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    control = model.control()
    contacts = model.collide(state_in)
    engine.step(state_in, state_out, control, contacts, DT)       # compile kernels before capture
    with wp.ScopedCapture() as capture:
        engine.step(state_in, state_out, control, contacts, DT)
    wp.capture_launch(capture.graph)
    assert np.all(np.isfinite(state_out.body_qd.numpy()))


def _backward(engine, buffer, steps, vel_seed):
    buffer.zero_grad()
    wp.copy(buffer.body_vel.grad[steps], wp.array(vel_seed, dtype=wp.spatial_vector, device=engine.device))
    for i in range(steps - 1, -1, -1):
        buffer.load_step(i, engine.data, engine.ostrich_contacts)
        engine.data.zero_gradients()
        engine.step_backward()
        buffer.save_gradients(i, engine.data)
        buffer.save_pose_gradients(i, engine.data)
    return buffer.joint_target_vel.grad.numpy()[:steps].reshape(steps, -1).astype(np.float64)


@pytest.mark.parametrize("reuse", [False, True])
def test_adjoint_matches_dense_adjoint(model, reuse):
    """The DPCR adjoint gradient equals the gradient with dense adjoint solves; reusing the
    forward factor stays close (it is one Newton update stale)."""
    steps = 8
    engine = make_engine(model, steps, LinearSolverConfig(deflation="joint_control", **TIGHT),
                         AdjointConfig(reuse_deflation_factor=reuse))
    buffer = TrajectoryBuffer(data=engine.data, contacts=engine.ostrich_contacts, dims=engine.dims,
                              num_steps=steps, device=model.device)
    rollout(model, engine, steps, buffer)
    seed = np.random.default_rng(0).standard_normal(engine.data.body_vel_grad.numpy().shape).astype(np.float32)
    g = _backward(engine, buffer, steps, seed)

    def dense_solve(A, b, x, preconditioner, iters, tol, atol, log=False):
        Ad, bd, act = dense_system(A, b)
        full = np.zeros(x.shape, dtype=np.float32)
        full[0, act] = np.linalg.solve(Ad, bd)
        x.assign(full)

    engine.cr_solver.solve = dense_solve
    try:
        g_ref = _backward(engine, buffer, steps, seed)
    finally:
        del engine.cr_solver.solve
    rel = np.linalg.norm(g - g_ref) / np.linalg.norm(g_ref)
    assert rel < 1e-3, f"rel. gradient error {rel:.1e}"
