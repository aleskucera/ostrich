"""Scene check for idea 7.2: contact-free articulated chains.

Question: is there a scene in which (a) the converged adjoint agrees with
finite differences, so the engine's linearization is right, and (b) the
engine's adjoint PCR is still cut off by its iteration cap? Only there can a
better adjoint solve give a better gradient. In the Helhest scenes (a) fails
once the contacts carry load (research_findings.md, "Why the adjoint
disagrees with FD"). A chain without contacts keeps the joints, which cause
the amplification, and drops the contact linearization.

Scene: N links in a row along x, revolute joints about y, pivot fixed at the
world, no collision shapes, gravity. Every joint is velocity-controlled
(target_ke = KE). Link masses can alternate by a factor --mass-ratio, which
makes A stiffer for Jacobi PCR. The chain starts horizontal and swings down.

Same harness as adjoint_trajectory.py: one rollout, then the backward pass is
repeated with the linear solve inside step_backward swapped. Gradient:
dL/d(joint target velocity) of every joint at every step before T. Losses: 8
random velocity weights and the y / z position of the tip link at T
(the joint axis is world x, so the chain swings in the y-z plane).

Run (from the repo root):
    PYTHONPATH=. .venv/bin/python test_scripts/deflation/adjoint_articulated.py --links 6
    ... --mass-ratio 100     alternating heavy/light links
    ... --fd                 converged adjoint vs finite differences
"""

import argparse

import newton
import numpy as np
import warp as wp

from ostrich.core.model_builder import OstrichModelBuilder
from ostrich.core.types import JointMode
from ostrich.simulation.trajectory_buffer import TrajectoryBuffer
from test_scripts.deflation.adjoint_trajectory import (SOLVERS, AdjointSolve, eliminated,
                                                       make_engine, window_error)
from test_scripts.recycle_probe import pct

END_STEPS = (5, 10, 20, 30, 40)
WINDOWS = (1, None)
N_VEL_DRAWS = 8
HX, HYZ = 0.25, 0.05                             # half extents of a link
KE = 100.0
N_JOINT_ROWS = None                              # engine.dims.offset_ctrl, set in main


def joints_only(cap, schur):
    """Eliminate only the joint rows; the control rows stay in F and are iterated on,
    as contact rows would be. Without contacts, S = joint/control rows is all rows."""
    inner = eliminated(cap, schur)

    def solve(A, b, S):
        assert len(S) >= N_JOINT_ROWS            # joint rows are always active and come first
        return inner(A, b, S[:N_JOINT_ROWS])
    return solve


SOLVERS = {**SOLVERS, "joints_diagA": joints_only(16, schur=False),
           "joints_schur": joints_only(16, schur=True)}


def build_chain(links, mass_ratio):
    builder = OstrichModelBuilder()
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi * 0.5)
    joints, parent = [], -1
    for k in range(links):
        rho = 1000.0 * (mass_ratio if k % 2 else 1.0)
        link = builder.add_link()
        builder.add_shape_box(link, hx=HX, hy=HYZ, hz=HYZ,
                              cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False, density=rho))
        if parent < 0:
            pxf = wp.transform(p=wp.vec3(0.0, 0.0, 5.0), q=rot)
        else:
            pxf = wp.transform(p=wp.vec3(HX, 0.0, 0.0), q=wp.quat_identity())
        joints.append(builder.add_joint_revolute(
            parent=parent, child=link, axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=pxf, child_xform=wp.transform(p=wp.vec3(-HX, 0.0, 0.0), q=wp.quat_identity()),
            target_ke=KE, target_kd=0.0, label=f"j{k}",
            custom_attributes={"joint_dof_mode": [JointMode.TARGET_VELOCITY]}))
        parent = link
    builder.add_articulation(joints, label="chain")
    return builder.finalize_replicated(num_worlds=1, gravity=-9.81, requires_grad=True)


def targets_for(model, links, seed=0):
    """Fixed, moderate target velocities, one per joint."""
    return np.random.default_rng(seed).uniform(-1.0, 1.0, links).astype(np.float32)


def rollout(model, engine, steps, dt, targets, buffer=None, newton_its=None):
    states = [model.state() for _ in range(steps + 1)]
    newton.eval_fk(model, model.joint_q, model.joint_qd, states[0])
    control = model.control()
    wp.copy(control.joint_target_vel, wp.array(targets.reshape(control.joint_target_vel.shape),
                                               dtype=wp.float32, device=model.device))
    for i in range(steps):
        states[i].clear_forces()
        engine.step(states[i], states[i + 1], control, model.collide(states[i]), dt)
        if newton_its is not None:
            newton_its.append(int(engine.data.iter_count.numpy()[0]))
        if buffer is not None:
            buffer.save_step(i, engine.data, engine.ostrich_contacts)
    return states


def backward(engine, buffer, hook, fn, T, pose_seed, vel_seed):
    """Backward pass from end step T. Returns dL/d(joint targets), shape (T, links)."""
    dev = engine.device
    buffer.zero_grad()
    if pose_seed is not None:
        wp.copy(buffer.body_pose.grad[T], wp.array(pose_seed, dtype=wp.transform, device=dev))
    if vel_seed is not None:
        wp.copy(buffer.body_vel.grad[T], wp.array(vel_seed, dtype=wp.spatial_vector, device=dev))
    hook.fn, hook.log = fn, []
    engine.cr_solver.solve = hook
    try:
        for i in range(T - 1, -1, -1):
            buffer.load_step(i, engine.data, engine.ostrich_contacts)
            engine.data.zero_gradients()
            engine.step_backward()
            buffer.save_gradients(i, engine.data)
            buffer.save_pose_gradients(i, engine.data)
    finally:
        del engine.cr_solver.solve
    return buffer.joint_target_vel.grad.numpy()[:T].reshape(T, -1).astype(np.float64), hook.log


def losses(engine, links, rng):
    pshape = engine.data.body_pose_grad.numpy().shape
    vshape = engine.data.body_vel_grad.numpy().shape
    out = []
    for axis, name in ((1, "tip y"), (2, "tip z")):   # the chain swings in the y-z plane
        seed = np.zeros(pshape, dtype=np.float32)
        seed[0, links - 1, axis] = 1.0
        out.append(("position", name, seed, None))
    for k in range(N_VEL_DRAWS):
        out.append(("velocity", f"random {k}", None, rng.standard_normal(vshape).astype(np.float32)))
    return out


def compare_solvers(engine, buffer, hook, links, steps):
    res, its = {}, {s: [] for s in SOLVERS}
    ends = [T for T in END_STEPS if T <= steps]
    for group, _, pose_seed, vel_seed in losses(engine, links, np.random.default_rng(0)):
        for T in ends:
            ref, _ = backward(engine, buffer, hook, SOLVERS["reference"], T, pose_seed, vel_seed)
            for name, fn in SOLVERS.items():
                if name == "reference":
                    continue
                g, log = backward(engine, buffer, hook, fn, T, pose_seed, vel_seed)
                its[name] += log
                for h in WINDOWS:
                    e = window_error(g, ref, h)
                    if e is not None:
                        res.setdefault((group, name, h), []).append(e)

    for group in ("position", "velocity"):
        for h in WINDOWS:
            n = len(res.get((group, "engine", h), []))
            label = "all steps" if h is None else f"last {h} step(s)"
            print(f"\n  {group} loss, {label} before T  ({n} gradients)")
            print(f"    {'adjoint solver':12s} | rel.err p50 / p90 / max    | >10%   >1% | min cos")
            for name in SOLVERS:
                v = np.array(res.get((group, name, h), []))
                if not len(v):
                    continue
                e, c = v[:, 0], v[:, 1]
                print(f"    {name:12s} | {pct(e, 50):7.1e} / {pct(e, 90):7.1e} / {e.max():7.1e} "
                      f"| {100 * (e > 0.1).mean():4.0f}% {100 * (e > 0.01).mean():4.0f}% | {c.min():7.4f}")

    print("\n  adjoint solves: iterations and share that reached the cap")
    for name, log in its.items():
        if log:
            n, cap = np.array(log).T
            print(f"    {name:12s} cap {cap[0]}: p50 {np.median(n):.0f}  max {n.max()}  "
                  f"hit cap {100 * (n >= cap).mean():.0f}%")


FD_SOLVERS = ("reference", "engine", "joints_schur")


def fd_errors(model, engine, links, steps, dt):
    """Adjoint (summed over steps) vs central FD, per joint target. Tight forward pass.
    Returns the Newton iteration counts and rows (loss, T, FD spread, {solver: rel. error})."""
    global N_JOINT_ROWS
    N_JOINT_ROWS = int(engine.dims.offset_ctrl)
    buffer = TrajectoryBuffer(data=engine.data, contacts=engine.ostrich_contacts,
                              dims=engine.dims, num_steps=steps, device=model.device)
    hook = AdjointSolve(engine)
    hook.checked = True
    base, newton_its = targets_for(model, links), []
    rollout(model, engine, steps, dt, base, buffer, newton_its)
    cases = losses(engine, links, np.random.default_rng(0))[:5]
    ends = [T for T in END_STEPS if T <= steps]

    def loss_values(states):
        out = {}
        for _, name, pose_seed, vel_seed in cases:
            for T in ends:
                q, qd = states[T].body_q.numpy(), states[T].body_qd.numpy()
                out[name, T] = (float(np.sum(pose_seed.reshape(q.shape) * q)) if pose_seed is not None
                                else float(np.sum(vel_seed.reshape(qd.shape) * qd)))
        return out

    def fd_gradients(eps):
        fd = {}
        for j in range(links):
            vals = []
            for sign in (1.0, -1.0):
                tv = base.copy()
                tv[j] += sign * eps
                vals.append(loss_values(rollout(model, engine, steps, dt, tv)))
            for key in vals[0]:
                fd.setdefault(key, np.zeros(links))[j] = (vals[0][key] - vals[1][key]) / (2 * eps)
        return fd

    fd, fd_small = fd_gradients(1e-2), fd_gradients(1e-3)
    rel = lambda a, b: np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)
    rows = []
    for _, name, pose_seed, vel_seed in cases:
        for T in ends:
            f = fd[name, T]
            errs = {s: rel(backward(engine, buffer, hook, SOLVERS[s], T, pose_seed, vel_seed)[0].sum(0), f)
                    for s in FD_SOLVERS}
            # FD = 0 (e.g. the tip has not moved yet) is no reference: mark it unusable
            spread = rel(fd_small[name, T], f) if np.linalg.norm(f) > 0.0 else np.inf
            rows.append((name, T, spread, errs))
    return newton_its, rows


def fd_check(model, engine, links, steps, dt):
    newton_its, rows = fd_errors(model, engine, links, steps, dt)
    print("\n  Adjoint (summed over steps) vs finite differences, tight forward pass.")
    print("  Newton iterations per step: " + " ".join(str(n) for n in newton_its))
    print("  'FD spread' = rel. difference between eps 1e-2 and 1e-3; errors are against eps 1e-2.")
    print(f"    {'loss':10s}  T | FD spread | " + " ".join(f"{s:>10s}" for s in FD_SOLVERS))
    for name, T, spread, errs in rows:
        print(f"    {name:10s} {T:2d} | {spread:9.3f} | " + " ".join(f"{errs[s]:10.3f}" for s in FD_SOLVERS))


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--links", type=int, default=6)
    ap.add_argument("--mass-ratio", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=END_STEPS[-1])
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--forward", choices=["default", "tight"], default="default")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--fd", action="store_true", help="run only the finite-difference check")
    args = ap.parse_args()

    with wp.ScopedDevice(args.device):
        model = build_chain(args.links, args.mass_ratio)
        forward = "tight" if args.fd else args.forward
        print(f"\n===== chain: {args.links} links, mass ratio {args.mass_ratio:g}, {args.steps} steps, "
              f"dt {args.dt}, forward pass '{forward}' =====")
        engine = make_engine(model, args.steps, forward)
        global N_JOINT_ROWS
        N_JOINT_ROWS = int(engine.dims.offset_ctrl)
        if args.fd:
            fd_check(model, engine, args.links, args.steps, args.dt)
            return
        buffer = TrajectoryBuffer(data=engine.data, contacts=engine.ostrich_contacts,
                                  dims=engine.dims, num_steps=args.steps, device=model.device)
        newton_its = []
        rollout(model, engine, args.steps, args.dt, targets_for(model, args.links), buffer, newton_its)
        print("  Newton iterations per step: " + " ".join(str(n) for n in newton_its))
        compare_solvers(engine, buffer, AdjointSolve(engine), args.links, args.steps)


if __name__ == "__main__":
    main()
