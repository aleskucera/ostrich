"""In-engine trajectory test of the adjoint solve (idea 7.2, prototype).

Question: on the scenes of the deflation study, is the engine's gradient
wrong because the adjoint PCR solve is cut off by the iteration cap, and
does eliminating the joint/control rows repair it at the same budget?

Method: one forward rollout per scene through the real engine (same scene,
dt, drive and default config as dump_linear_systems.py). Then the backward
pass is repeated on that same trajectory with different adjoint solvers.
Only the linear solve inside step_backward is swapped: engine.cr_solver.solve
is replaced for the duration of the backward pass (as SystemCapture does in
dump_linear_systems.py), the dense active-set system is rebuilt from A.data
and solved on the CPU, and the result is written back. Nothing in src/
changes, and everything else in step_backward runs as usual.

Adjoint solvers
  reference    dense solve, float64
  engine       the unmodified GPU PCR at the production budget (16, 1e-5)
  pcr16        CPU port of that PCR, float32 (checks the harness)
  pcr32        the same with cap 32
  elim_diagA   joint/control rows eliminated (unit-vector deflation),
               Jacobi diag(A), cap 16      -- deflation as in Section 4
  elim_schur   the same with the Jacobi diagonal of the Schur complement
               A_F, cap 16                 -- idea 7.2

Losses, seeded at an end step T: chassis x and y position (pose gradient),
and random velocity weights (as in adjoint_accuracy.py). The gradient is
dL/d(joint_target_vel) of the three wheels at every step before T. Errors
are reported for the last h steps before T (h = 1 is one adjoint step, the
offline study's quantity; "all" is the whole horizon).

Run (from the repo root):
    PYTHONPATH=. .venv/bin/python test_scripts/deflation/adjoint_trajectory.py --scene obstacle
    PYTHONPATH=. .venv/bin/python test_scripts/deflation/adjoint_trajectory.py --scene surface
    ... --forward tight     forward pass with the gradient tests' tight solver
    ... --fd                adjoint gradients against finite differences
    ... --scene flat --fd --dt 0.01 --drive-velocity 3
                            control: the gradient tests' flat-ground scene and regime
"""

import argparse

import newton
import numpy as np
import torch
import warp as wp

from examples.helhest.common import create_helhest_model
from ostrich import LoggingConfig, OstrichEngineConfig
from ostrich.core.engine_config import LinearSolverConfig, NewtonRaphsonConfig
from ostrich.core.model_builder import OstrichModelBuilder
from ostrich.simulation.trajectory_buffer import TrajectoryBuffer
from test_scripts.deflation.deflation import deflated_pcr
from test_scripts.dump_linear_systems import (build_helhest_obstacle_model,
                                              build_helhest_surface_model)
from test_scripts.precond_lab import _block_minv, jacobi_apply, pcr
from test_scripts.recycle_probe import pct

BUDGET = dict(iters=16, tol=1e-5, atol=1e-5)     # LinearSolverConfig defaults
TOL = BUDGET["tol"]
END_STEPS = (5, 10, 15, 20, 25)
WINDOWS = (1, 5, None)                           # last h steps before T; None = all
N_VEL_DRAWS = 8
CHASSIS = 0                                      # first link added by create_helhest_model
N_WHEELS = 3                                     # last joint DOFs, as in dump_linear_systems.py


def build_helhest_flat_model():
    """Flat ground, as in tests/differentiable_simulator/test_helhest_gradient.py."""
    builder = OstrichModelBuilder()
    builder.rigid_gap = 0.1
    builder.add_ground_plane()
    create_helhest_model(builder, xform=wp.transform(wp.vec3(0, 0, 0.6), wp.quat_identity()),
                         is_visible=False, control_mode="velocity", k_p=150.0, k_d=0.0,
                         friction_left_right=0.7, friction_rear=0.35)
    return builder.finalize_replicated(num_worlds=1, gravity=-9.81)


SCENES = {"obstacle": build_helhest_obstacle_model, "surface": build_helhest_surface_model,
          "flat": build_helhest_flat_model}


# ---------------------------------------------------------------- adjoint solvers

def dense_system(A, b, n_jc):
    """Dense active-set system of one PCR call (world 0), float64.
    Returns A, b, the active slot indices and the joint/control rows S."""
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
        Minv[6 * k:6 * k + 6, 6 * k:6 * k + 6] = _block_minv(float(minv[k]), Iinv[k])
    Ad = Jm @ Minv @ Jm.T + np.diag(C[act].astype(np.float64) + A.regularization)
    return (torch.tensor(Ad), torch.tensor(b.numpy()[0, act].astype(np.float64)),
            act, np.nonzero(act < n_jc)[0])


def schur_jacobi(A, S):
    """Jacobi on the Schur complement A_F = A_FF − A_FS A_SS⁻¹ A_SF (rows S keep diag(A))."""
    AS = A[:, S]
    dg = torch.diagonal(A) - (AS * torch.linalg.solve(A[S][:, S], AS.T).T).sum(1)
    dg[S] = torch.diagonal(A)[S]
    return lambda r: r / dg


def reference(A, b, S):
    return torch.linalg.solve(A, b), 0, None


def capped_pcr(cap):
    def solve(A, b, S):
        A, b = A.float(), b.float()
        x, it = pcr(A, b, jacobi_apply(A), max_iters=cap, tol=TOL, atol=TOL)
        return x.double(), it, cap
    return solve


def eliminated(cap, schur):
    def solve(A, b, S):
        A, b = A.float(), b.float()
        Z = torch.eye(A.shape[0], dtype=A.dtype)[:, S]
        M = schur_jacobi(A, S) if schur else jacobi_apply(A)
        x, it, _ = deflated_pcr(A, b, Z, M, max_iters=cap, tol=TOL, atol=TOL)
        return x.double(), it, cap
    return solve


SOLVERS = {
    "reference": reference,
    "engine": None,                              # unmodified GPU PCR
    "pcr16": capped_pcr(16),
    "pcr32": capped_pcr(32),
    "elim_diagA": eliminated(16, schur=False),
    "elim_schur": eliminated(16, schur=True),
}


class AdjointSolve:
    """Stands in for engine.cr_solver.solve during a backward pass."""

    def __init__(self, engine):
        self.engine, self.orig = engine, engine.cr_solver.solve
        self.fn, self.log, self.checked = None, [], False

    def __call__(self, A, b, x, preconditioner, iters, tol, atol, log=False):
        if self.fn is None:
            out = self.orig(A=A, b=b, x=x, preconditioner=preconditioner, log=log, **BUDGET)
            self.log.append((int(self.engine.cr_solver.iter_count.numpy()[0]), BUDGET["iters"]))
            return out
        Ad, bd, act, S = dense_system(A, b, self.engine.dims.offset_n)
        if not self.checked:
            self.check_operator(A, Ad, act)
        xs, it, cap = self.fn(Ad, bd, S)
        full = np.zeros(x.shape, dtype=np.float32)
        full[0, act] = xs.numpy()
        x.assign(full)
        self.log.append((it, cap))

    def check_operator(self, A, Ad, act):
        """The dense matrix must reproduce the engine's matrix-free operator."""
        v = np.random.default_rng(0).standard_normal((1, A.data.N_c)).astype(np.float32)
        xw = wp.array(v, dtype=wp.float32, device=A.device)
        zw = wp.zeros_like(xw)
        A.matvec(xw, zw, zw, 1.0, 0.0)
        z = zw.numpy()[0]
        ref = Ad.numpy() @ v[0, act]
        inactive = np.setdiff1d(np.arange(A.data.N_c), act)
        print(f"  check: dense A vs engine operator, rel. diff "
              f"{np.linalg.norm(z[act] - ref) / np.linalg.norm(ref):.1e} "
              f"(float32); inactive rows max |Ax| {np.abs(z[inactive]).max() if len(inactive) else 0.0:.1e}")
        self.checked = True


# ---------------------------------------------------------------- rollout / backward

def make_engine(model, steps, forward):
    cfg = OstrichEngineConfig() if forward == "default" else OstrichEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=20),
        linear=LinearSolverConfig(max_iters=200, tol=1e-8, atol=1e-8))
    return cfg.create_engine(model=model, sim_steps=steps, logging_config=LoggingConfig(),
                             differentiable_simulation=True)


def rollout(model, engine, steps, dt, wheel_vel, buffer=None):
    """Forward pass. wheel_vel: (3,) target velocities, held for all steps."""
    states = [model.state() for _ in range(steps + 1)]
    newton.eval_fk(model, model.joint_q, model.joint_qd, states[0])
    control = model.control()
    targets = np.zeros(control.joint_target_vel.shape[0], dtype=np.float32)
    targets[-N_WHEELS:] = wheel_vel
    wp.copy(control.joint_target_vel, wp.array(targets, dtype=wp.float32, device=model.device))
    for i in range(steps):
        states[i].clear_forces()
        contacts = model.collide(states[i])
        engine.step(states[i], states[i + 1], control, contacts, dt)
        if buffer is not None:
            buffer.save_step(i, engine.data, engine.ostrich_contacts)
    return states


def backward(engine, buffer, hook, fn, T, pose_seed, vel_seed):
    """Backward pass from end step T. Returns dL/d(wheel targets), shape (T, 3)."""
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
        del engine.cr_solver.solve               # back to the class method
    g = buffer.joint_target_vel.grad.numpy()[:T].reshape(T, -1)
    return g[:, -N_WHEELS:].astype(np.float64), hook.log


def losses(engine, rng):
    """(group, name, pose_seed, vel_seed) for every loss."""
    pshape = engine.data.body_pose_grad.numpy().shape
    vshape = engine.data.body_vel_grad.numpy().shape
    out = []
    for axis, name in ((0, "chassis x"), (1, "chassis y")):
        seed = np.zeros(pshape, dtype=np.float32)
        seed[0, CHASSIS, axis] = 1.0
        out.append(("position", name, seed, None))
    for k in range(N_VEL_DRAWS):
        out.append(("velocity", f"random {k}", None,
                    rng.standard_normal(vshape).astype(np.float32)))
    return out


def window_error(g, ref, h):
    """Relative error and cosine of the last h steps (None = all)."""
    a, r = (g, ref) if h is None else (g[-h:], ref[-h:])
    nr = np.linalg.norm(r)
    if nr < 1e-12:
        return None
    return (np.linalg.norm(a - r) / nr,
            float(np.sum(a * r) / (max(np.linalg.norm(a), 1e-300) * nr)))


# ---------------------------------------------------------------- experiments

def compare_solvers(engine, buffer, hook, rng):
    res = {}                                     # (group, solver, h) -> [(rel, cos)]
    its = {s: [] for s in SOLVERS}
    for group, _, pose_seed, vel_seed in losses(engine, rng):
        for T in END_STEPS:
            ref, _ = backward(engine, buffer, hook, reference, T, pose_seed, vel_seed)
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
            print(f"    {name:12s} cap {cap[0]}: p50 {np.median(n):.0f}  max {n.max()}  hit cap {100 * (n >= cap).mean():.0f}%")


def harness_checks(engine, buffer, hook, rng):
    _, _, pose_seed, vel_seed = losses(engine, rng)[0]
    T = END_STEPS[-1]
    ref, _ = backward(engine, buffer, hook, reference, T, pose_seed, vel_seed)
    eng, _ = backward(engine, buffer, hook, None, T, pose_seed, vel_seed)
    cpu, _ = backward(engine, buffer, hook, capped_pcr(16), T, pose_seed, vel_seed)
    def converged(A, b, S):                      # elimination in float64, run to convergence
        Z = torch.eye(A.shape[0], dtype=A.dtype)[:, S]
        x, it, _ = deflated_pcr(A, b, Z, schur_jacobi(A, S), max_iters=500, tol=1e-13, atol=0.0)
        return x, it, 500

    conv, _ = backward(engine, buffer, hook, converged, T, pose_seed, vel_seed)
    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
    print(f"  check: CPU pcr16 vs engine GPU PCR, rel. diff of the gradient {rel(cpu, eng):.1e} "
          f"(errors vs reference: {rel(cpu, ref):.1e} and {rel(eng, ref):.1e})")
    print(f"  check: converged float64 elimination vs reference, rel. diff {rel(conv, ref):.1e}")


def fd_check(model, steps, dt, drive):
    """Adjoint gradients against central finite differences, tight forward pass.
    The control is one array held for all steps, so the FD derivative equals
    the adjoint gradient summed over the steps. FD is taken with two step
    sizes; where they disagree, FD itself is not a usable reference."""
    engine = make_engine(model, steps, "tight")
    buffer = TrajectoryBuffer(data=engine.data, contacts=engine.ostrich_contacts,
                              dims=engine.dims, num_steps=steps, device=model.device)
    hook = AdjointSolve(engine)
    hook.checked = True
    base = np.full(N_WHEELS, drive, dtype=np.float32)
    newton_its, engine_step = [], engine.step

    def counted_step(*a):
        engine_step(*a)
        newton_its.append(int(engine.data.iter_count.numpy()[0]))

    engine.step = counted_step
    rollout(model, engine, steps, dt, base, buffer)
    del engine.step
    cases = losses(engine, np.random.default_rng(0))[:5]     # chassis x, y, three velocity draws
    ends = (4, 6, 8, 10, 12, 15, steps)

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
        for w in range(N_WHEELS):
            vals = []
            for sign in (1.0, -1.0):
                tv = base.copy()
                tv[w] += sign * eps
                vals.append(loss_values(rollout(model, engine, steps, dt, tv)))
            for key in vals[0]:
                fd.setdefault(key, np.zeros(N_WHEELS))[w] = (vals[0][key] - vals[1][key]) / (2 * eps)
        return fd

    fd, fd_small = fd_gradients(1e-2), fd_gradients(1e-3)
    rel = lambda a, b: np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)
    solvers = ("reference", "engine", "elim_schur")
    print("\n  Adjoint (summed over steps) vs finite differences, tight forward pass.")
    print("  Newton iterations per step: " + " ".join(str(n) for n in newton_its))
    print("  'FD spread' = rel. difference between eps 1e-2 and 1e-3; errors are against eps 1e-2.")
    print(f"    {'loss':10s}  T | FD spread | " + " ".join(f"{s:>10s}" for s in solvers))
    for _, name, pose_seed, vel_seed in cases:
        for T in ends:
            f = fd[name, T]
            errs = [rel(backward(engine, buffer, hook, SOLVERS[s], T, pose_seed, vel_seed)[0].sum(0), f)
                    for s in solvers]
            print(f"    {name:10s} {T:2d} | {rel(fd_small[name, T], f):9.3f} | "
                  + " ".join(f"{e:10.3f}" for e in errs))


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--scene", choices=list(SCENES), default="obstacle")
    ap.add_argument("--forward", choices=["default", "tight"], default="default")
    ap.add_argument("--dt", type=float, default=0.03)
    ap.add_argument("--drive-velocity", type=float, default=6.0)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--fd", action="store_true", help="run only the finite-difference check")
    args = ap.parse_args()
    steps = END_STEPS[-1]

    with wp.ScopedDevice(args.device):
        model = SCENES[args.scene]()
        forward = "tight" if args.fd else args.forward
        print(f"\n===== {args.scene}: {steps} steps, dt {args.dt}, forward pass '{forward}' =====")
        if args.fd:
            fd_check(model, steps, args.dt, args.drive_velocity)
            return

        engine = make_engine(model, steps, args.forward)
        buffer = TrajectoryBuffer(data=engine.data, contacts=engine.ostrich_contacts,
                                  dims=engine.dims, num_steps=steps, device=model.device)
        rollout(model, engine, steps, args.dt, np.full(N_WHEELS, args.drive_velocity, np.float32), buffer)
        hook = AdjointSolve(engine)
        harness_checks(engine, buffer, hook, np.random.default_rng(0))
        compare_solvers(engine, buffer, hook, np.random.default_rng(0))


if __name__ == "__main__":
    main()
