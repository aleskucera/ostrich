"""In-engine DPCR experiments: the engine analogues of Figures 2, 3 and 4.

Everything runs through the engine's own solver (linear.deflation in
OstrichEngineConfig), on the scenes of engine_scenes.py. Variants:

    Jacobi PCR            the engine today
    DPCR                  joint/control rows eliminated, Jacobi on diag(A_F)
    DPCR + 8 rows         also the 8 contact rows with the largest strength of connection

forward_budget   (Fig. 2)  production budget (cap 16): PCR iterations used, share at the
                           cap, relative residual ‖b − A x‖/‖b‖ reached, Newton iterations
                           per step. Newton solves of the last 80% of steps.
forward_time     (Fig. 3)  GPU time per Newton iteration from the engine's per-component
                           profiler (CUDA graph, one step per replay): the preconditioner
                           phase (for DPCR: row selection, A Z, Cholesky, E⁻¹, diag(A_F))
                           and the PCR solve, at states sampled along the rollout.
adjoint_errors   (Fig. 4)  relative error of trajectory gradients against the same backward
                           pass with dense float64 adjoint solves. Adjoint variants: Jacobi
                           PCR, DPCR (A_SS refactored at the saved state), DPCR reusing the
                           forward factor (adjoint.reuse_deflation_factor).
adjoint_fd       (Fig. 4)  chain only: step-summed gradients against central finite
                           differences, tight forward pass.

Run (from the repo root), e.g.:
    PYTHONPATH=. .venv/bin/python test_scripts/deflation/engine_dpcr.py budget --scene terrain
    ... time --scene marv --worlds 64
    ... adjoint --scene chain
    ... fd
"""

import argparse
import dataclasses

import newton
import numpy as np
import warp as wp

from ostrich import LoggingConfig, OstrichEngineConfig
from ostrich.core.engine_config import (LinearSolverConfig, NewtonRaphsonConfig,
                                        ProfilingConfig)
from ostrich.simulation.trajectory_buffer import TrajectoryBuffer
from test_scripts.deflation.adjoint_trajectory import dense_system
from test_scripts.deflation.engine_scenes import SCENES

VARIANTS = {
    "Jacobi PCR": LinearSolverConfig(),
    "DPCR": LinearSolverConfig(deflation="joint_control"),
    "DPCR + 8 rows": LinearSolverConfig(deflation="joint_control", deflation_strength_rows=8),
}
TIGHT = dict(nr=NewtonRaphsonConfig(max_iters=20), tight_linear=dict(max_iters=200, tol=1e-8, atol=1e-8))
HELD_OUT = 0.2                     # drop the first 20% of steps (landing), as the study's held-out split
N_VEL_DRAWS = 4


def make_engine(model, scene, linear, differentiable=False, tight=False, profiling=ProfilingConfig()):
    kwargs = dict(linear=linear, profiling=profiling)
    if tight:
        kwargs.update(nr=TIGHT["nr"], linear=dataclasses.replace(linear, **TIGHT["tight_linear"]))
    return OstrichEngineConfig(**kwargs).create_engine(
        model=model, sim_steps=scene.steps, logging_config=LoggingConfig(),
        differentiable_simulation=differentiable)


def set_targets(model, control, scene, step, offset=None):
    """Write the scene's targets for `step` (plus an optional per-DOF offset) into every world."""
    field = getattr(control, scene.control_field)
    t = scene.targets(step, model.joint_dof_count // model.world_count)
    if offset is not None:
        t = t + offset
    wp.copy(field, wp.array(np.tile(t, model.world_count).reshape(field.shape), dtype=wp.float32,
                            device=model.device))


def rollout(model, engine, scene, buffer=None, offset=None, on_step=None):
    states = [model.state() for _ in range(scene.steps + 1)]
    newton.eval_fk(model, model.joint_q, model.joint_qd, states[0])
    control = model.control()
    for i in range(scene.steps):
        set_targets(model, control, scene, i, offset)
        states[i].clear_forces()
        engine.step(states[i], states[i + 1], control, model.collide(states[i]), scene.dt)
        if buffer is not None:
            buffer.save_step(i, engine.data, engine.ostrich_contacts)
        if on_step is not None:
            on_step(i)
    return states


# ---------------------------------------------------------------- Fig. 2: production budget

class NewtonSolveRecorder:
    """Wraps engine.cr_solver.solve; records iterations and the true relative residual of
    the Newton solves (the warm-start solves write to a different x and are skipped)."""

    def __init__(self, engine):
        self.engine, self.orig = engine, engine.cr_solver.solve
        self.step, self.records = 0, []                  # (step, iterations, rel. residual)
        self._r = wp.zeros_like(engine.data.rhs)
        engine.cr_solver.solve = self

    def __call__(self, A, b, x, preconditioner, iters, tol, atol, log=False):
        self.orig(A=A, b=b, x=x, preconditioner=preconditioner, iters=iters, tol=tol, atol=atol, log=log)
        if x.ptr != self.engine.data.dconstr_force.full.ptr:
            return
        self.engine.A_op.matvec(x, b, self._r, -1.0, 1.0)        # r = b − A x
        rel = np.linalg.norm(self._r.numpy(), axis=1) / np.maximum(np.linalg.norm(b.numpy(), axis=1), 1e-30)
        self.records.append((self.step, int(self.engine.cr_solver.iter_count.numpy()[0]), float(rel.max())))


def forward_budget(scene_name):
    scene = SCENES[scene_name]
    model, out = scene.build(1), {}
    for name, linear in VARIANTS.items():
        engine = make_engine(model, scene, linear)
        rec, newton_its = NewtonSolveRecorder(engine), []

        def on_step(i):
            rec.step = i + 1
            newton_its.append(int(engine.data.iter_count.numpy()[0]))

        rollout(model, engine, scene, on_step=on_step)
        first = int(HELD_OUT * scene.steps)
        r = np.array([(it, res) for step, it, res in rec.records if step >= first])
        out[name] = dict(iters_p50=float(np.median(r[:, 0])), at_cap=float(np.mean(r[:, 0] >= linear.max_iters)),
                         res_p50=float(np.median(r[:, 1])), res_p90=float(np.percentile(r[:, 1], 90)),
                         newton_mean=float(np.mean(newton_its[first:])), solves=len(r))
    return out


# ---------------------------------------------------------------- Fig. 3: GPU time

def forward_time(scene_name, num_worlds=1, samples=5, replays=20):
    """Median ms per Newton iteration of the preconditioner phase and the PCR solve."""
    scene = SCENES[scene_name]
    model, out = scene.build(num_worlds), {}
    at = sorted({int(f * (scene.steps - 1)) for f in np.linspace(HELD_OUT, 1.0, samples)})
    for name, linear in VARIANTS.items():
        engine = make_engine(model, scene, linear, profiling=ProfilingConfig(mode="per_component"))
        states = rollout(model, engine, scene)
        control = model.control()
        engine.profiler.reset()                          # drop the rollout's records
        for i in at:
            set_targets(model, control, scene, i)
            contacts = model.collide(states[i])
            scratch = model.state()
            with wp.ScopedCapture() as capture:
                engine.step(states[i], scratch, control, contacts, scene.dt)
            for _ in range(replays):
                wp.capture_launch(capture.graph)
                engine.profiler.collect()
        summary = engine.profiler.summary()
        out[name] = {phase: summary[phase]["p50_ms"] for phase in ("preconditioner", "cr_solve", "linear_system")}
    return out


# ---------------------------------------------------------------- Fig. 4: adjoint

def backward(engine, buffer, scene, T, vel_seed, dense=False):
    """dL/d(control) at every step before T, for L = vel_seed · body velocities at T."""
    buffer.zero_grad()
    wp.copy(buffer.body_vel.grad[T], wp.array(vel_seed, dtype=wp.spatial_vector, device=engine.device))
    if dense:
        engine.cr_solver.solve = _dense_solve
    try:
        for i in range(T - 1, -1, -1):
            buffer.load_step(i, engine.data, engine.ostrich_contacts)
            engine.data.zero_gradients()
            engine.step_backward()
            buffer.save_gradients(i, engine.data)
            buffer.save_pose_gradients(i, engine.data)
    finally:
        if dense:
            del engine.cr_solver.solve
    g = getattr(buffer, scene.control_field).grad.numpy()[:T].reshape(T, -1)
    return g[:, list(scene.grad_dofs)].astype(np.float64)


def _dense_solve(A, b, x, preconditioner, iters, tol, atol, log=False):
    Ad, bd, act, _ = dense_system(A, b, 0)
    full = np.zeros(x.shape, dtype=np.float32)
    full[0, act] = np.linalg.solve(Ad.numpy(), bd.numpy())
    x.assign(full)


def _with_reuse(engine, reuse):
    adjoint = dataclasses.replace(engine.config.adjoint, reuse_deflation_factor=reuse)
    engine.config = dataclasses.replace(engine.config, adjoint=adjoint)


def adjoint_errors(scene_name, ends=None):
    """Relative gradient error vs. dense adjoint solves, per adjoint variant."""
    scene = SCENES[scene_name]
    model = scene.build(1)
    ends = ends or (scene.steps // 2, scene.steps)
    cases = [("Jacobi PCR", VARIANTS["Jacobi PCR"], False), ("DPCR", VARIANTS["DPCR"], False),
             ("DPCR, reused factor", VARIANTS["DPCR"], True)]
    out = {}
    for name, linear, reuse in cases:
        engine = make_engine(model, scene, linear, differentiable=True)
        buffer = TrajectoryBuffer(data=engine.data, contacts=engine.ostrich_contacts, dims=engine.dims,
                                  num_steps=scene.steps, device=model.device)
        rollout(model, engine, scene, buffer)
        _with_reuse(engine, reuse)
        rng, errs, its = np.random.default_rng(0), [], []
        shape = engine.data.body_vel_grad.numpy().shape
        for _ in range(N_VEL_DRAWS):
            seed = rng.standard_normal(shape).astype(np.float32)
            for T in ends:
                g = backward(engine, buffer, scene, T, seed)
                ref = backward(engine, buffer, scene, T, seed, dense=True)
                errs.append(np.linalg.norm(g - ref) / max(np.linalg.norm(ref), 1e-30))
        e = np.array(errs)
        out[name] = dict(p50=float(np.median(e)), p90=float(np.percentile(e, 90)), max=float(e.max()),
                         over10=float(np.mean(e > 0.1)), n=len(e))
    return out


def adjoint_fd(ends=(5, 10, 20, 30, 40), eps=(1e-2, 1e-3), max_spread=0.45):
    """Chain: step-summed adjoint gradients vs. central FD (tight forward pass).
    Returns {variant: relative errors of the gradients whose FD is usable}."""
    scene = SCENES["chain"]
    model = scene.build(1)
    dofs = list(scene.grad_dofs)
    rng = np.random.default_rng(0)
    seeds = [rng.standard_normal((1, model.body_count, 6)).astype(np.float32) for _ in range(N_VEL_DRAWS)]

    def losses(states):
        return {(k, T): float(np.sum(s * states[T].body_qd.numpy().reshape(s.shape[1:])))
                for k, s in enumerate(seeds) for T in ends}

    ref_engine = make_engine(model, scene, VARIANTS["Jacobi PCR"], tight=True)
    fd = {}
    for h in eps:
        grads = {}
        for j, dof in enumerate(dofs):
            offset = np.zeros(model.joint_dof_count, dtype=np.float32)
            offset[dof] = h
            lp, lm = losses(rollout(model, ref_engine, scene, offset=offset)), losses(rollout(model, ref_engine, scene, offset=-offset))
            for key in lp:
                grads.setdefault(key, np.zeros(len(dofs)))[j] = (lp[key] - lm[key]) / (2 * h)
        fd[h] = grads
    usable = [key for key in fd[eps[0]] if np.linalg.norm(fd[eps[0]][key]) > 0
              and np.linalg.norm(fd[eps[1]][key] - fd[eps[0]][key]) / np.linalg.norm(fd[eps[0]][key]) < max_spread]

    cases = [("Jacobi PCR", VARIANTS["Jacobi PCR"], False), ("DPCR", VARIANTS["DPCR"], False),
             ("DPCR, reused factor", VARIANTS["DPCR"], True), ("exact adjoint", VARIANTS["Jacobi PCR"], None)]
    out = {}
    for name, linear, reuse in cases:
        engine = make_engine(model, scene, linear, differentiable=True, tight=True)
        buffer = TrajectoryBuffer(data=engine.data, contacts=engine.ostrich_contacts, dims=engine.dims,
                                  num_steps=scene.steps, device=model.device)
        rollout(model, engine, scene, buffer)
        if reuse is not None:
            _with_reuse(engine, reuse)
        # Gradients are taken at the production adjoint budget, as in the engine's backward pass.
        engine.config = dataclasses.replace(engine.config, linear=dataclasses.replace(
            engine.config.linear, max_iters=16, tol=1e-5, atol=1e-5))
        errs = []
        for k, T in usable:
            g = backward(engine, buffer, scene, T, seeds[k], dense=reuse is None).sum(0)
            f = fd[eps[0]][k, T]
            errs.append(np.linalg.norm(g - f) / np.linalg.norm(f))
        out[name] = np.array(errs)
    return out


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("experiment", choices=["budget", "time", "adjoint", "fd"])
    ap.add_argument("--scene", choices=list(SCENES), default="terrain")
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    with wp.ScopedDevice(args.device):
        if args.experiment == "budget":
            result = forward_budget(args.scene)
        elif args.experiment == "time":
            result = forward_time(args.scene, args.worlds)
        elif args.experiment == "adjoint":
            result = adjoint_errors(args.scene)
        else:
            result = {k: dict(n=len(v), p50=float(np.median(v)), max=float(v.max())) for k, v in adjoint_fd().items()}
    print(f"\n{args.experiment} / {args.scene}")
    for name, row in result.items():
        print(f"  {name:22s} " + "  ".join(f"{k} {v:.3g}" for k, v in row.items()))


if __name__ == "__main__":
    main()
