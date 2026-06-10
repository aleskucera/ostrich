"""Phase 0: Can Genesis CPU backprop through contacts?

Original GENESIS_BACKWARD_BUG.md disabled both `enable_collision` and
`disable_constraint` because backward hung on GPU. The PR #2742 fix landed
the dynamic-loop ABD refactor but did not touch contact backward. We want
to test CPU specifically — never tested in the original bug report.

Two tests, each in its own subprocess with a timeout:

  drop    — free-floating box falls onto a ground plane under gravity.
            Tests: collision detection backward + constraint solver backward.

  wheels  — Helhest chassis with 3 hinge-jointed wheels rolling on a ground
            plane. Tests: articulated dynamics + revolute joint + contact +
            constraint solver, all in backward.

Each test optimises the initial state for one Adam step. We care only about
whether `loss.backward()` (a) completes within timeout, and (b) populates a
non-zero gradient on the parameter being optimised.

Outcomes:
    PASS    backward completes AND ctrl/init.grad has a non-zero norm
    PARTIAL backward completes but the gradient is None / all zeros
    FAIL    backward raises an exception
    TIMEOUT backward hung past --timeout
"""
import argparse
import multiprocessing
import os
import sys
import tempfile
import time

# Required to import genesis on Linux on certain numpy versions.
import numpy.typing  # noqa: F401

DROP_MJCF = """
<mujoco model="drop_test">
  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="10 10 0.1"
          friction="0.5 0.1 0.01"/>
    <body name="root" pos="0 0 0.5">
      <freejoint name="root_joint"/>
      <inertial mass="1.0" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
      <geom type="box" size="0.1 0.1 0.1" friction="0.5 0.1 0.01"/>
    </body>
  </worldbody>
</mujoco>
"""

WHEELS_MJCF = """
<mujoco model="helhest_on_ground">
  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="10 10 0.1"
          friction="0.5 0.1 0.01"/>
    <body name="chassis" pos="0 0 0.37">
      <freejoint name="base_joint"/>
      <inertial mass="85.0" pos="-0.047 0 0"
                diaginertia="0.6213 0.1583 0.6770"/>
      <geom type="box" pos="-0.047 0 0" size="0.13 0.3 0.09"
            contype="0" conaffinity="0"/>
      <body name="left_wheel" pos="0 0.36 0">
        <joint name="left_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="0.5 0.1 0.01"/>
      </body>
      <body name="right_wheel" pos="0 -0.36 0">
        <joint name="right_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="0.5 0.1 0.01"/>
      </body>
      <body name="rear_wheel" pos="-0.697 0 0">
        <joint name="rear_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="0.5 0.1 0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _write_mjcf(content: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def _drop_worker(q: multiprocessing.Queue) -> None:
    import numpy.typing  # noqa: F401
    import genesis as gs
    import torch

    gs.init(backend=gs.cpu, logging_level="warning")

    path = _write_mjcf(DROP_MJCF)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
            gravity=(0.0, 0.0, -9.81),
            requires_grad=True,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_self_collision=False,
            enable_joint_limit=False,
            disable_constraint=False,
        ),
        show_viewer=False,
    )
    box = scene.add_entity(gs.morphs.MJCF(file=path))
    scene.build()
    q.put(("info", f"n_dofs={box.n_dofs} n_links={box.n_links}"))

    init_pos = gs.tensor([0.0, 0.0, 0.5], requires_grad=True)
    target_pos = torch.tensor([0.2, 0.0, 0.05], device=gs.device)

    scene.reset()
    box.set_pos(init_pos)
    t0 = time.perf_counter()
    for _ in range(50):
        scene.step()
    fwd_ms = (time.perf_counter() - t0) * 1000.0
    q.put(("info", f"Forward OK ({fwd_ms:.0f} ms, 50 steps)"))

    state = box.get_state()
    final_pos = state.pos.squeeze()
    loss = ((final_pos - target_pos) ** 2).sum()
    q.put(("info", f"final_pos={final_pos.tolist()}  loss={float(loss):.4f}"))

    try:
        t0 = time.perf_counter()
        loss.backward()
        bwd_ms = (time.perf_counter() - t0) * 1000.0
        q.put(("info", f"Backward OK ({bwd_ms:.0f} ms)"))
        if init_pos.grad is None:
            q.put(("info", "init_pos.grad = None"))
            q.put(("result", "PARTIAL"))
        else:
            grad_norm = float(init_pos.grad.norm())
            q.put(("info", f"init_pos.grad = {init_pos.grad.tolist()} (norm={grad_norm:.3e})"))
            q.put(("result", "PASS" if grad_norm > 0 else "PARTIAL"))
    except Exception as e:
        q.put(("info", f"ERROR: {type(e).__name__}: {str(e).splitlines()[0]}"))
        q.put(("result", "FAIL"))


def _wheels_worker(q: multiprocessing.Queue) -> None:
    import numpy.typing  # noqa: F401
    import genesis as gs
    import torch

    gs.init(backend=gs.cpu, logging_level="warning")

    path = _write_mjcf(WHEELS_MJCF)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
            gravity=(0.0, 0.0, -9.81),
            requires_grad=True,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_self_collision=False,
            enable_joint_limit=False,
            disable_constraint=False,
        ),
        show_viewer=False,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=path))
    scene.build()
    q.put(("info", f"n_dofs={robot.n_dofs} n_links={robot.n_links}"))

    # All-DOF velocity vector. Indices 6-8 are the three wheel hinges.
    ctrl = gs.tensor([0.0] * robot.n_dofs, requires_grad=True)
    target = torch.tensor([0.5, 0.0, 0.37], device=gs.device)

    scene.reset()
    t0 = time.perf_counter()
    for _ in range(50):
        robot.set_dofs_velocity(ctrl)
        scene.step()
    fwd_ms = (time.perf_counter() - t0) * 1000.0
    q.put(("info", f"Forward OK ({fwd_ms:.0f} ms, 50 steps)"))

    state = robot.get_state()
    chassis_pos = state.pos.squeeze()
    loss = ((chassis_pos - target) ** 2).sum()
    q.put(("info", f"chassis_pos={chassis_pos.tolist()}  loss={float(loss):.4f}"))

    try:
        t0 = time.perf_counter()
        loss.backward()
        bwd_ms = (time.perf_counter() - t0) * 1000.0
        q.put(("info", f"Backward OK ({bwd_ms:.0f} ms)"))
        if ctrl.grad is None:
            q.put(("info", "ctrl.grad = None"))
            q.put(("result", "PARTIAL"))
        else:
            grad_norm = float(ctrl.grad.norm())
            q.put(("info", f"ctrl.grad = {ctrl.grad.tolist()} (norm={grad_norm:.3e})"))
            q.put(("result", "PASS" if grad_norm > 0 else "PARTIAL"))
    except Exception as e:
        q.put(("info", f"ERROR: {type(e).__name__}: {str(e).splitlines()[0]}"))
        q.put(("result", "FAIL"))


def run_test(name: str, worker, timeout_s: int) -> str:
    print(f"\n{'='*60}\nTEST: {name}\n{'='*60}")
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=worker, args=(q,), daemon=True)
    p.start()

    deadline = time.perf_counter() + timeout_s
    verdict = None
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        try:
            kind, msg = q.get(timeout=min(remaining, 1.0))
            print(f"  {msg}")
            if kind == "result":
                verdict = msg
                break
        except Exception:
            pass
        if not p.is_alive():
            break

    p.kill()
    p.join(timeout=3)
    if verdict is None:
        verdict = "TIMEOUT"
        print(f"  TIMEOUT (>{timeout_s}s)")
    print(f"  → {verdict}")
    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["drop", "wheels", "both"], default="both")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    results = {}
    if args.test in ("drop", "both"):
        results["drop"] = run_test(
            "Free body drop on ground (CPU + contacts)",
            _drop_worker, args.timeout,
        )
    if args.test in ("wheels", "both"):
        results["wheels"] = run_test(
            "Articulated Helhest wheels on ground (CPU + contacts)",
            _wheels_worker, args.timeout,
        )

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for k, v in results.items():
        print(f"  {k:30s}: {v}")
