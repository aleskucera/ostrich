"""Minimal reproducer for a Linux+CUDA backward bug on PR #2742.

PR #2742 fixes the original ABD backward hang reported in #2537 — backward
now completes and produces correct gradients on the **CPU** backend, including
the articulated freejoint+hinge case that previously hung indefinitely.

However, on the **CUDA GPU** backend, `kernel_forward_velocity.grad` fails to
launch with `CUDA_ERROR_ILLEGAL_ADDRESS` for *any* rigid-body backward — even
a single free-floating body with no contacts and no constraint solver. The
forward pass runs fine; the failure occurs at the first gradient kernel launch.

Note: the only GPU-flagged backend in `.github/workflows/generic.yml` is
`macos-15` (Apple Metal). There is no `ubuntu-*` + `gpu` row in the matrix,
so the Linux+CUDA backward path is not exercised by Genesis CI — which is
consistent with this bug being introduced and not caught by the PR's tests.

Usage:
    python pr2742_cuda_repro.py --backend cpu   # works, prints ctrl.grad
    python pr2742_cuda_repro.py --backend gpu   # fails at loss.backward()

Environment where this was observed:
    OS:        Arch Linux, kernel 6.19.11
    GPU:       NVIDIA RTX A500 Laptop
    CUDA:      12.8, driver 590.48.01
    PyTorch:   2.9.1+cu128
    Python:    3.12.12
    quadrants: 0.7.5 (from PyPI)
    Genesis:   PR #2742 head 4696cc8

Reproducer environment notes:
  - The PR's pyproject.toml currently pins `quadrants==0.7.4`, but the code
    requires 0.7.5+ (`qd.Tensor` was added in 0.7.5). With 0.7.4, import fails
    in genesis/utils/misc.py:416 (`qd.Tensor` AttributeError).
  - With CUDA_LAUNCH_BLOCKING=1, the failure point becomes synchronous and
    points at:
        rigid_solver.py:1282 → kernel_forward_velocity.grad(...)
        quadrants/lang/kernel.py:572 → prog.launch_kernel(...)
        RuntimeError: CUDA Error CUDA_ERROR_ILLEGAL_ADDRESS at cuLaunchKernel
"""
import argparse
import os
import tempfile

import genesis as gs
import torch

MJCF = """
<mujoco model="free_body_repro">
  <worldbody>
    <body name="root" pos="0 0 0">
      <freejoint name="root_joint"/>
      <inertial mass="1.0" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
      <geom type="box" size="0.1 0.1 0.1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    gs.init(backend=getattr(gs, args.backend), logging_level="warning")

    fd, mjcf_path = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd, "w") as f:
        f.write(MJCF)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, 0.0),
            requires_grad=True,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_self_collision=False,
            enable_joint_limit=False,
            disable_constraint=True,
        ),
        show_viewer=False,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=mjcf_path))
    scene.build()

    print(f"Backend: {args.backend}, n_dofs={robot.n_dofs}, n_links={robot.n_links}")

    # 6 DOFs from the freejoint: 3 linear + 3 angular velocity.
    ctrl = gs.tensor([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
    target = torch.tensor([0.05, 0.0, 0.0], device=gs.device)

    scene.reset()
    for _ in range(args.steps):
        robot.set_dofs_velocity(ctrl)
        scene.step()
    print(f"Forward OK ({args.steps} steps)")

    state = robot.get_state()
    loss = torch.nn.functional.mse_loss(state.pos.squeeze(), target)
    print(f"loss = {float(loss):.6f}")

    print("loss.backward() ...")
    loss.backward()
    print(f"Backward OK")
    print(f"ctrl.grad = {ctrl.grad}")


if __name__ == "__main__":
    main()
