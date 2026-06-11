## Linux + CUDA backward regression in current PR head

First — thanks for landing this. On the **CPU** backend, this PR fully fixes the original ABD hang from #2537: backward not only completes, it produces correct gradients for the articulated freejoint+hinge case that previously hung indefinitely. That part works great. 🎉

However, on the **CUDA GPU** backend, *every* rigid-body backward fails to launch with `CUDA_ERROR_ILLEGAL_ADDRESS` — even a single free-floating body with no contacts and no constraint solver. The forward pass runs fine; the failure occurs at the first gradient kernel launch (`kernel_forward_velocity.grad`).

### Minimal reproducer

<details>
<summary><code>pr2742_repro.py</code> — single free body, 5 steps, 50 LoC</summary>

```python
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
            dt=0.01, gravity=(0.0, 0.0, 0.0), requires_grad=True,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False, enable_self_collision=False,
            enable_joint_limit=False, disable_constraint=True,
        ),
        show_viewer=False,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=mjcf_path))
    scene.build()

    print(f"Backend: {args.backend}, n_dofs={robot.n_dofs}, n_links={robot.n_links}")

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
```
</details>

### Output

**`python pr2742_repro.py --backend cpu`** — works, gradient is correct (only x-component populated since target is offset only in x):
```
Backend: cpu, n_dofs=6, n_links=1
Forward OK (5 steps)
loss = 0.000675
Backward OK
ctrl.grad = tensor([-0.0015,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000])
```

**`python pr2742_repro.py --backend gpu`** — fails:
```
Backend: gpu, n_dofs=6, n_links=1
Forward OK (5 steps)
loss = 0.000675
loss.backward() ...
RuntimeError: CUDA Error CUDA_ERROR_ILLEGAL_ADDRESS at cuMemcpyDtoH_v2
```

### Synchronous traceback under `CUDA_LAUNCH_BLOCKING=1`

```
File ".../genesis/engine/solvers/rigid/rigid_solver.py", line 1282, in substep_pre_coupling_grad
    kernel_forward_velocity.grad(
        envs_idx=envs_idx,
        links_state=self.links_state,
        ...
        is_backward=True,
    )
File ".../quadrants/lang/kernel.py", line 572, in launch_kernel
    prog.launch_kernel(compiled_kernel_data, launch_ctx)
RuntimeError: CUDA Error CUDA_ERROR_ILLEGAL_ADDRESS at cuLaunchKernel
```

The crash is at `cuLaunchKernel` itself (not at a later sync), which suggests the launch arguments for `kernel_forward_velocity.grad` are bad on CUDA specifically — out-of-range indices, dangling buffer pointer, or wrong grid shape. CPU codegen for the same kernel works correctly with the same MJCF and the same `gs.tensor` inputs, so it looks like a backend-codegen issue rather than a logic bug in the kernel itself.

### Environment

| | |
|---|---|
| OS | Arch Linux, kernel 6.19.11 |
| GPU | NVIDIA RTX A500 Laptop |
| CUDA | 12.8, driver 590.48.01 |
| PyTorch | 2.9.1+cu128 |
| Python | 3.12.12 |
| `quadrants` | 0.7.5 (from PyPI) |
| Genesis | this PR head, `4696cc8` |

### CI coverage observation

Looking at `.github/workflows/generic.yml`, the only `gpu`-flagged row in the matrix is `macos-15` (Apple Metal); there's no `ubuntu-*` + `gpu` row. So the Linux+CUDA backward path isn't exercised by CI on this PR — which would explain how a regression here could slip through despite the test suite passing. Happy to test patches on this hardware if it helps.

### Minor: `quadrants` version pin

`pyproject.toml` currently pins `quadrants==0.7.4`, but the PR code uses `qd.Tensor` (e.g. `genesis/utils/misc.py:416`, `forward_kinematics.py:1213`, `noslip.py:87/382`, `base_solver.py:72`), which was added in 0.7.5. With 0.7.4 installed, `import genesis` fails with `AttributeError: module 'quadrants' has no attribute 'Tensor'`. Probably worth bumping to `quadrants>=0.7.5` before merge.
