"""Ball throw optimization using Genesis (Taichi-based differentiable physics).

Comparable to examples/comparison/ball_throw/ball_throw_axion.py.

Optimizes the initial velocity of a free-floating ball to reach a target
position after T steps of ballistic flight.

Gradient flow:
  - set_dofs_velocity(v0, dofs_idx_local=[0,1,2]) seeds the initial velocity.
  - scene.step() runs forward; Genesis registers each step as a custom PyTorch
    autograd Function via the PyTorch-Taichi bridge.
  - ball.get_state().pos returns a tensor with a grad_fn connected to v0.
  - loss.backward() propagates through the full rollout back to v0.grad.

IMPORTANT: use ball.get_state().pos, NOT ball.get_links_pos()[0].
  get_links_pos() returns a plain detached tensor with no grad_fn.
  get_state().pos returns a gs.Tensor that is part of the autograd graph.

Previous approach (now fixed):
  The original version injected gradients manually via ball.set_pos_grad() and
  called scene._backward() — this hangs indefinitely on GPU (deadlock in
  kernel_step_2.grad / kernel_forward_dynamics_without_qacc.grad).
  Using loss.backward() directly through PyTorch autograd avoids this.

IMPORTANT: rebuild the scene each iteration.
  Genesis accumulates the differentiation tape inside the scene object across
  calls. Even with a fresh gs.tensor for v0, the scene retains the full
  computational graph from all previous rollouts, causing backward() to propagate
  through stale states (gradient magnitudes grow unbounded across iterations).
  Fix: call build_scene(requires_grad=True) at the top of each loop iteration.

Setup:
    pip install genesis-world torch
    python examples/comparison/ball_throw/genesis.py
    python examples/comparison/ball_throw/genesis.py --save examples/comparison/ball_throw/results/genesis.json
"""
import argparse
import json
import pathlib
import time

import genesis as gs
import torch

gs.init(backend=gs.gpu, logging_level="warning")

DT = 3e-2
T = 50  # 1.5 s of ballistic flight

TARGET_V0 = [0.0, 4.0, 7.0]
INIT_V0 = [0.0, 2.0, 1.0]


def build_scene(requires_grad: bool) -> tuple:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -9.81),
            requires_grad=requires_grad,
        ),
        show_viewer=False,
    )
    ball = scene.add_entity(gs.morphs.Sphere(radius=0.2, pos=(0.0, 0.0, 1.0)))
    scene.build()
    return scene, ball


# --- Compute target position with a forward-only scene ---
scene_fwd, ball_fwd = build_scene(requires_grad=False)
print(f"n_dofs={ball_fwd.n_dofs}  (expected 6: freejoint linear+angular vel)")

ball_fwd.set_dofs_velocity(gs.tensor(TARGET_V0), dofs_idx_local=[0, 1, 2])
target_traj = []
for _ in range(T):
    scene_fwd.step()
    target_traj.append(torch.tensor(ball_fwd.get_state().pos[0].tolist(), device="cuda"))
target_traj = torch.stack(target_traj)  # (T, 3), plain tensors — no grad needed
print(f"Target final pos: ({target_traj[-1, 0]:.3f}, {target_traj[-1, 1]:.3f}, {target_traj[-1, 2]:.3f})")
scene_fwd.destroy()

LEARNING_RATE = 2e-2
MAX_GRAD = 100.0

# Current velocity values (plain Python list, updated each iteration)
_vel = list(INIT_V0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON")
    args = parser.parse_args()

    print(f"\nOptimizing: T={T}, dt={DT}, lr={LEARNING_RATE} (gradient descent, Genesis PyTorch autograd)")
    results = {
        "simulator": "Genesis",
        "problem": "ball_throw",
        "dt": DT,
        "T": T,
        "iterations": [],
        "loss": [],
        "time_ms": [],
    }

    for i in range(args.iters):
        t0 = time.perf_counter()

        # Rebuild the scene each iteration to clear Genesis's internal tape.
        # Genesis accumulates the full computational graph inside the scene object;
        # scene.reset() only resets physical state, not the differentiation tape.
        # Rebuilding ensures backward() sees only the current rollout.
        scene, ball = build_scene(requires_grad=True)
        v0 = gs.tensor(_vel, requires_grad=True)

        ball.set_dofs_velocity(v0, dofs_idx_local=[0, 1, 2])
        traj = []
        for t in range(T):
            scene.step()
            traj.append(ball.get_state().pos[0])  # gs.Tensor with grad_fn — do NOT use get_links_pos()

        loss = sum(((traj[t] - target_traj[t]) ** 2).sum() for t in range(T))
        loss.backward()

        t_ms = (time.perf_counter() - t0) * 1000
        loss_val = float(loss)
        grad = [float(g) for g in v0.grad.tolist()]
        grad_clamped = [max(-MAX_GRAD, min(MAX_GRAD, g)) for g in grad]
        for j in range(3):
            _vel[j] -= LEARNING_RATE * grad_clamped[j]

        v0_vals = [round(v, 3) for v in _vel]
        grad_vals = [round(g, 3) for g in grad]

        print(
            f"Iter {i:3d}: loss={loss_val:.4f} | "
            f"v0={v0_vals} | "
            f"grad={grad_vals} | "
            f"t={t_ms:.1f}ms"
        )
        results["iterations"].append(i)
        results["loss"].append(loss_val)
        results["time_ms"].append(t_ms)

        if loss_val < 1e-4:
            print("Converged!")
            break

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
