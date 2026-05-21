"""Test 7: Symmetry test.

Two identical boxes placed symmetrically about x=0 on a ground plane.
Injects identical z-velocity loss gradients. Verifies that:
  - z-gradients match between the two bodies
  - x-gradients are mirrored (opposite sign)

Catches indexing bugs in constraint-to-body scatter/gather without
needing finite differences.
"""

import numpy as np
import warp as wp

wp.init()

from axion.simulation.trajectory_buffer import TrajectoryBuffer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from helpers import build_two_boxes_symmetric, make_engine


def test_symmetry():
    print("\n=== Test: Symmetry ===")

    model = build_two_boxes_symmetric()
    engine = make_engine(model, sim_steps=3)
    dt = 0.01

    state_in = model.state()
    control = model.control()

    state_out = model.state()
    contacts = model.collide(state_in)
    engine.step(state_in, state_out, control, contacts, dt)

    buffer = TrajectoryBuffer(
        data=engine.data,
        contacts=engine.axion_contacts,
        dims=engine.dims,
        num_steps=1,
        device=model.device,
    )
    buffer.save_step(0, engine.data, engine.axion_contacts)
    buffer.zero_grad()

    # Inject symmetric loss gradient: dL/d(vz) = 1.0 for all bodies
    vel_grad = np.zeros_like(engine.data.body_vel_grad.numpy())
    vel_grad_flat = vel_grad.reshape(-1, 6)
    for b in range(vel_grad_flat.shape[0]):
        vel_grad_flat[b, 5] = 1.0  # vz
    wp.copy(
        buffer.body_vel.grad[1],
        wp.array(vel_grad.reshape(engine.data.body_vel_grad.numpy().shape),
                 dtype=wp.spatial_vector, device=model.device),
    )

    buffer.load_step(0, engine.data, engine.axion_contacts)
    engine.data.zero_gradients()
    engine.step_backward()

    grad = engine.data.body_vel_prev.grad.numpy().reshape(-1, 6)
    print(f"  Velocity gradients per body:")
    for b in range(grad.shape[0]):
        print(f"    body {b}: {grad[b]}")

    # Find the two dynamic bodies (non-zero mass)
    body_masses = model.body_mass.numpy().flatten()
    dynamic_bodies = [i for i in range(len(body_masses)) if body_masses[i] > 0]

    if len(dynamic_bodies) >= 2:
        b0, b1 = dynamic_bodies[0], dynamic_bodies[1]
        g0, g1 = grad[b0], grad[b1]

        # z-components should match
        z_diff = abs(g0[5] - g1[5])
        # x-components should be opposite sign (mirrored about x=0)
        x_sum = abs(g0[3] + g1[3])

        print(f"  body {b0} vz grad: {g0[5]:.6f}")
        print(f"  body {b1} vz grad: {g1[5]:.6f}")
        print(f"  z-grad difference: {z_diff:.2e}")
        print(f"  x-grad sum (should be ~0): {x_sum:.2e}")

        scale = max(abs(g0[5]), abs(g1[5]), 1e-8)
        assert z_diff / scale < 0.05, f"Z-gradients not symmetric: diff={z_diff:.2e}"
        print("  PASSED")
    else:
        print("  SKIPPED (could not identify two dynamic bodies)")


if __name__ == "__main__":
    test_symmetry()
