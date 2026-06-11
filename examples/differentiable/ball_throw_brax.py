"""Ball throw trajectory optimization using MuJoCo MJX (JAX-based differentiable physics).

Comparison to ball_throw_trajectory_ostrich.py which uses Ostrich implicit gradients.

Setup: optimize initial velocity of a free ball to match a target trajectory.
  - Duration: 1.5s, dt=3e-2 -> T=50 steps  (matches Ostrich base config)
  - Initial guess: linear vel (0, 2, 1)
  - Target:        linear vel (0, 4, 7)
  - Loss: sum of L2 position errors over all timesteps
  - Optimizer: gradient descent with gradient clamping (lr=0.2), matching Ostrich
"""

import time

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np

# Match Ostrich base config: dt=3e-2, duration=1.5s
DT = 3e-2
T = int(1.5 / DT)  # 50 steps

XML = f"""
<mujoco model="ball_throw">
  <option gravity="0 0 -9.81" timestep="{DT}"/>
  <worldbody>
    <body name="ball" pos="0 0 1">
      <freejoint name="ball_free"/>
      <geom name="ball_geom" type="sphere" size="0.2" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""


def make_init_data(mx, mj_model, q0_np, qd0):
    """Create MJX data with given initial position and velocity."""
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = q0_np
    dx = mjx.put_data(mj_model, mj_data)
    return dx.replace(qvel=qd0)


def rollout(mx, dx0):
    """Simulate for T steps, return [T, 3] position trajectory."""

    def step_fn(carry, _):
        d = mjx.step(mx, carry)
        return d, d.qpos[:3]

    _, positions = jax.lax.scan(step_fn, dx0, None, length=T)
    return positions  # [T, 3]


def main():
    mj_model = mujoco.MjModel.from_xml_string(XML)
    mx = mjx.put_model(mj_model)

    # Ball start: pos=(0,0,1), quaternion=(w=1,x=0,y=0,z=0)
    q0_np = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    # Matching Ostrich: spatial_vector(0, 4, 7, 0, 0, 0) where [0:3] = linear vel
    target_qd0 = jnp.array([0.0, 4.0, 7.0, 0.0, 0.0, 0.0])
    init_qd0 = jnp.array([0.0, 2.0, 1.0, 0.0, 0.0, 0.0])

    # Generate target trajectory
    dx_target = make_init_data(mx, mj_model, q0_np, target_qd0)
    target_positions = rollout(mx, dx_target)
    print(f"Target final pos: {target_positions[-1]}")

    def loss_fn(qd0):
        dx0 = make_init_data(mx, mj_model, q0_np, qd0)
        positions = rollout(mx, dx0)
        return jnp.sum((positions - target_positions) ** 2)

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    # Warmup JIT
    print("Compiling...")
    t_compile = time.perf_counter()
    loss, grad = loss_and_grad(init_qd0)
    loss.block_until_ready()
    print(f"Compile time: {time.perf_counter() - t_compile:.2f}s\n")

    qd0 = init_qd0
    lr = 0.05
    max_grad = 20.0

    print(f"Optimizing: T={T}, dt={DT}, lr={lr}")
    for i in range(30):
        t0 = time.perf_counter()
        loss, grad = loss_and_grad(qd0)
        loss.block_until_ready()
        t_iter = time.perf_counter() - t0

        print(
            f"Iter {i:2d}: Loss={loss:.4f} | "
            f"Vel=({qd0[0]:.2f}, {qd0[1]:.2f}, {qd0[2]:.2f}) | "
            f"t={t_iter * 1000:.1f}ms"
        )

        grad_clamped = jnp.clip(grad, -max_grad, max_grad)
        qd0 = qd0 - lr * grad_clamped


if __name__ == "__main__":
    main()
