"""Ball throw optimization using Axion (Newton/Warp, implicit differentiation).

Optimizes the initial velocity of a ball to match a target trajectory.
Uses gradient descent (not Adam) on the 6-DOF initial body velocity.
"""
import argparse
import json
import os
import pathlib
import time

import newton
import numpy as np
import warp as wp
from axion import AxionDifferentiableSimulator
from axion import AxionEngineConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion.simulation.sim_config import SyncMode
from axion import ComplianceConfig
from axion import ContactsConfig
from axion import LinearSolverConfig
from axion import LinesearchConfig
from axion import NewtonRaphsonConfig
from newton import Model

os.environ["PYOPENGL_PLATFORM"] = "glx"

DT = 3e-2
DURATION = 1.5


def bourke_color_map(v_min, v_max, v):
    c = wp.vec3(1.0, 1.0, 1.0)
    v = np.clip(v, v_min, v_max)
    dv = v_max - v_min
    if v < (v_min + 0.25 * dv):
        c[0] = 0.0
        c[1] = 4.0 * (v - v_min) / dv
    elif v < (v_min + 0.5 * dv):
        c[0] = 0.0
        c[2] = 1.0 + 4.0 * (v_min + 0.25 * dv - v) / dv
    elif v < (v_min + 0.75 * dv):
        c[0] = 4.0 * (v - v_min - 0.5 * dv) / dv
        c[2] = 0.0
    else:
        c[1] = 1.0 + 4.0 * (v_min + 0.75 * dv - v) / dv
        c[2] = 0.0
    return c


@wp.kernel
def loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_body_pose: wp.array(dtype=wp.transform, ndim=3),
    loss: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    pos = wp.transform_get_translation(body_pose[tid, 0, 0])
    target_pos = wp.transform_get_translation(target_body_pose[tid, 0, 0])
    delta = pos - target_pos
    wp.atomic_add(loss, 0, wp.dot(delta, delta))


@wp.kernel
def update_kernel(
    qd_grad: wp.array(dtype=wp.spatial_vector, ndim=2),
    alpha: float,
    qd: wp.array(dtype=wp.spatial_vector, ndim=1),
):
    tid = wp.tid()
    if tid > 0:
        return
    max_grad = 20.0
    g = qd_grad[0, 0]
    g_clamped = wp.spatial_vector(
        wp.clamp(g[0], -max_grad, max_grad),
        wp.clamp(g[1], -max_grad, max_grad),
        wp.clamp(g[2], -max_grad, max_grad),
        wp.clamp(g[3], -max_grad, max_grad),
        wp.clamp(g[4], -max_grad, max_grad),
        wp.clamp(g[5], -max_grad, max_grad),
    )
    qd[0] = qd[0] - g_clamped * alpha
    wp.printf("Gradient: [%f %f %f %f %f %f]\n", g[0], g[1], g[2], g[3], g[4], g[5])


class BallThrowOptimizer(AxionDifferentiableSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: AxionEngineConfig,
        logging_config: LoggingConfig,
        save_path: str = None,
    ):
        super().__init__(sim_config, render_config, engine_config, logging_config)

        self.save_path = save_path
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.learning_rate = 3.5e-1
        self.frame = 0

        self.init_vel = wp.spatial_vector(0.0, 2.0, 1.0, 0.0, 0.0, 0.0)
        self.target_init_vel = wp.spatial_vector(0.0, 4.0, 7.0, 0.0, 0.0, 0.0)

        self.track_body(body_idx=0, name="ball", color=(0.0, 1.0, 0.0))

    def build_model(self) -> Model:
        self.builder.rigid_gap = 1.0
        ball = self.builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
            mass=1.0,
        )
        self.builder.add_shape_sphere(body=ball, radius=0.2)
        self.builder.add_ground_plane()
        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=True,
        )

    def compute_loss(self):
        wp.launch(
            kernel=loss_kernel,
            dim=self.clock.total_sim_steps,
            inputs=[self.trajectory.body_pose, self.trajectory.target_body_pose],
            outputs=[self.loss],
            device=self.solver.model.device,
        )

    def update(self):
        wp.launch(
            kernel=update_kernel,
            dim=1,
            inputs=[self.trajectory.body_vel.grad[0], self.learning_rate],
            outputs=[self.states[0].body_qd],
        )

    def render(self, train_iter):
        if self.save_path:
            return
        if self.frame > 0 and train_iter % 3 != 0:
            return
        loss_val = self.loss.numpy()[0]
        color = bourke_color_map(0.0, 10.0, loss_val)
        self._tracked_bodies[0]["color"] = tuple(color)

        def draw_extras(viewer, step_idx, state):
            viewer.log_scalar("/loss", loss_val)
            viewer.log_shapes(
                "/target",
                newton.GeoType.SPHERE,
                0.18,
                wp.array(self.trajectory.target_body_pose.numpy()[-1, 0], dtype=wp.transform),
                wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3),
            )

        print(f"Rendering iteration {train_iter} (Loss: {loss_val:.4f})...")
        self.render_episode(
            iteration=train_iter, callback=draw_extras, loop=True, loops_count=1, playback_speed=2.0
        )
        self.frame += 1

    def train(self, iterations=30):
        wp.copy(
            self.target_states[0].body_qd, wp.array([self.target_init_vel], dtype=wp.spatial_vector)
        )
        self.run_target_episode()

        wp.copy(self.states[0].body_qd, wp.array([self.init_vel], dtype=wp.spatial_vector))
        print(
            f"\nOptimizing: T={self.clock.total_sim_steps}, dt={self.clock.dt:.4f}, lr={self.learning_rate}"
        )
        results = {
            "simulator": "Axion",
            "problem": "ball_throw",
            "dt": self.clock.dt,
            "T": self.clock.total_sim_steps,
            "iterations": [],
            "loss": [],
            "time_ms": [],
        }
        for i in range(iterations):
            t0 = time.perf_counter()
            self.diff_step()
            wp.synchronize()
            t_ms = (time.perf_counter() - t0) * 1000

            curr_loss = self.loss.numpy()[0]
            vel = self.states[0].body_qd.numpy()[0][0:3]
            print(
                f"Iter {i:3d}: loss={curr_loss:.4f} | vel=({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}) | t={t_ms:.0f}ms"
            )
            results["iterations"].append(i)
            results["loss"].append(float(curr_loss))
            results["time_ms"].append(t_ms)

            self.render(i)
            self.update()
            self.tape.zero()
            self.loss.zero_()

        if self.save_path:
            pathlib.Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.save_path).write_text(json.dumps(results, indent=2))
            print(f"Saved to {self.save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON and run headless")
    args = parser.parse_args()

    sim_config = SimulationConfig(
        duration_seconds=DURATION,
        target_timestep_seconds=DT,
        num_worlds=1,
        sync_mode=SyncMode.ALIGN_FPS_TO_DT,
    )
    render_config = RenderingConfig(
        vis_type="null" if args.save else "gl",
        target_fps=30,
        usd_file=None,
        world_offset_x=5.0,
        world_offset_y=5.0,
        start_paused=False,
    )
    engine_config = AxionEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=12, backtrack_min_iter=8, atol=0.001),
        linear=LinearSolverConfig(max_iters=12, atol=0.001, tol=0.001, regularization=1e-06),
        compliance=ComplianceConfig(joint=6e-08, contact=1e-06, friction=1e-06),
        linesearch=LinesearchConfig(enabled=False, conservative_step_count=16, conservative_upper_bound=0.05, min_step=1e-06, optimistic_step_count=48, optimistic_window=0.4),
        contacts=ContactsConfig(max_per_world=256),
    )
    logging_config = LoggingConfig()

    sim = BallThrowOptimizer(
        sim_config,
        render_config,
        engine_config,
        logging_config,
        save_path=args.save,
    )
    sim.train(iterations=15)


if __name__ == "__main__":
    main()
