"""Replay a real helhest run (recorded wheel-velocity setpoints) in the
helhest_junior digital twin and compare against the total-station ground truth.

This is the *open-loop sim-to-real* test: we drive the simulated robot with the
exact wheel-velocity commands recorded on the real robot, over the same box
obstacle as examples/helhest/gradient/trajectory_spline_box.py, and overlay the
resulting chassis trajectory on the recorded `/pose_world` ground truth.

Data source: ~/rosbags_experiment/synced/run_<id>.h5  (see that folder's README).

Two conventions must be reconciled between the recording and the sim:

  1. Column order. The HDF5 stores wheels as [left, rear, right]
     (`joint_names`), but the sim DOF layout is [left, right, rear]
     (DOFs 6, 7, 8). We remap on load.

  2. Sign. On the real robot, *forward* motion is recorded as
     left/rear negative, right positive (the left/right wheel joints have
     opposite axis orientations). In the sim all three wheel joints share
     +Y, so forward needs the *same* sign on every wheel. We flip the left
     and rear columns. If the robot drives backward in the viewer, flip
     WHEEL_SIGN.

Usage:
    # watch it in the GL viewer
    python -m examples.helhest_junior.replay_real --vis gl

    # headless + write comparison plot/npz
    python -m examples.helhest_junior.replay_real --vis headless \
        --out /tmp/replay_18_04_51
"""
import argparse
import pathlib

import h5py
import newton
import numpy as np
import warp as wp
from axion import AdjointConfig
from axion import AdjointLoggingConfig
from axion import AxionEngineConfig
from axion import ComplianceConfig
from axion import ContactsConfig
from axion import DatasetLoggingConfig
from axion import HDF5LoggingConfig
from axion import InteractiveSimulator
from axion import LinearSolverConfig
from axion import LinesearchConfig
from axion import LoggingConfig
from axion import NewtonRaphsonConfig
from axion import ProfilingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion import WarmStartConfig
from axion.collision import ContactReductionConfig
from axion.simulation.sim_config import SyncMode

try:
    from examples.helhest_junior.common import create_helhest_junior_model
except ImportError:
    from common import create_helhest_junior_model

# --- Scene: static box obstacle (real dimensions; placement from measurement) ---
# Real box is ~1 m ahead of where the front wheels contact the ground. The
# front-wheel axis is at robot X=0, so the box near face is at X≈1.0 →
# center X = 1.0 + hx = 1.37. Center z = hz so the box rests flush on the ground.
BOX_HALF_EXTENTS = (0.37, 0.575, 0.06)  # 73 cm long (X), 115 cm wide (Y), 12 cm tall
BOX_CENTER = (1.0 + BOX_HALF_EXTENTS[0], 0.0, BOX_HALF_EXTENTS[2])

# DOF layout: [0..5] free base, [6] left wheel, [7] right wheel, [8] rear wheel
WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3

# HDF5 wheel order is [left, rear, right]; sim wants [left, right, rear].
DATA_TO_SIM = [0, 2, 1]
# Sign flip per sim wheel [left, right, rear] so recorded forward → sim forward.
WHEEL_SIGN = np.array([-1.0, 1.0, -1.0], dtype=np.float32)

# Total-station prism (crystal) mount offset in the chassis frame. The real
# pose_world measures this point, NOT the chassis center — so the robot pitching
# onto the box swings it up/forward. Estimated "top-front of the chassis": the
# front box spans to X=+0.11 (front face) and Z=+0.10 (top), Y-centered.
PRISM_OFFSET = np.array([0.11, 0.0, 0.10], dtype=np.float32)


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vec3 v by quaternion q=[x,y,z,w] (numpy, single)."""
    u = q[:3]
    w = q[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def prism_track(poses: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """World position of the prism point for each chassis pose [T, 7]."""
    out = np.empty((poses.shape[0], 3), dtype=np.float32)
    for k in range(poses.shape[0]):
        p = poses[k, :3]
        q = poses[k, 3:7]
        out[k] = p + _quat_rotate(q, offset)
    return out


DEFAULT_RUN = "run_2026_05_20-18_04_51.h5"
SYNCED_DIR = pathlib.Path.home().joinpath("rosbags_experiment", "synced")


def load_setpoints(h5_path: pathlib.Path, drive: str, dt: float, duration: float):
    """Resample recorded wheel commands onto the sim timestep grid.

    Returns (setpoints[T, 3] in sim order+sign, real_pose dict for comparison).
    """
    with h5py.File(h5_path, "r") as f:
        src = "/joint_setpoint/velocity" if drive == "setpoint" else "/joint_states/velocity"
        t_src = "/joint_setpoint/t" if drive == "setpoint" else "/joint_states/t"
        sp_t = f[t_src][:]
        sp_v = f[src][:]  # [N, 3] in [left, rear, right] order

        real = {
            "t": f["/pose_world/t"][:],
            "position": f["/pose_world/position"][:],
            "orientation": f["/pose_world/orientation"][:],
            "yaw": f["/pose_world/yaw"][:],
        }
        run_id = f.attrs["run_id"]

    # Sim time grid, t=0 == motion start.
    T = int(round(duration / dt))
    t_grid = np.arange(T) * dt

    # Interpolate each recorded wheel column onto the grid (clamps outside range).
    resampled = np.zeros((T, 3), dtype=np.float32)
    for c in range(3):
        resampled[:, c] = np.interp(t_grid, sp_t, sp_v[:, c])

    # Remap [left, rear, right] -> [left, right, rear] and apply sign.
    setpoints = resampled[:, DATA_TO_SIM] * WHEEL_SIGN
    return setpoints.astype(np.float32), real, run_id, t_grid


def align_real_to_sim(real: dict, heading_dist: float = 1.0):
    """Rigid 2D transform of the real (subt-frame) trajectory into the sim frame:
    start at origin, initial heading along +X. NaN holes are preserved.

    The initial heading is taken from the *direction of travel* over the first
    ``heading_dist`` metres, not the single (noisy) first yaw sample — the robot
    drives nearly straight, so this is a far more robust +X reference.

    Returns aligned position [M, 3] (x, y in sim frame; z start-relative) and
    time [M].
    """
    pos = real["position"].copy()
    t = real["t"]

    valid = ~np.isnan(pos[:, 0])
    first = np.argmax(valid)  # first valid index

    # Translate so the first valid sample is at the origin (XY) / z baseline.
    origin = pos[first].copy()
    rel = pos - origin

    # Heading = direction from the start to the first sample ~heading_dist away
    # along the path (robust to per-sample yaw noise).
    vp = rel[valid]
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(vp[:, :2], axis=0), axis=1))])
    j = int(np.argmax(cum >= heading_dist))
    if j == 0:
        j = len(vp) - 1
    heading = np.arctan2(vp[j, 1], vp[j, 0])

    # Rotate XY so the initial heading points along +X.
    theta = -heading
    c, s = np.cos(theta), np.sin(theta)
    x = rel[:, 0] * c - rel[:, 1] * s
    y = rel[:, 0] * s + rel[:, 1] * c

    out = np.empty_like(pos)
    out[:, 0] = x
    out[:, 1] = y
    out[:, 2] = rel[:, 2]  # z relative to start (compared prism-vs-prism, start at 0)
    out[~valid] = np.nan
    return out, t


class HelhestJuniorReplaySimulator(InteractiveSimulator):
    def __init__(self, *args, control_mode="velocity",
                 mu_front=0.8, mu_rear=0.8, mu_rolling=0.7,
                 ground_ke=150.0, ground_kd=150.0, ground_kf=500.0,
                 box_ke=150.0, box_kd=150.0, box_kf=500.0,
                 **kwargs):
        self.control_mode = control_mode
        # Wheel-ground friction. mu_rear governs how easily the rear wheel skids
        # sideways → how readily the robot yaws. Low rear mu = turns too easily;
        # 0.8 (= front, same rubber) matches the real robot's near-straight
        # heading far better than the old artificial 0.35 "slippery rear" value.
        self.mu_front = mu_front
        self.mu_rear = mu_rear
        # mu_rolling is the wheel's resistance to free spin / sideways torsion.
        # It's the closest Axion analog to MuJoCo's torsional friction.
        self.mu_rolling = mu_rolling
        self.ground_cfg_kwargs = dict(ke=ground_ke, kd=ground_kd, kf=ground_kf)
        self.box_cfg_kwargs = dict(ke=box_ke, kd=box_kd, kf=box_kf)
        super().__init__(*args, **kwargs)
        # [left, right, rear] velocity command consumed by control_policy.
        self.target_velocities = wp.zeros(3, dtype=wp.float32, device=self.model.device)
        self.joint_target_buffer = wp.zeros_like(self.model.joint_target_vel)

    def build_model(self) -> newton.Model:
        self.builder.rigid_gap = 0.2

        ground_cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, **self.ground_cfg_kwargs)
        self.builder.add_ground_plane(cfg=ground_cfg)

        self.builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*BOX_CENTER), wp.quat_identity()),
            hx=BOX_HALF_EXTENTS[0],
            hy=BOX_HALF_EXTENTS[1],
            hz=BOX_HALF_EXTENTS[2],
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.8, **self.box_cfg_kwargs),
        )

        create_helhest_junior_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            control_mode=self.control_mode,
            k_p=250.0,
            k_d=0.0,
            friction_left_right=self.mu_front,
            friction_rear=self.mu_rear,
            mu_rolling=self.mu_rolling,
        )

        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)

    def control_policy(self, current_state: newton.State):
        wp.launch(
            kernel=_apply_wheel_velocity_kernel,
            dim=1,
            inputs=[
                self.target_velocities,
                self.joint_target_buffer,
                WHEEL_DOF_OFFSET,
                WHEEL_DOF_OFFSET + 1,
                WHEEL_DOF_OFFSET + 2,
            ],
            device=self.model.device,
        )
        wp.copy(self.control.joint_target_vel, self.joint_target_buffer)

    def _maybe_render(self, step_idx: int):
        if self.viewer is None:
            return
        sim_time = step_idx * self.clock.dt
        self.viewer.begin_frame(sim_time)
        self.viewer.log_state(self.current_state)
        self.viewer.log_contacts(self.contacts, self.current_state)
        self.viewer.end_frame()

    def reset_state(self):
        """Reset the robot to its spawn pose at rest (for reusing one simulator
        instance across multiple runs in a single process)."""
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.current_state)
        self.current_state.body_qd.zero_()

    def replay(self, setpoints: np.ndarray, settle_steps: int = 60):
        """Settle on the ground at zero velocity, then drive with recorded
        setpoints. Returns chassis pose [T, 7] (px,py,pz, qx,qy,qz,qw)."""
        is_gl = self.rendering_config.vis_type == "gl"

        # Settle: let the robot drop to the ground before motion starts.
        self.target_velocities.zero_()
        for _ in range(settle_steps):
            self._single_physics_step(0)
            if is_gl:
                self._maybe_render(0)

        T = setpoints.shape[0]
        poses = np.zeros((T, 7), dtype=np.float32)
        # Actual sim wheel angular velocities [left, right, rear] = joint_qd[6:9].
        # The solver works in maximal (body) coords, so joint_qd must be
        # recovered from the body state via eval_ik each step.
        wheel_qd = np.zeros((T, 3), dtype=np.float32)
        jq = wp.zeros_like(self.model.joint_q)
        jqd = wp.zeros_like(self.model.joint_qd)
        for k in range(T):
            wp.copy(
                self.target_velocities,
                wp.array(setpoints[k], dtype=wp.float32, device=self.model.device),
            )
            self._single_physics_step(k)
            poses[k] = self.current_state.body_q.numpy()[0]
            newton.eval_ik(self.model, self.current_state, jq, jqd)
            wheel_qd[k] = jqd.numpy()[WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS]
            if is_gl:
                self._maybe_render(k)
        return poses, wheel_qd

    def _graph_physics_step(self):
        """One physics step with GPU-side control + logging (capturable)."""
        self.current_state.clear_forces()
        self.contacts = self.model.collide(self.current_state)
        wp.launch(
            kernel=_graph_control_kernel,
            dim=1,
            inputs=[
                self._setpoints_wp,
                self._step_buf,
                self._T,
                self.control.joint_target_vel,
                WHEEL_DOF_OFFSET,
                WHEEL_DOF_OFFSET + 1,
                WHEEL_DOF_OFFSET + 2,
            ],
            device=self.model.device,
        )
        self.solver.step(
            state_in=self.current_state,
            state_out=self.next_state,
            control=self.control,
            contacts=self.contacts,
            dt=self.clock.dt,
        )
        self._copy_state(self.current_state, self.next_state)
        wp.launch(
            _graph_log_pose_kernel,
            dim=1,
            inputs=[self.current_state.body_q, self._step_buf, self._T, self._pose_log],
            device=self.model.device,
        )
        newton.eval_ik(self.model, self.current_state, self._jq, self._jqd)
        wp.launch(
            _graph_log_wheel_kernel,
            dim=1,
            inputs=[self._jqd, self._step_buf, self._T, self._wheel_log, WHEEL_DOF_OFFSET],
            device=self.model.device,
        )
        wp.launch(_graph_advance_kernel, dim=1, inputs=[self._step_buf], device=self.model.device)

    def replay_graph(self, setpoints: np.ndarray, settle_steps: int = 60):
        """CUDA-graph replay: capture one physics step (GPU-side control +
        logging via a step counter), then launch it T times with no Python in
        the physics loop. Works headless AND with the GL viewer — the captured
        graph only covers the *physics* step; rendering happens from Python
        between graph launches (the InteractiveSimulator pattern).
        Returns chassis pose [T, 7], wheel qd [T, 3]."""
        T = setpoints.shape[0]
        self._T = T
        self._setpoints_wp = wp.array(setpoints, dtype=wp.float32, device=self.model.device)
        self._step_buf = wp.zeros(1, dtype=wp.int32, device=self.model.device)
        self._pose_log = wp.zeros((T, 7), dtype=wp.float32, device=self.model.device)
        self._wheel_log = wp.zeros((T, 3), dtype=wp.float32, device=self.model.device)
        self._jq = wp.zeros_like(self.model.joint_q)
        self._jqd = wp.zeros_like(self.model.joint_qd)

        # Settle on the ground (uncaptured, zero command).
        self.target_velocities.zero_()
        for _ in range(settle_steps):
            self._single_physics_step(0)

        # Capture one physics step, then replay it.
        self._step_buf.zero_()
        with wp.ScopedCapture() as capture:
            self._graph_physics_step()
        graph = capture.graph

        is_gl = self.rendering_config.vis_type == "gl"
        if not is_gl:
            # Headless: fire all T launches back-to-back, read logs once.
            for _ in range(T):
                wp.capture_launch(graph)
            wp.synchronize()
            return self._pose_log.numpy(), self._wheel_log.numpy()

        # GL: launch the physics graph, then render the updated state each step.
        step = 0
        while self.viewer.is_running() and step < T:
            if not self.viewer.is_paused():
                wp.capture_launch(graph)
                step += 1
            self._maybe_render(step)
            wp.synchronize()
        return self._pose_log.numpy(), self._wheel_log.numpy()


@wp.kernel
def _apply_wheel_velocity_kernel(
    target_velocities: wp.array(dtype=wp.float32),
    joint_target_vel: wp.array(dtype=wp.float32),
    left_idx: int,
    right_idx: int,
    rear_idx: int,
):
    joint_target_vel[left_idx] = target_velocities[0]
    joint_target_vel[right_idx] = target_velocities[1]
    joint_target_vel[rear_idx] = target_velocities[2]


# --- CUDA-graph kernels: per-step indexing/logging happens on the GPU via a
# step counter, so the whole rollout is capturable (no Python in the loop). ---
@wp.kernel
def _graph_control_kernel(
    setpoints: wp.array(dtype=wp.float32, ndim=2),  # [T, 3] in sim order [L,R,rear]
    step_buf: wp.array(dtype=wp.int32),
    T: int,
    joint_target_vel: wp.array(dtype=wp.float32),
    left_idx: int,
    right_idx: int,
    rear_idx: int,
):
    s = wp.min(step_buf[0], T - 1)
    joint_target_vel[left_idx] = setpoints[s, 0]
    joint_target_vel[right_idx] = setpoints[s, 1]
    joint_target_vel[rear_idx] = setpoints[s, 2]


@wp.kernel
def _graph_log_pose_kernel(
    body_q: wp.array(dtype=wp.transform),
    step_buf: wp.array(dtype=wp.int32),
    T: int,
    pose_log: wp.array(dtype=wp.float32, ndim=2),  # [T, 7]
):
    s = wp.min(step_buf[0], T - 1)
    tf = body_q[0]
    p = wp.transform_get_translation(tf)
    q = wp.transform_get_rotation(tf)
    pose_log[s, 0] = p[0]
    pose_log[s, 1] = p[1]
    pose_log[s, 2] = p[2]
    pose_log[s, 3] = q[0]
    pose_log[s, 4] = q[1]
    pose_log[s, 5] = q[2]
    pose_log[s, 6] = q[3]


@wp.kernel
def _graph_log_wheel_kernel(
    jqd: wp.array(dtype=wp.float32),
    step_buf: wp.array(dtype=wp.int32),
    T: int,
    wheel_log: wp.array(dtype=wp.float32, ndim=2),  # [T, 3]
    off: int,
):
    s = wp.min(step_buf[0], T - 1)
    wheel_log[s, 0] = jqd[off + 0]
    wheel_log[s, 1] = jqd[off + 1]
    wheel_log[s, 2] = jqd[off + 2]


@wp.kernel
def _graph_advance_kernel(step_buf: wp.array(dtype=wp.int32)):
    step_buf[0] = step_buf[0] + 1


def best_time_shift(sim_rel, sim_t, real_aligned, real_t):
    """Cross-correlate forward progress (x) to recover the constant time offset
    between the sim and real streams. The robot/pose streams were zeroed on
    different events (wheels-spinning vs robot-translated), so a fixed lag is
    expected. Returns shift s such that real(t) ≈ sim(t + shift)."""
    v = ~np.isnan(real_aligned[:, 0])
    rt, rx = real_t[v], real_aligned[v, 0]
    sx = sim_rel[:, 0]
    tg = np.linspace(0.2, min(rt.max(), sim_t.max()) - 0.3, 60)
    rxi = np.interp(tg, rt, rx)
    shifts = np.linspace(-0.8, 0.8, 161)
    errs = [np.mean((np.interp(tg + s, sim_t, sx) - rxi) ** 2) for s in shifts]
    return float(shifts[int(np.argmin(errs))])


def save_comparison(out_prefix: str, sim_prism, real_aligned, real_t, sim_dt):
    # Both tracks are the prism point, expressed relative to their own start.
    sim_rel = sim_prism - sim_prism[0]
    np.savez_compressed(
        out_prefix + ".npz",
        dt=np.float32(sim_dt),
        sim_prism=sim_rel.astype(np.float32),
        real_position=real_aligned.astype(np.float32),
        real_t=real_t.astype(np.float32),
    )
    print(f"Saved trajectories to {out_prefix}.npz")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable — skipping plot")
        return

    sim_t = np.arange(sim_rel.shape[0]) * sim_dt
    shift = best_time_shift(sim_rel, sim_t, real_aligned, real_t)
    sim_t_aligned = sim_t - shift  # move sim onto the real time axis
    print(f"Stream time-alignment shift: {shift:+.3f} s")
    fig, (ax_xy, ax_z) = plt.subplots(1, 2, figsize=(13, 5))

    ax_xy.plot(sim_rel[:, 0], sim_rel[:, 1], "-", color="tab:blue", label="sim prism")
    ax_xy.plot(real_aligned[:, 0], real_aligned[:, 1], "-", color="tab:red", label="real (TS)")
    # Box in the start-relative frame: shift by the sim prism start (world).
    box_shift = sim_prism[0]
    box_x = [
        BOX_CENTER[0] - BOX_HALF_EXTENTS[0] - box_shift[0],
        BOX_CENTER[0] + BOX_HALF_EXTENTS[0] - box_shift[0],
    ]
    box_y = [
        BOX_CENTER[1] - BOX_HALF_EXTENTS[1] - box_shift[1],
        BOX_CENTER[1] + BOX_HALF_EXTENTS[1] - box_shift[1],
    ]
    ax_xy.add_patch(
        plt.Rectangle(
            (box_x[0], box_y[0]),
            box_x[1] - box_x[0],
            box_y[1] - box_y[0],
            color="gray",
            alpha=0.3,
            label="box",
        )
    )
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title("Top-down prism trajectory (start-aligned)")
    ax_xy.axis("equal")
    ax_xy.legend()
    ax_xy.grid(alpha=0.3)

    ax_z.plot(
        sim_t_aligned, sim_rel[:, 2], "-", color="tab:blue", label=f"sim prism z ({shift:+.2f}s)"
    )
    ax_z.plot(real_t, real_aligned[:, 2], "-", color="tab:red", label="real z")
    ax_z.set_xlabel("t [s]  (sim shifted onto real stream)")
    ax_z.set_ylabel("z rise from start [m]")
    ax_z.set_title("Prism elevation vs time (climb + pitch)")
    ax_z.legend()
    ax_z.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=130)
    print(f"Saved comparison plot to {out_prefix}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        default=str(SYNCED_DIR / DEFAULT_RUN),
        help="Path to synced run_*.h5 (default: cleanest run 18_04_51)",
    )
    parser.add_argument("--vis", choices=["gl", "headless"], default="gl")
    parser.add_argument(
        "--drive",
        choices=["setpoint", "measured"],
        default="setpoint",
        help="setpoint = commanded velocities (open-loop twin test); "
        "measured = recorded wheel velocities",
    )
    parser.add_argument("--dt", type=float, default=0.05, help="sim timestep [s]")
    parser.add_argument("--duration", type=float, default=5.0, help="replay duration [s]")
    parser.add_argument(
        "--out", default=None, help="output prefix for comparison .npz/.png (headless)"
    )
    parser.add_argument(
        "--prism",
        default=None,
        metavar="X,Y,Z",
        help="prism mount offset in chassis frame (default: top-front " f"{tuple(PRISM_OFFSET)})",
    )
    parser.add_argument("--mu-front", type=float, default=0.8,
                        help="front-wheel ground friction (default 0.8)")
    parser.add_argument("--mu-rear", type=float, default=0.8,
                        help="rear-wheel ground friction; higher = resists sideways "
                             "skid = turns less (default 0.8 = front; 0.35 was the old "
                             "artificial slippery-rear value)")
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="capture the physics step into a CUDA graph (faster; works headless and with GL)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="batch-process every run_*.h5 in the synced dir in ONE process "
        "(reuses the simulator; --out is treated as a directory). Headless.",
    )
    args = parser.parse_args()

    use_graph = args.cuda_graph and wp.get_device().is_cuda
    if args.cuda_graph and not use_graph:
        print("WARNING: --cuda-graph needs a CUDA device; falling back to the Python loop.")

    prism_offset = (
        np.array([float(v) for v in args.prism.split(",")], dtype=np.float32)
        if args.prism
        else PRISM_OFFSET
    )

    # --- Fully explicit configuration (no reliance on dataclass defaults) ---
    # Every field of every (sub-)config is written out so the run is fully
    # reproducible and self-documenting. Values equal to the library defaults
    # are still listed; only --dt/--duration/--vis vary by CLI.
    sim_config = SimulationConfig(
        duration_seconds=args.duration,  # CLI (default 5.0)
        target_timestep_seconds=args.dt,  # CLI (default 0.01)
        num_worlds=1,
        sync_mode=SyncMode.ALIGN_FPS_TO_DT,
        # BaseSimulator.run()'s own graph path is unused — replay drives its
        # own loop / manual capture (replay_graph) — so keep this False.
        use_cuda_graph=True,
    )
    render_config = RenderingConfig(
        vis_type="null" if args.all else ("gl" if args.vis == "gl" else "null"),
        target_fps=int(round(1.0 / args.dt)),
        usd_file="sim.usd",  # only used when vis_type == "usd"
        usd_scaling=100.0,
        start_paused=False,
        world_offset_x=20.0,
        world_offset_y=20.0,
    )
    engine_config = AxionEngineConfig(
        differentiable=False,
        nr=NewtonRaphsonConfig(
            max_iters=16,
            backtrack_min_iter=12,
            atol=1e-3,
        ),
        linear=LinearSolverConfig(
            max_iters=16,
            tol=1e-3,
            atol=1e-3,
            preconditioner_type="jacobi",
            regularization=1e-6,
        ),
        compliance=ComplianceConfig(
            joint=6e-8,
            contact=1e-6,
            friction=1e-6,
            contact_fb_smooth_eps_sq=1e-8,
        ),
        linesearch=LinesearchConfig(
            enabled=False,  # disabled → the rest is inactive
            min_step=1e-6,
            conservative_step_count=32,
            conservative_upper_bound=0.05,
            optimistic_step_count=32,
            optimistic_window=0.2,
        ),
        warm_start=WarmStartConfig(
            enabled=True,
            cold_gravity=True,
            cold_impact=True,
            cold_friction_v_threshold=0.1,
            method="position_match",
            seed_iterate=False,
        ),
        contacts=ContactsConfig(
            max_per_world=256,
            reduction=ContactReductionConfig(
                policy="none",
                max_per_pair=4,
                cluster_normal_dot_thresh=0.996,
                cluster_pos_thresh=5e-3,
            ),
        ),
        adjoint=AdjointConfig(  # inactive (differentiable=False)
            soft_blending=True,
            soft_blending_temperature=0.05,
            regularization=0.0,
            gradient_normalization=False,
        ),
        profiling=ProfilingConfig(
            segment_timing=False,
            mode="off",
        ),
    )
    logging_config = LoggingConfig(
        hdf5=HDF5LoggingConfig(
            enabled=False,
            file="simulation.h5",
            log_dynamics_state=True,
            log_linear_system_data=True,
            log_constraint_data=True,
        ),
        dataset=DatasetLoggingConfig(enabled=False, file="dataset.h5"),
        adjoint=AdjointLoggingConfig(enabled=False, file="adjoint.h5"),
    )

    # Build the simulator ONCE. Process startup (warp init, model build, module
    # loads) dominates a single replay, so batching all runs here — reusing this
    # instance via reset_state — is far faster than relaunching per run.
    sim = HelhestJuniorReplaySimulator(
        sim_config,
        render_config,
        engine_config,
        logging_config,
        control_mode="velocity",
        mu_front=args.mu_front,
        mu_rear=args.mu_rear,
    )

    if args.all:
        runs = sorted(SYNCED_DIR.glob("run_*.h5"))
        out_dir = pathlib.Path(args.out) if args.out else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
        for h5_path in runs:
            out_prefix = str(out_dir / h5_path.stem) if out_dir else None
            _replay_one(sim, h5_path, args, use_graph, prism_offset, out_prefix)
    else:
        _replay_one(sim, pathlib.Path(args.run), args, use_graph, prism_offset, args.out)


def _replay_one(sim, h5_path, args, use_graph, prism_offset, out_prefix):
    """Run a single recorded run on an already-built simulator."""
    import time

    setpoints, real, run_id, _ = load_setpoints(h5_path, args.drive, args.dt, args.duration)
    real_aligned, real_t = align_real_to_sim(real)
    print(f"\nRun {run_id}: {setpoints.shape[0]} steps @ dt={args.dt}s, drive={args.drive}")

    sim.reset_state()
    t0 = time.perf_counter()
    if use_graph:
        sim_pose, sim_wheel_qd = sim.replay_graph(setpoints)
    else:
        sim_pose, sim_wheel_qd = sim.replay(setpoints)
    elapsed = time.perf_counter() - t0
    mode = "cuda-graph" if use_graph else "python-loop"
    print(
        f"Replay ({mode}): {setpoints.shape[0]} steps in {elapsed:.3f}s "
        f"({1000 * elapsed / setpoints.shape[0]:.2f} ms/step)"
    )

    net = np.linalg.norm(sim_pose[-1, :2] - sim_pose[0, :2])
    print(f"Sim net XY displacement: {net:.3f} m  (final pose {sim_pose[-1, :3]})")

    _print_speed_decomposition(sim_pose, sim_wheel_qd, setpoints, args.dt, real_aligned, real_t)

    if out_prefix:
        sim_prism = prism_track(sim_pose, prism_offset)
        save_comparison(out_prefix, sim_prism, real_aligned, real_t, args.dt)


def _print_speed_decomposition(sim_pose, sim_wheel_qd, setpoints, dt, real_aligned, real_t):
    """Decompose flat-ground cruise into command → wheel → ground, exposing
    motor-tracking loss vs wheel slip. Window is before the box (x < 0.9 m)."""
    R = 0.35  # wheel radius
    x = sim_pose[:, 0]
    t = np.arange(len(x)) * dt
    pre = (t > 0.3) & (x < 0.9)
    if pre.sum() < 5:
        return

    cmd = np.abs(setpoints[pre]).mean()
    wheel = np.abs(sim_wheel_qd[pre]).mean()
    ground = np.median(np.diff(np.linalg.norm(sim_pose[:, :2], axis=1))[pre[:-1]]) / dt

    print("\n--- SIM flat-ground decomposition (x < 0.9 m) ---")
    print(f"  commanded wheel speed : {cmd:.3f} rad/s")
    print(f"  actual wheel speed    : {wheel:.3f} rad/s  ({100*wheel/cmd:.0f}% of command)")
    print(f"  no-slip ground (w*R)  : {wheel*R:.3f} m/s")
    print(f"  actual ground speed   : {ground:.3f} m/s  (slip {100*(1-ground/(wheel*R)):.0f}%)")
    rv = ~np.isnan(real_aligned[:, 0])
    rspd = np.linalg.norm(np.diff(real_aligned[rv][:, :2], axis=0), axis=1) / np.diff(real_t[rv])
    rt = real_t[rv][:-1]
    rwin = (rt > 0.3) & (rt < 1.7)
    if rwin.sum() > 3:
        print(f"  [real ground speed    : {np.nanmedian(rspd[rwin]):.3f} m/s]")


if __name__ == "__main__":
    main()
