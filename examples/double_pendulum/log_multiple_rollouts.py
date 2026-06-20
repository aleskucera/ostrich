"""log_multiple_rollouts_autoregressive.py

Run N independent double-pendulum rollouts from randomly sampled initial
conditions and save all trajectories to a single HDF5 file.

Top-level knobs
---------------
ENGINE      : "axion" | "gpt" | "teacher_forced_gpt" | "teacher_forced_mlp"   — which physics engine to use
N_ROLLOUTS  : int                — number of trajectories to collect
SEED        : int                — RNG seed for reproducible initial conditions
N_STEPS     : int                — timesteps per rollout (overrides yaml duration)
Q_RANGE     : (float, float)     — uniform range for joint angles q0, q1  [rad]
QD_RANGE    : (float, float)     — uniform range for joint velocities qd0, qd1 [rad/s]
PLANE_COEFF : (a, b, c, d)       — fixed plane ax+by+cz+d=0 when RANDOMIZE_PLANES is False
RANDOMIZE_PLANES : bool          — per rollout, sample a tilted plane (as in
                                  trajectory_sampler_pendulum + UniformSampler); if the
                                  rod would cross that plane at t=0, use (0,0,1,0) instead
MAX_D_COEFFICIENT_OFFSET_M : float — extra d offset uniform in [0, this] (matches training)
"""

import pathlib
import sys
from typing import override

import hydra
import newton
import numpy as np
import warp as wp
from tqdm import tqdm
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# ---------------------------------------------------------------------------
# USER-FACING KNOBS — edit these
# ---------------------------------------------------------------------------
ENGINE      = "teacher_forced_gpt"           # "axion" | "gpt" | "teacher_forced_gpt" | "teacher_forced_mlp" | "axion_neural_lambdas"
N_ROLLOUTS  = 25
SEED        = 0
N_STEPS     = 300               # timesteps per rollout
Q_RANGE     = (-np.pi, np.pi)  # q0, q1 sampled uniformly from this range
QD_RANGE    = (-3.0, 3.0)      # qd0, qd1 sampled uniformly from this range
PLANE_COEFF = (0.0, 0.0, 1.0, 0.0)  # horizontal ground plane (default)
RANDOMIZE_PLANES = True  # when True, per-rollout randomly sampled plane is used overriding PLANE_COEFF 
MAX_D_COEFFICIENT_OFFSET_M = 2.5
# ---------------------------------------------------------------------------

# Add the examples directory to sys.path so local modules resolve.
_EXAMPLES_DIR = pathlib.Path(__file__).parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

CONFIG_PATH = _EXAMPLES_DIR.parent / "conf"

from axion import EngineConfig, InteractiveSimulator, JointMode
from axion import LoggingConfig, RenderingConfig, SimulationConfig
from axion.neural_solver.logging.state_logger_for_examples import MultiRolloutStateLogger
from pendulum_articulation_definition import (
    LINK_LENGTH,
    PENDULUM_HEIGHT,
    build_pendulum_model,
)
from pendulum_utils import generalized_to_maximal, set_tilted_plane_from_coefficients

# Whether the chosen engine requires an eval_ik pass before reading joint_q/qd
_NEEDS_EVAL_IK = ENGINE == "axion"
# Whether to log the NN-predicted state from the engine rather than current_state
_LOGS_NN_PREDICTED = ENGINE == "axion_neural_lambdas"


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class Simulator(InteractiveSimulator):
    """Headless double-pendulum simulator supporting both Axion and GPT engines."""

    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
        plane_coefficients: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.0),
        initial_state: tuple[float, float, float, float] | None = None,
        state_logger: MultiRolloutStateLogger | None = None,
    ):
        self.plane_coefficients = plane_coefficients
        self.state_logger = state_logger
        super().__init__(sim_config, render_config, engine_config, logging_config)

        if initial_state is not None:
            q0, q1, qd0, qd1 = initial_state
            generalized_to_maximal(
                self.model, self.current_state,
                q0=q0, q1=q1, qd0=qd0, qd1=qd1,
            )

    @override
    def control_policy(self, state: newton.State):
        pass  # no control

    @override
    def _render(self, segment_num: int):
        """Minimal render: advance the ViewerNull frame counter and log state."""
        sim_time = segment_num * self.steps_per_segment * self.clock.dt

        # ViewerNull.is_running() counts calls to end_frame(); without this the
        # simulation loop never terminates.
        self.viewer.begin_frame(sim_time)
        self.viewer.end_frame()

        if self.state_logger is not None and self.use_cuda_graph:
            if _NEEDS_EVAL_IK:
                newton.eval_ik(
                    self.model,
                    self.current_state,
                    self.current_state.joint_q,
                    self.current_state.joint_qd,
                )
            self.state_logger.log_step(self.current_state, sim_time)

    @override
    def _run_segment_without_graph(self, segment_num: int):
        if segment_num == 0:
            prewarm_fn = getattr(self.solver, "prewarm", None)
            if prewarm_fn is not None:
                prewarm_fn(self.current_state, self.contacts, self.clock.dt)
        for step in range(self.steps_per_segment):
            self._single_physics_step(step)
            if self.state_logger is not None:
                global_step = segment_num * self.steps_per_segment + step + 1
                if _LOGS_NN_PREDICTED:
                    pred = getattr(self.solver, "last_predicted_next_states", None)
                    if pred is not None:
                        arr = pred[0].detach().cpu().numpy()
                        self.state_logger.log_step_from_array(arr, global_step * self.clock.dt)
                else:
                    if _NEEDS_EVAL_IK:
                        newton.eval_ik(
                            self.model,
                            self.current_state,
                            self.current_state.joint_q,
                            self.current_state.joint_qd,
                        )
                    self.state_logger.log_step(self.current_state, global_step * self.clock.dt)

    def build_model(self) -> newton.Model:
        model = build_pendulum_model(num_worlds=1, device="cuda:0")
        model.joint_dof_mode.assign(
            wp.array([JointMode.NONE, JointMode.NONE], dtype=wp.int32, device=model.device)
        )
        a, b, c, d = self.plane_coefficients
        set_tilted_plane_from_coefficients(model, a, b, c, d, world_idx=0)
        return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cfg():
    """Load Hydra config for the selected engine without the @hydra.main decorator.

    The ``project_name`` override resolves the ``${hydra:job.name}`` interpolation
    used in some logging configs, which is only available when Hydra is launched via
    ``@hydra.main``.  Providing it explicitly keeps the compose API path working.
    """
    GlobalHydra.instance().clear()
    if ENGINE == "axion":
        config_name = "pendulum"
    elif ENGINE == "gpt":
        config_name = "gpt_pendulum"
    elif ENGINE == "teacher_forced_gpt":
        config_name = "teacher_forced_gpt_pendulum"
    elif ENGINE == "teacher_forced_mlp":
        config_name = "teacher_forced_mlp_pendulum"
    elif ENGINE == "axion_neural_lambdas":
        config_name = "axion_w_neural_lambdas_pendulum"
    else:
        raise ValueError(f"Unknown ENGINE: {ENGINE!r}")
    with initialize_config_dir(config_dir=str(CONFIG_PATH), version_base=None):
        cfg = compose(
            config_name=config_name,
            overrides=[
                "rendering=headless",
                "project_name=multi_rollout_autoregressive",
            ],
        )
    return cfg


def _hdf5_filename_stem(dt: float) -> str:
    """Filesystem-safe HDF5 stem from USER-FACING KNOBS and resolved ``dt``."""
    if ENGINE == "axion":
        eng_label = "Axion"
    elif ENGINE == "gpt":
        eng_label = "GPT"
    elif ENGINE == "teacher_forced_gpt":
        eng_label = "TeacherForcedGPT"
    elif ENGINE == "teacher_forced_mlp":
        eng_label = "TeacherForcedMLP"
    elif ENGINE == "axion_neural_lambdas":
        eng_label = "AxionNeuralLambdas"
    else:
        eng_label = ENGINE.replace("-", "_").title()

    def fnum(x: float) -> str:
        return f"{x:g}".replace("-", "m").replace(".", "p")

    q_lo, q_hi = Q_RANGE
    qd_lo, qd_hi = QD_RANGE
    a, b, c, d = PLANE_COEFF
    stem = (
        f"{eng_label}_{N_ROLLOUTS}roll_seed{SEED}_{N_STEPS}steps_dt{fnum(dt)}_"
        f"q{fnum(q_lo)}_{fnum(q_hi)}_qd{fnum(qd_lo)}_{fnum(qd_hi)}_"
        f"pl{fnum(a)}_{fnum(b)}_{fnum(c)}_{fnum(d)}"
    )
    if RANDOMIZE_PLANES:
        stem += f"_rndplane_dmax{fnum(MAX_D_COEFFICIENT_OFFSET_M)}"
    return stem


def _sample_plane_normal_uniform(rng: np.random.Generator) -> np.ndarray:
    """Match UniformSampler.sample_plane_normals (single sample, shape (3,))."""
    x = rng.uniform(-3.0, 3.0)
    radicand = max((2.0 * LINK_LENGTH) ** 2 - x * x, 0.0)
    z = -np.sqrt(radicand)
    return np.array([x, 0.0, z], dtype=np.float64)


def _pendulum_keypoints(ic_q0: float, ic_q1: float) -> np.ndarray:
    """Pivot and link tips in world frame; same kinematics as TrajectorySamplerPendulum."""
    q0 = np.pi / 2.0 - ic_q0
    q1 = q0 - ic_q1
    x0 = LINK_LENGTH * np.sin(q0)
    z0 = -LINK_LENGTH * np.cos(q0) + PENDULUM_HEIGHT
    x1 = LINK_LENGTH * np.sin(q1) + x0
    z1 = -LINK_LENGTH * np.cos(q1) + z0
    return np.array(
        [
            [0.0, 0.0, PENDULUM_HEIGHT],
            [x0, 0.0, z0],
            [x1, 0.0, z1],
        ],
        dtype=np.float64,
    )


def _plane_crosses_pendulum(points_xyz: np.ndarray, n_hat: np.ndarray, d: float) -> bool:
    """True if min(n·p+d) < 0 and max(n·p+d) > 0 over keypoints (training sampler logic)."""
    signed = (points_xyz * n_hat).sum(axis=-1) + d
    return bool(signed.min() < 0.0 and signed.max() > 0.0)


def _plane_coeff_for_rollout(
    rng: np.random.Generator,
    ic: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Sample plane like trajectory generation; on geometric crossing use (0,0,1,0)."""
    q0, q1, _, _ = ic
    raw = _sample_plane_normal_uniform(rng)
    nrm = np.linalg.norm(raw)
    if nrm < 1e-8:
        n_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        n_hat = raw / nrm
    a, b, c = float(n_hat[0]), float(n_hat[1]), float(n_hat[2])
    x, y, z = float(raw[0]), float(raw[1]), float(raw[2])
    d0 = -a * x - b * y - c * (z + PENDULUM_HEIGHT)
    d = d0 + rng.uniform(0.0, MAX_D_COEFFICIENT_OFFSET_M)
    pts = _pendulum_keypoints(q0, q1)
    if _plane_crosses_pendulum(pts, n_hat, d):
        return (0.0, 0.0, 1.0, 0.0)
    return (-a, -b, -c, -float(d))


def _sample_initial_conditions(rng: np.random.Generator) -> tuple[float, float, float, float]:
    q0, q1 = rng.uniform(*Q_RANGE, size=2)
    qd0, qd1 = rng.uniform(*QD_RANGE, size=2)
    return (float(q0), float(q1), float(qd0), float(qd1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = _load_cfg()

    # Instantiate configs
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)

    # Override simulation duration based on N_STEPS
    dt = sim_config.target_timestep_seconds
    sim_config.duration_seconds = N_STEPS * dt

    rng = np.random.default_rng(seed=SEED)
    plane_crossing_fallbacks = 0

    multi_logger = MultiRolloutStateLogger(
        engine=ENGINE,
        n_rollouts=N_ROLLOUTS,
        seed=SEED,
        dt=dt,
        duration_seconds=sim_config.duration_seconds,
        filename_stem=_hdf5_filename_stem(dt),
    )

    with tqdm(total=N_ROLLOUTS, desc=f"Rollouts [{ENGINE}]", unit="rollout", position=1, leave=True) as pbar:
        for i in range(N_ROLLOUTS):
            ic = _sample_initial_conditions(rng)
            if RANDOMIZE_PLANES:
                plane_coeff = _plane_coeff_for_rollout(rng, ic)
                if plane_coeff == (0.0, 0.0, 1.0, 0.0):
                    plane_crossing_fallbacks += 1
            else:
                plane_coeff = PLANE_COEFF
            pbar.set_postfix(q0=f"{ic[0]:.2f}", q1=f"{ic[1]:.2f}", qd0=f"{ic[2]:.2f}", qd1=f"{ic[3]:.2f}")
            multi_logger.start_rollout(ic)

            sim = Simulator(
                sim_config=sim_config,
                render_config=render_config,
                engine_config=engine_config,
                logging_config=logging_config,
                plane_coefficients=plane_coeff,
                initial_state=ic,
                state_logger=multi_logger,
            )
            sim.run()
            multi_logger.finish_rollout()
            pbar.update(1)

    if RANDOMIZE_PLANES and plane_crossing_fallbacks:
        print(
            f"[log_multiple_rollouts] Plane IC crossing fallback (0,0,1,0): "
            f"{plane_crossing_fallbacks} / {N_ROLLOUTS} rollouts"
        )

    multi_logger.save()


if __name__ == "__main__":
    main()
