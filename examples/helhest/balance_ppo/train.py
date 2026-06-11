"""PPO trainer for the Helhest balance task, tied to Ostrich's parallel sim.

Architecture overview:
- Inherits OstrichDifferentiableSimulator for model build / state allocation /
  collision pipeline / viewer plumbing, but bypasses ``diff_step`` entirely.
- Each PPO iteration runs one fixed-length episode (T sim steps) across N
  parallel worlds. Per step: extract obs from state, sample action via the
  PyTorch policy, write into ``controls[t].joint_target_vel``, run
  ``collision + solver.step``, compute reward, push transition to buffer.
- Zero-copy obs/action handoff via ``wp.to_torch``: the policy reads/writes
  the same GPU memory the solver uses, so PPO 'tied to sim' has no sync cost
  beyond the policy forward/backward.
- Video and metric logging via wandb. Video is a 2D side-view animation of
  world 0's chassis, generated with matplotlib (no GL viewer dependency).

DOF layout: [0..5] free base joint, [6] left wheel, [7] right wheel,
[8] rear wheel — matches helhest_balance_bundled.py.
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import time
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import newton
import torch
import warp as wp
from newton import Model

from ostrich import (
    OstrichDifferentiableSimulator,
    OstrichEngineConfig,
    ComplianceConfig,
    ContactsConfig,
    LinearSolverConfig,
    LinesearchConfig,
    LoggingConfig,
    NewtonRaphsonConfig,
    RenderingConfig,
    SimulationConfig,
)
from examples.helhest.common import HelhestConfig, create_helhest_model
from examples.helhest.balance_ppo.actor_critic import ActorCritic
from examples.helhest.balance_ppo.ppo_trainer import (
    PPOConfig,
    PPOTrainer,
    explained_variance,
)
from examples.helhest.balance_ppo.rollout_buffer import RolloutBuffer

os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

# Same balance pose as helhest_balance_bundled.py: chassis tilted backward by
# ~47° around +y so its CoM sits over the rear-wheel contact patch.
BALANCE_PITCH = 0.825

WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3

OBS_DIM = 12
ACT_DIM = 3


# -----------------------------------------------------------------------------
# Reward (pure torch — runs on the same device as the policy).
# -----------------------------------------------------------------------------


@dataclass
class RewardWeights:
    alive_bonus: float = 1.0
    rot: float = 5.0
    pos: float = 0.5
    action_rate: float = 0.05
    action_l2: float = 1e-3
    alive_threshold: float = 0.85   # cos(pitch_err) above this counts as 'upright'


def compute_reward(
    cos_pitch_err: torch.Tensor,   # [N]
    x_err: torch.Tensor, y_err: torch.Tensor,   # [N]
    action: torch.Tensor, prev_action: torch.Tensor,   # [N, 3]
    w: RewardWeights,
) -> torch.Tensor:
    rot_pen = (1.0 - cos_pitch_err).clamp_min(0.0)        # 0 at target, → 2 when flipped
    pos_pen = x_err.pow(2) + y_err.pow(2)
    alive   = (cos_pitch_err > w.alive_threshold).float()
    a_rate  = (action - prev_action).pow(2).sum(-1)
    a_l2    = action.pow(2).sum(-1)
    return (
        w.alive_bonus * alive
        - w.rot * rot_pen
        - w.pos * pos_pen
        - w.action_rate * a_rate
        - w.action_l2 * a_l2
    )


# -----------------------------------------------------------------------------
# Side-view video rendering for wandb.
# -----------------------------------------------------------------------------


def _render_video_frames(
    chassis_xyz_pitch: np.ndarray,   # [T+1, 4] for one world: (x, y, z, pitch)
    target_pitch: float,
    iteration: int,
    fps: int = 30,
) -> np.ndarray:
    """Returns uint8 array [T+1, 3, H, W] suitable for ``wandb.Video``.

    Renders a side-view (x-z plane): chassis as a tilted rectangle, ground
    line, target-pitch indicator. Cheap matplotlib-only implementation —
    avoids any GL/USD dependency."""
    T_plus_1 = chassis_xyz_pitch.shape[0]
    chassis_w, chassis_h = 1.4, 0.4

    fig, ax = plt.subplots(figsize=(5.0, 3.5), dpi=80)
    fig.canvas.draw()  # initialize backend
    frames = []

    xs = chassis_xyz_pitch[:, 0]
    zs = chassis_xyz_pitch[:, 2]
    x_lo, x_hi = float(xs.min()) - 1.5, float(xs.max()) + 1.5
    z_hi = max(2.5, float(zs.max()) + 0.6)

    for t in range(T_plus_1):
        ax.clear()
        x, _y, z, pitch = chassis_xyz_pitch[t]

        ax.axhline(0.0, color="0.4", linewidth=1.5, zorder=0)

        # Chassis: rectangle centered at origin, then rotated by pitch and
        # translated to (x, z). Note matplotlib rotation is counter-clockwise
        # in screen coords, with +y up; pitch around +y in 3D appears as
        # clockwise rotation in the x-z view → negate.
        rect = mpatches.Rectangle(
            (-chassis_w / 2.0, -chassis_h / 2.0), chassis_w, chassis_h,
            facecolor="#3070c0", edgecolor="#1a3a60", linewidth=1.5,
        )
        rect.set_transform(
            mtransforms.Affine2D().rotate(-float(pitch)).translate(float(x), float(z))
            + ax.transData
        )
        ax.add_patch(rect)

        # Target-pitch ghost (faint outline at chassis position with target tilt).
        ghost = mpatches.Rectangle(
            (-chassis_w / 2.0, -chassis_h / 2.0), chassis_w, chassis_h,
            facecolor="none", edgecolor="#c04040", linewidth=1.0, linestyle="--",
        )
        ghost.set_transform(
            mtransforms.Affine2D().rotate(-target_pitch).translate(float(x), float(z))
            + ax.transData
        )
        ax.add_patch(ghost)

        # Trail of past chassis positions.
        if t > 0:
            ax.plot(xs[: t + 1], zs[: t + 1], color="#3070c0", alpha=0.4, linewidth=1.0)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(-0.3, z_hi)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"iter {iteration}  step {t}/{T_plus_1 - 1}", fontsize=10)

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]   # [H, W, 3] uint8
        frames.append(img.transpose(2, 0, 1))                  # [C, H, W]

    plt.close(fig)
    return np.stack(frames, axis=0)             # [T+1, 3, H, W]


# -----------------------------------------------------------------------------
# Checkpoint I/O.
# -----------------------------------------------------------------------------


def _build_checkpoint(
    policy: ActorCritic, optim: torch.optim.Optimizer, iter_num: int,
    best_alive: float, args_dict: dict | None = None,
) -> dict:
    return {
        "iter":                 iter_num,
        "policy_state_dict":    policy.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "best_alive_frac":      best_alive,
        # Self-describing fields so the file loads without the trainer.
        "obs_dim":              policy.obs_dim,
        "act_dim":              policy.act_dim,
        "v_max":                policy.v_max,
        "balance_pitch":        BALANCE_PITCH,
        "args":                 args_dict,
    }


def load_policy(
    path: str | pathlib.Path,
    device: torch.device | str = "cuda",
    hidden: int = 64,
) -> tuple[ActorCritic, dict]:
    """Standalone loader: rebuilds the actor-critic from a checkpoint without
    needing the trainer. Returns (policy, raw_ckpt_dict). Use this from the
    bundled-gradient fine-tune phase to warm-start from PPO weights."""
    device = torch.device(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    policy = ActorCritic(
        obs_dim=ckpt["obs_dim"], act_dim=ckpt["act_dim"],
        hidden=hidden, v_max=ckpt["v_max"],
    ).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    return policy, ckpt


# -----------------------------------------------------------------------------
# Trainer.
# -----------------------------------------------------------------------------


class HelhestBalancePPO(OstrichDifferentiableSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: OstrichEngineConfig,
        logging_config: LoggingConfig,
        policy_hidden: int = 64,
        v_max: float = 8.0,
        lr: float = 3e-4,
        ppo_cfg: PPOConfig | None = None,
        reward_weights: RewardWeights | None = None,
        wandb_run=None,
        video_every: int = 25,
        checkpoint_dir: pathlib.Path | None = None,
        checkpoint_every: int = 25,
        args_for_ckpt: dict | None = None,
    ):
        super().__init__(sim_config, render_config, engine_config, logging_config)

        self.N = sim_config.num_worlds
        self.T = self.clock.total_sim_steps
        self.dt = self.clock.dt

        wp_device = self.solver.model.device
        self.torch_device = torch.device("cuda" if wp_device.is_cuda else "cpu")

        self.policy = ActorCritic(
            obs_dim=OBS_DIM, act_dim=ACT_DIM,
            hidden=policy_hidden, v_max=v_max,
        ).to(self.torch_device)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.ppo = PPOTrainer(
            policy=self.policy, optimizer=self.optim,
            config=ppo_cfg if ppo_cfg is not None else PPOConfig(),
        )

        self.buf = RolloutBuffer(
            T=self.T, N=self.N, obs_dim=OBS_DIM, act_dim=ACT_DIM,
            device=self.torch_device,
        )

        self.w = reward_weights if reward_weights is not None else RewardWeights()
        self.target_cos = math.cos(BALANCE_PITCH)
        self.target_sin = math.sin(BALANCE_PITCH)

        self.wandb_run = wandb_run
        self.video_every = video_every
        self._shape_logged = False

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.args_for_ckpt = args_for_ckpt
        self.best_alive_frac = -float("inf")
        self.start_iter = 0
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.track_body(body_idx=0, name="chassis", color=(0.0, 0.5, 1.0))

    # --- Checkpointing ------------------------------------------------------

    def _save(self, path: pathlib.Path, iter_num: int, artifact_alias: str | None = None) -> None:
        """Write a checkpoint locally and (optionally) push it to wandb.

        ``artifact_alias=None`` → ``wandb.save`` (rolling file, no version history,
        suitable for ``latest.pt`` we may resume from). ``artifact_alias`` set →
        log as a wandb Artifact with that alias (``best`` / ``final``), giving
        proper version history and lineage for downstream runs that consume the
        weights."""
        ckpt = _build_checkpoint(
            self.policy, self.optim, iter_num,
            best_alive=self.best_alive_frac, args_dict=self.args_for_ckpt,
        )
        torch.save(ckpt, path)

        run = self.wandb_run
        if run is None or getattr(run, "disabled", False):
            return

        try:
            import wandb
            if artifact_alias is None:
                wandb.save(str(path), base_path=str(path.parent), policy="now")
                return

            artifact = wandb.Artifact(
                name=f"policy-{run.id}",
                type="model",
                metadata={
                    "iter": iter_num,
                    "best_alive_frac": self.best_alive_frac,
                    "balance_pitch": BALANCE_PITCH,
                    "v_max": self.policy.v_max,
                    "obs_dim": self.policy.obs_dim,
                    "act_dim": self.policy.act_dim,
                },
            )
            artifact.add_file(str(path))
            run.log_artifact(artifact, aliases=[artifact_alias])
        except Exception as e:
            print(f"[checkpoint] wandb upload failed ({artifact_alias or 'save'}): {e}",
                  flush=True)

    def resume_from(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path)
        ckpt = torch.load(path, map_location=self.torch_device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optim.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_iter = int(ckpt.get("iter", 0)) + 1
        self.best_alive_frac = float(ckpt.get("best_alive_frac", -float("inf")))
        print(f"[checkpoint] resumed from {path} at iter {self.start_iter} "
              f"(best_alive_frac={self.best_alive_frac:.3f})", flush=True)

    # --- Model build (mirrors helhest_balance_bundled.py) -------------------

    def build_model(self) -> Model:
        self.builder.rigid_gap = 0.1
        ground_cfg = newton.ModelBuilder.ShapeConfig(mu=0.7, ke=50.0, kd=50.0, kf=50.0)
        self.builder.add_ground_plane(cfg=ground_cfg)

        initial_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), BALANCE_PITCH)
        create_helhest_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.6), initial_rot),
            control_mode="velocity",
            k_p=HelhestConfig.TARGET_KE,
            k_d=HelhestConfig.TARGET_KD,
            friction_left_right=0.7,
            friction_rear=0.35,
        )
        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=False,    # PPO doesn't need sim gradients
        )

    # --- Required by base class but unused under PPO ------------------------

    def update(self):
        raise NotImplementedError("PPO uses its own update loop; do not call diff_step.")

    def compute_loss(self):
        raise NotImplementedError("PPO uses reward, not a Warp loss.")

    # --- Observation / action / reward --------------------------------------

    def _state_features(self, state) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (obs[N,12], chassis_xyz_pitch[N,4], cos_pitch_err[N])."""
        body_q  = wp.to_torch(state.body_q).view(self.N, -1, 7)        # [x,y,z, qx,qy,qz,qw]
        body_qd = wp.to_torch(state.body_qd).view(self.N, -1, 6)       # [vx,vy,vz, ωx,ωy,ωz]
        joint_qd = wp.to_torch(state.joint_qd).view(self.N, -1)

        chassis_q  = body_q[:, 0]
        chassis_qd = body_qd[:, 0]

        if not self._shape_logged:
            print(f"[HelhestBalancePPO] body_q.shape={tuple(body_q.shape)}, "
                  f"body_qd.shape={tuple(body_qd.shape)}, "
                  f"joint_qd.shape={tuple(joint_qd.shape)}", flush=True)
            self._shape_logged = True

        pos = chassis_q[:, :3]
        qx, qy, qz, qw = chassis_q[:, 3], chassis_q[:, 4], chassis_q[:, 5], chassis_q[:, 6]
        v_lin   = chassis_qd[:, :3]
        v_ang   = chassis_qd[:, 3:6]

        sin_pitch = (2.0 * (qw * qy - qz * qx)).clamp(-1.0, 1.0)
        pitch = torch.asin(sin_pitch)
        cos_pitch = torch.cos(pitch)
        roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))

        cos_pitch_err = cos_pitch * self.target_cos + sin_pitch * self.target_sin

        wheel_speeds = joint_qd[:, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS]

        obs = torch.stack([
            sin_pitch, cos_pitch,
            v_ang[:, 1],                           # pitch rate
            roll, v_ang[:, 0],                     # roll, roll rate
            pos[:, 0], v_lin[:, 0],                # x, vx
            pos[:, 1], v_lin[:, 1],                # y, vy
            wheel_speeds[:, 0], wheel_speeds[:, 1], wheel_speeds[:, 2],
        ], dim=-1)

        chassis_state = torch.stack([pos[:, 0], pos[:, 1], pos[:, 2], pitch], dim=-1)
        return obs, chassis_state, cos_pitch_err

    def _write_action(self, control, raw_action: torch.Tensor) -> torch.Tensor:
        """Squash raw action and write into ``control.joint_target_vel`` (zero-copy).

        ``wp.to_torch`` on a ``requires_grad=True`` Warp array returns a leaf
        tensor that rejects in-place ops, so we detach before writing — same
        memory, no autograd bookkeeping (PPO's gradient path is on the policy
        params, not on the sim controls)."""
        wheel_vel = self.policy.squash(raw_action)        # [N, 3]
        target = wp.to_torch(control.joint_target_vel).detach().view(self.N, -1)
        target.zero_()
        target[:, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = wheel_vel
        return wheel_vel

    # --- Rollout ------------------------------------------------------------

    def collect_rollout(self) -> dict:
        # Fresh start: rebuild state[0] from initial joint_q/qd.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])

        prev_action = torch.zeros(self.N, ACT_DIM, device=self.torch_device)
        episode_return = torch.zeros(self.N, device=self.torch_device)
        action_rate_sum = torch.zeros(self.N, device=self.torch_device)
        alive_count = torch.zeros(self.N, device=self.torch_device)

        # Initial chassis pose (state[0]).
        with torch.no_grad():
            obs0, chassis0, _ = self._state_features(self.states[0])
        self.buf.chassis_xyz_pitch[0] = chassis0

        for t in range(self.T):
            with torch.no_grad():
                if t == 0:
                    obs_t = obs0
                else:
                    obs_t, chassis_t, _ = self._state_features(self.states[t])
                    self.buf.chassis_xyz_pitch[t] = chassis_t

                raw_action, logp, value = self.policy.act(obs_t)

            wheel_vel = self._write_action(self.controls[t], raw_action)

            self.collision_pipeline.collide(self.states[t], self.contacts)
            self.solver.step(
                state_in=self.states[t],
                state_out=self.states[t + 1],
                control=self.controls[t],
                contacts=self.contacts,
                dt=self.dt,
            )

            with torch.no_grad():
                _, chassis_next, cos_err_next = self._state_features(self.states[t + 1])
                reward = compute_reward(
                    cos_pitch_err=cos_err_next,
                    x_err=chassis_next[:, 0], y_err=chassis_next[:, 1],
                    action=wheel_vel, prev_action=prev_action,
                    w=self.w,
                )

            self.buf.add(t, obs_t, raw_action, logp, value, reward)
            episode_return += reward
            action_rate_sum += (wheel_vel - prev_action).pow(2).sum(-1)
            alive_count += (cos_err_next > self.w.alive_threshold).float()
            prev_action = wheel_vel

        with torch.no_grad():
            obs_last, chassis_last, _ = self._state_features(self.states[self.T])
            self.buf.chassis_xyz_pitch[self.T] = chassis_last
            last_value = self.policy.value(obs_last)

        self.buf.compute_gae(last_value, gamma=0.99, lam=0.95)

        # Diagnostics (averaged across worlds).
        return {
            "episode_return":     float(episode_return.mean()),
            "alive_frac":         float(alive_count.mean() / self.T),
            "final_pitch_err":    float(torch.acos(
                                      (chassis_last[:, 3].cos() * self.target_cos
                                       + chassis_last[:, 3].sin() * self.target_sin).clamp(-1, 1)
                                  ).mean()),
            "final_x_err":        float(chassis_last[:, 0].abs().mean()),
            "action_rate_mean":   float(action_rate_sum.mean() / self.T),
            "log_std_mean":       float(self.policy.log_std.detach().mean()),
        }

    # --- Train --------------------------------------------------------------

    def train(self, iterations: int):
        for it in range(self.start_iter, iterations):
            t0 = time.perf_counter()
            rollout_metrics = self.collect_rollout()
            wp.synchronize()
            torch.cuda.synchronize() if self.torch_device.type == "cuda" else None
            rollout_wall = time.perf_counter() - t0

            ev = explained_variance(self.buf.values.flatten(), self.buf.returns.flatten())
            train_metrics = self.ppo.update(self.buf)
            update_wall = time.perf_counter() - t0 - rollout_wall

            log = {
                "iter": it,
                "rollout/episode_return": rollout_metrics["episode_return"],
                "rollout/alive_frac":     rollout_metrics["alive_frac"],
                "rollout/final_pitch_err_rad": rollout_metrics["final_pitch_err"],
                "rollout/final_x_err":    rollout_metrics["final_x_err"],
                "rollout/action_rate":    rollout_metrics["action_rate_mean"],
                "rollout/wall_s":         rollout_wall,
                "train/policy_loss":      train_metrics["policy_loss"],
                "train/value_loss":       train_metrics["value_loss"],
                "train/entropy":          train_metrics["entropy"],
                "train/approx_kl":        train_metrics["approx_kl"],
                "train/clip_fraction":    train_metrics["clip_fraction"],
                "train/grad_norm":        train_metrics["grad_norm"],
                "train/epochs_run":       train_metrics["epochs_run"],
                "train/explained_var":    ev,
                "train/log_std_mean":     rollout_metrics["log_std_mean"],
                "train/update_wall_s":    update_wall,
            }

            if self.video_every > 0 and (it % self.video_every == 0 or it == iterations - 1):
                chassis_np = self.buf.chassis_xyz_pitch[:, 0].detach().cpu().numpy()
                frames = _render_video_frames(
                    chassis_np, target_pitch=BALANCE_PITCH, iteration=it,
                )
                if self.wandb_run is not None:
                    import wandb
                    log["video/world0"] = wandb.Video(frames, fps=30, format="mp4")

            if self.wandb_run is not None:
                self.wandb_run.log(log, step=it)

            print(
                f"Iter {it:4d} | "
                f"return={rollout_metrics['episode_return']:>8.2f} "
                f"alive={rollout_metrics['alive_frac'] * 100:>5.1f}% "
                f"pitch_err={rollout_metrics['final_pitch_err']:.3f}rad "
                f"| pl={train_metrics['policy_loss']:+.3f} "
                f"vl={train_metrics['value_loss']:.3f} "
                f"H={train_metrics['entropy']:.3f} "
                f"kl={train_metrics['approx_kl']:+.4f} "
                f"ev={ev:+.2f} | "
                f"wall={rollout_wall + update_wall:.2f}s",
                flush=True,
            )

            if self.checkpoint_dir is not None:
                if rollout_metrics["alive_frac"] > self.best_alive_frac:
                    self.best_alive_frac = rollout_metrics["alive_frac"]
                    self._save(self.checkpoint_dir / "best.pt", it, artifact_alias="best")
                    if self.wandb_run is not None:
                        self.wandb_run.summary["best_alive_frac"] = self.best_alive_frac
                        self.wandb_run.summary["best_alive_iter"] = it
                if (it + 1) % self.checkpoint_every == 0:
                    self._save(self.checkpoint_dir / "latest.pt", it)
                if it == iterations - 1:
                    self._save(self.checkpoint_dir / "final.pt", it, artifact_alias="final")


# -----------------------------------------------------------------------------
# Engine config — copied from helhest_balance_bundled.py.
# -----------------------------------------------------------------------------


def _make_default_engine_config() -> OstrichEngineConfig:
    return OstrichEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=16, backtrack_min_iter=12, atol=1e-3),
        linear=LinearSolverConfig(max_iters=16, atol=1e-3, tol=1e-3, regularization=1e-6),
        compliance=ComplianceConfig(joint=6e-8, contact=1e-10, friction=1e-6),
        linesearch=LinesearchConfig(enabled=False),
        contacts=ContactsConfig(max_per_world=64),
    )


# -----------------------------------------------------------------------------
# CLI.
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-worlds", type=int, default=16,
                        help="Parallel envs (each is one PPO worker).")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--v-max", type=float, default=8.0,
                        help="Max wheel target velocity (rad/s); action is tanh-squashed.")

    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--target-kl", type=float, default=0.02)

    parser.add_argument("--w-alive", type=float, default=1.0)
    parser.add_argument("--w-rot", type=float, default=5.0)
    parser.add_argument("--w-pos", type=float, default=0.5)
    parser.add_argument("--w-action-rate", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", type=str, default="ostrich-helhest-balance")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--video-every", type=int, default=25,
                        help="Log a side-view video every K iters; 0 disables.")

    parser.add_argument("--checkpoint-dir", type=str, default="runs",
                        help="Parent dir for run checkpoints. Run-specific subdir "
                             "(named after wandb run) is created inside. Empty disables.")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="Save 'latest.pt' every K iters. 'best.pt' updates whenever "
                             "alive_frac improves; 'final.pt' is written at end.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint .pt to resume training from.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sim_config = SimulationConfig(
        duration_seconds=4.0,
        target_timestep_seconds=5e-2,
        num_worlds=args.num_worlds,
    )
    render_config = RenderingConfig(
        vis_type="null",
        target_fps=30, usd_file=None,
        world_offset_x=20.0, world_offset_y=20.0,
    )
    engine_config = _make_default_engine_config()
    logging_config = LoggingConfig()

    ppo_cfg = PPOConfig(
        epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        clip_ratio=args.clip_ratio,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        target_kl=args.target_kl,
    )
    reward_w = RewardWeights(
        alive_bonus=args.w_alive, rot=args.w_rot,
        pos=args.w_pos, action_rate=args.w_action_rate,
    )

    import wandb
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config={
            **vars(args),
            "obs_dim": OBS_DIM,
            "act_dim": ACT_DIM,
            "T": int(round(sim_config.duration_seconds / sim_config.target_timestep_seconds)),
            "dt": sim_config.target_timestep_seconds,
            "balance_pitch_rad": BALANCE_PITCH,
        },
    )

    ckpt_dir: pathlib.Path | None = None
    if args.checkpoint_dir:
        run_tag = run.name if (run is not None and run.name) else f"run_{int(time.time())}"
        ckpt_dir = pathlib.Path(args.checkpoint_dir) / run_tag
        print(f"[checkpoint] writing to {ckpt_dir}", flush=True)

    sim = HelhestBalancePPO(
        sim_config, render_config, engine_config, logging_config,
        policy_hidden=args.hidden, v_max=args.v_max,
        lr=args.lr, ppo_cfg=ppo_cfg, reward_weights=reward_w,
        wandb_run=run, video_every=args.video_every,
        checkpoint_dir=ckpt_dir,
        checkpoint_every=args.checkpoint_every,
        args_for_ckpt=vars(args),
    )

    if args.resume is not None:
        sim.resume_from(args.resume)

    try:
        sim.train(iterations=args.iterations)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
