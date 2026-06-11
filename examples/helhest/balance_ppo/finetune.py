"""Fine-tune a PPO-trained policy via Axion's exact gradients.

Pipeline:
  PPO → policy with ~88% alive_frac → load weights here → run T-step rollouts
  with the policy producing controls → Axion's adjoint backprops loss into
  ``joint_target_vel.grad`` → we chain that gradient through the saved policy
  graph at each timestep → Adam step on the policy parameters.

Why this works once PPO has done its job: gradient methods need a starting
point in a good basin (here, 'standing up most of the time'), and PPO's
stochastic gradients are bad at the *last* refinement (chatter, edge-of-
stable wobble) — exactly where smooth losses + exact gradients shine.

Compared to ``helhest_balance_bundled.py``: same loss kernels (threshold
orient + pos + smoothness + reg), same Axion ``diff_step`` adjoint, but the
optimization variable is the *MLP weights* instead of K spline knots. No
bundled smoothing — pure exact gradient (so ``num_worlds=1`` is the default;
worlds would be redundant when actions are deterministic).

Usage:
  python -m examples.helhest.balance_ppo.finetune --checkpoint runs/<run>/best.pt
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import time

import newton
import numpy as np
import torch
import warp as wp
from newton import Model

from axion import (
    AxionDifferentiableSimulator,
    AxionEngineConfig,
    ComplianceConfig,
    ContactsConfig,
    LinearSolverConfig,
    LinesearchConfig,
    LoggingConfig,
    NewtonRaphsonConfig,
    RenderingConfig,
    SimulationConfig,
)
from examples.helhest.balance_ppo.actor_critic import ActorCritic
from examples.helhest.balance_ppo.train import (
    ACT_DIM,
    BALANCE_PITCH,
    NUM_WHEEL_DOFS,
    OBS_DIM,
    WHEEL_DOF_OFFSET,
    HelhestBalancePPO,
    _build_checkpoint,
    _make_default_engine_config,
    _render_video_frames,
)
from examples.helhest.balance_ppo.ppo_trainer import PPOConfig
from examples.helhest.common import HelhestConfig, create_helhest_model

os.environ.setdefault("PYOPENGL_PLATFORM", "glx")


# -----------------------------------------------------------------------------
# Loss kernels (replicated from helhest_balance_bundled.py — same formulation
# the original spline-bundled trainer uses).
# -----------------------------------------------------------------------------


@wp.kernel
def threshold_balance_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_rot: wp.quat,
    target_pos: wp.vec3,
    weight_rot: float,
    weight_pos: float,
    alive_threshold: float,
    loss: wp.array(dtype=wp.float32),
):
    sim_step, w = wp.tid()
    xform = body_pose[sim_step, w, 0]
    pos = wp.transform_get_translation(xform)
    rot = wp.transform_get_rotation(xform)

    pos_err = wp.vec3(pos[0] - target_pos[0], pos[1] - target_pos[1], 0.0)
    p_loss = wp.dot(pos_err, pos_err)

    up_local = wp.vec3(0.0, 0.0, 1.0)
    current_up = wp.quat_rotate(rot, up_local)
    target_up = wp.quat_rotate(target_rot, up_local)
    similarity = wp.dot(current_up, target_up)
    deficit = wp.max(0.0, alive_threshold - similarity)
    r_loss = deficit * deficit
    wp.atomic_add(loss, 0, weight_pos * p_loss + weight_rot * r_loss)


@wp.kernel
def regularization_kernel(
    target_vel: wp.array(dtype=wp.float32, ndim=3),
    wheel_dof_offset: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    sim_step, w, wheel_idx = wp.tid()
    dof_idx = wheel_dof_offset + wheel_idx
    v = target_vel[sim_step, w, dof_idx]
    wp.atomic_add(loss, 0, weight * v * v)


@wp.kernel
def smoothness_kernel(
    target_vel: wp.array(dtype=wp.float32, ndim=3),
    wheel_dof_offset: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    sim_step, w, wheel_idx = wp.tid()
    dof_idx = wheel_dof_offset + wheel_idx
    diff = target_vel[sim_step + 1, w, dof_idx] - target_vel[sim_step, w, dof_idx]
    wp.atomic_add(loss, 0, weight * diff * diff)


@wp.kernel
def diag_alive_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_rot: wp.quat,
    threshold: float,
    out: wp.array(dtype=wp.int32),
):
    sim_step, w = wp.tid()
    xform = body_pose[sim_step, w, 0]
    rot = wp.transform_get_rotation(xform)
    up_local = wp.vec3(0.0, 0.0, 1.0)
    current_up = wp.quat_rotate(rot, up_local)
    target_up = wp.quat_rotate(target_rot, up_local)
    if wp.dot(current_up, target_up) > threshold:
        wp.atomic_add(out, 0, 1)


# -----------------------------------------------------------------------------
# Trainer.
# -----------------------------------------------------------------------------


class HelhestBalanceFineTune(AxionDifferentiableSimulator):
    """Replaces the spline-knot optimization variable with a full MLP, gradient
    descended via Axion's exact adjoint.

    Reuses HelhestBalancePPO's observation extraction and policy class without
    inheriting (we want the AxionDifferentiableSimulator pipeline, not the PPO
    pipeline). The helhest-specific helpers ``_state_features`` and the build
    are duplicated minimally below."""

    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: AxionEngineConfig,
        logging_config: LoggingConfig,
        checkpoint_path: pathlib.Path,
        policy_hidden: int = 64,
        v_max: float = 8.0,
        lr: float = 1e-4,
        weight_rot: float = 200.0,
        weight_pos: float = 1.0,
        weight_smooth: float = 1e-1,
        weight_reg: float = 1e-4,
        alive_threshold: float = 0.85,
        grad_clip: float = 0.5,
        wandb_run=None,
        video_every: int = 25,
        checkpoint_dir: pathlib.Path | None = None,
        checkpoint_every: int = 25,
        args_for_ckpt: dict | None = None,
    ):
        super().__init__(sim_config, render_config, engine_config, logging_config)

        self.N = sim_config.num_worlds
        self.T = self.clock.total_sim_steps

        wp_device = self.solver.model.device
        self.torch_device = torch.device("cuda" if wp_device.is_cuda else "cpu")

        # Build empty policy then load PPO weights.
        self.policy = ActorCritic(
            obs_dim=OBS_DIM, act_dim=ACT_DIM,
            hidden=policy_hidden, v_max=v_max,
        ).to(self.torch_device)
        ckpt = torch.load(checkpoint_path, map_location=self.torch_device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        print(f"[finetune] loaded PPO weights from {checkpoint_path} "
              f"(iter={ckpt.get('iter', '?')}, best_alive={ckpt.get('best_alive_frac', float('nan')):.3f})",
              flush=True)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.target_cos = math.cos(BALANCE_PITCH)
        self.target_sin = math.sin(BALANCE_PITCH)

        self.weight_rot = weight_rot
        self.weight_pos = weight_pos
        self.smoothness_weight = weight_smooth
        self.regularization_weight = weight_reg
        self.alive_threshold = alive_threshold
        self.grad_clip = grad_clip
        self.total_iterations = 1   # filled in by train()

        # Saved policy outputs per timestep — kept in the autograd graph so we
        # can backprop Axion's per-step grad on joint_target_vel back into the
        # MLP parameters in one shot.
        self.saved_actions: list[torch.Tensor] = []

        self.wandb_run = wandb_run
        self.video_every = video_every
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.args_for_ckpt = args_for_ckpt
        self.best_alive_frac = -float("inf")
        self.start_iter = 0
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._diag_alive = wp.zeros(1, dtype=wp.int32)

        self.track_body(body_idx=0, name="chassis", color=(0.0, 0.5, 1.0))

    # --- Model build (must require_grad=True for Axion adjoint) -------------

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
            requires_grad=True,
        )

    # --- Helhest helpers (mirrors HelhestBalancePPO; kept inline here so this
    # module can run independently of train.py's class) ---------------------

    def _state_features(self, state):
        body_q  = wp.to_torch(state.body_q).view(self.N, -1, 7)
        body_qd = wp.to_torch(state.body_qd).view(self.N, -1, 6)
        joint_qd = wp.to_torch(state.joint_qd).view(self.N, -1)

        chassis_q  = body_q[:, 0]
        chassis_qd = body_qd[:, 0]
        pos = chassis_q[:, :3]
        qx, qy, qz, qw = chassis_q[:, 3], chassis_q[:, 4], chassis_q[:, 5], chassis_q[:, 6]
        v_lin = chassis_qd[:, :3]
        v_ang = chassis_qd[:, 3:6]

        sin_pitch = (2.0 * (qw * qy - qz * qx)).clamp(-1.0, 1.0)
        pitch = torch.asin(sin_pitch)
        cos_pitch = torch.cos(pitch)
        roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))

        cos_pitch_err = cos_pitch * self.target_cos + sin_pitch * self.target_sin

        wheel_speeds = joint_qd[:, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS]
        obs = torch.stack([
            sin_pitch, cos_pitch, v_ang[:, 1],
            roll, v_ang[:, 0],
            pos[:, 0], v_lin[:, 0],
            pos[:, 1], v_lin[:, 1],
            wheel_speeds[:, 0], wheel_speeds[:, 1], wheel_speeds[:, 2],
        ], dim=-1)
        chassis_state = torch.stack([pos[:, 0], pos[:, 1], pos[:, 2], pitch], dim=-1)
        return obs, chassis_state, cos_pitch_err

    def update(self):
        # See _backprop_through_policy — deferred until after diff_step.
        pass

    # --- Custom forward + backward — overrides parent's --------------------

    def _forward_backward(self):
        """Replaces ``AxionDifferentiableSimulator._forward_backward``. The
        parent's version assumes ``controls[t].joint_target_vel`` is set in
        advance; here we set it on-the-fly from the policy at each step,
        keeping the (grad-tracked) policy output around so we can chain Axion's
        adjoint into the MLP after the backward pass."""
        self.trajectory.zero_grad()
        self.saved_actions = []

        for t in range(self.T):
            with torch.no_grad():
                obs_t, _, _ = self._state_features(self.states[t])

            # Policy forward keeps the autograd graph attached. Deterministic
            # mean (no exploration noise) — exact gradient flows through.
            mean_raw = self.policy.actor_head(self.policy.actor_trunk(obs_t))   # [N, 3]
            wheel_vel = self.policy.squash(mean_raw)                            # [N, 3]
            self.saved_actions.append(wheel_vel)

            # Detach for the sim path (we re-inject the gradient manually below).
            target = wp.to_torch(self.controls[t].joint_target_vel).detach().view(self.N, -1)
            target.zero_()
            target[:, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = wheel_vel.detach()

            self.collision_pipeline.collide(self.states[t], self.contacts)
            self.solver.step(
                state_in=self.states[t],
                state_out=self.states[t + 1],
                control=self.controls[t],
                contacts=self.contacts,
                dt=self.clock.dt,
            )
            self.trajectory.save_step(t, self.solver.data, self.solver.axion_contacts)

        self.tape.zero()
        with self.tape:
            self.compute_loss()
        self.tape.backward(self.loss)

        for t in range(self.T - 1, -1, -1):
            self.trajectory.load_step(t, self.solver.data, self.solver.axion_contacts)
            self.solver.data.zero_gradients()
            self.solver.step_backward()
            self.trajectory.save_gradients(t, self.solver.data)
            self.trajectory.save_pose_gradients(t, self.solver.data)
            if self.solver.config.adjoint.gradient_normalization and t > 0:
                self.trajectory.normalize_gradients(t)

    def compute_loss(self):
        T_plus_1 = self.trajectory.body_pose.shape[0]
        target_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), BALANCE_PITCH)
        target_pos = wp.vec3(0.0, 0.0, 0.0)
        device = self.solver.model.device

        wp.launch(
            kernel=threshold_balance_loss_kernel,
            dim=(T_plus_1, self.N),
            inputs=[
                self.trajectory.body_pose, target_rot, target_pos,
                self.weight_rot, self.weight_pos, float(self.alive_threshold),
            ],
            outputs=[self.loss], device=device,
        )
        wp.launch(
            kernel=regularization_kernel,
            dim=(self.T, self.N, NUM_WHEEL_DOFS),
            inputs=[self.trajectory.joint_target_vel, WHEEL_DOF_OFFSET,
                    self.regularization_weight],
            outputs=[self.loss], device=device,
        )
        wp.launch(
            kernel=smoothness_kernel,
            dim=(self.T - 1, self.N, NUM_WHEEL_DOFS),
            inputs=[self.trajectory.joint_target_vel, WHEEL_DOF_OFFSET,
                    self.smoothness_weight],
            outputs=[self.loss], device=device,
        )

    # --- Backprop into policy params via the saved graph -------------------

    def _backprop_through_policy(self) -> float:
        """Take Axion's per-step ``joint_target_vel.grad`` and chain it back
        into the MLP using the saved (still-attached) policy outputs.

        Mathematically: the surrogate ``Σ_t (saved_action[t] · grad_target_vel[t]).sum()``
        has gradient w.r.t. policy params equal to ``Σ_t (∂action_t / ∂θ)^T · grad_target_vel[t]`` —
        exactly ``∂L/∂θ``. One ``.backward()`` does the whole chain."""
        grad_tv = wp.to_torch(self.trajectory.joint_target_vel.grad).detach()
        grad_tv = grad_tv.view(self.T, self.N, -1)
        grad_wheel = grad_tv[:, :, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS]

        self.policy.zero_grad()
        # Average over worlds so the effective gradient magnitude doesn't scale with N.
        scale = 1.0 / float(self.N)
        surrogate = sum(
            (self.saved_actions[t] * grad_wheel[t]).sum() * scale
            for t in range(self.T)
        )
        surrogate.backward()

        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.grad_clip
        ))
        self.optim.step()

        self.tape.zero()
        self.loss.zero_()
        self.trajectory.joint_target_vel.grad.zero_()
        self.saved_actions = []
        return grad_norm

    # --- Diagnostics --------------------------------------------------------

    def _alive_frac(self) -> float:
        self._diag_alive.zero_()
        T_plus_1 = self.trajectory.body_pose.shape[0]
        target_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), BALANCE_PITCH)
        wp.launch(
            kernel=diag_alive_kernel,
            dim=(T_plus_1, self.N),
            inputs=[self.trajectory.body_pose, target_rot, float(self.alive_threshold)],
            outputs=[self._diag_alive], device=self.solver.model.device,
        )
        wp.synchronize()
        return float(self._diag_alive.numpy()[0]) / float(T_plus_1 * self.N)

    def _chassis_xyz_pitch_world0(self) -> np.ndarray:
        body_pose = self.trajectory.body_pose.numpy()    # [T+1, N, num_bodies]
        chassis = body_pose[:, 0, 0]                     # [T+1] of wp.transform → 7 floats
        out = np.zeros((chassis.shape[0], 4), dtype=np.float32)
        for t in range(chassis.shape[0]):
            xform = chassis[t]
            out[t, 0] = xform[0]
            out[t, 1] = xform[1]
            out[t, 2] = xform[2]
            qx, qy, qz, qw = xform[3], xform[4], xform[5], xform[6]
            sin_p = max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
            out[t, 3] = float(np.arcsin(sin_p))
        return out

    # --- Checkpointing ------------------------------------------------------

    def _save(self, path: pathlib.Path, iter_num: int, artifact_alias: str | None = None) -> None:
        ckpt = _build_checkpoint(
            self.policy, self.optim, iter_num,
            best_alive=self.best_alive_frac, args_dict=self.args_for_ckpt,
        )
        ckpt["finetuned"] = True   # marker so downstream code can distinguish PPO vs FT
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
                name=f"policy-finetune-{run.id}",
                type="model",
                metadata={
                    "iter": iter_num,
                    "best_alive_frac": self.best_alive_frac,
                    "balance_pitch": BALANCE_PITCH,
                    "v_max": self.policy.v_max,
                    "obs_dim": self.policy.obs_dim,
                    "act_dim": self.policy.act_dim,
                    "finetuned": True,
                },
            )
            artifact.add_file(str(path))
            run.log_artifact(artifact, aliases=[artifact_alias])
        except Exception as e:
            print(f"[checkpoint] wandb upload failed ({artifact_alias or 'save'}): {e}",
                  flush=True)

    # --- Train --------------------------------------------------------------

    def train(self, iterations: int):
        self.total_iterations = iterations

        for it in range(self.start_iter, iterations):
            t0 = time.perf_counter()
            self.diff_step()
            wp.synchronize()
            wall_fb = time.perf_counter() - t0

            curr_loss = float(self.loss.numpy()[0]) / self.N
            alive_frac = self._alive_frac()

            grad_norm = self._backprop_through_policy()
            wp.synchronize()
            wall_total = time.perf_counter() - t0

            log = {
                "iter": it,
                "finetune/loss":         curr_loss,
                "finetune/alive_frac":   alive_frac,
                "finetune/grad_norm":    grad_norm,
                "finetune/log_std_mean": float(self.policy.log_std.detach().mean()),
                "finetune/wall_s":       wall_total,
                "finetune/wall_fb_s":    wall_fb,
            }

            if self.video_every > 0 and (it % self.video_every == 0 or it == iterations - 1):
                chassis_np = self._chassis_xyz_pitch_world0()
                frames = _render_video_frames(
                    chassis_np, target_pitch=BALANCE_PITCH, iteration=it,
                )
                if self.wandb_run is not None:
                    import wandb
                    log["video/world0"] = wandb.Video(frames, fps=30, format="mp4")

            if self.wandb_run is not None:
                self.wandb_run.log(log, step=it)

            print(
                f"Iter {it:4d} | loss={curr_loss:>9.2f} alive={alive_frac * 100:>5.1f}% "
                f"‖∇‖={grad_norm:.3f} wall={wall_total:.2f}s",
                flush=True,
            )

            if self.checkpoint_dir is not None:
                if alive_frac > self.best_alive_frac:
                    self.best_alive_frac = alive_frac
                    self._save(self.checkpoint_dir / "best.pt", it, artifact_alias="best")
                    if self.wandb_run is not None:
                        self.wandb_run.summary["best_alive_frac"] = self.best_alive_frac
                        self.wandb_run.summary["best_alive_iter"] = it
                if (it + 1) % self.checkpoint_every == 0:
                    self._save(self.checkpoint_dir / "latest.pt", it)
                if it == iterations - 1:
                    self._save(self.checkpoint_dir / "final.pt", it, artifact_alias="final")


# -----------------------------------------------------------------------------
# CLI.
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to PPO checkpoint (e.g. runs/<run>/best.pt).")

    parser.add_argument("--num-worlds", type=int, default=1,
                        help="Parallel worlds. With deterministic policy + same init, worlds are "
                             "redundant; default 1 unless you want to run a quick parallel "
                             "benchmark.")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Fine-tune LR (default 3× smaller than PPO).")
    parser.add_argument("--hidden", type=int, default=64,
                        help="Must match the loaded checkpoint's hidden size.")
    parser.add_argument("--v-max", type=float, default=8.0)
    parser.add_argument("--grad-clip", type=float, default=0.5)

    parser.add_argument("--weight-rot", type=float, default=200.0)
    parser.add_argument("--weight-pos", type=float, default=1.0)
    parser.add_argument("--weight-smooth", type=float, default=0.1)
    parser.add_argument("--weight-reg", type=float, default=1e-4)
    parser.add_argument("--alive-threshold", type=float, default=0.85)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", type=str, default="axion-helhest-balance")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--video-every", type=int, default=25)

    parser.add_argument("--checkpoint-dir", type=str, default="runs")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sim_config = SimulationConfig(
        duration_seconds=4.0,
        target_timestep_seconds=5e-2,
        num_worlds=args.num_worlds,
        use_cuda_graph=False,   # forward has Python policy queries — graphs would not capture
    )
    render_config = RenderingConfig(
        vis_type="null",
        target_fps=30, usd_file=None,
        world_offset_x=20.0, world_offset_y=20.0,
    )

    import wandb
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode=args.wandb_mode,
        job_type="finetune",
        config={**vars(args), "balance_pitch_rad": BALANCE_PITCH},
    )

    ckpt_dir: pathlib.Path | None = None
    if args.checkpoint_dir:
        run_tag = run.name if (run is not None and run.name) else f"finetune_{int(time.time())}"
        ckpt_dir = pathlib.Path(args.checkpoint_dir) / run_tag
        print(f"[checkpoint] writing to {ckpt_dir}", flush=True)

    sim = HelhestBalanceFineTune(
        sim_config, render_config,
        _make_default_engine_config(), LoggingConfig(),
        checkpoint_path=pathlib.Path(args.checkpoint),
        policy_hidden=args.hidden,
        v_max=args.v_max,
        lr=args.lr,
        weight_rot=args.weight_rot,
        weight_pos=args.weight_pos,
        weight_smooth=args.weight_smooth,
        weight_reg=args.weight_reg,
        alive_threshold=args.alive_threshold,
        grad_clip=args.grad_clip,
        wandb_run=run,
        video_every=args.video_every,
        checkpoint_dir=ckpt_dir,
        checkpoint_every=args.checkpoint_every,
        args_for_ckpt=vars(args),
    )

    try:
        sim.train(iterations=args.iterations)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
