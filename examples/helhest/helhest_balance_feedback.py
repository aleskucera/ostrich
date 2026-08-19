"""Learned state-feedback balance for Helhest via differentiable simulation.

The open-loop spline formulation (helhest_balance_bundled.py) hits the
exponential-sensitivity wall of open-loop stabilization (~0.86 s max balance).
Feedback fixes the conditioning; this script learns the feedback.

Policies:
  linear : per wheel, u_i = theta_i . phi     (features below)
  mlp    : u = W2 tanh(W1 phi + b1) + b2      (small net, tanh hidden)

Features phi = [1, pitch_err, pitch_rate, v_x]:
  pitch from the chassis up-vector (rotation about +y, relative to
  BALANCE_PITCH), pitch_rate its finite difference, v_x the chassis forward
  velocity (keeps the policy from "balancing" by driving away).

Gradients — two modes:
  direct     : dL/dtheta = sum_t (dpi/dtheta)^T g_u(t), with g_u from the
               standard adjoint. Ignores du/ds inside the recursion; diverges
               once feedback gains matter.
  exact BPTT : during the descending backward sweep, after step i's adjoint
               is computed, inject the policy pathway
                   dL/ds_i += (du_i/ds_i)^T g_u(i)
               into the trajectory adjoint slots (pose slot i for the pitch
               feature, pose slot i-1 for the finite-difference rate, velocity
               slot i for v_x) BEFORE step i-1 consumes them. g_u(i) at that
               moment already contains every later step's policy pathway, so
               the recursion is exact. Parameter gradients accumulate from the
               same g_u(i).

Loss: for true balance-in-place use --orient-loss quadratic --weight-pos 10
(threshold orientation + weak position lets "drive away slightly tilted" win).

Usage:
    python examples/helhest/helhest_balance_feedback.py --policy mlp \
        --exact-bptt --duration 3.0 --iterations 300 --vis headless
    python examples/helhest/helhest_balance_feedback.py --replay \
        --init-policy policy.npz --vis gl
"""
import argparse
import pathlib
import sys

import numpy as np
import warp as wp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ostrich import LoggingConfig, RenderingConfig, SimulationConfig

from examples.helhest.helhest_balance_bundled import (
    BALANCE_PITCH, NUM_WHEEL_DOFS, WHEEL_DOF_OFFSET,
    HelhestBalanceBundledOptimizer, _make_default_engine_config)


N_FEATURES = 4  # [1, pitch_err, pitch_rate, v_x]
U_LIMIT = 25.0  # wheel target velocity clip [rad/s]


# ---------------------------------------------------------------------------
# Policies (numpy; tiny — host-side forward and Jacobians)
# ---------------------------------------------------------------------------
class LinearPolicy:
    def __init__(self, rng=None, scale=0.0):
        rng = rng or np.random.default_rng(0)
        self.W = scale * rng.standard_normal((NUM_WHEEL_DOFS, N_FEATURES))

    def forward(self, phi):
        u = self.W @ phi
        return u, {"phi": phi}

    def d_u_d_phi(self, cache):
        return self.W.copy()                      # [3, F]

    def d_u_d_theta(self, cache, g_u):
        return {"W": np.outer(g_u, cache["phi"])}

    def params(self):
        return {"W": self.W}

    def apply_update(self, deltas):
        self.W += deltas["W"]

    def save(self, path):
        np.savez(path, kind="linear", W=self.W)

    @staticmethod
    def load(path):
        d = np.load(path, allow_pickle=True)
        p = LinearPolicy()
        p.W = d["W"]
        return p


class MLPPolicy:
    def __init__(self, hidden=8, rng=None):
        rng = rng or np.random.default_rng(0)
        self.W1 = 0.3 * rng.standard_normal((hidden, N_FEATURES))
        self.b1 = np.zeros(hidden)
        self.W2 = 0.3 * rng.standard_normal((NUM_WHEEL_DOFS, hidden))
        self.b2 = np.zeros(NUM_WHEEL_DOFS)

    def forward(self, phi):
        z = self.W1 @ phi + self.b1
        h = np.tanh(z)
        u = self.W2 @ h + self.b2
        return u, {"phi": phi, "h": h}

    def d_u_d_phi(self, cache):
        dh = 1.0 - cache["h"] ** 2                # [H]
        return self.W2 @ (dh[:, None] * self.W1)  # [3, F]

    def d_u_d_theta(self, cache, g_u):
        dh = 1.0 - cache["h"] ** 2
        gh = (self.W2.T @ g_u) * dh               # [H]
        return {"W2": np.outer(g_u, cache["h"]), "b2": g_u.copy(),
                "W1": np.outer(gh, cache["phi"]), "b1": gh}

    def params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def apply_update(self, deltas):
        for k, d in deltas.items():
            getattr(self, k).__iadd__(d)

    def save(self, path):
        np.savez(path, kind="mlp", W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @staticmethod
    def load(path):
        d = np.load(path, allow_pickle=True)
        p = MLPPolicy(hidden=d["W1"].shape[0])
        p.W1, p.b1, p.W2, p.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
        return p


class ResidualPolicy(MLPPolicy):
    """u = base_W @ phi + MLP(phi); MLP output-layer zero-initialized, so the
    policy starts exactly at the (stabilizing) base PD and learns corrections."""

    def __init__(self, base_W, hidden=8, rng=None):
        super().__init__(hidden=hidden, rng=rng)
        self.W2 = np.zeros_like(self.W2)
        self.base_W = np.asarray(base_W, dtype=np.float64)

    def forward(self, phi):
        u_mlp, cache = super().forward(phi)
        return self.base_W @ phi + u_mlp, cache

    def d_u_d_phi(self, cache):
        return self.base_W + super().d_u_d_phi(cache)

    def save(self, path):
        np.savez(path, kind="residual", W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2, base_W=self.base_W)

    @staticmethod
    def load(path):
        d = np.load(path, allow_pickle=True)
        p = ResidualPolicy(d["base_W"], hidden=d["W1"].shape[0])
        p.W1, p.b1, p.W2, p.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
        return p


def load_policy(path):
    d = np.load(path, allow_pickle=True)
    kind = str(d["kind"])
    if kind == "mlp":
        return MLPPolicy.load(path)
    if kind == "residual":
        return ResidualPolicy.load(path)
    return LinearPolicy.load(path)


# ---------------------------------------------------------------------------
# Pitch helpers (quaternion x,y,z,w -> tilt about +y and its Jacobian)
# ---------------------------------------------------------------------------
def pitch_of_quat(q):
    x, y, z, w = q
    a = 2.0 * (x * z + w * y)        # up_x
    b = 1.0 - 2.0 * (x * x + y * y)  # up_z
    return float(np.arctan2(a, b))


def d_pitch_d_quat(q):
    x, y, z, w = q
    a = 2.0 * (x * z + w * y)
    b = 1.0 - 2.0 * (x * x + y * y)
    da = np.array([2 * z, 2 * w, 2 * x, 2 * y])
    db = np.array([-4 * x, -4 * y, 0.0, 0.0])
    return (b * da - a * db) / max(a * a + b * b, 1e-12)


class HelhestBalanceFeedback(HelhestBalanceBundledOptimizer):
    """Closed-loop rollout; exact or direct-term policy BPTT."""

    def _forward_backward(self):
        pol = self.policy
        dt = self.clock.dt
        T = self.clock.total_sim_steps
        self.trajectory.zero_grad()

        self._caches = []
        self._quats = np.zeros((T, 4))
        self._sat = np.zeros((T, NUM_WHEEL_DOFS))
        prev_pitch = None
        num_dofs = self.trajectory.joint_target_vel.shape[-1]

        kick = getattr(self, "current_kick", 0.0)
        if kick != 0.0:
            qd = self.states[0].body_qd.numpy()
            qd[0, self._omega_idx()] = kick
            wp.copy(self.states[0].body_qd,
                    wp.array(qd, dtype=wp.spatial_vector,
                             device=self.model.device))

        for i in range(T):
            q = self.states[i].body_q.numpy()[0][3:7]
            self._quats[i] = q
            pitch = pitch_of_quat(q)
            rate = 0.0 if prev_pitch is None else (pitch - prev_pitch) / dt
            prev_pitch = pitch
            # ostrich body_vel convention: spatial_top = linear -> v_x = [0]
            v_x = (float(self.trajectory.body_vel[i].numpy()[0, 0, 0])
                   if i > 0 else 0.0)
            phi = np.array([1.0, pitch - BALANCE_PITCH, rate, v_x])
            u_raw, cache = pol.forward(phi)
            u = np.clip(u_raw, -U_LIMIT, U_LIMIT)
            self._sat[i] = (np.abs(u_raw) >= U_LIMIT).astype(np.float64)
            self._caches.append(cache)

            ctrl = np.zeros((1, num_dofs), dtype=np.float32)
            ctrl[0, WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = u
            wp.copy(self.trajectory.joint_target_vel[i],
                    wp.array(ctrl, dtype=wp.float32, device=self.model.device))
            wp.copy(self.controls[i].joint_target_vel,
                    self.trajectory.joint_target_vel[i])

            self.collision_pipeline.collide(self.states[i], self.contacts)
            self.solver.step(
                state_in=self.states[i], state_out=self.states[i + 1],
                control=self.controls[i], contacts=self.contacts,
                dt=self.clock.dt)
            self.trajectory.save_step(i, self.solver.data,
                                      self.solver.ostrich_contacts)

        self.tape.reset()
        with self.tape:
            self.compute_loss()
        self.tape.backward(self.loss)

        grads = {k: np.zeros_like(v) for k, v in pol.params().items()}
        for i in range(T - 1, -1, -1):
            self.trajectory.load_step(i, self.solver.data,
                                      self.solver.ostrich_contacts)
            self.solver.data.zero_gradients()
            self.solver.step_backward()
            self.trajectory.save_gradients(i, self.solver.data)
            self.trajectory.save_pose_gradients(i, self.solver.data)

            # g_u(i): complete downstream gradient of this step's control.
            g_u = self.solver.data.joint_target_vel.grad.numpy()[
                0, WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS].copy()
            g_u *= (1.0 - self._sat[i])  # clip gate
            for k, d in pol.d_u_d_theta(self._caches[i], g_u).items():
                grads[k] += d

            if self.exact_bptt:
                J = pol.d_u_d_phi(self._caches[i])       # [3, F]
                g_phi = (J.T @ g_u) * getattr(self, "inject_scale", 1.0)
                dt_ = dt
                # pitch_err at step i: quat slot i (+ rate coupling)
                dpq_i = d_pitch_d_quat(self._quats[i])
                coeff_q_i = g_phi[1] + g_phi[2] / dt_
                self._add_pose_grad(i, dpq_i * coeff_q_i)
                if i > 0:
                    dpq_im1 = d_pitch_d_quat(self._quats[i - 1])
                    self._add_pose_grad(i - 1, -dpq_im1 * g_phi[2] / dt_)
                    # v_x read from body_vel slot i (valid for i>0)
                    self._add_vel_grad(i, 0, g_phi[3])
        self.policy_grads = grads

    def _omega_idx(self):
        # Which body_qd component is angular-y for newton State? Calibrate
        # once: kick each candidate, roll 3 passive steps, see which moves
        # pitch more.
        if hasattr(self, "_omega_idx_cached"):
            return self._omega_idx_cached
        base_qd = self.states[0].body_qd.numpy().copy()
        effects = {}
        for idx in (1, 4):
            qd = base_qd.copy(); qd[0, idx] = 1.0
            wp.copy(self.states[0].body_qd,
                    wp.array(qd, dtype=wp.spatial_vector, device=self.model.device))
            p0 = pitch_of_quat(self.states[0].body_q.numpy()[0][3:7])
            for i in range(3):
                self.collision_pipeline.collide(self.states[i], self.contacts)
                self.solver.step(state_in=self.states[i],
                                 state_out=self.states[i + 1],
                                 control=self.controls[i],
                                 contacts=self.contacts, dt=self.clock.dt)
            p3 = pitch_of_quat(self.states[3].body_q.numpy()[0][3:7])
            effects[idx] = abs(p3 - p0)
        wp.copy(self.states[0].body_qd,
                wp.array(base_qd, dtype=wp.spatial_vector, device=self.model.device))
        self._omega_idx_cached = max(effects, key=effects.get)
        return self._omega_idx_cached

    def episode_metrics(self):
        bp = self.trajectory.body_pose.numpy()
        T = self.clock.total_sim_steps
        xy = bp[:T + 1, 0, 0, 0:2]
        drift = float(np.max(np.linalg.norm(xy - xy[0], axis=1)))
        pitches = np.array([pitch_of_quat(bp[i, 0, 0, 3:7]) for i in range(T + 1)])
        pitch_rms = float(np.sqrt(np.mean((pitches - BALANCE_PITCH) ** 2)))
        return {"pos_drift_m": drift, "pitch_rms_rad": pitch_rms}

    def _add_pose_grad(self, slot, dquat):
        # Slot-only transfer: converting the full [T+1] buffer per injection
        # dominated the iteration time.
        g_slot = self.trajectory.body_pose.grad[slot]
        arr = g_slot.numpy()
        arr[0, 0, 3:7] += dquat
        wp.copy(g_slot, wp.array(arr, dtype=wp.transform,
                                 device=self.model.device))

    def _add_vel_grad(self, slot, comp, val):
        g_slot = self.trajectory.body_vel.grad[slot]
        arr = g_slot.numpy()
        arr[0, 0, comp] += val
        wp.copy(g_slot, wp.array(arr, dtype=wp.spatial_vector,
                                 device=self.model.device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", choices=["linear", "mlp", "residual"],
                    default="mlp")
    ap.add_argument("--base-kp", type=float, default=40.0,
                    help="residual policy: fixed base PD pitch gain")
    ap.add_argument("--base-kd", type=float, default=0.3,
                    help="residual policy: fixed base PD rate gain")
    ap.add_argument("--inject-scale", type=float, default=1.0,
                    help="scale on the exact-BPTT policy injections (<1 damps "
                         "the closed-loop adjoint explosion at the cost of bias)")
    ap.add_argument("--kick-std", type=float, default=0.0,
                    help="training: std of random initial pitch-rate kick [rad/s]")
    ap.add_argument("--eval-only", action="store_true",
                    help="single rollout, print metrics (optionally per --eval-kicks)")
    ap.add_argument("--eval-kicks", type=str, default="0",
                    help="comma list of initial pitch-rate kicks for --eval-only")
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--exact-bptt", action="store_true")
    ap.add_argument("--orient-loss", choices=["quadratic", "threshold"],
                    default="quadratic")
    ap.add_argument("--weight-pos", type=float, default=10.0)
    ap.add_argument("--weight-rot", type=float, default=200.0)
    ap.add_argument("--vis", choices=["gl", "headless"], default="headless")
    ap.add_argument("--save-policy", type=str, default=None)
    ap.add_argument("--init-policy", type=str, default=None)
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--replay-loops", type=int, default=3)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    sim_config = SimulationConfig(
        duration_seconds=args.duration, target_timestep_seconds=5e-2,
        num_worlds=1, use_cuda_graph=False)
    render_config = RenderingConfig(
        vis_type=("null" if args.vis == "headless" else "gl"),
        target_fps=30, usd_file=None, world_offset_x=20.0, world_offset_y=20.0)

    sim = HelhestBalanceFeedback(
        sim_config, render_config, _make_default_engine_config(),
        LoggingConfig(), num_control_points=10, sigma=0.0,
        sigma_min_ratio=0.1, antithetic=False, lr=args.lr,
        total_steps=args.iterations, beta1=0.9, beta2=0.999,
        orient_loss=args.orient_loss,
        weight_pos=args.weight_pos, weight_rot=args.weight_rot)

    import newton
    newton.eval_fk(sim.model, sim.model.joint_q, sim.model.joint_qd,
                   sim.states[0])
    newton.eval_fk(sim.model, sim.model.joint_q, sim.model.joint_qd,
                   sim.target_states[0])
    sim._build_target_episode()

    if args.init_policy:
        sim.policy = load_policy(args.init_policy)
    elif args.policy == "mlp":
        sim.policy = MLPPolicy(hidden=args.hidden, rng=rng)
    elif args.policy == "residual":
        base_W = np.zeros((NUM_WHEEL_DOFS, N_FEATURES))
        base_W[:, 1] = args.base_kp
        base_W[:, 2] = args.base_kd
        sim.policy = ResidualPolicy(base_W, hidden=args.hidden, rng=rng)
    else:
        sim.policy = LinearPolicy(rng=rng, scale=0.0)
    sim.exact_bptt = args.exact_bptt
    sim.inject_scale = args.inject_scale

    if args.eval_only:
        for kick in [float(k) for k in args.eval_kicks.split(",")]:
            sim.current_kick = kick
            sim.diff_step()
            wp.synchronize()
            loss = float(sim.loss.numpy()[0])
            diag = sim._compute_diagnostics()
            met = sim.episode_metrics()
            print(f"EVAL kick={kick:+.2f}: loss={loss:10.2f} "
                  f"alive={diag['alive_frac']*100:5.1f}% "
                  f"drift={met['pos_drift_m']:6.3f}m "
                  f"pitch_rms={met['pitch_rms_rad']:6.4f}rad", flush=True)
            sim.tape.zero(); sim.loss.zero_()
        return

    if args.replay:
        sim.diff_step()
        wp.synchronize()
        loss = float(sim.loss.numpy()[0])
        diag = sim._compute_diagnostics()
        print(f"replay: loss={loss:.2f} alive={diag['alive_frac']*100:.1f}%")
        sim.render_episode(iteration=0, loop=True,
                           loops_count=args.replay_loops, playback_speed=1.0)
        return

    params = sim.policy.params()
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v2 = {k: np.zeros_like(v) for k, v in params.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    best = (float("inf"), None)
    best_alive = 0.0

    for it in range(args.iterations):
        sim.current_kick = (float(rng.normal(0.0, args.kick_std))
                            if args.kick_std > 0 else 0.0)
        sim.diff_step()
        wp.synchronize()
        loss = float(sim.loss.numpy()[0])
        diag = sim._compute_diagnostics()
        g = sim.policy_grads
        gn = float(np.sqrt(sum(np.sum(x * x) for x in g.values())))
        clip = min(1.0, 10.0 / max(gn, 1e-12))
        deltas = {}
        for k in g:
            gk = g[k] * clip
            m[k] = b1 * m[k] + (1 - b1) * gk
            v2[k] = b2 * v2[k] + (1 - b2) * gk * gk
            mh = m[k] / (1 - b1 ** (it + 1))
            vh = v2[k] / (1 - b2 ** (it + 1))
            deltas[k] = -args.lr * mh / (np.sqrt(vh) + eps)
        sim.policy.apply_update(deltas)
        alive_now = diag["alive_frac"]
        if loss < best[0] and alive_now >= best_alive - 0.02:
            best = (loss, {k: v.copy() for k, v in sim.policy.params().items()})
            best_alive = max(best_alive, alive_now)
        print(f"Iter {it:3d}: loss={loss:10.2f}  alive={diag['alive_frac']*100:5.1f}%  "
              f"|g|={gn:9.3f}", flush=True)
        sim.tape.zero(); sim.loss.zero_()

    print("best loss:", best[0])
    if args.save_policy and best[1] is not None:
        for k, val in best[1].items():
            setattr(sim.policy, k, val) if hasattr(sim.policy, k) else None
        sim.policy.save(args.save_policy)
        print(f"saved policy -> {args.save_policy}")


if __name__ == "__main__":
    main()
