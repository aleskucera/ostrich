"""Helhest_junior box trajectory optimization using MuJoCo MJX (jax.grad).

MJX counterpart of optimize_ostrich.py: optimises a K-knot wheel-velocity
spline so the helhest_junior matches a recorded real trajectory while
crossing the box. Uses the junior MJCF + yaw-tuned best MuJoCo params from
experiments/1_sim_to_real_box (μ=1.5, tor=10, condim=6, implicitfast),
default dt=5e-3 (inside MuJoCo's accuracy plateau on this scene; small
enough that BPTT memory stays modest — ~0.7 GB at horizon=6 s).

Outputs results/mjx.json with the same per-trial loss-curve schema as
optimize_ostrich.py, so plot_results.py overlays them automatically.

Requires JAX + mujoco-mjx (not installed in ostrich's main env — install with
``pip install jax mujoco-mjx`` in a separate venv).

Usage:
    python experiments/3_gradient_quality_box/optimize_mjx.py
    python experiments/3_gradient_quality_box/optimize_mjx.py \
        --iterations 50 --num-trials 3 --horizon-s 6.0 --dt 5e-3
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "1_sim_to_real_box"))

import numpy as np

# Lazy/guarded JAX imports — let `--help` work without JAX installed and
# fail with a clear message if someone tries to run without it.
try:
    import jax
    import jax.numpy as jnp
    import mujoco
    import mujoco.mjx as mjx
    _HAVE_JAX = True
except ImportError as e:
    _HAVE_JAX = False
    _JAX_IMPORT_ERROR = e

# Reuse the junior+box MJCF template + tuned params from the MuJoCo sweep.
from sweep_mujoco import BASE_PARAMS, JUNIOR_BOX_XML  # noqa: E402

RESULTS_DIR = pathlib.Path(__file__).parent / "results"

# Junior qpos layout (with free base joint + 3 hinge wheel joints):
#   qpos[0:3]   chassis position (x, y, z)
#   qpos[3:7]   chassis orientation quaternion (w, x, y, z)  (MuJoCo's wxyz)
#   qpos[7:10]  wheel angles (left, right, rear)
# Loss only consumes qpos[:2] (chassis XY) — same as optimize_ostrich.py.


# -------------- box-scene MJCF params (locked to MuJoCo sweep best) ---------
MJX_PARAMS = dict(
    mu=1.5,
    tor=10.0,             # torsional friction (requires condim=6)
    kv=1000.0,            # velocity-actuator gain
    solref0=0.005,
    solref1=1.0,
    condim=6,
    integrator="implicitfast",
)


def make_interp_matrix(T, K):
    """Linear interpolation matrix W [T,K] — one spline knot becomes a triangle."""
    W = np.zeros((T, K), dtype=np.float32)
    for t in range(T):
        x = t * (K - 1) / max(T - 1, 1)
        lo = int(x); hi = min(lo + 1, K - 1); a = x - lo
        W[t, lo] += 1.0 - a
        W[t, hi] += a
    return W


def _patch_wheels_for_mjx(xml: str) -> str:
    """Swap wheel cylinders → capsules + disable wheel↔wheel collisions.

    MJX doesn't implement cylinder↔box collisions (the obstacle in this scene
    is a box), so the wheels can't stay as cylinders. Capsules preserve the
    cylinder's line-contact rolling behavior at ground level — the
    hemispherical caps don't poke below the cylindrical body — but they do
    extend ``radius`` (=0.35 m) past each axle endpoint in Y. With wheel
    centers at y=±0.365 and a 0.40 m capsule half-Y-extent, the L/R wheels
    overlap each other by ~7 cm at the spawn pose. We isolate the wheels into
    their own contype group so wheel↔wheel contacts are skipped while
    wheel↔ground and wheel↔box (both default 1/1) still fire.
    """
    return xml.replace(
        '<geom type="cylinder" fromto="0 -0.05 0 0 0.05 0" size="0.35"',
        '<geom type="capsule" fromto="0 -0.05 0 0 0.05 0" size="0.35" contype="1" conaffinity="2"',
    )


def build_mjx_model(dt, gt_box):
    """Compile MJCF (with our junior + box geometry) into an mjx.Model."""
    box = gt_box
    fmt = {**BASE_PARAMS, **MJX_PARAMS, "dt": dt,
           # mirror sweep_mujoco's per-surface friction
           "ground_friction": MJX_PARAMS["mu"], "box_friction": MJX_PARAMS["mu"],
           "front_friction": MJX_PARAMS["mu"], "rear_friction": MJX_PARAMS["mu"],
           "ground_torsional": MJX_PARAMS["tor"], "front_torsional": MJX_PARAMS["tor"],
           "rear_torsional": MJX_PARAMS["tor"],
           "box_x": box["center"][0], "box_y": box["center"][1], "box_z": box["center"][2],
           "box_hx": box["half_extents"][0], "box_hy": box["half_extents"][1],
           "box_hz": box["half_extents"][2]}
    xml = JUNIOR_BOX_XML.format(**fmt)
    xml = _patch_wheels_for_mjx(xml)  # MJX collision-matrix workaround
    mj_model = mujoco.MjModel.from_xml_string(xml)
    return mjx.put_model(mj_model), mj_model


def make_init_data(mx, mj_model):
    """Initial mjx.Data with the chassis at its spawn pose."""
    d = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, d)
    return mjx.put_data(mj_model, d)


def rollout_loss_fn(mx, dx0, W, target_xy):
    """Build a jit-able loss(params) -> scalar by closure over mx/dx0/W/target_xy."""

    def loss_fn(params):
        # params: [K, 3] -> ctrl: [T, 3]   (left, right, rear wheel velocities)
        ctrl_traj = W @ params

        def step_fn(d, ctrl_t):
            d = d.replace(ctrl=ctrl_t)
            d = mjx.step(mx, d)
            return d, d.qpos[:2]  # (x, y)

        _, xy = jax.lax.scan(step_fn, dx0, ctrl_traj)
        return jnp.mean(jnp.sum((xy - target_xy) ** 2, axis=1))

    return loss_fn


# -------------- Adam (numpy, like optimize_ostrich.py) -----------------------
class SplineAdam:
    def __init__(self, K, num_dofs, lr=0.1, lr_min_ratio=0.2, total_steps=50,
                 betas=(0.9, 0.999), eps=1e-8):
        self.lr_init = lr; self.lr_min = lr * lr_min_ratio
        self.total_steps = total_steps
        self.b1, self.b2 = betas; self.eps = eps
        self.m = np.zeros((K, num_dofs), dtype=np.float64)
        self.v = np.zeros((K, num_dofs), dtype=np.float64)
        self.t = 0

    def _lr(self):
        p = min(self.t / max(1, self.total_steps), 1.0)
        return self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (1.0 + np.cos(np.pi * p))

    def step(self, params, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        # bias correction: with b1=0 the formula reduces to m_t = grad, no
        # correction needed (1 - 0**t = 1); same handling drops out for b2.
        mh = self.m / (1 - self.b1**self.t) if self.b1 > 0 else self.m
        vh = self.v / (1 - self.b2**self.t)
        return params - self._lr() * mh / (np.sqrt(vh) + self.eps)


# ------------------------------ trial driver --------------------------------
def run_trial(target_xyz_rel, target_t, K, lr, iterations, seed, horizon, dt, gt_box,
              clip_grad_norm=None, beta1=0.9, beta2=0.999):
    """One full optimisation trial. Returns dict with losses + wall time.

    clip_grad_norm: if not None, scale the gradient so its global L2 norm is at
    most this value before passing to the optimizer. Surgically removes the
    occasional spike (typically at contact-event iterations) that otherwise
    knocks Adam out of discovered minima despite cosine LR decay.
    """
    # Build the MJX model + initial state.
    mx, mj_model = build_mjx_model(dt, gt_box)
    dx0 = make_init_data(mx, mj_model)

    # Initial sim chassis XY (model spawn pose) — shift target into sim frame.
    sim_origin = np.asarray(dx0.qpos[:2])  # MJX default: chassis spawn at (0,0)
    T = int(round(horizon / dt))
    t_grid = np.arange(T) * dt
    target_xy = np.zeros((T, 2), dtype=np.float32)
    for c in range(2):
        target_xy[:, c] = np.interp(t_grid, target_t, target_xyz_rel[:, c])
    target_xy += sim_origin.astype(np.float32)

    # JAX-side compiled value-and-grad of the rollout loss.
    W = make_interp_matrix(T, K)
    loss_fn = rollout_loss_fn(mx, dx0, jnp.asarray(W), jnp.asarray(target_xy))
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    # Initial spline params — small forward bias + per-trial noise.
    rng = np.random.default_rng(seed)
    params_np = (2.0 + 0.5 * rng.standard_normal((K, 3))).astype(np.float32)
    opt = SplineAdam(K=K, num_dofs=3, lr=lr, lr_min_ratio=0.2, total_steps=iterations,
                     betas=(beta1, beta2))

    losses, grad_norms, n_clipped = [], [], 0
    t0_total = time.perf_counter()
    for it in range(iterations):
        t0 = time.perf_counter()
        loss_val, grad = value_and_grad(jnp.asarray(params_np))
        loss_val = float(loss_val); grad_np = np.asarray(grad).astype(np.float64)
        gnorm = float(np.linalg.norm(grad_np))
        if clip_grad_norm is not None and gnorm > clip_grad_norm:
            grad_np = grad_np * (clip_grad_norm / gnorm)
            n_clipped += 1
        losses.append(loss_val); grad_norms.append(gnorm)
        params_np = opt.step(params_np, grad_np).astype(np.float32)
        clipped = " *" if (clip_grad_norm is not None and gnorm > clip_grad_norm) else ""
        print(f"    iter {it:3d}: loss={loss_val:.4f}  |g|={gnorm:.3f}{clipped}  "
              f"({time.perf_counter() - t0:.2f}s)", flush=True)
    elapsed = time.perf_counter() - t0_total
    if clip_grad_norm is not None:
        print(f"    grad clipped on {n_clipped}/{iterations} iters (threshold {clip_grad_norm})")
    return {"seed": int(seed), "losses": losses, "grad_norms": grad_norms,
            "n_clipped": int(n_clipped), "wall_s": elapsed,
            "best_loss": float(min(losses))}


# --------------------------------- main --------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", default=str(pathlib.Path(__file__).resolve().parents[1]
                                         / "1_sim_to_real_box" / "data"
                                         / "run_2026_05_20-18_10_33.json"),
                    help="ground-truth JSON (from 1_sim_to_real_box/data/)")
    ap.add_argument("--K", type=int, default=10, help="spline control points")
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--num-trials", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--horizon-s", type=float, default=6.0,
                    help="trajectory horizon (s). Box crossing happens around t≈3-5 s")
    # dt=5e-3 is inside MuJoCo's accuracy plateau on this scene
    # (experiments/2_dt_stability_box shows the plateau is 1e-4 → 1e-2).
    # BPTT memory ~ 0.7 GB at this dt + horizon=6 s (extrapolated from a
    # measured 1.1 GB / 2 s at typical dt).
    ap.add_argument("--dt", type=float, default=5e-3)
    ap.add_argument("--clip-grad-norm", type=float, default=1.0,
                    help="global-norm gradient clip (None to disable). Default 1.0 "
                         "stabilises the back half of optimisation against "
                         "contact-event gradient spikes that knock Adam out of "
                         "discovered minima.")
    ap.add_argument("--beta1", type=float, default=0.9,
                    help="Adam first-moment decay. Set to 0 to disable momentum "
                         "(reduces Adam to RMSprop). Without momentum, bad gradient "
                         "directions from contact-event spikes don't get averaged "
                         "into the running mean and carried forward ~10 iters.")
    ap.add_argument("--beta2", type=float, default=0.999,
                    help="Adam second-moment decay.")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    if not _HAVE_JAX:
        raise SystemExit(
            f"JAX or mujoco-mjx is not installed.\n  ({_JAX_IMPORT_ERROR})\n"
            "Install with:  pip install jax mujoco-mjx")

    with open(args.gt) as f:
        gt = json.load(f)
    real_t = np.asarray(gt["real"]["t"])
    # XY only — see loss; z is dominated by the climb and uninformative for
    # wheel-velocity gradients.
    real_xy = np.column_stack([gt["real"]["x"], gt["real"]["y"]])
    m = (real_t >= 0) & (real_t <= args.horizon_s)
    real_t = real_t[m]; real_xy = real_xy[m]

    print(f"Loaded GT {gt['run_id']}: {len(real_t)} target points over "
          f"t in [{real_t.min():.2f}, {real_t.max():.2f}] s")
    print(f"K={args.K}  iters={args.iterations}  lr={args.lr}  "
          f"trials={args.num_trials}  dt={args.dt}  horizon={args.horizon_s}s")
    print(f"JAX devices: {jax.devices()}")

    trials = []
    for k in range(args.num_trials):
        seed = args.seed_base + k
        print(f"\n--- trial {k + 1}/{args.num_trials} (seed={seed}) ---")
        trials.append(run_trial(real_xy, real_t, args.K, args.lr, args.iterations,
                                  seed, args.horizon_s, args.dt, gt["box"],
                                  clip_grad_norm=args.clip_grad_norm,
                                  beta1=args.beta1, beta2=args.beta2))

    out = {
        "simulator": "MJX",
        "gradient_method": "jax.grad (BPTT)",
        "gt": gt["run_id"],
        "K": args.K, "lr": args.lr, "iterations": args.iterations,
        "horizon_s": args.horizon_s, "dt": args.dt,
        "clip_grad_norm": args.clip_grad_norm,
        "beta1": args.beta1, "beta2": args.beta2,
        "num_trials": args.num_trials, "seed_base": args.seed_base,
        "params": MJX_PARAMS, "trials": trials,
    }
    save_path = args.save or str(RESULTS_DIR / "mjx.json")
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(out, f)
    best = min(t["best_loss"] for t in trials)
    print(f"\nBest loss across {args.num_trials} trials: {best:.4f}   -> {save_path}")


if __name__ == "__main__":
    main()
