"""Diff body trajectories from two diagnose runs."""
import sys

import numpy as np


def _final_nr_res(npz):
    nr_res = npz["nr_res_norm_sq"]   # (S, K, W)
    nr_iters = npz["nr_iters"]       # (S,)
    n_steps = nr_iters.shape[0]
    out = np.empty(n_steps, dtype=np.float64)
    for s in range(n_steps):
        k = int(nr_iters[s]) - 1
        out[s] = nr_res[s, k, 0]
    return out


def main(path_a: str, path_b: str):
    a = np.load(path_a)
    b = np.load(path_b)

    pose_a = a["body_pose"]
    pose_b = b["body_pose"]
    vel_a = a["body_vel"]
    vel_b = b["body_vel"]

    n_steps = min(pose_a.shape[0], pose_b.shape[0])
    pose_a = pose_a[:n_steps]
    pose_b = pose_b[:n_steps]
    vel_a = vel_a[:n_steps]
    vel_b = vel_b[:n_steps]

    body_count = pose_a.shape[2]
    print(f"Comparing {path_a} vs {path_b}")
    print(f"steps={n_steps}  bodies={body_count}")
    print()

    pos_a = pose_a[:, 0, :, :3]
    pos_b = pose_b[:, 0, :, :3]
    pos_err = np.linalg.norm(pos_a - pos_b, axis=-1)

    print("Per-body max position difference over the run:")
    for bi in range(body_count):
        max_pos = pos_err[:, bi].max()
        max_step = int(pos_err[:, bi].argmax())
        print(f"  body[{bi}]: pos_err max={max_pos:.4f} m at step {max_step}")
    print()
    print(f"chassis end-of-run a={pos_a[-1,0]}  b={pos_b[-1,0]}")
    print()

    # NR-residual analysis: steps where NR failed to converge.
    nr_a = _final_nr_res(a)
    nr_b = _final_nr_res(b)
    print("Final NR residual squared per step:")
    print(f"  a: p50={np.percentile(nr_a,50):.2e}  p95={np.percentile(nr_a,95):.2e}  "
          f"max={nr_a.max():.2e}")
    print(f"  b: p50={np.percentile(nr_b,50):.2e}  p95={np.percentile(nr_b,95):.2e}  "
          f"max={nr_b.max():.2e}")

    # Steps where one converged and the other didn't.
    threshold = 1e-3
    bad_a = (nr_a > threshold)
    bad_b = (nr_b > threshold)
    print(f"\nSteps with NR final ||r||² > {threshold}:")
    print(f"  a: {int(bad_a.sum())}/{n_steps}")
    print(f"  b: {int(bad_b.sum())}/{n_steps}")

    # Print steps where b's NR residual is dramatically worse than a's.
    ratio = nr_b / np.maximum(nr_a, 1e-30)
    worst = np.argsort(-ratio)[:10]
    print("\nTop 10 steps where b's NR residual >> a's:")
    print(f"  {'step':>5} {'nr_a':>14} {'nr_b':>14} {'b/a':>10} "
          f"{'pos_err':>10}  chassis_a -> chassis_b")
    for s in worst:
        print(f"  {s:>5d} {nr_a[s]:>14.2e} {nr_b[s]:>14.2e} {ratio[s]:>10.1e} "
              f"{pos_err[s,0]:>10.4f}  "
              f"{pos_a[s,0]} -> {pos_b[s,0]}")

    # Look at chassis Z (height): is it sinking?
    z_a = pos_a[:, 0, 2]
    z_b = pos_b[:, 0, 2]
    print(f"\nChassis z (height) over run:")
    print(f"  a: start={z_a[0]:.4f}  min={z_a.min():.4f}  max={z_a.max():.4f}  "
          f"end={z_a[-1]:.4f}")
    print(f"  b: start={z_b[0]:.4f}  min={z_b.min():.4f}  max={z_b.max():.4f}  "
          f"end={z_b[-1]:.4f}")

    # Check for NaN / inf
    if not np.isfinite(pose_b).all():
        n_nan = int(np.isnan(pose_b).sum())
        n_inf = int(np.isinf(pose_b).sum())
        print(f"\n*** b body_pose contains {n_nan} NaN, {n_inf} inf ***")

    # Velocity comparison (last 10 steps)
    print()
    print("Wheel ωy in last 10 steps (commanded ~5.0 rad/s):")
    for bi in range(1, body_count):
        wa = vel_a[-10:, 0, bi, 4]
        wb = vel_b[-10:, 0, bi, 4]
        print(f"  body[{bi}]:  a={wa.mean():+.3f}±{wa.std():.3f}  "
              f"b={wb.mean():+.3f}±{wb.std():.3f}")


if __name__ == "__main__":
    path_a = sys.argv[1] if len(sys.argv) > 1 else "data/baselines/helhest_traj_no_ls.npz"
    path_b = sys.argv[2] if len(sys.argv) > 2 else "data/baselines/helhest_traj_ls.npz"
    main(path_a, path_b)
