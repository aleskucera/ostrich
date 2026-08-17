"""Fit the euro-pallet obstacle pose from the campaign-2 lidar clouds.

The obstacle is a EUR pallet (1200 x 800 x 144 mm). Runs start from different
positions/orientations, so each run needs its own pallet pose (center + yaw)
in the aligned GT frame. The fit must be robust to the pallet's holes and to
partial views, so it works on the OUTER FOOTPRINT with known dimensions:

  1. Aggregate clouds from the approach phase (robot more than 1 m before the
     climb), transformed map -> aligned frame with the exact same alignment
     as prepare_gt_mcap.
  2. Slice the pallet-height band (0.05 < z < 0.20 m) near the coarse
     z-bump-based location estimate.
  3. Keep the largest planar cluster (grid connected components).
  4. Yaw from the SHARPEST boundary edge (RANSAC line on the density-onset
     boundary in all four cardinal directions; edge quality varies per run,
     so pick the crispest). Translation from a global density-weighted
     coverage search of the known 1.2 x 0.8 m footprint (integral image over
     a sqrt(count)-weighted 2 cm grid). Sharp edges pin the placement; fuzzy
     smeared sides and attached clutter carry little weight; holes are
     interior and do not matter.

Updates each GT JSON's box.center (x, y) and adds box.yaw [rad]; writes a
top-down validation figure per run.

Run with the helhest_stack venv (rosbags + matplotlib):
    ~/projects/helhest_stack/.venv/bin/python \
        experiments/1_sim_to_real_box/fit_pallet.py --bags-dir \
        ~/projects/helhest_stack/bags --runs ostrich0 ostrich1 ostrich2 ostrich3
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from prepare_gt_mcap import q_mul, q_rot, yaw_of  # noqa: E402

DATA_DIR = pathlib.Path(__file__).parent / "data"
PALLET_DIMS = (1.2, 0.8)     # EUR pallet footprint [m]
PALLET_HEIGHT = 0.144
Z_BAND = (0.05, 0.20)        # aligned-frame slice that isolates the pallet body


def read_poses_and_clouds(bag_dir):
    from rosbags.highlevel import AnyReader
    mo, ob = {}, {}
    clouds = []
    mount_t = np.zeros(3)
    mount_q = np.array([0.0, 0.0, 0.0, 1.0])
    with AnyReader([bag_dir]) as reader:
        for conn, ts, raw in reader.messages(
                connections=[c for c in reader.connections
                             if c.topic in ("/tf", "/tf_static")]):
            m = reader.deserialize(raw, conn.msgtype)
            for tr in m.transforms:
                t, r = tr.transform.translation, tr.transform.rotation
                rec = (np.array([t.x, t.y, t.z]), np.array([r.x, r.y, r.z, r.w]))
                if conn.topic == "/tf_static":
                    if (tr.header.frame_id == "odin1_base_link"
                            and tr.child_frame_id == "base_link"):
                        mount_t, mount_q = rec
                elif tr.child_frame_id == "odom_odin":
                    mo[ts] = rec
                elif tr.child_frame_id == "odin1_base_link":
                    ob[ts] = rec
        for conn, ts, raw in reader.messages(
                connections=[c for c in reader.connections
                             if c.topic == "/odin1/cloud_raw"]):
            m = reader.deserialize(raw, conn.msgtype)
            n = m.height * m.width
            buf = np.frombuffer(m.data, dtype=np.uint8).reshape(n, m.point_step)
            xyz = np.stack([buf[:, o:o + 4].copy().view(np.float32)[:, 0]
                            for o in (0, 4, 8)], axis=1)
            xyz = xyz[np.isfinite(xyz).all(axis=1)]
            clouds.append((ts, xyz))
    return mo, ob, mount_t, mount_q, clouds


def compose(mo, mo_t, ob_rec, ts):
    i = min(np.searchsorted(mo_t, ts), len(mo_t) - 1)
    p1, q1 = mo[mo_t[i]]
    p2, q2 = ob_rec
    return p1 + q_rot(q1, p2), q_mul(q1, q2)


def largest_cluster(pts, cell=0.15, anchor=None):
    """Grid connected-components. With ``anchor`` (x, y) set, returns the
    component nearest to it (the z-bump puts the robot ON the pallet, so the
    pallet is the cluster at that spot — the globally largest blob can be a
    vegetation bank at the field edge). Without anchor: largest component."""
    if len(pts) == 0:
        return pts
    ij = np.floor(pts[:, :2] / cell).astype(np.int64)
    keys = {}
    for k, (i, j) in enumerate(map(tuple, ij)):
        keys.setdefault((i, j), []).append(k)
    # density gate: sparse bridge cells (vegetation fringes) must not connect
    # the pallet to neighboring structures
    keys = {c: v for c, v in keys.items() if len(v) >= 8}
    seen, best, comps = set(), [], []
    for start in list(keys):
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            c = stack.pop()
            comp.append(c)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = (c[0] + di, c[1] + dj)
                    if nb in keys and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        if sum(len(keys[c]) for c in comp) > sum(len(keys[c]) for c in best):
            best = comp
        comps.append(comp)
    if anchor is not None:
        ax, ay = anchor
        scored = []
        for comp in comps:
            n = sum(len(keys[c]) for c in comp)
            if n < 300:
                continue
            d = min(np.hypot((c[0] + 0.5) * cell - ax, (c[1] + 0.5) * cell - ay)
                    for c in comp)
            scored.append((d, comp))
        if scored:
            best = min(scored, key=lambda t: t[0])[1]
    idx = [k for c in best for k in keys[c]]
    return pts[idx]


def _trace_min_x_edge(P):
    """Boundary points at the min-x side of P, via per-y-bin density onset.

    Sparse fringes (vegetation, smear) sit ahead of the dense pallet top -
    walk from min-x to the first pair of adjacent 2 cm cells that both carry
    real density, and call that the edge.
    """
    y_bins = np.arange(P[:, 1].min(), P[:, 1].max() + 0.04, 0.04)
    idx = np.digitize(P[:, 1], y_bins)
    edge = []
    for b in np.unique(idx):
        seg = P[idx == b]
        if len(seg) < 8:
            continue
        xb = np.arange(seg[:, 0].min(), seg[:, 0].max() + 0.02, 0.02)
        cnt, _ = np.histogram(seg[:, 0], bins=xb)
        if len(cnt) < 2:
            continue
        thr = max(3.0, 0.2 * cnt.max())
        onset = None
        for k in range(len(cnt) - 1):
            if cnt[k] >= thr and cnt[k + 1] >= thr:
                onset = xb[k]
                break
        if onset is None:
            continue
        near = seg[(seg[:, 0] >= onset) & (seg[:, 0] <= onset + 0.06)]
        if len(near) == 0:
            continue
        edge.append([np.median(near[:, 0]), np.median(near[:, 1])])
    return np.array(edge)


def _fit_line(edge):
    """Deterministic RANSAC line x = a*y + b: max-inlier over point pairs,
    LS refit on inliers. Locks onto the straight majority segment even when
    one end of the boundary is a long contaminated arc (which defeats plain
    sigma-trimming). Returns (a, b, rms, keep).
    """
    n = len(edge)
    best_keep = None
    step = max(1, n // 25)  # ~25x25 pair candidates
    for i in range(0, n, step):
        for j in range(i + 1, n, step):
            dy = edge[j, 1] - edge[i, 1]
            if abs(dy) < 0.10:
                continue
            a = (edge[j, 0] - edge[i, 0]) / dy
            if abs(a) > 1.0:  # > 45 deg off this cardinal direction
                continue
            b = edge[i, 0] - a * edge[i, 1]
            keep = np.abs(edge[:, 0] - (a * edge[:, 1] + b)) < 0.015
            if best_keep is None or keep.sum() > best_keep.sum():
                best_keep = keep
    if best_keep is None or best_keep.sum() < 5:
        best_keep = np.ones(n, dtype=bool)
    keep = best_keep
    for _ in range(2):  # LS refit on inliers, then refresh the inlier set
        a, b = np.polyfit(edge[keep, 1], edge[keep, 0], 1)
        r = edge[:, 0] - (a * edge[:, 1] + b)
        keep = np.abs(r) < 0.02
        if keep.sum() < 5:
            keep = best_keep
            break
    a, b = np.polyfit(edge[keep, 1], edge[keep, 0], 1)
    r = edge[:, 0] - (a * edge[:, 1] + b)
    return a, b, float(np.std(r[keep])), keep


def fit_sharpest_edge(pts_xy):
    """Fit the known-dims rectangle: yaw from the sharpest edge, translation
    from a global density-weighted coverage search.

    Edge quality varies per run: faces toward the sensor are crisp
    (vertical-wall returns), occluded/cluttered sides smear - and a smeared
    side's density-onset contour is straight but at a biased position, so
    edge lines can only be trusted for ORIENTATION, not position. Hence:
    1. Trace the boundary in all four cardinal directions, robust-line-fit
       each, and take YAW from the sharpest gate-passing edge (out-ratio gate:
       a real sharp edge has ~nothing beyond it).
    2. In the yaw-aligned frame, rasterize the cluster to a 2 cm grid with
       sqrt(count) cell weights (dense pallet top dominates sparse fuzz) and
       slide the 1.2 x 0.8 rectangle (both axis assignments) via integral
       image; the argmax pins the translation. Sharp edges bound the dense
       mass, so they pin the placement; fuzzy edges carry little weight.
    Returns (cx, cy, yaw, dims, edge_rms, edge_pts_world).
    """
    cands = {}
    for k in range(4):
        rot = k * np.pi / 2
        c, sn = np.cos(rot), np.sin(rot)
        Rk = np.array([[c, -sn], [sn, c]])  # p_local = Rk^T p_world
        P = pts_xy @ Rk
        edge = _trace_min_x_edge(P)
        if len(edge) < 10:
            continue
        a, b, rms, keep = _fit_line(edge)
        d = P[:, 0] - (a * P[:, 1] + b)
        ylim = ((P[:, 1] > edge[keep][:, 1].min())
                & (P[:, 1] < edge[keep][:, 1].max()))
        n_out = int(((d < -0.04) & (d > -0.25) & ylim).sum())
        n_in = int(((d >= 0.02) & (d < 0.25) & ylim).sum())
        ratio = n_out / max(n_in, 1)
        cands[k] = (rms, a, b, edge[keep], Rk, ratio)

    def _score(k):
        return cands[k][0] + (0.0 if cands[k][5] < 0.10 else 1.0)
    sides = {0: "min-x", 1: "min-y", 2: "max-x", 3: "max-y"}
    for k in sorted(cands):
        print(f"    edge {sides[k]}: rms={cands[k][0]:.3f} "
              f"out_ratio={cands[k][5]:.3f}")
    k0 = min(cands, key=_score)
    rms, a, b, edge, Rk, _ = cands[k0]

    # yaw of the rectangle frame (mod 90 deg; assignment resolved below)
    e_w = Rk @ (np.array([a, 1.0]) / np.hypot(a, 1.0))
    yaw = float(np.arctan2(e_w[1], e_w[0]))

    # rasterize in the yaw-aligned frame: 2 cm cells, sqrt(count) weights
    cell = 0.02
    c, sn = np.cos(yaw), np.sin(yaw)
    Ry = np.array([[c, sn], [-sn, c]])  # world -> rect frame
    q = pts_xy @ Ry.T
    q0 = q.min(axis=0) - cell
    ij = np.floor((q - q0) / cell).astype(np.int64)
    ni, nj = ij.max(axis=0) + 2
    grid = np.zeros((ni, nj))
    np.add.at(grid, (ij[:, 0], ij[:, 1]), 1.0)
    w = np.where(grid >= 3, np.sqrt(grid), 0.0)
    # integral image -> box sums for every translation
    I = np.zeros((ni + 1, nj + 1))
    I[1:, 1:] = np.cumsum(np.cumsum(w, axis=0), axis=1)

    def _best_box(du, dv):
        nu, nv = int(round(du / cell)), int(round(dv / cell))
        if nu >= ni or nv >= nj:
            return -1.0, (0, 0)
        S = (I[nu:, nv:] - I[:-nu or None, nv:] - I[nu:, :-nv or None]
             + I[:ni - nu + 1, :nj - nv + 1])
        u, v = np.unravel_index(np.argmax(S), S.shape)
        return float(S[u, v]), (u, v)

    best = None
    for along, depth in (PALLET_DIMS, PALLET_DIMS[::-1]):
        sc, (u, v) = _best_box(along, depth)
        if best is None or sc > best[0]:
            best = (sc, u, v, along, depth)
    _, u, v, along, depth = best
    center_q = q0 + cell * np.array([u + along / (2 * cell),
                                     v + depth / (2 * cell)])
    center = Ry.T @ center_q
    edge_w = edge @ Rk.T
    return (float(center[0]), float(center[1]), yaw, (along, depth),
            rms, edge_w)


def fit_rectangle(pts_xy):
    """Known-dims rectangle fit; returns (cx, cy, yaw, dims, score)."""
    best = None
    for deg in np.arange(0.0, 90.0, 0.5):
        a = np.deg2rad(deg)
        c, s = np.cos(a), np.sin(a)
        R = np.array([[c, s], [-s, c]])
        q = pts_xy @ R.T
        lo = np.percentile(q, 2, axis=0)
        hi = np.percentile(q, 98, axis=0)
        ext = hi - lo
        for dims in (PALLET_DIMS, PALLET_DIMS[::-1]):
            err = abs(ext[0] - dims[0]) + abs(ext[1] - dims[1])
            if best is None or err < best[0]:
                mid = (lo + hi) / 2.0
                center = R.T @ mid
                best = (err, center[0], center[1], a, dims)
    err, cx, cy, yaw, dims = best
    return cx, cy, yaw, dims, err


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bags-dir", type=pathlib.Path, required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--no-update", action="store_true",
                    help="fit and plot only; do not modify the GT JSONs")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for run in args.runs:
        gt = json.load(open(DATA_DIR / f"{run}.json"))
        # z-bump estimate recomputed from the climb interval so the script
        # stays idempotent (the JSON box center may hold a previous fit)
        _rx = np.array(gt["real"]["x"])
        _ry = np.array(gt["real"]["y"])
        _rz = np.array(gt["real"]["z"])
        _on = _rz > 0.06
        box_est_x = float(np.mean(_rx[_on])) if _on.any() else gt["box"]["center"][0]
        box_est_y = float(np.mean(_ry[_on])) if _on.any() else 0.0
        mo, ob, mount_t, mount_q, clouds = read_poses_and_clouds(
            args.bags_dir / run)
        mo_t = np.array(sorted(mo))
        ob_t = np.array(sorted(ob))

        # Alignment identical to prepare_gt_mcap (base_link at first kept pose).
        # Reconstruct t0 from the GT: first real sample corresponds to T[0]=0.
        first_ts = ob_t[np.searchsorted(ob_t, mo_t[0])]  # first composable
        # find the pose whose timestamp matches gt start: recompute exactly
        # as the converter: t0 = max(first pose ts, first setpoint ts). The
        # converter kept poses with T >= 0; its first kept pose defines the
        # alignment. We recover it by matching the stored duration grid:
        # simplest robust approach: recompute base poses for ALL ob stamps and
        # find the one whose relative trajectory reproduces gt real[0..2].
        poses = {}
        for ts in ob_t:
            p, q = compose(mo, mo_t, ob[ts], ts)
            poses[ts] = (p + q_rot(q, mount_t), q_mul(q, mount_q))
        # the converter's first kept pose: its absolute time = bag pose time
        # closest to (last pose time - gt duration ... ) — instead use yaw/pos
        # invariance: alignment origin P0/yaw0 is the pose at the FIRST ts
        # where ts/1e9 >= (first setpoint time). Setpoints not needed: the GT
        # real t[0] == 0 corresponds to that pose; and gt real x/y/z[0] == 0.
        # We find it by testing candidates until the transformed second sample
        # matches gt real (x,y)[1].
        gtx = np.array(gt["real"]["x"]); gty = np.array(gt["real"]["y"])
        cand = None
        for k in range(len(ob_t)):
            P0, Q0 = poses[ob_t[k]]
            yaw0 = yaw_of(Q0)
            c, s = np.cos(-yaw0), np.sin(-yaw0)
            R = np.array([[c, -s], [s, c]])
            ok = True
            for j, off in ((1, 1), (5, 5)):
                if k + off >= len(ob_t) or j >= len(gtx):
                    ok = False
                    break
                pj = poses[ob_t[k + off]][0]
                v = R @ (pj[:2] - P0[:2])
                if abs(v[0] - gtx[j]) > 0.02 or abs(v[1] - gty[j]) > 0.02:
                    ok = False
                    break
            if ok:
                cand = k
                break
        if cand is None:
            print(f"{run}: could not recover alignment, skipping")
            continue
        P0, Q0 = poses[ob_t[cand]]
        yaw0 = yaw_of(Q0)
        c, s = np.cos(-yaw0), np.sin(-yaw0)
        R2 = np.array([[c, -s], [s, c]])

        # Ground-plane estimate: aligned z is relative to base_link, which
        # rides ~0.33 m above ground; slice bands must be ground-relative.
        # Take the modal 5 cm z-bin of corridor points before the box.
        gz = []
        for ts, xyz in clouds[:80]:
            i = min(np.searchsorted(ob_t, ts), len(ob_t) - 1)
            po, qo = compose(mo, mo_t, ob[ob_t[i]], ob_t[i])
            xyz2 = xyz[np.linalg.norm(xyz[:, :2], axis=1) > 0.8]
            x, y, zc, w = qo
            u = np.array([x, y, zc])
            cr1 = np.cross(np.broadcast_to(u, xyz2.shape), xyz2) + w * xyz2
            pm = xyz2 + 2.0 * np.cross(np.broadcast_to(u, xyz2.shape), cr1) + po
            al = (R2 @ (pm[:, :2] - P0[:2]).T).T
            mcorr = ((np.abs(al[:, 1]) < 0.5) & (al[:, 0] > 0.3)
                     & (al[:, 0] < box_est_x - 0.8))
            gz.append(pm[mcorr, 2] - P0[2])
        gz = np.concatenate(gz)
        bins = np.arange(gz.min(), gz.max() + 0.05, 0.05)
        hist, edges = np.histogram(gz, bins=bins)
        z_ground = float(edges[np.argmax(hist)] + 0.025)
        z_lo, z_hi = z_ground + 0.09, z_ground + 0.20

        # Approach-phase clouds: robot aligned-x more than 1 m before the box.
        agg = []
        for ts, xyz in clouds:
            i = min(np.searchsorted(ob_t, ts), len(ob_t) - 1)
            p_r, q_r = poses[ob_t[i]]
            rx = (R2 @ (p_r[:2] - P0[:2]))[0]
            if rx > box_est_x - 0.5:
                continue
            # drop the robot's own near-field returns (chassis/wheels sit in
            # the pallet-height band and follow the sensor everywhere)
            xyz = xyz[np.linalg.norm(xyz[:, :2], axis=1) > 0.8]
            # cloud is in odin1_base_link; odin pose = base pose composed w/o mount
            # recompute odin pose directly:
            po, qo = compose(mo, mo_t, ob[ob_t[i]], ob_t[i])
            pm = q_rot(np.broadcast_to(qo, (len(xyz), 4))[0], np.zeros(3))  # noop
            # transform points: map = po + R(qo) * p
            x, y, z, w = qo
            u = np.array([x, y, z])
            pts = xyz + 0.0
            cr1 = np.cross(np.broadcast_to(u, pts.shape), pts) + w * pts
            pts_m = pts + 2.0 * np.cross(np.broadcast_to(u, pts.shape), cr1) + po
            al_xy = (R2 @ (pts_m[:, :2] - P0[:2]).T).T
            al_z = pts_m[:, 2] - P0[2]
            m = ((al_z > z_lo) & (al_z < z_hi)
                 & (np.abs(al_xy[:, 0] - box_est_x) < 1.3)
                 & (np.abs(al_xy[:, 1] - box_est_y) < 1.3))
            if m.any():
                agg.append(np.column_stack([al_xy[m], al_z[m]]))
        if not agg:
            print(f"{run}: no approach-phase pallet points, skipping")
            continue
        pts = np.concatenate(agg)
        cluster = largest_cluster(pts, anchor=(box_est_x, box_est_y))
        # no trim-refit here: re-fitting a trimmed cluster would trace the
        # artificial (perfectly straight) cut boundary as the sharpest edge
        cx, cy, yaw, dims, err, edge_pts = fit_sharpest_edge(cluster[:, :2])
        print(f"{run}: ground z={z_ground:+.3f}  pallet center=({cx:.3f},{cy:.3f}) "
              f"yaw={np.degrees(yaw):.1f}deg dims~{dims} fit_err={err:.3f}m "
              f"n_pts={len(cluster)} (z-bump est x={box_est_x:.2f})")

        # validation figure
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(pts[:, 0], pts[:, 1], s=1, c="#bbbbbb", label="z-slice points")
        ax.scatter(cluster[:, 0], cluster[:, 1], s=1, c="#3562D6", label="pallet cluster")
        ca, sa = np.cos(yaw), np.sin(yaw)
        Rr = np.array([[ca, -sa], [sa, ca]])
        hw, hh = dims[0] / 2, dims[1] / 2
        corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh], [-hw, -hh]])
        cc = (Rr @ corners.T).T + [cx, cy]
        ax.plot(cc[:, 0], cc[:, 1], "-", c="#B25E1C", lw=2, label="fitted pallet")
        ax.scatter(edge_pts[:, 0], edge_pts[:, 1], s=14, c="#C43131",
                   zorder=5, label="sharpest-edge anchor")
        ax.plot(gtx, gty, "-", c="#2a2a2a", lw=1, label="robot path")
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.set_title(f"{run}: pallet fit (yaw {np.degrees(yaw):.1f} deg, edge rms {err:.3f} m)")
        fig.savefig(DATA_DIR / f"{run}_pallet_fit.png", dpi=130,
                    bbox_inches="tight")
        plt.close(fig)

        if not args.no_update:
            gt["box"]["center"] = [float(cx), float(cy), PALLET_HEIGHT / 2]
            gt["box"]["yaw"] = float(yaw)
            gt["box"]["half_extents"] = [dims[0] / 2, dims[1] / 2,
                                          PALLET_HEIGHT / 2]
            gt["box"]["fit"] = {"err_m": float(err), "n_points": int(len(cluster))}
            json.dump(gt, open(DATA_DIR / f"{run}.json", "w"))


if __name__ == "__main__":
    main()
