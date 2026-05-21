"""Side-by-side Blender import of multiple dt-stability trajectories.

Each ``--input NAME=PATH`` adds one robot. They share the timeline (iteration
N of each input plays simultaneously), are laid out along Y at fixed
separation, and get a static identity label ("Axion" / "MuJoCo") above the
chassis plus the standard "dt = X ms" floating per-iteration label.

Run inside Blender:
    blender --background --python experiments/2_dt_stability/compare_to_blender.py -- \
        --input Axion=experiments/2_dt_stability/results/axion_dt.npz \
        --input MuJoCo=experiments/2_dt_stability/results/mujoco_dt.npz \
        --output experiments/2_dt_stability/results/compare.blend

All loaded npzs must have the same number of iterations and the same number
of frames per iteration (so they stay synced on the timeline).
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

# Make the per-experiment helpers importable when Blender runs this script.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

import import_to_blender as itb  # noqa: E402


SEPARATION_Y = 6.0  # metres between adjacent robots


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Add one robot. Repeatable. Example: --input Axion=axion_dt.npz",
    )
    p.add_argument("--output", type=Path, default=None, help="Save to this .blend after import")
    p.add_argument(
        "--separation",
        type=float,
        default=SEPARATION_Y,
        help=f"Y-axis spacing between robots in metres (default {SEPARATION_Y})",
    )
    return p.parse_args(argv)


def _parse_input(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"--input must look like NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    return name.strip(), Path(path.strip())


# ---------------------------------------------------------------------------
# Per-input pose + shape Y-offset application
# ---------------------------------------------------------------------------

def _shift_shape_y(shape: dict, dy: float) -> dict:
    """Static shapes: bake dy into the local_xform's translation Y."""
    out = dict(shape)
    out["local_xform"] = np.asarray(out["local_xform"], dtype=np.float32).copy()
    if int(out["body_idx"]) == -1:
        out["local_xform"][1] += dy
    return out


def _is_ground_plane(shape: dict) -> bool:
    return int(shape["body_idx"]) == -1 and int(shape["geo_type"]) == itb.GEO_PLANE


def _add_shared_ground(scene: bpy.types.Scene, half_extent_x: float, half_extent_y: float):
    """One large ground plane that spans every robot in the comparison."""
    coll = bpy.data.collections.new("shared_static")
    scene.collection.children.link(coll)
    bpy.ops.mesh.primitive_plane_add(size=2.0)
    obj = bpy.context.active_object
    obj.name = "shared_ground"
    obj.scale = (half_extent_x, half_extent_y, 1.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    mat = itb.make_material("shared_ground", itb.STATIC_COLOR, 1.0, roughness=0.95)
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    coll.objects.link(obj)
    try:
        bpy.context.collection.objects.unlink(obj)
    except RuntimeError:
        pass


def _shift_body_pose_iters(body_pose_iters: np.ndarray, dy: float) -> np.ndarray:
    """Dynamic bodies: shift per-step Y position; rotations untouched."""
    out = body_pose_iters.copy()
    out[..., 1] += dy
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    itb.clear_default_objects()

    inputs = [_parse_input(s) for s in args.input]
    if len(inputs) < 2:
        raise SystemExit("Need at least 2 --input entries; use import_to_blender.py for a single robot.")

    # Load all npzs and verify they have a compatible iteration count / frame count.
    loaded = []
    for name, path in inputs:
        data = np.load(path, allow_pickle=True)
        loaded.append(
            {
                "name": name,
                "fps": float(data["fps"]),
                "body_pose_iters": np.asarray(data["body_pose_iters"]),
                "iter_labels": [str(x) for x in data["iter_labels"]],
                "iter_stable": (
                    np.asarray(data["iter_stable"], dtype=bool)
                    if "iter_stable" in data.files
                    else np.ones(np.asarray(data["body_pose_iters"]).shape[0], dtype=bool)
                ),
                "shapes": list(data["shapes"]),
            }
        )

    n_iters = loaded[0]["body_pose_iters"].shape[0]
    T_motion = loaded[0]["body_pose_iters"].shape[1]
    T_iter = T_motion + itb.HOLD_FRAMES  # motion + freeze pause between iterations
    fps = loaded[0]["fps"]
    for entry in loaded[1:]:
        if entry["body_pose_iters"].shape[0] != n_iters:
            raise SystemExit(
                f"iteration count mismatch: {loaded[0]['name']}={n_iters}, "
                f"{entry['name']}={entry['body_pose_iters'].shape[0]}"
            )
        if entry["body_pose_iters"].shape[1] != T_motion:
            raise SystemExit(
                f"frame count mismatch: {loaded[0]['name']}={T_motion}, "
                f"{entry['name']}={entry['body_pose_iters'].shape[1]}"
            )

    T_total = n_iters * T_iter
    print(
        f"Comparing {len(loaded)} robots: "
        + ", ".join(e['name'] for e in loaded)
        + f" — {n_iters} iterations × ({T_motion} motion + {itb.HOLD_FRAMES} hold) "
        f"= {T_total} frames @ {fps:.1f} fps"
    )

    # Apply Y offsets so robots lay out along Y; drop per-input ground planes
    # (we add one shared plane below to avoid two coincident grounds z-fighting).
    n = len(loaded)
    for i, entry in enumerate(loaded):
        dy = (i - (n - 1) / 2.0) * args.separation
        entry["dy"] = float(dy)
        entry["body_pose_iters"] = _shift_body_pose_iters(entry["body_pose_iters"], dy)
        entry["shapes"] = [
            _shift_shape_y(s, dy) for s in entry["shapes"] if not _is_ground_plane(s)
        ]

    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = max(0, T_total - 1)
    scene.render.fps = max(1, int(round(fps)))

    itb.setup_lighting(scene)

    # Shared ground sized to cover every robot's chassis path along Y plus a
    # margin, with a fixed half-extent in X (the robots all drive along X).
    y_extents = [abs(entry["dy"]) for entry in loaded]
    shared_half_y = max(max(y_extents) + args.separation * 0.6, 6.0)
    _add_shared_ground(scene, half_extent_x=8.0, half_extent_y=shared_half_y)

    # Camera bbox: stable iterations from every loaded npz, clipped against
    # diverging chassis positions (per-input clip on the un-offset chassis,
    # but offsets already applied here).
    cam_pts: list[np.ndarray] = []
    for entry in loaded:
        bp = entry["body_pose_iters"]
        stable = entry["iter_stable"]
        pts = bp[stable, :, 0, :3].reshape(-1, 3) if stable.any() else bp[:, :, 0, :3].reshape(-1, 3)
        pts = pts[
            (np.abs(pts[:, 0]) < itb.CAMERA_BBOX_MAX_EXTENT)
            & (np.abs(pts[:, 1] - entry["dy"]) < itb.CAMERA_BBOX_MAX_EXTENT)
            & (np.abs(pts[:, 2]) < itb.CAMERA_BBOX_MAX_EXTENT)
        ]
        if len(pts) > 0:
            cam_pts.append(pts)
    if cam_pts:
        chassis_points = np.concatenate(cam_pts, axis=0)
    else:
        chassis_points = np.array(
            [[0.0, -args.separation, 0.0], [4.0, args.separation, 0.5]],
            dtype=np.float32,
        )
    itb.setup_camera(scene, chassis_points.astype(np.float32))

    camera = bpy.context.scene.camera

    # Per-input collections + per-input materials, robot, breadcrumbs, labels.
    for i, entry in enumerate(loaded):
        slug = entry["name"].lower().replace(" ", "_")
        static_coll = bpy.data.collections.new(f"{slug}_static")
        live_coll = bpy.data.collections.new(f"{slug}_live")
        scene.collection.children.link(static_coll)
        scene.collection.children.link(live_coll)

        live_materials: dict[str, bpy.types.Material] = {}

        def make_material_lookup(materials_dict, slug):
            def material_for_shape(shape: dict) -> bpy.types.Material:
                body_idx = int(shape["body_idx"])
                geo_type = int(shape["geo_type"])
                if body_idx == -1:
                    if geo_type == itb.GEO_PLANE:
                        key = "ground"
                        if key not in materials_dict:
                            materials_dict[key] = itb.make_material(
                                f"{slug}_ground", itb.STATIC_COLOR, 1.0, roughness=0.95
                            )
                    else:
                        key = "obstacle"
                        if key not in materials_dict:
                            materials_dict[key] = itb.make_material(
                                f"{slug}_obstacle",
                                itb.OBSTACLE_COLOR,
                                1.0,
                                roughness=0.5,
                                emission_strength=0.4,
                            )
                    return materials_dict[key]
                key = f"body_{body_idx}"
                if key not in materials_dict:
                    materials_dict[key] = itb.make_material(
                        f"{slug}_body_{body_idx}",
                        itb.live_color_for_body(body_idx),
                        1.0,
                    )
                return materials_dict[key]

            return material_for_shape

        material_for_shape = make_material_lookup(live_materials, slug)
        live_bodies = itb.build_robot(
            entry["shapes"], slug, material_for_shape, static_coll, live_coll
        )
        for empty in live_bodies.values():
            empty.empty_display_size = 0.0

        label_coll = bpy.data.collections.new(f"{slug}_labels")
        scene.collection.children.link(label_coll)

        # Static identity header — always visible, names the simulator.
        if 0 in live_bodies:
            itb.make_floating_label(
                name=f"{slug}_identity",
                follow_body=live_bodies[0],
                text=entry["name"],
                size=0.4,
                z_offset=1.5,
                collection=label_coll,
                color=itb.LIVE_PALETTE[0],
                camera=camera,
            )
            # Per-iteration "dt = X ms" label.
            itb.make_per_iteration_labels(
                follow_body=live_bodies[0],
                iter_labels=entry["iter_labels"],
                T=T_iter,
                collection=label_coll,
                camera=camera,
                color=itb.LIVE_PALETTE[0],
            )

        trail_coll = bpy.data.collections.new(f"{slug}_breadcrumbs")
        scene.collection.children.link(trail_coll)
        itb.make_breadcrumb_trails(
            entry["body_pose_iters"],
            T=T_iter,
            color=itb.LIVE_PALETTE[0],
            collection=trail_coll,
        )

        # Animate bodies — motion frames followed by a hold on the last pose.
        for it in range(n_iters):
            offset = it * T_iter
            itb.keyframe_bodies(
                live_bodies,
                entry["body_pose_iters"][it],
                frame_offset=offset,
                hold_frames=itb.HOLD_FRAMES,
            )

    if args.output:
        bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
