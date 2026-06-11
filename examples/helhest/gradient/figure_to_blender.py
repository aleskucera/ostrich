"""Render a publication still of the sim-to-real trajectory fit.

Run inside Blender:
    blender -b --python figure_to_blender.py -- TRAJ.npz --output figure

Unlike import_to_blender.py (which builds the full animation), this freezes a
single moment and stages three robots for a paper figure:

  * the *real* robot   — gunmetal, opaque: the reference trajectory we fit;
  * the *current* sim  — bright, opaque: this iteration's attempt;
  * the *previous* sim — bright, translucent: an earlier iteration (the "old"
                          attempt), to show convergence.

It renders the same frozen scene from several camera angles (hero / top /
side / front; top-down is orthographic) onto a transparent background, and
writes a per-angle JSON of projected 2D pixel coordinates (each robot, the
start + goal markers, the trail polylines, and the travel direction) so the
explanatory text, arrows, legend and scale bar can be added as a crisp
vector overlay in Inkscape / TikZ / matplotlib.

The npz is produced by trajectory_spline_surface_fast.py --export TRAJ.npz.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix
from mathutils import Vector

# Reuse the animation importer's scene-building helpers.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import import_to_blender as itb  # noqa: E402

# Gunmetal-industrial reference robot (matches the video's "real" robot).
REAL_CHASSIS = (0.20, 0.22, 0.26)
REAL_WHEEL = (0.05, 0.05, 0.06)

# Camera presets: (azimuth°, elevation°, orthographic?).
ANGLE_PRESETS: dict[str, dict] = {
    "hero": dict(azimuth=49.21, elevation=20.21, ortho=False),
    "top": dict(azimuth=49.21, elevation=89.0, ortho=True),
    "side": dict(azimuth=90.0, elevation=8.0, ortho=False),
    "front": dict(azimuth=0.0, elevation=12.0, ortho=False),
}


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=Path, help="trajectory.npz from --export")
    p.add_argument(
        "--output",
        type=str,
        default="figure",
        help=(
            "Output path stem. Renders <stem>_<angle>.png per angle and "
            "<stem>_anchors.json with the projected 2D coordinates."
        ),
    )
    p.add_argument(
        "--t-frame",
        type=int,
        default=-1,
        help=(
            "Timestep within the trajectory to freeze (0-based). Negative "
            "counts from the end, so -1 (default) is the final pose."
        ),
    )
    p.add_argument(
        "--current-snapshot",
        type=int,
        default=-1,
        help=(
            "Logged-snapshot position for the bright 'current simulation' "
            "robot (0-based; negative counts from the end). Default: the "
            "midpoint of the recorded set."
        ),
    )
    p.add_argument(
        "--previous-snapshot",
        type=int,
        default=-1,
        help=(
            "Logged-snapshot position for the translucent 'previous "
            "iteration' robot. Default: two snapshots before --current."
        ),
    )
    p.add_argument(
        "--previous-alpha",
        type=float,
        default=0.32,
        help="Opacity of the previous-iteration ghost robot.",
    )
    p.add_argument(
        "--angles",
        type=str,
        default="hero,top,side,front",
        help=f"Comma-separated camera angles from {sorted(ANGLE_PRESETS)}.",
    )
    p.add_argument("--lens", type=float, default=50.0, help="Camera focal length, mm.")
    p.add_argument("--res-x", type=int, default=2560, help="Render width in pixels.")
    p.add_argument("--res-y", type=int, default=1440, help="Render height in pixels.")
    p.add_argument("--samples", type=int, default=256, help="Eevee TAA samples.")
    p.add_argument(
        "--save-blend",
        type=Path,
        default=None,
        help="Also save the staged scene to this .blend for inspection.",
    )
    p.add_argument(
        "--views",
        type=str,
        nargs="*",
        default=None,
        metavar="AZ,EL[,ortho]",
        help=(
            "Custom camera views as AZ,EL degree pairs, rendered alongside "
            "--angles as view0, view1, ... e.g. --views 72,16 85,8. The "
            "hero angle is azimuth 49; push azimuth toward 90 for a more "
            "side-on shot, toward 0 for head-on. Append ',ortho' for an "
            "orthographic view (e.g. 49,89,ortho for top-down)."
        ),
    )
    return p.parse_args(argv)


def pose_robot(body_empties: dict[int, bpy.types.Object], pose_t: np.ndarray):
    """Set each body Empty to a single static pose (no keyframes)."""
    for body_idx, empty in body_empties.items():
        loc, quat = itb.warp_xform_to_blender(pose_t[body_idx])
        empty.location = loc
        empty.rotation_mode = "QUATERNION"
        empty.rotation_quaternion = quat
        empty.empty_display_size = 0.0


def make_translucent(materials, alpha: float):
    """Drop a set of materials to ``alpha`` and enable hashed transparency."""
    for mat in materials:
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Alpha"].default_value = float(alpha)
        for attr, value in (("blend_method", "HASHED"), ("shadow_method", "HASHED")):
            if hasattr(mat, attr):
                setattr(mat, attr, value)
        if hasattr(mat, "surface_render_method"):
            mat.surface_render_method = "DITHERED"


def static_trail(
    path: np.ndarray,
    color: tuple[float, float, float],
    alpha: float,
    collection: bpy.types.Collection,
    name: str,
    spacing: float = 0.4,
    seg_len: float = 0.18,
    seg_w: float = 0.04,
    emission: float = 1.2,
) -> np.ndarray:
    """Build one static breadcrumb mesh along ``path`` ([T, 3]).

    Returns the resampled center points (world coords) so the caller can
    project them for the overlay JSON.
    """
    sampled, _ = itb._resample_equidistant(path.astype(np.float32), spacing)
    tangents = itb._tangents(sampled)
    mat = itb.make_material(name, color, alpha, emission_strength=emission)
    if alpha < 1.0:
        make_translucent([mat], alpha)
    seg = itb._build_segment_cluster(name, sampled, tangents, length=seg_len, width=seg_w)
    seg.data.materials.append(mat)
    collection.objects.link(seg)
    return sampled


def place_camera(
    scene: bpy.types.Scene,
    points: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    ortho: bool,
    lens: float,
    margin: float = 1.12,
) -> bpy.types.Object:
    """Auto-fit a camera (perspective or orthographic) framing ``points``.

    The camera matrix is set directly (no Track-To constraint) so projection
    via world_to_camera_view is exact without a depsgraph round-trip.
    """
    for obj in list(scene.objects):
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)

    cam_data = bpy.data.cameras.new("figure_cam")
    cam_data.lens = float(lens)
    cam_obj = bpy.data.objects.new("figure_cam", cam_data)

    pad = np.array([0.5, 0.5, 0.4], dtype=np.float32)
    pmin = points.min(axis=0) - pad
    pmax = points.max(axis=0) + pad
    center = (pmin + pmax) / 2.0

    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    offset_dir = np.array(
        [np.cos(el) * np.sin(az), -np.cos(el) * np.cos(az), np.sin(el)], dtype=np.float32
    )
    view_dir = -offset_dir  # camera -> aim

    # Camera basis; fall back to +Y up when looking near-vertically (top-down).
    up_ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(view_dir, up_ref))) > 0.99:
        up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(view_dir, up_ref)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, view_dir)
    true_up /= np.linalg.norm(true_up)

    corners = np.array(
        [
            [pmin[0] if i & 1 else pmax[0], pmin[1] if i & 2 else pmax[1], pmin[2] if i & 4 else pmax[2]]
            for i in range(8)
        ],
        dtype=np.float32,
    )
    rel = corners - center
    half_h = float(np.max(np.abs(rel @ right)))
    half_v = float(np.max(np.abs(rel @ true_up)))
    diag = float(np.linalg.norm(pmax - pmin))

    res_x = max(1, scene.render.resolution_x)
    res_y = max(1, scene.render.resolution_y)
    aspect = res_x / res_y

    if ortho:
        cam_data.type = "ORTHO"
        if res_x >= res_y:
            cam_data.ortho_scale = max(2 * half_h, 2 * half_v * aspect) * margin
        else:
            cam_data.ortho_scale = max(2 * half_v, 2 * half_h / aspect) * margin
        cam_distance = diag * 2.0 + 1.0
    else:
        sensor_w = float(cam_data.sensor_width)
        if res_x >= res_y:
            sensor_w_eff, sensor_h_eff = sensor_w, sensor_w * res_y / res_x
        else:
            sensor_h_eff, sensor_w_eff = sensor_w, sensor_w * res_x / res_y
        half_fov_h = np.arctan(0.5 * sensor_w_eff / cam_data.lens)
        half_fov_v = np.arctan(0.5 * sensor_h_eff / cam_data.lens)
        d_h = half_h / np.tan(half_fov_h) if half_h > 0 else 0.0
        d_v = half_v / np.tan(half_fov_v) if half_v > 0 else 0.0
        cam_distance = max(max(d_h, d_v) * margin, 3.0)

    cam_data.clip_start = 0.01
    cam_data.clip_end = cam_distance + diag * 4.0 + 10.0

    loc = center + offset_dir * cam_distance
    back = -view_dir  # Blender camera looks down -Z, so local +Z = back
    cam_obj.matrix_world = Matrix(
        (
            (right[0], true_up[0], back[0], loc[0]),
            (right[1], true_up[1], back[1], loc[1]),
            (right[2], true_up[2], back[2], loc[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    return cam_obj


def project(scene, cam, res_x: int, res_y: int, world_xyz) -> list[float]:
    """Project a world point to [px, py] image coordinates (origin top-left)."""
    co = world_to_camera_view(scene, cam, Vector([float(c) for c in world_xyz]))
    return [round(co.x * res_x, 1), round((1.0 - co.y) * res_y, 1)]


def _downsample(arr: np.ndarray, n: int = 28) -> np.ndarray:
    if len(arr) <= n:
        return arr
    idx = np.linspace(0, len(arr) - 1, n).round().astype(int)
    return arr[idx]


def main():
    args = parse_args()
    itb.clear_default_objects()
    data = np.load(args.npz, allow_pickle=True)
    target_body_pose = np.asarray(data["target_body_pose"])  # [T, bodies, 7]
    body_pose_iters = np.asarray(data["body_pose_iters"])  # [N, T, bodies, 7]
    iter_indices = np.asarray(data["iter_indices"])
    iter_losses = np.asarray(data["iter_losses"])
    shapes = list(data["shapes"])

    n_logged = body_pose_iters.shape[0]
    T = target_body_pose.shape[0]

    t = args.t_frame if args.t_frame >= 0 else T + args.t_frame
    t = max(0, min(t, T - 1))

    cur = args.current_snapshot
    cur = (n_logged + cur) if cur < 0 else cur
    if args.current_snapshot == -1:
        cur = n_logged // 2
    cur = max(0, min(cur, n_logged - 1))

    prev = args.previous_snapshot
    if args.previous_snapshot == -1:
        prev = max(0, cur - 2)
    else:
        prev = (n_logged + prev) if prev < 0 else prev
    prev = max(0, min(prev, n_logged - 1))

    print(
        f"Figure @ t={t}/{T - 1}: real + current snapshot {cur} "
        f"(iter {int(iter_indices[cur])}, loss {float(iter_losses[cur]):.4f}) "
        f"+ previous snapshot {prev} "
        f"(iter {int(iter_indices[prev])}, loss {float(iter_losses[prev]):.4f})."
    )

    scene = bpy.context.scene
    scene.frame_start = scene.frame_end = 0
    scene.render.resolution_x = int(args.res_x)
    scene.render.resolution_y = int(args.res_y)
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    engine_items = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
    engines = {e.identifier for e in engine_items}
    if "BLENDER_EEVEE_NEXT" in engines:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    elif "BLENDER_EEVEE" in engines:
        scene.render.engine = "BLENDER_EEVEE"
    if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = max(1, int(args.samples))

    itb.setup_lighting(scene)

    static_coll = bpy.data.collections.new("static")
    real_coll = bpy.data.collections.new("real")
    sim_coll = bpy.data.collections.new("sim")
    trail_coll = bpy.data.collections.new("trails")
    for c in (static_coll, real_coll, sim_coll, trail_coll):
        scene.collection.children.link(c)

    # --- Real (reference) robot: gunmetal, opaque. Terrain is built here.
    real_mats: dict[int, bpy.types.Material] = {}

    def material_for_real(body_idx: int) -> bpy.types.Material:
        if body_idx not in real_mats:
            if body_idx == -1:
                real_mats[body_idx] = itb.make_material(
                    "fig_real_static", itb.STATIC_COLOR, 1.0, roughness=0.95
                )
            elif body_idx == 0:
                real_mats[body_idx] = itb.make_material(
                    "fig_real_chassis", REAL_CHASSIS, 1.0, roughness=0.35
                )
            else:
                real_mats[body_idx] = itb.make_material(
                    f"fig_real_wheel_{body_idx}", REAL_WHEEL, 1.0, roughness=0.6
                )
        return real_mats[body_idx]

    real_bodies = itb.build_robot(
        shapes, "fig_real", material_for_real, static_coll, real_coll, include_static=True
    )
    pose_robot(real_bodies, target_body_pose[t])
    itb.add_terrain_wireframe(static_coll)

    # --- Bright current-iteration sim robot, opaque.
    def make_sim(namespace: str, alpha: float):
        mats: dict[int, bpy.types.Material] = {}

        def factory(body_idx: int, _store=mats, _ns=namespace) -> bpy.types.Material:
            if body_idx not in _store:
                _store[body_idx] = itb.make_material(
                    f"{_ns}_body_{body_idx}", itb.live_color_for_body(body_idx), 1.0
                )
            return _store[body_idx]

        bodies = itb.build_robot(
            shapes, namespace, factory, static_coll, sim_coll, include_static=False
        )
        if alpha < 1.0:
            make_translucent(mats.values(), alpha)
        return bodies

    current_bodies = make_sim("fig_sim_cur", alpha=1.0)
    pose_robot(current_bodies, body_pose_iters[cur, t])

    previous_bodies = make_sim("fig_sim_prev", alpha=float(args.previous_alpha))
    pose_robot(previous_bodies, body_pose_iters[prev, t])

    # --- Static trails (the path each robot took up to the frozen moment).
    real_path = target_body_pose[:, 0, :3]
    cur_path = body_pose_iters[cur, :, 0, :3]
    prev_path = body_pose_iters[prev, :, 0, :3]
    REAL_TRAIL_COLOR = (0.95, 0.10, 0.50)  # magenta reference line
    static_trail(real_path, REAL_TRAIL_COLOR, 1.0, trail_coll, "fig_real_trail")
    static_trail(cur_path, itb.LIVE_PALETTE[0], 1.0, trail_coll, "fig_cur_trail")
    static_trail(prev_path, itb.LIVE_PALETTE[0], float(args.previous_alpha), trail_coll, "fig_prev_trail")

    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(args.save_blend.resolve()))
        print(f"Saved {args.save_blend}")

    # --- Render each requested angle + collect projected anchors.
    res_x, res_y = int(args.res_x), int(args.res_y)
    framing_points = np.concatenate([real_path, cur_path, prev_path], axis=0).astype(np.float32)

    # Build the view list: named presets from --angles, then any custom
    # AZ,EL[,ortho] views from --views (named view0, view1, ...).
    view_specs: list[tuple[str, float, float, bool]] = []
    for a in args.angles.split(","):
        a = a.strip()
        if not a:
            continue
        preset = ANGLE_PRESETS.get(a)
        if preset is None:
            print(f"  ! unknown angle '{a}', skipping (known: {sorted(ANGLE_PRESETS)})")
            continue
        view_specs.append((a, preset["azimuth"], preset["elevation"], preset["ortho"]))
    for i, spec in enumerate(args.views or []):
        parts = [s.strip() for s in spec.split(",")]
        if len(parts) < 2:
            print(f"  ! bad --views entry '{spec}', expected AZ,EL[,ortho]; skipping")
            continue
        az, el = float(parts[0]), float(parts[1])
        ortho = len(parts) > 2 and parts[2].lower() in ("ortho", "1", "true", "yes")
        view_specs.append((f"view{i}", az, el, ortho))

    anchors: dict[str, dict] = {}
    for name, az, el, ortho in view_specs:
        cam = place_camera(
            scene,
            framing_points,
            azimuth_deg=az,
            elevation_deg=el,
            ortho=ortho,
            lens=float(args.lens),
        )
        bpy.context.view_layer.update()

        out_png = os.path.abspath(f"{args.output}_{name}.png")
        scene.render.filepath = out_png
        bpy.ops.render.render(write_still=True)
        print(f"  rendered {out_png} (az={az:.1f} el={el:.1f} {'ortho' if ortho else 'persp'})")

        anchors[name] = {
            "resolution": [res_x, res_y],
            "azimuth": az,
            "elevation": el,
            "projection": "ortho" if ortho else "perspective",
            "points": {
                "real_chassis": project(scene, cam, res_x, res_y, target_body_pose[t, 0, :3]),
                "current_chassis": project(scene, cam, res_x, res_y, body_pose_iters[cur, t, 0, :3]),
                "previous_chassis": project(scene, cam, res_x, res_y, body_pose_iters[prev, t, 0, :3]),
                "start": project(scene, cam, res_x, res_y, real_path[0]),
                "goal": project(scene, cam, res_x, res_y, real_path[-1]),
                "direction_from": project(scene, cam, res_x, res_y, real_path[0]),
                "direction_to": project(
                    scene, cam, res_x, res_y, real_path[min(len(real_path) - 1, max(1, len(real_path) // 12))]
                ),
            },
            "trails": {
                "real": [project(scene, cam, res_x, res_y, p) for p in _downsample(real_path)],
                "current": [project(scene, cam, res_x, res_y, p) for p in _downsample(cur_path)],
                "previous": [project(scene, cam, res_x, res_y, p) for p in _downsample(prev_path)],
            },
        }

    anchors_meta = {
        "t_frame": t,
        "T": T,
        "current_snapshot": cur,
        "previous_snapshot": prev,
        "current_iter": int(iter_indices[cur]),
        "previous_iter": int(iter_indices[prev]),
        "current_loss": float(iter_losses[cur]),
        "previous_loss": float(iter_losses[prev]),
        "angles": anchors,
    }
    anchors_path = os.path.abspath(f"{args.output}_anchors.json")
    with open(anchors_path, "w") as f:
        json.dump(anchors_meta, f, indent=2)
    print(f"  wrote {anchors_path}")


if __name__ == "__main__":
    main()
