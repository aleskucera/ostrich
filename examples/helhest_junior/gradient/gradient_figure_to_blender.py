"""Static junior gradient figure with the DETAILED CAD robot.

Like examples/helhest/gradient/gradient_figure_to_blender.py (iterates fanning
out and tightening onto the target), but the robot at the start pose is the
detailed CAD model (appended with materials from the robot kit .blend) instead
of the npz's crude primitive shapes. Trajectory lines, heading arrows and the
terrain wireframe are drawn with the shared helpers exactly as before.

Run inside Blender:
    blender -b --python examples/helhest_junior/gradient/gradient_figure_to_blender.py -- \
        --npz   examples/helhest_junior/gradient/data/junior_traj.npz \
        --frame 5 --output junior_grad_fig --angles hero,side

The CAD bodies come from assets/junior_robot_kit.blend (built by
make_robot_kit.py). Each body mesh is in its body-local frame; we pose body k
at the Newton world transform target_body_pose[frame, k]. Per-body CALIB
matrices correct any residual offset between the CAD-local frame and the Newton
body frame (identity by default; tune if a body sits off).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix
from mathutils import Vector

# Shared, robot-agnostic figure helpers live in the helhest gradient dir.
_HELHEST_GRADIENT = Path(__file__).resolve().parents[2] / "helhest" / "gradient"
sys.path.insert(0, str(_HELHEST_GRADIENT))
import import_to_blender as itb  # noqa: E402
import figure_to_blender as fig  # noqa: E402
import gradient_figure_to_blender as gfb  # noqa: E402  (RAMPS, seq_color, arrows, selection)

KIT_BLEND = Path(__file__).resolve().parent / "assets" / "junior_robot_kit.blend"
DEFAULT_NPZ = Path(__file__).resolve().parent / "data" / "junior_traj.npz"

# Newton body index -> kit object name.
KIT_BODY_NAMES = {0: "chassis", 1: "wheel_left", 2: "wheel_right", 3: "wheel_rear"}

# Per-body calibration: CAD-local frame -> Newton body frame. Identity to start;
# adjust if a body sits off after the first render (see --calib-chassis-* flags).
CALIB: dict[int, Matrix] = {k: Matrix.Identity(4) for k in KIT_BODY_NAMES}


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, default=DEFAULT_NPZ, help="trajectory npz")
    p.add_argument("--kit-blend", type=Path, default=KIT_BLEND, help="robot kit .blend to append")
    p.add_argument(
        "--frame",
        type=int,
        default=5,
        help="Timestep to pose the CAD robot at (and where the trails begin). "
        "Not 0 — the robot is still falling/settling there. Default 5.",
    )
    p.add_argument("--output", type=str, default="junior_grad_fig")
    p.add_argument("--iterates", type=int, nargs="+", default=None, metavar="K")
    p.add_argument("--num-iterates", type=int, default=5)
    p.add_argument("--angles", type=str, default="hero")
    p.add_argument("--views", type=str, nargs="*", default=None, metavar="AZ,EL[,ortho]")
    p.add_argument("--colormap", type=str, default="mono_blue", choices=sorted(gfb.RAMPS))
    p.add_argument("--mono-color", type=float, nargs=3, default=None, metavar=("R", "G", "B"))
    p.add_argument("--mono-dark", type=float, default=0.22)
    p.add_argument("--target-color", type=float, nargs=3, default=(1.0, 0.2, 0.0))
    p.add_argument("--line-emission", type=float, default=2.5)
    p.add_argument("--target-emission", type=float, default=1.2)
    p.add_argument("--line-width", type=float, default=0.025)
    p.add_argument("--line-spacing", type=float, default=0.18)
    p.add_argument("--line-length", type=float, default=0.09)
    p.add_argument("--end-arrows", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--arrow-length", type=float, default=0.3)
    p.add_argument("--arrow-width", type=float, default=0.015)
    p.add_argument("--forward-axis", type=str, default="+x", choices=sorted(gfb.FORWARD_AXIS))
    p.add_argument("--lens", type=float, default=50.0)
    p.add_argument("--res-x", type=int, default=1920)
    p.add_argument("--res-y", type=int, default=1080)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--view-transform", type=str, default="Standard")
    # Calibration nudges for the chassis (the body most likely to need an offset).
    p.add_argument("--calib-chassis-xyz", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                   metavar=("X", "Y", "Z"), help="Translate chassis CAD mesh in its local frame.")
    p.add_argument("--save-blend", type=Path, default=None)
    return p.parse_args(argv)


def append_kit_body(kit_blend: Path, body_idx: int) -> bpy.types.Object:
    """Append one kit object (with its materials) and link it into the scene."""
    name = KIT_BODY_NAMES[body_idx]
    with bpy.data.libraries.load(str(kit_blend), link=False) as (src, dst):
        if name not in src.objects:
            raise ValueError(f"'{name}' not in kit {kit_blend} (have: {list(src.objects)})")
        dst.objects = [name]
    obj = dst.objects[0]
    bpy.context.scene.collection.objects.link(obj)
    return obj


def place_cad_robot(kit_blend: Path, frame_pose: np.ndarray, collection) -> dict[int, bpy.types.Object]:
    """Append + pose each CAD body at the Newton world transform frame_pose[body_idx]."""
    objs: dict[int, bpy.types.Object] = {}
    for body_idx in KIT_BODY_NAMES:
        obj = append_kit_body(kit_blend, body_idx)
        loc, quat = itb.warp_xform_to_blender(frame_pose[body_idx])
        newton_world = Matrix.Translation(loc) @ quat.to_matrix().to_4x4()
        obj.matrix_world = newton_world @ CALIB[body_idx]
        # Re-link into the figure's robot collection.
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        collection.objects.link(obj)
        objs[body_idx] = obj
    return objs


def main():
    args = parse_args()
    CALIB[0] = Matrix.Translation(Vector(args.calib_chassis_xyz))

    itb.clear_default_objects()
    data = np.load(args.npz, allow_pickle=True)
    target_body_pose = np.asarray(data["target_body_pose"])  # [T, bodies, 7]
    body_pose_iters = np.asarray(data["body_pose_iters"])  # [N, T, bodies, 7]
    iter_indices = np.asarray(data["iter_indices"])
    iter_losses = np.asarray(data["iter_losses"])
    shapes = list(data["shapes"])

    n_logged = body_pose_iters.shape[0]
    T = target_body_pose.shape[0]
    sel = gfb.select_iterates(n_logged, args)
    frame = max(0, min(args.frame, T - 1))
    print(f"CAD robot posed at frame {frame}/{T - 1}; {len(sel)} iterate lines at {sel}.")

    scene = bpy.context.scene
    scene.frame_start = scene.frame_end = 0
    scene.render.resolution_x = int(args.res_x)
    scene.render.resolution_y = int(args.res_y)
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    engines = {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    if "BLENDER_EEVEE_NEXT" in engines:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    elif "BLENDER_EEVEE" in engines:
        scene.render.engine = "BLENDER_EEVEE"
    if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = max(1, int(args.samples))
    try:
        scene.view_settings.view_transform = args.view_transform
    except TypeError:
        print(f"  ! view transform '{args.view_transform}' unavailable; leaving default")

    itb.setup_lighting(scene)

    static_coll = bpy.data.collections.new("static")
    robot_coll = bpy.data.collections.new("robot")
    trail_coll = bpy.data.collections.new("trails")
    for c in (static_coll, robot_coll, trail_coll):
        scene.collection.children.link(c)

    # --- Terrain only (body_idx == -1) from the npz shapes, then wireframe it. ---
    terrain_shapes = [s for s in shapes if int(s["body_idx"]) == -1]
    if terrain_shapes:
        terrain_mat = itb.make_material("jgf_terrain", itb.STATIC_COLOR, 1.0, roughness=0.95)
        itb.build_robot(
            terrain_shapes, "jgf_terrain", lambda _bi: terrain_mat,
            static_coll, robot_coll, include_static=True,
        )
        itb.add_terrain_wireframe(static_coll)

    # --- Detailed CAD robot at the chosen frame. ---
    place_cad_robot(args.kit_blend, target_body_pose[frame], robot_coll)

    # --- Iterate trajectory lines + target line (chassis path, body 0). ---
    spacing, seg_len, width = float(args.line_spacing), float(args.line_length), float(args.line_width)
    emission, target_emission = float(args.line_emission), float(args.target_emission)
    ramp = gfb.RAMPS[args.colormap]
    target_color = tuple(args.target_color)
    mono = tuple(args.mono_color) if args.mono_color is not None else None
    mono_dark = float(args.mono_dark)

    styles = []
    for rank in range(len(sel)):
        frac = rank / max(1, len(sel) - 1)
        if mono is not None:
            scale = mono_dark + (1.0 - mono_dark) * frac
            styles.append((tuple(c * scale for c in mono), emission))
        else:
            styles.append((gfb.seq_color(frac, ramp), emission))

    iterate_paths: dict[int, np.ndarray] = {}
    for rank, k in enumerate(sel):
        color, em = styles[rank]
        path = body_pose_iters[k, frame:, 0, :3].astype(np.float32)
        iterate_paths[k] = path
        fig.static_trail(path, color, 1.0, trail_coll, f"jgf_iter_{k}",
                         spacing=spacing, seg_len=seg_len, seg_w=width, emission=em)

    target_path = target_body_pose[frame:, 0, :3].astype(np.float32)
    fig.static_trail(target_path, target_color, 1.0, trail_coll, "jgf_target",
                     spacing=spacing, seg_len=seg_len, seg_w=width * 1.6, emission=target_emission)

    # --- Heading arrows at each trajectory end. ---
    if args.end_arrows:
        forward_local = Vector(gfb.FORWARD_AXIS[args.forward_axis])
        arrow_len, arrow_w = float(args.arrow_length), float(args.arrow_width)

        def heading_at(pose_end_7):
            loc, quat = itb.warp_xform_to_blender(pose_end_7)
            fwd = quat @ forward_local
            return (np.array([loc.x, loc.y, loc.z], dtype=np.float32),
                    np.array([fwd.x, fwd.y, fwd.z], dtype=np.float32))

        for rank, k in enumerate(sel):
            color, em = styles[rank]
            origin, direction = heading_at(body_pose_iters[k, -1, 0])
            gfb.make_heading_arrow(origin, direction, arrow_len, arrow_w, color,
                                   trail_coll, f"jgf_arrow_{k}", emission=em)
        origin, direction = heading_at(target_body_pose[-1, 0])
        gfb.make_heading_arrow(origin, direction, arrow_len, arrow_w * 1.4, target_color,
                               trail_coll, "jgf_arrow_target", emission=target_emission)

    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(args.save_blend.resolve()))
        print(f"Saved {args.save_blend}")

    # --- Render each requested angle + projected-anchor JSON. ---
    res_x, res_y = int(args.res_x), int(args.res_y)
    framing = np.concatenate([target_path] + list(iterate_paths.values()), axis=0).astype(np.float32)

    view_specs: list[tuple[str, float, float, bool]] = []
    for a in args.angles.split(","):
        a = a.strip()
        if not a:
            continue
        preset = fig.ANGLE_PRESETS.get(a)
        if preset is None:
            print(f"  ! unknown angle '{a}', skipping")
            continue
        view_specs.append((a, preset["azimuth"], preset["elevation"], preset["ortho"]))
    for i, spec in enumerate(args.views or []):
        parts = [s.strip() for s in spec.split(",")]
        if len(parts) < 2:
            print(f"  ! bad --views entry '{spec}'; skipping")
            continue
        az, el = float(parts[0]), float(parts[1])
        ortho = len(parts) > 2 and parts[2].lower() in ("ortho", "1", "true", "yes")
        view_specs.append((f"view{i}", az, el, ortho))

    anchors: dict[str, dict] = {}
    for name, az, el, ortho in view_specs:
        cam = fig.place_camera(scene, framing, az, el, ortho, float(args.lens))
        bpy.context.view_layer.update()
        out_png = os.path.abspath(f"{args.output}_{name}.png")
        scene.render.filepath = out_png
        bpy.ops.render.render(write_still=True)
        print(f"  rendered {out_png} (az={az:.1f} el={el:.1f} {'ortho' if ortho else 'persp'})")

        iters = {}
        for k in sel:
            path = iterate_paths[k]
            iters[str(int(iter_indices[k]))] = {
                "position": int(k),
                "loss": float(iter_losses[k]),
                "polyline": [fig.project(scene, cam, res_x, res_y, p) for p in fig._downsample(path)],
            }
        anchors[name] = {
            "resolution": [res_x, res_y], "azimuth": az, "elevation": el,
            "projection": "ortho" if ortho else "perspective",
            "robot_frame": frame,
            "target_polyline": [fig.project(scene, cam, res_x, res_y, p) for p in fig._downsample(target_path)],
            "iterates": iters,
        }

    anchors_path = os.path.abspath(f"{args.output}_anchors.json")
    with open(anchors_path, "w") as f:
        json.dump({"frame": frame, "selected_positions": sel, "angles": anchors}, f, indent=2)
    print(f"  wrote {anchors_path}")


if __name__ == "__main__":
    main()
