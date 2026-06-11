"""Static figure: gradient optimization iterates converging on the target.

Run inside Blender:
    blender -b --python gradient_figure_to_blender.py -- TRAJ.npz --output grad_fig

A robot sits at the start pose on the hill, and the gradient-descent iterates
fan out as an *ordered* set of trajectory lines: a sequential colour ramp
(early = cool/dark, final = warm/bright) plus visible tightening onto a bold,
distinct target line. That ordering + monotonic convergence is what stops it
reading like an MPPI sample cloud — these aren't unordered random rollouts,
they're a sequence each improving on the last.

Renders from several camera angles onto a transparent background and writes a
per-angle JSON of projected 2D pixel coordinates (robot start, each iterate's
trail polyline + a label anchor, the target line, the goal) so the iteration
numbers / legend can be added as a crisp vector overlay.

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
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import import_to_blender as itb  # noqa: E402
import figure_to_blender as fig  # noqa: E402

# Sequential colour ramps for the iterates. A ramp *is* an ordering, which
# reads as "sequence of steps" rather than "cloud of samples". Each is a list
# of anchor colours interpolated across the iterates (early -> final).
RAMPS: dict[str, list[tuple[float, float, float]]] = {
    # punchy magenta -> orange -> yellow; high contrast on grey terrain
    "plasma": [
        (0.05, 0.03, 0.53),
        (0.49, 0.01, 0.66),
        (0.80, 0.27, 0.47),
        (0.97, 0.58, 0.25),
        (0.94, 0.98, 0.13),
    ],
    # full rainbow, maximally distinct per-iterate
    "turbo": [
        (0.19, 0.07, 0.55),
        (0.10, 0.50, 0.99),
        (0.10, 0.90, 0.72),
        (0.55, 0.99, 0.23),
        (0.98, 0.73, 0.10),
        (0.88, 0.14, 0.10),
    ],
    # perceptually uniform purple -> green -> yellow
    "viridis": [
        (0.27, 0.00, 0.33),
        (0.23, 0.32, 0.55),
        (0.13, 0.57, 0.55),
        (0.37, 0.79, 0.38),
        (0.99, 0.91, 0.14),
    ],
    # black -> magenta -> orange -> cream
    "magma": [
        (0.02, 0.01, 0.07),
        (0.32, 0.07, 0.48),
        (0.71, 0.21, 0.47),
        (0.99, 0.55, 0.38),
        (0.99, 0.99, 0.75),
    ],
    # cool blue -> hot red diverging (clear ends)
    "coolwarm": [
        (0.23, 0.30, 0.75),
        (0.55, 0.69, 0.99),
        (0.87, 0.87, 0.87),
        (0.96, 0.60, 0.48),
        (0.71, 0.02, 0.15),
    ],
    # the original ocean ramp
    "ocean": [
        (0.12, 0.16, 0.58),
        (0.10, 0.55, 0.78),
        (0.28, 0.82, 0.42),
        (0.98, 0.78, 0.10),
    ],
    # single-hue blue, dark (old) -> bright azure (new); cool tone
    "mono_blue": [
        (0.02, 0.05, 0.20),
        (0.06, 0.22, 0.62),
        (0.12, 0.50, 1.00),
    ],
}
# Robot-local "forward" axis options for the heading arrow.
FORWARD_AXIS = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}


def to_hex(color: tuple[float, float, float]) -> str:
    return "#%02X%02X%02X" % tuple(int(round(255 * max(0.0, min(1.0, c)))) for c in color)


def seq_color(frac: float, stops: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    frac = float(np.clip(frac, 0.0, 1.0))
    x = frac * (len(stops) - 1)
    i = int(np.floor(x))
    if i >= len(stops) - 1:
        return stops[-1]
    f = x - i
    a, b = stops[i], stops[i + 1]
    return tuple(a[c] + (b[c] - a[c]) * f for c in range(3))


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=Path, help="trajectory.npz from --export")
    p.add_argument("--output", type=str, default="grad_fig", help="Output path stem.")
    p.add_argument(
        "--iterates",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 9],
        metavar="K",
        help="Explicit logged-snapshot positions to draw as iterate lines. "
        "Overrides --num-iterates.",
    )
    p.add_argument(
        "--num-iterates",
        type=int,
        default=5,
        help="If --iterates is not given, evenly pick this many (first+last kept).",
    )
    p.add_argument(
        "--start-frame",
        type=int,
        default=5,
        help="Timestep to begin from, skipping the initial fall/settle so the "
        "robot isn't flying. Default 5; pass -1 to auto-detect when the "
        "chassis stops dropping, or 0 to keep the full trajectory.",
    )
    p.add_argument(
        "--goal-robot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render a faint robot at the target's end pose to mark the goal "
        "(off by default; pass --goal-robot to enable).",
    )
    p.add_argument("--goal-alpha", type=float, default=0.28, help="Goal robot opacity.")
    p.add_argument(
        "--angles",
        type=str,
        default="hero",
        help=f"Comma-separated preset angles from {sorted(fig.ANGLE_PRESETS)}.",
    )
    p.add_argument(
        "--views",
        type=str,
        nargs="*",
        default=None,
        metavar="AZ,EL[,ortho]",
        help="Custom AZ,EL[,ortho] views rendered as view0, view1, ...",
    )
    p.add_argument(
        "--colormap",
        type=str,
        default="mono_blue",
        choices=sorted(RAMPS),
        help="Sequential colour map for the iterate lines (early -> final).",
    )
    p.add_argument(
        "--robot-color",
        type=float,
        nargs=3,
        default=(0.95, 0.45, 0.05),
        metavar=("R", "G", "B"),
        help="Chassis colour of the robot at the start pose (lit surface, not "
        "emissive). Default orange to match the target; wheels stay dark.",
    )
    p.add_argument(
        "--mono-color",
        type=float,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
        help="Use a SINGLE hue for all iterate lines and show age by "
        "lightness: oldest is a dark shade, newest is the full colour "
        "(ramped dark->bright at a fixed emission). Overrides --colormap.",
    )
    p.add_argument(
        "--mono-dark",
        type=float,
        default=0.22,
        help="Lightness of the oldest iterate in --mono-color mode, as a "
        "fraction of the full colour (newest = 1.0).",
    )
    p.add_argument(
        "--target-color",
        type=float,
        nargs=3,
        default=(1.00, 0.20, 0.00),
        metavar=("R", "G", "B"),
        help="Colour of the bold target reference line/arrow. Default warm "
        "orange (deep so it doesn't clip to yellow under emission), which "
        "contrasts the cool blue iterates.",
    )
    p.add_argument(
        "--line-emission",
        type=float,
        default=2.5,
        help="Emission strength of the iterate trajectory lines. Higher = "
        "brighter / more visible glow (the main visibility knob). Kept "
        "moderate so the bright end of a single-hue ramp stays saturated.",
    )
    p.add_argument(
        "--target-emission",
        type=float,
        default=1.2,
        help="Emission strength of the target line/arrow, kept low so the "
        "orange doesn't clip up into yellow/white.",
    )
    p.add_argument(
        "--line-width",
        type=float,
        default=0.025,
        help="Trajectory line half-width (m). Smaller = thinner lines.",
    )
    p.add_argument(
        "--line-spacing",
        type=float,
        default=0.18,
        help="Spacing between segment centers along each line (m). Smaller = "
        "denser.",
    )
    p.add_argument(
        "--line-length",
        type=float,
        default=0.09,
        help="Length of each individual segment/dash along the line (m). "
        "Equal to --line-spacing = solid line; smaller = dashed; larger = "
        "overlapping.",
    )
    p.add_argument(
        "--end-arrows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw an arrow at the end of each trajectory pointing in the "
        "robot's heading (from the chassis orientation at the final frame).",
    )
    p.add_argument("--arrow-length", type=float, default=0.3, help="Heading-arrow length (m).")
    p.add_argument("--arrow-width", type=float, default=0.015, help="Heading-arrow shaft half-width (m).")
    p.add_argument(
        "--forward-axis",
        type=str,
        default="+x",
        choices=sorted(FORWARD_AXIS),
        help="Robot-local forward axis used for the heading arrow. Flip/change "
        "if the arrows point the wrong way.",
    )
    p.add_argument("--lens", type=float, default=50.0, help="Camera focal length, mm.")
    p.add_argument("--res-x", type=int, default=1920)
    p.add_argument("--res-y", type=int, default=1080)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument(
        "--view-transform",
        type=str,
        default="Standard",
        help="Color-management view transform. 'Standard' keeps emissive "
        "colours vivid/true; 'AgX' (Blender default) desaturates bright "
        "values toward white. Use 'AgX' or 'Filmic' for a softer look.",
    )
    p.add_argument("--save-blend", type=Path, default=None)
    return p.parse_args(argv)


def make_heading_arrow(
    origin: np.ndarray,
    direction: np.ndarray,
    length: float,
    width: float,
    color: tuple[float, float, float],
    collection: bpy.types.Collection,
    name: str,
    emission: float = 1.6,
):
    """Build a single arrow (shaft + cone head) from ``origin`` along ``direction``."""
    o = np.asarray(origin, dtype=np.float32)
    d = np.asarray(direction, dtype=np.float32)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return None
    d = d / n
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(d @ ref)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = np.cross(d, ref)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)

    tip = o + d * length
    head_len = min(0.4 * length, 6.0 * width)
    c = o + d * (length - head_len)
    w, hw = width, 2.4 * width
    verts: list[tuple] = []
    faces: list[tuple] = []
    for center in (o, c):
        for su, sv in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            verts.append(tuple(center + su * w * u + sv * w * v))
    faces.extend(
        [
            (0, 1, 2, 3),
            (4, 7, 6, 5),
            (0, 4, 5, 1),
            (1, 5, 6, 2),
            (2, 6, 7, 3),
            (3, 7, 4, 0),
        ]
    )
    hb = len(verts)
    for su, sv in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        verts.append(tuple(c + su * hw * u + sv * hw * v))
    apex = len(verts)
    verts.append(tuple(tip))
    faces.extend(
        [
            (hb + 0, hb + 1, hb + 2, hb + 3),
            (hb + 0, apex, hb + 1),
            (hb + 1, apex, hb + 2),
            (hb + 2, apex, hb + 3),
            (hb + 3, apex, hb + 0),
        ]
    )
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(itb.make_material(f"{name}_mat", color, 1.0, emission_strength=emission))
    collection.objects.link(obj)
    return obj


def detect_settle_frame(
    chassis_z: np.ndarray, eps: float = 0.004, window: int = 3
) -> int:
    """First frame at which the chassis stops dropping (the robot has landed).

    The robot is spawned above the heightmap and falls/settles over the first
    few frames; until then it 'flies'. We find the first frame where the
    per-frame vertical motion stays below ``eps`` for ``window`` consecutive
    frames (searching only the first half of the trajectory so a later
    downhill stretch can't be mistaken for the landing). Returns 0 if it's
    already resting at t=0.
    """
    dz = np.abs(np.diff(np.asarray(chassis_z, dtype=np.float32)))
    horizon = max(1, len(dz) // 2)
    for t in range(horizon):
        seg = dz[t : t + window]
        if len(seg) and np.all(seg < eps):
            return t
    return 0


def select_iterates(n: int, args) -> list[int]:
    if args.iterates is not None:
        sel = sorted({k for k in args.iterates if 0 <= k < n})
        return sel or [0, n - 1]
    if args.num_iterates <= 0 or args.num_iterates >= n:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, args.num_iterates).round().astype(int).tolist()))


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
    sel = select_iterates(n_logged, args)

    # Skip the initial fall/settle so the robot isn't flying at the start.
    if args.start_frame >= 0:
        start_frame = min(args.start_frame, T - 1)
    else:
        start_frame = detect_settle_frame(target_body_pose[:, 0, 2])
    print(
        f"Drawing {len(sel)} iterate lines at positions {sel} "
        f"(real iters {[int(iter_indices[k]) for k in sel]}); "
        f"start frame {start_frame}/{T - 1} (after settle)."
    )

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

    # View transform: 'Standard' keeps emissive colours vivid (AgX, Blender's
    # default, washes bright saturated colours toward white).
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

    # Hero robot at the start pose: orange chassis, dark wheels, grey terrain.
    robot_color = tuple(args.robot_color)
    robot_mats: dict[int, bpy.types.Material] = {}

    def material_for_robot(body_idx: int) -> bpy.types.Material:
        if body_idx not in robot_mats:
            if body_idx == -1:
                robot_mats[body_idx] = itb.make_material(
                    "gf_static", itb.STATIC_COLOR, 1.0, roughness=0.95
                )
            elif body_idx == 0:
                robot_mats[body_idx] = itb.make_material(
                    "gf_chassis", robot_color, 1.0, roughness=0.4
                )
            else:
                robot_mats[body_idx] = itb.make_material(
                    f"gf_wheel_{body_idx}", itb.WHEEL_COLOR, 1.0, roughness=0.6
                )
        return robot_mats[body_idx]

    robot_bodies = itb.build_robot(
        shapes, "gf_robot", material_for_robot, static_coll, robot_coll, include_static=True
    )
    fig.pose_robot(robot_bodies, target_body_pose[start_frame])  # settled start pose
    itb.add_terrain_wireframe(static_coll)

    # Optional faint goal robot at the target's end pose.
    if args.goal_robot:
        goal_mats: dict[int, bpy.types.Material] = {}

        def material_for_goal(body_idx: int) -> bpy.types.Material:
            if body_idx not in goal_mats:
                goal_mats[body_idx] = itb.make_material(
                    f"gf_goal_{body_idx}", fig.REAL_CHASSIS if body_idx == 0 else fig.REAL_WHEEL, 1.0
                )
            return goal_mats[body_idx]

        goal_bodies = itb.build_robot(
            shapes, "gf_goal", material_for_goal, static_coll, robot_coll, include_static=False
        )
        fig.pose_robot(goal_bodies, target_body_pose[-1])
        fig.make_translucent(goal_mats.values(), float(args.goal_alpha))

    # Iterate trajectory lines, drawn from the settled start frame onward
    # (so they begin at the resting robot, not the airborne spawn). Ordered
    # by the sequential colour ramp, they tighten onto the bold target line.
    spacing = float(args.line_spacing)
    seg_len = float(args.line_length)
    width = float(args.line_width)
    emission = float(args.line_emission)
    target_emission = float(args.target_emission)
    ramp = RAMPS[args.colormap]
    target_color = tuple(args.target_color)
    mono = tuple(args.mono_color) if args.mono_color is not None else None
    mono_dark = float(args.mono_dark)

    # Per-iterate (colour, emission); emission is fixed for every line. In
    # mono mode the single hue is ramped dark->bright so age shows as
    # lightness (oldest dark, newest the full colour).
    styles: list[tuple[tuple[float, float, float], float]] = []
    for rank in range(len(sel)):
        frac = rank / max(1, len(sel) - 1)
        if mono is not None:
            scale = mono_dark + (1.0 - mono_dark) * frac
            styles.append((tuple(c * scale for c in mono), emission))
        else:
            styles.append((seq_color(frac, ramp), emission))

    iterate_paths: dict[int, np.ndarray] = {}
    print("Trajectory colours (base material RGB; render applies emission + view transform):")
    for rank, k in enumerate(sel):
        color, em = styles[rank]
        print(f"  pos {k:>2} (iter {int(iter_indices[k]):>3})  {to_hex(color)}  "
              f"rgb({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})  emission {em:.2f}")
        path = body_pose_iters[k, start_frame:, 0, :3].astype(np.float32)
        iterate_paths[k] = path
        fig.static_trail(
            path,
            color,
            1.0,
            trail_coll,
            f"gf_iter_{k}",
            spacing=spacing,
            seg_len=seg_len,
            seg_w=width,
            emission=em,
        )
    print(f"  target           {to_hex(target_color)}  "
          f"rgb({target_color[0]:.3f}, {target_color[1]:.3f}, {target_color[2]:.3f})")

    target_path = target_body_pose[start_frame:, 0, :3].astype(np.float32)
    fig.static_trail(
        target_path,
        target_color,
        1.0,
        trail_coll,
        "gf_target",
        spacing=spacing,
        seg_len=seg_len,
        seg_w=width * 1.6,
        emission=target_emission,
    )

    # Heading arrows at each trajectory end, oriented by the chassis
    # orientation (quaternion) at the final frame.
    if args.end_arrows:
        forward_local = Vector(FORWARD_AXIS[args.forward_axis])
        arrow_len = float(args.arrow_length)
        arrow_w = float(args.arrow_width)

        def heading_at(pose_end_7):
            loc, quat = itb.warp_xform_to_blender(pose_end_7)
            fwd = quat @ forward_local
            return np.array([loc.x, loc.y, loc.z], dtype=np.float32), np.array(
                [fwd.x, fwd.y, fwd.z], dtype=np.float32
            )

        for rank, k in enumerate(sel):
            color, em = styles[rank]
            origin, direction = heading_at(body_pose_iters[k, -1, 0])
            make_heading_arrow(
                origin,
                direction,
                arrow_len,
                arrow_w,
                color,
                trail_coll,
                f"gf_arrow_{k}",
                emission=em,
            )
        origin, direction = heading_at(target_body_pose[-1, 0])
        make_heading_arrow(
            origin,
            direction,
            arrow_len,
            arrow_w * 1.4,
            target_color,
            trail_coll,
            "gf_arrow_target",
            emission=target_emission,
        )

    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(args.save_blend.resolve()))
        print(f"Saved {args.save_blend}")

    # --- Render each view + collect projected anchors.
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
                "position": k,
                "loss": float(iter_losses[k]),
                "polyline": [fig.project(scene, cam, res_x, res_y, p) for p in fig._downsample(path)],
                "label_anchor": fig.project(scene, cam, res_x, res_y, path[len(path) // 2]),
            }
        anchors[name] = {
            "resolution": [res_x, res_y],
            "azimuth": az,
            "elevation": el,
            "projection": "ortho" if ortho else "perspective",
            "robot_start": fig.project(scene, cam, res_x, res_y, target_body_pose[start_frame, 0, :3]),
            "goal": fig.project(scene, cam, res_x, res_y, target_body_pose[-1, 0, :3]),
            "target_polyline": [fig.project(scene, cam, res_x, res_y, p) for p in fig._downsample(target_path)],
            "iterates": iters,
        }

    anchors_path = os.path.abspath(f"{args.output}_anchors.json")
    with open(anchors_path, "w") as f:
        json.dump({"selected_positions": sel, "angles": anchors}, f, indent=2)
    print(f"  wrote {anchors_path}")


if __name__ == "__main__":
    main()
