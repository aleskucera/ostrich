"""Import an axion trajectory.npz dump as a keyframed Blender scene.

Run inside Blender:
    blender [template.blend] --background --python import_to_blender.py -- TRAJ.npz
    blender [template.blend] --python import_to_blender.py -- TRAJ.npz --output animated.blend

The npz is produced by trajectory_spline_surface_fast.py --export TRAJ.npz.
Builds two robots (live + translucent ghost) and stacks every recorded
optimization iteration back-to-back on the timeline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Quaternion
from mathutils import Vector

GEO_PLANE = 0
GEO_HFIELD = 1
GEO_SPHERE = 2
GEO_CAPSULE = 3
GEO_ELLIPSOID = 4
GEO_CYLINDER = 5
GEO_BOX = 6
GEO_MESH = 7
GEO_CONE = 9
GEO_CONVEX_MESH = 10


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=Path, help="trajectory.npz from --export")
    p.add_argument("--output", type=Path, default=None, help="Save to this .blend after import")
    p.add_argument("--ghost-color", type=float, nargs=3, default=(0.95, 0.10, 0.50))
    p.add_argument("--ghost-alpha", type=float, default=0.3)
    p.add_argument(
        "--pause-seconds",
        type=float,
        default=1.0,
        help="Hold each iteration's final pose for this many seconds before the next starts.",
    )
    p.add_argument(
        "--video",
        type=str,
        default=None,
        help=(
            "Render output path baked into the .blend so `blender -b out.blend -a` "
            "produces an MP4. Defaults to <output stem>.mp4 next to --output, or "
            "//render.mp4 when --output is not set. Use Blender's `//` prefix for "
            "paths relative to the .blend."
        ),
    )
    p.add_argument("--lens", type=float, default=50.0, help="Camera focal length in mm.")
    p.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Multiplier on auto-computed camera distance (ignored when --distance is set).",
    )
    p.add_argument("--azimuth", type=float, default=49.21, help="Camera azimuth angle, degrees.")
    p.add_argument(
        "--elevation", type=float, default=20.21, help="Camera elevation angle, degrees."
    )
    p.add_argument(
        "--distance",
        type=float,
        default=16.340,
        help=(
            "Explicit camera distance from the aim point (meters). When set, "
            "bypasses the FOV/bbox auto-fit and --zoom — gives a fully "
            "reproducible framing across runs. Pass a negative value to "
            "re-enable the auto-fit."
        ),
    )
    p.add_argument(
        "--aim",
        type=float,
        nargs=3,
        default=(3.123, -0.929, 1.504),
        metavar=("X", "Y", "Z"),
        help="Explicit look-at point. Defaults to the padded bbox center.",
    )
    p.add_argument(
        "--fstop",
        type=float,
        default=1.0,
        help=(
            "Camera aperture f-stop for depth-of-field. Lower = stronger "
            "background blur (heightmap edges defocus into a blob). "
            "Focus tracks the camera-aim Empty automatically. Pass 0 to "
            "disable DoF."
        ),
    )
    p.add_argument(
        "--samples",
        type=int,
        default=256,
        help=(
            "Eevee TAA samples per rendered frame. Default 64 (Blender's "
            "stock); bump to 128–512 if DoF blur looks grainy. Render time "
            "scales roughly linearly with this."
        ),
    )
    p.add_argument(
        "--fog-density",
        type=float,
        default=0.04,
        help=(
            "Atmospheric haze density (volume-scatter coefficient, units of "
            "1/m). Larger = thicker fog, distant geometry fades faster. "
            "0.04 leaves the robot mostly clear at ~16 m and fades the "
            "heightmap edges out at ~30 m. Pass 0 to disable fog."
        ),
    )
    p.add_argument(
        "--fog-color",
        type=float,
        nargs=3,
        default=(0.95, 0.78, 0.55),
        metavar=("R", "G", "B"),
        help="Volume-scatter tint (warm golden-hour by default).",
    )
    p.add_argument(
        "--fog-start",
        type=float,
        default=14.0,
        help=(
            "Distance from camera at which fog begins (meters). With "
            "fog-start=5 the first 5 m of view stay completely clear; "
            "useful for keeping the robot crisp while still hazing out "
            "the heightmap edge."
        ),
    )
    p.add_argument(
        "--fog-falloff",
        type=str,
        default="QUADRATIC",
        choices=("QUADRATIC", "LINEAR", "INVERSE_QUADRATIC"),
        help=(
            "Shape of the fog curve. QUADRATIC ramps in slowly (clearer "
            "up close, denser far away). LINEAR is uniform. "
            "INVERSE_QUADRATIC bites fast and then levels off — feels "
            "like a wall of fog."
        ),
    )
    p.add_argument(
        "--labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Show floating labels above the robots: 'Target Trajectory' "
            "above the ghost, 'Iter N | Loss X' above the live robot. "
            "Off by default; pass --labels to enable."
        ),
    )
    return p.parse_args(argv)


# Vibrant per-body palette for the live robot.
WHEEL_COLOR = (0.08, 0.08, 0.08)  # matte black rubber — applied to every wheel
LIVE_PALETTE = {
    0: (0.98, 0.78, 0.10),  # chassis: caterpillar yellow
    1: WHEEL_COLOR,
    2: WHEEL_COLOR,
    3: WHEEL_COLOR,
}
LIVE_FALLBACK = WHEEL_COLOR  # extra bodies share the wheel color too
STATIC_COLOR = (0.22, 0.22, 0.24)  # terrain: muted cool gray


def live_color_for_body(body_idx: int) -> tuple[float, float, float]:
    if body_idx == -1:
        return STATIC_COLOR
    return LIVE_PALETTE.get(body_idx, LIVE_FALLBACK)


def make_material(
    name: str,
    color: tuple[float, float, float],
    alpha: float,
    roughness: float | None = None,
    emission_strength: float = 0.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Alpha"].default_value = alpha
    if roughness is not None and "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = float(roughness)
    if emission_strength > 0.0:
        for emission_color_key in ("Emission Color", "Emission"):
            if emission_color_key in bsdf.inputs:
                bsdf.inputs[emission_color_key].default_value = (*color, 1.0)
                break
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = float(emission_strength)
    if alpha < 1.0:
        for attr, value in (("blend_method", "HASHED"), ("shadow_method", "HASHED")):
            if hasattr(mat, attr):
                setattr(mat, attr, value)
        if hasattr(mat, "surface_render_method"):
            mat.surface_render_method = "DITHERED"
    return mat


def make_shape_object(shape: dict, name: str) -> bpy.types.Object:
    """Create a Blender mesh object whose local geometry matches the newton shape."""
    gt = int(shape["geo_type"])
    scale = np.asarray(shape["geo_scale"], dtype=np.float32)

    if gt == GEO_BOX:
        bpy.ops.mesh.primitive_cube_add(size=2.0)
        obj = bpy.context.active_object
        obj.scale = (float(scale[0]), float(scale[1]), float(scale[2]))
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    elif gt == GEO_SPHERE:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=float(scale[0]))
        obj = bpy.context.active_object
    elif gt == GEO_CYLINDER:
        bpy.ops.mesh.primitive_cylinder_add(radius=float(scale[0]), depth=2.0 * float(scale[1]))
        obj = bpy.context.active_object
    elif gt == GEO_CAPSULE:
        # Approximate as cylinder + two spheres joined; for visual purposes a stretched cylinder works.
        bpy.ops.mesh.primitive_cylinder_add(radius=float(scale[0]), depth=2.0 * float(scale[1]))
        obj = bpy.context.active_object
    elif gt == GEO_CONE:
        bpy.ops.mesh.primitive_cone_add(radius1=float(scale[0]), depth=2.0 * float(scale[1]))
        obj = bpy.context.active_object
    elif gt in (GEO_MESH, GEO_CONVEX_MESH):
        verts = np.asarray(shape["mesh_verts"], dtype=np.float32) * scale
        faces = np.asarray(shape["mesh_faces"], dtype=np.int32)
        mesh = bpy.data.meshes.new(name=f"{name}_mesh")
        mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
        mesh.update()
        obj = bpy.data.objects.new(name=name, object_data=mesh)
        bpy.context.collection.objects.link(obj)
    elif gt == GEO_PLANE:
        sx = float(scale[0]) if scale[0] > 0 else 50.0
        sy = float(scale[1]) if scale[1] > 0 else 50.0
        bpy.ops.mesh.primitive_plane_add(size=2.0)
        obj = bpy.context.active_object
        obj.scale = (sx, sy, 1.0)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    else:
        raise ValueError(f"Unsupported geo_type {gt} for shape {name}")

    obj.name = name
    return obj


def warp_xform_to_blender(xf7: np.ndarray) -> tuple[Vector, Quaternion]:
    """Newton transform [tx, ty, tz, qx, qy, qz, qw] → Blender (location, Quaternion(w, x, y, z))."""
    return (
        Vector((float(xf7[0]), float(xf7[1]), float(xf7[2]))),
        Quaternion((float(xf7[6]), float(xf7[3]), float(xf7[4]), float(xf7[5]))),
    )


def build_robot(
    shapes: list[dict],
    namespace: str,
    material_for_shape,
    static_collection: bpy.types.Collection,
    body_collection: bpy.types.Collection,
    include_static: bool = True,
) -> dict[int, bpy.types.Object]:
    """Create per-body Empties parented to the world; attach each shape mesh.

    `material_for_shape(body_idx) -> bpy.types.Material` assigns a material per shape.
    When ``include_static`` is False, body_idx == -1 shapes are skipped entirely
    (so the terrain isn't rebuilt on a second pass for the ghost robot).
    Returns a dict mapping body_idx -> body Empty (only for non-static bodies).
    """
    body_empties: dict[int, bpy.types.Object] = {}
    for i, shape in enumerate(shapes):
        body_idx = int(shape["body_idx"])
        if body_idx == -1 and not include_static:
            continue
        local_xf = np.asarray(shape["local_xform"], dtype=np.float32)
        loc, quat = warp_xform_to_blender(local_xf)

        obj_name = f"{namespace}_shape_{i}"
        obj = make_shape_object(shape, obj_name)
        obj.data.materials.clear()
        obj.data.materials.append(material_for_shape(body_idx))

        if body_idx == -1:
            obj.location = loc
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = quat
            static_collection.objects.link(obj)
            try:
                bpy.context.collection.objects.unlink(obj)
            except RuntimeError:
                pass
            continue

        if body_idx not in body_empties:
            empty = bpy.data.objects.new(f"{namespace}_body_{body_idx}", None)
            empty.empty_display_type = "ARROWS"
            empty.empty_display_size = 0.1
            body_collection.objects.link(empty)
            body_empties[body_idx] = empty

        body = body_empties[body_idx]
        obj.parent = body
        obj.location = loc
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = quat
        body_collection.objects.link(obj)
        try:
            bpy.context.collection.objects.unlink(obj)
        except RuntimeError:
            pass

    return body_empties


def keyframe_bodies(
    body_empties: dict[int, bpy.types.Object],
    poses: np.ndarray,  # [T_total, num_bodies, 7]
    frame_offset: int = 0,
):
    T_total = poses.shape[0]
    for body_idx, empty in body_empties.items():
        empty.rotation_mode = "QUATERNION"
        for t in range(T_total):
            xf = poses[t, body_idx]
            loc, quat = warp_xform_to_blender(xf)
            empty.location = loc
            empty.rotation_quaternion = quat
            f = frame_offset + t
            empty.keyframe_insert("location", frame=f)
            empty.keyframe_insert("rotation_quaternion", frame=f)


def clear_default_objects():
    """Remove Blender's startup Cube (and other default scene objects) if present."""
    for name in ("Cube",):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)


def setup_camera(
    scene: bpy.types.Scene,
    points: np.ndarray,
    lens: float = 50.0,
    azimuth_deg: float = 35.0,
    elevation_deg: float = 40.0,
    zoom: float = 1.0,
    distance: float | None = None,
    aim: tuple[float, float, float] | None = None,
    fstop: float = 0.0,
):
    """Static elevated 3/4 camera framing the bounding box of ``points`` ([N, 3]).

    The camera distance is sized so the padded bbox projects within the lens'
    horizontal *and* vertical FOV (axis-projection, not bounding-sphere — flat
    XY-heavy scenes won't be over-zoomed). ``zoom`` is a multiplicative knob:
    pass 0.7 to push closer, 1.5 to pull back.

    Pass ``distance`` and/or ``aim`` to override the auto-fit completely (e.g.
    to lock in a camera position you tuned in the GUI). When ``distance`` is
    given, ``zoom`` and the FOV math are bypassed entirely.
    """
    for obj in list(scene.objects):
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)

    cam_data = bpy.data.cameras.new("camera")
    cam_data.lens = float(lens)
    cam_obj = bpy.data.objects.new("camera", cam_data)

    # `points` are chassis centers only — pad in XY for body width and trail
    # spread, in +Z for the floating labels (ghost label sits at z=1.5) and
    # the robots' own height above the chassis frame.
    pad_min = np.array([0.5, 0.5, 0.3], dtype=np.float32)
    pad_max = np.array([0.5, 0.5, 2.0], dtype=np.float32)
    pmin = points.min(axis=0) - pad_min
    pmax = points.max(axis=0) + pad_max
    center = np.array(aim, dtype=np.float32) if aim is not None else (pmin + pmax) / 2.0

    aim_obj = bpy.data.objects.new("camera_aim", None)
    aim_obj.empty_display_type = "PLAIN_AXES"
    aim_obj.empty_display_size = 0.1
    aim_obj.location = (float(center[0]), float(center[1]), float(center[2]))
    scene.collection.objects.link(aim_obj)

    azimuth = np.radians(azimuth_deg)
    elevation = np.radians(elevation_deg)
    offset_dir = np.array(
        [
            np.cos(elevation) * np.sin(azimuth),
            -np.cos(elevation) * np.cos(azimuth),
            np.sin(elevation),
        ],
        dtype=np.float32,
    )
    # Camera basis in world coords (Track To with up_axis=UP_Y → world +Z is up).
    view_dir = -offset_dir  # camera-to-aim
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(view_dir, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, view_dir)
    up /= np.linalg.norm(up)

    # Project the 8 padded-bbox corners onto camera right/up axes; the longest
    # half-span on each axis is what the lens has to fit.
    corners = np.array(
        [
            [
                pmin[0] if i & 1 else pmax[0],
                pmin[1] if i & 2 else pmax[1],
                pmin[2] if i & 4 else pmax[2],
            ]
            for i in range(8)
        ],
        dtype=np.float32,
    )
    rel = corners - center
    half_h = float(np.max(np.abs(rel @ right)))
    half_v = float(np.max(np.abs(rel @ up)))

    # Per-axis half-FOV from sensor + lens (sensor fit AUTO).
    sensor_w = float(cam_data.sensor_width)
    res_x = max(1, scene.render.resolution_x)
    res_y = max(1, scene.render.resolution_y)
    if res_x >= res_y:
        sensor_w_eff = sensor_w
        sensor_h_eff = sensor_w * res_y / res_x
    else:
        sensor_h_eff = sensor_w
        sensor_w_eff = sensor_w * res_x / res_y
    half_fov_h = float(np.arctan(0.5 * sensor_w_eff / cam_data.lens))
    half_fov_v = float(np.arctan(0.5 * sensor_h_eff / cam_data.lens))

    if distance is not None and distance >= 0:
        cam_distance = float(distance)
    else:
        d_h = half_h / np.tan(half_fov_h) if half_h > 0 else 0.0
        d_v = half_v / np.tan(half_fov_v) if half_v > 0 else 0.0
        cam_distance = max(max(d_h, d_v) * 1.05 * float(zoom), 4.0)

    offset = offset_dir * cam_distance
    cam_obj.location = (
        float(center[0] + offset[0]),
        float(center[1] + offset[1]),
        float(center[2] + offset[2]),
    )

    track = cam_obj.constraints.new("TRACK_TO")
    track.target = aim_obj
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    if fstop > 0.0:
        cam_data.dof.use_dof = True
        cam_data.dof.focus_object = aim_obj
        cam_data.dof.aperture_fstop = float(fstop)
        # Eevee defaults to a "fast path" that drops below-threshold defocus to
        # save shading; with our distant subject the geometric blur is tiny so
        # the fast path culls it entirely. Force the high-quality path so f/4-ish
        # blurs are still rendered.
        if hasattr(scene, "eevee"):
            for attr in (
                "use_bokeh_high_quality_slight_defocus",
                "use_high_quality_slight_defocus",
            ):
                if hasattr(scene.eevee, attr):
                    setattr(scene.eevee, attr, True)

    dof_str = f"f/{fstop:.1f}" if fstop > 0.0 else "DoF off"
    print(
        f"Camera: lens={lens:.1f} mm, azimuth={azimuth_deg:.2f}°, "
        f"elevation={elevation_deg:.2f}°, distance={cam_distance:.3f} m, "
        f"aim=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}), {dof_str}"
    )


def setup_lighting(scene: bpy.types.Scene):
    """Replace the default sun with a tilted warm key + a cool dark ambient world."""
    # Remove startup lights so we control intensity from scratch.
    for obj in list(scene.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    light_data = bpy.data.lights.new(name="key_sun", type="SUN")
    light_data.energy = 4.0
    light_data.color = (1.0, 0.92, 0.78)  # warm sunlight
    if hasattr(light_data, "angle"):
        light_data.angle = np.radians(3.0)  # softer shadow penumbra
    light_obj = bpy.data.objects.new(name="key_sun", object_data=light_data)
    light_obj.rotation_euler = (np.radians(55), np.radians(20), np.radians(35))
    scene.collection.objects.link(light_obj)

    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node is not None:
        bg_node.inputs["Color"].default_value = (0.04, 0.05, 0.07, 1.0)  # cool dark ambient
        bg_node.inputs["Strength"].default_value = 0.3


def setup_atmospheric_fog(
    scene: bpy.types.Scene,
    density: float,
    color: tuple[float, float, float] = (0.7, 0.75, 0.85),
    start: float = 0.0,
    falloff: str = "QUADRATIC",
):
    """Add depth-based atmospheric haze via the Mist render pass + compositor.

    Implemented entirely in post: each pixel is blended toward ``color`` by
    a ``falloff``-shaped factor of camera-space depth that reaches full
    opacity at distance ``start + 1 / density`` meters. Surface lighting is
    untouched (unlike a world volume scatter, which absorbs the sun before
    it reaches surfaces).
    """
    if density <= 0.0:
        return

    world = scene.world
    if world is None:
        return

    fog_depth = max(1.0 / density, 1.0)
    mist = world.mist_settings
    mist.start = float(start)
    mist.depth = fog_depth
    mist.falloff = str(falloff)
    mist.height = 0.0
    mist.intensity = 0.0

    for vl in scene.view_layers:
        vl.use_pass_mist = True

    # Blender 5.0+ stores the compositor as a NodeGroup datablock assigned to
    # scene.compositing_node_group, with NodeGroupOutput as the result socket
    # (no more CompositorNodeComposite). Older Blender uses an embedded
    # scene.node_tree gated by scene.use_nodes, with CompositorNodeComposite.
    is_node_group = hasattr(scene, "compositing_node_group")
    if is_node_group:
        nt = scene.compositing_node_group
        if nt is None:
            nt = bpy.data.node_groups.new(
                name="HelhestComposite", type="CompositorNodeTree"
            )
            scene.compositing_node_group = nt
    else:
        scene.use_nodes = True
        nt = scene.node_tree
    nt.nodes.clear()

    rl = nt.nodes.new("CompositorNodeRLayers")
    rl.location = (0, 0)

    # AlphaOver with Fac=mist and a constant-RGBA upper image: outputs
    # `lower*(1-fac) + upper*fac`, the same MIX operation MixRGB gave us.
    # The legacy CompositorNodeMixRGB was removed in Blender 5.x; AlphaOver
    # is a primitive node that survives. Look sockets up by socket type so
    # we're robust to layout shifts (Blender 5.x added new float sockets
    # ahead of the upper-image slot, which moves index-based access).
    over = nt.nodes.new("CompositorNodeAlphaOver")
    over.location = (300, 0)
    image_inputs = [s for s in over.inputs if s.bl_idname in ("NodeSocketColor", "NodeSocketRGBA")]
    fac_inputs = [s for s in over.inputs if "Fac" in s.name or "Factor" in s.name]
    if len(image_inputs) < 2 or not fac_inputs:
        raise RuntimeError(
            f"AlphaOver node has unexpected socket layout on this Blender "
            f"version: {[ (s.name, s.bl_idname) for s in over.inputs ]}"
        )
    image_inputs[1].default_value = (*color, 1.0)

    if is_node_group:
        # Define an Image output on the NodeGroup's interface so the renderer
        # can read it; then a NodeGroupOutput inside the group writes to it.
        iface = nt.interface
        has_image_out = any(
            getattr(s, "in_out", None) == "OUTPUT" and s.name == "Image"
            for s in iface.items_tree
        )
        if not has_image_out:
            iface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        out_node = nt.nodes.new("NodeGroupOutput")
    else:
        out_node = nt.nodes.new("CompositorNodeComposite")
    out_node.location = (600, 0)

    nt.links.new(rl.outputs["Image"], image_inputs[0])
    nt.links.new(rl.outputs["Mist"], fac_inputs[0])
    nt.links.new(over.outputs[0], out_node.inputs[0])

    # Make sure the compositor actually runs at render time.
    scene.render.use_compositing = True


def configure_render_output(
    scene: bpy.types.Scene, output_filepath: str, samples: int = 64
) -> str | None:
    """Bake Eevee + video render settings into the scene.

    On Blender 4.x (FFmpeg encoder available) the scene renders straight to an
    MP4 at ``output_filepath``. On Blender 5.x (FFmpeg dropped from core), the
    scene falls back to a PNG image sequence sitting in ``<stem>_frames/`` next
    to ``output_filepath``; the returned string is a ready-to-paste ffmpeg
    command for stitching that sequence into the requested MP4. Returns ``None``
    when no fallback is needed.
    """
    engine_items = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
    available_engines = {e.identifier for e in engine_items}
    if "BLENDER_EEVEE_NEXT" in available_engines:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    elif "BLENDER_EEVEE" in available_engines:
        scene.render.engine = "BLENDER_EEVEE"

    # TAA render samples: Blender stores Eevee/Eevee-Next under scene.eevee on
    # 4.x/5.x. Setting the attribute is a no-op if the engine isn't Eevee.
    if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = max(1, int(samples))

    # The bl_rna enum still advertises "FFMPEG" on Blender 5.x even though the
    # setter rejects it (the encoder was dropped from core), so probe by
    # actually trying to assign it.
    image_settings = scene.render.image_settings
    try:
        image_settings.file_format = "FFMPEG"
        has_ffmpeg = hasattr(scene.render, "ffmpeg")
    except TypeError:
        has_ffmpeg = False

    if has_ffmpeg:
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "HIGH"
        scene.render.ffmpeg.audio_codec = "NONE"
        scene.render.filepath = output_filepath
        return None

    # Blender 5.x: render a PNG sequence and let the user stitch with ffmpeg.
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    if output_filepath.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
        stem, _ = output_filepath.rsplit(".", 1)
    else:
        stem = output_filepath.rstrip("/")
    frames_prefix = f"{stem}_frames/"
    scene.render.filepath = frames_prefix

    fps = max(1, int(scene.render.fps))
    # Blender pads frame numbers to 4 digits by default; %04d matches.
    pattern = frames_prefix.lstrip("/") if frames_prefix.startswith("//") else frames_prefix
    return (
        f"ffmpeg -framerate {fps} -i '{pattern}%04d.png' "
        f"-c:v libx264 -crf 18 -pix_fmt yuv420p '{output_filepath}'"
    )


class _ConstantInterpolation:
    """Context manager: keyframes inserted inside the block default to CONSTANT.

    Works across Blender 4.x / 5.x without touching action.fcurves directly,
    which was removed when Actions gained slots/layers.
    """

    def __enter__(self):
        edit_prefs = bpy.context.preferences.edit
        self._prev = edit_prefs.keyframe_new_interpolation_type
        edit_prefs.keyframe_new_interpolation_type = "CONSTANT"
        return self

    def __exit__(self, exc_type, exc, tb):
        bpy.context.preferences.edit.keyframe_new_interpolation_type = self._prev


def _attach_label_backing(
    text_obj: bpy.types.Object,
    text: str,
    size: float,
    collection: bpy.types.Collection,
    color: tuple[float, float, float] = (0.04, 0.04, 0.06),
    alpha: float = 0.6,
    pad_x: float = 0.35,
    pad_y: float = 0.25,
):
    """Translucent dark plate parented behind ``text_obj`` for legibility.

    Width/height are estimated from character count × glyph metrics, then padded.
    The plate is parented to the text so it inherits the Copy-Location and
    Track-To constraints, and offset along local -Z so it sits just behind the
    glyphs from the camera's perspective.
    """
    width = max(2, len(text)) * size * 0.55 + pad_x * size
    height = size * 1.1 + pad_y * size

    # Plate spans the text's local extents: text is align_x=CENTER, align_y=BOTTOM,
    # so its glyphs live in x ∈ [-width/2, width/2], y ∈ [0, height].
    verts = [
        (-width / 2.0, -pad_y * size * 0.5, 0.0),
        (width / 2.0, -pad_y * size * 0.5, 0.0),
        (width / 2.0, height - pad_y * size * 0.5, 0.0),
        (-width / 2.0, height - pad_y * size * 0.5, 0.0),
    ]
    faces = [(0, 1, 2, 3)]
    mesh = bpy.data.meshes.new(name=f"{text_obj.name}_backing_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    plate = bpy.data.objects.new(name=f"{text_obj.name}_backing", object_data=mesh)
    plate.parent = text_obj
    # Slightly behind the text along the camera-facing axis (text +Z faces camera).
    plate.location = (0.0, 0.0, -0.01)

    mat = make_material(f"{plate.name}_mat", color, alpha)
    plate.data.materials.append(mat)
    collection.objects.link(plate)
    return plate


def make_floating_label(
    name: str,
    follow_body: bpy.types.Object,
    text: str,
    size: float,
    z_offset: float,
    collection: bpy.types.Collection,
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    camera: bpy.types.Object | None = None,
    emission_strength: float = 0.0,
    backing: bool = False,
    roughness: float = 1.0,
) -> bpy.types.Object:
    """Text that follows ``follow_body``'s position with a fixed Z offset.

    The text inherits the body's location via a Copy Location constraint
    (so it doesn't spin with the chassis) and, if a camera is provided, gets
    a Track To constraint that keeps it facing the camera at all angles. The
    material is a flat Principled BSDF (matte, no emission) — contrast comes
    from the chosen color rather than glow.
    """
    curve = bpy.data.curves.new(name=f"{name}_curve", type="FONT")
    curve.body = text
    curve.size = size
    curve.align_x = "CENTER"
    curve.align_y = "BOTTOM"
    obj = bpy.data.objects.new(name=name, object_data=curve)

    mat = make_material(
        f"{name}_mat",
        color,
        1.0,
        roughness=roughness,
        emission_strength=emission_strength,
    )
    obj.data.materials.append(mat)

    obj.location = (0.0, 0.0, z_offset)
    copy_loc = obj.constraints.new("COPY_LOCATION")
    copy_loc.target = follow_body
    copy_loc.use_offset = True

    if camera is not None:
        track = obj.constraints.new("TRACK_TO")
        track.target = camera
        track.track_axis = "TRACK_Z"
        track.up_axis = "UP_Y"

    collection.objects.link(obj)

    if backing:
        _attach_label_backing(obj, text, size, collection)

    return obj


def make_floating_iteration_labels(
    follow_body: bpy.types.Object,
    iter_indices: np.ndarray,
    iter_losses: np.ndarray,
    T: int,
    collection: bpy.types.Collection,
    camera: bpy.types.Object | None = None,
    size: float = 0.4,
    z_offset: float = 0.8,
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    period: int | None = None,
):
    """One floating label per iteration above ``follow_body``; visibility keyframed.

    ``period`` is the per-iteration stride on the timeline (``T`` plus any pause
    between iterations). Defaults to ``T`` when no pause is configured.
    """
    if period is None:
        period = T
    n_iters = len(iter_indices)
    with _ConstantInterpolation():
        for it_idx in range(n_iters):
            it_label = int(iter_indices[it_idx])
            loss = float(iter_losses[it_idx])
            obj = make_floating_label(
                name=f"label_iter_{it_label}",
                follow_body=follow_body,
                text=f"Iter {it_label}  |  Loss {loss:.4f}",
                size=size,
                z_offset=z_offset,
                collection=collection,
                color=color,
                camera=camera,
            )
            start = it_idx * period
            end = (it_idx + 1) * period

            obj.hide_render = True
            obj.hide_viewport = True
            obj.keyframe_insert("hide_render", frame=0)
            obj.keyframe_insert("hide_viewport", frame=0)
            obj.hide_render = False
            obj.hide_viewport = False
            obj.keyframe_insert("hide_render", frame=start)
            obj.keyframe_insert("hide_viewport", frame=start)
            if it_idx < n_iters - 1:
                obj.hide_render = True
                obj.hide_viewport = True
                obj.keyframe_insert("hide_render", frame=end)
                obj.keyframe_insert("hide_viewport", frame=end)


def _resample_equidistant(path: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample a [T, 3] polyline at points ~spacing apart along its arc length.

    Returns (samples, sample_arc_lengths). sample_arc_lengths[i] is the cumulative
    arc length at sample i, which lets callers map each segment back to the
    timestep at which the chassis crossed that point.
    """
    seg = np.diff(path, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cumlen[-1])
    if total < 1e-6:
        return path[:1].astype(np.float32), np.zeros(1, dtype=np.float32)
    n = max(2, int(round(total / spacing)) + 1)
    targets = np.linspace(0.0, total, n)
    out = np.zeros((n, 3), dtype=np.float32)
    for i, s in enumerate(targets):
        idx = int(np.searchsorted(cumlen, s, side="right")) - 1
        idx = max(0, min(idx, len(seg_len) - 1))
        denom = cumlen[idx + 1] - cumlen[idx]
        t = 0.0 if denom < 1e-12 else (s - cumlen[idx]) / denom
        out[i] = path[idx] + t * (path[idx + 1] - path[idx])
    return out, targets.astype(np.float32)


def _spawn_steps(path: np.ndarray, arc_lengths: np.ndarray) -> np.ndarray:
    """Step index in `path` at which the chassis first reaches each arc length."""
    seg_len = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])
    return np.searchsorted(cumlen, arc_lengths, side="left").astype(np.int32)


_BOX_VERTS = np.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ],
    dtype=np.float32,
)
_BOX_FACES = [
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (2, 3, 7, 6),
    (1, 2, 6, 5),
    (0, 4, 7, 3),
]


def _rot_from_x_to(t: np.ndarray) -> np.ndarray:
    """3×3 rotation that maps the +X axis onto unit vector t."""
    n = float(np.linalg.norm(t))
    if n < 1e-9:
        return np.eye(3, dtype=np.float32)
    t = (t / n).astype(np.float32)
    x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    cosA = float(np.dot(x, t))
    if cosA > 1.0 - 1e-6:
        return np.eye(3, dtype=np.float32)
    if cosA < -1.0 + 1e-6:
        return np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
    axis = np.cross(x, t)
    sinA = float(np.linalg.norm(axis))
    axis = axis / sinA
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float32,
    )
    return (np.eye(3, dtype=np.float32) + sinA * K + (1.0 - cosA) * (K @ K)).astype(np.float32)


def _tangents(points: np.ndarray) -> np.ndarray:
    """Per-point unit tangent via central differences (forward/backward at ends)."""
    t = np.empty_like(points)
    if len(points) == 1:
        t[0] = (1.0, 0.0, 0.0)
        return t
    t[0] = points[1] - points[0]
    t[-1] = points[-1] - points[-2]
    if len(points) > 2:
        t[1:-1] = points[2:] - points[:-2]
    norms = np.linalg.norm(t, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return (t / norms).astype(np.float32)


def _build_segment_cluster(
    name: str,
    centers: np.ndarray,
    tangents: np.ndarray,
    length: float,
    width: float,
):
    """One mesh of thin boxes, each oriented along its tangent."""
    n_base = len(_BOX_VERTS)
    scale = np.array([length, width, width], dtype=np.float32)
    all_verts = np.empty((len(centers) * n_base, 3), dtype=np.float32)
    all_faces: list[tuple[int, ...]] = []
    for i, (c, t) in enumerate(zip(centers, tangents)):
        rot = _rot_from_x_to(t)
        all_verts[i * n_base : (i + 1) * n_base] = (_BOX_VERTS * scale) @ rot.T + c
        offset = i * n_base
        for f in _BOX_FACES:
            all_faces.append(tuple(int(idx + offset) for idx in f))
    mesh = bpy.data.meshes.new(name=f"{name}_mesh")
    mesh.from_pydata(all_verts.tolist(), [], all_faces)
    mesh.update()
    return bpy.data.objects.new(name=name, object_data=mesh)


def make_breadcrumb_trails(
    body_pose_iters: np.ndarray,  # [N_iters, T, num_bodies, 7]
    iter_indices: np.ndarray,
    T: int,
    chassis_body_idx: int,
    color: tuple[float, float, float],
    collection: bpy.types.Collection,
    spacing: float = 0.4,
    segment_length: float = 0.18,
    segment_width: float = 0.04,
    decay_rate: float = 0.5,
    min_alpha: float = 0.1,
    period: int | None = None,
):
    """Per-iteration chassis trail rendered as equidistant short segments.

    Each trail spawns at full alpha and decays exponentially each subsequent
    iteration: ``alpha[age] = min_alpha + (1 - min_alpha) * decay_rate ** age``.
    With ``decay_rate=0.4`` the current trail is opaque, the two previous are
    still legible, and older ones fade out toward ``min_alpha``.
    ``period`` is the per-iteration stride on the timeline (``T`` plus any pause
    between iterations). Defaults to ``T`` when no pause is configured.
    """
    if period is None:
        period = T
    n_iters = body_pose_iters.shape[0]
    for it_idx in range(n_iters):
        it_label = int(iter_indices[it_idx])
        path = body_pose_iters[it_idx, :, chassis_body_idx, :3].astype(np.float32)
        sampled, arc_lengths = _resample_equidistant(path, spacing)
        tangents = _tangents(sampled)
        spawn_steps = _spawn_steps(path, arc_lengths)

        # Shared material per iteration so alpha animation is one-and-done.
        mat = make_material(f"breadcrumb_{it_label}", color, 0.99, emission_strength=1.5)
        iter_start = it_idx * period

        # Per-segment objects so each can spawn at its own frame.
        for k in range(len(sampled)):
            seg = _build_segment_cluster(
                f"breadcrumb_iter_{it_label}_seg_{k}",
                sampled[k : k + 1],
                tangents[k : k + 1],
                length=segment_length,
                width=segment_width,
            )
            seg.data.materials.append(mat)
            collection.objects.link(seg)

            spawn_frame = iter_start + int(spawn_steps[k])
            with _ConstantInterpolation():
                seg.hide_render = True
                seg.hide_viewport = True
                seg.keyframe_insert("hide_render", frame=0)
                seg.keyframe_insert("hide_viewport", frame=0)
                seg.hide_render = False
                seg.hide_viewport = False
                seg.keyframe_insert("hide_render", frame=spawn_frame)
                seg.keyframe_insert("hide_viewport", frame=spawn_frame)

        # Keyframe alpha on the shared iteration material.
        # Hold full alpha for the entire birth iteration (drive + pause), then
        # decay one notch per subsequent iteration reached at that iteration's
        # end-of-period (so the trail stops fading during inter-iter pauses).
        alpha_input = mat.node_tree.nodes["Principled BSDF"].inputs["Alpha"]
        alpha_input.default_value = 1.0
        alpha_input.keyframe_insert("default_value", frame=iter_start)
        alpha_input.default_value = 1.0
        alpha_input.keyframe_insert("default_value", frame=iter_start + period)
        for later in range(it_idx + 1, n_iters):
            age = later - it_idx
            alpha = min_alpha + (1.0 - min_alpha) * (decay_rate**age)
            alpha_input.default_value = alpha
            alpha_input.keyframe_insert("default_value", frame=(later + 1) * period)


def make_target_trail(
    target_body_pose: np.ndarray,  # [T, num_bodies, 7]
    chassis_body_idx: int,
    color: tuple[float, float, float],
    collection: bpy.types.Collection,
    spacing: float = 0.4,
    segment_length: float = 0.18,
    segment_width: float = 0.04,
):
    """Single non-fading trail for the target chassis path.

    Each segment spawns at the frame the ghost first crosses its location during
    iteration 0, then stays visible at full opacity for the rest of the timeline
    (the ground truth never fades — it's the constant target).
    """
    path = target_body_pose[:, chassis_body_idx, :3].astype(np.float32)
    sampled, arc_lengths = _resample_equidistant(path, spacing)
    tangents = _tangents(sampled)
    spawn_steps = _spawn_steps(path, arc_lengths)

    mat = make_material("target_trail", color, 1.0, emission_strength=1.5)
    for k in range(len(sampled)):
        seg = _build_segment_cluster(
            f"target_trail_seg_{k}",
            sampled[k : k + 1],
            tangents[k : k + 1],
            length=segment_length,
            width=segment_width,
        )
        seg.data.materials.append(mat)
        collection.objects.link(seg)

        spawn_frame = int(spawn_steps[k])
        with _ConstantInterpolation():
            seg.hide_render = True
            seg.hide_viewport = True
            seg.keyframe_insert("hide_render", frame=0)
            seg.keyframe_insert("hide_viewport", frame=0)
            seg.hide_render = False
            seg.hide_viewport = False
            seg.keyframe_insert("hide_render", frame=spawn_frame)
            seg.keyframe_insert("hide_viewport", frame=spawn_frame)


def add_terrain_wireframe(collection: bpy.types.Collection, thickness: float = 0.008):
    """Overlay a thin dark wireframe on each static mesh for visible topology."""
    wire_mat = make_material("terrain_wires", (0.05, 0.05, 0.05), 1.0)
    for obj in collection.objects:
        if obj.type != "MESH":
            continue
        if wire_mat.name not in [m.name for m in obj.data.materials]:
            obj.data.materials.append(wire_mat)
        mod = obj.modifiers.new(name="Wireframe", type="WIREFRAME")
        mod.thickness = thickness
        mod.use_replace = False
        mod.use_relative_offset = False
        mod.material_offset = len(obj.data.materials) - 1


def main():
    args = parse_args()
    clear_default_objects()
    data = np.load(args.npz, allow_pickle=True)
    fps = float(data["fps"])
    target_body_pose = np.asarray(data["target_body_pose"])  # [T, num_bodies, 7]
    body_pose_iters = np.asarray(data["body_pose_iters"])  # [N_iters, T, num_bodies, 7]
    iter_indices = np.asarray(data["iter_indices"])
    iter_losses = np.asarray(data["iter_losses"])
    shapes = list(data["shapes"])

    n_iters = body_pose_iters.shape[0]
    T = target_body_pose.shape[0]
    pause_frames = max(0, int(round(float(args.pause_seconds) * fps)))
    period = T + pause_frames
    # No need to pause after the last iteration.
    T_total = n_iters * period - pause_frames if n_iters > 0 else 0

    print(
        f"Loaded {args.npz}: {n_iters} iterations × {T} frames "
        f"(+{pause_frames}-frame pause) = {T_total} frames @ {fps:.1f} fps; "
        f"{len(shapes)} shapes."
    )

    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = max(0, T_total - 1)
    scene.render.fps = max(1, int(round(fps)))

    if args.video is not None:
        video_path = args.video
    elif args.output is not None:
        # Sit next to the .blend with a matching stem (Blender resolves `//` to
        # the .blend directory at render time).
        video_path = f"//{args.output.stem}.mp4"
    else:
        video_path = "//render.mp4"
    ffmpeg_cmd = configure_render_output(scene, video_path, samples=int(args.samples))

    setup_lighting(scene)
    setup_atmospheric_fog(
        scene,
        density=float(args.fog_density),
        color=tuple(args.fog_color),
        start=float(args.fog_start),
        falloff=str(args.fog_falloff),
    )

    # Static 3/4 camera auto-framed to the bounding box of all chassis positions
    # (live across iterations + target). Done before label creation so the
    # floating labels' Track To constraint picks this camera up.
    chassis_points = np.concatenate(
        [
            body_pose_iters[:, :, 0, :3].reshape(-1, 3),
            target_body_pose[:, 0, :3],
        ],
        axis=0,
    )
    setup_camera(
        scene,
        chassis_points.astype(np.float32),
        lens=args.lens,
        azimuth_deg=args.azimuth,
        elevation_deg=args.elevation,
        zoom=args.zoom,
        distance=args.distance,
        aim=tuple(args.aim) if args.aim is not None else None,
        fstop=float(args.fstop),
    )

    static_coll = bpy.data.collections.new("static")
    live_coll = bpy.data.collections.new("live")
    ghost_coll = bpy.data.collections.new("ghost")
    for c in (static_coll, live_coll, ghost_coll):
        scene.collection.children.link(c)

    live_materials: dict[int, bpy.types.Material] = {}

    def material_for_live(body_idx: int) -> bpy.types.Material:
        if body_idx not in live_materials:
            color = live_color_for_body(body_idx)
            if body_idx == -1:
                live_materials[body_idx] = make_material("live_static", color, 1.0, roughness=0.95)
            else:
                live_materials[body_idx] = make_material(f"live_body_{body_idx}", color, 1.0)
        return live_materials[body_idx]

    ghost_mat = make_material("ghost", tuple(args.ghost_color), float(args.ghost_alpha))

    live_bodies = build_robot(shapes, "live", material_for_live, static_coll, live_coll)
    ghost_bodies = build_robot(
        shapes, "ghost", lambda _b: ghost_mat, static_coll, ghost_coll, include_static=False
    )
    for empty in live_bodies.values():
        empty.empty_display_size = 0.0
    add_terrain_wireframe(static_coll)

    if args.labels:
        # Floating labels: "Target Trajectory" above the ghost chassis,
        # per-iteration "Iter N | Loss X" above the live chassis.
        label_coll = bpy.data.collections.new("floating_labels")
        scene.collection.children.link(label_coll)
        camera = bpy.context.scene.camera
        if 0 in ghost_bodies:
            make_floating_label(
                name="target_label",
                follow_body=ghost_bodies[0],
                text="Target Trajectory",
                size=0.4,
                z_offset=1.7,
                collection=label_coll,
                color=tuple(args.ghost_color),
                camera=camera,
            )
        if 0 in live_bodies:
            make_floating_iteration_labels(
                follow_body=live_bodies[0],
                iter_indices=iter_indices,
                iter_losses=iter_losses,
                T=T,
                collection=label_coll,
                camera=camera,
                color=LIVE_PALETTE[0],
                period=period,
            )

    # Breadcrumbs: chassis path per iteration, accumulating across the timeline.
    trail_coll = bpy.data.collections.new("breadcrumbs")
    scene.collection.children.link(trail_coll)
    make_breadcrumb_trails(
        body_pose_iters,
        iter_indices,
        T,
        chassis_body_idx=0,
        color=LIVE_PALETTE[0],
        collection=trail_coll,
        period=period,
    )

    # Target trail: ground-truth chassis path, laid down by the ghost during iter 0
    # and kept visible (no fade) thereafter.
    target_trail_coll = bpy.data.collections.new("target_trail")
    scene.collection.children.link(target_trail_coll)
    make_target_trail(
        target_body_pose,
        chassis_body_idx=0,
        color=tuple(args.ghost_color),
        collection=target_trail_coll,
    )

    # Animate: ghost follows the constant target trajectory, live cycles through iterations.
    # Each iteration occupies `period` frames: T drive frames followed by `pause_frames`
    # frames of held end-pose (so the eye gets a beat between iterations).
    for it in range(n_iters):
        offset = it * period
        keyframe_bodies(live_bodies, body_pose_iters[it], frame_offset=offset)
        keyframe_bodies(ghost_bodies, target_body_pose, frame_offset=offset)
        if pause_frames > 0 and it < n_iters - 1:
            # Hold the iteration's last pose at the final pause frame so linear
            # interpolation between drive-end and pause-end stays constant; the
            # next iteration's first keyframe (one frame later) then "snaps"
            # the bodies back to the start pose.
            hold_frame = offset + period - 1
            keyframe_bodies(live_bodies, body_pose_iters[it, -1:], frame_offset=hold_frame)
            keyframe_bodies(ghost_bodies, target_body_pose[-1:], frame_offset=hold_frame)

    if args.output:
        bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))
        print(f"Saved {args.output}")

    if ffmpeg_cmd is None:
        print(
            f"Render with: blender -b {args.output or '<file>.blend'} -a " f"-> writes {video_path}"
        )
    else:
        print(
            "Blender 5.x has no built-in FFmpeg encoder; the scene is configured "
            "to render a PNG sequence instead."
        )
        print(f"  1) blender -b {args.output or '<file>.blend'} -a")
        print(f"  2) {ffmpeg_cmd}")
        print(
            "  (Run step 2 from the directory containing the .blend; the // prefix "
            "in step 1's output resolves to that directory.)"
        )


if __name__ == "__main__":
    main()
