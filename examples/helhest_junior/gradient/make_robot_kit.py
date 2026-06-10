"""Build a materials-preserving 'robot kit' .blend for the gradient figure.

OBJ can't carry the asset's shader-node materials (carpaint, chrome, rubber),
so we bake the detailed CAD bodies into a small .blend instead: 4 joined body
meshes (chassis + L/R/rear wheels), each in its body-local frame, each keeping
its materials. The gradient figure script appends these and poses them at the
Newton per-body trajectory poses.

Run headless against the saved sim_to_real_box.blend:
    blender -b path/to/sim_to_real_box.blend \
        --python examples/helhest_junior/gradient/make_robot_kit.py

Writes: examples/helhest_junior/gradient/assets/junior_robot_kit.blend
with objects named: chassis, wheel_left, wheel_right, wheel_rear.
"""
import pathlib

import bpy

OUT = (
    pathlib.Path(__file__).resolve().parent / "assets" / "junior_robot_kit.blend"
)

# (empty in sim_to_real_box.blend, kit object name, also pull root-parented meshes)
BODIES = [
    ("live_body_0", "chassis", True),   # chassis + chain/cover/joint meshes on the root
    ("live_body_1", "wheel_left", False),
    ("live_body_2", "wheel_right", False),
    ("live_body_3", "wheel_rear", False),
]


# Rear-wheel mounting-bracket / swingarm pieces that extend ~0.6 m past the
# wheel center (Wheel_Center.004/.005). They look like stray geometry when the
# wheel is shown in isolation (e.g. a translucent ghost), so drop them.
EXCLUDE_MESHES = {"Wheel_Center.004", "Wheel_Center.005"}


def collect_meshes(empty, include_root):
    meshes = [c for c in empty.children_recursive
              if c.type == "MESH" and not c.hide_render and c.name not in EXCLUDE_MESHES]
    if include_root:
        root = bpy.data.objects["live_body"]
        meshes += [c for c in root.children
                   if c.type == "MESH" and not c.hide_render and c.name not in EXCLUDE_MESHES]
    return meshes


def bake_body(empty_name, out_name, include_root):
    empty = bpy.data.objects[empty_name]
    meshes = collect_meshes(empty, include_root)
    src = [(m, m.matrix_world.copy()) for m in meshes]

    # Evaluate each mesh with MODIFIERS APPLIED. Critical: the tire's shape comes
    # entirely from Geometry-Nodes Array + Mirror + Subsurf modifiers (160-vert
    # base cage otherwise), and rims/centers use Subsurf/Bevel. join() drops
    # non-active modifiers, so we bake them here via the evaluated depsgraph.
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bpy.ops.object.select_all(action="DESELECT")
    dups = []
    for m, mw in src:
        eval_obj = m.evaluated_get(depsgraph)
        baked_mesh = bpy.data.meshes.new_from_object(eval_obj)  # modifiers applied, mats kept
        d = bpy.data.objects.new(f"{m.name}_baked", baked_mesh)
        bpy.context.collection.objects.link(d)
        d.matrix_world = mw
        dups.append(d)
    bpy.context.view_layer.update()

    for d in dups:
        d.select_set(True)
    bpy.context.view_layer.objects.active = dups[0]
    # Bake world transform into vertices, then join (keeps per-face materials).
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.join()
    joined = bpy.context.active_object

    # Express vertices in the body-local (empty) frame.
    joined.matrix_world = empty.matrix_world.inverted()
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    joined.name = out_name
    joined.data.name = f"{out_name}_mesh"
    n_mats = len(joined.data.materials)
    print(f"  baked {out_name}: {len(joined.data.vertices)} verts, {n_mats} materials")
    return joined


def main():
    kit_objs = []
    for empty_name, out_name, include_root in BODIES:
        kit_objs.append(bake_body(empty_name, out_name, include_root))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Write ONLY the 4 kit objects + their dependencies (meshes, materials, node
    # groups) to a fresh .blend. fake_user so they survive with no scene.
    data_blocks = set(kit_objs)
    bpy.data.libraries.write(str(OUT), data_blocks, fake_user=True, path_remap="NONE")
    print(f"Wrote robot kit -> {OUT}  ({len(kit_objs)} bodies)")


if __name__ == "__main__":
    main()
