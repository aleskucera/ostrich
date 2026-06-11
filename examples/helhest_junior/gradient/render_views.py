"""Headless render of the junior gradient figure's baked view cameras.

Designed to run on a remote Linux box with Blender 5.1.x + an NVIDIA GPU.
The scene renders with Eevee Next, which uses the GPU automatically when run
headless (working NVIDIA drivers required) — no Cycles device setup needed.

The .blend has the view cameras baked in (objects named ``view_*``); this
script just loops over them. Self-contained: only needs bpy + the .blend.

Usage:
    blender -b junior_gradient_figure.blend --python render_views.py

    # higher quality / different output dir / resolution via env vars:
    OUTDIR=~/junior_out SAMPLES=512 RES_X=3840 RES_Y=1620 \
        blender -b junior_gradient_figure.blend --python render_views.py

    # render only one view:
    ONLY=view_hero blender -b junior_gradient_figure.blend --python render_views.py
"""
import os
import time

import bpy

OUTDIR = os.environ.get("OUTDIR", os.path.expanduser("~/junior_views"))
SAMPLES = int(os.environ.get("SAMPLES", "256"))
RES_X = int(os.environ.get("RES_X", "2560"))
RES_Y = int(os.environ.get("RES_Y", "1080"))
ONLY = os.environ.get("ONLY", "").strip()
# Optional goal-patch recolor: a named preset or "r,g,b" floats. Empty = leave as saved.
PATCH_COLOR = os.environ.get("PATCH_COLOR", "").strip()
PATCH_PRESETS = {
    "gold":    (1.00, 0.80, 0.18),
    "green":   (0.20, 0.95, 0.30),
    "cyan":    (0.15, 0.90, 0.95),
    "magenta": (1.00, 0.15, 0.70),
}

os.makedirs(OUTDIR, exist_ok=True)

scene = bpy.context.scene
scene.render.resolution_x = RES_X
scene.render.resolution_y = RES_Y
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
if hasattr(scene.eevee, "taa_render_samples"):
    scene.eevee.taa_render_samples = SAMPLES


def _apply_patch_color(spec: str) -> str:
    """Recolor the goal terrain patch (Mix B + emission) from a preset or r,g,b."""
    rgb = PATCH_PRESETS.get(spec)
    if rgb is None:
        try:
            rgb = tuple(float(x) for x in spec.split(","))[:3]
        except ValueError:
            print(f"  ! bad PATCH_COLOR '{spec}', ignoring"); return ""
    mat = bpy.data.materials.get("jgf_terrain")
    if not mat or not mat.use_nodes:
        return ""
    nt = mat.node_tree
    mix = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeMix"), None)
    bsdf = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    if mix:
        mix.inputs[7].default_value = (*rgb, 1.0)
    if bsdf and "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = (*rgb, 1.0)
    print(f"  patch color -> {spec} {tuple(round(c,2) for c in rgb)}")
    return spec if spec in PATCH_PRESETS else "custom"


PATCH_TAG = _apply_patch_color(PATCH_COLOR) if PATCH_COLOR else ""

# If you ever switch the scene to Cycles, enable OptiX/CUDA GPU here.
# (No-op / harmless under Eevee Next, which is already GPU-rendered.)
try:
    cprefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            cprefs.compute_device_type = backend
            cprefs.get_devices()
            for d in cprefs.devices:
                d.use = d.type != "CPU"
            print(f"Cycles GPU backend set to {backend}")
            break
        except Exception:
            continue
    if hasattr(scene, "cycles"):
        scene.cycles.device = "GPU"
except Exception as exc:  # cycles addon not configured — fine for Eevee
    print(f"(Cycles GPU setup skipped: {exc})")

cams = sorted(
    (o for o in bpy.data.objects if o.type == "CAMERA" and o.name.startswith("view_")),
    key=lambda o: o.name,
)
if ONLY:
    cams = [o for o in cams if o.name == ONLY]
if not cams and scene.camera:
    cams = [scene.camera]

print(f"Engine={scene.render.engine}  {RES_X}x{RES_Y}  samples={SAMPLES}")
print(f"Rendering {len(cams)} camera(s) -> {OUTDIR}")
suffix = f"_{PATCH_TAG}" if PATCH_TAG else ""
for cam in cams:
    scene.camera = cam
    out = os.path.join(OUTDIR, f"{cam.name}{suffix}.png")
    scene.render.filepath = out
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"  {cam.name}: {time.time() - t0:.1f}s -> {out}")
print("done")
