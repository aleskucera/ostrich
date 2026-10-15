# GL viewer crashes: "CUDA error 719" is a driver fault, not physics

## The symptom

A GL example runs fine for up to a minute and then dies with a CUDA error --
usually 719 -- at a readback. It looks like the solver diverged. It is not: the
GPU context was already dead several launches earlier.

The kernel log has the real cause:

```
NVRM: Xid 13, Graphics Exception: SKEDCHECK22_INVALIDATE_ACTIVE_QMD failed
```

Check with:

```
journalctl -k | grep SKEDCHECK22
```

Headless runs never hit this. Only the GL viewer does.

## The cause

On a hybrid machine (integrated GPU + discrete NVIDIA), setting

```
__GLX_VENDOR_LIBRARY_NAME=nvidia
```

pins OpenGL to the **discrete** GPU -- the same one warp is launching CUDA
graphs on. GL rendering and CUDA compute then contend for it, and the NVIDIA
scheduler eventually fails to invalidate an active compute QMD and tears down
the context. The application only observes it at the next readback, by which
point the error is attributed to whatever kernel happened to run last.

That variable is a common desktop default; on this project's dev machine it came
from `~/.config/hypr/hyprland.conf`.

## The fix

`RenderingConfig.create_viewer` clears `__GLX_VENDOR_LIBRARY_NAME` when it is
set to `nvidia` and a GL viewer is being created, before any GL context exists.
OpenGL then renders on the integrated GPU and the discrete GPU is left to
compute. It prints a line saying so.

CUDA is unaffected: it never goes through libglvnd, and `ViewerGL` needs no
GL/CUDA interop, so nothing is lost by moving GL off the discrete card.

On a machine whose only GPUs are NVIDIA, clearing the pin is *worse* than the
contention: libglvnd falls back to Mesa, which without a second GPU means
llvmpipe -- software rendering. The viewer detects this and keeps the pin,
warning instead. `OSTRICH_ALLOW_NVIDIA_GLX=1` forces that behaviour anywhere.

The detection asks DRM whether a **non-NVIDIA render node** exists
(`/sys/class/drm/renderD*/device/vendor != 0x10de`), not whether a non-NVIDIA
libglvnd vendor file is installed. The latter was the first version of this
check and it was wrong: Mesa is installed nearly everywhere, so it reported
"second GPU available" on a 2x RTX 3090 box and dropped GL to llvmpipe, taking
a sim from real time to 0.09x. Render nodes exist only for GPUs with a 3D
engine, so display-only hardware -- a server's ASPEED BMC, say -- is correctly
ignored.

Machines that never set the variable, which is most of them, see none of this:
the check is a no-op and the viewer starts exactly as before.

## What the fix costs, and how bad the crash really is

Moving GL off the discrete card is not free. Measured on
`examples/helhest_junior/control.py` (84 viewer objects, dt=30ms), one whole
frame:

| GL on | frame | fps |
|---|---|---|
| Intel Arc iGPU (the safe default) | 58 ms | 17 |
| NVIDIA RTX A500 (`OSTRICH_ALLOW_NVIDIA_GLX=1`) | 12 ms | 82 |

Physics is not involved: the solver is 4.5 ms/step, 6.7x faster than real time,
and the window is just as slow with the solver never launched. Nor is it vsync
or the compositor -- `glFinish()` before the buffer swap accounts for 57 of the
58 ms, so it is real GPU work. `glxgears` gets 60 fps on the same iGPU.

### The iGPU cost is a fixed floor, so do not go tuning the scene

Hiding viewer objects one group at a time:

| visible objects | triangles | frame | fps |
|---|---|---|---|
| 84 (all) | 444 638 | 55.7 ms | 18.0 |
| 40 | 470 | 51.6 ms | 19.4 |
| 10 | 110 | 46.1 ms | 21.7 |
| **0** | **0** | **45.9 ms** | **21.8** |

An empty screen still costs 46 ms. The entire scene -- including a 444k-triangle
mesh -- is worth only ~10 ms on top. What is left is per-frame fixed work that
runs whatever the scene contains: a 4096x4096 shadow FBO bound and cleared every
frame, the sky pass, the 4x MSAA resolve, the blit to screen, the imgui panel.

That is why the quality knobs disappoint: shadow map 4096 -> 1024 buys 7 ms,
MSAA off buys 1.5 ms, together 17 -> 20.6 fps. Nothing reachable from
`RenderingConfig` closes a 5x gap, and neither does simplifying the world. The
only lever that matters is which GPU draws. (Note that
`renderer.draw_shadows = False` **crashes**: `_light_space_matrix` is assigned
inside `_render_shadow_map` but read unconditionally by `_render_scene`. Shrink
the map instead.)

Window size is also not a lever here: under a tiling WM the requested
`viewer_width`/`viewer_height` is ignored outright -- asking for 1280x720 got a
941x568 slot.

### How likely is the crash, really?

Less certain than the top of this document implies. Both Xid 13 events on this
machine came from sessions with a second heavy CUDA consumer (the elevation
node) on the same card. With `control.py` alone pinned to the discrete GPU:

```
SOAK SURVIVED 300.0s, 25514 steps, mean 11.8 ms/frame (85.1 fps)
```

-- five minutes, a state readback every single step (the operation that surfaces
a dead context), zero Xid. That is not proof the fault is gone; it is evidence
that the pin is a *risk*, not a certainty, and that it scales with how much other
CUDA work shares the GPU. For a single interactive example on a laptop, trying
the discrete GPU is reasonable:

```
OSTRICH_ALLOW_NVIDIA_GLX=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
    python examples/helhest_junior/control.py
```

If it does die, you lose the run and see `CUDA error 719`; check
`journalctl -k | grep SKEDCHECK22` to confirm it was the driver. Do not do this
alongside a training job or a second CUDA process -- that is the configuration
that actually broke.

### If you stay on the iGPU: silent slow motion

`target_fps` sets how much sim time one drawn frame covers
(`steps_per_segment = round(1/fps / dt)`). At the shipped `target_fps: 30` with
`dt = 30 ms` that is one step per frame, so the renderer has 30 ms and takes 58
-- and `_pace_real_time` resyncs rather than sleeping, so the window does not
merely stutter, it plays **slow**:

| `rendering.target_fps` | steps/frame | speed | drawn |
|---|---|---|---|
| 30 (default) | 1 | **0.54x** | 18 fps |
| 15 | 2 | 0.98x | 16 fps |
| 10 | 3 | 1.00x | 11 fps |

So ask for fewer frames and get correct-speed playback:

```
python examples/helhest_junior/control.py rendering.target_fps=15
```

Leave the default at 30 for machines whose GPU can serve it.

## Related: pinned host memory without a CUDA device

A second, independent GL viewer crash. `newton/_src/viewer/viewer_gl.py`
allocates page-locked host memory:

```python
self._packed_vbo_xforms_host = wp.empty(total, dtype=wp.mat44, device="cpu", pinned=True)
```

Pinned memory requires a CUDA context, so on a CPU-only machine this fails. The
guard is in `newton_local_changes.patch`:

```python
pinned = wp.get_cuda_device_count() > 0
```

`third_party/newton` is a submodule pointing at upstream
`newton-physics/newton`, so local edits there are **not** carried by a push of
this repo -- only the commit pointer is. Apply them after checkout with:

```
scripts/apply_newton_patch.sh
```

The durable fix is to fork newton, commit both viewer changes there, and point
`.gitmodules` at the fork; the pinned-memory guard is a genuine portability bug
worth upstreaming too.
