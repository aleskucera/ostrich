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

## Desktop-wide controls (this project's dev machine)

Independent of ostrich, three helpers manage the same pin for everything else:

| command | effect |
|---|---|
| `glx-pin on \| off \| status` | session default, by writing a Hyprland `env` line. Affects apps started afterwards; turning it off fully applies at next login. `status` also lists which processes are on the dGPU. |
| `nvidia-gl <cmd>` | run one app on the discrete GPU. Do not use while a CUDA job is running. |
| `igpu-only <cmd>` | force one app onto the integrated GPU |

`igpu-only` exists because clearing the variable is **not always enough**:
Chromium/Electron apps (Spotify) pick the discrete GPU themselves and ignore
the pin. It also restricts glvnd's EGL to the Mesa ICD, removing the NVIDIA GPU
from the set they can choose at all. Worth knowing if a long physics run dies
while such an app is open.

Note that ostrich's own guard overrides `glx-pin on` for its GL viewer: it
treats the pin as unsafe whenever it is about to run CUDA alongside GL.

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
