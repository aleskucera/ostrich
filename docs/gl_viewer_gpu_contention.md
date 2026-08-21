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

**Single-GPU NVIDIA machines** have no second vendor to fall back to. Set
`OSTRICH_ALLOW_NVIDIA_GLX=1` to keep the pin; the viewer then warns instead of
clearing it, and the contention above applies.

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
