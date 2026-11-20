"""Is MJX's terminal speed real creep, or chatter invisible at a coarse dt?

Rolls the robot forward with a drive-then-stop wheel profile and measures the
terminal speed the way the success criterion does (single-step finite
difference at the engine's own dt), then re-measures the SAME trajectory over
progressively longer physical windows. If the 2 ms number is much larger than
the 100 ms number, the criterion is resolution-dependent and penalises the
fine-timestep engine.
"""
import sys, pathlib, json
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

# optimize_mjx.py puts the v1 experiment dir on sys.path[0] and imports
# MJX_PARAMS from THAT file, so importing it by name from here would shadow
# v1 with box2 and self-import. Load box2's copy under a distinct name.
import importlib.util
_spec = importlib.util.spec_from_file_location("_omjx_box2", HERE / "optimize_mjx.py")
_omjx = importlib.util.module_from_spec(_spec)
sys.modules["_omjx_box2"] = _omjx
_spec.loader.exec_module(_omjx)

import jax, jax.numpy as jnp
build_mjx_model, make_init_dx, rollout = _omjx.build_mjx_model, _omjx.make_init_dx, _omjx.rollout
print("MJX_PARAMS in use:", _omjx.MJX_PARAMS)

GT = HERE.parent / "1_sim_to_real_box" / "data" / "run_2026_05_20-18_10_33.json"
HORIZON = 6.0

def drive_then_stop(T):
    """Wheel-velocity profile: cruise, decelerate, then commanded zero."""
    u = np.zeros((T, 3), dtype=np.float32)
    t_cruise, t_ramp = int(0.75 * T), int(0.85 * T)
    u[:t_cruise] = 2.0
    n = t_ramp - t_cruise
    u[t_cruise:t_ramp] = 2.0 * (1.0 - np.arange(n, dtype=np.float32)[:, None] / n)
    return u   # zero for the final 15% of the horizon

def windowed_speed(xy, dt, win_s):
    """Speed at the END of the trajectory over a window of win_s seconds."""
    k = max(1, int(round(win_s / dt)))
    if k >= len(xy): return float("nan")
    return float(np.linalg.norm(xy[-1] - xy[-1 - k]) / (k * dt))

def run(dt):
    gt = json.load(open(GT))
    T = int(HORIZON / dt)
    mx, mj_model = build_mjx_model(gt["box"], dt)
    dx0, _ = make_init_dx(mx, mj_model, [0.0, 0.0], 0.0, dt)
    u = jnp.asarray(drive_then_stop(T))
    traj = np.asarray(jax.jit(rollout)(mx, dx0, u))
    xy = traj[:, :2]

    print(f"\n=== MJX at dt = {dt*1000:.0f} ms  ({T} steps) ===")
    print(f"  final xy: ({xy[-1,0]:.3f}, {xy[-1,1]:.3f})   commanded zero for last {0.15*HORIZON:.2f} s")
    print("  terminal speed measured over a window of:")
    for w in (dt, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        if w < dt: continue
        tag = " <- success criterion uses this" if abs(w - dt) < 1e-12 else ""
        print(f"    {w*1000:6.1f} ms : {windowed_speed(xy, dt, w):.4f} m/s{tag}")

    # is the motion oscillatory or monotone?
    step = np.diff(xy[-int(0.5/dt):], axis=0)          # last 0.5 s of per-step displacement
    sgn = np.sign(step[:, 0])
    flips = int(np.sum(sgn[1:] != sgn[:-1]))
    net = float(np.linalg.norm(xy[-1] - xy[-1 - int(0.5/dt)]))
    path = float(np.sum(np.linalg.norm(step, axis=1)))
    print(f"  last 0.5 s: net displacement {net*1000:.2f} mm, path length {path*1000:.2f} mm")
    print(f"              straightness net/path = {net/max(path,1e-12):.3f}   x-direction reversals: {flips}")
    print("              (net/path ~1 = steady creep;  ~0 with many reversals = chatter)")

for dt in (0.002, 0.005):
    run(dt)
