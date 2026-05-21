"""Interactive tuning script for the Brax helhest simulation.

Runs a forward simulation with configurable parameters and plots the results.
Use this to find dt/kv/pipeline combinations that produce realistic trajectories.

Usage examples:
    python examples/comparison_gradient/helhest/brax_tune.py
    python examples/comparison_gradient/helhest/brax_tune.py --dt 0.01 --kv 150
    python examples/comparison_gradient/helhest/brax_tune.py --pipeline generalized --dt 0.005
    python examples/comparison_gradient/helhest/brax_tune.py --ctrl 2.0 2.0 0.5 --duration 2.0
    python examples/comparison_gradient/helhest/brax_tune.py --compare
        (runs positional+generalized side-by-side at current --dt/--kv)

Parameters to tune:
    --dt          Timestep. Positional stable at 0.01; generalized unstable at any dt tested.
    --kv          Motor gain. Original MuJoCo uses 150. Lower = weaker motors.
    --pipeline    positional (penalty contacts) or generalized (QP constraints).
    --duration    Seconds to simulate (JIT time scales with this).
    --ctrl        Three wheel velocities: left right rear (m/s targets).
    --mass        Chassis mass in kg (default 85).
    --wheel-mass  Wheel mass in kg (default 5.5).
"""

import argparse
import os
import time
import warnings

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
warnings.filterwarnings("ignore")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def make_xml(dt: float, kv: float, mass: float, wheel_mass: float) -> str:
    # Simplified single-body chassis (Brax rejects welded sub-bodies).
    # Inertias scale proportionally from the mjx.py reference values (85/5.5 kg).
    # Wheels use spheres — Brax does not support cylinder collision geometry.
    m = mass / 85.0
    w = wheel_mass / 5.5
    r = lambda v: round(v, 5)
    return f"""<mujoco model="helhest">
  <option gravity="0 0 -9.81" timestep="{dt}"/>
  <worldbody>
    <geom name="ground" type="plane" size="100 100 0.1" friction="0.7 0.1 0.01"/>
    <body name="chassis" pos="0 0 0.37">
      <freejoint/>
      <inertial mass="{mass}" pos="-0.047 0 0"
                diaginertia="{r(0.6213*m)} {r(0.1583*m)} {r(0.677*m)}"/>
      <geom type="box" pos="-0.047 0 0" size="0.13 0.3 0.09" contype="0" conaffinity="0"/>
      <body name="left_wheel" pos="0 0.36 0">
        <joint name="left_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="{r(5.5*w)}" pos="0 0 0"
                  diaginertia="{r(0.20045*w)} {r(0.20045*w)} {r(0.3888*w)}"/>
        <geom type="sphere" size="0.36" friction="0.7 0.1 0.01"/>
      </body>
      <body name="right_wheel" pos="0 -0.36 0">
        <joint name="right_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="{r(5.5*w)}" pos="0 0 0"
                  diaginertia="{r(0.20045*w)} {r(0.20045*w)} {r(0.3888*w)}"/>
        <geom type="sphere" size="0.36" friction="0.7 0.1 0.01"/>
      </body>
      <body name="rear_wheel" pos="-0.697 0 0">
        <joint name="rear_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="{r(5.5*w)}" pos="0 0 0"
                  diaginertia="{r(0.20045*w)} {r(0.20045*w)} {r(0.3888*w)}"/>
        <geom type="sphere" size="0.36" friction="0.35 0.1 0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="left_vel"  joint="left_wheel_j"  kv="{kv}"/>
    <velocity name="right_vel" joint="right_wheel_j" kv="{kv}"/>
    <velocity name="rear_vel"  joint="rear_wheel_j"  kv="{kv}"/>
  </actuator>
</mujoco>"""


def run_sim(xml: str, pipeline_name: str, ctrl_vals: list, duration: float, dt: float):
    """Run a forward simulation. Returns (times, xs, ys, zs) arrays."""
    import brax.io.mjcf as mjcf

    if pipeline_name == "generalized":
        import brax.generalized.pipeline as pipe
    elif pipeline_name == "spring":
        import brax.spring.pipeline as pipe
    else:
        import brax.positional.pipeline as pipe

    sys = mjcf.loads(xml)
    T = int(duration / dt)
    ctrl = jnp.array(ctrl_vals, dtype=jnp.float32)

    q0 = jnp.zeros(sys.q_size()).at[2].set(0.37).at[3].set(1.0)
    qd0 = jnp.zeros(sys.qd_size())

    print(f"  JIT compiling + running {T} steps ({duration:.1f}s) ...", flush=True)
    t0 = time.time()
    state = pipe.init(sys, q0, qd0)

    times = [0.0]
    xs = [float(state.x.pos[0, 0])]
    ys = [float(state.x.pos[0, 1])]
    zs = [float(state.x.pos[0, 2])]

    for i in range(T):
        state = pipe.step(sys, state, ctrl)
        times.append((i + 1) * dt)
        xs.append(float(state.x.pos[0, 0]))
        ys.append(float(state.x.pos[0, 1]))
        zs.append(float(state.x.pos[0, 2]))
        if abs(zs[-1]) > 20:
            print(f"  Exploded at step {i+1} (z={zs[-1]:.1f}), stopping.")
            break

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  |  final: x={xs[-1]:.3f}, y={ys[-1]:.3f}, z={zs[-1]:.3f}")
    return np.array(times), np.array(xs), np.array(ys), np.array(zs)


def plot(configs: list, save: str | None = None):
    """configs: list of (label, times, xs, ys, zs)"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax_z, ax_xy, ax_xyt = axes

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (label, times, xs, ys, zs) in enumerate(configs):
        c = colors[i % len(colors)]
        ok = 0.2 < zs[-1] < 0.6 and min(zs) > 0.0
        status = "OK" if ok else "BAD"
        full_label = f"{label} [{status}]"

        # z height over time
        ax_z.plot(times, zs, label=full_label, color=c)

        # xy top-down trajectory
        ax_xy.plot(xs, ys, label=full_label, color=c)
        ax_xy.plot(xs[0], ys[0], "o", color=c, ms=6)
        ax_xy.plot(xs[-1], ys[-1], "s", color=c, ms=6)

        # x and y vs time
        ax_xyt.plot(times, xs, label=f"{label} x", color=c, linestyle="-")
        ax_xyt.plot(times, ys, label=f"{label} y", color=c, linestyle="--")

    ax_z.axhline(0.37, color="gray", linestyle=":", linewidth=1, label="init z=0.37")
    ax_z.axhline(0.0, color="black", linestyle="-", linewidth=0.5)
    ax_z.set_xlabel("time (s)")
    ax_z.set_ylabel("chassis z (m)")
    ax_z.set_title("Z height (should stay ≈ 0.37)")
    ax_z.legend(fontsize=8)
    ax_z.set_ylim(-1.0, max(1.0, max(max(zs) for _, _, _, _, zs in configs) * 1.1))

    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.set_title("XY trajectory (top-down)\n○=start  □=end")
    ax_xy.set_aspect("equal")
    ax_xy.axhline(0, color="gray", linewidth=0.5)
    ax_xy.axvline(0, color="gray", linewidth=0.5)
    ax_xy.legend(fontsize=8)

    ax_xyt.set_xlabel("time (s)")
    ax_xyt.set_ylabel("position (m)")
    ax_xyt.set_title("X (solid) and Y (dashed) vs time")
    ax_xyt.legend(fontsize=8)
    ax_xyt.axhline(0, color="gray", linewidth=0.5)

    fig.suptitle("Brax helhest forward simulation", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=120, bbox_inches="tight")
        print(f"Saved: {save}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dt",       type=float, default=0.01, help="Timestep (default 0.01)")
    parser.add_argument("--kv",       type=float, default=150,  help="Motor kv gain (default 150)")
    parser.add_argument("--pipeline", choices=["positional", "generalized", "spring"],
                        default="positional", help="Brax pipeline (default positional)")
    parser.add_argument("--duration", type=float, default=1.0,  help="Sim duration in seconds (default 1.0)")
    parser.add_argument("--ctrl",     type=float, nargs=3, default=[1.0, 1.0, 0.0],
                        metavar=("LEFT", "RIGHT", "REAR"),
                        help="Wheel velocity targets (default 1.0 1.0 0.0)")
    parser.add_argument("--mass",       type=float, default=85.0, help="Chassis mass kg (default 85)")
    parser.add_argument("--wheel-mass", type=float, default=5.5,  help="Wheel mass kg (default 5.5)")
    parser.add_argument("--compare",  action="store_true",
                        help="Compare positional vs generalized at the given dt/kv")
    parser.add_argument("--save",     type=str, default=None,
                        help="Save plot to file instead of showing (e.g. --save out.png)")
    args = parser.parse_args()

    xml = make_xml(args.dt, args.kv, args.mass, args.wheel_mass)

    configs = []

    if args.compare:
        for pipeline in ["positional", "generalized"]:
            label = f"{pipeline} dt={args.dt} kv={args.kv}"
            print(f"\n[{pipeline}]")
            try:
                t, x, y, z = run_sim(xml, pipeline, args.ctrl, args.duration, args.dt)
                configs.append((label, t, x, y, z))
            except Exception as e:
                print(f"  FAILED: {e}")
    else:
        label = f"{args.pipeline} dt={args.dt} kv={args.kv}"
        print(f"\n[{args.pipeline}]")
        t, x, y, z = run_sim(xml, args.pipeline, args.ctrl, args.duration, args.dt)
        configs.append((label, t, x, y, z))

    if configs:
        plot(configs, save=args.save)


if __name__ == "__main__":
    main()
