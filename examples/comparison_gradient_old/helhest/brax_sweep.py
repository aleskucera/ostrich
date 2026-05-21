"""Sweep Brax helhest simulation parameters to find stable configurations.

Compiles the rollout function ONCE per pipeline by fixing T (steps).
Then patches sys fields (dt, kv, spring params, contact params) at runtime —
no recompilation since array shapes don't change, only values.

Actual simulated duration = T * dt (varies per config; shown in output).

Each pipeline has different stability levers:

  POSITIONAL  (penalty-based contacts)
  ──────────────────────────────────────
  dt             Timestep.  Stable range: 0.005–0.01.
                 Too large → sphere wheels sink through ground.
  kv             Motor gain.  Higher → more force → more bounce.
                 Useful range: 50–300.
  elasticity     Contact restitution (0=no bounce, 1=elastic).
                 Higher → more bounce → less stable at large dt.
  baumgarte_erp  Position error correction rate (0–1).
                 Higher → snappier constraint correction → can destabilise.

  SPRING  (spring-damper joints + penalty contacts)
  ──────────────────────────────────────────────────
  dt, kv, elasticity, baumgarte_erp  (same as above)
  spring_mass_scale     Extra mass contribution to spring force (default 0).
                        Positive values increase restoring force.
  spring_inertia_scale  Same for inertia.  Helps damp oscillations.

  GENERALIZED  (QP constraint solver)
  ─────────────────────────────────────
  dt, kv, baumgarte_erp  (same as above)
  solver_iterations   QP iterations (in --xml opt block, requires recompile).
                      Runtime-patchable via sys.opt.replace(iterations=N).

Usage:
    python examples/comparison_gradient/helhest/brax_sweep.py --pipeline positional
    python examples/comparison_gradient/helhest/brax_sweep.py --pipeline spring
    python examples/comparison_gradient/helhest/brax_sweep.py --pipeline generalized
    python examples/comparison_gradient/helhest/brax_sweep.py --pipeline positional --steps 100
    python examples/comparison_gradient/helhest/brax_sweep.py --pipeline positional \\
        --dt 0.05 0.02 0.01 0.005 \\
        --kv 50 100 150 200 \\
        --elasticity 0.0 0.3 \\
        --baumgarte 0.05 0.1 0.2
"""

import argparse
import itertools
import os
import time
import warnings

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import brax.io.mjcf as mjcf

# ─── Model ───────────────────────────────────────────────────────────────────
# Simplified model: single chassis body, sphere wheels (Brax has no cylinder).
# Inertias match mjx.py reference at mass=85/wheel=5.5 kg.
BASE_XML = """<mujoco model="helhest">
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <worldbody>
    <geom name="ground" type="plane" size="100 100 0.1" friction="0.7 0.1 0.01"/>
    <body name="chassis" pos="0 0 0.37">
      <freejoint/>
      <inertial mass="85.0" pos="-0.047 0 0" diaginertia="0.6213 0.1583 0.677"/>
      <geom type="box" pos="-0.047 0 0" size="0.13 0.3 0.09" contype="0" conaffinity="0"/>
      <body name="left_wheel" pos="0 0.36 0">
        <joint name="left_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="sphere" size="0.36" friction="0.7 0.1 0.01"/>
      </body>
      <body name="right_wheel" pos="0 -0.36 0">
        <joint name="right_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="sphere" size="0.36" friction="0.7 0.1 0.01"/>
      </body>
      <body name="rear_wheel" pos="-0.697 0 0">
        <joint name="rear_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="sphere" size="0.36" friction="0.35 0.1 0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="left_vel"  joint="left_wheel_j"  kv="150"/>
    <velocity name="right_vel" joint="right_wheel_j" kv="150"/>
    <velocity name="rear_vel"  joint="rear_wheel_j"  kv="150"/>
  </actuator>
</mujoco>"""

BASE_CTRL = jnp.array([1.0, 1.0, 0.0], dtype=jnp.float32)


# ─── Sys patching ────────────────────────────────────────────────────────────

def patch_sys(sys, dt=None, kv=None, elasticity=None, baumgarte=None,
              spring_mass_scale=None, spring_inertia_scale=None,
              solver_iterations=None):
    """Return a modified sys without recompiling the JIT.

    Only scalar/same-shape patches are applied; array shapes never change,
    so JAX reuses the compiled XLA graph.
    """
    changes = {}

    # dt
    if dt is not None:
        new_opt = sys.opt.replace(timestep=jnp.array(dt, jnp.float32))
        if solver_iterations is not None:
            new_opt = new_opt.replace(iterations=int(solver_iterations))
        changes["opt"] = new_opt
    elif solver_iterations is not None:
        changes["opt"] = sys.opt.replace(iterations=int(solver_iterations))

    # kv: gainprm[:, 0] = kv,  biasprm[:, 2] = -kv
    if kv is not None:
        kv_arr = jnp.array(kv, jnp.float32)
        changes["actuator_gainprm"] = sys.actuator_gainprm.at[:, 0].set(kv_arr)
        changes["actuator_biasprm"] = sys.actuator_biasprm.at[:, 2].set(-kv_arr)

    # contact params
    if elasticity is not None:
        changes["elasticity"] = jnp.full_like(sys.elasticity, elasticity)
    if baumgarte is not None:
        changes["baumgarte_erp"] = jnp.array(baumgarte, jnp.float32)

    # spring pipeline
    if spring_mass_scale is not None:
        changes["spring_mass_scale"] = jnp.array(spring_mass_scale, jnp.float32)
    if spring_inertia_scale is not None:
        changes["spring_inertia_scale"] = jnp.array(spring_inertia_scale, jnp.float32)

    return sys.replace(**changes) if changes else sys


# ─── Rollout ─────────────────────────────────────────────────────────────────

def make_rollout_fn(pipeline, T: int):
    """Build a JIT-able rollout that returns chassis (x, y, z) over T steps."""
    init_fn = pipeline.init
    step_fn = pipeline.step

    def rollout(sys, ctrl):
        q0 = jnp.zeros(sys.q_size()).at[2].set(0.37).at[3].set(1.0)
        qd0 = jnp.zeros(sys.qd_size())
        state = init_fn(sys, q0, qd0)

        def step(state, _):
            state = step_fn(sys, state, ctrl)
            pos = state.x.pos[0]          # chassis position
            return state, pos

        _, positions = jax.lax.scan(step, state, None, length=T)
        return positions   # (T, 3): x, y, z per step

    return jax.jit(rollout)


# Cache of rollout fns keyed by solver_iterations (each is a separate JIT trace)
_rollout_cache: dict = {}


# ─── Stability metrics ────────────────────────────────────────────────────────

def assess(positions):
    """Return dict with stability metrics from (T, 3) positions array."""
    zs = positions[:, 2]
    xs = positions[:, 0]
    z_min = float(jnp.min(zs))
    z_max = float(jnp.max(zs))
    z_final = float(zs[-1])
    x_final = float(xs[-1])
    # OK: chassis stays between 0.1m and 1.0m, never goes underground
    ok = z_min > 0.05 and z_max < 1.5 and abs(z_final - 0.37) < 0.5
    return dict(z_min=z_min, z_max=z_max, z_final=z_final, x_final=x_final, ok=ok)


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_sweep(results, sweep_axes, pipeline_name, T, save):
    """
    results: list of (param_dict, positions_array, metrics_dict)
    sweep_axes: list of param names that define the grid axes (first 2 used for grid)
    """
    import matplotlib.gridspec as gridspec

    # Determine unique values for each axis
    axis_vals = {}
    for name in sweep_axes:
        vals = sorted(set(r[0][name] for r in results))
        axis_vals[name] = vals

    # We'll always show the 2 primary axes (dt × kv) in a grid
    row_key = sweep_axes[0]   # dt
    col_key = sweep_axes[1]   # kv (or second axis)
    row_vals = axis_vals[row_key]
    col_vals = axis_vals[col_key]
    extra_keys = sweep_axes[2:]

    # Group extra configs
    extra_combos = list(itertools.product(*[axis_vals[k] for k in extra_keys])) if extra_keys else [()]

    n_extra = len(extra_combos)
    nrows = len(row_vals)
    ncols = len(col_vals)

    fig = plt.figure(figsize=(max(12, ncols * 3.5), max(8, nrows * 2.5 * n_extra + 2)))
    fig.suptitle(
        f"Brax helhest sweep — pipeline: {pipeline_name}  (T={T} steps, duration=T×dt)",
        fontsize=13, fontweight="bold"
    )

    # One heatmap + grid per extra combo
    outer = gridspec.GridSpec(n_extra, 2, figure=fig, wspace=0.4, hspace=0.6,
                               width_ratios=[1, ncols])

    for ei, extra in enumerate(extra_combos):
        extra_dict = dict(zip(extra_keys, extra))
        extra_label = "  ".join(f"{k}={v}" for k, v in extra_dict.items())

        # Filter results for this extra combo
        subset = [r for r in results if all(r[0][k] == v for k, v in extra_dict.items())]

        # ── heatmap ──
        ax_heat = fig.add_subplot(outer[ei, 0])
        heat_data = np.full((nrows, ncols), np.nan)
        for r in subset:
            ri = row_vals.index(r[0][row_key])
            ci = col_vals.index(r[0][col_key])
            heat_data[ri, ci] = r[2]["z_final"]

        ok_mask = np.zeros((nrows, ncols), dtype=bool)
        for r in subset:
            ri = row_vals.index(r[0][row_key])
            ci = col_vals.index(r[0][col_key])
            ok_mask[ri, ci] = r[2]["ok"]

        cmap = plt.cm.RdYlGn
        im = ax_heat.imshow(heat_data, cmap=cmap, vmin=0.0, vmax=1.5, aspect="auto")
        ax_heat.set_xticks(range(ncols)); ax_heat.set_xticklabels(col_vals, fontsize=8)
        ax_heat.set_yticks(range(nrows)); ax_heat.set_yticklabels(row_vals, fontsize=8)
        ax_heat.set_xlabel(col_key, fontsize=9)
        ax_heat.set_ylabel(row_key, fontsize=9)
        ax_heat.set_title(f"z_final heatmap\n{extra_label}" if extra_label else "z_final heatmap",
                          fontsize=9)
        # Mark OK cells
        for ri in range(nrows):
            for ci in range(ncols):
                v = heat_data[ri, ci]
                mark = "✓" if ok_mask[ri, ci] else "✗"
                color = "white" if v < 0.6 or v > 1.2 else "black"
                ax_heat.text(ci, ri, f"{v:.2f}\n{mark}", ha="center", va="center",
                             fontsize=7, color=color)
        plt.colorbar(im, ax=ax_heat, label="z_final (m)", fraction=0.05)

        # ── z-trace grid ──
        inner = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=outer[ei, 1],
                                                  wspace=0.1, hspace=0.4)
        for ri, rval in enumerate(row_vals):
            for ci, cval in enumerate(col_vals):
                ax = fig.add_subplot(inner[ri, ci])
                match = [r for r in subset
                         if r[0][row_key] == rval and r[0][col_key] == cval]
                if match:
                    pos = match[0][1]
                    met = match[0][2]
                    steps = np.arange(len(pos))
                    color = "#2ecc71" if met["ok"] else "#e74c3c"
                    ax.plot(steps, pos[:, 2], color=color, linewidth=0.8)
                    ax.axhline(0.37, color="gray", linewidth=0.5, linestyle=":")
                    ax.set_ylim(-0.5, 2.0)
                    ax.tick_params(labelsize=5)
                    status = "OK" if met["ok"] else "BAD"
                    ax.set_title(
                        f"dt={rval} kv={cval}\n{status} z∈[{met['z_min']:.2f},{met['z_max']:.2f}]",
                        fontsize=6
                    )
                ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save:
        plt.savefig(save, dpi=120, bbox_inches="tight")
        print(f"Saved: {save}")
    else:
        plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pipeline", choices=["positional", "generalized", "spring"],
                        default="positional")
    parser.add_argument("--steps", type=int, default=50,
                        help="Fixed T (steps per rollout). Duration = T×dt. Default: 50.")
    parser.add_argument("--ctrl", type=float, nargs=3, default=[1.0, 1.0, 0.0],
                        metavar=("L", "R", "REAR"), help="Wheel velocity targets")

    # Sweep axes — positional / generalized
    parser.add_argument("--dt",          type=float, nargs="+", default=[0.05, 0.02, 0.01, 0.005],
                        help="dt values to sweep (default: 0.05 0.02 0.01 0.005)")
    parser.add_argument("--kv",          type=float, nargs="+", default=[50, 100, 150, 200],
                        help="kv (motor gain) values to sweep")
    parser.add_argument("--elasticity",  type=float, nargs="+", default=[0.0],
                        help="Contact elasticity [0=inelastic, 1=elastic] (default: 0.0)")
    parser.add_argument("--baumgarte",   type=float, nargs="+", default=[0.1],
                        help="Baumgarte ERP rate (default: 0.1)")

    # Extra axes — spring pipeline
    parser.add_argument("--spring-mass-scale",    type=float, nargs="+", default=[0.0],
                        help="[spring only] spring_mass_scale values (default: 0.0)")
    parser.add_argument("--spring-inertia-scale", type=float, nargs="+", default=[0.0],
                        help="[spring only] spring_inertia_scale values (default: 0.0)")

    # Extra axes — generalized pipeline
    parser.add_argument("--solver-iter", type=int, nargs="+", default=[100],
                        help="[generalized only] solver iteration counts (default: 100)")

    parser.add_argument("--save", type=str, default=None,
                        help="Save plot to file (e.g. --save sweep.png)")
    args = parser.parse_args()

    # ── load pipeline ──
    if args.pipeline == "positional":
        import brax.positional.pipeline as pipe
    elif args.pipeline == "spring":
        import brax.spring.pipeline as pipe
    else:
        import brax.generalized.pipeline as pipe

    # ── load base sys ──
    print(f"Loading model...", flush=True)
    sys_base = mjcf.loads(BASE_XML)
    ctrl = jnp.array(args.ctrl, dtype=jnp.float32)

    # ── build rollout & JIT ──
    # solver_iterations is a compile-time constant in the JIT trace, so we keep
    # one cached JIT per unique value (retrace only when it changes).
    def get_rollout(solver_iters=None):
        key = solver_iters
        if key not in _rollout_cache:
            base = sys_base
            if solver_iters is not None:
                base = base.replace(opt=base.opt.replace(iterations=int(solver_iters)))
            fn = make_rollout_fn(pipe, args.steps)
            print(f"  JIT compiling (solver_iters={solver_iters})...", end=" ", flush=True)
            t0 = time.time()
            fn(base, ctrl).block_until_ready()
            print(f"done in {time.time()-t0:.1f}s", flush=True)
            _rollout_cache[key] = (fn, base)
        return _rollout_cache[key]

    print(f"JIT warmup ({args.pipeline}, T={args.steps})...")
    if args.pipeline == "generalized":
        for si in args.solver_iter:
            get_rollout(si)
    else:
        get_rollout(None)

    # ── build sweep configs ──
    if args.pipeline == "spring":
        sweep_keys = ["dt", "kv", "elasticity", "baumgarte",
                      "spring_mass_scale", "spring_inertia_scale"]
        sweep_vals = [args.dt, args.kv, args.elasticity, args.baumgarte,
                      args.spring_mass_scale, args.spring_inertia_scale]
    elif args.pipeline == "generalized":
        sweep_keys = ["dt", "kv", "baumgarte", "solver_iterations"]
        sweep_vals = [args.dt, args.kv, args.baumgarte, args.solver_iter]
    else:  # positional
        sweep_keys = ["dt", "kv", "elasticity", "baumgarte"]
        sweep_vals = [args.dt, args.kv, args.elasticity, args.baumgarte]

    combos = list(itertools.product(*sweep_vals))
    n = len(combos)
    print(f"\nSweeping {n} configs...\n")
    print(f"{'dt':>6} {'kv':>6} {'elast':>6} {'baum':>6} "
          + (f"{'sms':>5} {'sis':>5}" if args.pipeline == "spring" else "")
          + f"  {'z_min':>6} {'z_max':>6} {'z_fin':>6} {'x_fin':>7}  status")
    print("─" * 80)

    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(sweep_keys, combo))

        sys = patch_sys(
            sys_base,
            dt=params["dt"],
            kv=params["kv"],
            elasticity=params.get("elasticity"),
            baumgarte=params.get("baumgarte"),
            spring_mass_scale=params.get("spring_mass_scale"),
            spring_inertia_scale=params.get("spring_inertia_scale"),
            # solver_iterations handled via separate JIT trace (compile-time constant)
        )

        t0 = time.time()
        try:
            rollout_fn, _ = get_rollout(params.get("solver_iterations"))
            positions = rollout_fn(sys, ctrl)
            positions.block_until_ready()
            met = assess(positions)
            elapsed = time.time() - t0

            extra = (f" {params.get('spring_mass_scale',0):>5.1f} {params.get('spring_inertia_scale',0):>5.1f}"
                     if args.pipeline == "spring" else "")
            print(
                f"{params['dt']:>6.3f} {params['kv']:>6.0f} "
                f"{params.get('elasticity',0):>6.2f} {params.get('baumgarte',0.1):>6.2f}"
                f"{extra}"
                f"  {met['z_min']:>6.3f} {met['z_max']:>6.3f} {met['z_final']:>6.3f} "
                f"{met['x_final']:>7.3f}  {'✓ OK' if met['ok'] else '✗ BAD'}  ({elapsed:.2f}s)"
            )
            results.append((params, np.array(positions), met))
        except Exception as e:
            print(f"{params}  ERROR: {e}")
            results.append((params, None, dict(z_min=0, z_max=99, z_final=99, x_final=0, ok=False)))

    # ── summary ──
    ok_results = [r for r in results if r[2]["ok"]]
    print(f"\n{len(ok_results)}/{n} configs stable.")
    if ok_results:
        best = max(ok_results, key=lambda r: r[2]["x_final"])
        print(f"Best forward motion: {best[0]}  →  x={best[2]['x_final']:.3f}m")

    # ── plot ──
    valid = [(p, pos, m) for p, pos, m in results if pos is not None]
    if valid:
        # Always use dt and kv as the primary grid axes
        plot_sweep(valid, sweep_keys, args.pipeline, args.steps, args.save)


if __name__ == "__main__":
    main()
