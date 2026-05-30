"""Comprehensive, CORRECT Brax forward-tracking sweep on the box scene.

Fixes the earlier mistake: positional's stable timestep range is 0.005-0.01
(it launches at the dt<=1e-3 the first sweep used). This sweeps each pipeline
over its proper parameters via runtime sys-patching (no per-config recompile
except dt and generalized solver iterations), for both wheel geometries, and
saves every chassis pose [T,7] (xyz + quat xyzw) to npz for scoring/plotting.

Patches mirror examples/comparison_gradient_old/helhest/brax_sweep.py:
  dt -> sys.opt.timestep ;  kv -> actuator gainprm/biasprm ;  mu -> geom_friction
  elasticity -> sys.elasticity ;  baumgarte -> sys.baumgarte_erp
  spring_mass_scale / spring_inertia_scale (spring) ;  iterations -> sys.opt (generalized)

Run (per pipeline, in the Brax venv):
    .venv-brax/bin/python experiments/1_sim_to_real_box/forward_brax_sweep.py --pipeline positional
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
import argparse, json, pathlib, itertools
import numpy as np
import jax, jax.numpy as jnp
import brax.io.mjcf as mjcf

HERE = pathlib.Path(__file__).resolve().parent
RUNS = ["run_2026_05_20-18_04_51", "run_2026_05_20-18_10_33"]
DURATION = 8.0

WHEEL_GEOM = {
    "sphere":  'type="sphere" size="0.35"',
    "capsule": 'type="capsule" fromto="0 -0.05 0 0 0.05 0" size="0.35"',
}

XML = """<mujoco model="helhest_box_brax">
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <worldbody>
    <geom name="ground" type="plane" size="100 100 0.1" friction="1 0.1 0.01"/>
    <geom name="box" type="box" pos="1.37 0 0.06" size="0.37 0.575 0.06" friction="1 0.1 0.01"/>
    <body name="chassis" pos="0 0 0.36">
      <freejoint name="base_joint"/>
      <inertial mass="89.7" pos="-0.188 0 0" diaginertia="2.41 4.22 6.03"/>
      <geom type="box" pos="-0.13 0 0" size="0.24 0.28 0.10" contype="0" conaffinity="0"/>
      <body name="lw" pos="0 0.365 0"><joint name="lj" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.173 0.337 0.173"/>
        <geom {wheel} friction="1 0.1 0.01"/></body>
      <body name="rw" pos="0 -0.365 0"><joint name="rj" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.173 0.337 0.173"/>
        <geom {wheel} friction="1 0.1 0.01"/></body>
      <body name="rew" pos="-0.75 0 0"><joint name="rrj" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.173 0.337 0.173"/>
        <geom {wheel} friction="1 0.1 0.01"/></body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="la" joint="lj" kv="100"/><velocity name="ra" joint="rj" kv="100"/>
    <velocity name="rea" joint="rrj" kv="100"/>
  </actuator>
</mujoco>"""

# Per-pipeline parameter grids. dt ranges follow Brax's stable bands.
GRIDS = {
    "positional": dict(
        wheels=["sphere", "capsule"], dt=[0.002, 0.005, 0.01],
        kv=[50., 100., 200.], mu=[0.5, 1.0, 1.5], baumgarte=[0.1, 0.2],
        smass=[None], sinertia=[None], iters=[None]),
    "spring": dict(
        wheels=["sphere", "capsule"], dt=[0.002, 0.005, 0.01],
        kv=[50., 100., 200.], mu=[0.5, 1.0, 1.5], baumgarte=[0.1, 0.2],
        smass=[0.0, 1.0], sinertia=[0.0, 1.0], iters=[None]),
    # generalized+capsule is impractically slow on a 4 GB GPU (QP solver); sphere
    # only here. spring/positional cover capsule, and the overall best gates.
    "generalized": dict(
        wheels=["sphere"], dt=[0.005, 0.01],
        kv=[50., 100., 200.], mu=[0.5, 1.0, 1.5], baumgarte=[0.1],
        smass=[None], sinertia=[None], iters=[8, 16]),
}


def get_pipe(name):
    if name == "positional": import brax.positional.pipeline as p
    elif name == "spring": import brax.spring.pipeline as p
    else: import brax.generalized.pipeline as p
    return p


def patch(sys, kv, mu, baumgarte, smass, sinertia, iters):
    ch = {}
    ch["actuator_gainprm"] = sys.actuator_gainprm.at[:, 0].set(jnp.float32(kv))
    ch["actuator_biasprm"] = sys.actuator_biasprm.at[:, 2].set(jnp.float32(-kv))
    ch["geom_friction"] = sys.geom_friction.at[:, 0].set(
        jnp.full_like(sys.geom_friction[:, 0], jnp.float32(mu)))
    ch["elasticity"] = jnp.zeros_like(sys.elasticity)  # no restitution
    if hasattr(sys, "baumgarte_erp") and baumgarte is not None:
        ch["baumgarte_erp"] = jnp.float32(baumgarte)
    if smass is not None and hasattr(sys, "spring_mass_scale"):
        ch["spring_mass_scale"] = jnp.float32(smass)
    if sinertia is not None and hasattr(sys, "spring_inertia_scale"):
        ch["spring_inertia_scale"] = jnp.float32(sinertia)
    if iters is not None:
        ch["opt"] = sys.opt.replace(iterations=int(iters))
    return sys.replace(**ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", choices=list(GRIDS), required=True)
    args = ap.parse_args()
    g = GRIDS[args.pipeline]
    pipe = get_pipe(args.pipeline)
    (HERE / "results").mkdir(exist_ok=True)

    # preload control per (run, dt)
    gtctrl = {}
    for run in RUNS:
        gt = json.load(open(HERE / "data" / f"{run}.json"))
        gtctrl[run] = (np.asarray(gt["control"]["t"]), np.asarray(gt["control"]["lrr"]))

    out, n_ok, n_div = {}, 0, 0
    for wheel in g["wheels"]:
        base = mjcf.loads(XML.format(wheel=WHEEL_GEOM[wheel]))
        for dt in g["dt"]:
            T = int(DURATION / dt)
            # compile rollout once per (wheel, dt) [iters recompiles inside]
            for iters in g["iters"]:
                def make_roll(sysm):
                    def roll(c):
                        s = pipe.init(sysm, q0, qd0)
                        def step(s, u):
                            s = pipe.step(sysm, s, u)
                            r = s.x.rot[0]
                            return s, jnp.concatenate([s.x.pos[0], r[1:4], r[0:1]])
                        _, pose = jax.lax.scan(step, s, c)
                        return pose
                    return jax.jit(roll)
                q0 = jnp.zeros(base.q_size()).at[2].set(0.36).at[3].set(1.0)
                qd0 = jnp.zeros(base.qd_size())
                for kv, mu, baum, sm, si in itertools.product(
                        g["kv"], g["mu"], g["baumgarte"], g["smass"], g["sinertia"]):
                    sysm = patch(base.replace(opt=base.opt.replace(
                        timestep=jnp.float32(dt))), kv, mu, baum, sm, si, iters)
                    roll = make_roll(sysm)
                    for run in RUNS:
                        ct, lrr = gtctrl[run]
                        tg = np.arange(T) * dt
                        ctrl = np.stack([np.interp(tg, ct, lrr[:, i]) for i in range(3)],
                                        -1).astype(np.float32)
                        try:
                            pose = np.asarray(roll(jnp.asarray(ctrl)))
                        except Exception as e:
                            continue
                        if not np.all(np.isfinite(pose)) or np.max(np.abs(pose[:, :3])) > 10:
                            n_div += 1
                            continue
                        key = (f"{run}|{wheel}|dt{dt}|kv{kv}|mu{mu}|baum{baum}|"
                               f"sm{sm}|si{si}|it{iters}")
                        out[key] = pose
                        n_ok += 1
            print(f"[{args.pipeline}] wheel={wheel} dt={dt}: bounded so far {n_ok}, diverged {n_div}",
                  flush=True)
        # save incrementally after each wheel so a timeout preserves progress
        np.savez(HERE / "results" / f"forward_brax_{args.pipeline}.npz", **out)
        print(f"[{args.pipeline}] checkpoint after wheel={wheel}: {n_ok} bounded saved", flush=True)
    np.savez(HERE / "results" / f"forward_brax_{args.pipeline}.npz", **out)
    print(f"\n[{args.pipeline}] saved {n_ok} bounded poses ({n_div} diverged) "
          f"-> results/forward_brax_{args.pipeline}.npz")


if __name__ == "__main__":
    main()
