"""Forward-tracking test: does Brax track the real box trajectory?

Drives Brax (each candidate pipeline/wheel/params) with the recorded wheel
setpoints over the two clean GT runs and saves the chassis pose trajectory
[T,7] (xyz + quat xyzw) to an npz. score_brax.py then scores these with the
exact same metric as the MuJoCo/Ostrich/Semi sweeps (common_box.score), so the
combined position+yaw error is directly comparable to the paper's
0.062 / 0.054 / 0.110 m.

Run in the Brax venv:
    .venv-brax/bin/python experiments/1_sim_to_real_box/forward_brax.py
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
import json, pathlib
import numpy as np
import jax, jax.numpy as jnp
import brax.io.mjcf as mjcf

HERE = pathlib.Path(__file__).resolve().parent
RUNS = ["run_2026_05_20-18_04_51", "run_2026_05_20-18_10_33"]

WHEEL_GEOM = {
    "sphere":  'type="sphere" size="0.35"',
    "capsule": 'type="capsule" fromto="0 -0.05 0 0 0.05 0" size="0.35"',
}

XML = """<mujoco model="helhest_box_brax">
  <option gravity="0 0 -9.81" timestep="{dt}"/>
  <worldbody>
    <geom name="ground" type="plane" size="100 100 0.1" friction="{mu} 0.1 0.01"/>
    <geom name="box" type="box" pos="1.37 0 0.06" size="0.37 0.575 0.06" friction="{mu} 0.1 0.01"/>
    <body name="chassis" pos="0 0 0.35">
      <freejoint name="base_joint"/>
      <inertial mass="89.7" pos="-0.188 0 0" diaginertia="2.41 4.22 6.03"/>
      <geom type="box" pos="-0.13 0 0" size="0.24 0.28 0.10" contype="0" conaffinity="0"/>
      <body name="lw" pos="0 0.365 0"><joint name="lj" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.173 0.337 0.173"/>
        <geom {wheel} friction="{mu} 0.1 0.01"/></body>
      <body name="rw" pos="0 -0.365 0"><joint name="rj" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.173 0.337 0.173"/>
        <geom {wheel} friction="{mu} 0.1 0.01"/></body>
      <body name="rew" pos="-0.75 0 0"><joint name="rrj" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.173 0.337 0.173"/>
        <geom {wheel} friction="{mu} 0.1 0.01"/></body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="la" joint="lj" kv="{kv}"/><velocity name="ra" joint="rj" kv="{kv}"/>
    <velocity name="rea" joint="rrj" kv="{kv}"/>
  </actuator>
</mujoco>"""

# candidate forward configs (give Brax its best shot)
CONFIGS = [
    ("positional", "sphere"),
    ("spring", "capsule"),
]
KV_GRID = [50.0, 150.0]
MU_GRID = [0.5, 1.0]
DT_GRID = [1e-3, 5e-4, 2e-4]   # positional is unstable above ~1e-3; sweep down to be fair
DURATION = 8.0


def get_pipe(name):
    if name == "positional": import brax.positional.pipeline as p
    elif name == "spring": import brax.spring.pipeline as p
    else: import brax.generalized.pipeline as p
    return p


def main():
    out = {}
    for run in RUNS:
        gt = json.load(open(HERE / "data" / f"{run}.json"))
        ct = np.asarray(gt["control"]["t"]); lrr = np.asarray(gt["control"]["lrr"])
        for dt in DT_GRID:
            T = int(DURATION / dt)
            tg = np.arange(T) * dt
            ctrl = np.stack([np.interp(tg, ct, lrr[:, i]) for i in range(3)], -1).astype(np.float32)
            for (pipe_name, wheel) in CONFIGS:
                pipe = get_pipe(pipe_name)
                for kv in KV_GRID:
                    for mu in MU_GRID:
                        xml = XML.format(dt=dt, mu=mu, kv=kv, wheel=WHEEL_GEOM[wheel])
                        try:
                            sysm = mjcf.loads(xml)
                        except Exception as e:
                            print(f"{run} {pipe_name}/{wheel} kv={kv} mu={mu} dt={dt}: LOAD FAIL {str(e)[:50]}")
                            continue
                        q0 = jnp.zeros(sysm.q_size()).at[2].set(0.35).at[3].set(1.0)
                        qd0 = jnp.zeros(sysm.qd_size())
                        def roll(c):
                            s = pipe.init(sysm, q0, qd0)
                            def step(s, u):
                                s = pipe.step(sysm, s, u)
                                # chassis pose: pos[0] (xyz) + rot[0] (wxyz) -> xyzw
                                r = s.x.rot[0]
                                return s, jnp.concatenate([s.x.pos[0], r[1:4], r[0:1]])
                            _, pose = jax.lax.scan(step, s, c)
                            return pose
                        try:
                            pose = np.asarray(jax.jit(roll)(jnp.asarray(ctrl)))
                        except Exception as e:
                            print(f"{run} {pipe_name}/{wheel} kv={kv} mu={mu} dt={dt}: RUN FAIL {str(e)[:50]}")
                            continue
                        finite = bool(np.all(np.isfinite(pose)))
                        key = f"{run}|{pipe_name}|{wheel}|kv{kv}|mu{mu}|dt{dt}"
                        out[key] = pose if finite else None
                        print(f"{key}: finite={finite} "
                              f"x[{np.nanmin(pose[:,0]):.2f},{np.nanmax(pose[:,0]):.2f}] "
                              f"z[{np.nanmin(pose[:,2]):.2f},{np.nanmax(pose[:,2]):.2f}]")
    # save (drop None); dt is encoded in each key
    save = {k: v for k, v in out.items() if v is not None}
    np.savez(HERE / "results" / "forward_brax_poses.npz", **save)
    print(f"\nsaved {len(save)} pose sets -> results/forward_brax_poses.npz")


if __name__ == "__main__":
    (HERE / "results").mkdir(exist_ok=True)
    main()
