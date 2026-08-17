"""Gallery: current best identified config vs real on all 14 runs (top-down).

Best = round-1 Stribeck winner: k_p=4000, mu_lat=0.4 (aniso lateral),
mu_long (0.8, 1.2), mu_stiction_scale=4.0, v_stribeck=0.3, cmd_scale 0.937.

    .venv/bin/python experiments/1_sim_to_real_box/plot_best14.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import eval_campaign2 as ec
from common_box import DATA_DIR, RESULTS_DIR, load_gt
import examples.helhest_junior.replay_real as rr

BEST = dict(k_p=4000.0, mu_front=0.4, mu_rear=0.4,
            mu_long_front=0.8, mu_long_rear=1.2,
            mu_stiction_scale=4.0, v_stribeck=0.3)

_orig_init = rr.HelhestJuniorReplaySimulator.__init__


def _patched_init(self, *a, **kw):
    kw.update(BEST)
    _orig_init(self, *a, **kw)


rr.HelhestJuniorReplaySimulator.__init__ = _patched_init


def _run_mj_c3(gt):
    """MuJoCo at the campaign-3-identified config (rear 0.6, tor 0.05)."""
    import mujoco
    import numpy as _np
    from sweep_mujoco import BASE_PARAMS, JUNIOR_BOX_XML, _patch_wheel_geom
    p = ec.C1_MUJOCO
    dt, dur = p["dt"], float(gt["duration_s"])
    box = gt["box"]
    params = {**BASE_PARAMS, "dt": dt, "kv": p["kv"],
              "ground_friction": 0.3, "box_friction": 0.3,
              "front_friction": 1.2, "rear_friction": 0.3,
              "ground_torsional": 0.05, "front_torsional": 0.05,
              "rear_torsional": 0.05,
              "solref0": p["solref0"], "condim": p["condim"],
              "integrator": p["integrator"],
              "box_x": ec.sim_box_center(gt)[0],
              "box_y": ec.sim_box_center(gt)[1],
              "box_z": box["center"][2],
              "box_hx": box["half_extents"][0],
              "box_hy": box["half_extents"][1],
              "box_hz": box["half_extents"][2]}
    xml = JUNIOR_BOX_XML.format(**params)
    yaw_deg = _np.degrees(box.get("yaw", 0.0))
    xml = xml.replace('<geom name="box" type="box"',
                      f'<geom name="box" type="box" euler="0 0 {yaw_deg:.3f}"')
    xml = _patch_wheel_geom(xml, p["wheel_geom"])
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    T = int(dur / dt)
    pose = _np.zeros((T, 7), dtype=_np.float32)
    ts_t, cmd = gt["control"]["t"], gt["control"]["lrr"]
    for step in range(T):
        t = (step + 1) * dt
        for c in range(3):
            data.ctrl[c] = 0.9448 * _np.interp(t, ts_t, cmd[:, c])
        mujoco.mj_step(model, data)
        q = data.qpos
        pose[step, 0:3] = q[0:3]
        pose[step, 3:6] = q[4:7]
        pose[step, 6] = q[3]
    return pose, dt


def main():
    runs = [f"ostrich{i}" for i in range(14)]
    fig, axes = plt.subplots(7, 2, figsize=(13, 26))
    errs = []
    for ax, run in zip(axes.flat, runs):
        gt = load_gt(DATA_DIR / f"{run}.json")
        s = ec._score_run(*ec.run_ostrich(gt, 0.937), gt)
        errs.append(s["combined_with_yaw"])
        rx, ry = gt["real"]["x"], gt["real"]["y"]

        box = gt["box"]
        cx, cy = box["center"][:2]
        hx, hy = box["half_extents"][:2]
        yb = box.get("yaw", 0.0)
        c, sn = np.cos(yb), np.sin(yb)
        R = np.array([[c, -sn], [sn, c]])
        cc = (np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy],
                        [-hx, -hy]]) @ R.T) + [cx, cy]
        ax.fill(cc[:, 0], cc[:, 1], color="#B25E1C", alpha=0.2)
        ax.plot(cc[:, 0], cc[:, 1], "-", c="#B25E1C", lw=1.2)

        ax.plot(rx, ry, "-", c="k", lw=1.8, label="real")
        sim = s["sim_rel"]
        ax.plot(sim[:, 0], sim[:, 1], "-", c="#C43131", lw=1.4,
                label="Ostrich (Stribeck)")
        sm = ec._score_run(*_run_mj_c3(gt), gt)
        mj = sm["sim_rel"]
        ax.plot(mj[:, 0], mj[:, 1], "-", c="#3562D6", lw=1.2, alpha=0.85,
                label="MuJoCo")
        ax.plot(rx[0], ry[0], "o", c="k", ms=5)
        ax.set_aspect("equal")
        ax.set_xlim(min(np.min(rx), cc[:, 0].min()) - 0.4,
                    max(np.max(rx), cc[:, 0].max()) + 0.4)
        ax.set_ylim(min(np.min(ry), cc[:, 1].min()) - 0.4,
                    max(np.max(ry), cc[:, 1].max()) + 0.4)
        train = run in ("ostrich0", "ostrich1", "ostrich10", "ostrich13")
        ax.set_title(f"{run}{' (train)' if train else ''}: "
                     f"O {s['combined_with_yaw']:.2f} m | "
                     f"M {sm['combined_with_yaw']:.2f} m", fontsize=10)
        if run == "ostrich0":
            ax.legend(fontsize=8)
        print(f"{run}: {s['combined_with_yaw']:.3f} m", flush=True)

    fig.suptitle(f"Ostrich (Stribeck, identified) vs MuJoCo (identified) "
                 f"vs real - all 14 runs. Ostrich mean {np.mean(errs):.3f} m",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = RESULTS_DIR / "best14_gallery.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
