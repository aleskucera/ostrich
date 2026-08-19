"""Fit the wheel-actuator response from the in-air step bag (steps_air).

The bag holds the robot with wheels off the ground while the setpoint steps through
±{0.3, 0.6, 1, 2, 3, 4} rad/s on all three wheels, so the measured /joint_states
response is the motor + low-level controller alone — no contact. Every engine in the
comparison drives wheels with a velocity servo, so this answers two questions:

  1. tracking gain  — does commanded speed equal realized speed at steady state?
  2. lag            — first-order time constant tau of the realized speed.

Engines then either model the lag (a sweep row) or, if tau is small relative to the
replay dt, justifiably ignore it. Run with the helhest_stack venv (needs rosbags):

    ~/projects/helhest_stack/.venv/bin/python actuator_id.py
"""

import json
import pathlib

import numpy as np
from rosbags.highlevel import AnyReader

BAGS_DIR = pathlib.Path.home() / "projects" / "helhest_stack" / "bags"
DATA_DIR = pathlib.Path(__file__).parent / "data"
SIM_ORDER = ["left_wheel_j", "right_wheel_j", "rear_wheel_j"]
WHEELS = ["left", "right", "rear"]


def load(bag: str = "steps_air"):
    joint_cols = None
    t_sp, v_sp, t_js, v_js = [], [], [], []
    with AnyReader([BAGS_DIR / bag]) as reader:
        conns = [c for c in reader.connections
                 if c.topic in ("/joint_setpoints", "/joint_states")]
        for conn, ts, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if joint_cols is None:
                names = list(msg.name)
                joint_cols = [names.index(n) for n in SIM_ORDER]
            s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            t = s if s > 0 else ts * 1e-9
            vel = np.asarray(msg.velocity, dtype=np.float64)[joint_cols]
            if conn.topic == "/joint_setpoints":
                t_sp.append(t), v_sp.append(vel)
            else:
                t_js.append(t), v_js.append(vel)
    t_sp, t_js = np.asarray(t_sp), np.asarray(t_js)
    v_sp, v_js = np.stack(v_sp), np.stack(v_js)
    o = np.argsort(t_sp, kind="stable")
    t_sp, v_sp = t_sp[o], v_sp[o]
    o = np.argsort(t_js, kind="stable")
    t_js, v_js = t_js[o], v_js[o]
    t0 = min(t_sp[0], t_js[0])
    return t_sp - t0, v_sp, t_js - t0, v_js


def fit_steps(t_sp, v_sp, t_js, v_js):
    """Per-wheel step detection + 63%-rise tau + steady-state gain."""
    results = {w: {"tau": [], "gain": []} for w in WHEELS}
    for c, w in enumerate(WHEELS):
        sp = v_sp[:, c]
        steps = np.flatnonzero(np.abs(np.diff(sp)) > 0.15)
        # Merge step indices closer than 0.5 s (ramped transitions report many).
        keep = [steps[0]] if len(steps) else []
        for i in steps[1:]:
            if t_sp[i] - t_sp[keep[-1]] > 0.5:
                keep.append(i)
        for i in keep:
            ts0 = t_sp[i]
            v_from, v_to = sp[i], sp[i + 1]
            # Hold window = until the next command change (cap 6 s).
            nxt = t_sp[i + 1:][np.abs(np.diff(sp[i:])) > 0.15]
            t_hold_end = min(nxt[1] if len(nxt) > 1 else ts0 + 6.0, ts0 + 6.0)
            m = (t_js >= ts0) & (t_js <= t_hold_end)
            if m.sum() < 20 or (t_hold_end - ts0) < 1.0:
                continue
            tw, vw = t_js[m], v_js[m, c]
            # Steady state = last 30% of the hold.
            ss = vw[tw > ts0 + 0.7 * (t_hold_end - ts0)].mean()
            if abs(v_to) > 0.05:
                results[w]["gain"].append(ss / v_to)
            # 63% rise time toward the new target.
            target = v_from + 0.632 * (v_to - v_from)
            crossed = np.flatnonzero(
                (vw - target) * np.sign(v_to - v_from) >= 0)
            if len(crossed):
                results[w]["tau"].append(tw[crossed[0]] - ts0)
    return results


def main() -> None:
    t_sp, v_sp, t_js, v_js = load()
    res = fit_steps(t_sp, v_sp, t_js, v_js)
    out = {}
    print(f"{'wheel':>6} | {'n steps':>7} | {'tau median':>10} | {'gain median':>11}")
    for w in WHEELS:
        tau = np.asarray(res[w]["tau"])
        gain = np.asarray(res[w]["gain"])
        out[w] = {
            "n_steps": int(len(tau)),
            "tau_median_s": float(np.median(tau)),
            "tau_iqr_s": [float(np.percentile(tau, 25)), float(np.percentile(tau, 75))],
            "gain_median": float(np.median(gain)),
            "gain_iqr": [float(np.percentile(gain, 25)), float(np.percentile(gain, 75))],
        }
        print(f"{w:>6} | {len(tau):>7} | {np.median(tau):>9.3f}s | {np.median(gain):>11.3f}")
    DATA_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / "actuator_id.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {DATA_DIR / 'actuator_id.json'}")


if __name__ == "__main__":
    main()
