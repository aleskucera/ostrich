"""Records sim2d trajectories for the report's animations (results/animations.json).

Scenes:
  restitution  same drops with the handoff's target e max(0, -d_free) and the
               velocity target d_k - e h v_n^- (e = 0.8)
  tunneling    e = 0 drop from 0.3123 m: start-of-step detection (Ostrich)
               vs detection that covers the free step's gap
  slide        17 kg box kicked to 2 m/s: per-corner normal and friction forces
  roll         ball kicked to 1 m/s, sliding into rolling
  jump         two launches 0.001 m/s apart at 50 deg: bounce just before / after T

Run: PYTHONPATH=. python make_animations.py
"""

import json
import math

import torch

import ball_landscape as bl
import sim2d

R, H = 0.05, 0.01


class HandoffRestitution(sim2d.Sim2D):
    """The handoff's target d* = e max(0, -d_free) on impacts."""

    def gap_target(self, q_prev, v_prev, f_ext):
        _, d_free, v_n = self.free_gap(q_prev, v_prev, f_ext)
        impact = v_n < -self.cfg.restitution_v_thr
        return torch.where(impact, self.cfg.restitution * torch.clamp(-d_free, min=0.0), 0.0)


class StartOfStepDetection(sim2d.Sim2D):
    """Ostrich's rule: a contact is detected only if its start-of-step gap < margin."""

    def free_gap(self, q_prev, v_prev, f_ext):
        d0, _, v_n = super().free_gap(q_prev, v_prev, f_ext)
        return d0, d0, v_n


def run(sim, q, v, steps, record=("q",)):
    frames = {k: [] for k in record}
    frames["q"].append(q.tolist())
    for k in record:
        if k != "q":
            frames[k].append([[0.0] * sim.C for _ in range(len(q))])
    for _ in range(steps):
        q, v, info = sim.step(q, v)
        frames["q"].append(q.tolist())
        for k in record:
            if k != "q":
                frames[k].append(info[k].tolist())
    return frames, q, v


def rounded(x, nd=5):
    if isinstance(x, dict):
        return {k: rounded(y, nd) for k, y in x.items()}
    if isinstance(x, list):
        return [rounded(y, nd) for y in x]
    return round(float(x), nd)


def main():
    scenes = {}

    # Restitution: drop heights that sweep the impact's phase within the step.
    e, h0s, steps = 0.8, [0.300, 0.3075, 0.312, 0.318], 220
    q0 = torch.tensor([[0.0, h, 0.0] for h in h0s])
    v0 = torch.zeros(len(h0s), 3)
    cfg = sim2d.SimConfig(restitution=e, mu=0.0)
    handoff, _, _ = run(HandoffRestitution(sim2d.ball(1.0, R), cfg), q0, v0, steps)
    velocity, _, _ = run(sim2d.Sim2D(sim2d.ball(1.0, R), cfg), q0, v0, steps)
    scenes["restitution"] = {"dt": H, "r": R, "e": e, "h0": h0s,
                             "handoff": handoff["q"], "velocity": velocity["q"]}

    # Tunneling: e = 0, the landing step starts above the margin.
    q0, v0, steps = torch.tensor([[0.0, 0.3123, 0.0]]), torch.zeros(1, 3), 45
    cfg = sim2d.SimConfig(mu=0.0)
    ostrich, _, _ = run(StartOfStepDetection(sim2d.ball(1.0, R), cfg), q0, v0, steps)
    fixed, _, _ = run(sim2d.Sim2D(sim2d.ball(1.0, R), cfg), q0, v0, steps)
    scenes["tunneling"] = {"dt": H, "r": R, "ostrich": ostrich["q"], "fixed": fixed["q"]}

    # Sliding box: settle, then kick to 2 m/s.
    box = sim2d.Sim2D(sim2d.box(17.0, 0.2, 0.05))
    q, v = torch.tensor([[0.0, 0.0505, 0.0]]), torch.zeros(1, 3)
    for _ in range(20):
        q, v, _ = box.step(q, v)
    slide, _, _ = run(box, q, torch.tensor([[2.0, 0.0, 0.0]]), 60, record=("q", "lam_n", "lam_t"))
    scenes["slide"] = {"dt": H, "hx": 0.2, "hy": 0.05, "mu": 0.5, "mg": 17.0 * 9.81, **slide}

    # Ball sliding into rolling.
    ball = sim2d.Sim2D(sim2d.ball(1.0, R))
    q, v = torch.tensor([[0.0, R + 0.0005, 0.0]]), torch.zeros(1, 3)
    for _ in range(10):
        q, v, _ = ball.step(q, v)
    roll, _, _ = run(ball, q, torch.tensor([[1.0, 0.0, 0.0]]), 50)
    scenes["roll"] = {"dt": H, "r": R, "v0": 1.0, **roll}

    # Across a jump: launches 0.001 m/s apart, terminal-state loss at T = 1 s.
    speeds = [2.689, 2.690]
    u = torch.tensor([[(s - bl.SPEED[0]) / (bl.SPEED[1] - bl.SPEED[0]), 0.5] for s in speeds], dtype=torch.float64)
    q, v = bl.initial_state(u)
    sim = bl.make_sim()
    frames = {"q": [q.tolist()], "v": [v.tolist()]}
    for _ in range(bl.T_STEPS):
        q, v, _ = sim.step(q, v)
        frames["q"].append(q.tolist())
        frames["v"].append(v.tolist())
    scenes["jump"] = {"dt": H, "r": R, "speeds": speeds, "angle_deg": 50.0,
                      "loss": bl.terminal_cost(q, v).tolist(),
                      "target_q": bl.TARGET_Q.tolist(), "target_v": bl.TARGET_V.tolist(), **frames}

    with open(bl.RESULTS / "animations.json", "w") as f:
        json.dump(rounded(scenes), f, separators=(",", ":"))
    print("wrote", bl.RESULTS / "animations.json")


if __name__ == "__main__":
    main()
