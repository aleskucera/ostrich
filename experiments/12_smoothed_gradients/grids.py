"""Dense grids of L(u) and dL/du (adjoint) on [0, 1]^d for the benchmark.

The gradient benchmark (bench_gradients.py) reads all its samples from these
grids (nearest node), so estimator repeats cost nothing, and computes the
true smoothed gradient from them by quadrature. The corner drop's grid comes
from corner_drop_study.py (results/corner_drop.npz).

Run one slice:  PYTHONPATH=. python grids.py run NAME PART NPARTS
  (writes results/grids/NAME_PART.npz)
Then:           PYTHONPATH=. python grids.py merge NAME NPARTS
  (writes results/grid_NAME.npz: axis, u, L, g, and per node the number of
  steps whose Newton solve did not converge, unconverged)
"""

import importlib
import sys
import time

import numpy as np
import torch

import ball_landscape as bl

SPECS = {
    "wheel": {"module": "wheel_ledge", "d": 1, "n": 4001},
    "box": {"module": "box_tip", "d": 2, "n": 201},
}
PARTS = bl.RESULTS / "grids"


def nodes(spec):
    """Grid nodes (n^d, d), the last coordinate fastest."""
    axis = np.linspace(0.0, 1.0, spec["n"])
    mesh = np.meshgrid(*[axis] * spec["d"], indexing="ij")
    return axis, np.stack([m.ravel() for m in mesh], axis=1)


def run(name, part, nparts):
    spec = SPECS[name]
    mod = importlib.import_module(spec["module"])
    _, u = nodes(spec)
    idx = np.array_split(np.arange(len(u)), nparts)[part]
    sim = mod.make_sim()
    step, flags = sim.step, []

    def counting_step(*args, **kwargs):
        out = step(*args, **kwargs)
        flags.append(out[2]["residual_world"] > sim.cfg.tol)
        return out

    sim.step = counting_step
    t0 = time.time()
    L, g = mod.loss_and_grad(sim, torch.tensor(u[idx]), chunk=512)
    # loss_and_grad runs the chunks one after another, T_STEPS steps each
    unconverged = torch.cat([torch.stack(flags[i : i + mod.T_STEPS]).sum(dim=0)
                             for i in range(0, len(flags), mod.T_STEPS)])
    PARTS.mkdir(exist_ok=True)
    np.savez(PARTS / f"{name}_{part}.npz", idx=idx, L=L.numpy(), g=g.numpy(), unconverged=unconverged.numpy())
    print(f"{name} part {part}/{nparts}: {len(idx)} nodes, {time.time() - t0:.0f} s", flush=True)


def merge(name, nparts):
    spec = SPECS[name]
    axis, u = nodes(spec)
    L, g = np.full(len(u), np.nan), np.full((len(u), spec["d"]), np.nan)
    unconverged = np.zeros(len(u), dtype=int)
    for part in range(nparts):
        r = np.load(PARTS / f"{name}_{part}.npz")
        L[r["idx"]], g[r["idx"]], unconverged[r["idx"]] = r["L"], r["g"], r["unconverged"]
    assert not np.isnan(L).any(), "missing parts"
    np.savez(bl.RESULTS / f"grid_{name}.npz", axis=axis, u=u, L=L, g=g, unconverged=unconverged)
    print(f"wrote {bl.RESULTS / f'grid_{name}.npz'}: {len(u)} nodes, {np.mean(unconverged > 0):.2%} with an "
          f"unconverged step ({unconverged.sum()} steps in all)")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        merge(sys.argv[2], int(sys.argv[3]))
