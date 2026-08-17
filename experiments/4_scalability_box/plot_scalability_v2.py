"""Revised scalability figure: throughput as the headline, memory as support.

Panel (a): optimization throughput (world-iterations/s) vs #worlds, every
engine at its best memory configuration (MJX with jax.checkpoint per step).
Panel (b): peak GPU memory (NVML) vs #worlds. OOM points marked.

    .venv/bin/python experiments/4_scalability_box/plot_scalability_v2.py
"""
import collections
import glob
import json
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
RES = HERE / "results"

SERIES = {
    "ostrich": ("Ostrich", "#3562D6", "-", "o"),
    "ostrich_ckpt": ("Ostrich + checkpoint", "#3562D6", ":", "v"),
    "mjx_ckpt_step": ("MJX + checkpoint", "#C43131", "-", "s"),
    "mjx_ckpt_none": ("MJX plain BPTT", "#C43131", ":", "^"),
    "semi_implicit": ("Semi-Implicit", "#3B8C4E", "-", "D"),
}
# last world count that FAILED with OOM (annotation target)
OOM = {"mjx_ckpt_none": 16, "semi_implicit": 1024, "ostrich": 16384}


def load():
    data = collections.defaultdict(dict)
    for f in glob.glob(str(RES / "*.json")):
        m = re.match(r".*/([a-z_]+?)_(\d+)\.json", f)
        if not m:
            continue
        name, w = m.group(1), int(m.group(2))
        d = json.load(open(f))
        data[name][w] = d
    return data


def main():
    data = load()
    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(11, 4.2))

    for key, (label, color, ls, mk) in SERIES.items():
        if key not in data:
            continue
        ws = sorted(data[key])
        thr = [w / (data[key][w]["median_time_ms"] / 1000.0) for w in ws]
        ax_t.plot(ws, thr, ls, c=color, marker=mk, ms=4, lw=1.6, label=label)
        # memory: prefer NVML series file, else NVML field, else tracked
        mem_key = {"ostrich": "ostrich_nvml", "semi_implicit": "si_nvml"}.get(key, key)
        src = data.get(mem_key, data[key])
        wm = sorted(src)
        mem = [src[w].get("peak_gpu_mb_nvml") or src[w].get("peak_gpu_mb")
               for w in wm]
        ax_m.plot(wm, mem, ls, c=color, marker=mk, ms=4, lw=1.6, label=label)

        if key in OOM:
            for ax, ys in ((ax_t, thr), (ax_m, mem)):
                ax.plot([OOM[key]], [ys[-1] if ax is ax_t else ys[-1]], "x",
                        c=color, ms=9, mew=2)
            ax_t.annotate("OOM", (OOM[key], thr[-1]), textcoords="offset points",
                          xytext=(4, -11), fontsize=8, color=color)

    for ax in (ax_t, ax_m):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("parallel worlds")
        ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax_t.set_ylabel("throughput [world-iterations / s]")
    ax_t.set_title("(a) optimization throughput (fwd+bwd), best memory config",
                   fontsize=10)
    ax_m.set_ylabel("peak GPU memory [MB, NVML]")
    ax_m.set_title("(b) memory: checkpointing removes the ceiling",
                   fontsize=10)
    ax_t.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    out = RES / "scalability_v2.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
