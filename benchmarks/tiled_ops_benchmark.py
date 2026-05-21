from tqdm import tqdm
import warp as wp
import numpy as np
import time
from matplotlib import pyplot as plt

import axion.tiled as tu

np.random.seed(0)

device = "cuda:0"

TEST_ITERS = 10
M_N = [
    (1e4, 6e4),  # 1e4, 6e4
    (1e3, 5e3),
]
TS_BT = [
    (256, 128),
    (512, 256),
    (1024, 512),
    (2048, 1024),
    (128, 256),
    (256, 512),
    (512, 1024),
    (256, 256),
    (512, 512),
    (1024, 1024),
]

for i in range(len(M_N)):
    M_N[i] = (int(M_N[i][0]), int(M_N[i][1]))
dot_instances = {
    f"{M}x{N}": {
        f"{ts}x{bt}": tu.TiledDot((M, M), dtype=wp.float32, tile_size=ts, block_threads=bt)
        for ts, bt in TS_BT
    }
    for M, N in M_N
}
sum_instances = {
    f"{M}x{N}": {
        f"{ts}x{bt}": tu.TiledSum((M, M), dtype=wp.float32, tile_size=ts, block_threads=bt)
        for ts, bt in TS_BT
    }
    for M, N in M_N
}
sq_norm_instances = {
    f"{M}x{N}": {
        f"{ts}x{bt}": tu.TiledSqNorm((M, M), dtype=wp.float32, tile_size=ts, block_threads=bt)
        for ts, bt in TS_BT
    }
    for M, N in M_N
}

times_tiled = {
    "dot": {f"{M}x{N}": {f"{ts}x{bt}": [] for ts, bt in TS_BT} for M, N in M_N},
    "sum": {f"{M}x{N}": {f"{ts}x{bt}": [] for ts, bt in TS_BT} for M, N in M_N},
    "sq_norm": {f"{M}x{N}": {f"{ts}x{bt}": [] for ts, bt in TS_BT} for M, N in M_N},
}
times_np = {
    "dot": {f"{M}x{N}": [] for M, N in M_N},
    "sum": {f"{M}x{N}": [] for M, N in M_N},
    "sq_norm": {f"{M}x{N}": [] for M, N in M_N},
}

# Warmup
for _ in range(3):
    for M, N in M_N:
        a_np = np.random.rand(M, M).astype(np.float32) * 2 - 1
        b_np = np.random.rand(M, M).astype(np.float32) * 2 - 1
        a_wp = wp.array(a_np, dtype=wp.float32, device=device)
        b_wp = wp.array(b_np, dtype=wp.float32, device=device)
        out_wp = wp.zeros((M,), dtype=wp.float32, device=device)
        for ts, bt in TS_BT:
            dot_instances[f"{M}x{N}"][f"{ts}x{bt}"].compute(a_wp, b_wp, out_wp)
            sum_instances[f"{M}x{N}"][f"{ts}x{bt}"].compute(a_wp, out_wp)
            sq_norm_instances[f"{M}x{N}"][f"{ts}x{bt}"].compute(a_wp, out_wp)

for i in tqdm(range(TEST_ITERS)):
    for M, N in M_N:
        a_np = np.random.rand(M, M).astype(np.float32) * 2 - 1
        b_np = np.random.rand(M, M).astype(np.float32) * 2 - 1
        a_wp = wp.array(a_np, dtype=wp.float32, device=device)
        b_wp = wp.array(b_np, dtype=wp.float32, device=device)
        out_wp = wp.zeros((M,), dtype=wp.float32, device=device)
        for ts, bt in TS_BT:
            start = time.perf_counter()
            dot_instances[f"{M}x{N}"][f"{ts}x{bt}"].compute(a_wp, b_wp, out_wp)
            wp.synchronize()
            end = time.perf_counter()
            times_tiled["dot"][f"{M}x{N}"][f"{ts}x{bt}"].append((end - start) * 1000)
            start = time.perf_counter()
            out_np = np.sum(a_np * b_np, axis=1)
            end = time.perf_counter()
            times_np["dot"][f"{M}x{N}"].append((end - start) * 1000)
            assert np.allclose(
                out_wp.numpy(), out_np, atol=1e-3
            ), f"Dot product results do not match for {M}x{N}, tile_size={ts}, block_threads={bt}"

            start = time.perf_counter()
            sum_instances[f"{M}x{N}"][f"{ts}x{bt}"].compute(a_wp, out_wp)
            wp.synchronize()
            end = time.perf_counter()
            times_tiled["sum"][f"{M}x{N}"][f"{ts}x{bt}"].append((end - start) * 1000)
            start = time.perf_counter()
            out_np = np.sum(a_np, axis=1)
            end = time.perf_counter()
            times_np["sum"][f"{M}x{N}"].append((end - start) * 1000)
            assert np.allclose(
                out_wp.numpy(), out_np, atol=1e-3
            ), f"Sum results do not match for {M}x{N}, tile_size={ts}, block_threads={bt}"

            start = time.perf_counter()
            sq_norm_instances[f"{M}x{N}"][f"{ts}x{bt}"].compute(a_wp, out_wp)
            wp.synchronize()
            end = time.perf_counter()
            times_tiled["sq_norm"][f"{M}x{N}"][f"{ts}x{bt}"].append((end - start) * 1000)
            start = time.perf_counter()
            out_np = np.sum(a_np * a_np, axis=1)
            end = time.perf_counter()
            times_np["sq_norm"][f"{M}x{N}"].append((end - start) * 1000)
            assert np.allclose(
                out_wp.numpy(), out_np, atol=1e-3
            ), f"SqrNorm results do not match for {M}x{N}, tile_size={ts}, block_threads={bt}"

for op in ("dot", "sum", "sq_norm"):
    fig, axes = plt.subplots(1, len(M_N), figsize=(5 * max(1, len(M_N)), 4))
    if len(M_N) == 1:
        axes = [axes]
    for ax, (M, N) in zip(axes, M_N):
        key = f"{M}x{N}"
        labels = []
        means = []
        stds = []
        for ts, bt in TS_BT:
            tkey = f"{ts}x{bt}"
            data = np.array(times_tiled[op][key][tkey])
            labels.append(tkey)
            if data.size:
                means.append(data.mean())
                stds.append(data.std())
            else:
                means.append(np.nan)
                stds.append(0.0)

        np_data = np.array(times_np[op][key])
        np_mean = np_data.mean() if np_data.size else np.nan
        np_std = np_data.std() if np_data.size else 0.0

        x = np.arange(len(labels) + 1)
        bars_tiled = ax.bar(x[:-1], means, yerr=stds, capsize=4, label="tiled")
        bar_np = ax.bar(x[-1], np_mean, yerr=np_std, capsize=4, color="gray", label="numpy")

        # Add text labels on top of bars
        for b, h in zip(bars_tiled, means):
            if not np.isnan(h):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    h,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        # numpy bar label
        if not np.isnan(np_mean):
            b = bar_np[0]
            ax.text(
                b.get_x() + b.get_width() / 2,
                np_mean,
                f"{np_mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels + ["numpy"], rotation=45, ha="right")
        ax.set_title(f"{op} :: {M}x{N}")
        ax.set_ylabel("Time (ms)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.suptitle(f"{op} timings")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
