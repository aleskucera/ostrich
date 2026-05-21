"""Microbenchmark for the forward pass of the small tile-MLP from Warp's
example_tile_mlp.py.

Times only the forward kernel — no tape, no loss, no optimizer. Reports
mean +/- std microseconds per batch over many measured launches, both as
eager launches (one wp.launch per batch) and as a captured CUDA graph
(amortizes launch overhead).

Run:
    python test_scripts/bench_tile_mlp_forward.py
    python test_scripts/bench_tile_mlp_forward.py --batch-size 4096 --hidden 64
    python test_scripts/bench_tile_mlp_forward.py \
        --sweep-batch 256,1024,4096 --sweep-hidden 16,32,64
"""

import argparse
import statistics

import numpy as np
import warp as wp

# ---------------------------------------------------------------------------
# Network shape (matches example_tile_mlp.py defaults)
# ---------------------------------------------------------------------------
NUM_FREQ = wp.constant(8)
DIM_IN = wp.constant(4 * NUM_FREQ)  # sin/cos for x,y at each frequency
DIM_HID = 32
DIM_OUT = 3

NUM_THREADS = 32  # threads per block — required for tile ops
IMG_WIDTH = 512
IMG_HEIGHT = 512

dtype = wp.float16


@wp.func
def relu(x: dtype):
    return wp.max(x, dtype(0.0))


def make_compute_kernel(dim_hid: int):
    """Build a forward-only kernel specialized to dim_hid.

    Tile sizes must be compile-time constants, so we close over them here.
    Batch size is a runtime launch dim and does not require recompilation.
    """

    @wp.kernel
    def compute_fwd(
        indices: wp.array(dtype=int),
        weights_0: wp.array2d(dtype=dtype),
        bias_0: wp.array2d(dtype=dtype),
        weights_1: wp.array2d(dtype=dtype),
        bias_1: wp.array2d(dtype=dtype),
        weights_2: wp.array2d(dtype=dtype),
        bias_2: wp.array2d(dtype=dtype),
        weights_3: wp.array2d(dtype=dtype),
        bias_3: wp.array2d(dtype=dtype),
        out: wp.array2d(dtype=float),
    ):
        linear = indices[wp.tid()]

        row = linear / IMG_WIDTH
        col = linear % IMG_WIDTH

        x = (float(row) / float(IMG_WIDTH) - 0.5) * 2.0
        y = (float(col) / float(IMG_HEIGHT) - 0.5) * 2.0

        local = wp.types.vector(dtype=dtype, length=DIM_IN)
        for s in range(NUM_FREQ):
            scale = wp.pow(2.0, float(s)) * wp.pi
            local[s * 4 + 0] = dtype(wp.sin(x * scale))
            local[s * 4 + 1] = dtype(wp.cos(x * scale))
            local[s * 4 + 2] = dtype(wp.sin(y * scale))
            local[s * 4 + 3] = dtype(wp.cos(y * scale))

        f = wp.tile(local)

        w0 = wp.tile_load(weights_0, shape=(dim_hid, DIM_IN))
        b0 = wp.tile_load(bias_0, shape=(dim_hid, 1))
        z = wp.tile_map(relu, wp.tile_matmul(w0, f) + wp.tile_broadcast(b0, shape=(dim_hid, NUM_THREADS)))

        w1 = wp.tile_load(weights_1, shape=(dim_hid, dim_hid))
        b1 = wp.tile_load(bias_1, shape=(dim_hid, 1))
        z = wp.tile_map(relu, wp.tile_matmul(w1, z) + wp.tile_broadcast(b1, shape=(dim_hid, NUM_THREADS)))

        w2 = wp.tile_load(weights_2, shape=(dim_hid, dim_hid))
        b2 = wp.tile_load(bias_2, shape=(dim_hid, 1))
        z = wp.tile_map(relu, wp.tile_matmul(w2, z) + wp.tile_broadcast(b2, shape=(dim_hid, NUM_THREADS)))

        w3 = wp.tile_load(weights_3, shape=(DIM_OUT, dim_hid))
        b3 = wp.tile_load(bias_3, shape=(DIM_OUT, 1))
        o = wp.tile_map(relu, wp.tile_matmul(w3, z) + wp.tile_broadcast(b3, shape=(DIM_OUT, NUM_THREADS)))

        output = wp.untile(o)
        for i in range(DIM_OUT):
            out[i, linear] = float(output[i])

    return compute_fwd


def make_layer(dim_in: int, dim_hid: int, rng: np.random.Generator):
    scale = 1.0 / np.sqrt(dim_in)
    w = rng.uniform(-scale, scale, (dim_hid, dim_in))
    b = rng.uniform(-scale, scale, (dim_hid, 1))
    return wp.array(w, dtype=dtype), wp.array(b, dtype=dtype)


def time_eager(kernel, batch_size, args, num_iters):
    """Time num_iters eager wp.launch calls. Returns per-launch microseconds."""
    start = wp.Event(enable_timing=True)
    end = wp.Event(enable_timing=True)

    wp.synchronize_device()
    wp.record_event(start)
    for _ in range(num_iters):
        wp.launch(kernel, dim=[batch_size], inputs=args, block_dim=NUM_THREADS)
    wp.record_event(end)
    wp.synchronize_event(end)
    total_ms = wp.get_event_elapsed_time(start, end)
    return (total_ms * 1000.0) / num_iters  # us per launch


def time_graph(kernel, batch_size, args, launches_per_capture, num_replays):
    """Capture launches_per_capture launches in a graph and replay it.

    Returns per-launch microseconds (averaged across all replays).
    """
    # Warm up the kernel before capture (allocator + JIT).
    wp.launch(kernel, dim=[batch_size], inputs=args, block_dim=NUM_THREADS)
    wp.synchronize_device()

    wp.capture_begin()
    for _ in range(launches_per_capture):
        wp.launch(kernel, dim=[batch_size], inputs=args, block_dim=NUM_THREADS)
    graph = wp.capture_end()

    start = wp.Event(enable_timing=True)
    end = wp.Event(enable_timing=True)

    wp.synchronize_device()
    wp.record_event(start)
    for _ in range(num_replays):
        wp.capture_launch(graph)
    wp.record_event(end)
    wp.synchronize_event(end)
    total_ms = wp.get_event_elapsed_time(start, end)
    total_launches = launches_per_capture * num_replays
    return (total_ms * 1000.0) / total_launches


def repeated(measure_fn, repeats):
    """Run measure_fn() repeats times, return (mean, std) in microseconds."""
    samples = [measure_fn() for _ in range(repeats)]
    return statistics.mean(samples), statistics.stdev(samples) if len(samples) > 1 else 0.0


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def measure_one(hidden, batch_size, kernel, args, rng):
    """Run a full eager + graph measurement for one (hidden, batch) config."""
    weights_0, bias_0 = make_layer(int(DIM_IN), hidden, rng)
    weights_1, bias_1 = make_layer(hidden, hidden, rng)
    weights_2, bias_2 = make_layer(hidden, hidden, rng)
    weights_3, bias_3 = make_layer(hidden, DIM_OUT, rng)

    all_indices = np.arange(IMG_WIDTH * IMG_HEIGHT, dtype=np.int32)
    rng.shuffle(all_indices)
    indices = wp.array(all_indices[:batch_size])

    out = wp.zeros((DIM_OUT, IMG_WIDTH * IMG_HEIGHT), dtype=float)

    kernel_args = [
        indices,
        weights_0, bias_0,
        weights_1, bias_1,
        weights_2, bias_2,
        weights_3, bias_3,
        out,
    ]

    for _ in range(args.warmup):
        wp.launch(kernel, dim=[batch_size], inputs=kernel_args, block_dim=NUM_THREADS)
    wp.synchronize_device()

    eager_mean, eager_std = repeated(
        lambda: time_eager(kernel, batch_size, kernel_args, args.iters),
        args.repeats,
    )
    graph_mean, graph_std = repeated(
        lambda: time_graph(kernel, batch_size, kernel_args, args.graph_launches, args.graph_replays),
        args.repeats,
    )
    return {
        "eager_mean": eager_mean, "eager_std": eager_std,
        "graph_mean": graph_mean, "graph_std": graph_std,
    }


def print_single(hidden, batch_size, r):
    eager_tp = batch_size / (r["eager_mean"] * 1e-6) / 1e6
    graph_tp = batch_size / (r["graph_mean"] * 1e-6) / 1e6
    print()
    print(f"Network:  in={int(DIM_IN)}  hidden={hidden} x 3  out={DIM_OUT}  dtype={dtype}")
    print(f"Batch:    {batch_size}  (block_dim={NUM_THREADS})")
    print(f"Device:   {wp.get_device()}")
    print()
    print(f"  Eager launch:    {r['eager_mean']:8.2f} +/- {r['eager_std']:5.2f} us  "
          f"({eager_tp:7.2f} M samples/s)")
    print(f"  Captured graph:  {r['graph_mean']:8.2f} +/- {r['graph_std']:5.2f} us  "
          f"({graph_tp:7.2f} M samples/s)")
    print(f"  Launch overhead: {r['eager_mean'] - r['graph_mean']:8.2f} us / launch")
    print()


def print_sweep_table(batches, hiddens, results, label, key_mean, key_std):
    """Print a hidden-rows x batch-cols table of '<mean> +/- <std> us' cells."""
    col_w = 20
    print()
    print(f"=== {label} (us per launch) ===")
    header = f"{'hidden \\ batch':>14}" + "".join(f"{b:>{col_w}}" for b in batches)
    print(header)
    print("-" * len(header))
    for h in hiddens:
        row = f"{h:>14}"
        for b in batches:
            r = results[(h, b)]
            cell = f"{r[key_mean]:7.2f} +/- {r[key_std]:5.2f}"
            row += f"{cell:>{col_w}}"
        print(row)

    print()
    print(f"=== {label} throughput (M samples/s) ===")
    print(header)
    print("-" * len(header))
    for h in hiddens:
        row = f"{h:>14}"
        for b in batches:
            r = results[(h, b)]
            tp = b / (r[key_mean] * 1e-6) / 1e6
            row += f"{tp:>{col_w}.2f}"
        print(row)
    print()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=DIM_HID)
    parser.add_argument("--sweep-batch", type=str, default=None,
                        help="Comma-separated batch sizes (enables sweep mode).")
    parser.add_argument("--sweep-hidden", type=str, default=None,
                        help="Comma-separated hidden widths (enables sweep mode).")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup launches before timing.")
    parser.add_argument("--iters", type=int, default=2000, help="Eager launches per measurement.")
    parser.add_argument("--graph-launches", type=int, default=64, help="Launches per captured graph.")
    parser.add_argument("--graph-replays", type=int, default=200, help="Graph replays per measurement.")
    parser.add_argument("--repeats", type=int, default=5, help="Repeat each measurement.")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    sweep_mode = args.sweep_batch is not None or args.sweep_hidden is not None
    batches = parse_int_list(args.sweep_batch) if args.sweep_batch else [args.batch_size]
    hiddens = parse_int_list(args.sweep_hidden) if args.sweep_hidden else [args.hidden]

    with wp.ScopedDevice(args.device):
        rng = np.random.default_rng(45)

        # Compile (and cache) one kernel per unique hidden width.
        kernels: dict[int, object] = {}
        for h in hiddens:
            if h not in kernels:
                k = make_compute_kernel(h)
                wp.load_module(module=k.module, device=wp.get_device(), block_dim=NUM_THREADS)
                kernels[h] = k

        results: dict[tuple[int, int], dict] = {}
        for h in hiddens:
            for b in batches:
                if sweep_mode:
                    print(f"  measuring  hidden={h:>3}  batch={b:>5} ...", flush=True)
                results[(h, b)] = measure_one(h, b, kernels[h], args, rng)

        if not sweep_mode:
            print_single(hiddens[0], batches[0], results[(hiddens[0], batches[0])])
        else:
            print()
            print(f"Network:  in={int(DIM_IN)}  hidden=H x 3  out={DIM_OUT}  dtype={dtype}")
            print(f"Device:   {wp.get_device()}  (block_dim={NUM_THREADS})")
            print_sweep_table(batches, hiddens, results, "Captured graph",
                              "graph_mean", "graph_std")
            print_sweep_table(batches, hiddens, results, "Eager launch",
                              "eager_mean", "eager_std")


if __name__ == "__main__":
    main()
