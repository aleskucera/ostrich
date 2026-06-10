"""For specific simulation steps, dump the per-NR-iter residual trace and (with
linesearch on) the chosen α."""
import sys

import numpy as np


def main(npz_path: str, *step_ids: int):
    a = np.load(npz_path)
    nr_res = a["nr_res_norm_sq"]   # (S, K, W)
    nr_iters = a["nr_iters"]       # (S,)
    pcr = a["pcr_iter_counts"]     # (S, K)

    has_ls = "ls_min_idx" in a.files
    if has_ls:
        ls_idx = a["ls_min_idx"]   # (S, K, W)
        grid = a["ls_step_sizes"]  # (step_count,)

    if not step_ids:
        # Default: top 5 worst steps
        final = np.array([nr_res[s, int(nr_iters[s]) - 1, 0] for s in range(nr_res.shape[0])])
        step_ids = tuple(int(s) for s in np.argsort(-final)[:5])
        print(f"(no step IDs given; showing top-5 worst final NR residuals: {step_ids})\n")

    for sid in step_ids:
        K = int(nr_iters[sid])
        print(f"=== step {sid}  NR iters used: {K}/{nr_res.shape[1]} ===")
        print(f"{'iter':>4} {'log10 ||r||²':>14} {'PCR iters':>10}", end="")
        if has_ls:
            print(f" {'α':>10}", end="")
        print()
        for k in range(K):
            r = nr_res[sid, k, 0]
            log_r = np.log10(max(r, 1e-30))
            line = f"{k:>4d} {log_r:>14.3f} {int(pcr[sid, k]):>10d}"
            if has_ls:
                a_val = float(grid[ls_idx[sid, k, 0]])
                line += f" {a_val:>10.3e}"
            print(line)
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python inspect_bad_step.py <npz_path> [step_id ...]")
        sys.exit(1)
    npz_path = sys.argv[1]
    step_ids = [int(x) for x in sys.argv[2:]]
    main(npz_path, *step_ids)
