"""For specific (step, NR-iter) pairs, dump the full linesearch residual
landscape: residual at each candidate α, and which one was chosen.
"""
import sys

import numpy as np


def main(npz_path: str, step_id: int, iter_ids):
    a = np.load(npz_path)
    ls_res = a["ls_res_norm_sq"]      # (S, K, A, W)
    grid = a["ls_step_sizes"]         # (A,)
    ls_idx = a["ls_min_idx"]          # (S, K, W)
    nr_res = a["nr_res_norm_sq"]      # (S, K, W)
    nr_iters = a["nr_iters"]          # (S,)

    K_used = int(nr_iters[step_id])
    print(f"=== step {step_id}  (NR ran {K_used} iters) ===\n")

    for k in iter_ids:
        if k >= K_used:
            print(f"--- iter {k}: did not run ---")
            continue

        chosen = int(ls_idx[step_id, k, 0])
        chosen_alpha = float(grid[chosen])
        chosen_res = float(ls_res[step_id, k, chosen, 0])

        # Sort candidates by α and print residuals.
        order = np.argsort(grid)
        print(f"--- step {step_id}, NR iter {k} ---")
        print(f"chosen idx={chosen}  α={chosen_alpha:.3e}  residual²={chosen_res:.3e}  "
              f"(log10 = {np.log10(max(chosen_res,1e-30)):+.3f})")
        print(f"min residual² across all 64 candidates = "
              f"{ls_res[step_id, k, :, 0].min():.3e}")
        print(f"argmin = idx {int(np.argmin(ls_res[step_id, k, :, 0]))}, "
              f"α = {grid[int(np.argmin(ls_res[step_id, k, :, 0]))]:.3e}")
        print(f"\n{'idx':>4} {'α':>12} {'residual²':>14} {'log10':>10} {'mark':>6}")
        for j in order:
            r = float(ls_res[step_id, k, j, 0])
            mark = " <-" if j == chosen else ""
            print(f"{int(j):>4d} {float(grid[j]):>12.3e} {r:>14.3e} "
                  f"{np.log10(max(r,1e-30)):>10.3f}{mark}")
        # post-step residual saved to candidates
        print(f"\npost-step nr_res[{k}] = {float(nr_res[step_id, k, 0]):.3e} "
              f"(log10 = {np.log10(max(float(nr_res[step_id, k, 0]),1e-30)):+.3f})")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python inspect_ls_landscape.py <npz_path> <step_id> [iter_id ...]")
        sys.exit(1)
    npz_path = sys.argv[1]
    step_id = int(sys.argv[2])
    iter_ids = [int(x) for x in sys.argv[3:]] or [0, 1, 2]
    main(npz_path, step_id, iter_ids)
