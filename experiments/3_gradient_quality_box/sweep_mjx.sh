#!/usr/bin/env bash
# MJX optimiser sweep: (lr x beta1) at fixed clip=1.0
#
# Each MJX iter is ~17 s. We use 30 iters x 1 trial per config for screening
# (~10 min per config). After picking the winning (lr, beta1) from the summary
# table, re-run optimize_mjx.py with --num-trials 3 --iterations 50 for the
# headline figure.
#
# Why this grid:
# - lr: 0.05 (touched 0.27 transiently but bounced), 0.02 (cleaner but slow),
#   0.005 (much smaller step, see if no-momentum needs less LR damping).
# - beta1: 0.9 (standard Adam — bad gradient directions averaged ~10 iters
#   into m, then carried forward) vs 0.0 (RMSprop — direction comes only
#   from current clipped gradient, no historical contamination).
#
# Clipping at 1.0 stays on for all configs: the iter-36 |g|=504,224 spike in
# the prior run confirmed BPTT-through-contact produces pathological gradients
# that would otherwise destabilise any optimiser.

set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_DIR="$DIR/results/sweep_mjx"
mkdir -p "$SWEEP_DIR"

ITERATIONS="${ITERATIONS:-30}"
TRIALS="${TRIALS:-1}"
CLIP="${CLIP:-1.0}"

LRS=(0.05 0.02 0.005)
BETA1S=(0.9 0.0)

echo "=== MJX sweep (lr x beta1) ==="
echo "iters=$ITERATIONS  trials=$TRIALS  clip=$CLIP"
echo "lrs: ${LRS[*]}"
echo "beta1s: ${BETA1S[*]}"
echo "results: $SWEEP_DIR/"
echo ""

for lr in "${LRS[@]}"; do
    for b1 in "${BETA1S[@]}"; do
        tag="lr${lr}_b1${b1}"
        out="$SWEEP_DIR/${tag}.json"
        if [[ -f "$out" ]]; then
            echo "[skip] $tag (already exists: $out)"
            continue
        fi
        echo "=== $tag ==="
        python "$DIR/optimize_mjx.py" \
            --lr "$lr" --beta1 "$b1" --clip-grad-norm "$CLIP" \
            --iterations "$ITERATIONS" --num-trials "$TRIALS" \
            --save "$out"
        echo ""
    done
done

echo ""
echo "=== SUMMARY ==="
python3 - <<PY
import json, pathlib
sweep_dir = pathlib.Path("$SWEEP_DIR")
rows = []
for p in sorted(sweep_dir.glob("*.json")):
    d = json.load(open(p))
    t = d["trials"][0]
    final = t["losses"][-1]
    best = min(t["losses"])
    best_iter = t["losses"].index(best)
    n_clipped = t.get("n_clipped", 0)
    rows.append((d.get("lr"), d.get("beta1"), final, best, best_iter,
                 n_clipped, d["iterations"], p.name))

print(f"{'lr':>6s}  {'beta1':>5s}  {'final':>7s}  {'best':>7s}  {'@iter':>6s}  {'clip/total':>10s}  file")
print("-" * 80)
for lr, b1, fin, best, bi, nc, it, name in sorted(rows, key=lambda r: r[3]):
    star = "  <- best" if (lr, b1, fin, best, bi, nc, it, name) == min(rows, key=lambda r: r[3]) else ""
    print(f"{lr:>6.3f}  {b1:>5.2f}  {fin:>7.4f}  {best:>7.4f}  {bi:>6d}  {nc:>4d}/{it:<4d}   {name}{star}")
PY
