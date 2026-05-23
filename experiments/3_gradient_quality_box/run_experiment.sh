#!/usr/bin/env bash
# Run gradient-quality optimization for all available engines, box scene.
#
# Each script loads a real GT trajectory, uses calibrated params from
# experiments/1_sim_to_real_box, and optimizes a K-knot wheel-velocity
# spline to fit it using its native gradient mechanism.
#
# Usage:
#   ./run_experiment.sh                       # run all available
#   ./run_experiment.sh --axion               # only Axion
#   ./run_experiment.sh --iterations 100      # forward extra args to every runner
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="$DIR/results"
mkdir -p "$RESULTS"

GT="${GT:-$DIR/../1_sim_to_real_box/data/run_2026_05_20-18_10_33.json}"
HORIZON="${HORIZON:-6.0}"
ITERATIONS="${ITERATIONS:-50}"
NUM_TRIALS="${NUM_TRIALS:-3}"
SEED_BASE="${SEED_BASE:-42}"
K="${K:-10}"
LR="${LR:-0.1}"

RUN_AXION=false
RUN_ALL=true
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --axion) RUN_AXION=true; RUN_ALL=false; shift;;
        *)       EXTRA_ARGS+=("$1"); shift;;
    esac
done

COMMON_ARGS=(--gt "$GT" --horizon-s "$HORIZON" --iterations "$ITERATIONS" \
             --num-trials "$NUM_TRIALS" --seed-base "$SEED_BASE" \
             --K "$K" --lr "$LR")

if $RUN_ALL || $RUN_AXION; then
    echo "=== Axion (adjoint)  [horizon=${HORIZON}s, iters=${ITERATIONS}, trials=${NUM_TRIALS}, K=${K}] ==="
    python "$DIR/optimize_axion.py" \
        "${COMMON_ARGS[@]}" --save "$RESULTS/axion.json" "${EXTRA_ARGS[@]}"
    echo ""
fi

echo "Done. Results in $RESULTS/"
echo "Plot with:  python $DIR/plot_results.py"
