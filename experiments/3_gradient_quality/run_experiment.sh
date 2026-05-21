#!/usr/bin/env bash
# Run gradient-quality optimization for all simulators.
#
# Each script loads the real-robot ground truth, uses calibrated params from
# Experiment 1, and optimizes a K-knot spline to match the real trajectory.
#
# Usage:
#   ./run_all.sh                             # run all
#   ./run_all.sh --axion                     # run only Axion
#   ./run_all.sh --axion --mjx               # run Axion and MJX
#   ./run_all.sh --iterations 100 --K 15     # forward extra args to every script
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$DIR/../data"
RESULTS="$DIR/results"
mkdir -p "$RESULTS"

GT="${GT:-$DATA/right_turn_b.json}"   # override with GT=$DATA/acceleration.json ./run_experiment.sh
HORIZON="${HORIZON:-2.0}"     # seconds;     override with HORIZON=9.2 ./run_experiment.sh
ITERATIONS="${ITERATIONS:-50}"   # opt iters; override with ITERATIONS=200 ./run_experiment.sh
NUM_TRIALS="${NUM_TRIALS:-3}"    # trials per sim; override with NUM_TRIALS=5 ./run_experiment.sh
SEED_BASE="${SEED_BASE:-42}"     # seed for trial k is SEED_BASE + k

# Save-file suffix derived from GT filename (so runs on different trajectories don't overwrite each other)
GT_STEM="$(basename "$GT" .json)"
if [ "$GT_STEM" != "right_turn_b" ]; then
    SAVE_SUFFIX="_${GT_STEM}"
else
    SAVE_SUFFIX=""
fi

RUN_AXION=false; RUN_MJX=false; RUN_SEMI=false; RUN_TINY=false
RUN_ALL=true
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --axion)          RUN_AXION=true; RUN_ALL=false; shift;;
        --mjx)            RUN_MJX=true;   RUN_ALL=false; shift;;
        --semi-implicit)  RUN_SEMI=true;  RUN_ALL=false; shift;;
        --tinydiffsim)    RUN_TINY=true;  RUN_ALL=false; shift;;
        *)                EXTRA_ARGS+=("$1"); shift;;
    esac
done

COMMON_ARGS=(--ground-truth "$GT" --horizon-s "$HORIZON" --iterations "$ITERATIONS" \
             --num-trials "$NUM_TRIALS" --seed-base "$SEED_BASE")

if $RUN_ALL || $RUN_AXION; then
    echo "=== Axion (adjoint)  [horizon=${HORIZON}s, iters=${ITERATIONS}, trials=${NUM_TRIALS}] ==="
    python "$DIR/optimize_axion.py" \
        "${COMMON_ARGS[@]}" --save "$RESULTS/axion${SAVE_SUFFIX}.json" "${EXTRA_ARGS[@]}"
    echo ""
fi

if $RUN_ALL || $RUN_MJX; then
    echo "=== MJX (jax.grad / BPTT)  [horizon=${HORIZON}s, iters=${ITERATIONS}, trials=${NUM_TRIALS}] ==="
    python "$DIR/optimize_mjx.py" \
        "${COMMON_ARGS[@]}" --save "$RESULTS/mjx${SAVE_SUFFIX}.json" "${EXTRA_ARGS[@]}"
    echo ""
fi

if $RUN_ALL || $RUN_SEMI; then
    echo "=== Semi-Implicit (warp tape / BPTT)  [horizon=${HORIZON}s, iters=${ITERATIONS}, trials=${NUM_TRIALS}] ==="
    python "$DIR/optimize_semi_implicit.py" \
        "${COMMON_ARGS[@]}" --save "$RESULTS/semi_implicit${SAVE_SUFFIX}.json" "${EXTRA_ARGS[@]}"
    echo ""
fi

if $RUN_ALL || $RUN_TINY; then
    echo "=== TinyDiffSim (CppAD)  [horizon=${HORIZON}s, iters=${ITERATIONS}, trials=${NUM_TRIALS}] ==="
    python "$DIR/optimize_tinydiffsim.py" \
        "${COMMON_ARGS[@]}" --save "$RESULTS/tinydiffsim${SAVE_SUFFIX}.json" "${EXTRA_ARGS[@]}"
    echo ""
fi

echo "Done. Results in $RESULTS/"
