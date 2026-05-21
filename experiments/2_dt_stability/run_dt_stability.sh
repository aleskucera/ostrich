#!/usr/bin/env bash
# Run dt stability sweeps for all simulators on the obstacle scene.
# Uses calibrated params from Experiment 1 (sim_to_real).
#
# Usage:
#   ./run_dt_stability.sh                   # run all
#   ./run_dt_stability.sh --axion           # run only Axion
#   ./run_dt_stability.sh --axion --mujoco  # run Axion and MuJoCo
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="$DIR/results"
mkdir -p "$RESULTS"

RUN_AXION=false; RUN_MUJOCO=false; RUN_SEMI=false
RUN_ALL=true
NUM_TRIALS=""
SEED=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --axion) RUN_AXION=true; RUN_ALL=false; shift;;
        --mujoco) RUN_MUJOCO=true; RUN_ALL=false; shift;;
        --semi-implicit) RUN_SEMI=true; RUN_ALL=false; shift;;
        --num-trials) NUM_TRIALS="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

EXTRA=()
[[ -n "$NUM_TRIALS" ]] && EXTRA+=(--num-trials "$NUM_TRIALS")
[[ -n "$SEED" ]] && EXTRA+=(--seed "$SEED")

if $RUN_ALL || $RUN_AXION; then
    echo "=== Axion ==="
    python "$DIR/sweep_axion.py" --save "$RESULTS/sweep_axion.json" "${EXTRA[@]}"
    echo ""
fi

if $RUN_ALL || $RUN_MUJOCO; then
    echo "=== MuJoCo ==="
    python "$DIR/sweep_mujoco.py" --save "$RESULTS/sweep_mujoco.json" "${EXTRA[@]}"
    echo ""
fi

if $RUN_ALL || $RUN_SEMI; then
    echo "=== Semi-Implicit ==="
    python "$DIR/sweep_semi_implicit.py" --save "$RESULTS/sweep_semi_implicit.json" "${EXTRA[@]}"
    echo ""
fi

echo "Done. Results in $RESULTS/"
