#!/usr/bin/env bash
# Run full parameter sweeps for all simulators against real robot trajectories.
#
# Usage:
#   ./run_full_sweeps.sh                    # run all
#   ./run_full_sweeps.sh --axion            # run only Axion
#   ./run_full_sweeps.sh --axion --mujoco   # run Axion and MuJoCo
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$DIR/../data"
RESULTS="$DIR/results"
mkdir -p "$RESULTS"

GT="$DATA/right_turn_b.json $DATA/acceleration.json"

# Parse args: if none given, run all
RUN_AXION=false; RUN_MUJOCO=false; RUN_SEMI=false; RUN_TINY=false; RUN_DOJO=false
RUN_ALL=true
for arg in "$@"; do
    case $arg in
        --axion) RUN_AXION=true; RUN_ALL=false;;
        --mujoco) RUN_MUJOCO=true; RUN_ALL=false;;
        --semi-implicit) RUN_SEMI=true; RUN_ALL=false;;
        --tinydiffsim) RUN_TINY=true; RUN_ALL=false;;
        --dojo) RUN_DOJO=true; RUN_ALL=false;;
    esac
done

if $RUN_ALL || $RUN_AXION; then
    echo "=== Axion ==="
    python "$DIR/sweep_axion.py" \
        --ground-truth $GT \
        --dt 0.05 0.08 \
        --mu 0.1 0.15 \
        --fc 1e-3 5e-3 1.2e-2 2e-2 5e-2 \
        --cc 1e-1 \
        --save "$RESULTS/sweep_axion.json"
    echo ""
fi

if $RUN_ALL || $RUN_MUJOCO; then
    echo "=== MuJoCo ==="
    python "$DIR/sweep_mujoco.py" \
        --ground-truth $GT \
        --dt 0.001 0.002 0.005 \
        --kv 1000 2000 4000 8000 16000 \
        --mu 0.1 0.2 0.35 0.5 0.7 1.0 \
        --save "$RESULTS/sweep_mujoco.json"
    echo ""
fi

if $RUN_ALL || $RUN_SEMI; then
    echo "=== Semi-Implicit ==="
    python "$DIR/sweep_semi_implicit.py" \
        --ground-truth $GT \
        --dt 0.0005 \
        --k-d 200 400 800 \
        --mu 0.005 0.01 0.02 0.05 \
        --kf 400 800 1500 \
        --save "$RESULTS/sweep_semi_implicit.json"
    echo ""
fi

if $RUN_ALL || $RUN_TINY; then
    echo "=== TinyDiffSim ==="
    python "$DIR/sweep_tinydiffsim.py" \
        --ground-truth $GT \
        --save "$RESULTS/sweep_tinydiffsim.json"
    echo ""
fi

if $RUN_ALL || $RUN_DOJO; then
    echo "=== Dojo ==="
    ~/.juliaup/bin/julia +1.10 --startup-file=no "$DIR/sweep_dojo.jl" \
        --ground-truth $GT \
        --save "$RESULTS/sweep_dojo.json"
    echo ""
fi

echo "Done. Results in $RESULTS/"
