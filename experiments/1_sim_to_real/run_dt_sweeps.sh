#!/usr/bin/env bash
# Run dt-only sweeps with fixed calibrated params from the full sweep.
#
# Usage:
#   ./run_dt_sweeps.sh                    # run all
#   ./run_dt_sweeps.sh --axion            # run only Axion
#   ./run_dt_sweeps.sh --axion --mujoco   # run Axion and MuJoCo
#
# Update the fixed param values below after running the full sweep.
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$DIR/../data"
RESULTS="$DIR/results"
mkdir -p "$RESULTS"

GT="$DATA/right_turn_b.json $DATA/acceleration.json"

# Parse args
RUN_AXION=false; RUN_MUJOCO=false; RUN_SEMI=false; RUN_TINY=false
RUN_ALL=true
for arg in "$@"; do
    case $arg in
        --axion) RUN_AXION=true; RUN_ALL=false;;
        --mujoco) RUN_MUJOCO=true; RUN_ALL=false;;
        --tinydiffsim) RUN_TINY=true; RUN_ALL=false;;
        --semi-implicit) RUN_SEMI=true; RUN_ALL=false;;
    esac
done

if $RUN_ALL || $RUN_AXION; then
    # Axion: calibrated mu=0.1, fc=0.02, cc=0.1 — sweep dt only
    echo "=== Axion dt sweep ==="
    python "$DIR/sweep_axion.py" \
        --ground-truth $GT \
        --dt 0.02 0.05 0.08 0.1 0.125 0.15 0.2 \
        --mu 0.1 \
        --fc 2e-2 \
        --cc 1e-1 \
        --save "$RESULTS/sweep_axion_dt.json"
    echo ""
fi

if $RUN_ALL || $RUN_MUJOCO; then
    # MuJoCo: calibrated kv=4000, mu=0.2, implicitfast — sweep dt only
    echo "=== MuJoCo dt sweep ==="
    python "$DIR/sweep_mujoco.py" \
        --ground-truth $GT \
        --dt 0.0005 0.001 0.002 0.005 0.01 0.02 \
        --kv 4000 \
        --mu 0.2 \
        --save "$RESULTS/sweep_mujoco_dt.json"
    echo ""
fi

if $RUN_ALL || $RUN_SEMI; then
    # Semi-Implicit: calibrated k_d=400, mu=0.02, kf=1500 — sweep dt only
    echo "=== Semi-Implicit dt sweep ==="
    python "$DIR/sweep_semi_implicit.py" \
        --ground-truth $GT \
        --dt 0.0001 0.0002 0.0005 0.001 0.002 \
        --k-d 400 \
        --mu 0.02 \
        --kf 1500 \
        --save "$RESULTS/sweep_semi_implicit_dt.json"
    echo ""
fi

if $RUN_ALL || $RUN_TINY; then
    # TinyDiffSim: calibrated kv=36, friction=0.12 — sweep dt only
    echo "=== TinyDiffSim dt sweep ==="
    python "$DIR/sweep_tinydiffsim.py" \
        --ground-truth $GT \
        --dt 0.0005 0.001 0.002 0.005 \
        --kv 36 \
        --friction 0.12 \
        --save "$RESULTS/sweep_tinydiffsim_dt.json"
    echo ""
fi

echo "Done. Results in $RESULTS/"
