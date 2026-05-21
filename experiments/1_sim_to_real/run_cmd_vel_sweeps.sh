#!/usr/bin/env bash
# Sweep Axion and MuJoCo on the turn bag using /cmd_vel-derived wheel targets
# (diff-drive kinematic) instead of /joint_states. Only the turn bag has
# cmd_vel populated, so we restrict to it.
#
# Usage:
#   ./run_cmd_vel_sweeps.sh              # both
#   ./run_cmd_vel_sweeps.sh --axion      # axion only
#   ./run_cmd_vel_sweeps.sh --mujoco     # mujoco only
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$DIR/../data"
BAGS="$DIR/../../data/rosbags/nuc"
RESULTS="$DIR/results"
mkdir -p "$RESULTS"

GT="$DATA/right_turn_b.json"
BAG="$BAGS/helhest_2026_04_10-14_46_18"

RUN_AXION=false; RUN_MUJOCO=false; RUN_ALL=true
for arg in "$@"; do
    case $arg in
        --axion) RUN_AXION=true; RUN_ALL=false;;
        --mujoco) RUN_MUJOCO=true; RUN_ALL=false;;
    esac
done

if $RUN_ALL || $RUN_AXION; then
    echo "=== Axion (cmd_vel kinematic target) ==="
    python "$DIR/sweep_axion.py" \
        --ground-truth "$GT" \
        --cmd-vel-bag "$BAG" \
        --dt 0.05 0.08 \
        --mu 0.1 0.15 0.2 0.3 0.5 \
        --fc 1e-3 5e-3 1.2e-2 2e-2 5e-2 \
        --cc 1e-1 \
        --save "$RESULTS/sweep_axion_cmdvel.json"
    echo ""
fi

if $RUN_ALL || $RUN_MUJOCO; then
    echo "=== MuJoCo (cmd_vel kinematic target) ==="
    python "$DIR/sweep_mujoco.py" \
        --ground-truth "$GT" \
        --cmd-vel-bag "$BAG" \
        --dt 0.001 0.002 0.005 \
        --kv 1000 2000 4000 8000 16000 \
        --mu 0.1 0.2 0.35 0.5 0.7 1.0 \
        --save "$RESULTS/sweep_mujoco_cmdvel.json"
    echo ""
fi

echo "Done. Results:"
echo "  $RESULTS/sweep_axion_cmdvel.json"
echo "  $RESULTS/sweep_mujoco_cmdvel.json"
