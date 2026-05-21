#!/usr/bin/env bash
# Sweep num_worlds for Axion and MJX, skipping gracefully on OOM or crash.
#
# Usage: bash examples/comparison/helhest_scalability/run_sweep.sh

set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DIR/results"

AXION_WORLDS=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144)
MJX_WORLDS=(1 2 4 8 16 32)   # MJX OOMs around 32 worlds (BPTT tape ~1.3 GB/world)

run_sim() {
    local script="$1"
    local sim_name="$2"
    local n="$3"
    local out="$4"

    if [ -f "$out" ]; then
        echo "[$sim_name | worlds=$n] Already exists, skipping."
        return 0
    fi

    echo "[$sim_name | worlds=$n] Running..."
    if python "$script" --num-worlds "$n" --save "$out"; then
        echo "[$sim_name | worlds=$n] Done."
    else
        echo "[$sim_name | worlds=$n] FAILED (OOM or crash) — skipping."
        rm -f "$out"
    fi
}

for N in "${AXION_WORLDS[@]}"; do
    run_sim "$DIR/axion_sim.py"   "Axion"      "$N" "$DIR/results/axion_${N}.json"
done

for N in "${MJX_WORLDS[@]}"; do
    run_sim "$DIR/mjx.py"         "MJX-grad"   "$N" "$DIR/results/mjx_grad_${N}.json"
done

# TinyDiffSim, MJX-jacfwd, Brax: kept on disk as references; not part of §IV-D
# (TinyDiffSim is CPU-bound; Semi-Implicit's gradients fail to converge in §IV-C).

echo ""
echo "Sweep complete. Plotting..."
python "$DIR/plot_results.py"
