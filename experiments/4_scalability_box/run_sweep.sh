#!/usr/bin/env bash
# Sweep num_worlds for Axion, MJX, and Semi-Implicit on the box scene.
# Skip-on-OOM (so the sweep continues past the first failure) and skip
# already-existing result files (so the sweep is resumable).
#
# Per-engine compute (3090, dasenka):
#   - Axion : ~1.5 min/run (~30 min for the full 18-point grid)
#   - MJX   : ~3 min/run   (~25 min for 1..32 grid; OOM expected ~32-64)
#   - SI    : ~20 min/run (each invocation pays ~20 min cold CUDA-graph
#             capture; the actual 5-iter measurement is ~30 s)
#
# Usage:
#   bash experiments/4_scalability_box/run_sweep.sh                # all engines
#   bash experiments/4_scalability_box/run_sweep.sh --axion        # subset

set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="$DIR/results"
mkdir -p "$RESULTS"

AXION_WORLDS=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072)
MJX_WORLDS=(1 2 4 8 16 32 64)
# SI surprisingly memory-efficient (~26 MB/world at N=32), should scale much
# higher than the original cap. Extended to 1024 to find the real wall.
SI_WORLDS=(1 2 4 8 16 32 64 128 256 512 1024)

RUN_ALL=true; RUN_AXION=false; RUN_MJX=false; RUN_SI=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --axion)         RUN_AXION=true; RUN_ALL=false; shift;;
        --mjx)           RUN_MJX=true;   RUN_ALL=false; shift;;
        --semi-implicit) RUN_SI=true;    RUN_ALL=false; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

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
        echo "[$sim_name | worlds=$n] FAILED (OOM or crash) — stopping this engine."
        rm -f "$out"
        return 1
    fi
}

if $RUN_ALL || $RUN_AXION; then
    echo "=== Axion sweep ==="
    for N in "${AXION_WORLDS[@]}"; do
        run_sim "$DIR/axion_sim.py" "Axion" "$N" "$RESULTS/axion_${N}.json" || break
    done
fi

if $RUN_ALL || $RUN_MJX; then
    echo ""
    echo "=== MJX sweep ==="
    for N in "${MJX_WORLDS[@]}"; do
        run_sim "$DIR/mjx_sim.py" "MJX-grad" "$N" "$RESULTS/mjx_grad_${N}.json" || break
    done
fi

if $RUN_ALL || $RUN_SI; then
    echo ""
    echo "=== Semi-Implicit sweep (each ~20 min cold capture) ==="
    for N in "${SI_WORLDS[@]}"; do
        run_sim "$DIR/semi_implicit_sim.py" "Semi-Implicit" "$N" \
            "$RESULTS/semi_implicit_${N}.json" || break
    done
fi

echo ""
echo "Sweep complete. Plot with:"
echo "  python $DIR/plot_results.py"
