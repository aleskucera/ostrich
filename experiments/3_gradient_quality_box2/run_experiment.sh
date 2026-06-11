#!/usr/bin/env bash
# Random-IC final-pose experiment: N=25 trials per engine, each with
# different IC + target + spline init. Loss = pos + yaw + terminal-vel +
# smoothness + reg (deployable on the real robot open-loop).
#
# Per-engine wall (3090, N=25, iters=100):
#   - Ostrich: ~30 min (~70 s/trial)
#   - MJX:   ~12 hr (~30 min/trial at horizon=6 s, 100 iters)
#   - SI:    ~25 hr (~60 min/trial; ~20 min cold capture + 100 warm iters)
#
# MJX and SI are the long poles. For a smoke check use --iterations 30 and
# --num-trials 5 first.
#
# Usage:
#   bash run_experiment.sh                    # all engines
#   bash run_experiment.sh --ostrich            # only Ostrich
#   bash run_experiment.sh --iterations 30 --num-trials 5   # quick screen
#
# Skips already-existing JSONs so the script is resumable.

set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="$DIR/results"
mkdir -p "$RESULTS"

ITERATIONS="${ITERATIONS:-100}"
NUM_TRIALS="${NUM_TRIALS:-25}"
SEED_BASE="${SEED_BASE:-42}"

RUN_ALL=true; RUN_OSTRICH=false; RUN_MJX=false; RUN_SI=false
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --ostrich)         RUN_OSTRICH=true; RUN_ALL=false; shift;;
        --mjx)           RUN_MJX=true;   RUN_ALL=false; shift;;
        --semi-implicit) RUN_SI=true;    RUN_ALL=false; shift;;
        --iterations)    ITERATIONS="$2"; shift 2;;
        --num-trials)    NUM_TRIALS="$2"; shift 2;;
        --seed-base)     SEED_BASE="$2"; shift 2;;
        *)               EXTRA_ARGS+=("$1"); shift;;
    esac
done

COMMON=(--iterations "$ITERATIONS" --num-trials "$NUM_TRIALS"
        --seed-base "$SEED_BASE" "${EXTRA_ARGS[@]}")

run_one() {
    local script="$1"
    local name="$2"
    local out="$3"
    if [[ -f "$out" ]]; then
        echo "[$name] Already exists ($out) — skipping."
        return 0
    fi
    echo "=== $name  (iters=$ITERATIONS, trials=$NUM_TRIALS) ==="
    python "$script" "${COMMON[@]}" --save "$out"
    echo ""
}

if $RUN_ALL || $RUN_OSTRICH; then
    run_one "$DIR/optimize_ostrich.py" "Ostrich" "$RESULTS/ostrich.json"
fi
if $RUN_ALL || $RUN_MJX; then
    run_one "$DIR/optimize_mjx.py" "MJX" "$RESULTS/mjx.json"
fi
if $RUN_ALL || $RUN_SI; then
    run_one "$DIR/optimize_semi_implicit.py" "Semi-Implicit" "$RESULTS/semi_implicit.json"
fi

echo "Done. Plot with:"
echo "  python $DIR/plot_results.py"
