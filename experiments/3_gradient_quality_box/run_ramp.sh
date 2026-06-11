#!/usr/bin/env bash
# Gradient difficulty-ramp matrix: every engine x every horizon.
#
# Runs the same box inverse problem at increasing horizons to show which
# baselines descend on the easy (pre-contact) problem and collapse on the full
# box climb. Produces results/ramp_<engine>_<H>s.json for plot_ramp.py.
#
# Intended for the 24 GB machine (dasenka): MJX needs its own venv, Brax
# generalized OOMs below ~8 GB, and 6 s runs are heavy. Set the venv paths
# below for the target machine, then: bash run_ramp.sh
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
RES="$HERE/results"
mkdir -p "$RES"

# --- per-machine interpreter paths (EDIT FOR DASENKA) -----------------------
PY_OSTRICH="${PY_OSTRICH:-$HOME/projects/ostrich/.venv/bin/python}"        # ostrich + Newton/Warp (Ostrich, Semi, XPBD)
PY_BRAX="${PY_BRAX:-$HOME/projects/ostrich/.venv-brax/bin/python}"     # brax + jax
PY_MJX="${PY_MJX:-$HOME/projects/ostrich/.venv-mjx/bin/python}"        # mujoco-mjx + jax  (set to your MJX venv)

# --- ramp: horizon (s) -> spline knots K ------------------------------------
HORIZONS=(1 3 6)
declare -A K_FOR=( [1]=3 [3]=6 [6]=10 )

ITERS=50
TRIALS=3

run () {  # run <label> <python> <script+args...>
  local label="$1"; shift
  local py="$1"; shift
  for H in "${HORIZONS[@]}"; do
    local K="${K_FOR[$H]}"
    local out="$RES/ramp_${label}_${H}s.json"
    echo "=== $label  H=${H}s  K=$K -> $out ==="
    "$py" "$@" --horizon-s "$H" --K "$K" --iterations "$ITERS" \
        --num-trials "$TRIALS" --save "$out" \
      || echo "!!! $label H=${H}s FAILED (continuing)"
  done
}

# Ostrich (reference; works throughout)
run ostrich        "$PY_OSTRICH" "$HERE/optimize_ostrich.py"
# MJX (reference; works throughout)
run mjx            "$PY_MJX"   "$HERE/optimize_mjx.py"
# Newton/Warp solvers
run semi_implicit  "$PY_OSTRICH" "$HERE/optimize_semi_implicit.py"
run xpbd           "$PY_OSTRICH" "$HERE/optimize_xpbd.py"
# Brax pipelines (positional uses spheres - capsule unstable in positional;
# spring/generalized use capsules, matching MJX's capsule wheels)
run brax_positional "$PY_BRAX" "$HERE/optimize_brax.py" --pipeline positional --wheel sphere
run brax_spring     "$PY_BRAX" "$HERE/optimize_brax.py" --pipeline spring     --wheel capsule
run brax_generalized "$PY_BRAX" "$HERE/optimize_brax.py" --pipeline generalized --wheel capsule

echo "ALL DONE -> $RES"
