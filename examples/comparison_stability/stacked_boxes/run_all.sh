#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DIR/results"

python "$DIR/axion_sim.py"        --save "$DIR/results/axion.json"
python "$DIR/semi_implicit.py"    --save "$DIR/results/semi_implicit.json"
python "$DIR/mujoco_sim.py"       --save "$DIR/results/mujoco.json"
python "$DIR/mjx.py"              --save "$DIR/results/mjx.json"
python "$DIR/genesis_sim.py"      --save "$DIR/results/genesis.json"
python "$DIR/tinydiffsim.py"      --save "$DIR/results/tinydiffsim.json"
# Dojo: no native box-box collision; SphereBoxCollision point-contact
# approximation is unstable at all tested timesteps. Skipped.

python "$DIR/plot_results.py"
