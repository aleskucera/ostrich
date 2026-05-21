#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DIR/results"

# Axion — implicit servo (dt_sweep, gain_sweep, binary_search)
python "$DIR/axion_sim.py" --experiment dt_sweep   --save "$DIR/results/axion_implicit_dt.json"
python "$DIR/axion_sim.py" --experiment gain_sweep  --save "$DIR/results/axion_implicit_gain.json"
python "$DIR/axion_sim.py" --experiment binary_search --save "$DIR/results/axion_threshold.json"

# Semi-Implicit — explicit PD (binary_search)
python "$DIR/semi_implicit.py" --experiment binary_search --save "$DIR/results/semi_implicit_threshold.json"

# MuJoCo — explicit PD (dt_sweep, gain_sweep, binary_search)
python "$DIR/mujoco_sim.py" --experiment dt_sweep   --save "$DIR/results/mujoco_dt.json"
python "$DIR/mujoco_sim.py" --experiment gain_sweep  --save "$DIR/results/mujoco_gain.json"
python "$DIR/mujoco_sim.py" --experiment binary_search --save "$DIR/results/mujoco_threshold.json"

# Genesis — binary_search only
python "$DIR/genesis_sim.py" --experiment binary_search --save "$DIR/results/genesis_threshold.json"

# MJX — binary_search only
python "$DIR/mjx.py" --experiment binary_search --save "$DIR/results/mjx_threshold.json"

# TinyDiffSim — binary_search only
python "$DIR/tinydiffsim.py" --experiment binary_search --save "$DIR/results/tinydiffsim_threshold.json"

# Dojo — explicit PD (binary_search)
~/.juliaup/bin/julia +1.10 --startup-file=no "$DIR/dojo.jl" \
    --experiment binary_search --save "$DIR/results/dojo_threshold.json"

# Plot threshold comparison
python "$DIR/plot_results.py" --mode threshold

# Plot trajectory/gain comparison
python "$DIR/plot_results.py" --mode trajectory
