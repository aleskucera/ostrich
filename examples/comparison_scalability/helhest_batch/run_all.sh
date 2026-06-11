#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DIR/results"

python "$DIR/ostrich_sim.py"   --save "$DIR/results/ostrich.json"
python "$DIR/mjx.py"         --save "$DIR/results/mjx_grad.json"
python "$DIR/mjx_jacfwd.py"  --save "$DIR/results/mjx_jacfwd.json"
python "$DIR/tinydiffsim.py" --save "$DIR/results/tinydiffsim.json"
python "$DIR/brax_sim.py"        --save "$DIR/results/brax.json"
python "$DIR/plot_results.py"
