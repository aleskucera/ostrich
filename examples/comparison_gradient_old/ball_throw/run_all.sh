#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DIR/results"

python "$DIR/axion_sim.py"       --save "$DIR/results/axion.json"
python "$DIR/genesis_sim.py"     --save "$DIR/results/genesis.json"
python "$DIR/mjfd.py"        --save "$DIR/results/mjfd.json"
python "$DIR/mjx.py"         --save "$DIR/results/mjx.json"
python "$DIR/mjx_jacfwd.py"  --save "$DIR/results/mjx_jacfwd.json"
python "$DIR/featherstone.py" --save "$DIR/results/featherstone.json"
python "$DIR/xpbd.py"        --save "$DIR/results/xpbd.json"
python "$DIR/tinydiffsim.py" --save "$DIR/results/tinydiffsim.json"
python "$DIR/brax_sim.py"        --save "$DIR/results/brax.json"
python "$DIR/diffarticulated_sim.py" --save "$DIR/results/diffarticulated.json"
