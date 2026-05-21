#!/bin/bash
set -e

for N in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536; do
    echo "=== Running Axion with $N worlds ==="
    python examples/comparison_scalability/helhest_scalability/axion_sim.py \
        --num-worlds $N --save results/scalability_axion_${N}.json
done

echo "Done!"
