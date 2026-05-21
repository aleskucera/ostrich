#!/bin/bash
# Run terrain traversal optimization across multiple seeds.
#
# Usage:
#   ./examples/terrain_traversal/run_batch.sh
#   ./examples/terrain_traversal/run_batch.sh 100   # custom number of seeds

NUM_SEEDS=${1:-50}
RESULTS_DIR="results/terrain_batch"
DURATION=10.0
ITERATIONS=100
SIGMA=1.5
CURVATURE=3.0
ROUGHNESS=2.0
TERRAIN_FREQ=1.5

export WARP_QUIET=1
export WARP_LOG_LEVEL=error

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "  Terrain Traversal Batch Run"
echo "  Seeds: 0..$((NUM_SEEDS-1))"
echo "  Duration: ${DURATION}s | Iterations: ${ITERATIONS}"
echo "  Sigma: ${SIGMA} | Curvature: ${CURVATURE}"
echo "  Roughness: ${ROUGHNESS} | Terrain freq: ${TERRAIN_FREQ}"
echo "  Results: ${RESULTS_DIR}/"
echo "============================================"

for seed in $(seq 0 $((NUM_SEEDS-1))); do
    OUT="$RESULTS_DIR/seed_${seed}.json"
    if [ -f "$OUT" ]; then
        echo "[$seed/$((NUM_SEEDS-1))] Skipping (exists: $OUT)"
        continue
    fi
    echo ""
    echo "[$seed/$((NUM_SEEDS-1))] Running..."
    python -m examples.terrain_traversal.optimize \
        --seed "$seed" \
        --iterations "$ITERATIONS" \
        --duration "$DURATION" \
        --sigma "$SIGMA" \
        --curvature "$CURVATURE" \
        --roughness "$ROUGHNESS" \
        --terrain-freq "$TERRAIN_FREQ" \
        --save "$OUT"
    echo "[$seed/$((NUM_SEEDS-1))] Done -> $OUT"
done

echo ""
echo "============================================"
echo "  All done. Aggregating..."
echo "============================================"

# Print summary from individual JSONs
python -c "
import json, pathlib, numpy as np
results_dir = pathlib.Path('$RESULTS_DIR')
files = sorted(results_dir.glob('seed_*.json'))
if not files:
    print('No results found.')
    exit()
best_rmses = []
final_rmses = []
times = []
for f in files:
    d = json.loads(f.read_text())
    best_rmses.append(min(d['rmse_m']))
    final_rmses.append(d['rmse_m'][-1])
    times.append(float(np.median(d['time_ms'][1:])))
best_rmses = np.array(best_rmses)
final_rmses = np.array(final_rmses)
times = np.array(times)
print(f'  Seeds completed: {len(files)}')
print(f'  Best  RMSE: {np.median(best_rmses):.3f}m median, {np.mean(best_rmses):.3f} +/- {np.std(best_rmses):.3f}m')
print(f'  Final RMSE: {np.median(final_rmses):.3f}m median, {np.mean(final_rmses):.3f} +/- {np.std(final_rmses):.3f}m')
print(f'  Iter  time: {np.median(times):.0f}ms median')
print(f'  Worst best RMSE: {np.max(best_rmses):.3f}m (seed {np.argmax(best_rmses)})')
print(f'  Best  best RMSE: {np.min(best_rmses):.3f}m (seed {np.argmin(best_rmses)})')
"
