#!/usr/bin/env bash
# Render the junior gradient figure on the dasenka multi-GPU box.
#
# Inspired by experiments/2_dt_stability/render_dasenka.sh: EEVEE Next in
# Blender 5.x renders through Vulkan, so we pin its backend to a single GPU via
# VK_DEVICE_INDEX (and CUDA_VISIBLE_DEVICES) so it doesn't fight CUDA workloads
# on GPU 0. The baked "view_*" cameras are driven by render_views.py.
#
# Usage:
#   ./render_junior_dasenka.sh [gpu_index]
#   ONLY=view_high_3q ./render_junior_dasenka.sh 1
#   SAMPLES=768 RES_X=3840 RES_Y=1620 OUTDIR=~/out ./render_junior_dasenka.sh 2
#
# Defaults: gpu_index=1, all views, SAMPLES=512, 3840x1620, OUTDIR=~/junior_out
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLEND="$HERE/junior_gradient_figure.blend"
PYSCRIPT="$HERE/render_views.py"
GPU_INDEX="${1:-1}"

[[ -f "$BLEND" ]]    || { echo "error: blend not found: $BLEND" >&2; exit 1; }
[[ -f "$PYSCRIPT" ]] || { echo "error: render script not found: $PYSCRIPT" >&2; exit 1; }

# Forwarded to render_views.py (override any via env before the call).
export OUTDIR="${OUTDIR:-$HOME/junior_out}"
export SAMPLES="${SAMPLES:-512}"
export RES_X="${RES_X:-3840}"
export RES_Y="${RES_Y:-1620}"
export ONLY="${ONLY:-}"

echo "Rendering $BLEND on GPU $GPU_INDEX (Vulkan / EEVEE Next) -> $OUTDIR"
echo "  samples=$SAMPLES  res=${RES_X}x${RES_Y}  only=${ONLY:-<all views>}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --id="$GPU_INDEX" --query-gpu=name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader 2>/dev/null | sed 's/^/  GPU '"$GPU_INDEX"': /' || true
fi

VK_DEVICE_INDEX="$GPU_INDEX" CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
    blender --gpu-backend vulkan -b "$BLEND" --python "$PYSCRIPT"
