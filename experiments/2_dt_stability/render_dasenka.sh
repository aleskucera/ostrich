#!/usr/bin/env bash
# Render a .blend on a specific GPU on the dasenka multi-GPU box.
#
# Pins Blender's Vulkan backend (used by EEVEE Next in Blender 5.x) to a single
# GPU via VK_DEVICE_INDEX so it doesn't fight other CUDA workloads on GPU 0.
#
# Usage:
#   ./render_dasenka.sh <file.blend> [gpu_index]            # render animation
#   ./render_dasenka.sh <file.blend> [gpu_index] -f 42      # single frame 42
#   ./render_dasenka.sh <file.blend> [gpu_index] -- ...     # extra blender args
#
# Defaults: gpu_index=1.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <file.blend> [gpu_index] [-- extra blender args]" >&2
    exit 1
fi

BLEND_FILE="$1"; shift
GPU_INDEX="${1:-1}"
if [[ $# -ge 1 && "$1" =~ ^[0-9]+$ ]]; then
    shift
fi

if [[ ! -f "$BLEND_FILE" ]]; then
    echo "error: blend file not found: $BLEND_FILE" >&2
    exit 1
fi

# If no extra args after gpu index, default to "render full animation".
EXTRA_ARGS=("$@")
if [[ ${#EXTRA_ARGS[@]} -eq 0 ]]; then
    EXTRA_ARGS=(-a)
fi

echo "Rendering $BLEND_FILE on GPU $GPU_INDEX (Vulkan backend)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --id="$GPU_INDEX" --query-gpu=name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader 2>/dev/null | sed 's/^/  GPU '"$GPU_INDEX"': /' || true
fi

VK_DEVICE_INDEX="$GPU_INDEX" CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
    blender --gpu-backend vulkan -b "$BLEND_FILE" "${EXTRA_ARGS[@]}"
