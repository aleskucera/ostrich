#!/usr/bin/env bash
# Apply the local newton viewer fixes to the third_party/newton submodule.
#
# The submodule points at upstream newton-physics/newton, so these edits cannot
# be committed and pushed with this repo -- only the commit pointer travels.
# Run this after `git submodule update`, or the GL viewer will crash on
# CPU-only machines. See docs/gl_viewer_gpu_contention.md.
#
# Idempotent: does nothing if the patch is already applied.
set -euo pipefail
cd "$(dirname "$0")/.."

PATCH="$PWD/newton_local_changes.patch"
cd third_party/newton

if git apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "newton patch already applied."
    exit 0
fi

if ! git apply --check "$PATCH" 2>/dev/null; then
    echo "ERROR: $PATCH does not apply cleanly to newton $(git rev-parse --short HEAD)." >&2
    echo "The submodule pointer probably moved. Rebase the patch by hand." >&2
    exit 1
fi

git apply "$PATCH"
echo "Applied newton patch: pinned-memory guard + USD prototype specifier."
