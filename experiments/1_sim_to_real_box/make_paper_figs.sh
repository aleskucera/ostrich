#!/usr/bin/env bash
# Single source of truth for the box experiments' PAPER figures.
#
# Two figures, two canonical generators — do NOT cross them:
#   - box_sim_to_real.png  <- plot_paper_panels.py  (clean 3-panel: xy, z, bar)
#   - box_dt_stability.png / dt_stability_box.png
#                          <- 2_dt_stability_box/plot_dt_vs_error.py
#                             (the 0.2 m usable-threshold version with the red
#                              shaded over-threshold block and the "~Nx larger
#                              usable dt" annotation). plot_dt_vs_error.py
#                              self-installs BOTH filenames into the paper.
#
# IMPORTANT: plot_paper_panels.py ALSO emits a box_dt_stability.png (its make_fig2,
# a plainer 0.5 m-threshold variant). We deliberately do NOT install that one —
# the paper's dt figure must be the plot_dt_vs_error.py version. Likewise never
# install plot_results.py output (dev/diagnostic only).
#
# Usage:  bash experiments/1_sim_to_real_box/make_paper_figs.sh
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP2_DIR="$(cd "$EXP_DIR/../2_dt_stability_box" && pwd)"
SRC_DIR="$EXP_DIR/results/paper_panels"
# axion_paper is a SIBLING of the axion repo (../../../axion_paper from here).
PAPER_FIG_DIR="$(cd "$EXP_DIR/../../.." && pwd)/axion_paper/figures"

if [[ ! -d "$PAPER_FIG_DIR" ]]; then
  echo "ERROR: paper figures dir not found: $PAPER_FIG_DIR" >&2
  exit 1
fi

echo ">> regenerating sim-to-real panel (plot_paper_panels.py)..."
python "$EXP_DIR/plot_paper_panels.py"
cp "$SRC_DIR/box_sim_to_real.png" "$PAPER_FIG_DIR/box_sim_to_real.png"
echo ">> installed box_sim_to_real.png -> $PAPER_FIG_DIR/box_sim_to_real.png"

echo ">> regenerating dt-stability figure (plot_dt_vs_error.py: 0.2 m threshold + red block)..."
# plot_dt_vs_error.py self-installs box_dt_stability.png AND dt_stability_box.png
# straight into the paper figures dir, so no cp needed here.
python "$EXP2_DIR/plot_dt_vs_error.py"

echo ">> done."
