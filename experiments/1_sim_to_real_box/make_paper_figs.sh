#!/usr/bin/env bash
# Single source of truth for the box experiment's PAPER figures.
#
# Regenerates the clean, paper-styled panels with plot_paper_panels.py and
# installs them into the paper tree. This is the ONLY supported way to update
# axion_paper/figures/box_sim_to_real.png — never hand-`cp` a figure there, and
# never install the output of plot_results.py (that script is dev/diagnostic
# only: it has per-panel titles, a suptitle, and verbose bar labels that do not
# belong in the paper).
#
# Usage:  bash experiments/1_sim_to_real_box/make_paper_figs.sh
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$EXP_DIR/results/paper_panels"
# axion_paper is a SIBLING of the axion repo (../../../axion_paper from here).
PAPER_FIG_DIR="$(cd "$EXP_DIR/../../.." && pwd)/axion_paper/figures"

if [[ ! -d "$PAPER_FIG_DIR" ]]; then
  echo "ERROR: paper figures dir not found: $PAPER_FIG_DIR" >&2
  exit 1
fi

echo ">> regenerating paper panels..."
python "$EXP_DIR/plot_paper_panels.py"

for fig in box_sim_to_real.png box_dt_stability.png; do
  if [[ -f "$SRC_DIR/$fig" ]]; then
    cp "$SRC_DIR/$fig" "$PAPER_FIG_DIR/$fig"
    echo ">> installed $fig -> $PAPER_FIG_DIR/$fig"
  fi
done

echo ">> done."
