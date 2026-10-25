# Builds the PDF figures for the theory document from figure_scripts/*.py.
#
# HOW TO USE (run from this directory, docs/deflation-study/latex-theory):
#   make figures            build every figure that is missing or out of date
#   make figures/NAME.pdf   build just one figure
#   make -n figures         dry run: print what would be rebuilt, run nothing
#   make -B figures         force-rebuild everything (e.g. after a torch upgrade)
#   make clean-figures      delete all generated figures
#
# HOW IT WORKS:
#   figure_scripts/NAME.py  ->  figures/NAME.pdf   (the name is the mapping)
#   make compares timestamps: a figure is rebuilt only if its script or
#   figure_scripts/_common.py is newer than the PDF. The PDFs are gitignored,
#   so on a fresh clone `make figures` builds all of them.
#   `uv run` uses the project's pinned environment (uv.lock), so it needs no
#   manual venv activation.
#
# ADDING A FIGURE:
#   1. Write figure_scripts/NAME.py ending in `save_fig(fig, __file__)`
#      (import it with `from _common import save_fig`).
#   2. In the .tex: \includegraphics{latex-theory/figures/NAME.pdf}
#   3. `make figures`. Nothing to edit in this file.
#
# CHECK PROVENANCE of a PDF:  pdfinfo figures/NAME.pdf | grep Subject
#   -> "<script> @ <git hash>"; a "+dirty" suffix means the script had
#   uncommitted changes when the figure was made.

SCRIPTS := $(wildcard figure_scripts/[!_]*.py)
FIGURES := $(patsubst figure_scripts/%.py,figures/%.pdf,$(SCRIPTS))

.PHONY: figures clean-figures

figures: $(FIGURES)

figures/%.pdf: figure_scripts/%.py figure_scripts/_common.py
	uv run python $<

clean-figures:
	rm -f $(FIGURES)
