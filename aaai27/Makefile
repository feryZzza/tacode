.PHONY: all clean figures

PYTHON ?= python3
FIGURE_SCRIPT := ../scripts/draw_aaai_figures.py
SUITE_DIR ?= ../reports/v2_fc_paper_suite
FIGURE_PDFS := \
	figures/fig_method_overview.pdf \
	figures/fig_detection_taxonomy.pdf \
	figures/fig_safety_utility.pdf \
	figures/fig_ablation_summary.pdf \
	figures/fig_stress_transfer.pdf

all: figures main.pdf

figures: $(FIGURE_PDFS)

$(FIGURE_PDFS) &: $(FIGURE_SCRIPT)
	MPLCONFIGDIR=/tmp/tacode_aaai_mpl $(PYTHON) $(FIGURE_SCRIPT) \
		--output-dir figures --suite-dir $(SUITE_DIR)

main.pdf: main.tex references.bib aaai2027.sty aaai2027.bst $(FIGURE_PDFS)
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	bibtex main
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex

clean:
	$(RM) main.aux main.bbl main.blg main.log main.out main.pdf
