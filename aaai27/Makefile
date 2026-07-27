.PHONY: all clean figures zh checklist supplement submission check-figures

PYTHON ?= python3
FIGURE_SCRIPT := ../scripts/draw_aaai_figures.py
SUITE_DIR ?= ../reports/v2_fc_paper_suite
# 逐 timestep 导出，图 1(c) 的数据来源。缺这个参数脚本会按 cwd 找默认路径，
# 从本目录运行时找不到，于是静默退回示意曲线（overview=schematic）。
OVERVIEW_NPZ ?= figures/data/timeseries_overview.npz
FIGURE_PDFS := \
	figures/fig_method_overview.pdf \
	figures/fig_detection_taxonomy.pdf \
	figures/fig_detection_full.pdf \
	figures/fig_safety_utility.pdf \
	figures/fig_ablation_summary.pdf \
	figures/fig_stress_transfer.pdf

all: figures main.pdf

submission: all checklist supplement

figures: $(FIGURE_PDFS)

# 出图脚本一次写全六张，用 grouped target 表达，避免并行 make 重复调用。
$(FIGURE_PDFS) &: $(FIGURE_SCRIPT) $(OVERVIEW_NPZ) $(SUITE_DIR)/paper_numbers.json
	MPLCONFIGDIR=/tmp/tacode_aaai_mpl $(PYTHON) $(FIGURE_SCRIPT) \
		--output-dir figures \
		--suite-dir $(SUITE_DIR) \
		--overview-timeseries $(OVERVIEW_NPZ)

# 出图前确认汇总目录和导出文件都在，否则脚本会退回内置常量或示意曲线，
# 且退出码仍是 0。这里主动失败，把问题暴露在编译之前。
check-figures:
	@test -f $(SUITE_DIR)/paper_numbers.json || { \
		echo "缺 $(SUITE_DIR)/paper_numbers.json：图里的数字会退回内置 FALLBACK 常量。"; \
		echo "先解开指标包： tar xzf ../aaai27_metrics.tar.gz -C .."; exit 1; }
	@test -f $(OVERVIEW_NPZ) || { \
		echo "缺 $(OVERVIEW_NPZ)：图 1(c) 会退回示意曲线，与 caption 声明的真实导出不符。"; \
		echo "重新导出： cd .. && $(PYTHON) scripts/export_timeseries.py --help"; exit 1; }
	@echo "汇总目录与逐 timestep 导出均在位。"

main.pdf: main.tex references.bib aaai2027.sty aaai2027.bst $(FIGURE_PDFS)
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	bibtex main
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex

checklist: ReproducibilityChecklist.pdf

ReproducibilityChecklist.pdf: ReproducibilityChecklist.tex aaai2027.sty
	pdflatex -interaction=nonstopmode -halt-on-error ReproducibilityChecklist.tex
	pdflatex -interaction=nonstopmode -halt-on-error ReproducibilityChecklist.tex

supplement: supplement.pdf

supplement.pdf: supplement.tex aaai2027.sty figures/fig_detection_full.pdf figures/fig_ablation_summary.pdf figures/fig_stress_transfer.pdf
	pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
	pdflatex -interaction=nonstopmode -halt-on-error supplement.tex

# 中文版用 ctexart，必须走 XeLaTeX；pdflatex 编不过。
zh: main_zh.pdf

main_zh.pdf: main_zh.tex references.bib aaai2027.bst $(FIGURE_PDFS)
	xelatex -interaction=nonstopmode -halt-on-error main_zh.tex
	bibtex main_zh
	xelatex -interaction=nonstopmode -halt-on-error main_zh.tex
	xelatex -interaction=nonstopmode -halt-on-error main_zh.tex

clean:
	$(RM) main.aux main.bbl main.blg main.log main.out main.pdf
	$(RM) main_zh.aux main_zh.bbl main_zh.blg main_zh.log main_zh.out main_zh.pdf
	$(RM) ReproducibilityChecklist.aux ReproducibilityChecklist.log ReproducibilityChecklist.out ReproducibilityChecklist.pdf
	$(RM) supplement.aux supplement.log supplement.out supplement.pdf
