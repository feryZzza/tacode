#!/usr/bin/env bash
# 任务 D：把 v2_fc_* 报告汇总成论文表格 + gap report + LOSO 置信区间。
#
# 只吃 reports/v2_fc_*/reliability_report.json（修正后 forecast 口径 + 含 stuck_imu），
# 旧的 v2_* 报告一律不进汇总，避免两套口径被当成同一批样本平均。
set -euo pipefail

SUITE_DIR="${SUITE_DIR:-reports/v2_fc_paper_suite}"
PY="${PY:-python3}"
PRIMARY="${PRIMARY:-reports/v2_fc_main_seed7/reliability_report.json}"

cd "$(dirname "$0")/.."

reports=()
for candidate in reports/v2_fc_*/reliability_report.json; do
	[[ -s "${candidate}" ]] && reports+=("${candidate}")
done
if [[ "${#reports[@]}" -eq 0 ]]; then
	echo "No reports/v2_fc_*/reliability_report.json found." >&2
	exit 2
fi
echo "==> aggregating ${#reports[@]} report(s) into ${SUITE_DIR}"

# CORE_SCENARIOS 已含 stuck_imu，无需 --scenarios 覆盖。
${PY} summarize_reliability_suite.py \
	--output-dir "${SUITE_DIR}" \
	--primary-report "${PRIMARY}" \
	"${reports[@]}"

${PY} scripts/build_aaai_tables.py \
	--suite-dir "${SUITE_DIR}" \
	--output-dir "${SUITE_DIR}"

if compgen -G "reports/v2_fc_loso_*/reliability_report.json" > /dev/null; then
	${PY} scripts/analyze_loso_effect.py \
		--glob "reports/v2_fc_loso_*/reliability_report.json" \
		--output "${SUITE_DIR}/loso_effect.json" \
		| tee "${SUITE_DIR}/loso_effect.txt"
fi

# 汇总 TSV 不含 NLL / coverage ECE，且 macro 必须先在 seed 内平均；这一步补上。
${PY} scripts/extract_paper_numbers.py \
	--output "${SUITE_DIR}/paper_numbers.json" > /dev/null

# 图的数据常量直接读 SUITE_DIR，重新出图即同步正文数字。
MPLCONFIGDIR=/tmp/tacode_aaai_mpl ${PY} scripts/draw_aaai_figures.py \
	--output-dir aaai27/figures --suite-dir "${SUITE_DIR}"

echo
echo "==== paper_gap_report.md ===="
sed -n '1,120p' "${SUITE_DIR}/paper_gap_report.md"
