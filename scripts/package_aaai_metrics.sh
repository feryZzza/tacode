#!/usr/bin/env bash
# 任务清单 §12：拷回前验收 + 打包（不含 .pt）。
#
# 验收项按 EXPERIMENTS_TODO.md §12 逐条检查，任何一条不过就退出非零，
# 避免把口径不一致的报告打进 tarball。
set -euo pipefail

PY="${PY:-/home/zfy/miniconda3/envs/pytorch/bin/python}"
SUITE_DIR="${SUITE_DIR:-reports/v2_fc_paper_suite}"
EXPECT_SEEDS="${EXPECT_SEEDS:-3}"

cd "$(dirname "$0")/.."

fail=0
note() { printf '%-6s %s\n' "$1" "$2"; }
check() { if [[ "$1" == "0" ]]; then note "PASS" "$2"; else note "FAIL" "$2"; fail=1; fi; }

echo "==== §12 验收 ===="

# 1. forecast_residual 单测必须在当前 commit 上通过。
if ${PY} -m unittest tests.test_forecast_residual > /tmp/aaai27_forecast_test.log 2>&1; then
	check 0 "tests.test_forecast_residual 通过"
else
	check 1 "tests.test_forecast_residual 失败（见 /tmp/aaai27_forecast_test.log）"
fi

# 2. 所有 prob_aug_recon_fc 报告都要晚于 forecast 修正的生效时间。
# 要判的是"修正后的代码内容从什么时候开始存在"，这个时刻既不等于 commit 时间
# 也不等于 mtime：修正先写后提交时 commit 时间偏晚（会把合法报告误判为过期），
# 新克隆的仓库里 mtime 是 checkout 时间又偏晚。两者取较早者在两种情形下都正确。
fix_commit="$(git log -1 --format=%H -- reliability/metrics.py)"
fix_epoch="$(git log -1 --format=%ct -- reliability/metrics.py)"
metrics_mtime="$(stat -c %Y reliability/metrics.py)"
(( metrics_mtime < fix_epoch )) && fix_epoch="${metrics_mtime}"
if ! git diff --quiet -- reliability/metrics.py; then
	note "NOTE" "reliability/metrics.py 有未提交改动；AAAI27_GIT_STATUS.txt 会记录脏树状态"
fi
stale=()
for report in reports/v2_fc_*/reliability_report.json; do
	[[ -s "${report}" ]] || continue
	mtime="$(stat -c %Y "${report}")"
	(( mtime < fix_epoch )) && stale+=("${report}")
done
check "$(( ${#stale[@]} > 0 ? 1 : 0 ))" \
	"v2_fc_* 报告均晚于 forecast 修正 ${fix_commit:0:8}（过期 ${#stale[@]} 份）"
((${#stale[@]})) && printf '       stale: %s\n' "${stale[@]}"

# 3. 三个主报告都要有 stuck_imu 的场景键和检测键。
${PY} - <<'EOF'
import json, sys
from pathlib import Path

missing = []
for seed in (7, 13, 23):
	path = Path(f"reports/v2_fc_main_seed{seed}/reliability_report.json")
	if not path.is_file():
		missing.append(f"{path} 不存在")
		continue
	results = json.loads(path.read_text(encoding="utf-8")).get("results", {})
	# 检测键与场景键都在 results 下，前者带 fault_detection/ 前缀。
	for split in ("test_id", "test_ood"):
		for key in (f"{split}/stuck_imu", f"fault_detection/{split}/stuck_imu"):
			if key not in results:
				missing.append(f"seed{seed} {key}")
print("\n".join(missing))
sys.exit(1 if missing else 0)
EOF
check "$?" "三个主报告含 test_{id,ood}/stuck_imu 与对应 detection key"

# 4. 汇总只吃 v2_fc_*：detection aggregate 的 n 列必须等于期望 seed 数。
if [[ -s "${SUITE_DIR}/main_detection_aggregate.tsv" ]]; then
	bad="$(awk -F'\t' -v want="${EXPECT_SEEDS}" '
		NR == 1 { for (i = 1; i <= NF; i++) if ($i == "n") c = i; next }
		c && $c != want { print $1 "/" $2 " n=" $c }
	' "${SUITE_DIR}/main_detection_aggregate.tsv")"
	check "$([[ -z "${bad}" ]] && echo 0 || echo 1)" "main_detection_aggregate.tsv 每行 n=${EXPECT_SEEDS}"
	[[ -n "${bad}" ]] && echo "${bad}" | head -10
else
	check 1 "缺 ${SUITE_DIR}/main_detection_aggregate.tsv（先跑 aggregate_forecast_suite.sh）"
fi

# 5. gap report 不再提示缺 action profile / stress seeds / paired baseline。
if [[ -s "${SUITE_DIR}/paper_gap_report.md" ]]; then
	blockers="$(grep -nE 'Action-input ablations are incomplete|Fault stress curves are missing|paired seed' \
		"${SUITE_DIR}/paper_gap_report.md" || true)"
	check "$([[ -z "${blockers}" ]] && echo 0 || echo 1)" "paper_gap_report.md 无证据缺口"
	[[ -n "${blockers}" ]] && echo "${blockers}"
else
	check 1 "缺 ${SUITE_DIR}/paper_gap_report.md"
fi

echo
if [[ "${fail}" != "0" ]]; then
	echo "验收未通过，不打包。" >&2
	exit 1
fi

echo "==== 打包 ===="
git rev-parse HEAD > reports/AAAI27_CODE_COMMIT.txt
git status --short > reports/AAAI27_GIT_STATUS.txt
find reports -path '*/reliability_report.json' -size +0 -print | sort \
	> reports/AAAI27_REPORT_MANIFEST.txt

find reports -type f \( \
	-path 'reports/v2_*/reliability_report.json' -o \
	-path "${SUITE_DIR}/*" -o \
	-name 'AAAI27_CODE_COMMIT.txt' -o \
	-name 'AAAI27_GIT_STATUS.txt' -o \
	-name 'AAAI27_REPORT_MANIFEST.txt' \
\) -print0 | tar --null -T - -czf aaai27_metrics.tar.gz
sha256sum aaai27_metrics.tar.gz > aaai27_metrics.tar.gz.sha256

echo "commit: $(cat reports/AAAI27_CODE_COMMIT.txt)"
echo "reports in manifest: $(wc -l < reports/AAAI27_REPORT_MANIFEST.txt)"
ls -lh aaai27_metrics.tar.gz
cat aaai27_metrics.tar.gz.sha256
