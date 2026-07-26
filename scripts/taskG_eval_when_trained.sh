#!/usr/bin/env bash
# 任务 G 的 eval 触发器。
#
# 关键约束：run_reliability_experiment.py 在每个 epoch 结束就覆盖 reliability_tcn_best.pt，
# 所以「checkpoint 存在」不等于「训练完成」——直接按文件存在触发会评测到第 1 个 epoch 的权重。
# 训练脚本在收尾时写 comparison.tsv，用它作为完成标记。
set -uo pipefail
cd "$(dirname "$0")/.."
MODES="det_noaug det_aug prob_aug prob_aug_recon prob_aug_recon_fc"

for _ in $(seq 1 120); do
	ready=0
	for seed in 13 23; do
		for mode in ${MODES}; do
			src="reports/v2_ablation_${mode}_seed${seed}"
			# comparison.tsv 只在训练+自评测结束后才写出。
			[[ -s "${src}/comparison.tsv" ]] && ready=$((ready + 1))
		done
	done
	done_n=$(ls reports/v2_fc_ablation_*_seed13/reliability_report.json \
	            reports/v2_fc_ablation_*_seed23/reliability_report.json 2>/dev/null | wc -l)
	echo "$(date +%H:%M:%S) trained=${ready}/10 evaluated=${done_n}/10"
	[[ "${done_n}" -ge 10 ]] && { echo "所有 10 个消融 seed eval 完成"; exit 0; }
	if [[ "${ready}" -gt 0 ]]; then
		# eval_job 自己会跳过缺 checkpoint 和已完成的项；这里只把「训练已完成」的放进队列。
		TRAINED_ONLY=1 GPUS="${GPUS:-3 5}" JOBS_PER_GPU="${JOBS_PER_GPU:-2}" \
			bash scripts/refresh_forecast_suite.sh ablation_seeds >> logs/taskG_eval_poll.log 2>&1
	fi
	sleep 150
done
echo "轮询超时退出"
