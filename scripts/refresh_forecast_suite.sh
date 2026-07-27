#!/usr/bin/env bash
# AAAI-27 EXPERIMENTS_TODO 任务 A/B/C/E/F/G/H 的统一执行器。
#
# 关键事实：所有 run 目录都留有 reliability_tcn_best.pt，因此 LOSO / action /
# ablation 全部退化为 eval-only——不需要任何重训。每个 job 用原 checkpoint 在修正后的
# forecast_residual 口径下重新评测，并补上 stuck_imu 场景。
#
# 用法：
#   GPUS="3 5" bash scripts/refresh_forecast_suite.sh main stress ablation action loso ensemble
#   GPUS="3 5" JOBS_PER_GPU=2 bash scripts/refresh_forecast_suite.sh all
set -uo pipefail

GPUS="${GPUS:-3 5}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
DATA_ROOT="${DATA_ROOT:-data/Parsed}"
LOG_DIR="${LOG_DIR:-logs/forecast_refresh}"
PY="${PY:-python3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-6}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
AMP="${AMP:-bf16}"
MC_SAMPLES="${MC_SAMPLES:-10}"
SEEDS="${SEEDS:-7 13 23}"
LOSO_SUBJECTS="${LOSO_SUBJECTS:-BT01 BT02 BT03 BT06 BT07 BT08 BT09 BT10 BT11 BT12 BT13 BT14 BT15 BT16 BT17}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
DRY_RUN="${DRY_RUN:-0}"

# 十个评测场景：六个训练期故障(含 stuck_imu) + 三个未见生成器 + clean。
FULL_FAULTS="clean,insole_missing,encoder_dropout,imu_bias,packet_loss,stuck_imu,packet_loss_burst,packet_loss_partial,sensor_delay,sensor_delay_jitter"
STRESS_FAULTS="clean,packet_loss@0.05,packet_loss@0.15,packet_loss@0.30,packet_loss_burst@0.05,packet_loss_burst@0.15,packet_loss_partial@0.15,sensor_delay@5,sensor_delay@10,sensor_delay@20,sensor_delay_jitter@10,sensor_delay_jitter@20"
GATE_VAL_FAULTS="insole_missing,encoder_dropout,packet_loss,sensor_delay"
GATE_GRID="0.25,0.5,1.0,2.0,4.0,8.0"
HELDOUT_TASKS="jump,cutting,lift_weight,lunges"

mkdir -p "${LOG_DIR}"

JOB_NAMES=()
JOB_CMDS=()

# eval_job <name> <checkpoint> <output_dir> <profile> <training_mode> <seed> <faults> [extra...]
eval_job() {
	local name="$1"; shift
	local ckpt="$1"; shift
	local out="$1"; shift
	local profile="$1"; shift
	local mode="$1"; shift
	local seed="$1"; shift
	local faults="$1"; shift
	if [[ ! -f "${ckpt}" ]]; then
		echo "SKIP ${name}: missing checkpoint ${ckpt}" >&2
		return 0
	fi
	if [[ "${SKIP_COMPLETED}" == "1" && -s "${out}/reliability_report.json" ]]; then
		echo "SKIP ${name}: ${out}/reliability_report.json already exists" >&2
		return 0
	fi
	# 本脚本会被 taskG 轮询器反复调用；同一个 output-dir 已有在跑的 eval 时不能再起一个，
	# 否则两个进程互相覆盖同一份报告和日志。
	if pgrep -f -- "--output-dir ${out} " >/dev/null 2>&1; then
		echo "SKIP ${name}: 已有进程在写 ${out}" >&2
		return 0
	fi
	local log="${LOG_DIR}/${name}.log"
	local extra=""; local a
	for a in "$@"; do extra+=" $(printf '%q' "$a")"; done
	# 评测期 gate sweep 场景要落在实际评测的场景里，否则 sweep 行为空。
	local sweep="test_id/clean,test_ood/clean,test_id/insole_missing,test_id/encoder_dropout,test_id/sensor_delay,test_ood/insole_missing,test_ood/encoder_dropout,test_ood/sensor_delay"
	[[ "${faults}" == "${STRESS_FAULTS}" ]] && sweep="test_id/clean,test_ood/clean,test_id/sensor_delay@20,test_ood/sensor_delay@20"
	local cmd
	cmd="${PY} run_reliability_experiment.py \
		--mode eval --device cuda \
		--data-root $(printf '%q' "${DATA_ROOT}") \
		--checkpoint $(printf '%q' "${ckpt}") \
		--output-dir $(printf '%q' "${out}") \
		--input-profile ${profile} --training-mode ${mode} \
		--batch-size ${BATCH_SIZE} \
		--num-workers ${NUM_WORKERS} --prefetch-factor ${PREFETCH_FACTOR} --amp ${AMP} \
		--window-size 768 --stride 384 \
		--max-windows-per-trial 4 --min-valid-fraction 0.5 \
		--heldout-tasks $(printf '%q' "${HELDOUT_TASKS}") \
		--limit-trials 0 \
		--fault-scenarios $(printf '%q' "${faults}") \
		--eval-ignore-history 248 --mc-samples ${MC_SAMPLES} \
		--select-gate-on-val --gate-softness 0.5 --gate-deadband 0.1 \
		--gate-validation-faults $(printf '%q' "${GATE_VAL_FAULTS}") \
		--ood-deadband-quantile 0.9 --gate-max-deadband 2.0 \
		--gate-softness-grid $(printf '%q' "${GATE_GRID}") \
		--gate-sweep-scenarios $(printf '%q' "${sweep}") \
		--log-interval 20 --progress-style line --seed ${seed}${extra} \
		> $(printf '%q' "${log}") 2>&1"
	JOB_NAMES+=("${name}")
	JOB_CMDS+=("${cmd}")
}

# --- 任务 A：主模型三 seed，补 stuck_imu ---
plan_main() {
	local seed
	for seed in ${SEEDS}; do
		eval_job "main_seed${seed}" \
			"reports/v2_main_seed${seed}/reliability_tcn_best.pt" \
			"reports/v2_fc_main_seed${seed}" human prob_aug_recon_fc "${seed}" "${FULL_FAULTS}"
	done
}

# --- 任务 B：stress 强度曲线三 seed ---
plan_stress() {
	local seed
	for seed in ${SEEDS}; do
		eval_job "stress_seed${seed}" \
			"reports/v2_main_seed${seed}/reliability_tcn_best.pt" \
			"reports/v2_fc_stress_seed${seed}" human prob_aug_recon_fc "${seed}" "${STRESS_FAULTS}"
	done
}

# --- 任务 C：五个消融模式（归因种子 7）。det_* / prob_aug* 无 forecast 头，但同样需要
# stuck_imu 列才能和主结果同口径，所以五个模式全部重评（都是 eval，成本低）。---
plan_ablation() {
	local mode
	for mode in det_noaug det_aug prob_aug prob_aug_recon prob_aug_recon_fc; do
		eval_job "ablation_${mode}" \
			"reports/v2_ablation_${mode}_seed7/reliability_tcn_best.pt" \
			"reports/v2_fc_ablation_${mode}_seed7" human "${mode}" 7 "${FULL_FAULTS}"
	done
}

# --- 任务 G：种子 13/23 的消融重训后评测。checkpoint 由 taskG 的训练队列产出，
# 训练未完成时 eval_job 会自己 SKIP，所以本函数可以反复调用直到全部补齐。---
plan_ablation_seeds() {
	local mode seed src
	for seed in 13 23; do
		for mode in det_noaug det_aug prob_aug prob_aug_recon prob_aug_recon_fc; do
			src="reports/v2_ablation_${mode}_seed${seed}"
			# best.pt 每个 epoch 都会被覆盖，所以不能只看它存在；训练收尾才写 comparison.tsv。
			if [[ "${TRAINED_ONLY:-0}" == "1" && ! -s "${src}/comparison.tsv" ]]; then
				echo "SKIP ablation_${mode}_seed${seed}: 训练尚未完成（无 comparison.tsv）" >&2
				continue
			fi
			eval_job "ablation_${mode}_seed${seed}" \
				"${src}/reliability_tcn_best.pt" \
				"reports/v2_fc_ablation_${mode}_seed${seed}" human "${mode}" "${seed}" "${FULL_FAULTS}"
		done
	done
}

# --- 任务 F：action-input 四个 profile ---
plan_action() {
	local profile
	for profile in human_desired human_measured human_execution human_interaction; do
		eval_job "action_${profile}" \
			"reports/v2_action_${profile}_seed7/reliability_tcn_best.pt" \
			"reports/v2_fc_action_${profile}_seed7" "${profile}" prob_aug_recon_fc 7 "${FULL_FAULTS}"
	done
}

# --- 任务 E：LOSO 15 折。每折的 val/test 被试沿用原报告的环形规则。---
plan_loso() {
	local subjects=(${LOSO_SUBJECTS})
	local n=${#subjects[@]}
	local i
	for i in "${!subjects[@]}"; do
		local test_subj="${subjects[$i]}"
		local val_idx=$(( (i + n - 1) % n ))
		local val_subj="${subjects[$val_idx]}"
		eval_job "loso_${test_subj}" \
			"reports/v2_loso_${test_subj}/reliability_tcn_best.pt" \
			"reports/v2_fc_loso_${test_subj}" human prob_aug_recon_fc 7 "${FULL_FAULTS}" \
			--val-participants "${val_subj}" --test-participants "${test_subj}"
	done
}

# --- 任务 H：deep ensemble（seed7 主模型 + seed13/23 成员）---
plan_ensemble() {
	eval_job "ensemble_seed7" \
		"reports/v2_main_seed7/reliability_tcn_best.pt" \
		"reports/v2_fc_ensemble_seed7" human prob_aug_recon_fc 7 "${FULL_FAULTS}" \
		--ensemble-checkpoints "reports/v2_main_seed13/reliability_tcn_best.pt,reports/v2_main_seed23/reliability_tcn_best.pt" \
		--mc-samples 0
}

# --- 三个 paired deterministic baseline：无 forecast 头，但要 stuck_imu 列 ---
plan_baseline() {
	local seed
	for seed in ${SEEDS}; do
		eval_job "baseline_det_seed${seed}" \
			"reports/v2_baseline_det_seed${seed}/reliability_tcn_best.pt" \
			"reports/v2_fc_baseline_det_seed${seed}" human det_noaug "${seed}" "${FULL_FAULTS}"
	done
}

TASKS=("$@")
[[ "${#TASKS[@]}" -eq 0 ]] && TASKS=(all)
for task in "${TASKS[@]}"; do
	case "${task}" in
		main)      plan_main ;;
		stress)    plan_stress ;;
		ablation)  plan_ablation ;;
		ablation_seeds) plan_ablation_seeds ;;
		action)    plan_action ;;
		loso)      plan_loso ;;
		ensemble)  plan_ensemble ;;
		baseline)  plan_baseline ;;
		all)       plan_main; plan_stress; plan_ablation; plan_baseline; plan_action; plan_loso; plan_ensemble ;;
		*) echo "Unknown task: ${task}" >&2; exit 2 ;;
	esac
done

echo "==> ${#JOB_CMDS[@]} eval job(s) planned on GPUs: ${GPUS} (x${JOBS_PER_GPU} per GPU)"
if [[ "${DRY_RUN}" == "1" ]]; then
	for i in "${!JOB_NAMES[@]}"; do echo "  [${i}] ${JOB_NAMES[$i]}"; done
	exit 0
fi
[[ "${#JOB_CMDS[@]}" -eq 0 ]] && { echo "Nothing to run."; exit 0; }

# 每个 GPU 开 JOBS_PER_GPU 个并发槽位；槽位空出就取下一个 job。
SLOTS=()
for g in ${GPUS}; do
	for ((k = 0; k < JOBS_PER_GPU; k++)); do SLOTS+=("${g}"); done
done

nslots=${#SLOTS[@]}
# 每个槽位一格：pid 为空串表示空闲。用定长数组而不是关联数组，避免 set -u 下
# 空关联数组展开被当成未绑定变量。
slot_pid=()
slot_job=()
for ((si = 0; si < nslots; si++)); do slot_pid+=(""); slot_job+=(""); done

FAILED=()
ji=0
njobs=${#JOB_CMDS[@]}
running=0
succeeded=0

reap() {
	local s
	for ((s = 0; s < nslots; s++)); do
		local pid="${slot_pid[$s]}"
		[[ -z "${pid}" ]] && continue
		if ! kill -0 "${pid}" 2>/dev/null; then
			if wait "${pid}"; then
				echo "    OK   ${slot_job[$s]}"
				succeeded=$(( succeeded + 1 ))
			else
				echo "    FAIL ${slot_job[$s]} (see ${LOG_DIR}/${slot_job[$s]}.log)" >&2
				FAILED+=("${slot_job[$s]}")
			fi
			slot_pid[$s]=""
			slot_job[$s]=""
			running=$(( running - 1 ))
		fi
	done
}

while (( ji < njobs || running > 0 )); do
	launched=0
	for ((si = 0; si < nslots; si++)); do
		(( ji < njobs )) || break
		[[ -n "${slot_pid[$si]}" ]] && continue
		gpu="${SLOTS[$si]}"
		echo ">>> [GPU ${gpu}] ($(( ji + 1 ))/${njobs}) ${JOB_NAMES[$ji]}"
		CUDA_VISIBLE_DEVICES="${gpu}" bash -c "${JOB_CMDS[$ji]}" &
		slot_pid[$si]=$!
		slot_job[$si]="${JOB_NAMES[$ji]}"
		running=$(( running + 1 ))
		ji=$(( ji + 1 ))
		launched=1
	done
	(( launched == 0 )) && sleep 10
	reap
done

echo "==> done: ${succeeded}/${njobs} succeeded"
if [[ "${#FAILED[@]}" -gt 0 ]]; then
	printf 'FAILED: %s\n' "${FAILED[*]}" >&2
	exit 1
fi
