#!/usr/bin/env bash
# 过夜训练编排脚本 —— 容错 + 断点续跑 + 全程日志。
#
# 设计目标（过夜跑专用，区别于 reliability_paper_v2.sh）：
#   - 不用 set -e：单个 run 失败不会中断整夜，继续跑下一个。
#   - 断点续跑：已写出 reliability_report.json 的 run 自动跳过（FORCE=1 可强制重跑）。
#     —— 若一晚没跑完，第二晚再执行同一命令会接着没跑完的部分继续。
#   - 全程日志：主进度写 logs/overnight_<stamp>.log，每个 run 另写独立日志。
#   - 每个 run 记录耗时与退出码，结尾打印状态汇总并自动汇总报告。
#
# 推荐启动方式（断开 SSH 也不会中断）：
#   tmux new -s train
#   scripts/reliability_overnight.sh 2>&1 | tee logs/overnight_console.log
#   # 然后 Ctrl-b d 脱离；回来用 tmux attach -t train
# 或：
#   nohup scripts/reliability_overnight.sh > logs/overnight_console.log 2>&1 &
#   tail -f logs/overnight_console.log
#
# 常用覆盖：
#   DEVICE=cuda EPOCHS=20 SEEDS='7 13 23' MC_SAMPLES=10 RUN_LOSO=1 LIMIT_TRIALS=0
#   RUN_LOSO=0  跳过留一交叉（默认 1）；想先验证可 LIMIT_TRIALS=8 EPOCHS=3 MC_SAMPLES=4
set -uo pipefail

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-20}"
DATA_ROOT="${DATA_ROOT:-/home/zfy/dataset/tcn/Parsed}"
SEEDS="${SEEDS:-7 13 23}"
ABLATION_SEED="${ABLATION_SEED:-7}"
LIMIT_TRIALS="${LIMIT_TRIALS:-0}"
MC_SAMPLES="${MC_SAMPLES:-10}"
RUN_LOSO="${RUN_LOSO:-1}"
FORCE="${FORCE:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
PROGRESS_STYLE="${PROGRESS_STYLE:-line}"   # 过夜默认 line（日志友好，不刷屏）

HELDOUT_TASKS="${HELDOUT_TASKS:-jump,cutting,lift_weight,lunges}"
CORE_FAULTS="${CORE_FAULTS:-clean,insole_missing,encoder_dropout,imu_bias,packet_loss,sensor_delay}"
TRAIN_FAULTS="${TRAIN_FAULTS:-clean,insole_missing,encoder_dropout,imu_bias,packet_loss,stuck_imu,sensor_delay}"
GATE_GRID="${GATE_GRID:-0.25,0.5,1.0,2.0}"
GATE_DEADBAND="${GATE_DEADBAND:-0.1}"
STRESS_FAULTS="${STRESS_FAULTS:-clean,packet_loss@0.05,packet_loss@0.15,packet_loss@0.30,sensor_delay@5,sensor_delay@10,sensor_delay@20}"
LOSO_SUBJECTS="${LOSO_SUBJECTS:-BT01 BT02 BT03 BT06 BT07 BT08 BT09 BT10 BT11 BT12 BT13 BT14 BT15 BT16 BT17}"

PY="conda run -n pytorch python"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/overnight_${STAMP}.log"
MAIN_SEED7_CKPT="reports/v2_main_seed7/reliability_tcn_best.pt"

# 状态汇总累加器。
declare -a STATUS_LINES=()

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${MASTER_LOG}"; }

# 已完成判定：报告 JSON 存在且非空。
is_done() { [[ "${FORCE}" != "1" && -s "$1/reliability_report.json" ]]; }

# 运行一个训练 run，带容错与计时。参数: <output_dir> <profile> <mode> <seed> [extra...]
train_run() {
	local output_dir="$1"; shift
	local profile="$1"; shift
	local mode="$1"; shift
	local seed="$1"; shift
	local name; name="$(basename "${output_dir}")"
	if is_done "${output_dir}"; then
		log "SKIP  ${name} (report exists; FORCE=1 to rerun)"
		STATUS_LINES+=("SKIP    ${name}")
		return 0
	fi
	local run_log="${LOG_DIR}/${name}_${STAMP}.log"
	log "START ${name} | profile=${profile} mode=${mode} seed=${seed} -> ${run_log}"
	local t0 t1 rc
	t0=$(date +%s)
	${PY} run_reliability_experiment.py \
		--mode train --device "${DEVICE}" --data-root "${DATA_ROOT}" \
		--output-dir "${output_dir}" --input-profile "${profile}" --training-mode "${mode}" \
		--epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" \
		--window-size 768 --stride 384 --max-windows-per-trial 4 --min-valid-fraction 0.5 \
		--heldout-tasks "${HELDOUT_TASKS}" --limit-trials "${LIMIT_TRIALS}" \
		--num-channels 80,80,80,80,80 --kernel-size 5 --dropout 0.15 --spatial-dropout \
		--fault-loss-weight 0.2 --recon-loss-weight 0.1 \
		--training-faults "${TRAIN_FAULTS}" --fault-scenarios "${CORE_FAULTS}" \
		--train-ignore-history 248 --eval-ignore-history 248 \
		--mc-samples "${MC_SAMPLES}" --select-gate-on-val \
		--gate-softness 0.5 --gate-deadband "${GATE_DEADBAND}" --gate-softness-grid "${GATE_GRID}" \
		--log-interval "${LOG_INTERVAL}" --progress-style "${PROGRESS_STYLE}" \
		--seed "${seed}" "$@" > "${run_log}" 2>&1
	rc=$?
	t1=$(date +%s)
	local mins=$(( (t1 - t0) / 60 ))
	if [[ ${rc} -eq 0 ]]; then
		${PY} summarize_reliability_report.py "${output_dir}/reliability_report.json" >> "${run_log}" 2>&1 || true
		log "OK    ${name} (${mins} min)"
		STATUS_LINES+=("OK      ${name} (${mins} min)")
	else
		log "FAIL  ${name} (rc=${rc}, ${mins} min) -- see ${run_log}; 继续下一个"
		STATUS_LINES+=("FAIL    ${name} (rc=${rc})")
	fi
	return 0
}

# stress 评估（eval 模式，复用 seed7 主模型 checkpoint）。
stress_run() {
	local output_dir="reports/v2_stress"
	if is_done "${output_dir}"; then
		log "SKIP  v2_stress (report exists)"; STATUS_LINES+=("SKIP    v2_stress"); return 0
	fi
	if [[ ! -s "${MAIN_SEED7_CKPT}" ]]; then
		log "SKIP  v2_stress (缺少主模型 checkpoint ${MAIN_SEED7_CKPT})"
		STATUS_LINES+=("SKIP    v2_stress (no checkpoint)"); return 0
	fi
	local run_log="${LOG_DIR}/v2_stress_${STAMP}.log"
	log "START v2_stress | ckpt=${MAIN_SEED7_CKPT} -> ${run_log}"
	local t0 t1 rc; t0=$(date +%s)
	${PY} run_reliability_experiment.py \
		--mode eval --device "${DEVICE}" --data-root "${DATA_ROOT}" \
		--checkpoint "${MAIN_SEED7_CKPT}" --output-dir "${output_dir}" --input-profile human \
		--batch-size "${BATCH_SIZE}" --window-size 768 --stride 384 \
		--max-windows-per-trial 4 --min-valid-fraction 0.5 \
		--heldout-tasks "${HELDOUT_TASKS}" --limit-trials "${LIMIT_TRIALS}" \
		--fault-scenarios "${STRESS_FAULTS}" --eval-ignore-history 248 \
		--mc-samples "${MC_SAMPLES}" --gate-softness 0.5 --gate-deadband "${GATE_DEADBAND}" \
		--gate-softness-grid "${GATE_GRID}" \
		--gate-sweep-scenarios "test_id/clean,test_ood/clean,test_id/sensor_delay@20,test_ood/sensor_delay@20" \
		--log-interval "${LOG_INTERVAL}" --progress-style "${PROGRESS_STYLE}" \
		> "${run_log}" 2>&1
	rc=$?; t1=$(date +%s); local mins=$(( (t1 - t0) / 60 ))
	if [[ ${rc} -eq 0 ]]; then
		${PY} summarize_reliability_report.py "${output_dir}/reliability_report.json" >> "${run_log}" 2>&1 || true
		log "OK    v2_stress (${mins} min)"; STATUS_LINES+=("OK      v2_stress (${mins} min)")
	else
		log "FAIL  v2_stress (rc=${rc})"; STATUS_LINES+=("FAIL    v2_stress (rc=${rc})")
	fi
	return 0
}

# LOSO 一折（用显式 val/test 被试）。
loso_fold() {
	local test_subj="$1"; local val_subj="$2"
	local output_dir="reports/v2_loso_${test_subj}"
	if is_done "${output_dir}"; then
		log "SKIP  loso/${test_subj} (report exists)"; STATUS_LINES+=("SKIP    loso_${test_subj}"); return 0
	fi
	local run_log="${LOG_DIR}/v2_loso_${test_subj}_${STAMP}.log"
	log "START loso/${test_subj} | val=${val_subj} -> ${run_log}"
	local t0 t1 rc; t0=$(date +%s)
	${PY} run_reliability_experiment.py \
		--mode train --device "${DEVICE}" --data-root "${DATA_ROOT}" \
		--output-dir "${output_dir}" --input-profile human --training-mode prob_aug_recon \
		--epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" \
		--window-size 768 --stride 384 --max-windows-per-trial 4 --min-valid-fraction 0.5 \
		--heldout-tasks "${HELDOUT_TASKS}" \
		--val-participants "${val_subj}" --test-participants "${test_subj}" \
		--num-channels 80,80,80,80,80 --kernel-size 5 --dropout 0.15 --spatial-dropout \
		--fault-loss-weight 0.2 --recon-loss-weight 0.1 \
		--training-faults "${TRAIN_FAULTS}" --fault-scenarios "${CORE_FAULTS}" \
		--train-ignore-history 248 --eval-ignore-history 248 \
		--mc-samples "${MC_SAMPLES}" --select-gate-on-val \
		--gate-softness 0.5 --gate-deadband "${GATE_DEADBAND}" --gate-softness-grid "${GATE_GRID}" \
		--log-interval "${LOG_INTERVAL}" --progress-style "${PROGRESS_STYLE}" \
		--seed "${ABLATION_SEED}" > "${run_log}" 2>&1
	rc=$?; t1=$(date +%s); local mins=$(( (t1 - t0) / 60 ))
	if [[ ${rc} -eq 0 ]]; then
		${PY} summarize_reliability_report.py "${output_dir}/reliability_report.json" >> "${run_log}" 2>&1 || true
		log "OK    loso/${test_subj} (${mins} min)"; STATUS_LINES+=("OK      loso_${test_subj} (${mins} min)")
	else
		log "FAIL  loso/${test_subj} (rc=${rc})"; STATUS_LINES+=("FAIL    loso_${test_subj} (rc=${rc})")
	fi
	return 0
}

# ===================== 主流程 =====================
GLOBAL_T0=$(date +%s)
log "==== Overnight run start: device=${DEVICE} epochs=${EPOCHS} seeds='${SEEDS}' mc=${MC_SAMPLES} limit=${LIMIT_TRIALS} loso=${RUN_LOSO} ===="
log "Master log: ${MASTER_LOG}"

# 阶段 1：主结果（多 seed）。seed7 须最先完成，stress 依赖它的 checkpoint。
log "---- Stage 1/6: main (multi-seed) ----"
for seed in ${SEEDS}; do
	train_run "reports/v2_main_seed${seed}" human prob_aug_recon "${seed}"
done

# 阶段 2：归因消融（E5）。det_noaug 同时充当无泄漏 baseline（E6）。
log "---- Stage 2/6: attribution ablations ----"
for mode in det_noaug det_aug prob_aug prob_aug_recon; do
	train_run "reports/v2_ablation_${mode}_seed${ABLATION_SEED}" human "${mode}" "${ABLATION_SEED}"
done

# 阶段 3：显式无泄漏 baseline（与 ablation/det_noaug 等价，单独命名便于论文引用）。
log "---- Stage 3/6: leakage-free deterministic baseline ----"
train_run "reports/v2_baseline_det_seed${ABLATION_SEED}" human det_noaug "${ABLATION_SEED}"

# 阶段 4：故障强度曲线（E9）。
log "---- Stage 4/6: stress curves ----"
stress_run

# 阶段 5：action 输入消融（E10）。
log "---- Stage 5/6: action-input ablations ----"
for profile in human_desired human_measured human_execution human_interaction; do
	train_run "reports/v2_action_${profile}_seed${ABLATION_SEED}" "${profile}" prob_aug_recon "${ABLATION_SEED}"
done

# 阶段 6：留一被试交叉验证（E7，最重，可用 RUN_LOSO=0 跳过）。
if [[ "${RUN_LOSO}" == "1" ]]; then
	log "---- Stage 6/6: leave-one-subject-out ----"
	subjects=(${LOSO_SUBJECTS})
	n=${#subjects[@]}
	for i in "${!subjects[@]}"; do
		val_idx=$(( (i + n - 1) % n ))
		loso_fold "${subjects[$i]}" "${subjects[$val_idx]}"
	done
else
	log "---- Stage 6/6: LOSO skipped (RUN_LOSO=0) ----"
fi

# 汇总。
log "---- Summarizing all v2 reports ----"
reports=()
for candidate in reports/v2_*/reliability_report.json; do
	[[ -s "${candidate}" ]] && reports+=("${candidate}")
done
if [[ "${#reports[@]}" -gt 0 ]]; then
	${PY} summarize_reliability_suite.py \
		--output-dir reports/v2_paper_suite \
		--primary-report reports/v2_main_seed7/reliability_report.json \
		"${reports[@]}" >> "${MASTER_LOG}" 2>&1 \
		&& log "Summary written: reports/v2_paper_suite/" \
		|| log "Summary step failed -- check ${MASTER_LOG}"
else
	log "No completed reports to summarize."
fi

GLOBAL_T1=$(date +%s)
TOTAL_MIN=$(( (GLOBAL_T1 - GLOBAL_T0) / 60 ))
log "==== Overnight run done in ${TOTAL_MIN} min ===="
log "Status summary:"
for line in "${STATUS_LINES[@]}"; do log "  ${line}"; done
log "Key outputs: reports/v2_paper_suite/{core_aggregate,detection_aggregate,gate_policy,paper_gap_report}"
