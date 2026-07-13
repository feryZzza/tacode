#!/usr/bin/env bash
# 可靠性论文实验套件 v2 —— 按 PAPER_DESIGN.md 的实验矩阵 E1–E10 组织。
#
# 用法（在 GPU 机器上）：
#   scripts/reliability_paper_v2.sh main           # 主结果：多 seed 概率+残差模型
#   scripts/reliability_paper_v2.sh ablations      # E5 归因消融：det_noaug/det_aug/prob_aug/prob_aug_recon
#   scripts/reliability_paper_v2.sh baseline       # E6 同划分自训确定性 TCN（无泄漏 baseline）
#   scripts/reliability_paper_v2.sh stress         # E9 故障强度曲线
#   scripts/reliability_paper_v2.sh action         # E10 action 输入消融
#   scripts/reliability_paper_v2.sh loso           # E7 留一被试交叉验证（重，最后跑）
#   scripts/reliability_paper_v2.sh summary        # 汇总所有报告
#   scripts/reliability_paper_v2.sh core           # main + ablations + baseline + stress + action + summary
#
# 每个 run 同时写终端（带进度条/周期行）并 tee 到 logs/ 下的时间戳日志，便于回看训练过程。
set -euo pipefail

TASK="${1:-core}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-20}"
DATA_ROOT="${DATA_ROOT:-/home/zfy/dataset/tcn/Parsed}"
SEEDS="${SEEDS:-7 13 23}"
ABLATION_SEED="${ABLATION_SEED:-7}"
LIMIT_TRIALS="${LIMIT_TRIALS:-0}"
MC_SAMPLES="${MC_SAMPLES:-10}"
CLEAN_SAMPLE_PROB="${CLEAN_SAMPLE_PROB:--1}"
FAULT_POSITIVE_WEIGHT="${FAULT_POSITIVE_WEIGHT:-0}"
FAULT_HEAD_MODE="${FAULT_HEAD_MODE:-binary}"
RECON_DETACH="${RECON_DETACH:-0}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
PROGRESS_STYLE="${PROGRESS_STYLE:-line}"
NUM_WORKERS="${NUM_WORKERS:-12}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
AMP="${AMP:-bf16}"
MAIN_OUTPUT_PREFIX="${MAIN_OUTPUT_PREFIX:-reports/v2_main_seed}"
STRESS_SEEDS="${STRESS_SEEDS:-${SEEDS}}"
STRESS_OUTPUT_PREFIX="${STRESS_OUTPUT_PREFIX:-reports/v2_refreshed_stress_seed}"
STRESS_SKIP_COMPLETED="${STRESS_SKIP_COMPLETED:-1}"

# ---------- 多卡并行调度 ----------
# PARALLEL=1 时把彼此独立的 run 分发到多张空闲 GPU 上并行跑；=0 退回串行（单卡）。
PARALLEL="${PARALLEL:-1}"
# 自动探测空闲 GPU 的显存阈值（MiB），低于此值视为空闲可用。
GPU_FREE_MEM_MB="${GPU_FREE_MEM_MB:-2000}"
# 低显存不等于空闲；共享机器上还要限制瞬时计算利用率。
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
# 自动模式找不到空闲卡时等待，而不是危险地回退到物理 GPU 0。
WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
# 显式指定可用 GPU 列表（空格分隔）可覆盖自动探测，例如 GPUS="1 6"。
GPUS="${GPUS:-}"

# 与原 Nature-capacity 一致的网络规模与对齐设置。
HELDOUT_TASKS="${HELDOUT_TASKS:-jump,cutting,lift_weight,lunges}"
# CORE includes held-out fault variants that are not in TRAIN_FAULTS, so Table-I/II
# can test whether the reliability monitor generalizes beyond the exact simulator.
CORE_FAULTS="${CORE_FAULTS:-clean,insole_missing,encoder_dropout,imu_bias,packet_loss,packet_loss_burst,packet_loss_partial,sensor_delay,sensor_delay_jitter}"
TRAIN_FAULTS="${TRAIN_FAULTS:-clean,insole_missing,encoder_dropout,imu_bias,packet_loss,stuck_imu,sensor_delay}"
GATE_VAL_FAULTS="${GATE_VAL_FAULTS:-insole_missing,encoder_dropout,packet_loss,sensor_delay}"
GATE_GRID="${GATE_GRID:-0.25,0.5,1.0,2.0,4.0,8.0}"
GATE_DEADBAND="${GATE_DEADBAND:-0.1}"
GATE_MAX_DEADBAND="${GATE_MAX_DEADBAND:-2.0}"
# OOD 域移 deadband：用 val_ood(验证被试×heldout任务) clean 风险的该分位设带宽，
# 让门控在 OOD-clean 上不误伤、故障仍抑制。<=0 关闭。0.9 经验值恢复 OOD-clean 保留力矩。
OOD_DEADBAND_Q="${OOD_DEADBAND_Q:-0.9}"
STRESS_FAULTS="${STRESS_FAULTS:-clean,packet_loss@0.05,packet_loss@0.15,packet_loss@0.30,packet_loss_burst@0.05,packet_loss_burst@0.15,packet_loss_partial@0.15,sensor_delay@5,sensor_delay@10,sensor_delay@20,sensor_delay_jitter@10,sensor_delay_jitter@20}"
# 用于 LOSO 的被试列表（可用 LOSO_SUBJECTS 覆盖以跑子集快速验证）。
LOSO_SUBJECTS="${LOSO_SUBJECTS:-BT01 BT02 BT03 BT06 BT07 BT08 BT09 BT10 BT11 BT12 BT13 BT14 BT15 BT16 BT17}"

PY="${PY:-conda run --no-capture-output -n pytorch python}"
mkdir -p "${LOG_DIR}"

# ---------- GPU 发现与并行调度 ----------
JOBS=()          # 待调度的 run 命令（每个元素是一条完整 shell 命令）
JOB_NAMES=()     # 对应的可读名字，仅用于日志
RESOLVED_GPUS="" # 首次调度时缓存，供训练队列和后续 eval 复用同一组物理 GPU。
EVAL_GPU=""
EVAL_DEVICE="${DEVICE}"

detect_gpus() {
	# 返回空闲 GPU 索引（显存占用低于阈值），或回显用户在 GPUS 中显式给定的列表。
	if [[ -n "${GPUS}" ]]; then
		echo "${GPUS}"; return
	fi
	if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
		echo "${CUDA_VISIBLE_DEVICES//,/ }"; return
	fi
	if ! command -v nvidia-smi >/dev/null 2>&1; then
		echo "0"; return
	fi
	local free=()
	while IFS=',' read -r idx used util; do
		idx="$(echo "${idx}" | tr -d ' ')"
		used="$(echo "${used}" | tr -d ' ')"
		util="$(echo "${util}" | tr -d ' ')"
		if [[ "${used}" =~ ^[0-9]+$ && "${util}" =~ ^[0-9]+$ ]] \
			&& (( used < GPU_FREE_MEM_MB && util <= GPU_MAX_UTIL )); then
			free+=("${idx}")
		fi
	done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
	if [[ "${#free[@]}" -eq 0 ]]; then
		return 1
	fi
	echo "${free[*]}"
}

resolve_gpus() {
	while [[ -z "${RESOLVED_GPUS}" ]]; do
		if RESOLVED_GPUS="$(detect_gpus)"; then
			break
		fi
		RESOLVED_GPUS=""
		if [[ "${WAIT_FOR_GPU}" != "1" ]]; then
			echo "No GPU meets GPU_FREE_MEM_MB=${GPU_FREE_MEM_MB} and GPU_MAX_UTIL=${GPU_MAX_UTIL}. Set GPUS explicitly or WAIT_FOR_GPU=1." >&2
			return 1
		fi
		echo "==> 暂无空闲 GPU（显存 < ${GPU_FREE_MEM_MB} MiB 且利用率 <= ${GPU_MAX_UTIL}%），${GPU_POLL_SECONDS}s 后重试" >&2
		sleep "${GPU_POLL_SECONDS}"
	done
}

prepare_eval_device() {
	EVAL_GPU=""
	EVAL_DEVICE="${DEVICE}"
	if [[ "${PARALLEL}" == "1" && "${DEVICE}" == cuda* ]]; then
		resolve_gpus
		local gpus=(${RESOLVED_GPUS})
		EVAL_GPU="${gpus[0]}"
		# CUDA_VISIBLE_DEVICES 把物理卡重映射为进程内的 cuda:0。
		EVAL_DEVICE="cuda"
		echo "==> 后处理评估复用物理 GPU ${EVAL_GPU} (进程内 ${EVAL_DEVICE})"
	fi
}

run_eval_python() {
	if [[ -n "${EVAL_GPU}" ]]; then
		CUDA_VISIBLE_DEVICES="${EVAL_GPU}" ${PY} "$@"
	else
		${PY} "$@"
	fi
}

enqueue() {
	# enqueue <name> <command-string>
	JOB_NAMES+=("$1")
	JOBS+=("$2")
}

run_queue() {
	# 把 JOBS 里的命令分发到空闲 GPU 上并行执行，每卡同一时刻一个 job。
	resolve_gpus
	local gpus=(${RESOLVED_GPUS})
	local ngpu=${#gpus[@]}
	echo "==> 调度 ${#JOBS[@]} 个 job 到 GPU: ${gpus[*]} (并发=${ngpu})"
	declare -A slot_pid     # gpu_index -> 当前运行的 pid
	local ji=0
	local njobs=${#JOBS[@]}
	while (( ji < njobs )); do
		local launched=0
		for g in "${gpus[@]}"; do
			(( ji < njobs )) || break
			local pid="${slot_pid[$g]:-}"
			if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
				echo ">>> [GPU ${g}] 启动: ${JOB_NAMES[$ji]}"
				CUDA_VISIBLE_DEVICES="${g}" bash -c "${JOBS[$ji]}" &
				slot_pid[$g]=$!
				ji=$(( ji + 1 ))
				launched=1
			fi
		done
		(( launched == 0 )) && sleep 5
	done
	wait
	echo "==> 全部 job 完成"
}

# ---------- 通用训练封装 ----------
# 参数: <output_dir> <input_profile> <training_mode> <seed> [extra args...]
# PARALLEL=1 时把命令推入队列由 run_queue 多卡分发；否则串行直接执行。
train_run() {
	local output_dir="$1"; shift
	local profile="$1"; shift
	local mode="$1"; shift
	local seed="$1"; shift
	local stamp; stamp="$(date +%Y%m%d_%H%M%S)"
	local log="${LOG_DIR}/$(basename "${output_dir}")_${stamp}.log"
	# 并行时每个 job 只看到一张卡（CUDA_VISIBLE_DEVICES 已重映射），故用 cuda:0。
	local dev="${DEVICE}"; [[ "${PARALLEL}" == "1" && "${DEVICE}" == cuda* ]] && dev="cuda"
	local recon_detach_flag="--no-recon-detach"
	[[ "${RECON_DETACH}" == "1" ]] && recon_detach_flag="--recon-detach"
	local extra_q=""; local a
	for a in "$@"; do extra_q+=" $(printf '%q' "$a")"; done
	local cmd
	cmd="${PY} run_reliability_experiment.py \
		--mode train --device ${dev} \
		--data-root $(printf '%q' "${DATA_ROOT}") \
		--output-dir $(printf '%q' "${output_dir}") \
		--input-profile ${profile} --training-mode ${mode} \
		--epochs ${EPOCHS} --batch-size ${BATCH_SIZE} \
		--num-workers ${NUM_WORKERS} --prefetch-factor ${PREFETCH_FACTOR} --amp ${AMP} \
		--window-size 768 --stride 384 \
		--max-windows-per-trial 4 --min-valid-fraction 0.5 \
		--heldout-tasks $(printf '%q' "${HELDOUT_TASKS}") \
		--limit-trials ${LIMIT_TRIALS} \
		--num-channels 80,80,80,80,80 --kernel-size 5 --dropout 0.15 --spatial-dropout \
		--fault-loss-weight 0.2 --fault-positive-weight ${FAULT_POSITIVE_WEIGHT} --fault-head-mode ${FAULT_HEAD_MODE} \
		--clean-sample-prob ${CLEAN_SAMPLE_PROB} \
		--recon-loss-weight 0.1 ${recon_detach_flag} --forecast-loss-weight 0.1 --forecast-detach \
		--training-faults $(printf '%q' "${TRAIN_FAULTS}") \
		--fault-scenarios $(printf '%q' "${CORE_FAULTS}") \
		--train-ignore-history 248 --eval-ignore-history 248 \
		--mc-samples ${MC_SAMPLES} \
		--select-gate-on-val --gate-softness 0.5 --gate-deadband $(printf '%q' "${GATE_DEADBAND}") \
		--gate-validation-faults $(printf '%q' "${GATE_VAL_FAULTS}") \
		--ood-deadband-quantile $(printf '%q' "${OOD_DEADBAND_Q}") \
		--gate-max-deadband $(printf '%q' "${GATE_MAX_DEADBAND}") \
		--gate-softness-grid $(printf '%q' "${GATE_GRID}") \
		--log-interval ${LOG_INTERVAL} --progress-style ${PROGRESS_STYLE} \
		--seed ${seed}${extra_q} > $(printf '%q' "${log}") 2>&1; \
		${PY} summarize_reliability_report.py $(printf '%q' "${output_dir}/reliability_report.json") || true"
	if [[ "${PARALLEL}" == "1" ]]; then
		enqueue "TRAIN ${output_dir} (profile=${profile} mode=${mode} seed=${seed}) -> ${log}" "${cmd}"
	else
		echo ">>> TRAIN ${output_dir} | profile=${profile} mode=${mode} seed=${seed} | log=${log}"
		bash -c "${cmd}"
	fi
}

# E1/E2/main：主结果，多 seed，概率+残差+MC。
# E1/E2/main：主结果，多 seed，概率+残差+预测+MC/ensemble。
run_main() {
	for seed in ${SEEDS}; do
		train_run "${MAIN_OUTPUT_PREFIX}${seed}" human prob_aug_recon_fc "${seed}"
	done
}

# E5：归因消融，固定 seed，逐一比较五种训练模式。
# prob_aug_recon (无预测头) 与 prob_aug_recon_fc (含预测头) 的对比直接隔离
# 预测头对 packet_loss/sensor_delay/imu_bias 时序故障检测的贡献。
run_ablations() {
	for mode in det_noaug det_aug prob_aug prob_aug_recon prob_aug_recon_fc; do
		train_run "reports/v2_ablation_${mode}_seed${ABLATION_SEED}" human "${mode}" "${ABLATION_SEED}"
	done
}

# E6：同划分自训确定性 TCN，作为无泄漏 baseline；与 main 使用相同 seeds 才能配对统计。
run_baseline() {
	for seed in ${SEEDS}; do
		train_run "reports/v2_baseline_det_seed${seed}" human det_noaug "${seed}"
	done
}

# E9：故障强度 stress 曲线，复用多 seed 主模型 checkpoint 做 eval。
stress_eval_run() {
	local seed="$1"
	local ckpt="reports/v2_main_seed${seed}/reliability_tcn_best.pt"
	local output_dir="${STRESS_OUTPUT_PREFIX}${seed}"
	local stamp; stamp="$(date +%Y%m%d_%H%M%S)"
	local log="${LOG_DIR}/$(basename "${output_dir}")_${stamp}.log"
	local dev="${DEVICE}"
	[[ "${PARALLEL}" == "1" && "${DEVICE}" == cuda* ]] && dev="cuda"
	if [[ ! -f "${ckpt}" ]]; then
		echo "Missing stress checkpoint: ${ckpt}" >&2
		return 1
	fi
	if [[ "${STRESS_SKIP_COMPLETED}" == "1" && -s "${output_dir}/reliability_report.json" ]]; then
		echo "==> Skip completed stress seed ${seed}: ${output_dir}/reliability_report.json"
		return 0
	fi
	local cmd
	cmd="${PY} run_reliability_experiment.py \
		--mode eval --device ${dev} \
		--data-root $(printf '%q' "${DATA_ROOT}") \
		--checkpoint $(printf '%q' "${ckpt}") \
		--output-dir $(printf '%q' "${output_dir}") \
		--input-profile human \
		--batch-size ${BATCH_SIZE} \
		--num-workers ${NUM_WORKERS} --prefetch-factor ${PREFETCH_FACTOR} --amp ${AMP} \
		--window-size 768 --stride 384 \
		--max-windows-per-trial 4 --min-valid-fraction 0.5 \
		--heldout-tasks $(printf '%q' "${HELDOUT_TASKS}") \
		--limit-trials ${LIMIT_TRIALS} \
		--fault-scenarios $(printf '%q' "${STRESS_FAULTS}") \
		--eval-ignore-history 248 --mc-samples ${MC_SAMPLES} \
		--select-gate-on-val --gate-softness 0.5 --gate-deadband $(printf '%q' "${GATE_DEADBAND}") \
		--gate-validation-faults $(printf '%q' "${GATE_VAL_FAULTS}") \
		--ood-deadband-quantile $(printf '%q' "${OOD_DEADBAND_Q}") \
		--gate-max-deadband $(printf '%q' "${GATE_MAX_DEADBAND}") \
		--gate-softness-grid $(printf '%q' "${GATE_GRID}") \
		--gate-sweep-scenarios test_id/clean,test_ood/clean,test_id/sensor_delay@20,test_ood/sensor_delay@20 \
		--log-interval ${LOG_INTERVAL} --progress-style ${PROGRESS_STYLE} \
		--seed ${seed} > $(printf '%q' "${log}") 2>&1 && \
		${PY} summarize_reliability_report.py $(printf '%q' "${output_dir}/reliability_report.json") >> $(printf '%q' "${log}") 2>&1"
	if [[ "${PARALLEL}" == "1" ]]; then
		enqueue "STRESS ${output_dir} (seed=${seed}) -> ${log}" "${cmd}"
	else
		echo ">>> STRESS ${output_dir} | seed=${seed} | ckpt=${ckpt} | log=${log}"
		bash -c "${cmd}"
	fi
}

run_stress() {
	local seed
	for seed in ${STRESS_SEEDS}; do
		stress_eval_run "${seed}"
	done
	flush_queue
}

# E2-ensemble：deep ensemble OOD 评估。以 seed7 为主模型，seed13/23 为集成成员，
# 跨模型预测方差作为 epistemic 不确定性（优于单模型 MC-dropout）。需先训完 main 三 seed。
run_ensemble() {
	local main_ckpt="reports/v2_main_seed7/reliability_tcn_best.pt"
	local ens_ckpts="reports/v2_main_seed13/reliability_tcn_best.pt,reports/v2_main_seed23/reliability_tcn_best.pt"
	if [[ ! -f "${main_ckpt}" ]]; then
		echo "WARNING ensemble: 主 checkpoint 缺失 (${main_ckpt})，跳过。先跑 main。" >&2
		return 0
	fi
	local stamp; stamp="$(date +%Y%m%d_%H%M%S)"
	local log="${LOG_DIR}/v2_ensemble_seed7_${stamp}.log"
	prepare_eval_device
	echo ">>> ENSEMBLE eval | main=${main_ckpt} | members=${ens_ckpts} | log=${log}"
	run_eval_python run_reliability_experiment.py \
		--mode eval \
		--device "${EVAL_DEVICE}" \
		--data-root "${DATA_ROOT}" \
		--checkpoint "${main_ckpt}" \
		--ensemble-checkpoints "${ens_ckpts}" \
		--output-dir reports/v2_ensemble_seed7 \
		--input-profile human \
		--batch-size "${BATCH_SIZE}" \
		--num-workers "${NUM_WORKERS}" --prefetch-factor "${PREFETCH_FACTOR}" --amp "${AMP}" \
		--window-size 768 --stride 384 \
		--max-windows-per-trial 4 --min-valid-fraction 0.5 \
		--heldout-tasks "${HELDOUT_TASKS}" \
		--limit-trials "${LIMIT_TRIALS}" \
		--fault-scenarios "${CORE_FAULTS}" \
		--eval-ignore-history 248 \
		--mc-samples 0 \
		--select-gate-on-val --gate-softness 0.5 --gate-deadband "${GATE_DEADBAND}" \
		--gate-validation-faults "${GATE_VAL_FAULTS}" \
		--ood-deadband-quantile "${OOD_DEADBAND_Q}" \
		--gate-max-deadband "${GATE_MAX_DEADBAND}" \
		--gate-softness-grid "${GATE_GRID}" \
		--log-interval "${LOG_INTERVAL}" --progress-style "${PROGRESS_STYLE}" \
		2>&1 | tee "${log}"
	${PY} summarize_reliability_report.py reports/v2_ensemble_seed7/reliability_report.json || true
}

# E10：action 输入消融。
run_action() {
	for profile in human_desired human_measured human_execution human_interaction; do
		train_run "reports/v2_action_${profile}_seed${ABLATION_SEED}" "${profile}" prob_aug_recon_fc "${ABLATION_SEED}"
	done
}

# E7：留一被试交叉验证。每折留 1 个被试做 test，前一个被试做 val。
# 复用 train_run（含多卡队列），通过 extra args 传入显式 val/test 被试。
run_loso() {
	local subjects=(${LOSO_SUBJECTS})
	local n=${#subjects[@]}
	for i in "${!subjects[@]}"; do
		local test_subj="${subjects[$i]}"
		local val_idx=$(( (i + n - 1) % n ))
		local val_subj="${subjects[$val_idx]}"
		train_run "reports/v2_loso_${test_subj}" human prob_aug_recon_fc "${ABLATION_SEED}" \
			--val-participants "${val_subj}" \
			--test-participants "${test_subj}"
	done
}

run_summary() {
	local reports=()
	local candidate
	for candidate in reports/v2_*/reliability_report.json; do
		[[ -f "${candidate}" ]] && reports+=("${candidate}")
	done
	if [[ "${#reports[@]}" -eq 0 ]]; then
		echo "No v2 reports found to summarize." >&2; exit 2
	fi
	${PY} summarize_reliability_suite.py \
		--output-dir reports/v2_paper_suite \
		--primary-report reports/v2_main_seed7/reliability_report.json \
		"${reports[@]}"
}

# 串行模式下 train_run 已即时执行，队列为空；并行模式下在此统一分发。
flush_queue() {
	if [[ "${PARALLEL}" == "1" && "${#JOBS[@]}" -gt 0 ]]; then
		run_queue
		JOBS=(); JOB_NAMES=()
	fi
}

case "${TASK}" in
	main)       run_main; flush_queue ;;
	ablations)  run_ablations; flush_queue ;;
	baseline)   run_baseline; flush_queue ;;
	stress)     run_stress ;;
	ensemble)   run_ensemble ;;
	action)     run_action; flush_queue ;;
	loso)       run_loso; flush_queue ;;
	summary)    run_summary ;;
	# core：训练类全部入队并行跑完，再做依赖主 checkpoint 的 stress/ensemble 与汇总。
	core)       run_main; run_ablations; run_baseline; run_action; flush_queue; run_stress; run_ensemble; run_summary ;;
	all)        run_main; run_ablations; run_baseline; run_action; run_loso; flush_queue; run_stress; run_ensemble; run_summary ;;
	*)
		echo "Usage: $0 {main|ablations|baseline|stress|ensemble|action|loso|summary|core|all}" >&2
		echo "Overrides: DEVICE=cuda EPOCHS=20 SEEDS='7 13 23' MC_SAMPLES=10 LIMIT_TRIALS=0" >&2
		echo "          PARALLEL=1 GPUS='1 6' GPU_FREE_MEM_MB=2000 GPU_MAX_UTIL=10 WAIT_FOR_GPU=1 GPU_POLL_SECONDS=30" >&2
		echo "          BATCH_SIZE=64 NUM_WORKERS=12 AMP=bf16" >&2
		echo "          CLEAN_SAMPLE_PROB=-1 FAULT_POSITIVE_WEIGHT=0 FAULT_HEAD_MODE=binary RECON_DETACH=0" >&2
		echo "          MAIN_OUTPUT_PREFIX=reports/v2_main_seed STRESS_SEEDS='7 13 23'" >&2
		exit 2 ;;
esac

echo "Done: ${TASK}"
