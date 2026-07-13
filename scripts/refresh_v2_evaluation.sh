#!/usr/bin/env bash
# Re-evaluate completed v2 checkpoints with the current metrics implementation.
# Original training reports are preserved under reports/v2_main_seed*.
set -euo pipefail

SEEDS=(${SEEDS:-7 13 23})
GPUS=(${GPUS:-3 4 5})
DATA_ROOT="${DATA_ROOT:-/home/zfy/dataset/tcn/Parsed}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-reports/v2_refreshed_main_seed}"
LOG_DIR="${LOG_DIR:-logs}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-12}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
AMP="${AMP:-bf16}"
MC_SAMPLES="${MC_SAMPLES:-10}"
NATURE_BASELINE_CHECKPOINT="${NATURE_BASELINE_CHECKPOINT:-}"
PY="${PY:-conda run --no-capture-output -n pytorch python}"

if [[ "${#GPUS[@]}" -eq 0 ]]; then
	echo "GPUS must contain at least one physical GPU index." >&2
	exit 2
fi

mkdir -p "${LOG_DIR}"
pids=()
names=()

for index in "${!SEEDS[@]}"; do
	seed="${SEEDS[$index]}"
	gpu="${GPUS[$((index % ${#GPUS[@]}))]}"
	checkpoint="reports/v2_main_seed${seed}/reliability_tcn_best.pt"
	output_dir="${OUTPUT_PREFIX}${seed}"
	log="${LOG_DIR}/$(basename "${output_dir}")_$(date +%Y%m%d_%H%M%S).log"
	if [[ ! -f "${checkpoint}" ]]; then
		echo "Missing checkpoint: ${checkpoint}" >&2
		exit 2
	fi
	echo ">>> [physical GPU ${gpu}] refresh seed=${seed} -> ${output_dir} | log=${log}"
	(
		CUDA_VISIBLE_DEVICES="${gpu}" ${PY} run_reliability_experiment.py \
			--mode eval --device cuda \
			--data-root "${DATA_ROOT}" \
			--checkpoint "${checkpoint}" \
			--nature-baseline-checkpoint "${NATURE_BASELINE_CHECKPOINT}" \
			--output-dir "${output_dir}" \
			--input-profile human --training-mode prob_aug_recon_fc \
			--batch-size "${BATCH_SIZE}" \
			--num-workers "${NUM_WORKERS}" --prefetch-factor "${PREFETCH_FACTOR}" --amp "${AMP}" \
			--window-size 768 --stride 384 \
			--max-windows-per-trial 4 --min-valid-fraction 0.5 \
			--heldout-tasks jump,cutting,lift_weight,lunges \
			--limit-trials 0 \
			--fault-scenarios clean,insole_missing,encoder_dropout,imu_bias,packet_loss,packet_loss_burst,packet_loss_partial,sensor_delay,sensor_delay_jitter \
			--eval-ignore-history 248 \
			--mc-samples "${MC_SAMPLES}" \
			--select-gate-on-val --gate-softness 0.5 --gate-deadband 0.1 \
			--gate-validation-faults insole_missing,encoder_dropout,packet_loss,sensor_delay \
			--ood-deadband-quantile 0.9 --gate-max-deadband 2.0 \
			--gate-softness-grid 0.25,0.5,1.0,2.0,4.0,8.0 \
			--log-interval 20 --progress-style line \
			--seed "${seed}" > "${log}" 2>&1
		${PY} summarize_reliability_report.py "${output_dir}/reliability_report.json"
		${PY} scripts/merge_refreshed_report.py \
			--source "reports/v2_main_seed${seed}/reliability_report.json" \
			--refreshed "${output_dir}/reliability_report.json"
	) &
	pids+=("$!")
	names+=("seed${seed}")
done

status=0
for index in "${!pids[@]}"; do
	if wait "${pids[$index]}"; then
		echo "OK: ${names[$index]}"
	else
		echo "FAIL: ${names[$index]}" >&2
		status=1
	fi
done
exit "${status}"
