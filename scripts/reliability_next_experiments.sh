#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-aligned-quick}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-20}"
LIMIT_TRIALS="${LIMIT_TRIALS:-16}"
DATA_ROOT="${DATA_ROOT:-/home/zfy/dataset/tcn/Parsed}"
FAULTS="clean,insole_missing,encoder_dropout,imu_bias,packet_loss,sensor_delay"
TRAIN_FAULTS="clean,insole_missing,encoder_dropout,imu_bias,packet_loss,stuck_imu,sensor_delay"
HELDOUT_TASKS="jump,cutting,lift_weight,lunges"
GATE_GRID="${GATE_GRID:-0.25,0.5,1.0,2.0}"

run_aligned_eval() {
	if [[ "${LIMIT_TRIALS}" == "0" ]]; then
		output_dir="reports/reliability_human_aligned"
	else
		output_dir="reports/reliability_human_aligned_quick"
	fi

	conda run -n pytorch python run_reliability_experiment.py \
		--mode eval \
		--device "${DEVICE}" \
		--data-root "${DATA_ROOT}" \
		--checkpoint reports/reliability_human/reliability_tcn_best.pt \
		--output-dir "${output_dir}" \
		--input-profile human \
		--batch-size "${BATCH_SIZE}" \
		--window-size 768 \
		--stride 384 \
		--max-windows-per-trial 4 \
		--min-valid-fraction 0.5 \
		--heldout-tasks "${HELDOUT_TASKS}" \
		--limit-trials "${LIMIT_TRIALS}" \
		--fault-scenarios "${FAULTS}" \
		--eval-ignore-history 248 \
		--gate-softness 0.5 \
		--gate-softness-grid "${GATE_GRID}"

	conda run -n pytorch python summarize_reliability_report.py \
		"${output_dir}/reliability_report.json"
}

run_capacity_train() {
	conda run -n pytorch python run_reliability_experiment.py \
		--mode train \
		--device "${DEVICE}" \
		--data-root "${DATA_ROOT}" \
		--output-dir reports/reliability_human_nature_capacity \
		--input-profile human \
		--epochs "${EPOCHS}" \
		--batch-size "${BATCH_SIZE}" \
		--window-size 768 \
		--stride 384 \
		--max-windows-per-trial 4 \
		--min-valid-fraction 0.5 \
		--heldout-tasks "${HELDOUT_TASKS}" \
		--num-channels 80,80,80,80,80 \
		--kernel-size 5 \
		--dropout 0.15 \
		--spatial-dropout \
		--fault-loss-weight 0.2 \
		--training-faults "${TRAIN_FAULTS}" \
		--fault-scenarios "${FAULTS}" \
		--train-ignore-history 248 \
		--eval-ignore-history 248 \
		--gate-softness 0.5 \
		--gate-softness-grid "${GATE_GRID}"

	conda run -n pytorch python summarize_reliability_report.py \
		reports/reliability_human_nature_capacity/reliability_report.json
}

run_action_ablations() {
	for profile in human_desired human_measured human_execution human_interaction; do
		conda run -n pytorch python run_reliability_experiment.py \
			--mode train \
			--device "${DEVICE}" \
			--data-root "${DATA_ROOT}" \
			--output-dir "reports/reliability_${profile}" \
			--input-profile "${profile}" \
			--epochs "${EPOCHS}" \
			--batch-size "${BATCH_SIZE}" \
			--window-size 768 \
			--stride 384 \
			--max-windows-per-trial 4 \
			--min-valid-fraction 0.5 \
			--heldout-tasks "${HELDOUT_TASKS}" \
			--fault-loss-weight 0.2 \
			--training-faults "${TRAIN_FAULTS}" \
			--fault-scenarios "${FAULTS}" \
			--train-ignore-history 90 \
			--eval-ignore-history 90 \
			--gate-softness 0.5 \
			--gate-softness-grid "${GATE_GRID}"

		conda run -n pytorch python summarize_reliability_report.py \
			"reports/reliability_${profile}/reliability_report.json"
	done
}

case "${TASK}" in
	aligned-quick|aligned-eval)
		run_aligned_eval
		;;
	capacity-train)
		run_capacity_train
		;;
	action-ablations)
		run_action_ablations
		;;
	all)
		run_aligned_eval
		run_capacity_train
		;;
	*)
		echo "Usage: $0 {aligned-quick|aligned-eval|capacity-train|action-ablations|all}" >&2
		echo "Set LIMIT_TRIALS=0 for a full aligned eval." >&2
		exit 2
		;;
esac
