#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-summary}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-20}"
DATA_ROOT="${DATA_ROOT:-/home/zfy/dataset/tcn/Parsed}"
SEEDS="${SEEDS:-7 13 23}"
ABLATION_SEED="${ABLATION_SEED:-7}"
LIMIT_TRIALS="${LIMIT_TRIALS:-0}"
CHECKPOINT="${CHECKPOINT:-reports/reliability_human_nature_capacity/reliability_tcn_best.pt}"
PAPER_DIR="${PAPER_DIR:-reports/reliability_paper_suite}"

HELDOUT_TASKS="${HELDOUT_TASKS:-jump,cutting,lift_weight,lunges}"
CORE_FAULTS="${CORE_FAULTS:-clean,insole_missing,encoder_dropout,imu_bias,packet_loss,sensor_delay}"
TRAIN_FAULTS="${TRAIN_FAULTS:-clean,insole_missing,encoder_dropout,imu_bias,packet_loss,stuck_imu,sensor_delay}"
GATE_GRID="${GATE_GRID:-0.25,0.5,1.0,2.0}"
STRESS_FAULTS="${STRESS_FAULTS:-clean,insole_missing,encoder_dropout,imu_bias,packet_loss@0.05,packet_loss@0.15,packet_loss@0.30,sensor_delay@5,sensor_delay@10,sensor_delay@20}"
STRESS_GATE_SCENARIOS="${STRESS_GATE_SCENARIOS:-test_id/clean,test_ood/clean,test_id/packet_loss@0.30,test_id/sensor_delay@20,test_ood/packet_loss@0.30,test_ood/sensor_delay@20}"

train_human_seed() {
	local seed="$1"
	local output_dir="reports/reliability_human_nature_capacity_seed${seed}"

	conda run -n pytorch python run_reliability_experiment.py \
		--mode train \
		--device "${DEVICE}" \
		--data-root "${DATA_ROOT}" \
		--output-dir "${output_dir}" \
		--input-profile human \
		--epochs "${EPOCHS}" \
		--batch-size "${BATCH_SIZE}" \
		--window-size 768 \
		--stride 384 \
		--max-windows-per-trial 4 \
		--min-valid-fraction 0.5 \
		--heldout-tasks "${HELDOUT_TASKS}" \
		--limit-trials "${LIMIT_TRIALS}" \
		--num-channels 80,80,80,80,80 \
		--kernel-size 5 \
		--dropout 0.15 \
		--spatial-dropout \
		--fault-loss-weight 0.2 \
		--training-faults "${TRAIN_FAULTS}" \
		--fault-scenarios "${CORE_FAULTS}" \
		--train-ignore-history 248 \
		--eval-ignore-history 248 \
		--gate-softness 0.5 \
		--gate-softness-grid "${GATE_GRID}" \
		--seed "${seed}"

	conda run -n pytorch python summarize_reliability_report.py \
		"${output_dir}/reliability_report.json"
}

run_seed_repeats() {
	for seed in ${SEEDS}; do
		train_human_seed "${seed}"
	done
}

run_stress_eval() {
	local output_dir="reports/reliability_human_stress"

	conda run -n pytorch python run_reliability_experiment.py \
		--mode eval \
		--device "${DEVICE}" \
		--data-root "${DATA_ROOT}" \
		--checkpoint "${CHECKPOINT}" \
		--output-dir "${output_dir}" \
		--input-profile human \
		--batch-size "${BATCH_SIZE}" \
		--window-size 768 \
		--stride 384 \
		--max-windows-per-trial 4 \
		--min-valid-fraction 0.5 \
		--heldout-tasks "${HELDOUT_TASKS}" \
		--limit-trials "${LIMIT_TRIALS}" \
		--fault-scenarios "${STRESS_FAULTS}" \
		--eval-ignore-history 248 \
		--gate-softness 0.5 \
		--gate-softness-grid "${GATE_GRID}" \
		--gate-sweep-scenarios "${STRESS_GATE_SCENARIOS}"

	conda run -n pytorch python summarize_reliability_report.py \
		"${output_dir}/reliability_report.json"
}

run_action_ablations() {
	for profile in human_desired human_measured human_execution human_interaction; do
		local output_dir="reports/reliability_ablation_${profile}_seed${ABLATION_SEED}"

		conda run -n pytorch python run_reliability_experiment.py \
			--mode train \
			--device "${DEVICE}" \
			--data-root "${DATA_ROOT}" \
			--output-dir "${output_dir}" \
			--input-profile "${profile}" \
			--epochs "${EPOCHS}" \
			--batch-size "${BATCH_SIZE}" \
			--window-size 768 \
			--stride 384 \
			--max-windows-per-trial 4 \
			--min-valid-fraction 0.5 \
			--heldout-tasks "${HELDOUT_TASKS}" \
			--limit-trials "${LIMIT_TRIALS}" \
			--num-channels 80,80,80,80,80 \
			--kernel-size 5 \
			--dropout 0.15 \
			--spatial-dropout \
			--fault-loss-weight 0.2 \
			--training-faults "${TRAIN_FAULTS}" \
			--fault-scenarios "${CORE_FAULTS}" \
			--train-ignore-history 248 \
			--eval-ignore-history 248 \
			--gate-softness 0.5 \
			--gate-softness-grid "${GATE_GRID}" \
			--seed "${ABLATION_SEED}"

		conda run -n pytorch python summarize_reliability_report.py \
			"${output_dir}/reliability_report.json"
	done
}

run_suite_summary() {
	local reports=()
	local candidate
	for candidate in \
		reports/reliability_human_nature_capacity/reliability_report.json \
		reports/reliability_human_nature_capacity_seed*/reliability_report.json \
		reports/reliability_human_stress/reliability_report.json \
		reports/reliability_ablation_*/reliability_report.json; do
		if [[ -f "${candidate}" ]]; then
			reports+=("${candidate}")
		fi
	done

	if [[ "${#reports[@]}" -eq 0 ]]; then
		echo "No reliability reports found to summarize." >&2
		exit 2
	fi

	conda run -n pytorch python summarize_reliability_suite.py \
		--output-dir "${PAPER_DIR}" \
		"${reports[@]}"
}

case "${TASK}" in
	seed-repeats)
		run_seed_repeats
		;;
	stress)
		run_stress_eval
		;;
	action-ablations)
		run_action_ablations
		;;
	summary)
		run_suite_summary
		;;
	all)
		run_seed_repeats
		run_stress_eval
		run_action_ablations
		run_suite_summary
		;;
	*)
		echo "Usage: $0 {seed-repeats|stress|action-ablations|summary|all}" >&2
		echo "Useful overrides: DEVICE=cuda EPOCHS=20 SEEDS='7 13 23' LIMIT_TRIALS=0" >&2
		exit 2
		;;
esac
