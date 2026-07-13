#!/usr/bin/env bash
# Reproducibility entrypoint for the AAAI-oriented reliability paper.
#
# Usage:
#   scripts/reproduce_aaai.sh smoke    # quick CPU/GPU sanity run
#   scripts/reproduce_aaai.sh core     # main + ablations + baseline + action + stress + ensemble + summary
#   scripts/reproduce_aaai.sh all      # core + LOSO
#   scripts/reproduce_aaai.sh summary  # rebuild reports/v2_paper_suite from completed runs
#
# Common overrides:
#   DEVICE=cuda DATA_ROOT=/path/to/Parsed EPOCHS=20 SEEDS="7 13 23"
#   PARALLEL=1 GPUS="0 1" BATCH_SIZE=64 NUM_WORKERS=12 AMP=bf16
set -euo pipefail

TASK="${1:-smoke}"
DEVICE="${DEVICE:-cpu}"
DATA_ROOT="${DATA_ROOT:-/home/zfy/dataset/tcn/Parsed}"
PY="${PY:-conda run --no-capture-output -n pytorch python}"

case "${TASK}" in
	smoke)
		${PY} run_reliability_experiment.py \
			--mode smoke \
			--device "${DEVICE}" \
			--data-root "${DATA_ROOT}" \
			--output-dir reports/aaai_smoke \
			--input-profile human \
			--batch-size 2 \
			--num-workers 0 \
			--amp off \
			--window-size 128 \
			--stride 128 \
			--num-channels 8,8 \
			--stats-batches 1 \
			--limit-trials 2 \
			--max-eval-batches 1 \
			--min-valid-fraction 0.2 \
			--max-windows-per-trial 1 \
			--training-faults clean,packet_loss,sensor_delay \
			--fault-scenarios clean,packet_loss,packet_loss_burst,packet_loss_partial,sensor_delay,sensor_delay_jitter \
			--select-gate-on-val \
			--gate-validation-faults packet_loss,sensor_delay \
			--gate-softness-grid 0.5,1.0,2.0,4.0 \
			--progress-style line \
			--log-interval 1
		${PY} summarize_reliability_report.py reports/aaai_smoke/reliability_report.json || true
		;;
	core|all|summary|main|ablations|baseline|stress|ensemble|action|loso)
		DEVICE="${DEVICE}" DATA_ROOT="${DATA_ROOT}" scripts/reliability_paper_v2.sh "${TASK}"
		;;
	*)
		echo "Usage: $0 {smoke|core|all|summary|main|ablations|baseline|stress|ensemble|action|loso}" >&2
		exit 2
		;;
esac
