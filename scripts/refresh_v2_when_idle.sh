#!/usr/bin/env bash
# Queue refreshed main or stress evaluations until genuinely idle GPUs are available.
set -u

SEEDS=(${SEEDS:-7 13 23})
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
JOB_KIND="${JOB_KIND:-main}"
MIN_FREE_MB="${MIN_FREE_MB:-60000}"
MAX_UTIL="${MAX_UTIL:-10}"
IDLE_CONFIRMATIONS="${IDLE_CONFIRMATIONS:-3}"
POLL_SECONDS="${POLL_SECONDS:-20}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
FORCE="${FORCE:-0}"
LOG_DIR="${LOG_DIR:-logs}"
PID_FILE="${PID_FILE:-${LOG_DIR}/coherence_refresh_scheduler.pid}"
STATUS_FILE="${STATUS_FILE:-${LOG_DIR}/coherence_refresh_scheduler.status}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-reports/v2_refreshed_main_seed}"
SUITE_DIR="${SUITE_DIR:-reports/v2_refreshed_suite}"
PRIMARY_REPORT="${PRIMARY_REPORT:-${OUTPUT_PREFIX}7/reliability_report.json}"
PY="${PY:-conda run --no-capture-output -n pytorch python}"

if [[ "${JOB_KIND}" != "main" && "${JOB_KIND}" != "stress" ]]; then
	echo "JOB_KIND must be main or stress, got: ${JOB_KIND}" >&2
	exit 2
fi

mkdir -p "${LOG_DIR}"
echo "$$" > "${PID_FILE}"

declare -A active_pid=()
declare -A active_seed=()
declare -A idle_count=()
pending=()
completed=()
failed=()
last_snapshot=""

report_is_current() {
	local seed="$1"
	local report="${OUTPUT_PREFIX}${seed}/reliability_report.json"
	[[ -f "${report}" ]] || return 1
	if command -v jq >/dev/null 2>&1; then
		if [[ "${JOB_KIND}" == "stress" ]]; then
			jq -e --argjson seed "${seed}" \
				'.args.seed == $seed and .results["fault_detection/test_id/sensor_delay@20"].auroc_coherence != null' \
				"${report}" >/dev/null 2>&1
		else
			jq -e '.results["fault_detection/test_id/sensor_delay"].auroc_coherence != null' "${report}" >/dev/null 2>&1
		fi
	else
		rg -q '"auroc_coherence"' "${report}"
	fi
}

for seed in "${SEEDS[@]}"; do
	if [[ "${FORCE}" != "1" ]] && report_is_current "${seed}"; then
		completed+=("seed${seed}:existing")
	else
		pending+=("${seed}")
	fi
done

gpu_snapshot() {
	nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
}

gpu_is_candidate() {
	local target="$1"
	local gpu
	for gpu in "${GPUS[@]}"; do
		[[ "${gpu}" == "${target}" ]] && return 0
	done
	return 1
}

refresh_idle_counts() {
	local index free_mb util
	last_snapshot="$(gpu_snapshot)"
	while IFS=',' read -r index free_mb util; do
		index="${index//[[:space:]]/}"
		free_mb="${free_mb//[[:space:]]/}"
		util="${util//[[:space:]]/}"
		gpu_is_candidate "${index}" || continue
		if [[ -n "${active_pid[$index]:-}" ]]; then
			idle_count[$index]=0
		elif [[ "${free_mb}" =~ ^[0-9]+$ && "${util}" =~ ^[0-9]+$ ]] \
			&& (( free_mb >= MIN_FREE_MB && util <= MAX_UTIL )); then
			idle_count[$index]=$(( ${idle_count[$index]:-0} + 1 ))
		else
			idle_count[$index]=0
		fi
	done <<< "${last_snapshot}"
}

active_count() {
	local count=0 gpu
	for gpu in "${GPUS[@]}"; do
		[[ -n "${active_pid[$gpu]:-}" ]] && count=$((count + 1))
	done
	echo "${count}"
}

write_status() {
	local tmp="${STATUS_FILE}.tmp"
	{
		echo "timestamp=$(date '+%F %T')"
		echo "scheduler_pid=$$"
		echo "job_kind=${JOB_KIND}"
		echo "pending=${pending[*]:-none}"
		echo "completed=${completed[*]:-none}"
		echo "failed=${failed[*]:-none}"
		echo "thresholds=free_mb>=${MIN_FREE_MB},util<=${MAX_UTIL},confirmations=${IDLE_CONFIRMATIONS}"
		echo "candidate_gpus=${GPUS[*]}"
		local gpu
		for gpu in "${GPUS[@]}"; do
			if [[ -n "${active_pid[$gpu]:-}" ]]; then
				echo "active_gpu${gpu}=seed${active_seed[$gpu]},pid${active_pid[$gpu]}"
			else
				echo "idle_confirm_gpu${gpu}=${idle_count[$gpu]:-0}/${IDLE_CONFIRMATIONS}"
			fi
		done
		echo "gpu_snapshot=index,free_mb,util"
		printf '%s\n' "${last_snapshot}"
	} > "${tmp}"
	mv "${tmp}" "${STATUS_FILE}"
}

launch_seed() {
	local seed="$1"
	local gpu="$2"
	local log="${LOG_DIR}/${JOB_KIND}_refresh_seed${seed}_gpu${gpu}_$(date +%Y%m%d_%H%M%S).log"
	echo ">>> $(date '+%F %T') launch seed=${seed} gpu=${gpu} log=${log}"
	if [[ "${JOB_KIND}" == "stress" ]]; then
		(
			CUDA_VISIBLE_DEVICES="${gpu}" \
			STRESS_SEEDS="${seed}" STRESS_OUTPUT_PREFIX="${OUTPUT_PREFIX}" STRESS_SKIP_COMPLETED=0 \
			DEVICE=cuda PARALLEL=0 LOG_DIR="${LOG_DIR}" scripts/reliability_paper_v2.sh stress
		) > "${log}" 2>&1 &
	else
		(
			SEEDS="${seed}" GPUS="${gpu}" OUTPUT_PREFIX="${OUTPUT_PREFIX}" \
				scripts/refresh_v2_evaluation.sh
		) > "${log}" 2>&1 &
	fi
	active_pid[$gpu]=$!
	active_seed[$gpu]="${seed}"
	idle_count[$gpu]=0
}

reap_jobs() {
	local gpu pid seed
	for gpu in "${GPUS[@]}"; do
		pid="${active_pid[$gpu]:-}"
		[[ -n "${pid}" ]] || continue
		if ! kill -0 "${pid}" 2>/dev/null; then
			seed="${active_seed[$gpu]}"
			if wait "${pid}"; then
				completed+=("seed${seed}:gpu${gpu}")
				echo "<<< $(date '+%F %T') complete seed=${seed} gpu=${gpu}"
			else
				failed+=("seed${seed}:gpu${gpu}")
				echo "<<< $(date '+%F %T') FAILED seed=${seed} gpu=${gpu}" >&2
			fi
			unset "active_pid[$gpu]"
			unset "active_seed[$gpu]"
		fi
	done
}

while (( ${#pending[@]} > 0 )) || (( $(active_count) > 0 )); do
	reap_jobs
	refresh_idle_counts
	while (( ${#pending[@]} > 0 )) && (( $(active_count) < MAX_PARALLEL )); do
		selected_gpu=""
		for gpu in "${GPUS[@]}"; do
			if [[ -z "${active_pid[$gpu]:-}" ]] && (( ${idle_count[$gpu]:-0} >= IDLE_CONFIRMATIONS )); then
				selected_gpu="${gpu}"
				break
			fi
		done
		[[ -n "${selected_gpu}" ]] || break
		seed="${pending[0]}"
		pending=("${pending[@]:1}")
		launch_seed "${seed}" "${selected_gpu}"
	done
	write_status
	if (( ${#pending[@]} > 0 )) || (( $(active_count) > 0 )); then
		sleep "${POLL_SECONDS}"
	fi
done

refresh_idle_counts
write_status
if (( ${#failed[@]} > 0 )); then
	echo "Refresh failed: ${failed[*]}" >&2
	exit 1
fi

reports=()
for report in reports/v2_*/reliability_report.json; do
	[[ -f "${report}" ]] && reports+=("${report}")
done
${PY} summarize_reliability_suite.py \
	--output-dir "${SUITE_DIR}" \
	--primary-report "${PRIMARY_REPORT}" \
	"${reports[@]}"
${PY} scripts/build_aaai_tables.py --suite-dir "${SUITE_DIR}"

completed+=("suite:rebuilt")
refresh_idle_counts
write_status
echo "All ${JOB_KIND} refreshed evaluations and suite aggregation completed."
