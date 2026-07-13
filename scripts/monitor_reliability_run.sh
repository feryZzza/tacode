#!/usr/bin/env bash
set -u

PID_FILE="${PID_FILE:-logs/reliability_autorun.pid}"
WATCH_GPU="${WATCH_GPU:-2}"
INTERVAL="${INTERVAL:-60}"
RUN_NAME="${RUN_NAME:-reliability_paper_v2_core}"
TRAIN_LOG_PATTERN="${TRAIN_LOG_PATTERN:-v2_*.log}"
LOG="${LOG:-logs/reliability_monitor_$(date +%Y%m%d_%H%M%S).log}"
LATEST_STATUS="${LATEST_STATUS:-logs/reliability_monitor_latest.status}"
REPORT_DIR="${REPORT_DIR:-reports/v2_paper_suite}"
EXTRA_STATUS_FILE="${EXTRA_STATUS_FILE:-}"
MONITOR_PID_FILE="${MONITOR_PID_FILE:-}"

mkdir -p logs
if [[ -n "${MONITOR_PID_FILE}" ]]; then
	echo "$$" > "${MONITOR_PID_FILE}"
fi

read_pid() {
	if [[ -s "${PID_FILE}" ]]; then
		tr -d '[:space:]' < "${PID_FILE}"
	fi
}

latest_train_log() {
	find logs -maxdepth 1 -type f -name "${TRAIN_LOG_PATTERN}" -printf "%T@ %p\n" 2>/dev/null \
		| sort -nr \
		| awk 'NR == 1 {print $2}'
}

write_snapshot() {
	local pid="$1"
	local running="$2"
	local latest_log="$3"
	local ts
	ts="$(date '+%F %T')"
	{
		echo "==== ${ts} ===="
		echo "run=${RUN_NAME}"
		echo "pid=${pid:-none}"
		echo "running=${running}"
		echo "gpu=${WATCH_GPU}"
		echo
		if command -v nvidia-smi >/dev/null 2>&1; then
			echo "[gpu]"
			nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
				--format=csv,noheader,nounits 2>/dev/null \
				| awk -F, -v gpu="${WATCH_GPU}" '{
					gsub(/^ +| +$/, "", $1);
					if (gpu == "all" || $1 == gpu) print $0;
				}' || true
			echo
			echo "[gpu processes]"
			nvidia-smi pmon -c 1 -s um 2>/dev/null | awk -v gpu="${WATCH_GPU}" 'NR <= 2 || gpu == "all" || $1 == gpu {print}' || true
			echo
		else
			echo "nvidia-smi not found"
			echo
		fi
		echo "[latest train log]"
		if [[ -n "${latest_log}" && -f "${latest_log}" ]]; then
			echo "${latest_log}"
			tail -30 "${latest_log}"
		else
			echo "no training log found yet"
		fi
		echo
		if [[ -n "${EXTRA_STATUS_FILE}" && -f "${EXTRA_STATUS_FILE}" ]]; then
			echo "[scheduler status]"
			cat "${EXTRA_STATUS_FILE}"
			echo
		fi
		echo "[paper suite outputs]"
		find "${REPORT_DIR}" -maxdepth 1 -type f \
			\( -name "paper_gap_report.md" -o -name "*aggregate*.tsv" -o -name "gate_policy.tsv" \) \
			-printf "%TY-%Tm-%Td %TH:%TM %s %p\n" 2>/dev/null \
			| sort
		echo
	} | tee -a "${LOG}" > "${LATEST_STATUS}"
}

while true; do
	pid="$(read_pid)"
	running=0
	if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
		running=1
	fi
	write_snapshot "${pid}" "${running}" "$(latest_train_log)"
	if [[ "${running}" != "1" ]]; then
		exit 0
	fi
	sleep "${INTERVAL}"
done
