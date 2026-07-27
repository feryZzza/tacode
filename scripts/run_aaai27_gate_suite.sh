#!/usr/bin/env bash
# EXPERIMENTS_TODO §2/§3/§4/§5 的全量派发：三 seed × 四个 stage，按 GPU 分流。
#
# 四个 stage 都是纯 eval（复用已有 checkpoint，不重训），彼此独立，所以可以并行。
# 每个 GPU 上串行跑一个 seed 的全部 stage，避免同一张卡上四个进程抢显存。
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/home/zfy/miniconda3/envs/pytorch/bin/python}
DATA_ROOT=${DATA_ROOT:-/home/zfy/dataset/tcn/Parsed}
REPORTS=${REPORTS:-reports}
SEEDS=${SEEDS:-"7 13 23"}
GPUS=${GPUS:-"4 5 6"}
DRAWS=${DRAWS:-1000}
BOOTSTRAP=${BOOTSTRAP:-10000}
WORKERS=${WORKERS:-6}
LOG_DIR=${LOG_DIR:-reports/aaai27_logs}

# 单卡多进程时的线程风暴防护：不设的话 256 核会被每个进程各开满一遍。
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

mkdir -p "$LOG_DIR"

run_seed() {
	local seed=$1 gpu=$2
	local report="$REPORTS/v2_fc_main_seed${seed}/reliability_report.json"
	local ckpt="$REPORTS/v2_main_seed${seed}/reliability_tcn_best.pt"
	if [[ ! -f $report || ! -f $ckpt ]]; then
		echo "SKIP seed $seed：缺 $report 或 $ckpt"
		return 1
	fi
	local common=(--report "$report" --checkpoint "$ckpt" --data-root "$DATA_ROOT"
		--splits test_id,test_ood --num-workers "$WORKERS" --device cuda)

	echo "[seed $seed / gpu $gpu] §2 gate baselines"
	CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_gate_baselines.py "${common[@]}" \
		--output-dir "$REPORTS/aaai27_gate_baselines" --draws "$DRAWS" \
		> "$LOG_DIR/gate_baselines_seed${seed}.log" 2>&1
	echo "[seed $seed / gpu $gpu] §2 exit=$?"

	echo "[seed $seed / gpu $gpu] §3 gate dynamics"
	CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_gate_dynamics.py "${common[@]}" \
		--output-dir "$REPORTS/aaai27_gate_dynamics" \
		> "$LOG_DIR/gate_dynamics_seed${seed}.log" 2>&1
	echo "[seed $seed / gpu $gpu] §3 exit=$?"

	echo "[seed $seed / gpu $gpu] §4 transient faults"
	CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_transient_faults.py "${common[@]}" \
		--output-dir "$REPORTS/aaai27_transient_faults" --dump-timesteps \
		> "$LOG_DIR/transient_faults_seed${seed}.log" 2>&1
	echo "[seed $seed / gpu $gpu] §4 exit=$?"
}

pids=()
i=0
gpu_arr=($GPUS)
for seed in $SEEDS; do
	gpu=${gpu_arr[$((i % ${#gpu_arr[@]}))]}
	run_seed "$seed" "$gpu" &
	pids+=($!)
	i=$((i + 1))
done
for pid in "${pids[@]}"; do wait "$pid"; done

# §5 一次跑完三个 seed（种子敏感性表要在同一个进程里汇总）。
EXTRA=""
for seed in 13 23; do
	rep="$REPORTS/v2_fc_main_seed${seed}/reliability_report.json"
	ckpt="$REPORTS/v2_main_seed${seed}/reliability_tcn_best.pt"
	[[ -f $rep && -f $ckpt ]] && EXTRA="${EXTRA:+$EXTRA,}${rep}:${ckpt}"
done
echo "[§5] paired analysis (extra runs: ${EXTRA:-none})"
CUDA_VISIBLE_DEVICES=${gpu_arr[0]} $PY scripts/run_paired_analysis.py \
	--report "$REPORTS/v2_fc_main_seed7/reliability_report.json" \
	--checkpoint "$REPORTS/v2_main_seed7/reliability_tcn_best.pt" \
	--extra-runs "$EXTRA" --data-root "$DATA_ROOT" \
	--output-dir "$REPORTS/aaai27_paired_analysis" \
	--bootstrap "$BOOTSTRAP" --num-workers "$WORKERS" \
	> "$LOG_DIR/paired_analysis.log" 2>&1
echo "[§5] exit=$?"
echo "ALL DONE"
