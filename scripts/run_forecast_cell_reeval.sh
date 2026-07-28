#!/usr/bin/env bash
# EXPERIMENTS_TODO §6 第四格（minus forecasting）的纯 eval 补齐。
#
# 为什么需要这一步：`reports/v2_ablation_prob_aug_recon_seed*` 三个 seed 的模型早就训好了，
# 但那批 report 产生于 X_opp 指标落地之前，results 里只有 `*_wrong_direction_ratio`
# 与 `*_mean_abs_jerk`，没有 `*_wrong_energy` / `*_wrong_torque_product_integral`。
# 别名回落救不了它——键根本不存在。所以这里复用同一份 checkpoint 重跑一遍评测，
# 只为把 §6 要求的七项指标补全，**不重训**。
#
# 输出写到新目录 reports/v2_reverse_prob_aug_recon_seed<N>/，不覆盖原归档：
# 原 report 是旧口径的既有结果，改掉它会让历史数字不可复算。目录名沿用
# `v2_reverse_*` 前缀，这样 aggregate_reverse_ablation.py 会自动把它当训练侧格子收进去。
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/home/zfy/miniconda3/envs/pytorch/bin/python}
DATA_ROOT=${DATA_ROOT:-/home/zfy/dataset/tcn/Parsed}
SEEDS=${SEEDS:-"7 13 23"}
GPUS=${GPUS:-"7"}
SRC_PREFIX=${SRC_PREFIX:-reports/v2_ablation_prob_aug_recon_seed}
OUT_PREFIX=${OUT_PREFIX:-reports/v2_reverse_prob_aug_recon_seed}
LOG_DIR=${LOG_DIR:-reports/aaai27_logs}
CORE_FAULTS=${CORE_FAULTS:-"clean,insole_missing,encoder_dropout,imu_bias,packet_loss,packet_loss_burst,packet_loss_partial,stuck_imu,sensor_delay,sensor_delay_jitter"}
HELDOUT_TASKS=${HELDOUT_TASKS:-}

export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
mkdir -p "$LOG_DIR"

gpu_arr=($GPUS)
pids=()
i=0
for seed in $SEEDS; do
	gpu=${gpu_arr[$((i % ${#gpu_arr[@]}))]}
	i=$((i + 1))
	src="${SRC_PREFIX}${seed}"
	ckpt="$src/reliability_tcn_best.pt"
	if [[ ! -f $ckpt ]]; then
		echo "SKIP seed $seed：缺 $ckpt"
		continue
	fi
	out="${OUT_PREFIX}${seed}"
	log="$LOG_DIR/forecast_cell_seed${seed}.log"
	# 训练期超参与门控工作点全部从原 report 读回：这一格与其他六格的差异必须只有
	# 「有没有 forecast 头」，任何评测侧参数漂移都会让 Pareto 比较失去意义。
	read -r window stride profile side limit_trials report_heldout mc softness deadband fall rise < <("$PY" - "$src/reliability_report.json" <<'EOF'
import json, sys
report = json.loads(open(sys.argv[1]).read())
a = report["args"]
pol = report.get("results", {}).get("_gate_policy", {}) or {}
heldout = a.get("heldout_tasks", "")
if isinstance(heldout, (list, tuple)):
	heldout = ",".join(map(str, heldout))
else:
	heldout = ",".join(item.strip() for item in str(heldout).split(",") if item.strip())
print(
	a["window_size"], a["stride"], a["input_profile"], a["side"],
	a.get("limit_trials", 0), heldout, a.get("mc_samples", 0) or 0,
	pol.get("softness", a.get("gate_softness", 0.5)),
	pol.get("deadband", a.get("gate_deadband", 0.0)),
	pol.get("gate_max_fall_per_step", a.get("gate_max_fall_per_step", 0.0)),
	pol.get("gate_max_rise_per_step", a.get("gate_max_rise_per_step", 0.0)),
)
EOF
	)
	heldout="${HELDOUT_TASKS:-$report_heldout}"
	if [[ -z $heldout ]]; then
		echo "FAIL seed $seed：原报告未记录 heldout_tasks" >&2
		exit 1
	fi
	echo "[seed $seed / gpu $gpu] forecast cell -> $out (softness=$softness deadband=$deadband)"
	(
		CUDA_VISIBLE_DEVICES=$gpu "$PY" run_reliability_experiment.py \
			--mode eval --device cuda --data-root "$DATA_ROOT" \
			--output-dir "$out" --checkpoint "$ckpt" \
			--input-profile "$profile" --side "$side" \
			--window-size "$window" --stride "$stride" \
			--max-windows-per-trial 4 --min-valid-fraction 0.5 \
			--heldout-tasks "$heldout" --limit-trials "$limit_trials" \
			--fault-scenarios "$CORE_FAULTS" \
			--eval-ignore-history 248 --mc-samples "$mc" \
			--gate-softness "$softness" --gate-deadband "$deadband" \
			--gate-max-fall-per-step "$fall" --gate-max-rise-per-step "$rise" \
			--num-workers 6 --batch-size 32 \
			--log-interval 50 --progress-style off \
			--seed "$seed" > "$log" 2>&1
		echo "[seed $seed] forecast cell exit=$?"
	) &
	pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "FORECAST CELL DONE"
