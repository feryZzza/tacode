#!/usr/bin/env bash
# 把 §6 反向消融的三个训练侧格子统一到主重选的门控协议下重新评测（纯 eval，不重训）。
#
# 为什么需要：这三格的归档 eval 用的门控选择协议彼此不同——
#   det_aug_recon_fc         三个 seed 都用旧目标 dimensionless_relative_wrong_v2，无速率网格
#   prob_aug_fc              seed7 用旧目标；seed13/23 用 x_opp 但只有 6 个候选（fall=rise=0）
#   prob_aug_recon_fc_nofault三个 seed 用 x_opp，同样只有 6 个候选
# 而主重选（aaai27_gate_reselection）与 prob_aug_recon 重评都是 216 候选、x_opp 目标、带速率限幅。
# 混在一张 Pareto 表里比较，等于把「组件差异」和「门控搜索空间差异」叠在一起，
# 正文「seven cells under one frozen gate policy」的说法就不成立。
#
# 复用 reports/v2_reverse_<mode>_seed<seed>/reliability_tcn_best.pt，window/stride/split/
# 故障集全部从原报告读回，只统一门控选择协议，与 run_gate_reselection.sh 及
# run_reverse_forecast_cell_reeval.sh 同口径。
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/home/zfy/miniconda3/envs/pytorch/bin/python}
DATA_ROOT=${DATA_ROOT:-/home/zfy/dataset/tcn/Parsed}
MODES=${MODES:-"det_aug_recon_fc prob_aug_fc prob_aug_recon_fc_nofault"}
SEEDS=${SEEDS:-"7 13 23"}
GPUS=${GPUS:-"4 5 6 7"}
OUT_ROOT=${OUT_ROOT:-reports/aaai27_reverse_cells}
LOG_DIR=${LOG_DIR:-reports/aaai27_logs}
SOFTNESS_GRID=${SOFTNESS_GRID:-"0.25,0.5,1.0,2.0,4.0,8.0"}
FALL_GRID=${FALL_GRID:-"0,0.0025,0.005,0.01,0.02,0.05"}
RISE_GRID=${RISE_GRID:-"0,0.001,0.0025,0.005,0.01,0.02"}
GATE_VAL_FAULTS=${GATE_VAL_FAULTS:-"insole_missing,encoder_dropout,packet_loss,sensor_delay"}
CORE_FAULTS=${CORE_FAULTS:-"clean,insole_missing,encoder_dropout,imu_bias,packet_loss,packet_loss_burst,packet_loss_partial,stuck_imu,sensor_delay,sensor_delay_jitter"}

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
mkdir -p "$OUT_ROOT" "$LOG_DIR"

gpu_arr=($GPUS)
pids=()
i=0
for mode in $MODES; do
	for seed in $SEEDS; do
		gpu=${gpu_arr[$((i % ${#gpu_arr[@]}))]}
		i=$((i + 1))
		src_dir="reports/v2_reverse_${mode}_seed${seed}"
		ckpt="$src_dir/reliability_tcn_best.pt"
		src="$src_dir/reliability_report.json"
		if [[ ! -f $ckpt || ! -f $src ]]; then
			echo "FAIL $mode seed $seed：缺 $ckpt 或 $src" >&2
			exit 1
		fi
		out="$OUT_ROOT/${mode}_seed${seed}"
		log="$LOG_DIR/reverse_cell_${mode}_seed${seed}.log"
		read -r window stride profile side limit_trials heldout max_windows min_valid ignore_history mc_samples val_count test_count val_people test_people training_mode batch_size amp < <("$PY" - "$src" <<'EOF'
import json, sys
a = json.loads(open(sys.argv[1]).read())["args"]
heldout = a.get("heldout_tasks", "")
if isinstance(heldout, (list, tuple)):
    heldout = ",".join(map(str, heldout))
else:
    heldout = ",".join(item.strip() for item in str(heldout).split(",") if item.strip())
print(
    a["window_size"], a["stride"], a["input_profile"], a["side"],
    a.get("limit_trials", 0), heldout,
    a.get("max_windows_per_trial", 4), a.get("min_valid_fraction", 0.5),
    a.get("eval_ignore_history", 248), a.get("mc_samples", 10) or 0,
    a.get("val_count", 2), a.get("test_count", 2),
    a.get("val_participants", "") or "-", a.get("test_participants", "") or "-",
    a.get("training_mode", ""),
    a.get("batch_size", 32), a.get("amp", "bf16"),
)
EOF
		)
		if [[ -z $heldout ]]; then
			echo "FAIL $mode seed $seed：原报告未记录 heldout_tasks" >&2
			exit 1
		fi
		split_args=(--val-count "$val_count" --test-count "$test_count")
		[[ $val_people != "-" ]] && split_args+=(--val-participants "$val_people")
		[[ $test_people != "-" ]] && split_args+=(--test-participants "$test_people")
		echo "[$mode seed $seed / gpu $gpu] reeval -> $out (heldout=$heldout mode=$training_mode)"
		(
			CUDA_VISIBLE_DEVICES=$gpu "$PY" run_reliability_experiment.py \
				--mode eval --device cuda --data-root "$DATA_ROOT" \
				--output-dir "$out" --checkpoint "$ckpt" \
				--input-profile "$profile" --side "$side" --training-mode "$training_mode" \
				--window-size "$window" --stride "$stride" \
				--max-windows-per-trial "$max_windows" --min-valid-fraction "$min_valid" \
				--heldout-tasks "$heldout" --limit-trials "$limit_trials" \
				"${split_args[@]}" \
				--fault-scenarios "$CORE_FAULTS" \
				--eval-ignore-history "$ignore_history" --mc-samples "$mc_samples" \
				--select-gate-on-val \
				--gate-selection-objective x_opp \
				--gate-softness 0.5 --gate-deadband 0.1 \
				--gate-softness-grid "$SOFTNESS_GRID" \
				--gate-max-fall-grid "$FALL_GRID" \
				--gate-max-rise-grid "$RISE_GRID" \
				--gate-validation-faults "$GATE_VAL_FAULTS" \
				--ood-deadband-quantile 0.9 --gate-max-deadband 2.0 \
				--num-workers 6 --batch-size "$batch_size" --amp "$amp" \
				--log-interval 50 --progress-style off \
				--seed "$seed" > "$log" 2>&1
			echo "[$mode seed $seed] reeval exit=$?"
		) &
		pids+=($!)
	done
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "REVERSE CELLS REEVAL DONE"
