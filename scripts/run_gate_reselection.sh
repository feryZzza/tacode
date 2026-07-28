#!/usr/bin/env bash
# EXPERIMENTS_TODO §1.2：三 seed 的工作点重选（X_opp 目标 + rate-limit 网格）。
#
# 纯 eval：复用 reports/v2_main_seed{7,13,23}/reliability_tcn_best.pt，只重跑门控选择。
# 这里没有改 scripts/reliability_paper_v2.sh，因为那份脚本正在跑 §6 的训练波，
# 而且主结果的工作点必须由**这一次**记录在案的网格产生，不能混进旧模板。
#
# 网格按 §1.2：fall ∈ {0,0.0025,0.005,0.01,0.02,0.05}，rise ∈ {0,0.001,0.0025,0.005,0.01,0.02}。
# 含 0 是必要的：不含 0 就无法说明「加 rate limit 比不加好」。
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/home/zfy/miniconda3/envs/pytorch/bin/python}
DATA_ROOT=${DATA_ROOT:-/home/zfy/dataset/tcn/Parsed}
SEEDS=${SEEDS:-"7 13 23"}
GPUS=${GPUS:-"7"}
OUT_ROOT=${OUT_ROOT:-reports/aaai27_gate_reselection}
LOG_DIR=${LOG_DIR:-reports/aaai27_logs}
FALL_GRID=${FALL_GRID:-"0,0.0025,0.005,0.01,0.02,0.05"}
RISE_GRID=${RISE_GRID:-"0,0.001,0.0025,0.005,0.01,0.02"}
SOFTNESS_GRID=${SOFTNESS_GRID:-"0.25,0.5,1.0,2.0,4.0,8.0"}
GATE_VAL_FAULTS=${GATE_VAL_FAULTS:-"insole_missing,encoder_dropout,packet_loss,sensor_delay"}
CORE_FAULTS=${CORE_FAULTS:-"clean,insole_missing,encoder_dropout,imu_bias,packet_loss,packet_loss_burst,packet_loss_partial,stuck_imu,sensor_delay,sensor_delay_jitter"}
# Split/window settings are read from each archived main report below. Optional
# environment overrides exist only for an intentional, separately documented run.
HELDOUT_TASKS=${HELDOUT_TASKS:-}
MC_SAMPLES=${MC_SAMPLES:-}

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
mkdir -p "$OUT_ROOT" "$LOG_DIR"

gpu_arr=($GPUS)
pids=()
i=0
for seed in $SEEDS; do
	gpu=${gpu_arr[$((i % ${#gpu_arr[@]}))]}
	i=$((i + 1))
	ckpt="reports/v2_main_seed${seed}/reliability_tcn_best.pt"
	src="reports/v2_fc_main_seed${seed}/reliability_report.json"
	if [[ ! -f $ckpt || ! -f $src ]]; then
		echo "FAIL seed $seed：缺 $ckpt 或 $src" >&2
		exit 1
	fi
	out="$OUT_ROOT/seed${seed}"
	log="$LOG_DIR/gate_reselection_seed${seed}.log"
	# 训练期的超参从原 report 里读回，保证除门控选择以外一切同口径。
	read -r window stride profile side limit_trials report_heldout max_windows min_valid ignore_history report_mc val_count test_count val_people test_people training_mode batch_size amp < <("$PY" - "$src" <<'EOF'
import json, sys
a = json.loads(open(sys.argv[1]).read())["args"]
heldout = a.get("heldout_tasks", "")
if isinstance(heldout, (list, tuple)):
    heldout = ",".join(map(str, heldout))
else:
    heldout = ",".join(item.strip() for item in str(heldout).split(",") if item.strip())
val_people = a.get("val_participants", "") or "-"
test_people = a.get("test_participants", "") or "-"
print(
    a["window_size"],
    a["stride"],
    a["input_profile"],
    a["side"],
    a.get("limit_trials", 0),
    heldout,
    a.get("max_windows_per_trial", 4),
    a.get("min_valid_fraction", 0.5),
    a.get("eval_ignore_history", 248),
    a.get("mc_samples", 10) or 0,
    a.get("val_count", 2),
    a.get("test_count", 2),
    val_people,
    test_people,
    a.get("training_mode", "prob_aug_recon_fc"),
    a.get("batch_size", 32),
    a.get("amp", "bf16"),
)
EOF
	)
	heldout="${HELDOUT_TASKS:-$report_heldout}"
	mc_samples="${MC_SAMPLES:-$report_mc}"
	if [[ -z $heldout ]]; then
		echo "FAIL seed $seed：原报告未记录 heldout_tasks" >&2
		exit 1
	fi
	split_args=(--val-count "$val_count" --test-count "$test_count")
	[[ $val_people != "-" ]] && split_args+=(--val-participants "$val_people")
	[[ $test_people != "-" ]] && split_args+=(--test-participants "$test_people")
	echo "[seed $seed / gpu $gpu] reselect -> $out (window=$window profile=$profile heldout=$heldout)"
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
		echo "[seed $seed] reselect exit=$?"
		"$PY" summarize_reliability_report.py "$out/reliability_report.json" >> "$log" 2>&1 || true
	) &
	pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "RESELECTION DONE"
