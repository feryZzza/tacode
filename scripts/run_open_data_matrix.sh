#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
STAGE=${1:-smoke}
SEEDS=${SEEDS:-"7 17 27"}
EPOCHS=${EPOCHS:-20}

common=(
  --feature-profile minimal
  --window-size 256
  --windows-per-trial 2
  --batch-size 32
  --epochs "$EPOCHS"
  --device cuda
)

if [[ "$STAGE" == "smoke" ]]; then
  "$PYTHON_BIN" train_open_data.py --mode smoke --protocol pooled --device cpu
  "$PYTHON_BIN" train_open_data.py --mode smoke --protocol lodo \
    --target-dataset camargo --adaptation-participants 2 \
    --target-supervision none --alignment dann --device cpu \
    --run-name smoke_uda_camargo
  exit 0
fi

if [[ "$STAGE" == "pooled" ]]; then
  for seed in $SEEDS; do
    "$PYTHON_BIN" train_open_data.py "${common[@]}" --protocol pooled \
      --alignment none --seed "$seed" --run-name "pooled_none_seed${seed}"
    "$PYTHON_BIN" train_open_data.py "${common[@]}" --protocol pooled \
      --alignment dann --seed "$seed" --run-name "pooled_dann_seed${seed}"
  done
  exit 0
fi

if [[ "$STAGE" == "transfer" ]]; then
  for target in nature dryad camargo; do
    for seed in $SEEDS; do
      "$PYTHON_BIN" train_open_data.py "${common[@]}" --protocol lodo \
        --target-dataset "$target" --adaptation-participants 0 \
        --alignment none --seed "$seed" --run-name "source_only_${target}_seed${seed}"
      "$PYTHON_BIN" train_open_data.py "${common[@]}" --protocol lodo \
        --target-dataset "$target" --adaptation-participants 4 \
        --target-supervision none --alignment dann --seed "$seed" \
        --run-name "uda_${target}_n4_seed${seed}"
      for people in 1 2 4; do
        "$PYTHON_BIN" train_open_data.py "${common[@]}" --protocol lodo \
          --target-dataset "$target" --adaptation-participants "$people" \
          --target-supervision labels --alignment dann --seed "$seed" \
          --run-name "fewshot_${target}_n${people}_seed${seed}"
      done
    done
  done
  exit 0
fi

if [[ "$STAGE" == "ablation" ]]; then
  for dropout in 0 0.15; do
    for alignment in none dann; do
      "$PYTHON_BIN" train_open_data.py "${common[@]}" --protocol pooled \
        --feature-profile hip_imu --sensor-dropout "$dropout" \
        --alignment "$alignment" \
        --run-name "ablation_hip_imu_${alignment}_drop${dropout}"
    done
  done
  exit 0
fi

echo "Unknown stage: $STAGE (choose smoke, pooled, transfer, or ablation)" >&2
exit 2
