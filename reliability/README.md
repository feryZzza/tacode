# Reliability-Aware Moment Estimation Code

This folder contains the first implementation pass for the proposed follow-up
paper:

> Reliability-aware biological joint moment estimation for risk-sensitive
> task-agnostic exoskeleton assistance.

The code is intentionally separate from the original Nature 2024 reproduction
files (`example.py`, `tcn.py`, `dataloader.py`) so the baseline remains intact.

## 中文速览

- `features.py`: 定义 human/action 输入消融。
- `nature_dataset.py`: 扫描 Parsed 数据并切窗口。
- `faults.py`: 注入传感器故障，训练可靠性信号。
- `model.py`: 输出 moment、uncertainty、fault logit。
- `metrics.py`: 计算校准、OOD、故障检测和 torque 风险。

## What It Implements

- Parsed Nature 2024 dataset indexing from `/home/zfy/dataset/tcn/Parsed`.
- Windowed loading for `_exo.csv` plus `_moment_filt_bio.csv` labels.
- Valid-label filtering to avoid training on windows where OpenSim moment labels
  are missing.
- Input-profile ablations:
  - `human`
  - `human_desired`
  - `human_measured`
  - `human_execution`
  - `human_interaction`
  - `full`
- Synthetic deployment faults:
  - insole missing
  - encoder dropout
  - IMU bias drift
  - IMU scale error
  - packet loss
  - stuck IMU
  - sensor delay
- Probabilistic TCN outputs:
  - hip/knee moment mean
  - hip/knee log variance
  - sensor-fault logit
- Evaluation metrics:
  - RMSE, MAE, R2
  - uncertainty-error correlation
  - interval calibration and coverage ECE
  - OOD risk AUROC
  - clean-vs-fault detection AUROC
  - Nature-style torque replay risk metrics
- Optional official Nature TCN checkpoint replay when `--input-profile human`.

## Smoke Test

Use the PyTorch conda environment:

```bash
conda run -n pytorch python run_reliability_experiment.py \
  --mode smoke \
  --device cpu \
  --output-dir reports/reliability_smoke \
  --batch-size 2 \
  --window-size 128 \
  --stride 128 \
  --num-channels 8,8 \
  --stats-batches 2 \
  --min-valid-fraction 0.2
```

## First Real Run

```bash
conda run -n pytorch python run_reliability_experiment.py \
  --mode train \
  --device cuda \
  --output-dir reports/reliability_human \
  --input-profile human \
  --epochs 20 \
  --batch-size 16 \
  --window-size 768 \
  --stride 384 \
  --max-windows-per-trial 4 \
  --min-valid-fraction 0.5 \
  --heldout-tasks jump,cutting,lift_weight,lunges
```

For the actuator/action ablation, rerun with:

```bash
--input-profile human_desired
--input-profile human_measured
--input-profile human_execution
--input-profile human_interaction
```

## Main Output

The script writes:

- `reports/<run>/reliability_tcn_best.pt`
- `reports/<run>/reliability_report.json`

The JSON report is structured so tables can be made directly from keys such as:

- `test_id/clean`
- `test_ood/clean`
- `test_id/insole_missing`
- `fault_detection/test_id/insole_missing`
- `ood_detection/clean`
- `nature_baseline/test_id/clean`
