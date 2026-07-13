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
- `model.py`: 输出 moment、uncertainty 和 per-fault logits。
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
  - one supervised logit per training fault plus an aggregate fault logit
- Evaluation metrics:
  - RMSE, MAE, R2
  - uncertainty-error correlation
  - interval calibration and coverage ECE
  - OOD risk AUROC
  - clean-vs-fault detection AUROC
  - causal cross-IMU coherence detection for fixed and jittered sensor delay
  - Nature-style torque replay risk metrics
- Optional official Nature TCN checkpoint replay when `--input-profile human`.
  This checkpoint is not split-matched and is a secondary reference only; the
  dedicated `det_noaug` runs are the primary fair baseline.

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

Training prints live progress by default. In a normal terminal it uses a
single-line progress bar; when output is captured by `conda run` or a log file,
it falls back to periodic progress lines. Useful controls:

```bash
--progress-style auto   # default: bar on TTY, line otherwise
--progress-style bar    # force a live progress bar
--progress-style line   # print periodic progress lines
--progress-style off    # disable live progress
--log-interval 10       # line-mode batch interval; 0 disables live progress
```

For the actuator/action ablation, rerun with:

```bash
--input-profile human_desired
--input-profile human_measured
--input-profile human_execution
--input-profile human_interaction
```

## Next Experiments

Use the helper script for the follow-up experiments:

```bash
# Quick re-evaluation of the existing checkpoint with Nature-style
# history masking and a gate-softness sweep.
scripts/reliability_next_experiments.sh aligned-quick

# Full aligned evaluation. This is slow on CPU because Parsed CSV files
# are re-read across multiple fault scenarios.
LIMIT_TRIALS=0 scripts/reliability_next_experiments.sh aligned-eval

# Train a Nature-capacity probabilistic TCN: 5 layers, 80 channels,
# kernel size 5, spatial dropout, and sensor-delay fault supervision.
scripts/reliability_next_experiments.sh capacity-train

# Train action/actuator input ablations.
scripts/reliability_next_experiments.sh action-ablations
```

Each run writes `summary.md`, `comparison.tsv`, and, when enabled,
`gate_sweep.tsv` next to the JSON report.

## Paper-Grade Experiment Suite

For a Q1 journal or A-conference submission, the current single best run is not
enough. Use the paper suite to add repeatability, ablations, stress curves, and
automatic gap reporting:

```bash
# Summarize all completed paper-suite reports.
scripts/reliability_paper_experiments.sh summary

# Repeat the main human-profile model over several seeds.
DEVICE=cuda SEEDS="7 13 23" scripts/reliability_paper_experiments.sh seed-repeats

# Evaluate fault-intensity curves such as packet_loss@0.30 and sensor_delay@20.
DEVICE=cuda scripts/reliability_paper_experiments.sh stress

# Train actuator/action input ablations.
DEVICE=cuda scripts/reliability_paper_experiments.sh action-ablations

# Run the full suite. This can be slow.
DEVICE=cuda scripts/reliability_paper_experiments.sh all
```

For the current AAAI-oriented paper route, prefer the reproducibility wrapper:

```bash
# Fast sanity check, including held-out packet-loss/delay variants.
scripts/reproduce_aaai.sh smoke

# Main paper suite without LOSO.
DEVICE=cuda scripts/reproduce_aaai.sh core

# Full suite including LOSO.
DEVICE=cuda scripts/reproduce_aaai.sh all

# Rebuild aggregate tables from completed v2 runs.
scripts/reproduce_aaai.sh summary
```

The current v2 entrypoint is `scripts/reliability_paper_v2.sh`. Its validated
defaults retain uniform sampling over `TRAIN_FAULTS`, a binary fault head, and
reconstruction gradients through the shared trunk. The failed v3 alternatives
remain available as explicit ablations with `CLEAN_SAMPLE_PROB=0.5`,
`FAULT_HEAD_MODE=per_fault`, and `RECON_DETACH=1`. To fill missing paired
deterministic baselines without rerunning seed 7:

```bash
SEEDS="13 23" scripts/reliability_paper_v2.sh baseline
```

In automatic GPU mode, the script waits when no card is below
`GPU_FREE_MEM_MB`; it no longer falls back to physical GPU 0. Use `GPUS="7"`
only when intentionally overriding that availability check.

The v2 suite evaluates the base training faults plus held-out fault variants
that are not included in `TRAIN_FAULTS`: `packet_loss_burst`,
`packet_loss_partial`, and `sensor_delay_jitter`. These challenge scenarios are
intended to test whether the reliability monitor generalizes beyond the exact
fault simulator used during training.

The suite-level summarizer writes:

- `reports/v2_paper_suite/paper_gap_report.md`
- `reports/v2_paper_suite/core_aggregate.tsv`
- `reports/v2_paper_suite/detection_aggregate.tsv`
- `reports/v2_paper_suite/gate_policy.tsv`
- `reports/v2_paper_suite/fair_baseline_runs.tsv`
- `reports/v2_paper_suite/fair_baseline_aggregate.tsv`
- `reports/v2_paper_suite/stress_aggregate.tsv`

The stress syntax is handled by `faults.py`: `packet_loss@0.30` changes packet
loss severity, and `sensor_delay@20` changes the delay to 20 samples. The same
syntax works for challenge variants such as `packet_loss_burst@0.15` and
`sensor_delay_jitter@20`.

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
