# AAAI-27 论文结果数据需求与服务器拷贝清单

本文档用于收集 `main.tex` 中全部 `[TBD]` 所需的数据。目标是只从服务器拷回足以复算论文表格、摘要数字和 Discussion 结论的结果文件，避免搬运原始传感器数据和不必要的模型权重。

> **评测版本要求：** `forecast_residual` 已改为在预测目标到达的 `t+h` 时刻发布。旧报告若由修正前代码生成，其 forecast AUROC、fused detector、gate、stress 和相关 ablation 指标均应使用原 checkpoint 重新评测后再拷回；预测头训练目标未变，因此不要求重新训练主模型。

## 1. 结论先行

### 必须拷回

每个实验目录中的：

```text
reliability_report.json
```

以及服务器汇总后生成的：

```text
reports/v2_paper_suite/*.tsv
reports/v2_paper_suite/*.md
```

`reliability_report.json` 已包含运行参数、participant/task split、训练历史、门控策略、检测策略以及各场景的 RMSE、R2、NLL、coverage ECE、AUROC 和控制指标。只要 JSON 完整，就可以在本机重新汇总绝大多数 `[TBD]`。

### 通常不需要拷回

- 原始 `Parsed` 数据集；
- 训练日志；
- `reliability_tcn_best.pt` 模型权重；
- 临时缓存和 TensorBoard 文件。

如果服务器之后可能无法继续访问，建议额外保留三个主模型 checkpoint。因为当前默认评测遗漏了 `stuck_imu`，必要时可利用 checkpoint 补评测，而无需重新训练。

## 2. 实验版本必须满足的条件

用于论文的报告必须满足：

| 项目 | 要求 |
|---|---|
| 数据规模 | `limit_trials=0`，不能使用 smoke 或受限数据 |
| 输入 | `input_profile=human`，25 个 wearable channels |
| 主模型 | `training_mode=prob_aug_recon_fc` |
| 公平 baseline | `training_mode=det_noaug`，与主模型具有相同 seed 和 split |
| 主结果 seeds | `7, 13, 23` |
| 训练轮数 | 20 epochs |
| Held-out tasks | `jump, cutting, lift_weight, lunges` |
| Participant split | 所有主模型和 baseline 必须完全一致 |
| Gate | 仅由 validation participant、validation fault 和 validation OOD-clean 选择 |
| Detector | 信号集合和 q90 reference 仅由 validation 数据确定 |
| 测试数据 | 不能参与 detector subset、deadband 或 softness 选择 |

每个 JSON 中至少应存在以下顶层字段：

```text
args
splits
history
results
```

`results` 中至少需要：

```text
_gate_policy
_detector_policy
test_id/clean
test_ood/clean
test_id/<fault>
test_ood/<fault>
fault_detection/test_id/<fault>
fault_detection/test_ood/<fault>
```

## 3. 需要拷回的实验目录

### P0：摘要和三张主表必需

#### 3.1 主模型，3 个 seeds

```text
reports/v2_main_seed7/reliability_report.json
reports/v2_main_seed13/reliability_report.json
reports/v2_main_seed23/reliability_report.json
```

对应论文：摘要、Table 1 Proposed、Table 2、Table 3、Discussion。

#### 3.2 同划分确定性 baseline，3 个 seeds

```text
reports/v2_baseline_det_seed7/reliability_report.json
reports/v2_baseline_det_seed13/reliability_report.json
reports/v2_baseline_det_seed23/reliability_report.json
```

对应论文：摘要中的 clean RMSE change、Table 1 Deterministic、paired difference。

不能用 Nature 官方 checkpoint 替代这三份报告，因为它的训练划分与本文不匹配。

#### 3.3 顺序消融

最低要求为 seed 7 的五个模式：

```text
reports/v2_ablation_det_noaug_seed7/reliability_report.json
reports/v2_ablation_det_aug_seed7/reliability_report.json
reports/v2_ablation_prob_aug_seed7/reliability_report.json
reports/v2_ablation_prob_aug_recon_seed7/reliability_report.json
reports/v2_ablation_prob_aug_recon_fc_seed7/reliability_report.json
```

论文更强的推荐配置是五个模式均运行 seeds 7、13、23，即共 15 份 JSON。否则 Table 4 必须明确标注为单 seed attribution，不能报告标准差或稳定性结论。

#### 3.4 故障强度实验，3 个 seeds

默认输出目录为：

```text
reports/v2_refreshed_stress_seed7/reliability_report.json
reports/v2_refreshed_stress_seed13/reliability_report.json
reports/v2_refreshed_stress_seed23/reliability_report.json
```

每份报告需要至少包含：

```text
packet_loss@0.05
packet_loss@0.15
packet_loss@0.30
packet_loss_burst@0.05
packet_loss_burst@0.15
packet_loss_partial@0.15
sensor_delay@5
sensor_delay@10
sensor_delay@20
sensor_delay_jitter@10
sensor_delay_jitter@20
```

对应论文：最大 packet-loss rate、最大 delay 下的 AUROC、aligned assistance retention 和 wrong-direction change。

### P1：填完当前正文全部 `[TBD]` 所需

#### 3.5 Action-input ablation

```text
reports/v2_action_human_desired_seed7/reliability_report.json
reports/v2_action_human_measured_seed7/reliability_report.json
reports/v2_action_human_execution_seed7/reliability_report.json
reports/v2_action_human_interaction_seed7/reliability_report.json
```

这些数据用于判断 actuator/action signals 是否提高检测能力，或形成 device-specific shortcut。当前正文只需要定性结论，单 seed 可作为辅助实验；若将其升级为核心贡献，应补 3 seeds。

#### 3.6 LOSO

需要 15 个测试被试各一份报告：

```text
reports/v2_loso_BT01/reliability_report.json
reports/v2_loso_BT02/reliability_report.json
reports/v2_loso_BT03/reliability_report.json
reports/v2_loso_BT06/reliability_report.json
reports/v2_loso_BT07/reliability_report.json
reports/v2_loso_BT08/reliability_report.json
reports/v2_loso_BT09/reliability_report.json
reports/v2_loso_BT10/reliability_report.json
reports/v2_loso_BT11/reliability_report.json
reports/v2_loso_BT12/reliability_report.json
reports/v2_loso_BT13/reliability_report.json
reports/v2_loso_BT14/reliability_report.json
reports/v2_loso_BT15/reliability_report.json
reports/v2_loso_BT16/reliability_report.json
reports/v2_loso_BT17/reliability_report.json
```

LOSO 用于计算跨被试的 principal safety effect、置信区间和个体差异，而不是替代主表的 3-seed 实验。

### P2：可选增强证据

```text
reports/v2_ensemble_seed7/reliability_report.json
```

该报告用于比较 MC dropout 与 deep-ensemble epistemic uncertainty。它不是当前表格的硬性必需项，但可用于解释 epistemic channel 的贡献。

## 4. 故障集合与当前阻塞问题

### 4.1 论文所需故障分组

训练期已见故障：

```text
insole_missing
encoder_dropout
imu_bias
packet_loss
stuck_imu
sensor_delay
```

训练和 detector selection 均未见的生成器：

```text
packet_loss_burst
packet_loss_partial
sensor_delay_jitter
```

所有故障均需要同时评测：

```text
test_id/<fault>
test_ood/<fault>
fault_detection/test_id/<fault>
fault_detection/test_ood/<fault>
```

### 4.2 必须注意：当前默认脚本遗漏 `stuck_imu` 评测

当前 `TRAIN_FAULTS` 包含 `stuck_imu`，但 `CORE_FAULTS`、`CORE_SCENARIOS` 和 `build_aaai_tables.py` 的默认列表没有它。这会导致现有默认报告无法验证摘要中的“six training-time fault types”。

推荐在服务器上先补齐：

1. 使用三个主模型 checkpoint 对包含 `stuck_imu` 的完整 fault scenarios 重新 eval；
2. 确认 JSON 中出现 `test_id/stuck_imu`、`test_ood/stuck_imu` 和两个 detection keys；
3. 汇总时把 `stuck_imu` 加入 seen-fault macro。

如果不补该评测，则提交前必须把摘要和正文改成只声称评测当前实际包含的故障，且不能写“full table reports every preregistered fault”。

## 5. 每个 `[TBD]` 对应的数据字段

### 5.1 Abstract

| 摘要数字 | 数据来源 | 推荐定义 |
|---|---|---|
| clean RMSE change | 主模型与 paired baseline 的 `test_id/clean`、`test_ood/clean` | 每个 seed 先对 ID/OOD 求 macro，再计算 proposed minus baseline；报告 mean ± SD，正文可用绝对差或百分比但必须注明 |
| wrong-direction reduction | 主模型全部 fault scenarios | 每个 seed 先对 ID/OOD 和固定 fault set 求 macro；使用 `baseline_wrong_direction_ratio - gated_wrong_direction_ratio`，建议报告 absolute percentage points |
| clean aligned torque retained | 主模型 `test_id/clean`、`test_ood/clean` | `gated_retained_aligned_torque / baseline_retained_aligned_torque`，每个 seed 先对 ID/OOD 求 macro |

不要把 `baseline_*` 误解为确定性 TCN baseline：在单个 proposed report 中，它表示同一 proposed estimator 的 **ungated command**。

### 5.2 Table 1：Accuracy and Calibration

公平 baseline 数据来自：

```text
reports/v2_paper_suite/fair_baseline_aggregate.tsv
```

关键列：

```text
scenario
main_rmse_mean / main_rmse_std
baseline_rmse_mean / baseline_rmse_std
rmse_delta_mean / rmse_delta_std
main_r2_mean / main_r2_std
baseline_r2_mean / baseline_r2_std
r2_delta_mean / r2_delta_std
```

Proposed 的 NLL 和 coverage ECE 当前没有被 `summarize_reliability_suite.py` 写入 aggregate TSV，必须从三份主 JSON 的以下字段提取后计算 mean ± sample SD：

```text
results.test_id/clean.nll
results.test_id/clean.coverage_ece
results.test_ood/clean.nll
results.test_ood/clean.coverage_ece
```

### 5.3 Table 2：Fault Detection

来源：

```text
reports/v2_paper_suite/main_detection_aggregate.tsv
reports/v2_paper_suite/table1_detection.tsv
```

关键 detector row：

```text
fault_detection/test_id/<fault>
fault_detection/test_ood/<fault>
```

关键列：

```text
auroc_mean / auroc_std                 # frozen fused detector
auroc_logit_mean / std
auroc_residual_mean / std              # reconstruction residual
auroc_forecast_mean / std
auroc_staleness_mean / std
auroc_coherence_mean / std
auroc_drift_mean / std
```

Macro-AUROC 的固定口径：

1. 每个 seed 内对 fault types 等权平均；
2. seen 和 unseen generator 分开计算；
3. ID 和 OOD 分开计算；
4. 最后跨 seeds 报告 mean ± sample SD。

不能把所有 windows 合并后重新计算一个 pooled AUROC，也不能根据 test AUROC 重新选择 detector channels。

“strongest signal”“hardest fault”和 Discussion 中的 fault family 解释，均从完整的 fault-by-signal 矩阵产生，不能只看论文中展示的五个代表行。

### 5.4 Table 3：Safety--Utility Trade-off

来源：

```text
reports/v2_paper_suite/main_aggregate.tsv
reports/v2_paper_suite/table2_safety.tsv
```

关键列：

```text
our_ungated_wrong_mean / std       # Wrong-U
our_wrong_mean / std               # Wrong-G
gate_wrong_delta_mean / std        # Wrong-G minus Wrong-U
gate_retained_ratio_mean / std     # Ret.
mean_gate_mean / std               # Gate
```

六个论文场景的分组方式：

| 论文行 | scenarios |
|---|---|
| ID clean | `test_id/clean` |
| OOD clean | `test_ood/clean` |
| ID seen faults | `test_id/` + 六个 seen faults |
| ID unseen faults | `test_id/` + 三个 unseen generators |
| OOD seen faults | `test_ood/` + 六个 seen faults |
| OOD unseen faults | `test_ood/` + 三个 unseen generators |

对于 fault macro，必须先在每个 seed 内对 fault types 等权平均，再跨 seeds 统计。不能直接平均 aggregate TSV 中的标准差。

### 5.5 Table 4：Ablation

每个 ablation mode 使用固定定义：

| 列 | 定义 |
|---|---|
| RMSE | `test_ood/clean.rmse` |
| AUC | ID/OOD × 固定 fault set 的 macro fused AUROC |
| Wrong-D | faults 上 `gated_wrong - ungated_wrong` 的 macro；没有 gate 的 deterministic row 写 `--` |
| Ret. | `test_id/clean` 与 `test_ood/clean` 的 gate retained ratio macro |

顺序差值必须按下列相邻行计算：

```text
det_noaug
→ det_aug
→ prob_aug
→ prob_aug_recon
→ prob_aug_recon_fc
```

### 5.6 Stress、LOSO 与 Discussion

Stress 最大强度固定使用：

```text
packet_loss@0.30
sensor_delay@20
```

分别提取 fused AUROC，并对这两个场景的 ID/OOD 计算 gate retention 和 wrong-direction change。不要在看完 test 结果后改选“最好看的最大强度”。

LOSO principal safety effect 推荐定义为：每个 test participant 上固定 fault set 的 macro `ungated_wrong - gated_wrong`。报告 15 folds 的均值、95% bootstrap CI、中位数和改善被试比例，同时报告 clean retention，防止通过全面关闭助力获得表面安全提升。

Discussion 中以下文字需要从数据矩阵填写：

- 每个 fault family 的 strongest signal；
- 最困难故障及其 AUROC；
- 困难来自 detection、calibration 还是 gate response；
- wrong-direction reduction 与 clean retention 的最终 operating point；
- LOSO effect 的方向、置信区间和个体异质性。

## 6. 服务器端推荐操作顺序

### 6.1 先检查报告完整性

```bash
find reports -path '*/reliability_report.json' -size +0 -print | sort
```

运行汇总：

```bash
scripts/reproduce_aaai.sh summary
python scripts/build_aaai_tables.py \
  --suite-dir reports/v2_paper_suite \
  --output-dir reports/v2_paper_suite
```

检查 gap report：

```bash
sed -n '1,240p' reports/v2_paper_suite/paper_gap_report.md
```

### 6.2 生成运行清单和代码版本

```bash
git rev-parse HEAD > reports/AAAI27_CODE_COMMIT.txt
git status --short > reports/AAAI27_GIT_STATUS.txt
find reports -path '*/reliability_report.json' -size +0 -print | sort \
  > reports/AAAI27_REPORT_MANIFEST.txt
```

### 6.3 打包仅指标版本

下面的命令会打包所有 v2 JSON、论文汇总结果和版本清单，不包含 `.pt`：

```bash
find reports -type f \( \
  -path 'reports/v2_*/reliability_report.json' -o \
  -path 'reports/v2_paper_suite/*' -o \
  -name 'AAAI27_CODE_COMMIT.txt' -o \
  -name 'AAAI27_GIT_STATUS.txt' -o \
  -name 'AAAI27_REPORT_MANIFEST.txt' \
\) -print0 | tar --null -T - -czf aaai27_metrics.tar.gz

sha256sum aaai27_metrics.tar.gz > aaai27_metrics.tar.gz.sha256
```

### 6.4 可选 checkpoint 包

仅在需要离线补评测或服务器即将释放时打包：

```bash
tar -czf aaai27_main_checkpoints.tar.gz \
  reports/v2_main_seed7/reliability_tcn_best.pt \
  reports/v2_main_seed13/reliability_tcn_best.pt \
  reports/v2_main_seed23/reliability_tcn_best.pt

sha256sum aaai27_main_checkpoints.tar.gz \
  > aaai27_main_checkpoints.tar.gz.sha256
```

## 7. 拷回本机后的验收标准

### 最低核心包

- 3 个主模型 JSON；
- 3 个 paired deterministic baseline JSON；
- 5 个单 seed ablation JSON；
- 3 个 stress JSON；
- `v2_paper_suite` 全部 TSV/MD。

共至少 14 份 `reliability_report.json`。

### 填完当前正文所有 `[TBD]`

在最低核心包基础上增加：

- 4 个 action profile JSON；
- 15 个 LOSO JSON。

共至少 33 份 JSON。若五个 ablation 均跑 3 seeds，则为 43 份 JSON。Ensemble 报告另计。

### 验收检查

1. `sha256sum -c aaai27_metrics.tar.gz.sha256` 通过；
2. 三个主模型和三个 baseline 的 seed、split signature 完全匹配；
3. 所有正式报告 `limit_trials=0`；
4. 每个主 seed 同时具有 clean、seen fault、unseen generator 的 ID/OOD 结果；
5. `_gate_policy` 和 `_detector_policy` 均存在；
6. `stuck_imu` 已补评测，或论文相关声称已同步缩减；
7. `paper_gap_report.md` 不再提示缺失主 seeds、paired baseline 或 stress curves；
8. 论文中的 mean/std 使用相同 seed 集合，不能混用单 seed best run。

## 8. 不足以填论文的数据

以下内容不能作为最终证据：

- `reports/reliability_smoke/`；
- 旧版 `reports/reliability_human/`；
- 任意 `limit_trials > 0` 的快速实验；
- 只有 checkpoint、没有 `reliability_report.json` 的目录；
- 只有最终均值、没有 per-seed JSON 的截图或手工表格；
- 使用不同 participant/task split 的 proposed 和 baseline；
- 根据 test AUROC 重新选择 detector signal 或 gate 参数的结果。

如果还要制作代表性故障时序图，仅有当前 JSON 不够，因为当前训练程序不保存逐 timestep prediction、risk、gate 和 torque。该图需要在服务器上额外导出时序数组，或保留 checkpoint 后重新运行带时序导出的评测。
