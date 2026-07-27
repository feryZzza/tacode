# AAAI-27 新服务器实验 TODO

本文件只保留**当前仍需在新服务器运行或实现后运行**的内容。旧清单中的主模型三
seed、stress、ordered ablation、LOSO、action-input、deep ensemble、完整九故障汇总
和现有图表均已完成，不要重复执行。

当前论文只能声称“modest offline evidence”，完成以下 P0 之前不能声称在线安全性。

## 0. 服务器准备与结果冻结

```bash
cd ~/tacode
git fetch origin
git switch paper/aaai27-maintenance
git pull --ff-only origin paper/aaai27-maintenance
git rev-parse HEAD | tee reports/AAAI27_SERVER_COMMIT.txt

conda run --no-capture-output -n pytorch \
  python -m unittest discover -s tests -t . -v
```

验收：

- 全部测试通过；
- `git status --short` 为空；
- 完整数据和三个主模型 checkpoint 可读；
- 所有正式作业均使用 `--limit-trials 0`；
- 每个输出目录保存完整 `reliability_report.json`、运行参数和代码 commit。

## 1. P0：按连续响应指标重新选择门控工作点

### 1.1 先补齐选择逻辑

当前归档工作点主要由 wrong-direction fraction 选择，但对非负门控有

\[
w(g u)\leq w(u),
\]

因此其 non-increase 约束恒成立。服务器运行前应修改门控选择，使主要目标为验证故障
上的 \(X_{\mathrm{opp}}\)，clean 条件至少同时约束：

- aligned-command retention；
- capped-overlap retention；
- torque tracking RMSE；
- full-shutdown fraction \(P(g=0)\)；
- mean absolute torque rate；
- false shutdowns per minute。

不得继续用 wrong-direction non-increase 作为实质安全约束。wrong-direction fraction
仅保留为 hard-shutdown 诊断量。

### 1.2 工作点扫描

固定现有三个主模型 checkpoint 和验证参与者，只在验证集选择参数，测试集只评一次。
至少扫描：

- 原始无 rate limit；
- `gate-max-fall-per-step`：
  `0.0025,0.005,0.01,0.02,0.05`；
- `gate-max-rise-per-step`：
  `0.001,0.0025,0.005,0.01,0.02`；
- 原 gate softness/deadband 网格。

示例参数接口：

```bash
--select-gate-on-val \
--gate-max-fall-per-step 0.01 \
--gate-max-rise-per-step 0.005
```

需要为选择器新增的结果字段：

```text
selection_objective = x_opp
val_clean_x_opp
val_fault_x_opp
val_clean_tracking_rmse
val_clean_capped_overlap_retention
val_clean_full_shutdown_fraction
val_clean_shutdowns_per_minute
val_clean_mean_abs_torque_rate
selected_gate_deadband
selected_gate_softness
selected_gate_max_fall_per_step
selected_gate_max_rise_per_step
```

## 2. P0：匹配对照与实际门控基线

所有基线必须使用同一模型输出、同一底层窗口、同一 active mask 和固定随机种子。每个
seed、task split、fault operator 分别计算，不能只比较宏平均。

必须加入：

1. `fault-only` gate；
2. `reconstruction-only` gate；
3. `staleness-only` gate；
4. validation-selected-subset gate；
5. all-gate-channels gate（论文实际门控）；
6. hard-threshold gate；
7. oracle prediction-error gate（只作离线上界）；
8. 与实际 \(P(g=0)\) 匹配的 random shutdown；
9. 与实际 clean aligned-command retention 匹配的 random attenuation。

两个随机基线至少使用 1,000 个固定随机抽样，并报告随机化分布及实际门控相对该分布
的 percentile。不能只报告一次随机结果。

每种 gate 必须输出：

```text
x_opp
wrong_direction_fraction
peak_wrong_torque
aligned_command_retention
capped_overlap_retention
over_assistance_fraction
tracking_rmse
mean_abs_torque_rate
full_shutdown_fraction
gate_below_half_fraction
shutdowns_per_minute
mean_shutdown_duration_ms
p95_shutdown_duration_ms
```

主要比较：

- 实际 gate vs full-closure-matched random；
- 实际 gate vs retention-matched random；
- continuous gate vs hard threshold；
- all-channel gate vs selected subset；
- all-channel gate vs component-only gates；
- actual gate vs oracle upper bound。

## 3. P0：门控动态与 torque-rate

对第 1 节的 rate-limit 网格生成 safety–utility Pareto 表，横轴至少包含 clean
retention 或 tracking RMSE，纵轴同时展示 \(X_{\mathrm{opp}}\) 和 torque rate。

最终候选工作点必须满足预注册的 clean 约束，不能看到测试结果后再改变阈值。至少报告：

- 无平滑原门控；
- 最优 causal rate-limited gate；
- causal hysteretic gate；
- hard threshold；
- ungated estimator。

如果任何门控降低 \(X_{\mathrm{opp}}\) 但增加 torque rate，应明确列为 trade-off，不能
用“更安全”概括。

## 4. P0：瞬态故障、检测延迟与恢复

现有故障覆盖完整 768-sample window，只测稳态异常。新增 partial-window 注入：

- onset 位于窗口的 20%、40%、60%、80%；
- duration：25、50、100、250、500 ms；
- 有恢复和无恢复；
- 单故障与两种同时故障；
- clean pause / near-stationary motion 作为冻结故障的负对照。

至少覆盖：

- insole missing；
- encoder dropout；
- IMU bias ramp；
- packet loss；
- stuck IMU；
- sensor delay；
- burst loss；
- partial-channel loss；
- delay jitter。

逐 timestep 保存：

```text
timestamp
participant
trial
task_split
fault_operator
fault_onset
fault_offset
fault_active
all anomaly channels
normalized anomaly score
gate
ungated torque
gated torque
ideal torque
active mask
```

报告：

- onset-to-first-\(g<0.5\) 延迟；
- onset-to-first-\(g=0\) 延迟；
- recovery-to-\(g>0.5\) 和 recovery-to-\(g=1\) 延迟；
- 每分钟错误关闭次数；
- alarm chatter / gate transitions per minute；
- 短故障 miss rate；
- clean pause false-positive rate。

## 5. P0：同窗口 clean–fault 配对分析

对每个底层 clean window 保存其九种 corrupted counterpart，使用相同模型和 gate。
主要估计量为：

\[
\Delta_{\mathrm{fault-specific}}
=
\bigl(M_{\mathrm{gated}}^f-M_{\mathrm{ungated}}^f\bigr)
-
\bigl(M_{\mathrm{gated}}^{clean}-M_{\mathrm{ungated}}^{clean}\bigr),
\]

其中 \(M\) 至少包含 \(X_{\mathrm{opp}}\)、tracking RMSE、torque rate 和
wrong-direction fraction。

统计要求：

- 先在 window 内做配对差分；
- 再在 participant 内聚合；
- participant 作为最终不确定性单位；
- 报告 participant-level bootstrap 95% CI；
- 同时给出 seed sensitivity；
- 不把高度重叠的模型训练集解释为独立总体样本。

现有 `scripts/analyze_gate_specificity.py` 只提供 seed × task-split 汇总层面的
clean-adjusted contrast，不能替代本节。

## 6. P0：反向消融或 factorial ablation

现有 ordered ablation 只能描述固定添加顺序。至少补以下 leave-one-component-out：

- full model；
- minus probabilistic regression；
- minus reconstruction；
- minus forecasting；
- minus fault head；
- minus staleness；
- full model using validation-selected gate subset。

如果 GPU 预算允许，再运行
`probabilistic × reconstruction × forecasting` 的 \(2^3\) factorial ablation。

每个模型至少三 seeds，并同时报告：

- clean ID / training-held-out-task RMSE；
- actual-gate AUROC 或 risk–coverage；
- \(X_{\mathrm{opp}}\)；
- tracking RMSE；
- clean retention；
- torque rate；
- full-shutdown fraction。

结论必须基于 Pareto 比较；如果简单模型支配完整模型，应修改 proposed configuration，
而不是只解释 ordered stage。

## 7. P1：真实故障与在线实验

若硬件、伦理审批和被试资源允许：

1. bench test：真实拔线、冻结、延迟和漂移；
2. 无人佩戴条件下验证低级扭矩限制、rate limit 和恢复逻辑；
3. 受控人体实验，先低扭矩、急停和安全员；
4. 报告舒适度、误关闭、恢复时间、交互稳定性和 adverse events。

没有完成本节时，论文必须保留“offline fixed-kinematics replay does not establish
human safety or metabolic benefit”。

## 8. 推荐执行顺序

```text
代码与测试
  ↓
按 X_opp 重写并冻结验证选择规则
  ↓
三 seed 工作点 + rate-limit 网格
  ↓
matched-random / component / hard / oracle gates
  ↓
partial-window transient faults
  ↓
window-level paired analysis
  ↓
leave-one-component-out / factorial ablation
  ↓
汇总、重绘、重新编译论文
  ↓
可选：bench 与在线实验
```

## 9. 拷回与验收

必须拷回：

```text
reports/AAAI27_SERVER_COMMIT.txt
reports/aaai27_gate_reselection/
reports/aaai27_gate_baselines/
reports/aaai27_gate_dynamics/
reports/aaai27_transient_faults/
reports/aaai27_paired_analysis/
reports/aaai27_reverse_ablation/
```

每个目录至少包含：

- `reliability_report.json` 或等价逐 run JSON；
- `run_manifest.tsv`；
- `aggregate.tsv`；
- 运行日志；
- 完整 CLI 参数；
- checkpoint 哈希；
- 数据 split 与 participant IDs；
- 随机种子。

最终验收：

1. 所有正式实验为 full data、三 seeds，无 smoke 或 trial limit；
2. 测试集没有参与工作点选择；
3. random baselines 确实匹配 closure/retention，而非近似口头说明；
4. paired analysis 能追溯到同一 clean window；
5. torque-rate、shutdown dynamics 和失败结果没有被省略；
6. `python3 scripts/check_paper_numbers.py` 通过；
7. `make main.pdf supplement.pdf ReproducibilityChecklist.pdf` 通过；
8. 主文、补充材料、中文版、图表和 provenance 同步更新。

