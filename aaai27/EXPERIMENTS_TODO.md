# AAAI-27 最终服务器实验 TODO（覆盖 2026-07-27 旧版）

本清单对应 `paper/aaai27-maintenance` 的最终门控一致性复核。主模型、三 seed、九类
稳态故障、stress、LOSO、ordered ablation、reverse-ablation checkpoint，以及第一轮
§1--§6 实验均已完成，**不要重新训练主模型**。

目前必须重跑的是同一批 checkpoint 上的纯评测。原因不是实验缺失，而是第一轮 §2--§5
误把 Detector-gate 的 validation-selected AUROC 子集当成了实际执行门控；论文定义的
执行门控应始终融合六个 command-time 通道：

```text
logit, aleatoric, residual, forecast, epistemic, staleness
```

第一轮随机 attenuation 也只是打乱 gate 值，没有严格匹配论文定义的
\(\kappa=\text{gated aligned command}/\text{ungated aligned command}\)。当前代码已经
把“aligned-command adequacy”和上述 retention ratio 分开，并实现了精确 retention
匹配。旧的 `reports/aaai27_*` 数字只能作为探索性结果，不能直接替换进最终结论。
另外，第一轮 gate-reselection driver 曾把 held-out tasks 写成 `stairs,ramp`，与主论文
报告保存的 `jump,cutting,lift_weight,lunges` 不一致。当前 driver 改为逐 seed 从原始
主报告读取 split/window 参数，validator 会拒绝任何配置漂移。

## P0：提交前必须完成的最终门控复评

### 0. 冻结代码与环境

先把本地修改提交、推送，再在服务器执行：

```bash
cd ~/tacode
git fetch origin
git switch paper/aaai27-maintenance
git pull --ff-only origin paper/aaai27-maintenance
git status --short
git rev-parse HEAD

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

PY=/home/zfy/miniconda3/envs/pytorch/bin/python
DATA_ROOT=/home/zfy/dataset/tcn/Parsed

$PY -m unittest discover -s tests -t . -v

mkdir -p reports/aaai27_environment
$PY -VV > reports/aaai27_environment/python-version.txt 2>&1
$PY -m pip freeze > reports/aaai27_environment/pip-freeze.txt
$PY -c 'import torch; print(torch.__version__); print(torch.version.cuda)' \
  > reports/aaai27_environment/torch-version.txt
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv,noheader > reports/aaai27_environment/gpu.csv
lscpu > reports/aaai27_environment/lscpu.txt
uname -a > reports/aaai27_environment/uname.txt
free -h > reports/aaai27_environment/memory.txt
cp /etc/os-release reports/aaai27_environment/os-release.txt
```

验收：

- `git status --short` 为空；
- 单测全部通过；
- `reports/aaai27_environment/` 中记录 CPU、GPU、内存、OS、Python、PyTorch/CUDA
  和完整 `pip freeze`；
- `reports/v2_main_seed{7,13,23}/reliability_tcn_best.pt` 与原始
  `reports/v2_fc_main_seed{7,13,23}/reliability_report.json` 均存在；
- 正式运行保持 `limit_trials=0`，不得用 smoke-test 截断。

### 1. 归档第一轮 §1--§6 产物

不要删除旧结果，也不要让新旧 TSV 混在同一目录。以下目录存在才移动：

```bash
mkdir -p reports/archive_aaai27_gate_20260728
for name in \
  aaai27_gate_reselection \
  aaai27_gate_baselines \
  aaai27_gate_dynamics \
  aaai27_transient_faults \
  aaai27_paired_analysis \
  aaai27_reverse_ablation \
  aaai27_logs
do
  if [ -e "reports/$name" ]; then
    mv "reports/$name" reports/archive_aaai27_gate_20260728/
  fi
done
```

### 2. 重选三 seed 的六通道门控工作点

纯 eval，复用三个主 checkpoint：

```bash
PY=$PY \
DATA_ROOT=$DATA_ROOT \
GPUS="4 5 6" \
SEEDS="7 13 23" \
bash scripts/run_gate_reselection.sh
```

每个新报告必须同时满足：

- `results._gate_policy.gate_signals` 是上述六通道；
- `results._detector_policy.online_signals` 单独保存检测器子集，且不再覆盖执行门控；
- `results._gate_policy.selection_objective == "x_opp"`；
- `val_clean_aligned_command_retention` 与
  `val_clean_aligned_command_adequacy` 分别存在；
- `val_clean_capped_overlap_retention` 与
  `val_clean_capped_overlap_adequacy` 分别存在；
- `n_feasible > 0`；
- held-out tasks、window/stride、input profile、side、trial limit 与对应
  `v2_fc_main_seed*` 原报告一致；
- 选择只使用 validation split，test 只评一次。

### 3. 重跑 §2--§5

仍然是纯 eval。四个 stage 必须读取第 2 步的新工作点，而不是
`reports/v2_fc_main_seed*` 的归档工作点：

```bash
PY=$PY \
DATA_ROOT=$DATA_ROOT \
POLICY_ROOT=reports/aaai27_gate_reselection \
GPUS="4 5 6" \
SEEDS="7 13 23" \
DRAWS=1000 \
BOOTSTRAP=10000 \
WORKERS=6 \
bash scripts/run_aaai27_gate_suite.sh
```

产物：

```text
reports/aaai27_gate_baselines/
reports/aaai27_gate_dynamics/
reports/aaai27_transient_faults/
reports/aaai27_paired_analysis/
reports/aaai27_logs/
```

关键验收口径：

- baseline provenance 的 `reference_gate` 必须是 `all_gate_channels`；
- baseline、dynamics、transient、paired provenance 的 `gate_policy.signals`
  都必须是六通道；
- `selected_subset` 只作 Detector-gate 诊断对照；
- `oracle_error` 只作不可实现诊断对照，不称为理论上界；
- `random_attenuation.matched_quantity == aligned_command_retention`；
- `random_retention_mean` 与 `actual_retention` 的绝对差不超过 `1e-5`；
- 所有新表同时保留 `aligned_command_retention` 与
  `aligned_command_adequacy`，不能再用同一列名混写两个量；
- baseline 与 dynamics 表也要同时保留 `capped_overlap_retention` 与
  `capped_overlap_adequacy`；
- transient 必须含 ID、held-out-task、clean pseudo-onset control，以及两种同时故障；
- paired analysis 的 bootstrap 单位是 participant，不是重叠窗口。

### 4. 重新汇总 reverse ablation

这一步不训练。训练侧格子继续读取已有 `v2_reverse_*` 与
`v2_ablation_prob_aug_recon_seed*` 报告；full-stack、selected-subset 与
minus-staleness 改用第 3 步同一前向/同一故障实现产生的新 gate-baselines 表。
当前汇总器会用 `all_gate_channels` 作为唯一 full-stack 参照，不再同时混入旧 `full` 格：

```bash
$PY scripts/aggregate_reverse_ablation.py \
  --reports-dir reports \
  --output-dir reports/aaai27_reverse_ablation
```

该表仍属于描述性分析：训练侧旧报告与 gate-only 格不是同一次统一前向，不能据此宣称
某个组件被严格支配。若要写强组件因果结论，执行下面的 P1。

### 5. 汇总、自动验收、提取数字

```bash
$PY scripts/aggregate_aaai27_suite.py \
  --reports-dir reports \
  --log-dir reports/aaai27_logs

$PY scripts/validate_aaai27_final_gate_audit.py \
  --reports-dir reports

$PY scripts/extract_aaai27_gate_numbers.py \
  --reports-dir reports \
  --output reports/aaai27_gate_numbers.json
```

只有 validator 退出码为 0，且
`reports/aaai27_suite_status.json` 的 `problems` 为空，才可把数字写入稿件。
旧的 `scripts/check_paper_numbers.py` 在稿件更新数字前可能按预期失败；它应在新 JSON
拷回本地、更新英文正文后再次运行。

### 6. 回传清单

将以下内容完整拷回本地，不要只抄宏平均数字：

```text
reports/aaai27_gate_reselection/
reports/aaai27_gate_baselines/
reports/aaai27_gate_dynamics/
reports/aaai27_transient_faults/
reports/aaai27_paired_analysis/
reports/aaai27_reverse_ablation/
reports/aaai27_logs/
reports/aaai27_environment/
reports/aaai27_suite_manifest.tsv
reports/aaai27_suite_status.json
reports/aaai27_gate_numbers.json
```

另外保存服务器 commit 与文件清单：

```bash
git rev-parse HEAD > reports/AAAI27_FINAL_GATE_COMMIT.txt
git status --short > reports/AAAI27_FINAL_GATE_GIT_STATUS.txt
find reports/aaai27_* -type f -print | sort \
  > reports/AAAI27_FINAL_GATE_MANIFEST.txt
tar czf aaai27_final_gate_audit.tar.gz \
  reports/aaai27_gate_reselection \
  reports/aaai27_gate_baselines \
  reports/aaai27_gate_dynamics \
  reports/aaai27_transient_faults \
  reports/aaai27_paired_analysis \
  reports/aaai27_reverse_ablation \
  reports/aaai27_logs \
  reports/aaai27_environment \
  reports/aaai27_suite_manifest.tsv \
  reports/aaai27_suite_status.json \
  reports/aaai27_gate_numbers.json \
  reports/AAAI27_FINAL_GATE_COMMIT.txt \
  reports/AAAI27_FINAL_GATE_GIT_STATUS.txt \
  reports/AAAI27_FINAL_GATE_MANIFEST.txt
sha256sum aaai27_final_gate_audit.tar.gz \
  > aaai27_final_gate_audit.tar.gz.sha256
```

### 7. 回传后的唯一封版清单

P0 回传后，旧探索性数字不能与新六通道结果并存。只维护 `aaai27/` 这一份英文投稿源，
按以下顺序封版：

1. 用 `aaai27_gate_numbers.json` 更新摘要、`Reselection Audit and Matched Controls`、
   `Transients, Paired Effects, and Reverse Ablation`、Discussion、Limitations、Conclusion，
   以及 supplement 的对应段落。
2. 删除正文中关于旧 `stairs,ramp` 误跑、archived selected-subset operating point 和
   “first pass” 的过程性叙述；最终稿只报告 split-locked 六通道 P0 结果。错误批次仅保留在
   `RESULTS_PROVENANCE.md`，不作为论文内容。
3. 旧 gate-baseline、dynamics、transient、paired、reverse-ablation 数字全部替换或删除；
   不得把新工作点与旧图、旧表或旧 selected-subset 数字拼在一起。
4. 用 P0 汇总重新生成所有受门控工作点影响的图；若某张图仍读取
   `reports/v2_fc_paper_suite`，必须明确证明它与新工作点无关，否则改为读取 P0 汇总。
5. 在写入 P0 数字前扩展 `scripts/check_paper_numbers.py`，让它读取
   `aaai27_gate_numbers.json` 并逐项核对所有新增正文数字；不得只靠人工抄写。
   同时把 `aaai27_environment/` 的 CPU 型号和依赖版本写入 supplement；
   只有信息完整时才把 reproducibility checklist 的 computing-infrastructure 项从
   `partial` 改为 `yes`。
6. 重新运行：

```bash
$PY scripts/validate_aaai27_final_gate_audit.py --reports-dir reports
$PY scripts/check_paper_numbers.py
$PY -m unittest discover -s tests -t . -v
make -C aaai27 check-figures
make -C aaai27 submission
```

7. 对三个最终 PDF 检查页数、正文/参考文献边界、overfull/undefined、字体嵌入、
   PDF 元数据匿名性和逐页视觉结果。只有这些检查全部通过，才把 P0 后的版本称为最终稿。

## P1：只有要保留强 reverse-ablation 结论时才运行

P0 足以完成最终门控主张。若正文仍想声称“完整组件栈优于移除任一组件”，则必须把每个
已有 reverse-ablation checkpoint 都在当前 commit 上重新 eval，并对每个模型分别在
validation 上重选门控工作点；所有格必须使用同一 fault list、split、窗口、随机种子和
六通道定义。checkpoint 已存在时不重训。

最低要求：

- 每格三 seed；
- `limit_trials=0`；
- 同一 `CORE_FAULTS`；
- 每格 `--select-gate-on-val --gate-selection-objective x_opp`；
- 同一 softness/deadband/fall/rise 网格与 clean constraints；
- 用完全相同的 corruption realization 比较；
- 只在 `(fault X_opp, clean retention/tracking RMSE)` Pareto 平面上写结论。

如果任何格缺 checkpoint，才训练缺失格；不要重训已有格。未完成这套公平复评时，保留
当前稿件中的“exploratory/descriptive”限定，不写组件支配关系。

## P2：不属于本轮服务器离线训练

以下是后续真实系统验证，不应与 P0 混为一谈：

- 实机闭环、真实通信栈与 actuator saturation；
- 人体实验中的舒适度、稳定性与 adverse event；
- 长时序故障、跨窗口状态与恢复；
- 单一受控协议下更广的 simultaneous/compound faults；
- latency、功耗和部署资源测量。

完成 P0 仍只能支持“offline reliability evaluation”，不能声称临床安全、在线安全或
真实设备上的故障保护效果。P0 协议是在检查旧 test 诊断后修订的，因此最终论文还必须
把它标为 post-hoc reevaluation，而不能写成 preregistered 或 fresh confirmatory test。
