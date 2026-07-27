# AAAI-27 稿件

风险敏感型任务无关外骨骼辅助的可靠性感知力矩估计。中英双语稿件，图表全部由脚本从
实验汇总目录确定性生成。

## 文件

- `main.tex` — 英文投稿主文件，单文件匿名格式，符合官方 author kit 要求。
- `main_zh.tex` — 中文版，`ctexart` 文档类，必须用 XeLaTeX 编译。
- `references.bib` — 参考文献。
- `aaai2027.sty`、`aaai2027.bst` — AAAI-27 官方 author kit（2026 年 5 月版）原文件，不要修改。
- `figures/` — 投稿用矢量 PDF，附 SVG 源和 300 dpi PNG 预览。
- `figures/data/timeseries_overview.npz` — 图 1(c) 的逐 timestep 导出数据。
- `OUTLINE.md`、`DATA_REQUIREMENTS.md`、`EXPERIMENTS_TODO.md` — 写作与实验规划笔记。

## 编译

```bash
make          # 出图 + 英文版
make zh       # 中文版（XeLaTeX）
make figures  # 只出图
```

英文版走 pdflatex，中文版走 xelatex。两者共用同一批图。

## 数字从哪来

出图脚本是 `../scripts/draw_aaai_figures.py`。所有数字优先从汇总目录读取：

| 面板 | 数据源 |
| --- | --- |
| 检测热力图、宏平均 AUROC | `table1_detection.tsv`、`paper_numbers.json` |
| 安全—效用 | `table2_safety.tsv` |
| 消融、压力迁移 | `paper_numbers.json` |
| LOSO 逐折效应量 | `loso_effect.json` |
| 图 1(c) 门控时序 | `figures/data/timeseries_overview.npz` |

默认汇总目录是 `../reports/v2_fc_paper_suite`（`SUITE_DIR` 可覆盖）。`reports/` 不入版本库，
从指标包解开：

```bash
tar xzf ../aaai27_metrics.tar.gz -C ..
```

脚本每次运行都会打印数据来源，**这一行是唯一可靠的验收凭据**：

```
data sources: detection=suite, macro=suite, safety=suite, ablation=suite, stress=suite, loso=suite, overview=export
```

七项全是 `suite`/`export` 才是对的。出现 `fallback` 说明汇总目录没读到，图里是脚本内置的
`FALLBACK` 常量（旧数字）；出现 `overview=schematic` 说明 npz 没读到，图 1(c) 退成了示意
曲线，与 caption 声明的「真实导出」不符。**两种退化都不会报错，退出码仍是 0。**

`make figures` 和 `make check-figures` 已经把路径钉死并在缺文件时主动失败。如果直接调脚本，
注意它的默认路径是相对仓库根目录的，必须显式给参数：

```bash
python3 scripts/draw_aaai_figures.py \
    --output-dir aaai27/figures \
    --suite-dir reports/v2_fc_paper_suite \
    --overview-timeseries aaai27/figures/data/timeseries_overview.npz
```

## 图 1(c) 溯源

npz 由 `../scripts/export_timeseries.py` 从 checkpoint 重放单个窗口导出，`meta` 字段自带
完整溯源：种子 7、`test_id/encoder_dropout`、参与者 BT16、trial
`dynamic_walk_1_1_high-knees_on`，门控策略 `val_utility_aware`。这与 caption 的声明一致。

## 视觉规范

四角色分类色（助力蓝 / 传感紫 / 故障红 / 力矩绿），蓝色顺序渐变表示阶段量，以 0.5 为中心的
蓝-灰-红发散渐变表示 AUROC。每组分类对都带第二编码（marker 形状、填充或直接标注），
保证色觉障碍下可读。字体嵌成 TrueType（`pdf.fonttype=42`）而不是 Type 3。

四张全宽图按 AAAI 正文宽度 7.0 in（`figure*`）出，`fig_stress_transfer` 按单栏 3.35 in（`figure`）出。

## 已完成：服务器侧的 S1 / S2（2026-07-27）

两项都做完了，全部 `v2_fc_*` 已按新口径重评一遍（纯 eval，复用现成 checkpoint，没有重训；
`RESULTS_PROVENANCE.md` 末节的「别删主 checkpoint」依然有效）。落地情况：

| 项 | 代码落点 | 状态 |
| --- | --- | --- |
| S1 `E_wrong` | `reliability/metrics.py:441`（能量）与 `:448`（返回 `*_wrong_energy`） | 已进汇总与正文 §Safety--Utility |
| S1 峰值 / jerk | 同一函数已有的 `peak_wrong_torque`、`mean_abs_jerk`，补进 `summarize_reliability_suite.py` | 已有数字支撑 |
| S2 Detector-online | `run_reliability_experiment.py:585` `select_detector_signals_on_val` 在 `K_gate` 内**重选**一次（不是事后过滤），`--detector-online-signals` 给候选池 | 摘要与主结论已改用 online |
| S2 出图 | `draw_aaai_figures.py` 检测热图多一列融合口径；图 5(a) 实心 = all、空心 = online | 已重出 |

重跑规模 44 份报告（main / baseline / ablation × 3 seed、stress × 3 seed、loso 15 折）。
运行时踩到的坑记一笔：eval 进程按 256 核推 OMP 线程池，每进程 135 线程，22 个并发就是
3232 线程抢 256 核，表现为「进程在跑但一份报告都不落地」。派发前必须
`export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8`，
`refresh_forecast_suite.sh` 自身不设线程数。另外 `SKIP_COMPLETED=1` 只判文件存在、不判新鲜度，
换口径重评时要显式传 `SKIP_COMPLETED=0`。

下面保留两项的原始需求描述，便于复核当初的验收标准。

### S1 —— 补 wrong-direction 能量指标 `E_wrong`

现在的安全指标只有比例（`wrong_direction_ratio`）和峰值（`peak_wrong_torque`）。审稿意见
是「多少步反向」不等于「反向了多大伤害」：一步 0.45 Nm/kg 的反向和一百步 0.01 的反向，
比例差 100 倍，实际风险相反。需要一个能量口径：

```
E_wrong = Σ_{(j,t)∈A} max(0, −τ^cmd_jt · τ*_jt) · Δt      # Δt = 1/200 s
```

- 落点：`reliability/metrics.py:_torque_metrics_for_command`，紧挨着现有四项返回值加
  `f"{prefix}_wrong_energy"`。`active`/`ideal` 都已在作用域里，改动约 3 行。
- 然后对 main 三 seed（7/13/23）跑 `--mode eval`，重跑 `summarize_reliability_suite.py`，
  `E_wrong` 会自动进 `core_aggregate.tsv`。
- `peak_wrong_torque` 和 `mean_abs_jerk` **已经算出来了**（同一函数），只是没进汇总表也没进
  正文。`summarize_reliability_suite.py` 要把这两列一起带出来，正文 §Safety--Utility 那句
  「Peak wrong torque and command jerk follow the same comparison」现在没有数字支撑。
- 正文的形式定义（active / wrong / retained 三个公式 + 「收益只来自 `g_t=0`」那句）已经写进
  `main.tex` §Metrics and Implementation，不需要服务器；只有 `E_wrong` 的数值要等这一步。

### S2 —— 拆 Detector-all / Detector-online，摘要改用 online-only AUROC

冻结的检测子集是 `residual, staleness, coherence`（见 `paper_gap_report.md:35`），但
**coherence 和 drift 不参与 command-time 门控**（`run_reliability_experiment.py:1055`,
`796`, `1114` 传 `drift=None`，`main.tex` 里 `K_gate` 也不含这两项）。所以现在摘要里的
0.849 / 0.773 是一个门控用不上的检测器的成绩——审稿人抓的就是这一点，而且 coherence 在
partial dropout 和 stuck_imu 上是 1.000，delay 行去掉它会掉到约 0.55，差距不是小数点后第三位。

要跑的是：

```bash
# 在服务器仓库根目录，DEVICE/DATA_ROOT 按 scripts/reliability_paper_v2.sh 的默认值
for seed in 7 13 23; do
  conda run --no-capture-output -n pytorch python run_reliability_experiment.py \
    --mode eval --checkpoint reports/v2_main_seed${seed}/reliability_tcn_best.pt \
    --output-dir reports/v2_fc_online_seed${seed} \
    ...（其余参数照抄 reliability_paper_v2.sh 的 stress_eval_run，换成 CORE_FAULTS）
done
```

需要一个只在 online 通道里选子集的开关（现在 `detector_aurocs` 会在全部八个候选里搜），
候选集限制成 `K_gate = {fault, ale, rec, pred, epi, stale}`，其余流程不变。产出两行：

| 检测器 | 候选通道 | 用途 |
| --- | --- | --- |
| Detector-all | 八个（含 coherence、drift） | 离线诊断上限，Table 2 保留 |
| Detector-online | 六个（`K_gate`） | 门控实际可用，**摘要和主结论改用这一组** |

改完要同步：摘要四个 AUROC、§Fault Detection under Task Shift 的四个数、
`fig_detection_taxonomy` 的融合列（`draw_aaai_figures.py` 要多读一列）、
`RESULTS_PROVENANCE.md` 的「主要数字」表。中文版 `main_zh.tex` 同步。

## 投稿前

- 检查 `[TBD]` 占位符和内部 TODO 注释是否清完。
- 检查 log 里的 overfull box。
- 确认页数符合 AAAI-27 限制，浮动体没有跑到章节外。
- 不要修改官方 `.sty` / `.bst`。

**页数规则（官方投稿说明）**：正文最多 7 页，第 8–9 页只能放参考文献，
全文不超过 9 页；伦理声明算在 7 页正文内，附录另投 Supplementary Document
（正文 7 月 28 日截稿，附录 7 月 31 日）。

开发机已装 TinyTeX，前三项现已本机验证：

```bash
export PATH=$HOME/bin:$HOME/.TinyTeX/bin/x86_64-linux:$PATH
make            # 英文，pdflatex
make zh         # 中文，xelatex
```

当前编译结果：英文 8 页（正文 1–7 页，参考文献从第 8 页开始），overfull 0、
underfull hbox 0、undefined 0、error 0；中文 14 页、四项全 0。
验收方式见 `RESULTS_PROVENANCE.md`「7 页版面合规」一节——只数「第 8 页有没有正文词」，
不要用词数系数估算。
