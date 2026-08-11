# AAAI-27 稿件

从故障证据到控制后果的任务无关外骨骼可靠性评估。只维护英文投稿稿件，图表全部由
脚本从实验汇总目录确定性生成。

## 文件

- `main.tex` — 英文投稿主文件，单文件匿名格式，符合官方 author kit 要求。
- `references.bib` — 参考文献。
- `aaai2027.sty`、`aaai2027.bst` — AAAI-27 官方 author kit（2026 年 5 月版）原文件，不要修改。
- `figures/` — 投稿用矢量 PDF，附 SVG 源和 600 dpi PNG 预览。
- `figures/data/timeseries_overview.npz` — 图 1(a) 与图 2(b) 的逐 timestep 导出数据。
- `supplement.tex` — 匿名补充材料，另投 Supplementary Document。
- `ReproducibilityChecklist.tex` — AAAI-27 官方清单原题与逐项回答，单独提交。

以下是仓库内的工作笔记，**不随稿件提交**：

- `REPRODUCIBILITY.md` — 运行顺序、数据布局与核验方法。
- `RESULTS_PROVENANCE.md` — 每个稿件数字对应的 run 目录与提取路径。
- `FIGURE_QA.md` — 图件数据来源、统计语义和 AAAI 可读性审计。
- `OUTLINE.md`、`DATA_REQUIREMENTS.md`、`EXPERIMENTS_TODO.md` — 写作与实验规划笔记。

构建产物（`*.aux`/`*.bbl`/`*.blg`/`*.log`/`*.pdf`）不入库，`make clean` 会清掉；
`figures/*.pdf` 是例外，入库以便无 TeX 环境时也能核对图件。

## 编译

```bash
make          # 出图 + 英文版
make figures  # 只出图
make submission  # 英文正文 + 官方复现清单 + 匿名补充材料
```

投稿稿件使用 `pdflatex`。

## 数字从哪来

出图脚本是 `../scripts/draw_aaai_figures.py`。所有数字优先从汇总目录读取：

| 面板 | 数据源 |
| --- | --- |
| 检测热力图、宏平均 AUROC | `table1_detection.tsv`、`paper_numbers.json` |
| 控制后果、指令保留、力矩变化率 | `table2_safety.tsv`、`paper_numbers.json` |
| 消融、压力迁移 | `paper_numbers.json` |
| LOSO 逐折效应量 | `loso_effect.json` |
| 图 1(a) teaser 与图 2(b) 门控时序 | `figures/data/timeseries_overview.npz` |

默认汇总目录是 `../reports/v2_fc_paper_suite`（`SUITE_DIR` 可覆盖）。`reports/` 不入版本库，
从指标包解开：

```bash
tar xzf ../aaai27_metrics.tar.gz -C ..
```

脚本每次运行都会打印数据来源，**这一行是唯一可靠的验收凭据**：

```
data sources: detection=suite, macro=suite, safety=suite, ablation=suite, stress=suite, loso=suite, overview=export, teaser=export
```

八项全是 `suite`/`export` 才是对的。出现 `fallback` 说明汇总目录没读到，图里是脚本内置的
`FALLBACK` 常量（旧数字）；出现 `overview=schematic` 说明 npz 没读到，图 2(b) 退成了示意
曲线，与 caption 声明的「真实导出」不符；出现 `teaser=absent` 说明第一页那张图没画出来。
**这三种退化都不会报错，退出码仍是 0。**

`make figures` 和 `make check-figures` 已经把路径钉死并在缺文件时主动失败。如果直接调脚本，
注意它的默认路径是相对仓库根目录的，必须显式给参数：

```bash
python3 scripts/draw_aaai_figures.py \
    --output-dir aaai27/figures \
    --suite-dir reports/v2_fc_paper_suite \
    --overview-timeseries aaai27/figures/data/timeseries_overview.npz
```

## 图 1(a) / 图 2(b) 溯源

npz 由 `../scripts/export_timeseries.py` 从 checkpoint 重放单个窗口导出，`meta` 字段自带
完整溯源：种子 7、`test_id/encoder_dropout`、参与者 BT16、trial
`dynamic_walk_1_1_high-knees_on`，门控策略 `val_utility_aware`。这与 caption 的声明一致。

## 视觉规范

四角色分类色（助力蓝 / 传感紫 / 故障红 / 力矩绿），蓝色顺序渐变表示阶段量，以 0.5 为中心的
蓝-灰-红发散渐变表示 AUROC。每组分类对都带第二编码（marker 形状、填充或直接标注），
保证色觉障碍下可读。投稿 PDF 中的字形转成轮廓，因而既没有 Type 3，也没有 AAAI
author kit 要求移除的 Identity-H 字体；可编辑文字保留在配套 SVG 中。所有图内文字按
8.5 pt 生成（正文 10 pt），线宽至少 0.5 pt。

**图内字号看的是纸面磅值，不是代码里的磅值。** `bbox="tight"` 会把画布裁到内容边界，
原生宽因此不等于 `figsize`；再按 `\includegraphics[width=...]` 缩放，纸面字号就是
`代码磅值 x 排版宽 / 原生宽`。三张正文图曾被裁成 5.90 / 6.88 / 6.55 in 又统一按
`0.72\textwidth` 放置，同一个 9.2 pt 印出来是 7.86 / 6.74 / 7.08 pt。现在出图时
`fit_canvas_to_print_width` 把画布收敛到「裁剪后正好等于排版宽」，`assert_prints_one_to_one`
在原生宽偏离超过 0.01 in 时直接失败，七张图实测 8.48–8.54 pt。

五张跨栏图按 AAAI 正文宽度 7.0 in（`figure*`）出；`fig_stress_transfer` 与 teaser 按单栏
3.35 in（`figure`）出。teaser 只能是单栏：`figure*` 是双栏浮动体，LaTeX 不会把它排到带
`\twocolumn` 标题块的第一页，通栏版本必然被推到第 2 页。
`.tex` 里的宽度必须和脚本里传给 `save()` 的 `print_width_in` 一致，改一处就要改另一处。

## 已完成：服务器侧的 S1 / S2（2026-07-27）

两项都做完了，全部 `v2_fc_*` 已按新口径重评一遍（纯 eval，复用现成 checkpoint，没有重训；
`RESULTS_PROVENANCE.md` 末节的「别删主 checkpoint」依然有效）。落地情况：

> **2026-07-28 最终门控审计更新：**S1/S2 的主模型与检测数字仍有效，但后续第一轮
> §2--§5 门控审稿实验误把 Detector-gate 的 validation-selected AUROC 子集当成实际
> 六通道执行门控，并把 aligned-command adequacy 写成 retention。正文现已把该批结果
> 降为探索性证据；最终六通道纯评测及自动验收命令见 `EXPERIMENTS_TODO.md`。无需重训
> 三个主模型。

| 项 | 代码落点 | 状态 |
| --- | --- | --- |
| S1 $X_{\mathrm{opp}}$ | `reliability/metrics.py:_torque_metrics_for_command` 计算反向力矩乘积积分，同时保留旧 JSON 键兼容历史报告 | 已进汇总与正文 §Safety--Utility |
| S1 峰值 / torque rate | 同一函数计算 `peak_wrong_torque`、`mean_abs_torque_rate`，同时保留旧键兼容历史报告 | 已有数字支撑 |
| S2 Detector-gate | `run_reliability_experiment.py:select_detector_signals_on_val` 在当前门控实现的 $\mathcal K_{\rm gate}$ 内重新选择；它是评估口径，实际 gate 使用该集合全部可用通道 | 摘要与主结论采用此口径 |
| S2 出图 | `draw_aaai_figures.py` 检测热图并列 Detector-all / Detector-gate；压力图实心 = all、空心 = gate | 已重出 |

重跑规模 44 份报告（main / baseline / ablation × 3 seed、stress × 3 seed、loso 15 折）。
运行时踩到的坑记一笔：eval 进程按 256 核推 OMP 线程池，每进程 135 线程，22 个并发就是
3232 线程抢 256 核，表现为「进程在跑但一份报告都不落地」。派发前必须
`export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8`，
`refresh_forecast_suite.sh` 自身不设线程数。另外 `SKIP_COMPLETED=1` 只判文件存在、不判新鲜度，
换口径重评时要显式传 `SKIP_COMPLETED=0`。

下面保留两项的原始需求描述，便于复核当初的验收标准。

### S1 —— 补 opposition-weighted torque-product integral $X_{\mathrm{opp}}$

现在的安全指标只有比例（`wrong_direction_ratio`）和峰值（`peak_wrong_torque`）。审稿意见
是「多少步反向」不等于「反向了多大伤害」：一步 0.45 Nm/kg 的反向和一百步 0.01 的反向，
比例差 100 倍，暴露幅值却不同。需要一个幅值敏感的离线代理量：

```
X_opp = Σ_{(j,t)∈A} max(0, −τ^cmd_jt · τ*_jt) · Δt      # Δt = 1/200 s
```

- 落点：`reliability/metrics.py:_torque_metrics_for_command`，主键为
  `f"{prefix}_wrong_torque_product_integral"`；`active`/`ideal` 在同一作用域。
  历史结果仍使用 `*_wrong_energy`，代码保留该别名以便复算。
- 然后对 main 三 seed（7/13/23）跑 `--mode eval`，重跑 `summarize_reliability_suite.py`，
  $X_{\mathrm{opp}}$ 会自动进 `core_aggregate.tsv`。
- `peak_wrong_torque` 和 `mean_abs_torque_rate` **已经算出来了**（同一函数）；汇总脚本
  把这两列一起带进论文结果。历史报告的 `mean_abs_jerk` 只作为兼容别名保留。
- 正文的形式定义（active / wrong / retained 三个公式 + 「收益只来自 `g_t=0`」那句）已经写进
  `main.tex` §Metrics and Implementation，不需要服务器；只有 $X_{\mathrm{opp}}$ 的数值要等这一步。

### S2 —— 拆 Detector-all / Detector-gate，摘要改用门控候选池 AUROC

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
| Detector-gate | 六个（`K_gate`） | 当前评测门控实际实现的候选池，**摘要和主结论改用这一组** |

改完要同步：摘要四个 AUROC、§Fault Detection under Task Shift 的四个数、
`fig_detection_taxonomy` 的融合列（`draw_aaai_figures.py` 要多读一列）、
`RESULTS_PROVENANCE.md` 的「主要数字」表。

## 投稿前

- 检查 `[TBD]` 占位符和内部 TODO 注释是否清完。
- 检查 log 里的 overfull box。
- 确认页数符合 AAAI-27 限制，浮动体没有跑到章节外。
- 不要修改官方 `.sty` / `.bst`。

**页数规则（官方投稿说明）**：正文最多 7 页，第 8–9 页只能放参考文献，
全文不超过 9 页；伦理声明算在 7 页正文内，附录另投 Supplementary Document
（正文 7 月 28 日截稿，附录 7 月 31 日）。

验收命令（TinyTeX 在本机 `$HOME/.TinyTeX`）：

```bash
export PATH=$HOME/bin:$HOME/.TinyTeX/bin/x86_64-linux:$PATH
make submission
pdfinfo main.pdf | grep Pages                        # 必须 <= 9
pdftotext -f 8 -l 8 main.pdf - | grep -n References  # 必须是 1:References
grep -c Overfull main.log                            # 必须是 0
```

当前封版结果：`main.pdf` 8 页，正文与伦理声明止于第 7 页，参考文献自第 8 页第一行开始；
overfull / undefined / error 均为 0；三个 PDF 均无 Type 3 字形、字体全部嵌入、元数据不含
作者信息。`supplement.pdf` 7 页，`ReproducibilityChecklist.pdf` 2 页。消融与压力/LOSO
全图在补充材料中，主文保留对应数值和统计限定。

改动稿件后必须重新执行上面整套验收，不能沿用上一版页数——正文接近 7 页上限，一句话的
增删就会把参考文献推离第 8 页首行。
