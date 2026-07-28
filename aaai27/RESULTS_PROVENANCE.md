# 结果溯源与验收状态

记录稿件里每个数字的来源、如何复算，以及哪些项还没验证。

## 2026-07-28 最终门控一致性审计

最新代码审计确认：2026-07-27 完成的主模型、Detector-all/Detector-gate AUROC、稳态安全
表、stress 与 LOSO 数字仍可复算；但随后第一轮 §2--§5 审稿实验有两处口径错误，不能
作为六通道实际门控的最终证据：

1. `GatePolicy.from_report` 读取了 `_detector_policy.online_signals`，导致 downstream
   gate-baseline、dynamics、transient 与 paired analysis 实际评估 Detector-gate 的
   validation-selected AUROC 子集，而不是论文定义的六通道执行门控；
2. 历史 `*_retained_aligned_torque` 以 ideal command 为分母，语义是
   aligned-command adequacy；论文的 retention \(\kappa\) 应以 ungated estimator 为
   分母。第一轮 retention-matched random control 也只置换 gate 值，没有严格匹配
   \(\kappa\)；
3. 第一轮 gate-reselection driver 使用了 `stairs,ramp`，而主论文报告保存的
   held-out tasks 是 `jump,cutting,lift_weight,lunges`，因此其重选工作点也必须按原
   split 重跑。

本地已修复 policy/metric 语义、精确 retention-matched random control、统一
reselected-policy 流水线和交付 validator。稿件把第一轮 §2--§6 数字明确标成
post-hoc exploratory，并移除 oracle upper-bound、全栈支配及瞬态范围等过强或错误表述。
最终数字必须按 `EXPERIMENTS_TODO.md` 的 P0 在服务器纯 eval 后再写回；不需要重训主模型。
当前容器没有 `pdflatex`，而 AAAI 官方样式拒绝 XeTeX/Tectonic；因此本轮英文源文件只完成
结构、引用和数字键检查，页数、浮动体和 overfull 必须在有 pdfTeX 的环境重新验收。

## 最终 AAAI-27 验收口径（2026-07-27）

本节及文首表格记录最终提交状态；后文保留的多轮排版记录仅用于审计修改
过程，若与本节冲突，以本节为准。最终英文稿共 8 页：正文与 Ethical
Statement 均在第 7 页结束，参考文献占第 7--8 页。主文保留图 1--3 和表 1；
消融、压力测试及 LOSO 汇总图移入匿名补充材料。最终稿没有使用
`\vspace`、`\resizebox`、浮动比例重定义或其他压缩版面命令。

最终构建的正文、补充材料和复现清单均为 US Letter、PDF 1.7；字体全部嵌入，
没有 Type 3 或 `Identity-H` 字体。正文无 overfull box、未定义引用或编译错误，
并且不含超链接、书签、附件、作者元数据或外部材料链接。

## 数据来源链

```
checkpoint (reports/v2_*/reliability_tcn_best.pt)
  └─ run_reliability_experiment.py      →  reports/v2_fc_*/reliability_report.json
       └─ summarize_reliability_suite.py →  reports/v2_fc_paper_suite/{paper_numbers.json, table*.tsv, loso_effect.json}
            ├─ scripts/draw_aaai_figures.py  →  aaai27/figures/*.pdf
            └─ 手工核对                        →  main.tex
       └─ scripts/export_timeseries.py   →  aaai27/figures/data/timeseries_overview.npz  →  图 1(c)
```

`reports/` 不入版本库（79 MB）。可复算的输入全部打进 `aaai27_metrics.tar.gz`：

```bash
tar xzf aaai27_metrics.tar.gz          # 从仓库根目录
sha256sum -c aaai27_metrics.tar.gz.sha256
```

包内 2.4 MB / 118 条：95 份 report JSON、22 个 suite 汇总文件、
`AAAI27_CODE_COMMIT.txt`、`AAAI27_GIT_STATUS.txt`、`AAAI27_REPORT_MANIFEST.txt`（114 行）。

## 主要数字

三个种子（7 / 13 / 23），参与者互斥划分。

检测器有两个评估口径：**Detector-all** 在八个候选通道里选子集（含 coherence / drift，
只能离线用），**Detector-gate** 把候选集限制在当前门控实现的 $K_{gate}$
六通道内重选一次。摘要与正文结论以 Detector-gate 为准。实际 gate
不复用该评估子集，而是融合 $K_{gate}$ 中全部可用通道。
两者由同一次 eval 的同一批验证分数产生，见 `run_reliability_experiment.py`
的 `select_detector_signals_on_val`。

| 量 | Detector-gate | Detector-all | 来源键 |
| --- | --- | --- | --- |
| 宏平均 AUROC，已见故障 / ID | 0.815 ± 0.006 | 0.849 ± 0.005 | `table2_detection.macro_{online,fused}_auroc.test_id.seen` |
| 宏平均 AUROC，已见故障 / 留出任务 | 0.740 ± 0.002 | 0.773 ± 0.002 | `…test_ood.seen` |
| 宏平均 AUROC，未见故障 / ID | 0.812 ± 0.011 | 0.842 ± 0.010 | `…test_id.unseen` |
| 宏平均 AUROC，未见故障 / 留出任务 | 0.715 ± 0.002 | 0.778 ± 0.004 | `…test_ood.unseen` |
| 冻结子集 | residual, staleness | residual, staleness, coherence | `paper_gap_report.md` |

差距集中在延迟类故障（丢掉 coherence：`sensor_delay` ID 0.878→0.628、
OOD 0.774→0.557），其余故障上更小的子集不差甚至更好
（`packet_loss_burst` ID 0.743→0.886，干净侧误报更少）。

| 量 | 值 | 来源键 |
| --- | --- | --- |
| LOSO 逐参与者降低量 | 0.44 pp，95% CI [0.23, 0.67] | `loso_effect.json` |
| LOSO 折数 / 为正折数 | 15 / 15 | 同上 |
| 干净试验同向力矩保留 | 96.1% | `clean_retention_mean` |
| $X_{\mathrm{opp}}$，留出任务已见故障 | 0.0439 → 0.0298 $(\mathrm{Nm/kg})^2\mathrm{s}$ | `table3_safety.test_ood.seen_faults.wrong_energy_{ungated,gated}`（历史键名） |
| $X_{\mathrm{opp}}$，留出任务留出算子 | 0.0412 → 0.0270 | `…unseen_faults.…`（兼容历史字段名） |
| 反向指令峰值，留出任务已见故障 | 0.175 → 0.137 Nm/kg | `…peak_wrong_{ungated,gated}` |
| 平均绝对力矩变化率，留出任务已见故障 | 0.237 → 0.279 Nm kg$^{-1}$ s$^{-1}$ | `…jerk_{ungated,gated}`（历史键名） |

训练已见故障六种：`insole_missing`、`encoder_dropout`、`imu_bias`、`packet_loss`、
`stuck_imu`、`sensor_delay`。留出损坏算子三种：`packet_loss_burst`、
`packet_loss_partial`、`sensor_delay_jitter`（模型训练与检测器选择都没用过，
但仍属于已知丢包或延迟故障族，不作 open-set 故障声明）。

审稿核查新增的 clean-adjusted 对比由 `scripts/analyze_gate_specificity.py` 从
`core_runs.tsv` 确定性生成到 `gate_specificity.{json,md}`。错误方向比例的 fault /
matched-clean / adjusted 变化为 -0.736 / -0.397 / -0.339 pp；$X_{\mathrm{opp}}$
为 -0.0114 / -0.0044 / -0.0070 $(\mathrm{Nm/kg})^2\mathrm{s}$。均为 seed × task-split
匹配的场景汇总对比，不是逐窗口置信区间。

正文 §Fault Detection under Task Shift、主图 `fig_detection_taxonomy`（九种算子、
八个单通道、两个融合列）和补充图 `fig_detection_full`（18 个 split/operator 行）
均由 `table1_detection.tsv` 生成。
行号会随修改漂移，这里只记小节名。

## 出图脚本的两处静默退化

`scripts/draw_aaai_figures.py` 读不到输入时不报错、退出码 0，只在末行改口径：

- 汇总目录缺失 → 用内置 `FALLBACK` 常量（`:69`），宏平均 AUROC 会变成
  **0.826 / 0.760**。这是 forecast 口径修正前的旧数字，稿件里已不使用。
- npz 缺失 → 图 1(c) 退成示意曲线（`overview=schematic`），与 caption 声明的
  「真实导出」不符。

验收只看这一行，七项必须全是 `suite` / `export`：

```
data sources: detection=suite, macro=suite, safety=suite, ablation=suite, stress=suite, loso=suite, overview=export
```

`make figures` / `make check-figures` 已把路径钉死并在缺文件时主动失败。脚本默认路径相对
仓库根目录，从 `aaai27/` 直接调用而不给 `--overview-timeseries` 就会命中第二种退化。

## 图 1(c) 溯源

`scripts/export_timeseries.py` 从 checkpoint 重放单窗口导出，`npz` 的 `meta` 自带溯源：

| 字段 | 值 |
| --- | --- |
| checkpoint | `reports/v2_main_seed7/reliability_tcn_best.pt` |
| report | `reports/v2_fc_main_seed7/reliability_report.json` |
| split / fault | `test_id` / `encoder_dropout` |
| 参与者 / trial | BT16 / `dynamic_walk_1_1_high-knees_on` |
| 门控策略 | `val_utility_aware`（deadband 1.052，softness 2.0） |
| 检测器信号 | residual, staleness, coherence |

## 代码版本

`reports/AAAI27_CODE_COMMIT.txt` 记 `e498853`。forecast_residual 口径修正
`5d12366` 是它的祖先（`git merge-base --is-ancestor 5d12366 e498853` 通过），
所以记录的 commit 确实包含产出这批数字的代码。

## 尚未验证

| 项 | 状态 | 原因 |
| --- | --- | --- |
| 页数是否符合 AAAI-27 限制 | 上一版已核；2026-07-28 英文修订待重编 | 上一版为 8 页且正文止于第 7 页；当前环境无 `pdflatex` |
| overfull box | 上一版已核；当前待重编 | 上一版英文 0；本轮只完成静态结构检查 |
| 浮动体落位 | 上一版已核；当前待重编 | 上一版图 1 在 p3、表 1 在 p5、图 2 在 p6、图 3 在 p7 |
| 图 PDF 的 MediaBox 宽度 | 已核 | 五张导出图的 `pdfinfo` 宽度均 ≤ `figsize`，最宽 494 pt = 6.86 in |
| `tests/test_forecast_residual.py` | 已通过（5 case） | 2026-07-26 本机 |
| `tests/test_gate_selection.py` | 已通过（2 case） | 同上 |
| `tests/test_wrong_energy_and_online_detector.py` | 已通过（7 case） | 2026-07-27 本机 |

页数 / 溢出框 / 浮动体三项要用 `pdflatex` 验收。开发机已装 TinyTeX：

```bash
export PATH=$HOME/bin:$HOME/.TinyTeX/bin/x86_64-linux:$PATH
make
```

单测在开发机上跑通，14/14（S1/S2 新增 `test_wrong_energy_and_online_detector.py` 7 个 case）。
系统 `python3` 没有 torch，用 conda 的 pytorch 环境；该环境没装 pytest，所以走 unittest：

```bash
python3 -m unittest discover -s tests -t . -v
```

有 pytest 的环境（如训练机）用 `python -m pytest tests/ -q` 等价。

## 已复核的正文数字（2026-07-27 重跑后，开发机）

按 §13 的映射表把 `main.tex` 与 `paper_numbers.json` / `table*.md` / `loso_effect.txt`
逐位对照。S1（$X_{\mathrm{opp}}$）与 S2（Detector-gate；历史键名 online）落地后全部 `v2_fc_*` 重评过一遍，
Detector-all 的四个宏平均 AUROC 与重跑前逐位一致（说明改动没有动既有口径），
新增的 online 与能量数字见下。行号按本次修改后的 `main.tex`：

| 正文位置 | 稿件值 | suite 值 |
| --- | --- | --- |
| Abstract 未见故障宏平均 AUROC，online（ID / OOD） | 0.812 / 0.715 | 0.8119 / 0.7148 |
| Abstract 未见故障宏平均 AUROC，all（ID / OOD） | 0.842 / 0.778 | 0.8416 / 0.7779 |
| Abstract clean RMSE 增量（ID / OOD） | +0.011 / +0.021 | +0.0110 / +0.0213 |
| Abstract 反向力矩降幅、干净保留 | ≈0.7 pp、96.1% | 0.00736、0.9607 |
| Abstract LOSO | 0.44 pp，CI [0.23,0.67]，15/15 | 0.440 pp，[0.227,0.674]，15/15 |
| §Fault Detection 八个宏平均 AUROC | online 0.815/0.740/0.812/0.715；all 0.849/0.773/0.842/0.778 | 一致 |
| §Fault Detection 延迟通道代价 | 0.878→0.628（ID）、0.774→0.557（OOD）；burst 0.743→0.886 | `main_detection_aggregate.tsv` 一致 |
| 保留率与四个 wrong delta | 0.960/0.961，−0.0005/−0.0074/−0.0052/−0.0109/−0.0017/−0.0103 | 一致 |
| §Safety 幅值敏感四项 | $X_{\mathrm{opp}}$ 0.0439→0.0298、0.0412→0.0270、峰值 0.175→0.137、力矩变化率 0.237→0.279 | `table3_safety` 一致 |
| `:343` 平均门控区间 | 0.904–0.978 | `gate_mean_range_safety_rows` 0.9043–0.9781 |
| `:358` 确定性阶段种子散布 | 0.831 ± 0.036 | 0.8310 ± 0.0361 |
| `:372`–`:373` stress 极值六项 | 见正文 | `stress_max_severity` 一致 |
| `:382` 强度极值两个融合口径 | 丢包 0.961/0.914（两口径同）；延迟 all 0.899/0.788、causal 0.743±0.019 / 0.588±0.003 | `stress_max_severity.*.{fused,online}_auroc` 一致（n=3；online 为历史键名） |
| `:375` action 消融三组区间 | 0.842–0.847 / 0.771–0.784 / 0.259–0.278 | 一致 |

注意 `:343` 那句的口径：0.904–0.978 来自 `table3_safety` 的六行聚合
（clean / seen / unseen × ID / OOD），**不是** `table2_safety.md` 里逐故障行的范围
（那个是 0.794–0.979）。核对时别拿错表。

参考文献 27 条，与 `main.tex` 的 `\cite` 键集合完全一致，无未引用项、无缺失项。

出图脚本重跑确认无静默退化，末行七项全是 `suite` / `export`。

## 图面复核（2026-07-26，开发机）

五张图逐面板看过位图，修掉的排版缺陷（数字口径未变，只改几何）：

| 图 / 面板 | 问题 | 处理 |
| --- | --- | --- |
| 1(b) | 框宽是手填常量，`sensor window` / `causal TCN` 文字溢出到邻框；四条斜箭线斜穿蓝框边线；底部汇流线三段都带箭头，方向看着互相矛盾 | 框宽改成按实测文字宽度计算并按序排布，排不下直接报错；扇出改走竖直汇流线；转角段去掉箭头 |
| 1(a) | 标题过长，尾部伸到面板 (b) 的标签 `b` 上 | 标题缩短，并加 `assert_no_overlapping_titles` 拦截 |
| 2(b) | 组标题的 x 传的是数据值（≈0.77），在 blended 变换里被当成 axes 分数，标题被推到右边框并被裁 | 改用 axes 分数；x 范围按数值加余量，最左的 0.773 不再探出边框 |
| 3(b) | 标签碰撞盒按数据坐标比例算、`annotate` 按排版点落位，两者差数倍，判定不重叠却叠成一团；边缘点标签被挤到对侧、拉长引线横穿画面 | 碰撞盒与偏移统一用排版点；候选方向按背离点云中心排序；标记与参考线注记都登记为障碍 |
| 3(c) | x 轴标签 `mean gate $g_t$, active steps` 比面板宽，被 `bbox=tight` 静默裁成 `active step` | 标签缩短（口径在 caption 里）；新增 `assert_no_clipping`，出图时量每个轴标签是否越过裁剪框，越界就报错 |
| 4(a) | `+0.024` / `-0.001` 挤在一起，`-0.001` 压着连线 | 差值标签沿垂直方向逐步找空位，数据点与误差棒分别占位 |
| 4(b) | `deterministic reference` 横排压在虚线和柱体上 | 先改竖排放在虚线右端，2026-07-27 那轮直接删掉该注记，含义移进 caption（见下面「版面瘦身」） |

`assert_no_clipping` 和 `assert_no_overlapping_titles` 在 `save()` 里无条件执行，所以
以后改标签文案或 `width_ratios` 时，这两类缺陷会让 `make figures` 直接失败，而不是
产出一张缺字或叠字的图。

`make check-figures` 与 `make figures` 重跑通过，末行七项仍全是 `suite` / `export`；
单测 7/7。

## 编译稿复核（2026-07-27，依据另一台机器编译出的 tacode.pdf）

之前记为「未验证（开发机无 TeX Live）」的三项——页数、溢出框、浮动体位置——这次是
按编译出的 10 页 PDF 逐页看的。单张图看不出来的问题，排进双栏后才暴露：

| 位置 | 问题 | 处理 |
| --- | --- | --- |
| 1(a) | 三条图例横排一行比面板宽 269 px，尾部伸进面板 (b) 的框图，`Insole force` 看着像 (b) 的说明 | 图例改竖排一列（宽 235 px），放在上方留白带 |
| 1(c) | 两列图例比面板宽 97 px，右轴那条竖线正好从 `gate $g_t$` 的下标上穿过 | 图例改竖排一列（宽 242 px），落在 1.18–1.52 留白带 |
| 1 脚注 | 手填的 `fig.text(0.5, -0.045)` 压在两个面板的 `time (samples)` 上，两行字叠在一起 | 新增 `figure_note()`：量出所有面板内容的最低边缘再往下让 3 pt |
| 5(a) | 组标题 x 传 −0.02（axes 分数），跨过左边框压在 y 轴刻度上 | 改成 0.02，与图 2(b) 的组标题一致 |
| 5(b) | 两条图例并排比单栏面板宽 39 px，右端 `CI` 越出边框 | 字号降到 5.4 pt（宽 684 px），文案不变 |
| 浮动体 | 第 8、9 页没有任何正文，只放着图 4 和图 5：正文第 7 页就结束，整篇却占 10 页 | 三个 figure 环境移到首次引用它的段落之前；单栏的图 5 放宽成 `[tb]` |

新增 `assert_legends_inside_panels`，同样在 `save()` 里无条件执行：图例横向溢出面板
即报错。纵向不查——图例常故意放在轴下方，那是有意的。已用一个窄面板 + 长标签的
最小例子确认守卫会触发。

浮动体那条是这次唯一改 `.tex` 的：LaTeX 不会把浮动体排到声明点之前，声明得比引用晚
就只剩末尾几个页顶可用。`figure*` 不能用 `[b]`（那要 `stfloats`，AAAI 样式明确禁用），
所以只有单栏的图 5 能加 `b`。改后的页数需要在有 TeX 的机器上重编译确认。

任务 I（逐 timestep 导出）已完成，不再是可选项：`aaai27/figures/data/timeseries_overview.npz`
在位，英文 caption 已改成真实导出的表述。`EXPERIMENTS_TODO.md` §10 仍按「P3 可选、
保留示意图」写，那是执行前的计划口径，已过期。

## 版面瘦身与审稿意见回应（2026-07-27 第二轮）

上一轮把浮动体声明提前之后，编译稿仍是 10 页。按 `tacode.pdf` 逐页量：正文到第 7 页
结束，第 8、9 两页只有图 4 和图 5（各 212 和 169 个词的 caption），参考文献在第 10 页。
所以超出的量里约 23 栏英寸是浮动页留白（上一轮已处理），另外约 12 栏英寸是真实体积。
两条轴一起压：

| 轴 | 动作 | 量 |
| --- | --- | --- |
| 图高 | 四张图的 `figsize` 高度下调：overview 2.1→1.95、detection 2.45→2.15、safety 2.35→2.15、stress 3.45→3.05（`hspace` 0.85→0.78），ablation 保持 2.05 | 图形占版 26.1 → 24.1 栏英寸 |
| 正文 | 合并 LOSO 逐故障六个数字为一句趋势、归因段落与门控段落各并句、去掉三处过渡引导句 | 正文 3855 → 3723 词，按实测 44.9 词/栏英寸约 −2.9 栏英寸 |

44.9 词/栏英寸是从 `tacode.pdf` 第 2 页实测的（809 词占约 18 栏英寸），不是估的；后续
再要砍字可以直接用这个系数换算。

`figure_note()` 的 MediaBox 膨胀（新守卫 `assert_tight_within_canvas`）：

`savefig(bbox="tight")` 不裁超宽内容，而是把裁剪框撑大。图 4 的单行脚注实际宽 9.3 in，
画在 7.0 in 的画布上，导出的 PDF MediaBox 就变成 9.32 in；`\includegraphics[width=\textwidth]`
再把整张图连字号一起缩到约 75%。位图看起来完全正常，只有排进双栏才显小。修法两步：
`figure_note()` 按图宽实测折行（图 4 现为 6.86 in，两行），并新增第四个守卫
`assert_tight_within_canvas`，`get_tightbbox` 比 `figsize` 宽出 0.05 in 就报错。已用
2.0 in 画布 + 300 字符脚注确认会触发（报 `裁剪框宽 14.75 in 超过 figsize 的 2.00 in`）。

至此 `save()` 里无条件执行四个守卫：`assert_no_clipping`、
`assert_no_overlapping_titles`、`assert_legends_inside_panels`、
`assert_tight_within_canvas`。

同轮按审稿意见改的内容（数字口径未变）：

| 项 | 处理 |
| --- | --- |
| 安全指标形式定义 | §Metrics 补 `active`（判据用真力矩 $\|\tau^*\|>0.02$，与门控无关）、$w$、$\kappa$ 三个式子；点明 $\tau^{\mathrm{cmd}}=g_t\,\mathcal{C}(\mu)$ 且 $g_t\ge0$，故反向力矩的减少只能来自 $g_t=0$ |
| 图 4 新增面板 (d) | 各阶段在自身验证选定门控下的反向力矩变化（pp）。前三阶段恒为 0：选出的死区从未被测试风险超过，$g_t\equiv1$ |
| 消融解释 | 正文点明只看 AUROC 会把阶段 2 排第一，但它对执行零收益；重建残差入风险后才有 $-0.69\pm0.02$ pp，加预测后 $-0.62\pm0.19$ pp |
| 4(b) `deterministic reference` | 注记删除，含义写进 caption；阶段刻度改成裸数字，消掉与刻度标签的碰撞 |
| Limitations 补 OOD 混淆 | 明确写出干净留出任务上门控仍降 0.74 pp，与故障留出的 1.09 / 1.03 pp 同量级，所以聚合效应里有一部分是抑制普通任务偏移误差而非故障；分离需要当前协议没有的无故障 OOD 对照 |
| 可复现细节 | 种子 7/13/23、检测器与门控在验证集上冻结、burst 丢包 / 独立通道丢失 / 随机延迟抖动不参与训练与检测器选择 |

`main.tex` 结构检查通过：环境配对、花括号配对、无悬空 `\ref`、无未用 `\label`。

审稿意见里的第 2、3 项各有一半要在服务器上跑（$X_{\mathrm{opp}}$ 需纯 eval 重跑；
Detector-all / Detector-gate 拆分需 checkpoint），已写进 `README.md` 的执行记录。

## 全量重评的执行记录（2026-07-27，服务器）

S1 / S2 落地后所有 `v2_fc_*` 重评了一遍，44 份报告：main / baseline / 五个消融 × 三 seed、
stress × 三 seed、loso 15 折。纯 eval，复用 `reports/v2_main_seed*/reliability_tcn_best.pt`，
没有重训任何模型。新鲜度判据是报告 JSON 里含 `fault_auroc_online` 键（旧口径没有该键），
比 mtime 可靠：

```bash
grep -l fault_auroc_online reports/v2_fc_*/reliability_report.json | wc -l   # 期望 44
```

两个坑值得记住，都会让人误判「任务卡住」：

| 现象 | 真实原因 | 处理 |
| --- | --- | --- |
| 22 个并发任务跑了 20 分钟零报告，load 从 736 掉到 90 | eval 进程按 256 核推 OMP 线程池 → 每进程 135 线程，3232 线程抢 256 核，绝大多数线程在 futex 上等 | 派发前 `export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8`，降到 15 线程/任务、总 359（1.4×），报告随即恢复 |
| 派发器打印一堆 SKIP、0 job planned | `refresh_forecast_suite.sh` 的 `SKIP_COMPLETED=1` 只判 `reliability_report.json` 是否存在，不判是否按新口径产出 | 换口径重评必须显式 `SKIP_COMPLETED=0` |

另有一条诊断教训：主进程是 `do_wait` 包装层，自身 CPU 时间不增长，所以「主进程 CPU 没动」
不能推断任务停滞，要按进程树递归统计子进程 CPU。GPU 4–7 利用率只有 10–50%、显存 2–4 GB，
这批 eval 是 CPU / dataloader 瓶颈，不是 GPU 瓶颈；GPU 0–3 属于其他用户，全程未动。

`scripts/check_paper_numbers.py` 是本轮新增的核对工具：把稿件引用的数字和
`paper_numbers.json` / `loso_effect.json` 并排打印，键缺失直接标出来，替代逐个 grep 表格。
重跑后 37 项全部在位、与稿件逐位一致（`缺失键数量: 0`）。注意 suite 用的是**含点的扁平键**
（`macro_fused_auroc.test_id.seen` 是一个键，不是三层嵌套），按点拆路径取值会全部落空；
脚本里的 `get()` 因此每层都先试最长匹配。另外 `collect()` 输出的标准差字段名是 `sd` 不是 `std`。

图 5(a) 的一处渲染缺陷（数值口径未变，只改标记样式）：Detector-gate 原本画成白色填充的空心圆，
而丢包两行两个口径逐位相同，白填充把实心的 Detector-all 整个盖掉，看上去像 all 缺失。
改成透明填充（`markerfacecolor="none"`）+ 半径 6.0（实心 4.2），重合时呈同心圆、分离时各自可读。

## 逻辑一致性复核（2026-07-27 第三轮，开发机）

重跑后又逐段核了一遍论证链，查的不是数字对不对，而是数字与叙述是否互相矛盾。
七条里六条改了正文（英中同步），第七条挂起。数字口径全部未变。

| # | 问题 | 证据 | 处理 |
| --- | --- | --- | --- |
| 1 | 正文称幅值敏感指标降幅「两个划分的干净试验上都可忽略」，与 §Limitations 已承认的 OOD 混淆自相矛盾 | 干净留出 $X_{\mathrm{opp}}$ 0.0316→0.0236，降 25.3%，与故障侧 32%/34% 同量级；干净 ID 才是 0.0408→0.0400 | 拆成两句，ID 说可忽略、OOD 明写 25% 并指向局限性 |
| 2 | 消融用 Detector-all 的理由写成「前几个阶段缺少区分两个口径的通道」——事实相反 | coherence / drift 由输入直接算，五个阶段都有；缺的是 residual / forecast，而这两个恰属 online 通道。`fault_auroc_online` 在每个 stage × seed 都在位 | 改成「Detector-all 是各阶段自身检测器的选择池，故作归因基准」，并披露 `+recon` 一步两口径反号：all $+0.0235\pm0.0042$ vs online $-0.0130\pm0.0018$ |
| 3 | coherence 被三处标为「离线 / 窗口级」，同时 §Complementary 又称其为 causal | `reliability/metrics.py:107` 用因果 `cumsum` 滚动相关，输出 `[B,1,T]`，逐 timestep；真正窗口级的只有 `drift_score`（整窗最小二乘斜率）。`calibrate_risk` 的签名里根本没有 coherence 参数 | 改为：drift 是定义使然，coherence 是被评估系统的实现选择而非因果性限制；并点明 drift 从未被选中 |
| 4 | 「最强单通道是 coherence」 | 九种故障宏平均：staleness 0.785(ID)/0.796(OOD)，coherence 只有 0.658/0.663 | 改成「staleness 最广、coherence 最极端」，峰值 1.000 的说法保留 |
| 5 | 「靠选择性衰减而非近乎关断」与 §Metrics 的 $g_t=0$ 论证冲突 | $\tau^{cmd}=g_t\mathcal{C}(\mu)$、$g_t\ge0$，只有 $g_t=0$ 能改变 $w$ | 改成「由稀疏的完全关断时刻承担，中间档衰减的代价由保留率给出上界」 |
| 6 | 中英表格数不对等（zh 4 表 / en 1 表） | — | **挂起**：英文版版面不够，补表要先砍正文 |
| 7 | burst 丢包「去掉 coherence 反而更好」只说了「不含信息」 | coherence 在 burst 上 AUROC 0.055(ID)/0.054(OOD)，即干净侧分数系统性高于故障侧，max 融合把干净基线一起抬高 | 改成「不是无信息而是反号」 |

顺带修掉一处曾加错的交叉引用：`aaai2027.sty:234` 设 `\setcounter{secnumdepth}{0}`，小节不编号，
`Section~\ref{...}` 会印出过期计数器。所以正文一律用散文指路（「下文的故障强度分析」），
不给小节加 `\label`。

### 版面：本轮净增与回压

改完先量后果。按同一口径（去注释、去公式/表格/图环境、保留 caption）比 HEAD：

| 阶段 | 词数 | 相对 HEAD |
| --- | --- | --- |
| HEAD (`0b919df`) | 4179 | — |
| 六条修完 | 4693 | +514 词 ≈ +11.4 栏英寸 |
| 回压后 | 4667 | +488 词 ≈ +10.9 栏英寸 |

回压只动措辞不动结论：合并 Related Work / Problem Formulation / Limitations 的并列短句、
压缩 $X_{\mathrm{opp}}$ 公式、图 4/5 的 caption、
删掉「执行器回放仅作离线代理」等与前句重复的收尾句。

这条已在下一节闭合：本机装上 TeX 后直接量页数，不再用 44.9 词/栏英寸估算。

`main.tex` 结构检查通过（无悬空 `\ref`、无未用 `\label`、环境与花括号配对、`$` 成对）；
`scripts/check_paper_numbers.py` 37 项在位、`缺失键数量: 0`。

## 7 页版面合规（2026-07-27，本机 TinyTeX）

之前所有版面判断都是拿 44.9 词/栏英寸估的，而且默认限制是 10 页。两个前提都要改。

### 真实限制是 7 页正文，不是 10 页

官方投稿说明：正文 PDF 最多 9 页，**第 8–9 页只能放参考文献**，即正文 ≤7 页。
伦理声明算在 7 页内，附录另投 Supplementary Document。按这条口径，
HEAD `0b919df`（9 页，正文到第 7 页尾但第 8 页仍有正文）和本轮修完的工作副本（10 页）
**都不合规**——这个限制从来没有满足过，之前只是没量。

### 工具链

conda 的 `texlive-core` 不可用（`find share/texmf-dist -name '*.sty' | wc -l` = 0，
只有二进制没有 texmf 树，`pdftex -ini` 找不到 `latex.ini`）；tectonic 也不行，
`aaai2027.sty` 显式要求 pdftex 而 tectonic 是 xetex。最后用 TinyTeX：

```bash
export PATH=$HOME/bin:$HOME/.TinyTeX/bin/x86_64-linux:$PATH
tlmgr install <pkg>     # 缺包时；包名一般就是 .sty 的基名（newtxtext → newtx）
```

### 从 10 页压到 8 页（正文 7 页）

| 动作 | 页数效应 | 内容代价 |
| --- | --- | --- |
| 放宽浮动体页顶占比（`\dbltopfraction` 等，`main.tex:23`） | −1 页 | **零**。图 4/5 原本被 LaTeX 默认的 0.7 页高上限推到一个几乎没有正文的浮动页上 |
| 删 5 处与前句重复的收尾/过渡句 | −1 页 | 无结论损失 |
| 五张图 `figsize` 高度再降一轮（overview 1.82→1.64、detection 1.98→1.78、safety 1.98→1.68、ablation 1.92→1.62、stress 2.80→2.44） | 与下一行合计 −1 页 | 零面板、零数字变化；四个 `save()` 守卫全过 |
| 正文逐句压缩：并句、把三个独立 equation 收成行内公式（$\mathbf{z}_t$ 归一化、感受野 $H$、总损失 $\mathcal{L}$、检测器 $\rho^{det}$/$R^{det}$） | 同上 | 所有论断、数字、披露（含重建一步两口径反号、OOD 混淆）逐条保留 |

**验收只看一件事**：第 8 页有没有正文词。

```bash
pdftotext -f 8 -l 8 main.pdf - | sed -n '1,3p'   # 第一行应当就是 References
```

当前状态：英文 8 页，正文 1–7 页（p1 701 / p2 783 / p3 716 / p4 827 / p5 883 /
p6 519 / p7 651 词），第 8 页只有参考文献；overfull 0、underfull hbox 0、
undefined 0、error 0；`\cite` 键与 `references.bib` 双向一致，24 条全部命中。
浮动体：图 1 p3、表 1 p5、图 2/3 p6、图 4/5 p7，无纯浮动页。

## 图例纵向压字（2026-07-27 第四轮，本机）

上一轮为了腾版面把五张图的 `figsize` 高度又降了一次。四个守卫全过、`data sources`
七项全对、退出码 0——但看位图发现三处图例压在轴标签上：

| 图 / 面板 | 压字量 | 处理 |
| --- | --- | --- |
| 5(a) | 图例压 xlabel 7.7 pt，`Detector-all AUROC` 与 `AUROC $\cdot$ retained aligned torque` 叠字 | **删掉该 xlabel**。三条图例已逐项写明横轴画的是什么，再加一行是同一件事说两遍；caption 里也已声明「三者共用同一坐标轴」。删标签比把图例推远省版面 |
| 5(b) | 图例压 xlabel 4.5 pt | xlabel 带单位不能删，图例 `bbox_to_anchor` 的 y 从 −0.62 降到 −0.92 |
| 2(b) | 图例首行字底离 xlabel 字顶只有 2 px | y 从 −0.42 降到 −0.52 |

另外 5(a) 的图例底边离 5(b) 的面板标题只差几像素（守卫按 1 pt 余量放行，但读起来连成
一块），`hspace` 0.78 → 0.95。

### 第五个守卫 `assert_legends_clear_of_text`

原来的 `assert_legends_inside_panels` 注释里写明「纵向不检查——图例常故意放在轴下方，
那是有意的」。放在轴下方确实是有意的，**放多远却没人量**，所以压缩图高时这类缺陷零成本
通过。新守卫改判据：不查「图例 vs 面板边框」，查「图例文字 vs 其他文字」（xlabel /
ylabel / 面板标题 / `fig.text` 脚注 / 另一个图例），横纵都查。同样在 `save()` 里无条件执行。

两个判据细节，写错任何一个守卫就没用：

- 量的是**图例条目文字的并集**，不是 `legend.get_window_extent()`。后者含
  `borderpad` / `handletextpad` 的空白边，图例还差十来像素才碰到 xlabel 就误报，
  逼着每张图把图例推远，白掉版面。
- 余量按排版点给（`1.0 * fig.dpi / 72`）。写成像素常量在 300 dpi 下只有 0.24 pt，
  边框相切就判成压字。

负例已验证：3.0×1.5 in 画布、图例 `bbox_to_anchor` y = −0.30 压在 xlabel 上时报
`图例 'series one' 压在文字 'a long x axis label' 上`。

### 图高回压与页数复核

图例往下挪会撑大 `bbox="tight"` 的裁剪框，正文被顶回第 8 页（`pdftotext -f 8` 首行变成
正文句子）。从 `figsize` 里扣回来：detection 1.78→1.70、stress 2.44→2.34。复编后：

- 英文 8 页，第 8 页首行 `References`；p1 685 / p2 863 / p3 748 / p4 931 / p5 1028 /
  p6 617 / p7 677 词。浮动体：图 1 p3、表 1 p5、图 2/3 p6、图 4/5 p7，无纯浮动页。
- 英文 overfull 0、undefined 0、error 0；`Underfull \hbox (badness 1014)` 1 处，
  在摘要末段 `main.tex:41--48`，是不满行的正常收尾，不是缺陷。
- 中文 14 页，overfull / missing character / undefined / error 全 0。
- 单测 14/14、`check_paper_numbers.py` 37 项在位（`缺失键数量: 0`）、`make check-figures` 通过。

数字口径全程未变，只改几何与一处冗余标签。

## 数值标注越界压邻面板（2026-07-27 第五轮，本机）

第四轮修完图例，再看图 4 的位图，发现另一类缺陷：`annotate` 出来的数值标签不受面板
边框约束，居中标在末柱/末点上时有一半探出右边框，而右邻面板的 ylabel 就贴在那儿。
五个守卫全过、退出码 0。图 4 压到 1.62 in 之后四个面板全中，实测间距：

| 标注 | 标注 x 区间 | 被压对象 | 间距 |
| --- | --- | --- | --- |
| (a) `0.813` | [662, 734] | (b) ylabel `clean OOD RMSE (Nm/kg)` x0=737 | 0.7 pt |
| (b) `0.2545` | [1019, 1106] | (c) ylabel `retained torque` x0=1120 | 0.5 pt |
| (c) `0.889` | [1443, 1515] | (d) ylabel `wrong-direction change (pp)` x0=1517 | 0.5 pt |
| (d) `-0.62` | [1833, 1898] | 面板右边框 x1=1890 | 探出 8 px |

四处一律改成右对齐到 `ax.get_xlim()[1]`（不再硬编码 5.55），偏移 −1…−2 pt。(d) 虽是最右侧
面板、不会压到别的面板，但探出边框会把 `bbox="tight"` 的裁剪框撑宽，等于连字号一起缩小
（第二轮记过这个机制），所以同样收回边框内。

右对齐会把标注推到本面板的数据上，两处要额外让位：

- (a) 的 `0.813` 右对齐后横向覆盖 x=5.0–5.85，正好压住末点的标记。改成走面板 (a) 已有的
  `reserve()` / `free()` 避让：从 `va="center"` 起试 ±8/±14/±20 pt，撞了就往外推，找不到
  空位直接报错而不是叠字画出来。
- (b) 的 `0.2545` 右对齐后横向占到第 4 根柱子上方，贴末柱顶画会盖住那根误差棒。把 `ylim`
  顶部按实测字高多留 1.35 行，标注抬到所有柱和棒之上，作为整条的右上角读数。

### 第六个守卫 `assert_annotations_clear_of_other_panels`

前五个守卫查的是「标签 vs 裁剪框」「标题 vs 标题」「图例 vs 面板边框」「图例文字 vs 其他
文字」，都不覆盖数值标注。新守卫判据：本面板的 `ax.texts` vs **其他面板**的 ylabel /
xlabel / 标题 / 刻度标签 / 数值标注，横纵都查，余量同样按排版点（`1.0 * fig.dpi / 72`）。

同面板内部**不查**——那是各图自己的避让逻辑（面板 (a) 的 `reserve`/`free`），在守卫里查会把
有意的紧凑排布判成缺陷。

负例已验证：3.0×1.4 in 双面板、左面板末柱上居中标 `0.8891`、右面板 ylabel
`wrong-direction change (pp)`，报
`面板标注 '0.8891' 压在邻面板的文字 'wrong-direction change (pp)' 上`。

至此 `save()` 里无条件执行六个守卫：`assert_no_clipping`、
`assert_no_overlapping_titles`、`assert_legends_inside_panels`、
`assert_legends_clear_of_text`、`assert_annotations_clear_of_other_panels`、
`assert_tight_within_canvas`。

### 复核

只改几何，数字口径未变（`0.813` / `0.2545` / `0.889` / `-0.62` 与 `main.tex:315` 一致）。

- 英文 8 页，第 8 页首行 `References`；p1 685 / p2 863 / p3 748 / p4 931 / p5 1028 /
  p6 617 / p7 677 词。浮动体：图 1 p3、表 1 p5、图 2/3 p6、图 4/5 p7，无纯浮动页。
- 英文 overfull 0、undefined 0、error 0；`Underfull \hbox` 那处（摘要末段）本轮已消失，
  现在 `\hbox` 计数为 0，只剩两条 `Underfull \vbox ... while \output is active`
  ——那是双栏页底留白，不是缺陷。
- 中文 14 页，overfull / missing character / undefined / error 全 0。
- 五张图逐张看过位图：无叠字、无越界、图例都在留白带里。
- 图 PDF 宽度（pt）：ablation 511、detection 474、overview 433、safety 445、stress 215，
  全部 ≤ `figsize`。
- 单测 14/14、`check_paper_numbers.py`「缺失键数量: 0」、`make check-figures` 通过、
  `data sources` 七项全是 `suite` / `export`。

## §1–§6 进正文 + 7 页正文合规（2026-07-28，本机）

新增的六组实验（工作点重选、匹配对照、限速/迟滞网格、瞬态、配对双重差分、反向消融）
这一轮全部写进正文，英文版从「正文溢到第 8 页 55 行」压回**参考文献从第 8 页第 1 行开始**。

### 验收口径

AAAI-27 的硬约束是「正文最多 7 页，第 8–9 页只放参考文献，伦理声明算正文」，
所以验收命令是这两条，而不是看总页数：

```bash
pdftotext -f 8 -l 8 main.pdf - | grep -n References   # 必须是 1:References
pdfinfo main.pdf | grep Pages                          # 必须 ≤ 9
```

现在：`1:References`、`Pages: 8`、overfull 0、undefined 0、warning 0。

### 怎么压下来的

只动措辞与版式，**没有删任何数字或结论**（删实质内容需要先定夺）：

- 把 6 个展示公式改成行内：预测分布、$\mathcal{L}_{\mathrm{mom}}$（aligned→单式）、
  $\mathcal{L}_{\mathrm{rec}}/\mathcal{L}_{\mathrm{pred}}$、$q^{\mathrm{rec}}/q^{\mathrm{pred}}$、
  $r_t$ 融合式、$g_t$ 增益式。$X_{\mathrm{opp}}$ 与 $J_{\mathrm{det}}$ 保留为展示式。
- 三张 `figure*` 宽度 `\textwidth` → `0.84\textwidth`。这是关键一步：
  0.9 时 References 落在 p8 第 19 行，0.87 → 6 行，0.84 → 4 行。
- 最后 4 行靠 6 处等义改写吃掉（结论、伦理声明、讨论、局限性、消融段）。

### 回到 0.9 图宽（同日追加）

图放大回 `0.9\textwidth` 后重新溢出 19→5 行，继续做纯措辞压缩到 **0:溢出**，
最终仍是 `1:References` / `Pages: 8` / overfull 0 / warning 0，图宽定在 `0.9\textwidth`。
关键一步是术语缩写：在 Experimental Setup 定义 `HT`（training-held-out）对 `ID`，
正文 9 处 “training-held-out” 换成 HT，表 1 表头同步改为 `Training-held-out (HT) tasks`。
其余是等义改写：Discussion 首句、Limitations 首段、Conclusion、消融段、匹配对照段。
数字与结论一个没删。

同轮验收（上文早期章节里的「单测 14/14」是当时的计数，现已扩到 49）：
`check_paper_numbers.py` 缺失键 0、`make check-figures` 通过、
`data sources` 七项全是 `suite`/`export`、`unittest discover` 49/49 OK、
三份投稿 PDF 分别 8 / 5 / 2 页且 overfull 全 0。

页面分布：p3 图 1、p5 表 1、p6 图 2+图 3、正文与伦理声明止于 p7、p8 全是参考文献。
p6 仍是双栏浮动页（332 词），`\dbltopfraction`/`\dblfloatpagefraction`/`[!t]` 都推不动它，
唯一有效的杠杆就是图宽。

### ReproducibilityChecklist 之前编译不过

`ts1-qtmr.tfm` 缺失导致 fatal error，`tlmgr install tex-gyre tex-gyre-math` 后 2 页正常。
「computing infrastructure」一项据实改成 `yes`，并把环境写进 supplement 的复现说明：
A100 80GB PCIe（驱动 570.211.01）、503 GB 内存、Ubuntu 22.04.5（Linux 6.2.1）、
Python 3.12 + PyTorch 2.11.0/CUDA 12.8，单 seed 训练 30–50 GPU-min（从 `logs/v2_reverse_*`
的起止时间实测），§1–§6 全部是纯 eval。

## 服务器侧建议保留

主 checkpoint 不要删。补测某个故障或某个指标时，有 checkpoint 是几十分钟纯 eval；
没有就得重训。`stuck_imu` 那次漏评测就是靠 checkpoint 补回来的。
