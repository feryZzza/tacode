# 结果溯源与验收状态

记录稿件里每个数字的来源、如何复算，以及哪些项还没验证。

## 数据来源链

```
checkpoint (reports/v2_*/reliability_tcn_best.pt)
  └─ run_reliability_experiment.py      →  reports/v2_fc_*/reliability_report.json
       └─ summarize_reliability_suite.py →  reports/v2_fc_paper_suite/{paper_numbers.json, table*.tsv, loso_effect.json}
            ├─ scripts/draw_aaai_figures.py  →  aaai27/figures/*.pdf
            └─ 手工核对                        →  main.tex / main_zh.tex
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

| 量 | 值 | 来源键 |
| --- | --- | --- |
| 宏平均 AUROC，已见故障 / ID | 0.849 ± 0.005 | `table2_detection.macro_fused_auroc.test_id.seen` |
| 宏平均 AUROC，已见故障 / 留出任务 | 0.773 ± 0.002 | `…test_ood.seen` |
| 宏平均 AUROC，未见故障 / ID | 0.842 ± 0.010 | `…test_id.unseen` |
| 宏平均 AUROC，未见故障 / 留出任务 | 0.778 ± 0.004 | `…test_ood.unseen` |
| LOSO 逐参与者降低量 | 0.44 pp，95% CI [0.23, 0.67] | `loso_effect.json` |
| LOSO 折数 / 为正折数 | 15 / 15 | 同上 |
| 干净试验同向力矩保留 | 96.1% | `clean_retention_mean` |

已见故障六种：`insole_missing`、`encoder_dropout`、`imu_bias`、`packet_loss`、
`stuck_imu`、`sensor_delay`。未见故障三种：`packet_loss_burst`、
`packet_loss_partial`、`sensor_delay_jitter`（模型训练与检测器选择都没用过）。

正文（`main.tex:323-324`）、图 `fig_detection_taxonomy` 与 suite 三者一致，已逐 token 核对。

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
| 页数是否符合 AAAI-27 限制 | **未验证** | 开发机无 TeX Live，本轮不装（用户选「跳过，先做别的」）；瘦身后需在产出 `tacode.pdf` 的机器上重编译 |
| overfull box | **未验证** | 同上 |
| 浮动体落位 | **未验证** | 同上 |
| 图 PDF 的 MediaBox 宽度 | 已核 | 五张导出图的 `pdfinfo` 宽度均 ≤ `figsize`，最宽 494 pt = 6.86 in |
| `tests/test_forecast_residual.py` | 已通过（5 case） | 2026-07-26 本机 |
| `tests/test_gate_selection.py` | 已通过（2 case） | 同上 |

页数 / 溢出框 / 浮动体三项需要 `pdflatex`（英文）和 `xelatex`（中文），在有 TeX 的机器上跑
`make` 与 `make zh`。

单测在开发机上跑通，7/7。系统 `python3` 没有 torch，用 conda 的 pytorch 环境；该环境没装
pytest，所以走 unittest：

```bash
/home/fery/anaconda3/envs/pytorch/bin/python -m unittest discover -s tests -t . -v
```

有 pytest 的环境（如训练机）用 `python -m pytest tests/ -q` 等价。

## 已复核的正文数字（2026-07-26，开发机）

按 §13 的映射表把 `main.tex` 与 `paper_numbers.json` / `table*.md` / `loso_effect.txt`
逐位对照，全部一致，无需回填：

| 正文位置 | 稿件值 | suite 值 |
| --- | --- | --- |
| Abstract 未见故障宏平均 AUROC（ID / OOD） | 0.842 / 0.778 | 0.8417 / 0.7780 |
| Abstract clean RMSE 增量（ID / OOD） | +0.011 / +0.021 | +0.0110 / +0.0213 |
| Abstract 反向力矩降幅、干净保留 | ≈0.7 pp、96.1% | 0.00736、0.9607 |
| Abstract LOSO | 0.44 pp，CI [0.23,0.67]，15/15 | 0.440 pp，[0.227,0.674]，15/15 |
| `:323`–`:324` 四个宏平均 AUROC | 0.849/0.773/0.842/0.778 | 一致 |
| `:340`–`:341` 保留率与四个 wrong delta | 0.960/0.961，−0.0005/−0.0074/−0.0052/−0.0109/−0.0017/−0.0103 | 一致 |
| `:343` 平均门控区间 | 0.904–0.978 | `gate_mean_range_safety_rows` 0.9043–0.9781 |
| `:358` 确定性阶段种子散布 | 0.831 ± 0.036 | 0.8310 ± 0.0361 |
| `:372`–`:373` stress 极值六项 | 见正文 | `stress_max_severity` 一致 |
| `:375` action 消融三组区间 | 0.842–0.847 / 0.771–0.784 / 0.259–0.278 | 一致 |

注意 `:343` 那句的口径：0.904–0.978 来自 `table3_safety` 的六行聚合
（clean / seen / unseen × ID / OOD），**不是** `table2_safety.md` 里逐故障行的范围
（那个是 0.794–0.979）。核对时别拿错表。

参考文献 24 条，与 `main.tex` 的 `\cite` 键集合完全一致，无未引用项、无缺失项。

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
在位，中英文 caption 都已改成真实导出的表述。`EXPERIMENTS_TODO.md` §10 仍按「P3 可选、
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

`main_zh.tex` 已同步以上全部改动（含面板 (d) caption、消融段落、Limitations 首句）。
两份 `.tex` 结构检查通过：环境配对、花括号配对、无悬空 `\ref`、无未用 `\label`。

审稿意见里的第 2、3 项各有一半要在服务器上跑（`E_wrong` 需纯 eval 重跑；Detector-all /
Detector-online 拆分需 checkpoint），已写进 `README.md` 的「待办：需要在服务器上跑」。

## 服务器侧建议保留

主 checkpoint 不要删。补测某个故障或某个指标时，有 checkpoint 是几十分钟纯 eval；
没有就得重训。`stuck_imu` 那次漏评测就是靠 checkpoint 补回来的。
