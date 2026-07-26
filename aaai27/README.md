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

## 投稿前

- 检查 `[TBD]` 占位符和内部 TODO 注释是否清完。
- 检查 log 里的 overfull box。
- 确认页数符合 AAAI-27 限制，浮动体没有跑到章节外。
- 不要修改官方 `.sty` / `.bst`。

以上三项需要本地 TeX 发行版。当前开发机未安装 TeX Live，这三项**尚未验证**。
