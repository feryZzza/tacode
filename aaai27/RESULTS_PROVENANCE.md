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
| 页数是否符合 AAAI-27 限制 | **未验证** | 开发机无 TeX Live |
| overfull box | **未验证** | 同上 |
| 浮动体落位 | **未验证** | 同上 |
| `tests/test_forecast_residual.py` | **未运行** | 缺 torch 与 pytest |
| `tests/test_gate_selection.py` | **未运行** | 同上 |

前三项需要 `pdflatex`（英文）和 `xelatex`（中文），装好后跑 `make` 与 `make zh`。
单测需要 torch 环境，在训练机上跑 `python -m pytest tests/ -q`。

## 服务器侧建议保留

主 checkpoint 不要删。补测某个故障或某个指标时，有 checkpoint 是几十分钟纯 eval；
没有就得重训。`stuck_imu` 那次漏评测就是靠 checkpoint 补回来的。
