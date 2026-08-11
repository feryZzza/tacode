"""确定性生成 AAAI-27 稿件的五张图。

数字优先从 `--suite-dir` 读取（`paper_numbers.json` / `table1_detection.tsv` /
`table2_safety.tsv` / `loso_effect.json`）；读不到就退回内置常量，这样在没有报告的
机器上也能重现出图。所以正文数字变了只要重跑汇总再 `make figures`，不必手改常量。

注意两处静默退化：读不到汇总目录就用下面的 FALLBACK 常量（宏平均 AUROC 会变成
0.826/0.760，那是 forecast 口径修正前的旧数字，稿件已不使用）；读不到
`--overview-timeseries` 的 npz 就把图 1(c) 退成示意曲线，与 caption 声明的真实导出
不符。两种情况都不抛异常、退出码仍是 0，只在末行的 "data sources:" 里改口径——
验收就看那一行，七项必须全是 suite/export。默认路径相对仓库根目录，从 aaai27/ 里
调用请走 `make figures`，它已把路径钉死。

视觉系统：四角色分类色（助力蓝 / 传感紫 / 故障红 / 力矩绿）、蓝色顺序渐变表示阶段量、
以 0.5 为中心的蓝-灰-红发散渐变表示 AUROC。每组分类对都带第二编码（marker 形状、
填充或直接标注），保证色觉障碍下仍可读。投稿 PDF 将文字转成轮廓，避免 Type 3
与 Identity-H；可编辑文字保留在 SVG。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.transforms as mtransforms
import numpy as np

# AAAI-27 two-column text block: 7.0 in = 177.8 mm; one column: 3.35 in.
full_width_mm = 177.8
column_width_mm = 85.09
FULL_WIDTH = full_width_mm / 25.4
COL_WIDTH = column_width_mm / 25.4

#: 每张图在正文里的实际排版宽度，必须和 .tex 里的 `\includegraphics[width=...]` 对齐。
#: 出图时按这个宽度收敛画布（fit_canvas_to_print_width），保证 1:1 放置。
#: 跨栏图一律按满 \textwidth 排。曾经按 0.72\textwidth 留白，但版面成本在高度上，
#: 宽度留白不省页数：字号被收敛器锁在 8.5 pt，窄画布只是把同样的内容挤进 5.04 in——
#: 图 1 面板 b 的三个框横排要 2.18 in，而 0.72 宽下那个面板只有 1.24 in，怎么排都会撞。
MAIN_FIGURE_WIDTH = FULL_WIDTH
# 图内字号按「印到纸上的磅值」定，不是画布磅值：只有每张图都以 1:1 放进正文
# （figsize 宽 == \includegraphics 宽）时两者才相等，所以下面的 assert_prints_one_to_one
# 会把「原生宽 != 排版宽」当错误拦下来。正文是 10 pt，图内 8.5 pt 是常规下限。
TARGET_PRINT_FONT = 8.5
BASE_FONT = TARGET_PRINT_FONT
TITLE_FONT = 8.8
BLUE = "#2166ac"
VIOLET = "#762a83"
RED = "#b2182b"
GREEN = "#1b7837"
GREY = "#5b6068"
STAGE_RAMP = ("#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c")
DIVERGING = LinearSegmentedColormap.from_list(
	"auroc_div", ["#b2182b", "#ef8a62", "#f7f7f7", "#8fb8da", "#2166ac"]
)

#: 引子图右栏的全协议汇总。数值与 main.tex 摘要/正文同源（reports/v2_fc_paper_suite），
#: 改动必须同步 check_paper_numbers.py 覆盖的那几处，否则第一页会和正文对不上。
#: 第四行是代价项，故意与前三行并列：只列收益的引子图会被当成选择性汇报。
TEASER_SUMMARY_ROWS = (
	(r"clean-adjusted $X_{\mathrm{opp}}$ drop", r"$0.0070\pm0.0015$", BLUE),
	("wrong-direction timesteps", r"$-0.74$ pp", BLUE),
	("clean aligned command kept", r"$96.1\%$", GREEN),
	("torque rate", r"$+18\%$ to $+28\%$", RED),
)

SIGNAL_COLUMNS = (
	"logit",
	"aleatoric",
	"residual",
	"forecast",
	"epistemic",
	"staleness",
	"coherence",
	"drift",
)
SIGNAL_LABELS = {
	"logit": "Fault\nlogit",
	"aleatoric": "Aleat.",
	"residual": "Recon.\nresidual",
	"forecast": "Forecast\nresidual",
	"epistemic": "Epist.",
	"staleness": "Stale-\nness",
	"coherence": "Coher-\nence",
	"drift": "Drift",
}
FAULT_ROWS = (
	("insole_missing", "insole missing", "seen"),
	("encoder_dropout", "encoder dropout", "seen"),
	("imu_bias", "IMU bias", "seen"),
	("packet_loss", "packet loss", "seen"),
	("stuck_imu", "stuck IMU", "seen"),
	("sensor_delay", "sensor delay", "seen"),
	("packet_loss_burst", "burst packet loss", "held-out op."),
	("packet_loss_partial", "partial dropout", "held-out op."),
	("sensor_delay_jitter", "delay jitter", "held-out op."),
)
FULL_DETECTION_ROWS = tuple(
	(split, fault, label, group)
	for split in ("test_id", "test_ood")
	for fault, label, group in FAULT_ROWS
)
SEEN_FAULTS = ("insole_missing", "encoder_dropout", "imu_bias", "packet_loss", "stuck_imu", "sensor_delay")
UNSEEN_FAULTS = ("packet_loss_burst", "packet_loss_partial", "sensor_delay_jitter")

# ---- 内置回退常量：与提交版正文一致，仅在没有汇总目录时使用 ----
FALLBACK: dict[str, Any] = {
	"macro_auroc": {"id_seen": 0.826, "ood_seen": 0.760, "id_unseen": 0.842, "ood_unseen": 0.778},
	# 兜底常量取自修正后 forecast 口径的 v2_fc 三 seed 均值，和正文一致。
	"detection": {
		"test_id/insole_missing": {"fused": 0.928, "logit": 0.478, "residual": 0.945, "forecast": 0.929, "staleness": 0.859, "coherence": 0.500, "drift": 0.318},
		"test_id/imu_bias": {"fused": 0.502, "logit": 0.534, "residual": 0.504, "forecast": 0.504, "staleness": 0.502, "coherence": 0.500, "drift": 0.514},
		"test_id/stuck_imu": {"fused": 0.960, "logit": 0.500, "residual": 0.659, "forecast": 0.659, "staleness": 0.965, "coherence": 1.000, "drift": 0.500},
		"test_id/packet_loss_burst": {"fused": 0.743, "logit": 0.499, "residual": 0.504, "forecast": 0.515, "staleness": 0.922, "coherence": 0.055, "drift": 0.503},
		"test_ood/packet_loss_partial": {"fused": 0.813, "logit": 0.499, "residual": 0.501, "forecast": 0.507, "staleness": 0.961, "coherence": 1.000, "drift": 0.500},
		"test_ood/sensor_delay_jitter": {"fused": 0.766, "logit": 0.481, "residual": 0.550, "forecast": 0.543, "staleness": 0.499, "coherence": 0.963, "drift": 0.504},
	},
	"safety": [
		("ID clean", 0.0752, 0.0747, 0.960, 0.978, False),
		("HT clean", 0.0774, 0.0700, 0.961, 0.950, False),
		("ID seen", 0.0910, 0.0858, 0.829, 0.904, True),
		("ID unseen", 0.0868, 0.0851, 0.918, 0.956, True),
		("HT seen", 0.0983, 0.0874, 0.878, 0.915, True),
		("HT unseen", 0.0868, 0.0765, 0.948, 0.936, True),
	],
	"ablation": {
		"auroc": (0.846, 0.870, 0.790, 0.818, 0.808),
		"ood_rmse": (0.2309, 0.2598, 0.2472, 0.2613, 0.2578),
		"retained": (0.989, 0.983, 0.989, 0.878, 0.916),
		"wrong_delta": (0.0, 0.0, 0.0, -0.00667, -0.00403),
	},
	# 末位是 Detector-gate AUROC；回退路径不编造该值，留 None（面板只画 all）。
	"stress": [
		("packet loss 0.30", "ID", 0.961, 0.939, -0.0007, None),
		("packet loss 0.30", "HT", 0.914, 0.938, -0.0080, None),
		("delay 20 samples", "ID", 0.899, 0.751, -0.0132, None),
		("delay 20 samples", "HT", 0.788, 0.933, -0.0230, None),
	],
	# LOSO 没有可信的硬编码回退：折数不足时面板 (b) 留空，而不是画一组假的分布。
	"loso": None,
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output-dir", default="aaai27/figures")
	parser.add_argument("--suite-dir", default="reports/v2_fc_paper_suite")
	parser.add_argument("--formats", default="pdf,svg,png")
	parser.add_argument(
		"--overview-timeseries",
		default="aaai27/figures/data/timeseries_overview.npz",
		help="scripts/export_timeseries.py 的导出文件；缺失时面板 (c) 退回示意曲线",
	)
	return parser.parse_args()


def setup_style() -> None:
	plt.rcParams.update(
		{
			"pdf.fonttype": 42,
			"ps.fonttype": 42,
			"svg.fonttype": "none",
			"font.family": "sans-serif",
			"font.sans-serif": ["Nimbus Sans", "Helvetica"],
			"font.size": BASE_FONT,
			"axes.labelsize": BASE_FONT,
			"axes.titlesize": TITLE_FONT,
			"xtick.labelsize": BASE_FONT,
			"ytick.labelsize": BASE_FONT,
			"legend.fontsize": BASE_FONT,
			"axes.linewidth": 0.5,
			"axes.edgecolor": "#9aa0a6",
			"xtick.major.width": 0.5,
			"ytick.major.width": 0.5,
			"grid.linewidth": 0.5,
			"grid.color": "#dfe1e5",
			"figure.dpi": 300,
			"savefig.dpi": 300,
			"savefig.bbox": "tight",
			"savefig.pad_inches": 0.02,
		}
	)


def recessive(ax, *, grid_axis: str = "x") -> None:
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	ax.grid(True, axis=grid_axis, zorder=0)
	ax.set_axisbelow(True)
	ax.tick_params(length=2.0, pad=1.5)


def panel_tag(ax, tag: str, text: str) -> None:
	ax.set_title(f"$\\bf{{{tag}}}$  {text}", loc="left", pad=4.0)


# ------------------------------------------------------------------ data loading


class Suite:
	"""汇总目录的薄封装：缺任何一块就退回 FALLBACK，绝不抛异常中断出图。"""

	def __init__(self, suite_dir: Path) -> None:
		self.dir = suite_dir
		self.numbers = self._json("paper_numbers.json")
		self.loso = self._json("loso_effect.json")
		self.detection = self._tsv("table1_detection.tsv")
		self.safety = self._tsv("table2_safety.tsv")
		self.sources: list[str] = []

	def _json(self, name: str) -> dict[str, Any]:
		path = self.dir / name
		if not path.is_file():
			return {}
		try:
			return json.loads(path.read_text(encoding="utf-8"))
		except (OSError, json.JSONDecodeError):
			return {}

	def _tsv(self, name: str) -> list[dict[str, str]]:
		path = self.dir / name
		if not path.is_file():
			return []
		with path.open(encoding="utf-8", newline="") as handle:
			return list(csv.DictReader(handle, delimiter="\t"))

	def note(self, label: str, live: bool) -> None:
		self.sources.append(f"{label}={'suite' if live else 'fallback'}")


def as_float(value: Any) -> float | None:
	if value in ("", None):
		return None
	try:
		out = float(value)
	except (TypeError, ValueError):
		return None
	return None if out != out else out


def stat(node: Any, key: str = "mean") -> float | None:
	return as_float(node.get(key)) if isinstance(node, dict) else None


def resolve_detection(suite: Suite) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
	"""All split/operator/signal cells plus the four detector macro AUROCs."""
	cells: dict[str, dict[str, float]] = {}
	by_key = {f"{row['split']}/{row['fault']}": row for row in suite.detection}
	for split, fault, _, _ in FULL_DETECTION_ROWS:
		row = by_key.get(f"{split}/{fault}")
		if row is None:
			raise RuntimeError(f"Missing detection row {split}/{fault}")
		entry = {
			"fused": as_float(row.get("fused_mean")),
			# Detector-gate：候选集限制在当前门控实现的 K_gate 六路。旧 suite 没有这一列时为 None，
			# 单元格留空（imshow 用 nan），不触发 fallback。
			"fused_online": as_float(row.get("fused_online_mean")),
		}
		for signal in SIGNAL_COLUMNS:
			entry[signal] = as_float(row.get(f"{signal}_mean"))
		if entry["fused"] is None:
			raise RuntimeError(f"Missing fused AUROC for {split}/{fault}")
		cells[f"{split}/{fault}"] = entry
	live_cells = len(cells) == len(FULL_DETECTION_ROWS)
	suite.note("detection", live_cells)

	macro: dict[str, float] = {}
	table = suite.numbers.get("table2_detection", {})
	for split_key, split in (("id", "test_id"), ("ood", "test_ood")):
		for group in ("seen", "unseen"):
			value = stat(table.get(f"macro_fused_auroc.{split}.{group}"))
			if value is not None:
				macro[f"{split_key}_{group}"] = value
	if len(macro) < 4 and suite.detection:
		# 没有 paper_numbers.json 时，直接从 table1 重算 macro。
		for split_key, split in (("id", "test_id"), ("ood", "test_ood")):
			for group, faults in (("seen", SEEN_FAULTS), ("unseen", UNSEEN_FAULTS)):
				values = [
					as_float(by_key[f"{split}/{f}"].get("fused_mean"))
					for f in faults
					if f"{split}/{f}" in by_key
				]
				values = [v for v in values if v is not None]
				if values:
					macro[f"{split_key}_{group}"] = float(np.mean(values))
	# Detector-gate 的 macro：摘要与主结论用这一组（当前门控实现的 K_gate 六路候选）。
	online: dict[str, float] = {}
	for split_key, split in (("id", "test_id"), ("ood", "test_ood")):
		for group in ("seen", "unseen"):
			node = table.get(f"macro_online_auroc.{split}.{group}")
			value = stat(node)
			if value is not None:
				online[f"{split_key}_{group}"] = value
				sd = stat(node, "sd")
				if sd is not None:
					online[f"{split_key}_{group}_sd"] = sd
	if len(online) < 4 and suite.detection:
		for split_key, split in (("id", "test_id"), ("ood", "test_ood")):
			for group, faults in (("seen", SEEN_FAULTS), ("unseen", UNSEEN_FAULTS)):
				values = [
					as_float(by_key[f"{split}/{f}"].get("fused_online_mean"))
					for f in faults
					if f"{split}/{f}" in by_key
				]
				values = [v for v in values if v is not None]
				if len(values) == len(faults):
					online[f"{split_key}_{group}"] = float(np.mean(values))
	live_macro = len(macro) == 4
	if not live_macro:
		macro = dict(FALLBACK["macro_auroc"])
	suite.note("macro", live_macro)
	if all(f"{split}_{group}" in online for split in ("id", "ood") for group in ("seen", "unseen")):
		# 面板 b 与摘要口径一致：门控实际可用的检测器成绩。
		macro = online
	return cells, macro


def resolve_safety(suite: Suite) -> list[dict[str, float | str | bool]]:
	"""Magnitude-sensitive command consequences and their cross-seed SDs."""
	table = suite.numbers.get("table3_safety", {})
	rows: list[dict[str, float | str | bool]] = []
	plan = (
		("ID clean", "test_id", "clean", False),
		("Held-task clean", "test_ood", "clean", False),
		("ID seen", "test_id", "seen_faults", True),
		("ID held-op.", "test_id", "unseen_faults", True),
		("Held-task seen", "test_ood", "seen_faults", True),
		("Held-task held-op.", "test_ood", "unseen_faults", True),
	)
	for label, split, group, faulty in plan:
		nodes = {
			"xopp_ungated": table.get(f"{split}.{group}.wrong_energy_ungated"),
			"xopp_gated": table.get(f"{split}.{group}.wrong_energy_gated"),
			"retained": table.get(f"{split}.{group}.retained_ratio"),
			"rate_ungated": table.get(f"{split}.{group}.jerk_ungated"),
			"rate_gated": table.get(f"{split}.{group}.jerk_gated"),
			"wrong_ungated": table.get(f"{split}.{group}.wrong_ungated"),
			"wrong_gated": table.get(f"{split}.{group}.wrong_gated"),
		}
		values = {key: stat(node) for key, node in nodes.items()}
		if any(value is None for value in values.values()):
			raise RuntimeError(f"Missing safety statistic for {split}/{group}")
		row: dict[str, float | str | bool] = {"label": label, "faulty": faulty}
		for key, value in values.items():
			row[key] = float(value)
			sd = stat(nodes[key], "sd")
			row[f"{key}_sd"] = float(sd) if sd is not None else 0.0
		rows.append(row)
	live = len(rows) == len(plan)
	suite.note("safety", live)
	return rows


def resolve_ablation(suite: Suite) -> dict[str, tuple]:
	# 任务 G 补齐后优先用三种子版本：同一个梯度多了跨种子离散度，
	# 才能判断相邻阶段的差值是不是比训练随机性更大。
	multi = suite.numbers.get("table4_ablation_multiseed", [])
	if isinstance(multi, list) and len(multi) == 5:
		def col(field: str, key: str) -> list[float | None]:
			return [as_float((s.get(field) or {}).get(key)) for s in multi]
		auroc, rmse = col("macro_fused_auroc", "mean"), col("ood_clean_rmse", "mean")
		seeds = [int((s.get("macro_fused_auroc") or {}).get("n") or 0) for s in multi]
		if all(v is not None for v in auroc + rmse) and min(seeds) >= 3:
			suite.note("ablation", True)
			return {
				"auroc": tuple(auroc), "ood_rmse": tuple(rmse),
				"retained": tuple(col("fault_retained_ratio", "mean")),
				# 前三阶段的门控从不闭合，wrong_delta 恒为 0：不画这一列，读者会以为
				# AUROC 最高的阶段 2 就是最好的系统，而它在控制层面毫无作用。
				"wrong_delta": tuple(col("wrong_delta_gated_minus_ungated", "mean")),
				"auroc_sd": tuple(col("macro_fused_auroc", "sd")),
				"rmse_sd": tuple(col("ood_clean_rmse", "sd")),
				"retained_sd": tuple(col("fault_retained_ratio", "sd")),
				"wrong_delta_sd": tuple(col("wrong_delta_gated_minus_ungated", "sd")),
				"n_seeds": tuple(seeds),
			}
	stages = suite.numbers.get("table4_ablation", [])
	if isinstance(stages, list) and len(stages) == 5:
		auroc = [as_float(s.get("macro_fused_auroc")) for s in stages]
		rmse = [as_float(s.get("ood_clean_rmse")) for s in stages]
		# 图注写的是「在固定故障集合上取宏平均」，所以 retention 用故障集合那一列。
		retained = [as_float(s.get("fault_retained_ratio")) for s in stages]
		wrong_delta = [as_float(s.get("wrong_delta_gated_minus_ungated")) for s in stages]
		if all(v is not None for v in auroc) and all(v is not None for v in rmse):
			suite.note("ablation", True)
			# 五个阶段都按同一套验证选门控流程评测，所以每阶段都有定义好的
			# retained ratio（含确定性阶段），不再把阶段 1 当成「无门控」。
			return {
				"auroc": tuple(auroc), "ood_rmse": tuple(rmse),
				"retained": tuple(retained), "wrong_delta": tuple(wrong_delta),
			}
	suite.note("ablation", False)
	return {key: tuple(value) for key, value in FALLBACK["ablation"].items()}


def resolve_stress(suite: Suite) -> list[tuple]:
	table = suite.numbers.get("stress_max_severity", {})
	rows: list[tuple] = []
	# 划分标 ID / HT，跟正文一致（"OOD" 在正文里只出现在 Related Work 里指别人工作的
	# 分布外语义，本文自己的留出任务集一律写 HT，图里混用会读成两个不同的划分）。
	plan = (
		("packet loss 0.30", "ID", "test_id/packet_loss@0.30"),
		("packet loss 0.30", "HT", "test_ood/packet_loss@0.30"),
		("delay 20 samples", "ID", "test_id/sensor_delay@20"),
		("delay 20 samples", "HT", "test_ood/sensor_delay@20"),
	)
	for label, split, key in plan:
		auroc_node = table.get(f"{key}.fused_auroc")
		retained_node = table.get(f"{key}.retained_ratio")
		delta_node = table.get(f"{key}.wrong_delta")
		online_node = table.get(f"{key}.online_auroc")
		auroc = stat(auroc_node)
		retained = stat(retained_node)
		delta = stat(delta_node)
		if None in (auroc, retained, delta):
			break
		online = stat(online_node)
		rows.append((
			label, split,
			auroc, stat(auroc_node, "sd"),
			retained, stat(retained_node, "sd"),
			delta, stat(delta_node, "sd"),
			online, stat(online_node, "sd"),
		))
	live = len(rows) == len(plan)
	if not live:
		rows = [
			(label, split, auroc, None, retained, None, delta, None, online, None)
			for label, split, auroc, retained, delta, online in FALLBACK["stress"]
		]
	suite.note("stress", live)
	return rows


def resolve_loso(suite: Suite) -> tuple[dict[str, Any] | None, dict[str, Any]]:
	"""LOSO：每折一个效应量，加 bootstrap 均值与 95% CI。

	任务 E 的问题是「效应在被试间是否稳定、是否跨过 0」，所以画的是逐折分布本身，
	而不是按故障机理分组的平均——分组会把被试间变异藏起来。
	"""
	node = suite.loso
	folds = node.get("folds") if isinstance(node, dict) else None
	effects: list[float] = []
	retentions: list[float] = []
	if isinstance(folds, list):
		for entry in folds:
			if not isinstance(entry, dict):
				continue
			effect = as_float(entry.get("effect"))
			if effect is None:
				continue
			effects.append(effect)
			retention = as_float(entry.get("clean_retention"))
			if retention is not None:
				retentions.append(retention)
	mean = as_float(node.get("effect_mean")) if isinstance(node, dict) else None
	lo = as_float(node.get("ci95_low")) if isinstance(node, dict) else None
	hi = as_float(node.get("ci95_high")) if isinstance(node, dict) else None
	live = len(effects) >= 10 and None not in (mean, lo, hi)
	suite.note("loso", live)
	if not live:
		return None, suite.loso
	return {
		"effects": effects,
		"mean": float(mean),
		"ci": (float(lo), float(hi)),
		"improved": sum(1 for e in effects if e > 0),
		"retention_min": min(retentions) if retentions else None,
	}, suite.loso


# ------------------------------------------------------------------ figure 1


def draw_overview_gate_panel(ax, series: dict[str, Any]) -> tuple[list, list[str]]:
	"""面板 (c) 的真实数据版本：风险（右轴）、门控与指令幅度比（左轴）。

	返回图例条目交给调用方统一排在图下方，不在面板内画：四条图例竖排要占
	0.34 个数据单位的留白带，为此把 ylim 抬到 1.52，门控曲线长时间贴着 1.0，
	图例还是压在 1.00 刻度和曲线上。排到图外后 ylim 收回 1.05，曲线占满面板。
	"""
	risk = series["risk"]
	gate = series["gate"]
	baseline = series["baseline_mag"]
	commanded = series["commanded_mag"]
	t = np.arange(risk.size)
	# 指令幅度归一到未门控峰值，和门控共用 0–1 左轴。
	scale = float(np.max(baseline)) or 1.0
	ax.fill_between(t, 0, baseline / scale, color=GREY, alpha=0.18, lw=0, label="ungated |torque|")
	ax.plot(t, commanded / scale, color=GREEN, lw=0.9, label="gated |torque|")
	ax.plot(t, gate, color=BLUE, lw=0.9, label="gate $g_t$")
	# 图例排到图外，面板内不再留图例带。
	ax.set_ylim(0, 1.05)
	ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
	ax.set_xlim(0, t[-1])
	ax.set_xlabel("time (samples)")
	ax.set_ylabel("gate / rel. |torque|")

	twin = ax.twinx()
	twin.plot(t, risk, color=RED, lw=0.7, alpha=0.85, label="normalized score")
	twin.axhline(series["deadband_risk"], color=GREY, lw=0.5, ls=":")
	# 右轴与左轴同比例：左轴 1.05 装满，右轴也只留 5% 顶部余量。
	risk_top = max(float(np.max(risk)), series["deadband_risk"]) * 1.09
	twin.set_ylim(0, risk_top)
	twin.set_yticks([tick for tick in twin.get_yticks() if tick <= float(np.max(risk)) * 1.05])
	twin.set_ylabel("score", color=RED)
	twin.tick_params(axis="y", colors=RED, length=2.0, width=0.5)
	# 死区线很靠下，正好穿过灰色力矩带；加白底才读得出来。
	twin.text(
		t[-1] * 0.985, series["deadband_risk"] + risk_top * 0.012,
		"deadband", fontsize=BASE_FONT, color=GREY, ha="right", va="bottom",
		bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.85},
	)
	for spine in ("top", "left"):
		twin.spines[spine].set_visible(False)
	twin.spines["right"].set_linewidth(0.5)
	twin.spines["right"].set_color(RED)

	recessive(ax, grid_axis="y")
	handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
	labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
	return handles, labels


def load_overview_timeseries(path: Path) -> dict[str, Any] | None:
	"""读取 scripts/export_timeseries.py 导出的逐 timestep 数组。

	缺文件就返回 None，面板 (c) 退回示意曲线；出图脚本不能因为可选数据缺失而失败。
	"""
	if not path.is_file():
		return None
	try:
		with np.load(path, allow_pickle=True) as data:
			meta = json.loads(str(data["meta"]))
			start = int(meta.get("eval_ignore_history", 0))
			risk = data["risk"][0][start:]
			gate = data["gate"][0][start:]
			baseline = data["baseline_torque"][:, start:]
			commanded = data["commanded_torque"][:, start:]
			truth = data["truth"][:, start:].astype(float)
	except (OSError, ValueError, KeyError, json.JSONDecodeError):
		return None
	if risk.size == 0:
		return None
	# 引子图画髋一路：这一窗髋的对抗量比膝大一个量级，两路叠在一张轴上只会互相盖住。
	# 目标方向按髋的力矩换算系数 0.20 缩放（与主评测同一常数），单位仍是 Nm/kg。
	j = 0
	hip_scale = 0.20
	u0, ug, tau = baseline[j].astype(float), commanded[j].astype(float), truth[j]
	opp_ungated, opp_gated = (u0 * tau) < 0, (ug * tau) < 0
	exposure_0 = float(np.maximum(0.0, -(u0 * tau)).sum())
	exposure_g = float(np.maximum(0.0, -(ug * tau)).sum())
	hip = {
		"target": hip_scale * tau,
		"ungated": u0,
		"gated": ug,
		"opposing_ungated": opp_ungated,
		"opposing_gated": opp_gated,
		"opposing_drop": 1.0 - exposure_g / exposure_0 if exposure_0 > 0 else 0.0,
	}
	return {
		"meta": meta,
		"risk": risk,
		"gate": gate,
		"hip": hip,
		# 髋膝两路力矩取绝对值均值，作为「指令幅度」的单条曲线。
		"baseline_mag": np.abs(baseline).mean(axis=0),
		"commanded_mag": np.abs(commanded).mean(axis=0),
		"deadband_risk": 1.0 + float(meta.get("gate_deadband", 0.0)),
	}


def draw_teaser(path_stem: Path, formats: list[str], series: dict[str, Any] | None) -> bool:
	"""第一页引子图：一窗真实逐 timestep 指令，右侧接全协议汇总。

	没有导出数据就不画：这张图的全部说服力来自它是实测轨迹，示意版本放第一页
	等于把机制示意冒充结果。返回是否画出，由调用方决定 data sources 怎么记。
	"""
	if series is None or "hip" not in series:
		return False
	hip = series["hip"]
	t = np.arange(hip["target"].size)
	want, u0, ug, gate = hip["target"], hip["ungated"], hip["gated"], series["gate"]
	# 门控带排在数据下方的专用条带里：它是 [0,1] 无量纲量，与力矩不共轴，只标形状不标刻度。
	# floor 必须低于三条曲线的最小值（这一窗是 -0.108），否则带子和曲线叠在一起；
	# ylim 也随之下探，并把 y 刻度限定在力矩量程内，免得刻度看着像在给带子标单位。
	floor, band = -0.196, 0.034
	# 单栏宽而非通栏：`figure*` 是双栏浮动体，LaTeX 不会把它排到有 \twocolumn 标题块的
	# 那一页顶部，于是通栏版本必然被推到第 2 页——而这张图的全部作用是落在第一页。
	# 两个面板上下叠：单栏宽度容不下并排的时序 + 汇总表。
	# hspace 要留够：面板 (a) 的 xlabel 排在它自己的框外，hspace 小于约 0.9 时会压到
	# 面板 (b) 的标题上（六个几何断言都查不出这一类跨面板碰撞）。
	fig, axes = plt.subplots(
		2, 1, figsize=(COL_WIDTH, 2.26),
		gridspec_kw={"height_ratios": [1.60, 1.0], "hspace": 0.86},
	)
	ax = axes[0]
	panel_tag(ax, "a", "One fault window (ID task, seen fault, seed 7)")
	ax.fill_between(t, floor, floor + band * gate, color="#c6dbef", lw=0, zorder=1)
	ax.fill_between(
		t, u0, 0.0, where=hip["opposing_ungated"], color=RED, alpha=0.30, lw=0, zorder=2,
		interpolate=True, label="opposing, removed",
	)
	ax.fill_between(
		t, ug, 0.0, where=hip["opposing_gated"], color="#4c4c4c", alpha=0.55, lw=0, zorder=4,
		interpolate=True, label="opposing, left",
	)
	ax.plot(t, want, color=GREY, lw=1.0, zorder=3, label=r"target $0.20\,\tau^\star$")
	ax.plot(t, u0, color=RED, lw=0.9, zorder=5, label="ungated")
	ax.plot(t, ug, color=BLUE, lw=1.0, zorder=6, label="risk-gated")
	ax.axhline(0.0, color="#9aa0a6", lw=0.5, zorder=1)
	ax.set_xlim(0, t[-1])
	ax.set_ylim(floor - 0.004, max(want.max(), u0.max()) + 0.012)
	# 刻度只给力矩量程：带子占的那段不打刻度，否则会被读成门控值也用这根轴。
	ax.set_yticks([-0.10, -0.05, 0.0, 0.05])
	ax.set_xlabel("time (samples)")
	ax.set_ylabel("hip torque (Nm/kg)")
	recessive(ax, grid_axis="x")
	# 两条注释共用带子上方那一行：左端注带子、右端注效应量。放在带子上方而不是数据区里，
	# 才不会压住曲线；单栏宽度下二者横向合计约 1.8 in，仍互不相接。
	label_y = floor + band + 0.004
	ax.text(4, label_y, "gate $g_t$", color=BLUE, ha="left", va="bottom", fontsize=BASE_FONT)
	ax.text(
		t[-1], label_y, f"$-{hip['opposing_drop'] * 100:.0f}\\%$ opposing exposure",
		color=RED, ha="right", va="bottom", fontsize=BASE_FONT,
	)

	# (b) 纯数字，不画条：四个量的分母互不相干，共用一根无刻度的轴会让条长看着可比。
	# 单栏放不下「名称 + 数值」上下两行 × 4 组，改成一行一条：左对齐名称、右对齐数值。
	ax = axes[1]
	panel_tag(ax, "b", "Across the full protocol")
	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)
	ax.axis("off")
	for i, (name, value, color) in enumerate(TEASER_SUMMARY_ROWS):
		# 首行从 0.78 起：0.86 会顶在面板标题的下缘上。
		y = 0.78 - i * 0.26
		ax.text(0.0, y, name, ha="left", va="baseline", fontsize=BASE_FONT, color="#222222")
		ax.text(1.0, y, value, ha="right", va="baseline", fontsize=BASE_FONT, color=color)
		if i < len(TEASER_SUMMARY_ROWS) - 1:
			ax.plot([0.0, 1.0], [y - 0.095] * 2, color="#dfe1e5", lw=0.5, clip_on=False)

	handles, labels = axes[0].get_legend_handles_labels()
	order = [2, 3, 4, 0, 1]  # 先三条曲线再两块填色，和读图顺序一致
	figure_legend_below(
		fig, [handles[i] for i in order], [labels[i] for i in order],
		ncol=3, handlelength=1.2, columnspacing=0.9, handletextpad=0.4,
	)
	save(fig, path_stem, formats, COL_WIDTH)
	return True


def draw_method_overview(path_stem: Path, formats: list[str], series: dict[str, Any] | None = None) -> None:
	"""方法总览。面板 (a)(b) 是机制示意；(c) 在有导出数据时使用真实逐 timestep 时序。

	有引子图（fig_teaser）时不画面板 (c)：引子图就是同一条实测轨迹的放大版，两张并存
	是同一份证据画两遍，而正文卡在 7 页上限，重复的那一栏正好是要腾出来的高度。
	面板 b 的框宽按实测文字撑开，去掉一栏后剩下两栏平分宽度反而更宽松。
	"""
	# 面板 c 带右轴 + 双 y 标签，默认间距会让标题和轴标签互相压住。
	# 高度按 1:1 排版后的观感定：宽 5.04 in 配 1.72 in 高，纵横比与原来 7.0x2.05 一致。
	panels = 2 if series is not None else 3
	fig, axes = plt.subplots(
		1, panels, figsize=(MAIN_FIGURE_WIDTH, 1.48),
		gridspec_kw=(
			{"width_ratios": [0.90, 1.34], "wspace": 0.30} if panels == 2
			else {"width_ratios": [0.90, 1.34, 1.00], "wspace": 0.46}
		),
	)
	rng = np.random.default_rng(11)
	t = np.arange(0, 400)

	# (a) 传感输入 + 一段保持值故障
	ax = axes[0]
	# 标题必须短到不会伸进面板 b 的标签位（assert_no_overlapping_titles 会拦）。
	panel_tag(ax, "a", "Causal input and fault")
	base = 0.55 * np.sin(2 * np.pi * t / 95.0) + 0.12 * np.sin(2 * np.pi * t / 31.0)
	imu = base + 0.03 * rng.standard_normal(t.size)
	insole = 0.45 * np.clip(np.sin(2 * np.pi * t / 95.0 + 0.6), 0, None) + 0.02 * rng.standard_normal(t.size)
	held = slice(210, 285)
	imu_f = imu.copy()
	imu_f[held] = imu[held.start - 1]
	ax.axvspan(held.start, held.stop, color=RED, alpha=0.10, lw=0)
	ax.plot(t, imu + 1.05, color=VIOLET, lw=0.8, label="IMU clean")
	ax.plot(t, imu_f + 1.05, color=RED, lw=0.8, ls="--", label="IMU held")
	ax.plot(t, insole, color=BLUE, lw=0.8, label="insole")
	# 两条曲线上下几乎相接，标注只能放在最下方的留白带里。
	ax.text(held.start + 37, -0.47, "held value", color=RED, ha="center", va="bottom", fontsize=BASE_FONT)
	ax.set_xlabel("time (samples)")
	ax.set_yticks([])
	ax.set_xlim(0, t[-1])
	# 图例排到图下方，面板内只给曲线和 "held value" 标注留白。
	ax.set_ylim(-0.55, 2.05)
	recessive(ax, grid_axis="x")
	trace_handles, trace_labels = ax.get_legend_handles_labels()

	# (b) 共享 TCN、学习头和完整性检查。细节由图注解释，图内只保留可读标签。
	schema_ax = axes[1]
	panel_tag(schema_ax, "b", "Estimator and evidence")
	schema_ax.set_xlim(0, 1)
	schema_ax.set_ylim(0, 1)
	schema_ax.axis("off")

	def draw_schematic() -> None:
		"""重画面板 b：2x2 流程格。

		框的位置和大小全部由实测文字尺寸算出，不能手填。原先是横排三框、宽度硬编码
		0.20/0.30/0.25，而 "TCN + heads" 在轴坐标里要占 0.42，文字直接从框里溢到左邻
		"sensor window" 的尾巴上——PDF 里读作 "sensorwindowTCN + heads"。

		改成 2x2 也不是排版偏好，是横排放不下：三个框并排量出来要 2.18 in，而这个面板
		即使整幅图按满 \\textwidth 出也只有 1.72 in 宽。竖排两行后每列只需 ~1.6 in，
		语义不变（传感窗 -> TCN -> 异常证据，完整性检查从传感窗分支后并入证据）。

		做成函数是因为画布宽会被 fit_canvas_to_print_width 反复调整，而
		text_extent_in_data 把磅值换算成数据坐标时要用当前的换算比例：
		画布一变，之前量出来的框宽就不再匹配文字，必须整块重画。
		"""
		ax = schema_ax
		for patch in list(ax.patches):
			patch.remove()
		for text in list(ax.texts):
			text.remove()
		for line in list(ax.lines):
			line.remove()

		pad_x, pad_y = 0.045, 0.085
		gap = 0.010  # 框边到箭头端点的间隙

		def size(text: str) -> tuple[float, float]:
			w, h = text_extent_in_data(ax, text, BASE_FONT)
			return w + 2 * pad_x, h + 2 * pad_y

		def box(cx, cy, text, color):
			"""按中心点摆放，框大小由文字撑开。返回 (左, 右, 上, 下) 供箭头对接。"""
			w, h = size(text)
			ax.add_patch(
				FancyBboxPatch(
					(cx - w / 2, cy - h / 2), w, h,
					boxstyle="round,pad=0.010,rounding_size=0.022",
					linewidth=0.7, edgecolor=color, facecolor=color + "1f",
				)
			)
			ax.text(cx, cy, text, ha="center", va="center", fontsize=BASE_FONT, color="#202124")
			return cx - w / 2 - gap, cx + w / 2 + gap, cy + h / 2 + gap, cy - h / 2 - gap

		def arrow(x0, y0, x1, y1):
			ax.add_patch(
				FancyArrowPatch(
					(x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7.0,
					lw=0.7, color=GREY, shrinkA=0, shrinkB=0,
				)
			)

		sensor, tcn = "sensor\nwindow", "causal TCN\n+ heads"
		integ, evid = "integrity\nchecks", "anomaly\nevidence"
		# 列宽取该列两个框的较大者，行高同理：格子按内容撑开，谁也不会挤到谁。
		left = max(size(sensor)[0], size(integ)[0])
		right = max(size(tcn)[0], size(evid)[0])
		top = max(size(sensor)[1], size(tcn)[1])
		bottom = max(size(integ)[1], size(evid)[1])
		col_gap, row_gap = 0.13, 0.16  # 留给箭头的走线宽度
		x0 = (1.0 - (left + right + col_gap)) / 2
		y0 = (1.0 - (top + bottom + row_gap)) / 2
		# 排不下就报错，别默默让框互相压住：几何守卫查的是文字与文字，两个半透明框
		# 叠在一起、文字正好错开时它们全过，只有看位图才发现。
		if x0 < 0 or y0 < 0:
			raise RuntimeError(
				f"面板 b 的 2x2 流程格排不进面板（水平余量 {x0:.3f}、垂直余量 {y0:.3f} 轴单位）；"
				"请加大 width_ratios 里 b 的份额或图高，或缩短框内文案"
			)
		cx_l, cx_r = x0 + left / 2, x0 + left + col_gap + right / 2
		cy_t, cy_b = 1.0 - y0 - top / 2, y0 + bottom / 2

		_, sensor_r, _, sensor_b = box(cx_l, cy_t, sensor, VIOLET)
		tcn_l, _, _, tcn_b = box(cx_r, cy_t, tcn, BLUE)
		_, integ_r, integ_t, _ = box(cx_l, cy_b, integ, GREY)
		evid_l, _, evid_t, _ = box(cx_r, cy_b, evid, RED)
		arrow(sensor_r, cy_t, tcn_l, cy_t)          # 传感窗 -> TCN
		arrow(cx_l, sensor_b, cx_l, integ_t)        # 传感窗 -> 完整性检查
		arrow(cx_r, tcn_b, cx_r, evid_t)            # TCN -> 异常证据
		arrow(integ_r, cy_b, evid_l, cy_b)          # 完整性检查 -> 异常证据

	draw_schematic()

	if series is not None:
		# 面板 (c) 的实测轨迹已由引子图承担，这里只留 (a)(b) 两栏机制示意。
		# 图例仍排到图下方：面板 (a) 只有 1.2 in 高，三行图例会占掉上方 1/4。
		figure_legend_below(
			fig, trace_handles, trace_labels,
			ncol=3, handlelength=1.3, columnspacing=1.1, handletextpad=0.5,
		)
		save(fig, path_stem, formats, MAIN_FIGURE_WIDTH, relayout=draw_schematic)
		return

	# (c) 风险 -> 门控 -> 力矩衰减（无导出数据时的示意版本）
	ax = axes[2]
	panel_tag(ax, "c", "Risk-gated command")
	risk = 0.18 + 0.05 * np.sin(2 * np.pi * t / 120.0)
	risk[held] = np.linspace(0.2, 0.95, held.stop - held.start)
	risk[held.stop:] = np.linspace(0.95, 0.2, t.size - held.stop) ** 1.5
	deadband = 0.35
	gate = 1.0 / (1.0 + np.exp((risk - deadband) / 0.09))
	gate = np.clip(gate, 0.0, 1.0)
	ax.plot(t, risk, color=RED, lw=0.8, label="normalized score")
	ax.axhline(deadband, color=GREY, lw=0.5, ls=":")
	ax.text(6, deadband + 0.03, "deadband", fontsize=BASE_FONT, color=GREY)
	ax.plot(t, gate, color=BLUE, lw=0.9, label="gate $g_t$")
	ax.fill_between(t, 0, gate, color=BLUE, alpha=0.10, lw=0)
	ax.set_ylim(0, 1.05)
	ax.set_xlim(0, t[-1])
	ax.set_xlabel("time (samples)")
	ax.set_ylabel("score / gate")
	recessive(ax, grid_axis="y")
	figure_legend_below(
		fig,
		trace_handles + ax.get_legend_handles_labels()[0],
		trace_labels + ax.get_legend_handles_labels()[1],
		ncol=5, handlelength=1.3, columnspacing=1.3, handletextpad=0.5,
	)
	figure_note(
		fig,
		"Panels a and c are illustrative traces of the mechanism, not measured per-timestep exports.",
	)
	save(fig, path_stem, formats, MAIN_FIGURE_WIDTH, relayout=draw_schematic)


def figure_note(fig, text: str, fontsize: float = BASE_FONT, gap_points: float = 3.0) -> None:
	"""把脚注放在所有面板内容的下方。

	原来是手填 `fig.text(0.5, -0.045, ...)`：那个 -0.045 是按当时的面板高度目测的，
	面板一变（图 1 的 x 轴标签换行、图例挪到轴下）脚注就压在 "time (samples)" 上，
	PDF 里两行字直接叠在一起。这里改成量出最低的内容边缘再往下让 gap_points 排版点。

	脚注还要按图宽折行。一行排不下时它不会被裁掉，而是把 `bbox="tight"` 的裁剪框
	向两侧撑开：图 4 的两句脚注让 PDF 的 MediaBox 变成 9.32 in，再按 `\\textwidth`
	（7.0 in）放进正文就等于把整张图连字号一起缩到 75%，比同页其他图明显小一号。
	"""
	fig.canvas.draw()
	renderer = fig.canvas.get_renderer()
	bottoms = [ax.get_tightbbox(renderer).y0 for ax in fig.axes]
	# 轴下方的图例算不算进 tightbbox 取决于版本，索性显式再量一遍。
	bottoms += [ax.get_legend().get_window_extent(renderer).y0 for ax in fig.axes if ax.get_legend()]
	# 图级图例不属于任何 ax，漏掉它脚注就会排到图例上。
	bottoms += [legend.get_window_extent(renderer).y0 for legend in fig.legends]
	lowest = min(bottoms)
	height_px = fig.get_figheight() * fig.dpi
	y = (lowest - gap_points * fig.dpi / 72.0) / height_px
	limit_px = fig.get_figwidth() * fig.dpi

	def width_of(candidate: str) -> float:
		probe = fig.text(0.5, y, candidate, ha="center", va="top", fontsize=fontsize)
		extent = probe.get_window_extent(renderer)
		probe.remove()
		return extent.width

	lines: list[str] = []
	pending = text.split()
	while pending:
		line = pending.pop(0)
		while pending and width_of(f"{line} {pending[0]}") <= limit_px:
			line = f"{line} {pending.pop(0)}"
		lines.append(line)
	fig.text(0.5, y, "\n".join(lines), ha="center", va="top", fontsize=fontsize, color=GREY, linespacing=1.35)


def figure_legend_below(fig, *args, gap_points: float = 4.0, **kwargs):
	"""把图级图例排在所有面板内容（含 xlabel）下方，纵向位置量出来而不是手填。

	`bbox_to_anchor=(0.5, 0.0)` 是按图坐标算的，0.0 是画布底边——不是「最低那行字的
	下面」。xlabel 本身就排在画布下边界之外（负图坐标），所以这个写法等于把图例摆到
	xlabel 中间：图 3 的 "base / gated" 正好压在 "aligned-command retention κ" 上。
	图高一变偏移量就得重调，手填必然过期。这里量出最低内容边缘，再往下让 gap_points。
	"""
	fig.canvas.draw()
	renderer = fig.canvas.get_renderer()
	lowest = min(ax.get_tightbbox(renderer).y0 for ax in fig.axes)
	y = (lowest - gap_points * fig.dpi / 72.0) / (fig.get_figheight() * fig.dpi)
	kwargs.setdefault("loc", "upper center")
	kwargs.setdefault("frameon", False)
	kwargs.setdefault("borderpad", 0.0)
	return fig.legend(*args, bbox_to_anchor=(0.5, y), **kwargs)


def assert_no_clipping(fig) -> None:
	"""检查每个轴标签是否落在最终裁剪框内，越界就报错。

	`savefig.bbox="tight"` 计算裁剪框时不把轴标签的横向溢出算进去（它按面板的
	tightbbox 汇总）。一个比自己面板还宽的 xlabel 因此会被静默截掉尾字母——
	"active steps" 输出成 "active step"，PDF 里也一样，看图才发现。
	与其每张图目测，不如出图时自己量：宁可 make figures 失败，也不要交一张缺字的图。
	"""
	fig.canvas.draw()
	renderer = fig.canvas.get_renderer()
	clip = fig.get_tightbbox(renderer)  # 英寸
	dpi = fig.dpi
	x0, x1 = clip.x0 * dpi, clip.x1 * dpi
	y0, y1 = clip.y0 * dpi, clip.y1 * dpi
	for ax in fig.axes:
		for label in (ax.xaxis.label, ax.yaxis.label, ax.title):
			if not label.get_text():
				continue
			bbox = label.get_window_extent(renderer=renderer)
			# 半个像素的浮点余量：tight 框本身就是由这些文字的 extent 拼出来的。
			if bbox.x0 < x0 - 0.5 or bbox.x1 > x1 + 0.5 or bbox.y0 < y0 - 0.5 or bbox.y1 > y1 + 0.5:
				raise RuntimeError(
					f"标签 {label.get_text()!r} 会被 bbox=tight 裁掉"
					f"（文字 x=[{bbox.x0:.0f},{bbox.x1:.0f}]，裁剪框 x=[{x0:.0f},{x1:.0f}]）；"
					"请加宽该面板的 width_ratios 或缩短标签"
				)


def assert_no_overlapping_titles(fig) -> None:
	"""相邻面板的标题不能互相压住。

	标题是 loc="left" 的左对齐文本，宽度不受面板宽度约束：面板 a 的标题一长，
	尾部就会伸到面板 b 的标签 "b" 上（"plausible fault" 直接和 "b" 挤成一团）。
	改 width_ratios 时最容易触发这个，所以出图时一并检查。
	"""
	fig.canvas.draw()
	renderer = fig.canvas.get_renderer()
	boxes = []
	for ax in fig.axes:
		if ax.title.get_text():
			boxes.append((ax.title.get_text(), ax.title.get_window_extent(renderer=renderer)))
	for i in range(len(boxes)):
		for j in range(i + 1, len(boxes)):
			(t1, b1), (t2, b2) = boxes[i], boxes[j]
			if b1.x1 > b2.x0 + 0.5 and b2.x1 > b1.x0 + 0.5 and b1.y1 > b2.y0 + 0.5 and b2.y1 > b1.y0 + 0.5:
				raise RuntimeError(f"面板标题 {t1!r} 与 {t2!r} 互相重叠；请缩短标题或调整 width_ratios")


def assert_legends_inside_panels(fig) -> None:
	"""图例不能横向溢出自己的面板。

	`bbox=tight` 只保证图例不被裁掉，不保证它待在自己面板里：面板 (a) 的三列图例
	一旦比面板宽，尾部就伸进面板 (b)，读者会以为那条图例属于隔壁的图；面板 (c) 的
	图例更糟——右轴那条竖线正好从 "gate $g_t$" 的下标上穿过去。
	纵向不检查：图例常常故意放在轴下方（bbox_to_anchor 负 y），那是有意的。
	"""
	fig.canvas.draw()
	renderer = fig.canvas.get_renderer()
	for ax in fig.axes:
		legend = ax.get_legend()
		if legend is None:
			continue
		lb = legend.get_window_extent(renderer)
		ab = ax.get_window_extent(renderer)
		# 1 pt 余量：图例边框和面板边框重合时不算溢出。
		if lb.x0 < ab.x0 - 1.0 or lb.x1 > ab.x1 + 1.0:
			raise RuntimeError(
				f"图例 {[t.get_text() for t in legend.get_texts()]} 超出面板宽度"
				f"（图例 x=[{lb.x0:.0f},{lb.x1:.0f}]，面板 x=[{ab.x0:.0f},{ab.x1:.0f}]）；"
				"请减少 ncol、缩短标签或放到面板下方"
			)


def assert_legends_clear_of_text(fig) -> None:
	"""图例不能压在轴标签、面板标题、脚注或另一个图例上。

	`assert_legends_inside_panels` 只查横向，注释里写明「纵向是有意的」——把图例放在
	轴下方确实是有意的，但放多远没人量。图 5 从 2.80 in 压到 2.44 in 后，两个面板的
	图例都退到了自己的 xlabel 上：(a) 的 "Detector-all AUROC" 压着
	"AUROC · retained aligned torque"，(b) 的两条图例压着 "wrong-direction reduction
	(pp)"，字叠字。四个守卫全过、退出码 0，只有看位图才发现。
	所以纵向也要查，只是查的对象换成「图例与其他文字」而不是「图例与面板边框」。

	图级图例（`fig.legend`）也要查：它挂在 `fig.legends` 上，只遍历 `ax.get_legend()`
	的话整类都漏检。图 1/2/3 为了给面板腾地方把图例全改成了 `fig.legend`，于是图 1
	那两行图例正好压在 "time (samples)" 上——守卫全过、位图上是字叠字。
	"""
	fig.canvas.draw()
	renderer = fig.canvas.get_renderer()
	legends = []
	others = []

	def collect(legend) -> None:
		if legend is None:
			return
		# 量图例条目文字的并集，不用 legend.get_window_extent()：后者含 borderpad /
		# handletextpad 的空白边，图例明明还差十来个像素才碰到 xlabel 就会误报。
		# 判据要贴着「字压字」，否则守卫会逼着每张图都把图例推远，白掉版面。
		boxes = [t.get_window_extent(renderer=renderer) for t in legend.get_texts()]
		if not boxes:
			return
		label = ", ".join(t.get_text() for t in legend.get_texts())
		legends.append((label, mtransforms.Bbox.union(boxes)))

	for legend in fig.legends:
		collect(legend)
	for ax in fig.axes:
		collect(ax.get_legend())
		for artist in (ax.xaxis.label, ax.yaxis.label, ax.title):
			if artist.get_text():
				others.append((artist.get_text(), artist.get_window_extent(renderer=renderer)))
	for text in fig.texts:
		if text.get_text():
			others.append((text.get_text()[:40], text.get_window_extent(renderer=renderer)))

	# 余量按排版点给（1 pt = dpi/72 px）。写成像素常量在 300 dpi 下只有 0.24 pt，
	# 图例外框与 xlabel 刚好相切也会被判成压字。
	pad = 1.0 * fig.dpi / 72.0

	def overlaps(a, b) -> bool:
		return a.x1 > b.x0 + pad and b.x1 > a.x0 + pad and a.y1 > b.y0 + pad and b.y1 > a.y0 + pad

	for i, (label, lb) in enumerate(legends):
		for other_label, ob in others:
			if overlaps(lb, ob):
				raise RuntimeError(
					f"图例 {label!r} 压在文字 {other_label!r} 上"
					f"（图例 y=[{lb.y0:.0f},{lb.y1:.0f}]，文字 y=[{ob.y0:.0f},{ob.y1:.0f}]）；"
					"请把 bbox_to_anchor 的 y 再降一些、加大 hspace 或缩短图例文案"
				)
		for other_label, ob in legends[i + 1:]:
			if overlaps(lb, ob):
				raise RuntimeError(f"图例 {label!r} 与图例 {other_label!r} 互相重叠；请加大 hspace")


def assert_annotations_clear_of_other_panels(fig) -> None:
	"""面板内的数据标注不能压到别的面板的文字上。

	前面五个守卫查的都是「标签 vs 裁剪框」「标题 vs 标题」「图例 vs 文字」，唯独漏了
	最常见的一类：`annotate` 出来的数值标签。它不受面板边框约束，居中标在末柱上时有
	一半探出右边框——右邻面板的 ylabel 就贴在那儿。图 4 压到 1.62 in 后四个面板全中，
	最窄处只差 0.5 pt，位图上是 `0.889` 和 `wrong-direction change (pp)` 叠成一团。
	判据同样是「字压字」：本面板的 Text vs 其他面板的 ylabel / xlabel / 标题 /
	刻度标签 / 数值标注。同面板内部不查——那是各图自己的避让逻辑（见 panel (a) 的
	`reserve`/`free`），在这里查会把有意的紧凑排布判成缺陷。
	"""
	fig.canvas.draw()
	renderer = fig.canvas.get_renderer()

	def texts_of(ax, *, own: bool) -> list[tuple[str, Any]]:
		items = [t for t in ax.texts if t.get_text()]
		if not own:
			for artist in (ax.xaxis.label, ax.yaxis.label, ax.title):
				if artist.get_text():
					items.append(artist)
			items.extend(
				label for axis in (ax.xaxis, ax.yaxis)
				for label in axis.get_ticklabels() if label.get_text() and label.get_visible()
			)
		return [(t.get_text()[:40], t.get_window_extent(renderer=renderer)) for t in items]

	pad = 1.0 * fig.dpi / 72.0
	axes = list(fig.axes)
	for i, ax in enumerate(axes):
		mine = texts_of(ax, own=True)
		for j, other in enumerate(axes):
			if i == j:
				continue
			for label, box in mine:
				for other_label, other_box in texts_of(other, own=False):
					if (box.x1 > other_box.x0 + pad and other_box.x1 > box.x0 + pad
							and box.y1 > other_box.y0 + pad and other_box.y1 > box.y0 + pad):
						raise RuntimeError(
							f"面板标注 {label!r} 压在邻面板的文字 {other_label!r} 上"
							f"（标注 x=[{box.x0:.0f},{box.x1:.0f}]，被压 x=[{other_box.x0:.0f},{other_box.x1:.0f}]）；"
							"请把标注右对齐到 ax.get_xlim()[1] 或加大 wspace"
						)


def fit_canvas_to_print_width(fig, print_width_in: float, relayout=None) -> None:
	"""把画布宽调到「裁剪后正好等于排版宽度」，让图以 1:1 放进正文。

	图内字号是磅值，放进正文时会随 `\\includegraphics[width=...]` 一起缩放，所以
	纸面上的实际字号 = 代码里的字号 x (排版宽 / PDF 原生宽)。`savefig.bbox="tight"`
	把画布裁到内容外框，原生宽因此既不等于 figsize 也不可预测：三张 7.0 in 的图
	分别裁成 5.90 / 6.88 / 6.55 in，再统一按 0.72\\textwidth 放进正文，就变成
	0.854 / 0.733 / 0.769 三种缩放——代码里写着同一个 9.2 pt，纸上却印成
	7.86 / 6.74 / 7.08 pt，同一页的两张图字号差一号还都不到下限。

	所以出图时反解一次：裁剪宽比目标差多少就把画布加宽多少。文字宽度不随画布变，
	面板宽度是按比例分配的，误差对画布宽单调，几轮就收敛到千分之一英寸内。
	收敛后原生宽 == 排版宽，缩放恒为 1.0，代码里的磅值就是纸面上的磅值。

	`relayout` 给那些「尺寸由实测文字宽度决定」的图元用（图 1 面板 b 的框）。改画布宽会
	改数据坐标与像素的换算，按文字量出来的框宽必须在每次改宽后重算，否则框会重新对不上字。
	"""
	# savefig 会在裁剪框外再补一圈 pad_inches，左右各一次，导出宽因此比 tightbbox 宽
	# 2*pad。收敛目标要先扣掉这圈边，否则原生宽会稳定地比排版宽多 0.04 in（约 0.8%）。
	target = print_width_in - 2 * plt.rcParams["savefig.pad_inches"]
	for _ in range(16):
		if relayout is not None:
			relayout()
		fig.canvas.draw()
		width = fig.get_tightbbox(fig.canvas.get_renderer()).width
		deficit = target - width
		if abs(deficit) <= 0.002:
			return
		# 有些图元宽度不随画布缩放（图例、脚注这类图级文本按磅值排），缩到某一点后
		# 裁剪宽就卡在它们的宽度上不再变小。此时把画布继续缩只会缩到负数，
		# 报错要指向真正的原因，而不是 matplotlib 的 "size must be positive"。
		if fig.get_figwidth() + deficit <= 0.5:
			raise RuntimeError(
				f"裁剪宽 {width:.2f} in 收不进排版宽 {print_width_in:.2f} in："
				"图例或图级文本本身就比排版宽度宽，请增加 ncol 行数、缩短标签或折行"
			)
		fig.set_figwidth(fig.get_figwidth() + deficit)
	raise RuntimeError(
		f"画布宽无法收敛到 {print_width_in:.3f} in（当前裁剪宽 "
		f"{fig.get_tightbbox(fig.canvas.get_renderer()).width:.3f} in）；"
		"通常是某个标签比整幅图还宽，请折行或缩短"
	)


def assert_prints_one_to_one(fig, path_stem: Path, print_width_in: float) -> None:
	"""导出的 PDF 原生宽必须等于排版宽度，否则纸面字号不是代码里的字号。"""
	# 和 fit_canvas_to_print_width 一致：导出宽 = 裁剪宽 + 左右两圈 pad_inches。
	width = fig.get_tightbbox(fig.canvas.get_renderer()).width
	width += 2 * plt.rcParams["savefig.pad_inches"]
	if abs(width - print_width_in) > 0.01:
		raise RuntimeError(
			f"{path_stem.name}: 原生宽 {width:.3f} in != 排版宽 {print_width_in:.3f} in，"
			f"纸面字号会变成 {BASE_FONT * print_width_in / width:.2f} pt"
		)


def save(fig, path_stem: Path, formats: list[str], print_width_in: float, relayout=None) -> None:
	# 先把画布调到 1:1，再跑几何守卫：守卫量的是像素，画布一变所有 extent 都变，
	# 顺序反了就等于在一个不会被导出的布局上做检查。
	fit_canvas_to_print_width(fig, print_width_in, relayout)
	assert_no_clipping(fig)
	assert_no_overlapping_titles(fig)
	assert_legends_inside_panels(fig)
	assert_legends_clear_of_text(fig)
	assert_annotations_clear_of_other_panels(fig)
	assert_prints_one_to_one(fig, path_stem, print_width_in)
	for fmt in formats:
		target = path_stem.with_suffix(f".{fmt}")
		if fmt != "pdf":
			# Export bundle: editable vector .svg and 600-dpi raster preview .png.
			if fmt == "png":
				fig.savefig(target, format="png", dpi=600)
			else:
				fig.savefig(target, format=fmt)
				if fmt == "svg":
					# Matplotlib emits trailing spaces in multi-line SVG path data.
					# Normalize generated text so repository whitespace checks remain useful.
					svg = target.read_text(encoding="utf-8")
					target.write_text(
						"\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
						encoding="utf-8",
					)
			continue
		# Matplotlib embeds even standard sans fonts as Identity-H CID subsets. AAAI-27
		# explicitly asks authors to remove or outline Identity-H fonts, so keep editable
		# text in the companion SVG and convert only the submission PDF glyphs to paths.
		fonted = path_stem.with_name(path_stem.name + "_fonted").with_suffix(".pdf")
		fig.savefig(fonted, format="pdf")
		subprocess.run(
			[
				"gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE",
				"-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.5",
				"-dNoOutputFonts", f"-sOutputFile={target}", str(fonted),
			],
			check=True,
		)
		fonted.unlink()
	plt.close(fig)


def text_extent_in_data(ax, text: str, fontsize: float) -> tuple[float, float]:
	"""标签的真实渲染尺寸，换算成 ax 的数据坐标 (宽, 高)。

	自动避让需要知道标签占多大地方。按字符数乘常数估算在比例字体下不可靠——宽字母
	（O、W、m）会被低估，标签就会被判为放得下然后溢出边界。这里画一个临时 Text、
	量它的窗口尺寸、再用逆变换换算回数据坐标，量完就删掉。

	需要 renderer，所以调用前 figure 必须已经有 canvas（plt 创建的 figure 都有）。
	"""
	fig = ax.figure
	renderer = fig.canvas.get_renderer()
	probe = ax.text(0, 0, text, fontsize=fontsize)
	bbox = probe.get_window_extent(renderer=renderer)
	probe.remove()
	inverse = ax.transData.inverted()
	x0, y0 = inverse.transform((bbox.x0, bbox.y0))
	x1, y1 = inverse.transform((bbox.x1, bbox.y1))
	return abs(x1 - x0), abs(y1 - y0)


def points_to_data(ax) -> tuple[float, float]:
	"""1 pt 在 ax 数据坐标里有多长 (x, y)。

	`annotate(textcoords="offset points")` 的偏移是排版点，碰撞检测却在数据坐标里做。
	两边口径不统一时，盒子会算在实际落位以外几倍距离的地方——判定不重叠、画出来却叠着。
	所以偏移量一律以点为单位声明，用这个换算成数据坐标后再拼盒子。
	"""
	px = ax.figure.dpi / 72.0
	inverse = ax.transData.inverted()
	x0, y0 = inverse.transform((0.0, 0.0))
	x1, y1 = inverse.transform((px, px))
	return abs(x1 - x0), abs(y1 - y0)


# ------------------------------------------------------------------ figure 2


def draw_detection_taxonomy(path_stem: Path, formats: list[str], cells, macro) -> None:
	# 面板 (b) 的 y 轴刻度标签 "held-out operators" 很长（8.5 pt 下量出来 1.11 in），
	# 而它自己只有 4 个数据点。wspace 要留得下这条标签，否则热力图最后一列的数字会
	# 撞上去（守卫会拦）。1.18 是画布窄到 5.04 in 时凑出来的，满宽下它留出 2.01 in 的
	# 空档：图里是热力图和面板 b 之间一条明显的空白带。0.62 给标签 1.28 in，够。
	fig, axes = plt.subplots(
		1, 2, figsize=(MAIN_FIGURE_WIDTH, 1.82),
		gridspec_kw={"width_ratios": [3.75, 1.0], "wspace": 0.62},
	)
	fig.subplots_adjust(left=0.18, right=0.98, bottom=0.25, top=0.90)

	ax = axes[0]
	panel_tag(ax, "a", "Per-operator AUROC (task-split mean)")
	has_online = any(entry.get("fused_online") is not None for entry in cells.values())
	fusion_columns = ("fused", "fused_online") if has_online else ("fused",)
	fusion_labels = ["All", "Gate"] if has_online else ["Fused"]
	signal_labels = {
		"logit": "Fault",
		"aleatoric": "Aleat.",
		"residual": "Recon.",
		"forecast": "Forecast",
		"epistemic": "Epist.",
		"staleness": "Stale",
		"coherence": "Coherence",
		"drift": "Drift",
	}
	columns = (*fusion_columns, *SIGNAL_COLUMNS)
	matrix = np.full((len(FAULT_ROWS), len(columns)), np.nan)
	for r, (fault, _, _) in enumerate(FAULT_ROWS):
		for c, column in enumerate(columns):
			values = [
				cells[f"{split}/{fault}"].get(column)
				for split in ("test_id", "test_ood")
			]
			values = [value for value in values if value is not None]
			if values:
				matrix[r, c] = float(np.mean(values))
	norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
	ax.imshow(matrix, cmap=DIVERGING, norm=norm, aspect="auto")
	for r in range(matrix.shape[0]):
		for c in range(matrix.shape[1]):
			value = matrix[r, c]
			if value != value:
				continue
			shade = "#ffffff" if abs(value - 0.5) > 0.36 else "#202124"
			ax.text(c, r, f"{value:.2f}".lstrip("0"), ha="center", va="center", fontsize=BASE_FONT, color=shade)
	ax.set_xticks(range(len(columns)))
	ax.set_xticklabels(
		[*fusion_labels, *(signal_labels[s] for s in SIGNAL_COLUMNS)],
		rotation=55,
		ha="right",
		rotation_mode="anchor",
	)
	ax.set_yticks(range(len(FAULT_ROWS)))
	ax.set_yticklabels(
		[f"{name} [{'S' if tag == 'seen' else 'H'}]" for _, name, tag in FAULT_ROWS]
	)
	ax.tick_params(length=0)
	for side in ("top", "right", "left", "bottom"):
		ax.spines[side].set_visible(False)
	split_at = len(fusion_columns) - 0.5
	ax.axvline(split_at, color="#202124", lw=0.7)
	ax.set_ylim(len(FAULT_ROWS) - 0.5, -0.5)

	ax = axes[1]
	panel_tag(ax, "b", "Macro AUROC")
	groups = (("seen faults", "seen"), ("held-out operators", "unseen"))
	for gi, (title, key) in enumerate(groups):
		y = 1 - gi
		for split_key, marker, face, offset in (
			("id", "o", BLUE, 0.12), ("ood", "s", "#ffffff", -0.12)
		):
			value = macro.get(f"{split_key}_{key}")
			if value is None:
				continue
			sd = macro.get(f"{split_key}_{key}_sd")
			ax.errorbar(
				value, y + offset, xerr=sd, fmt=marker, markersize=5.0,
				color=BLUE, markerfacecolor=face, markeredgecolor=BLUE,
				markeredgewidth=0.8, elinewidth=0.8, capsize=2.0,
			)
	ax.set_yticks([1, 0])
	ax.set_yticklabels([title for title, _ in groups])
	ax.set_ylim(-0.45, 1.45)
	values = [v for v in (macro.get(f"{s}_{k}") for s in ("id", "ood") for _, k in groups) if v is not None]
	if values:
		pad = max(0.18 * (max(values) - min(values)), 0.012)
		ax.set_xlim(min(values) - pad, max(values) + pad)
	ax.set_xlabel("macro AUROC")
	handles = [
		plt.Line2D([], [], marker="o", color=BLUE, markerfacecolor=BLUE, lw=0, markersize=5.0, label="ID"),
		# 标 "HT" 而不是 "held-out"：这一栏的 y 轴刻度已经有 "held-out operators"，
		# 两个 "held-out" 指的却是不同东西（任务划分 vs 损坏算子），并排读会串。
		# 正文用 ID / HT 这一对写法，图里跟着用。
		plt.Line2D([], [], marker="s", color=BLUE, markerfacecolor="#ffffff", lw=0, markersize=5.0, label="HT"),
	]
	# 图例排到整幅图下方：面板只有两行数据点，"lower right" 那格正好是
	# held-out operators 那组 ID/held-out 两个点的位置，图例标记直接压在数据点上；
	# 而这个面板只占 1/4 幅宽，两列图例横排又装不进面板，所以挂到 figure 上。
	figure_legend_below(
		fig, handles=handles, ncol=2, handlelength=1.0,
		columnspacing=1.6, handletextpad=0.4,
	)
	recessive(ax, grid_axis="x")
	save(fig, path_stem, formats, MAIN_FIGURE_WIDTH)


def draw_detection_full(path_stem: Path, formats: list[str], cells) -> None:
	"""Supplementary split-specific matrix with no omitted operator or signal."""
	fig, ax = plt.subplots(figsize=(FULL_WIDTH, 6.65))
	fig.subplots_adjust(left=0.20, right=0.98, bottom=0.15, top=0.96)
	has_online = any(entry.get("fused_online") is not None for entry in cells.values())
	fusion_columns = ("fused", "fused_online") if has_online else ("fused",)
	columns = (*fusion_columns, *SIGNAL_COLUMNS)
	labels = (
		("All", "Gate") if has_online else ("Fused",)
	) + ("Fault", "Aleat.", "Recon.", "Forecast", "Epist.", "Stale", "Coherence", "Drift")
	matrix = np.full((len(FULL_DETECTION_ROWS), len(columns)), np.nan)
	for r, (split, fault, _, _) in enumerate(FULL_DETECTION_ROWS):
		entry = cells[f"{split}/{fault}"]
		for c, column in enumerate(columns):
			value = entry.get(column)
			if value is not None:
				matrix[r, c] = value
	norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
	ax.imshow(matrix, cmap=DIVERGING, norm=norm, aspect="auto")
	for r in range(matrix.shape[0]):
		for c in range(matrix.shape[1]):
			value = matrix[r, c]
			if value != value:
				continue
			shade = "#ffffff" if abs(value - 0.5) > 0.36 else "#202124"
			ax.text(c, r, f"{value:.2f}".lstrip("0"), ha="center", va="center", fontsize=8.0, color=shade)
	ax.set_xticks(range(len(columns)))
	ax.set_xticklabels(labels, rotation=50, ha="right", rotation_mode="anchor")
	ax.set_yticks(range(len(FULL_DETECTION_ROWS)))
	ax.set_yticklabels(
		[
			f"{'ID' if split == 'test_id' else 'HT'} · {label} · {'S' if group == 'seen' else 'H'}"
			for split, _, label, group in FULL_DETECTION_ROWS
		],
		fontsize=8.2,
	)
	ax.tick_params(length=0)
	for side in ("top", "right", "left", "bottom"):
		ax.spines[side].set_visible(False)
	ax.axvline(len(fusion_columns) - 0.5, color="#202124", lw=0.7)
	ax.axhline(len(FAULT_ROWS) - 0.5, color="#202124", lw=0.7)
	ax.set_ylim(len(FULL_DETECTION_ROWS) - 0.5, -0.5)
	save(fig, path_stem, formats, FULL_WIDTH)


# ------------------------------------------------------------------ figure 3


def draw_safety_utility(path_stem: Path, formats: list[str], rows) -> None:
	fig, axes = plt.subplots(
		1, 3, figsize=(MAIN_FIGURE_WIDTH, 1.80),
		gridspec_kw={"width_ratios": [1.35, 0.82, 1.25], "wspace": 0.62},
	)
	labels = [str(row["label"]) for row in rows]
	faulty = [bool(row["faulty"]) for row in rows]
	y = np.arange(len(rows))[::-1]

	def paired_panel(ax, key: str, title: str, xlabel: str, annotation) -> None:
		ungated = np.array([float(row[f"{key}_ungated"]) for row in rows])
		gated = np.array([float(row[f"{key}_gated"]) for row in rows])
		ungated_sd = np.array([float(row[f"{key}_ungated_sd"]) for row in rows])
		gated_sd = np.array([float(row[f"{key}_gated_sd"]) for row in rows])
		panel_tag(ax, title[0], title[1:])
		for yi, before, after in zip(y, ungated, gated):
			ax.plot([before, after], [yi, yi], color="#c9ccd1", lw=0.8, zorder=1)
		ax.errorbar(
			ungated, y, xerr=ungated_sd, fmt="o", markersize=3.8,
			color=GREY, markerfacecolor="#ffffff", markeredgewidth=0.8,
			elinewidth=0.7, capsize=1.5, linestyle="none", label="base", zorder=2,
		)
		ax.errorbar(
			gated, y, xerr=gated_sd, fmt="D", markersize=3.4,
			color=BLUE, markerfacecolor=BLUE, markeredgewidth=0.8,
			elinewidth=0.7, capsize=1.5, linestyle="none", label="gated", zorder=3,
		)
		span = max(float(np.max(np.r_[ungated + ungated_sd, gated + gated_sd]) - np.min(np.r_[ungated - ungated_sd, gated - gated_sd])), 1e-6)
		# 标注锚在「误差棒左端」而不是标记中心：锚在中心时那 4 pt 偏移只够躲开标记本身，
		# 误差棒和端帽还在左边伸出去，"+28%" 就压在 base 的空心圆和它的横线上。
		# 端帽半宽 1.5 pt + 线宽，留 3 pt 再加 4 pt 文字间距。
		left = np.minimum(ungated - ungated_sd, gated - gated_sd)
		for yi, before, after, anchor in zip(y, ungated, gated, left):
			ax.annotate(
				annotation(before, after), (anchor, yi),
				textcoords="offset points", xytext=(-7, 0), ha="right",
				va="center", fontsize=8.0, color="#202124",
			)
		ax.set_xlim(
			float(np.min(np.r_[ungated - ungated_sd, gated - gated_sd])) - 0.45 * span,
			float(np.max(np.r_[ungated + ungated_sd, gated + gated_sd])) + 0.08 * span,
		)
		ax.set_xlabel(xlabel)
		recessive(ax, grid_axis="x")

	ax = axes[0]
	paired_panel(
		ax,
		"xopp",
		"a$X_{\\mathrm{opp}}$ exposure",
		"$X_{\\mathrm{opp}}$ ($(\\mathrm{Nm/kg})^2\\mathrm{s}$)",
		lambda before, after: f"{after - before:+.3f}",
	)
	ax.set_yticks(y)
	ax.set_yticklabels(labels)
	# 图例排到整幅图下方居中：放在面板 (a) 的 "upper left" 时，两行图例正好落在
	# ID clean / Held-task clean 两行的数据点和 -0.001 标注上（面板里没有空行留白）。
	base_handles, base_labels = ax.get_legend_handles_labels()

	ax = axes[2]
	paired_panel(
		ax,
		"rate",
		"cTorque-rate increase",
		"mean $|\\Delta u|/\\Delta t$",
		lambda before, after: f"{(after / before - 1.0) * 100:+.0f}%",
	)
	ax.set_yticks(y)
	ax.set_yticklabels([])

	ax = axes[1]
	panel_tag(ax, "b", "Command retention")
	retained = np.array([float(row["retained"]) for row in rows])
	retained_sd = np.array([float(row["retained_sd"]) for row in rows])
	for yi, value, error, is_fault in zip(y, retained, retained_sd, faulty):
		marker = "^" if is_fault else "o"
		face = RED if is_fault else "#ffffff"
		edge = RED if is_fault else BLUE
		ax.errorbar(
			value, yi, xerr=error, fmt=marker, markersize=4.0,
			color=edge, markerfacecolor=face, markeredgecolor=edge,
			elinewidth=0.7, capsize=1.5,
		)
	ax.axvline(0.95, color=GREY, lw=0.6, ls="--")
	ax.set_yticks(y)
	ax.set_yticklabels([])
	ax.set_xlim(min(0.80, float(np.min(retained - retained_sd)) - 0.02), 1.005)
	ax.set_xlabel("aligned-command retention $\\kappa$")
	recessive(ax, grid_axis="x")
	# 图例最后画：figure_legend_below 量的是全部面板的最低内容边缘，三个面板的 xlabel
	# 还没排完就量，得到的位置会偏高，图例就压回到后画的那个 xlabel 上。
	figure_legend_below(
		fig, base_handles[:2], base_labels[:2], ncol=2,
		handlelength=1.2, columnspacing=1.6, handletextpad=0.5,
	)
	save(fig, path_stem, formats, MAIN_FIGURE_WIDTH)


# ------------------------------------------------------------------ figure 4


def draw_wrong_delta_panel(ax, stages, wrong_delta, wrong_sd) -> None:
	"""每阶段门控前后的反向力矩变化（负号=更安全）。

	前三阶段恒为 0：那些阶段选出的 deadband 落在 0.45–0.71、softness 固定 8.0，
	门控从不闭合，所以再高的 AUROC 也不改变作动。零值柱画不出来，用一条贴零线的
	文字标出来，否则面板看起来像缺数据。
	"""
	if wrong_delta is None or any(v is None for v in wrong_delta):
		ax.set_axis_off()
		ax.text(0.5, 0.5, "wrong-direction delta\nunavailable", transform=ax.transAxes,
		        ha="center", va="center", fontsize=BASE_FONT, color=GREY)
		return
	values = np.array([float(v) for v in wrong_delta], dtype=float) * 100.0
	sd = None if wrong_sd is None else np.array([float(v) for v in wrong_sd], dtype=float) * 100.0
	for stage, value, color in zip(stages, values, STAGE_RAMP):
		ax.bar(stage, value, width=0.62, color=color, linewidth=0)
	if sd is not None:
		ax.errorbar(stages, values, yerr=sd, fmt="none", ecolor="#202124", elinewidth=0.7, capsize=1.6, zorder=5)
	ax.axhline(0.0, color=GREY, lw=0.6, zorder=4)
	lo = float(np.min(values - (sd if sd is not None else 0.0)))
	span = max(abs(lo), 0.1)
	ax.set_ylim(lo - 0.22 * span, 0.42 * span)
	zero = [int(s) for s, v in zip(stages, values) if abs(v) < 1e-9]
	if zero:
		ax.annotate(
			"gate never closes", (float(np.mean(zero)), 0.0), textcoords="offset points",
			xytext=(0, 3), ha="center", va="bottom", fontsize=BASE_FONT, color=GREY,
		)
	# 这是最右侧面板，居中标注探出右边框不会压到别的面板，但会把 bbox="tight" 的裁剪框
	# 撑宽（进而缩小 \includegraphics 的实际字号）。同样右对齐到边框内。
	ax.set_xlim(0.45, 5.55)
	ax.annotate(
		f"{values[-1]:+.2f}", (ax.get_xlim()[1], values[-1] - (sd[-1] if sd is not None else 0.0)),
		textcoords="offset points", xytext=(-1, -3), ha="right", va="top", fontsize=BASE_FONT,
	)
	ax.set_xticks(stages)
	ax.set_xticklabels(stages)
	ax.set_ylabel("wrong-direction change (pp)")
	recessive(ax, grid_axis="y")


def draw_ablation_summary(path_stem: Path, formats: list[str], ablation) -> None:
	fig, axes = plt.subplots(
		1, 4, figsize=(FULL_WIDTH, 2.60),
		gridspec_kw={"width_ratios": [1.38, 0.82, 0.88, 0.88], "wspace": 0.72},
	)
	stages = np.arange(1, 6)
	auroc = np.array([v for v in ablation["auroc"]], dtype=float)
	rmse = np.array([v for v in ablation["ood_rmse"]], dtype=float)
	retained = ablation["retained"]
	wrong_delta = ablation.get("wrong_delta")
	# 三种子版本才有 *_sd；单种子扫描下这些键缺失，误差棒自动消失。
	def sd_of(key: str) -> np.ndarray | None:
		values = ablation.get(key)
		if not values or any(v is None for v in values):
			return None
		return np.array([float(v) for v in values], dtype=float)
	auroc_sd, rmse_sd, retained_sd = sd_of("auroc_sd"), sd_of("rmse_sd"), sd_of("retained_sd")
	wrong_sd = sd_of("wrong_delta_sd")
	n_seeds = max(ablation.get("n_seeds") or [1])

	# (a) macro AUROC 与相邻差值
	ax = axes[0]
	panel_tag(ax, "a", "Detection AUROC")
	ax.plot(stages, auroc, color=BLUE, lw=0.8, zorder=1)
	if auroc_sd is not None:
		ax.errorbar(stages, auroc, yerr=auroc_sd, fmt="none", ecolor=BLUE, elinewidth=0.7, capsize=1.8, zorder=2)
	for stage, value, color in zip(stages, auroc, STAGE_RAMP):
		ax.plot(stage, value, "o", markersize=5.0, color=color, markeredgecolor=BLUE, markeredgewidth=0.7, linestyle="none", zorder=3)
	# 加了面板 (d) 之后 (a) 变窄，五个两行阶段名（"+ recon-/struction" 最宽）横向挤在一起，
	# "+ prob-/head" 和 "+ recon-" 直接连成一串。阶段序列在 caption 里已经写全，
	# 四个面板统一只标序号；顺带省掉两行刻度标签的高度。
	ax.set_xticks(stages)
	ax.set_xticklabels(stages)
	ax.set_ylabel("macro fused AUROC")
	# 坐标范围先定，之后的标签避让要用到边界。
	ax.set_xlim(0.6, 5.85)
	pad = max(0.35 * float(np.ptp(auroc)), 0.01)
	lo = float(np.min(auroc - (auroc_sd if auroc_sd is not None else 0)))
	hi = float(np.max(auroc + (auroc_sd if auroc_sd is not None else 0)))
	ax.set_ylim(lo - pad, hi + pad)
	recessive(ax, grid_axis="y")

	# (b) clean HT RMSE
	ax = axes[1]
	panel_tag(ax, "b", "Clean HT RMSE")
	for stage, value, color in zip(stages, rmse, STAGE_RAMP):
		ax.bar(stage, value, width=0.62, color=color, linewidth=0)
	if rmse_sd is not None:
		ax.errorbar(stages, rmse, yerr=rmse_sd, fmt="none", ecolor="#202124", elinewidth=0.7, capsize=1.6, zorder=5)
	ax.axhline(rmse[0], color=GREY, lw=0.6, ls="--", zorder=4)
	ax.set_xticks(stages)
	ax.set_xticklabels(stages)
	# 标 HT 不标 OOD：正文和 caption 都把留出任务集写作 HT，"OOD" 在正文里指的是别人
	# 工作里的分布外语义，图里换个词读者会当成第三种划分。
	ax.set_ylabel("clean HT RMSE (Nm/kg)")
	rmse_pad = rmse_sd if rmse_sd is not None else 0
	rmse_hi = float(np.max(rmse + rmse_pad))
	ax.set_ylim(float(np.min(rmse - rmse_pad)) - 0.008, rmse_hi + 0.008)
	ax.set_xlim(0.45, 5.55)
	recessive(ax, grid_axis="y")

	# (c) retained aligned torque；阶段 1 无门控
	ax = axes[2]
	panel_tag(ax, "c", "Retention")
	shown = [(s, v, c) for s, v, c in zip(stages, retained, STAGE_RAMP) if v is not None]
	for stage, value, color in shown:
		ax.bar(stage, value, width=0.62, color=color, linewidth=0)
	if shown and retained_sd is not None:
		idx = [int(s) - 1 for s, _, _ in shown]
		ax.errorbar([s for s, _, _ in shown], [v for _, v, _ in shown], yerr=retained_sd[idx],
		            fmt="none", ecolor="#202124", elinewidth=0.7, capsize=1.6, zorder=5)
	if shown:
		values = [v for _, v, _ in shown]
		spread = float(np.max(retained_sd)) if retained_sd is not None else 0.0
		ax.set_ylim(min(values) - 0.03 - spread, max(values) + 0.03 + spread)
	if retained[0] is None:
		ax.text(1, ax.get_ylim()[0] + 0.004, "no gate", rotation=90, ha="center", va="bottom", fontsize=BASE_FONT, color=GREY)
	ax.set_xlim(0.45, 5.55)
	ax.set_xticks(stages)
	ax.set_xticklabels(stages)
	ax.set_ylabel("retained torque")
	recessive(ax, grid_axis="y")

	# (d) 门控带来的反向力矩变化。没有这一列，读者只能从 (a) 读出「阶段 2 的 AUROC 最高」，
	# 而阶段 1–3 的门控在全部 60 行评测里 wrong_delta 精确为 0——检测再准也没有作动后果。
	ax = axes[3]
	panel_tag(ax, "d", "Wrong-direction change")
	draw_wrong_delta_panel(ax, stages, wrong_delta, wrong_sd)

	save(fig, path_stem, formats, FULL_WIDTH)


# ------------------------------------------------------------------ figure 5


def draw_stress_transfer(path_stem: Path, formats: list[str], stress, loso_rows) -> None:
	# hspace 0.78 时 (a) 的图例底边离 (b) 的面板标题只有几个像素，守卫按 1 pt 余量放行但
	# 读起来像连成一块。0.95 拉开一行的距离，图高不变。
	fig, axes = plt.subplots(
		2, 1, figsize=(COL_WIDTH, 3.60),
		gridspec_kw={"height_ratios": [1.12, 0.88], "hspace": 1.05},
	)

	# (a) 最高强度点：AUROC 与保留比例共轴
	ax = axes[0]
	panel_tag(ax, "a", "Highest fault intensity")
	labels: list[str] = []
	positions: list[float] = []
	y = 0.0
	last_group = None
	for group, split, auroc, auroc_sd, retained, retained_sd, delta, delta_sd, online, online_sd in stress:
		if group != last_group:
			if last_group is not None:
				y -= 0.55
			# x 是 axes 分数：-0.02 会让组标题跨过左边框、压在 y 轴刻度文字和第一条网格线上。
			# 挪到边框内 0.02，和图 2(b) 的组标题一致。
			ax.text(0.02, y + 0.55, group, transform=ax.get_yaxis_transform(), fontsize=BASE_FONT, color=GREY, ha="left")
			last_group = group
		positions.append(y)
		labels.append(split)
		ax.errorbar(
			auroc, y, xerr=auroc_sd, fmt="o", markersize=5.0, color=BLUE,
			elinewidth=0.8, capsize=2.0, linestyle="none", zorder=3,
		)
		if online is not None:
			# 空心圆 = Detector-gate；与实心的 Detector-all 同轴，两者间距本身就是要传达的量。
			# 填充必须透明而非白色：丢包两行两个口径逐位相同，白填充会把实心点整个盖掉，
			# 看上去像 Detector-all 缺失。透明 + 略大半径 → 重合时是同心圆，分离时各自可读。
			ax.errorbar(
				online, y, xerr=online_sd, fmt="o", markersize=6.0, markerfacecolor="none",
				markeredgecolor=BLUE, markeredgewidth=0.9, color=BLUE,
				elinewidth=0.8, capsize=2.0, linestyle="none", zorder=4,
			)
		ax.errorbar(
			retained, y, xerr=retained_sd, fmt="D", markersize=5.0,
			markerfacecolor="#ffffff", markeredgecolor=GREEN, markeredgewidth=0.8,
			color=GREEN, elinewidth=0.8, capsize=2.0, linestyle="none", zorder=3,
		)
		# pp 标注统一贴右边界，避免跟着点走出坐标区。
		ax.annotate(
			f"{delta * 100:+.2f} pp", (0.995, y), xycoords=ax.get_yaxis_transform(),
			ha="right", va="center", fontsize=BASE_FONT, color="#202124",
		)
		y -= 1.0
	ax.set_yticks(positions)
	ax.set_yticklabels(labels)
	ax.set_ylim(y + 0.5, 1.05)
	# 这个面板不设 xlabel：三条图例已经逐项写明横轴上画的是什么（Detector-all AUROC /
		# Detector-gate AUROC / retained torque），再加一行 "AUROC · retained aligned torque"
	# 是同一件事说两遍，而且这张单栏图压到 2.44 in 后图例正好退到该标签上（压字 7.7 pt）。
	# 删标签比把图例推远省版面，信息不减。
	values = [v for row in stress for v in (row[2], row[4]) if v is not None]
	values += [row[8] for row in stress if row[8] is not None]
	# 右侧要留出「pp 标注 + 标记半径」的宽度：标注右对齐在 0.995 轴分数处，而 0.075 的
	# 余量是按数据单位手填的，装不下 "-2.30 pp"（8.5 pt 下约 0.55 in），于是文字压在
	# 0.96 那个实心圆上。按实测文字宽换算成数据单位来定 xlim，改字号或位数都不会再撞。
	lo, marker_hi = min(values) - 0.03, max(values) + 0.02
	# 迭代到不动点：text_extent_in_data 按当前 xlim 把磅值换算成数据单位，而这里就在改
	# xlim——量一次就定的话换算比例已经过期，留出的余量会偏小。三轮之内稳定。
	ax.set_xlim(lo, marker_hi)
	for _ in range(6):
		# 按实测宽度取最宽的一条，不是字符串的字典序最大：比例字体下位数相同也不等宽。
		text_w = max(
			text_extent_in_data(ax, f"{row[6] * 100:+.2f} pp", BASE_FONT)[0] for row in stress
		)
		per_pt, _ = points_to_data(ax)
		hi = marker_hi + text_w + 6.0 * per_pt
		if abs(hi - ax.get_xlim()[1]) < 1e-4:
			break
		ax.set_xlim(lo, hi)
	has_online = any(row[8] is not None for row in stress)
	handles = [plt.Line2D([], [], marker="o", color=BLUE, lw=0, markersize=4.0, label="all AUROC")]
	if has_online:
		handles.append(plt.Line2D(
			[], [], marker="o", color=BLUE, markerfacecolor="none", lw=0,
				markersize=5.2, label="gate AUROC",
		))
	handles.append(plt.Line2D([], [], marker="D", color=GREEN, markerfacecolor="#ffffff", lw=0, markersize=4.0, label="retention"))
	ax.legend(
		handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.66),
		frameon=False, ncol=2, handlelength=0.8,
		columnspacing=0.8,
	)
	recessive(ax, grid_axis="x")

	# (b) LOSO：逐折效应量 + bootstrap 均值与 95% CI
	ax = axes[1]
	panel_tag(ax, "b", "Leave-one-subject-out transfer")
	if loso_rows is None:
		ax.axis("off")
		save(fig, path_stem, formats, COL_WIDTH)
		return
	effects = np.sort(np.asarray(loso_rows["effects"], dtype=float) * 100.0)
	mean_pp = loso_rows["mean"] * 100.0
	lo_pp, hi_pp = (v * 100.0 for v in loso_rows["ci"])
	jitter = np.linspace(-0.22, 0.22, effects.size)
	ax.plot(
		effects, jitter, "o", markersize=3.4, color=RED, markerfacecolor="#ffffff",
		markeredgecolor=RED, markeredgewidth=0.7, linestyle="none", zorder=3,
		label=f"subjects ($n={effects.size}$)",
	)
	ax.errorbar(
		mean_pp, -0.72, xerr=[[mean_pp - lo_pp], [hi_pp - mean_pp]], fmt="D", markersize=4.2,
		color=BLUE, ecolor=BLUE, elinewidth=0.9, capsize=2.2, capthick=0.9, zorder=4,
		label="mean, 95% CI",
	)
	ax.axvline(0.0, color=GREY, lw=0.5)
	# CI 完全在 0 右侧才是「跨被试稳定」的证据，直接把这句话写在图里。
	ax.annotate(
		f"CI excludes zero; {loso_rows['improved']}/{effects.size} subjects improve",
		(0.985, -1.16), xycoords=ax.get_yaxis_transform(), ha="right", va="center",
		fontsize=BASE_FONT, color=GREY,
	)
	ax.set_yticks([0.0, -0.72])
	ax.set_yticklabels(["folds", "pooled"])
	ax.set_ylim(-1.45, 0.55)
	ax.set_xlabel("wrong-direction reduction (pp)")
	span = max(float(effects.max()), hi_pp)
	ax.set_xlim(-0.06 * span, span * 1.12)
	# 两条图例并排时比这个单栏面板宽 39 px，右端 "CI" 越出边框。缩到 5.4 pt 后
	# 宽 684 px，留得下余量；标签文案照旧，n 和置信水平都还写在图里。
	# y 从 −0.62 降到 −0.92：这里的 xlabel 带单位，不能像 (a) 那样删，只能让图例落到它下面。
	ax.legend(
		loc="lower center", bbox_to_anchor=(0.5, -1.18), frameon=False, ncol=1,
		handlelength=1.0, columnspacing=0.8,
	)
	recessive(ax, grid_axis="x")
	save(fig, path_stem, formats, COL_WIDTH)


def main() -> None:
	args = parse_args()
	setup_style()
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	formats = [f.strip() for f in args.formats.split(",") if f.strip()]
	suite = Suite(Path(args.suite_dir))

	cells, macro = resolve_detection(suite)
	safety = resolve_safety(suite)
	ablation = resolve_ablation(suite)
	stress = resolve_stress(suite)
	loso_rows, _ = resolve_loso(suite)

	overview_series = load_overview_timeseries(Path(args.overview_timeseries))
	suite.sources.append("overview=" + ("export" if overview_series else "schematic"))
	# 引子图只在有实测导出时才画，所以它自己单独记一条来源：没画出来时验收行会显示
	# teaser=absent，而不是静默少一张图（main.tex 会因为缺 PDF 直接编译失败）。
	drew_teaser = draw_teaser(output_dir / "fig_teaser", formats, overview_series)
	suite.sources.append("teaser=" + ("export" if drew_teaser else "absent"))
	draw_method_overview(output_dir / "fig_method_overview", formats, overview_series)
	draw_detection_taxonomy(output_dir / "fig_detection_taxonomy", formats, cells, macro)
	draw_detection_full(output_dir / "fig_detection_full", formats, cells)
	draw_safety_utility(output_dir / "fig_safety_utility", formats, safety)
	draw_ablation_summary(output_dir / "fig_ablation_summary", formats, ablation)
	draw_stress_transfer(output_dir / "fig_stress_transfer", formats, stress, loso_rows)

	print(f"Wrote {6 + int(drew_teaser)} figure(s) x {len(formats)} format(s) to {output_dir}")
	print("data sources: " + ", ".join(suite.sources))


if __name__ == "__main__":
	main()
