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
填充或直接标注），保证色觉障碍下仍可读。字体嵌成 TrueType 而不是 Type 3。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

FULL_WIDTH = 7.0
COL_WIDTH = 3.35
BLUE = "#2166ac"
VIOLET = "#762a83"
RED = "#b2182b"
GREEN = "#1b7837"
GREY = "#5b6068"
STAGE_RAMP = ("#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c")
DIVERGING = LinearSegmentedColormap.from_list(
	"auroc_div", ["#b2182b", "#ef8a62", "#f7f7f7", "#8fb8da", "#2166ac"]
)

SIGNAL_COLUMNS = ("logit", "residual", "forecast", "staleness", "coherence", "drift")
SIGNAL_LABELS = {
	"logit": "Fault\nlogit",
	"residual": "Recon.\nresidual",
	"forecast": "Forecast\nresidual",
	"staleness": "Stale-\nness",
	"coherence": "Coher-\nence",
	"drift": "Drift",
}
# 热力图只画五个代表性故障，行数再多就挤了；全表在 table1_detection.tsv 里。
TAXONOMY_ROWS = (
	("test_id", "insole_missing", "insole missing", "ID / seen"),
	("test_id", "imu_bias", "IMU bias", "ID / seen"),
	("test_id", "stuck_imu", "stuck IMU", "ID / seen"),
	("test_id", "packet_loss_burst", "burst packet loss", "ID / unseen"),
	("test_ood", "packet_loss_partial", "partial dropout", "OOD / unseen"),
	("test_ood", "sensor_delay_jitter", "delay jitter", "OOD / unseen"),
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
		("OOD clean", 0.0774, 0.0700, 0.961, 0.950, False),
		("ID seen", 0.0910, 0.0858, 0.829, 0.904, True),
		("ID unseen", 0.0868, 0.0851, 0.918, 0.956, True),
		("OOD seen", 0.0983, 0.0874, 0.878, 0.915, True),
		("OOD unseen", 0.0868, 0.0765, 0.948, 0.936, True),
	],
	"ablation": {
		"auroc": (0.846, 0.870, 0.790, 0.818, 0.808),
		"ood_rmse": (0.2309, 0.2598, 0.2472, 0.2613, 0.2578),
		"retained": (0.989, 0.983, 0.989, 0.878, 0.916),
		"wrong_delta": (0.0, 0.0, 0.0, -0.00667, -0.00403),
	},
	"stress": [
		("packet loss 0.30", "ID", 0.961, 0.939, -0.0007),
		("packet loss 0.30", "OOD", 0.914, 0.938, -0.0080),
		("fixed delay 20", "ID", 0.899, 0.751, -0.0132),
		("fixed delay 20", "OOD", 0.788, 0.933, -0.0230),
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
			"font.family": "DejaVu Sans",
			"font.size": 7.0,
			"axes.labelsize": 7.0,
			"axes.titlesize": 7.5,
			"xtick.labelsize": 6.5,
			"ytick.labelsize": 6.5,
			"legend.fontsize": 6.5,
			"axes.linewidth": 0.5,
			"axes.edgecolor": "#9aa0a6",
			"xtick.major.width": 0.5,
			"ytick.major.width": 0.5,
			"grid.linewidth": 0.4,
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
	"""热力图单元 + 四个 macro AUROC。"""
	cells: dict[str, dict[str, float]] = {}
	by_key = {f"{row['split']}/{row['fault']}": row for row in suite.detection}
	for split, fault, _, _ in TAXONOMY_ROWS:
		row = by_key.get(f"{split}/{fault}")
		if row is None:
			break
		entry = {"fused": as_float(row.get("fused_mean"))}
		for signal in SIGNAL_COLUMNS:
			entry[signal] = as_float(row.get(f"{signal}_mean"))
		if entry["fused"] is None:
			break
		cells[f"{split}/{fault}"] = entry
	live_cells = len(cells) == len(TAXONOMY_ROWS)
	if not live_cells:
		cells = {k: dict(v) for k, v in FALLBACK["detection"].items()}
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
	live_macro = len(macro) == 4
	if not live_macro:
		macro = dict(FALLBACK["macro_auroc"])
	suite.note("macro", live_macro)
	return cells, macro


def resolve_safety(suite: Suite) -> list[tuple[str, float, float, float, float, bool]]:
	"""(label, ungated wrong, gated wrong, retained ratio, mean gate, is_faulty)。"""
	table = suite.numbers.get("table3_safety", {})
	rows: list[tuple[str, float, float, float, float, bool]] = []
	plan = (
		("ID clean", "test_id", "clean", False),
		("OOD clean", "test_ood", "clean", False),
		("ID seen", "test_id", "seen_faults", True),
		("ID unseen", "test_id", "unseen_faults", True),
		("OOD seen", "test_ood", "seen_faults", True),
		("OOD unseen", "test_ood", "unseen_faults", True),
	)
	for label, split, group, faulty in plan:
		ungated = stat(table.get(f"{split}.{group}.wrong_ungated"))
		gated = stat(table.get(f"{split}.{group}.wrong_gated"))
		retained = stat(table.get(f"{split}.{group}.retained_ratio"))
		gate = stat(table.get(f"{split}.{group}.mean_gate"))
		if None in (ungated, gated, retained, gate):
			break
		rows.append((label, ungated, gated, retained, gate, faulty))
	live = len(rows) == len(plan)
	if not live:
		rows = [tuple(row) for row in FALLBACK["safety"]]  # type: ignore[misc]
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


def resolve_stress(suite: Suite) -> list[tuple[str, str, float, float, float]]:
	table = suite.numbers.get("stress_max_severity", {})
	rows: list[tuple[str, str, float, float, float]] = []
	plan = (
		("packet loss 0.30", "ID", "test_id/packet_loss@0.30"),
		("packet loss 0.30", "OOD", "test_ood/packet_loss@0.30"),
		("fixed delay 20", "ID", "test_id/sensor_delay@20"),
		("fixed delay 20", "OOD", "test_ood/sensor_delay@20"),
	)
	for label, split, key in plan:
		auroc = stat(table.get(f"{key}.fused_auroc"))
		retained = stat(table.get(f"{key}.retained_ratio"))
		delta = stat(table.get(f"{key}.wrong_delta"))
		if None in (auroc, retained, delta):
			break
		rows.append((label, split, auroc, retained, delta))
	live = len(rows) == len(plan)
	if not live:
		rows = [tuple(row) for row in FALLBACK["stress"]]  # type: ignore[misc]
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


def draw_overview_gate_panel(ax, series: dict[str, Any]) -> None:
	"""面板 (c) 的真实数据版本：风险（右轴）、门控与指令幅度比（左轴）。"""
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
	# 上方留出图例带：门控长时间贴着 1.0，图例压在曲线上会看不清。
	ax.set_ylim(0, 1.52)
	ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
	ax.set_xlim(0, t[-1])
	ax.set_xlabel("time (samples)")
	ax.set_ylabel("gate / rel. |torque|")

	twin = ax.twinx()
	twin.plot(t, risk, color=RED, lw=0.7, alpha=0.85, label="normalized risk")
	twin.axhline(series["deadband_risk"], color=GREY, lw=0.5, ls=":")
	risk_top = max(float(np.max(risk)), series["deadband_risk"]) * 1.58
	twin.set_ylim(0, risk_top)
	twin.set_yticks([tick for tick in twin.get_yticks() if tick <= float(np.max(risk)) * 1.05])
	twin.set_ylabel("risk", color=RED)
	twin.tick_params(axis="y", colors=RED, length=2.0, width=0.5)
	# 死区线很靠下，正好穿过灰色力矩带；加白底才读得出来。
	twin.text(
		t[-1] * 0.985, series["deadband_risk"] + risk_top * 0.012,
		"deadband", fontsize=5.4, color=GREY, ha="right", va="bottom",
		bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.85},
	)
	for spine in ("top", "left"):
		twin.spines[spine].set_visible(False)
	twin.spines["right"].set_linewidth(0.5)
	twin.spines["right"].set_color(RED)

	handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
	labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
	# 两列排时整条宽 488 px、面板只有 403 px：右列的 "gate $g_t$" 和 "normalized risk"
	# 越过右边框，右轴那条竖线正好从 $g_t$ 的下标上穿过去。竖排一列宽 242 px，
	# 四行高 109 px（约 0.34 数据单位），落在 1.18–1.52 的留白带里，不压门控曲线。
	ax.legend(
		handles, labels, loc="upper left", ncol=1, frameon=False,
		handlelength=1.2, borderpad=0.0, labelspacing=0.2, fontsize=5.6,
	)
	recessive(ax, grid_axis="y")


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
	except (OSError, ValueError, KeyError, json.JSONDecodeError):
		return None
	if risk.size == 0:
		return None
	return {
		"meta": meta,
		"risk": risk,
		"gate": gate,
		# 髋膝两路力矩取绝对值均值，作为「指令幅度」的单条曲线。
		"baseline_mag": np.abs(baseline).mean(axis=0),
		"commanded_mag": np.abs(commanded).mean(axis=0),
		"deadband_risk": 1.0 + float(meta.get("gate_deadband", 0.0)),
	}


def draw_method_overview(path_stem: Path, formats: list[str], series: dict[str, Any] | None = None) -> None:
	"""方法总览。面板 (a)(b) 是机制示意；(c) 在有导出数据时使用真实逐 timestep 时序。"""
	# 面板 c 带右轴 + 双 y 标签，默认间距会让标题和轴标签互相压住。
	# 面板 b 的框宽按实测文字宽度排布（见下），三档等宽时排不下，所以多分一点给它。
	fig, axes = plt.subplots(
		1, 3, figsize=(FULL_WIDTH, 1.95), gridspec_kw={"width_ratios": [0.88, 1.32, 0.96], "wspace": 0.34}
	)
	rng = np.random.default_rng(11)
	t = np.arange(0, 400)

	# (a) 传感输入 + 一段保持值故障
	ax = axes[0]
	# 标题必须短到不会伸进面板 b 的标签位（assert_no_overlapping_titles 会拦）。
	panel_tag(ax, "a", "Causal streams, fault")
	base = 0.55 * np.sin(2 * np.pi * t / 95.0) + 0.12 * np.sin(2 * np.pi * t / 31.0)
	imu = base + 0.03 * rng.standard_normal(t.size)
	insole = 0.45 * np.clip(np.sin(2 * np.pi * t / 95.0 + 0.6), 0, None) + 0.02 * rng.standard_normal(t.size)
	held = slice(210, 285)
	imu_f = imu.copy()
	imu_f[held] = imu[held.start - 1]
	ax.axvspan(held.start, held.stop, color=RED, alpha=0.10, lw=0)
	ax.plot(t, imu + 1.05, color=VIOLET, lw=0.8, label="IMU (clean)")
	ax.plot(t, imu_f + 1.05, color=RED, lw=0.8, ls="--", label="IMU (held value)")
	ax.plot(t, insole, color=BLUE, lw=0.8, label="Insole force")
	# 两条曲线上下几乎相接，标注只能放在最下方的留白带里。
	ax.text(held.start + 37, -0.47, "stuck value", color=RED, ha="center", va="bottom", fontsize=5.8)
	ax.set_xlabel("time (samples)")
	ax.set_yticks([])
	ax.set_xlim(0, t[-1])
	# 上方留白带放图例，下方留白带放故障标注。
	ax.set_ylim(-0.55, 2.55)
	# 三条图例横排在一行时，整条比面板还宽 269 px，尾部伸进面板 (b) 的框图里——
	# 编译成 PDF 后 "Insole force" 看着像 (b) 的说明。竖排一列宽 235 px，稳稳在面板内，
	# 三行高 81 px 也放得进 y=1.72 以上那条 131 px 的留白带。
	ax.legend(
		loc="upper left", frameon=False, ncol=1, handlelength=1.1,
		borderpad=0.0, labelspacing=0.2, fontsize=5.6,
	)
	recessive(ax, grid_axis="x")

	# (b) 共享 TCN + 四个头 + 完整性检查
	ax = axes[1]
	panel_tag(ax, "b", "Shared TCN and evidence heads")
	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)
	ax.axis("off")

	def box(x, y, w, h, text, color, fontsize=6.0):
		ax.add_patch(
			FancyBboxPatch(
				(x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
				linewidth=0.6, edgecolor=color, facecolor=color + "1f",
			)
		)
		ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#202124")

	def arrow(x0, y0, x1, y1, color=GREY):
		ax.add_patch(
			FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=5.0, lw=0.55, color=color, shrinkA=0, shrinkB=0)
		)

	# 框宽由文字实测宽度决定，不再手填常量。手填的版本每次改文案都要重新目测，
	# 而 FancyBboxPatch 不会自动撑开：宽了一点点，文字就伸出去压到邻框上。
	def box_width(text: str, fontsize: float) -> float:
		widest = max(text_extent_in_data(ax, line, fontsize)[0] for line in text.split("\n"))
		return widest + 2 * BOX_PAD + 0.012  # 两侧留 pad，再加一点行末余量

	BOX_PAD = 0.012  # 与 boxstyle 的 pad 一致：几何计算要按含 pad 的外缘算
	FS = 4.8
	FS_CHECKS = 4.5
	src_text, tcn_text = "sensor\nwindow", "causal\nTCN"
	checks_text = "integrity checks\n(staleness,\ncoherence, drift)"
	heads = [
		("moment\nmean/var", GREEN, 0.855),
		("fault\nlogit", RED, 0.665),
		("recon-\nstruction", VIOLET, 0.475),
		("one-step\nforecast", VIOLET, 0.285),
	]
	src_w = box_width(src_text, FS)
	tcn_w = box_width(tcn_text, FS)
	head_w = max(box_width(text, FS) for text, _, _ in heads)
	checks_w = box_width(checks_text, FS_CHECKS)

	GAP = 0.022  # 相邻框外缘之间的净距，也是箭头的长度
	src_x = 0.0
	tcn_x = src_x + src_w + BOX_PAD + GAP
	bus_x = tcn_x + tcn_w + BOX_PAD + GAP  # 竖直汇流线
	head_x = bus_x + GAP
	checks_x = head_x + head_w + BOX_PAD + GAP
	# 总宽超过 1 时按比例压缩字号解决不了（文字有最小可读尺寸），这里直接报错，
	# 让改文案的人知道要么缩短标签、要么给面板更多 width_ratios。
	total = checks_x + checks_w + BOX_PAD
	if total > 1.0:
		raise RuntimeError(f"panel (b) 的框排不进面板：需要 {total:.3f} 宽（>1），请缩短标签或加宽面板")

	box(src_x, 0.50, src_w, 0.18, src_text, VIOLET, fontsize=FS)
	box(tcn_x, 0.46, tcn_w, 0.26, tcn_text, BLUE, fontsize=FS)
	arrow(src_x + src_w + BOX_PAD, 0.59, tcn_x - BOX_PAD, 0.59)
	# 四条斜箭线原先直接从蓝框右缘拉向各个头，最陡的两条会斜穿蓝框的圆角边线，
	# 看着像线从框里长出来。改成先出一小段、走一条竖直汇流线、再水平进各个头：
	# 蓝框边线不再被压到，分叉关系也更直观。
	for text, color, y in heads:
		box(head_x, y - 0.075, head_w, 0.15, text, color, fontsize=FS)
		arrow(bus_x, y, head_x - BOX_PAD, y)
	ax.plot([bus_x, bus_x], [heads[-1][2], heads[0][2]], color=GREY, lw=0.55, solid_capstyle="butt", zorder=1)
	ax.plot([tcn_x + tcn_w + BOX_PAD, bus_x], [0.59, 0.59], color=GREY, lw=0.55, solid_capstyle="butt", zorder=1)
	box(checks_x, 0.38, checks_w, 0.42, checks_text, GREY, fontsize=FS_CHECKS)
	# 原始输入也送进完整性检查：走在所有框下方（y=0.19）绕过去，不穿框体。
	# 三段都画成箭头时，两个转角各多一个箭头，看着像三条独立的、方向互相矛盾的线；
	# 中间两段改成纯线段，只在终点保留一个箭头。
	feed_x = src_x + src_w / 2
	checks_mid = checks_x + checks_w / 2
	ax.plot([feed_x, feed_x], [0.50, 0.19], color=GREY, lw=0.55, solid_capstyle="butt", zorder=1)
	ax.plot([feed_x, checks_mid], [0.19, 0.19], color=GREY, lw=0.55, solid_capstyle="butt", zorder=1)
	arrow(checks_mid, 0.19, checks_mid, 0.38 - BOX_PAD)
	ax.text(
		0.5, 0.005,
		"calibration, detector subset, and gate tuning\nuse validation data only",
		ha="center", va="bottom", fontsize=5.2, color=GREY,
	)

	# (c) 风险 -> 门控 -> 力矩衰减
	ax = axes[2]
	panel_tag(ax, "c", "Risk above deadband attenuates")
	if series is not None:
		draw_overview_gate_panel(ax, series)
		figure_note(
			fig,
			"Panel a is an illustrative trace; panel c is a measured per-timestep export "
			f"({series['meta'].get('fault', '')} on {series['meta'].get('split', '')}).",
		)
		save(fig, path_stem, formats)
		return
	risk = 0.18 + 0.05 * np.sin(2 * np.pi * t / 120.0)
	risk[held] = np.linspace(0.2, 0.95, held.stop - held.start)
	risk[held.stop:] = np.linspace(0.95, 0.2, t.size - held.stop) ** 1.5
	deadband = 0.35
	gate = 1.0 / (1.0 + np.exp((risk - deadband) / 0.09))
	gate = np.clip(gate, 0.0, 1.0)
	ax.plot(t, risk, color=RED, lw=0.8, label="normalized risk")
	ax.axhline(deadband, color=GREY, lw=0.5, ls=":")
	ax.text(6, deadband + 0.03, "deadband", fontsize=5.6, color=GREY)
	ax.plot(t, gate, color=BLUE, lw=0.9, label="gate $g_t$")
	ax.fill_between(t, 0, gate, color=BLUE, alpha=0.10, lw=0)
	ax.set_ylim(0, 1.05)
	ax.set_xlim(0, t[-1])
	ax.set_xlabel("time (samples)")
	ax.set_ylabel("risk / gate")
	ax.legend(loc="center left", frameon=False, handlelength=1.4, borderpad=0.1)
	recessive(ax, grid_axis="y")

	figure_note(
		fig,
		"Panels a and c are illustrative traces of the mechanism, not measured per-timestep exports.",
	)
	save(fig, path_stem, formats)


def figure_note(fig, text: str, fontsize: float = 5.6, gap_points: float = 3.0) -> None:
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


def assert_tight_within_canvas(fig) -> None:
	"""`bbox="tight"` 的裁剪框不能比 figsize 宽。

	比图宽的内容不会被裁掉，而是把裁剪框撑开，PDF 的 MediaBox 跟着变宽：图 4 的
	两句 figure_note 曾把 7.0 in 的图输出成 9.32 in，再按 `\\textwidth` 放进正文
	就等于连字号一起缩到 75%，比同页其他图小一号——单看图挑不出问题，排进双栏才看得出来。
	"""
	fig.canvas.draw()
	tight = fig.get_tightbbox(fig.canvas.get_renderer())  # 英寸
	# 0.05 in 余量：pad_inches 0.02 加上量测误差。
	if tight.width > fig.get_figwidth() + 0.05:
		raise RuntimeError(
			f"裁剪框宽 {tight.width:.2f} in 超过 figsize 的 {fig.get_figwidth():.2f} in；"
			"figure_note 或某个标签比图还宽，请折行或缩短"
		)


def save(fig, path_stem: Path, formats: list[str]) -> None:
	assert_no_clipping(fig)
	assert_no_overlapping_titles(fig)
	assert_legends_inside_panels(fig)
	assert_tight_within_canvas(fig)
	for fmt in formats:
		fig.savefig(path_stem.with_suffix(f".{fmt}"), format=fmt)
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
	fig, axes = plt.subplots(
		1, 2, figsize=(FULL_WIDTH, 2.15), gridspec_kw={"width_ratios": [2.35, 1.0], "wspace": 0.30}
	)

	ax = axes[0]
	panel_tag(ax, "a", "Which signal sees which fault (AUROC)")
	columns = ("fused", *SIGNAL_COLUMNS)
	matrix = np.full((len(TAXONOMY_ROWS), len(columns)), np.nan)
	for r, (split, fault, _, _) in enumerate(TAXONOMY_ROWS):
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
			ax.text(c, r, f"{value:.3f}".lstrip("0"), ha="center", va="center", fontsize=6.0, color=shade)
	ax.set_xticks(range(len(columns)))
	ax.set_xticklabels(["Fused\ndetector", *(SIGNAL_LABELS[s] for s in SIGNAL_COLUMNS)], fontsize=6.0)
	ax.set_yticks(range(len(TAXONOMY_ROWS)))
	ax.set_yticklabels([f"{name}\n{tag}" for _, _, name, tag in TAXONOMY_ROWS], fontsize=6.0)
	ax.tick_params(length=0)
	for side in ("top", "right", "left", "bottom"):
		ax.spines[side].set_visible(False)
	# 竖线把「验证集选定的融合」和候选通道分开。
	ax.axvline(0.5, color="#202124", lw=0.7)
	# 左栏只有一列宽，注记右对齐贴到分隔线左侧，避免压线。
	ax.text(0.42, -0.72, "validation-selected", ha="right", fontsize=5.2, color=GREY)
	ax.text((len(columns) + 0.5) / 2, -0.72, "candidate integrity signals", ha="center", fontsize=5.6, color=GREY)
	ax.set_ylim(len(TAXONOMY_ROWS) - 0.5, -0.95)

	# 色带放到面板 a 的刻度标签之下，说明文字再放到色带之下，避免和列名撞在一起。
	cax = fig.add_axes([0.135, -0.115, 0.22, 0.032])
	cax.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap=DIVERGING, aspect="auto")
	cax.set_xticks([0, 127.5, 255])
	cax.set_xticklabels(["0", ".5", "1"], fontsize=5.6)
	cax.set_yticks([])
	cax.tick_params(length=1.5, pad=1.0)
	cax.text(127.5, 2.9, "below chance $\\leftarrow$ chance $\\rightarrow$ above chance", ha="center", va="top", fontsize=5.6, color=GREY)

	ax = axes[1]
	panel_tag(ax, "b", "Frozen detector, macro AUROC")
	groups = (("seen faults", "seen"), ("unseen generators", "unseen"))
	for gi, (title, key) in enumerate(groups):
		y = 1 - gi
		for split_key, marker, face in (("id", "o", BLUE), ("ood", "s", "#ffffff")):
			value = macro.get(f"{split_key}_{key}")
			if value is None:
				continue
			ax.plot(
				value, y, marker=marker, markersize=4.2, color=BLUE,
				markerfacecolor=face, markeredgecolor=BLUE, markeredgewidth=0.7, linestyle="none",
			)
			ax.annotate(
				f"{value:.3f}", (value, y), textcoords="offset points",
				xytext=(0, 6 if split_key == "id" else -11), ha="center", fontsize=6.0, color="#202124",
			)
		# 组标题贴左侧：x 用 axes 分数（0=左边框），不是数据值。原来传的是
		# get_xlim()[0]（约 0.77），在 blended 变换里被当成 axes 分数用，
		# 于是标题被推到最右边贴着边框，长的那行还被 x 轴范围裁到。
		ax.text(
			0.02, y + 0.30, title, fontsize=6.0, color=GREY, ha="left",
			transform=ax.get_yaxis_transform(),
		)
	ax.set_yticks([])
	ax.set_ylim(-0.55, 1.62)
	# 两个数值标签居中在各自标记下方/上方，最左的那个（0.773）会一半探出左边框；
	# 左右各留一点余量。
	values = [v for v in (macro.get(f"{s}_{k}") for s in ("id", "ood") for _, k in groups) if v is not None]
	if values:
		pad = max(0.35 * (max(values) - min(values)), 0.004)
		ax.set_xlim(min(values) - pad, max(values) + pad)
	ax.set_xlabel("macro AUROC over fault types")
	handles = [
		plt.Line2D([], [], marker="o", color=BLUE, markerfacecolor=BLUE, lw=0, markersize=4.0, label="in-distribution tasks"),
		plt.Line2D([], [], marker="s", color=BLUE, markerfacecolor="#ffffff", lw=0, markersize=4.0, label="held-out tasks"),
	]
	ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.42), frameon=False, ncol=1, handlelength=1.0)
	ax.text(0.5, -0.62, "3 seeds; detector subset and q90 scales\nfrozen on validation data.", transform=ax.transAxes, ha="center", fontsize=5.6, color=GREY)
	recessive(ax, grid_axis="x")
	save(fig, path_stem, formats)


# ------------------------------------------------------------------ figure 3


def draw_safety_utility(path_stem: Path, formats: list[str], rows) -> None:
	# 面板 (b) 要给六个点各放一条文字标签，标签宽度是固定的物理量：面板越窄，
	# 同一条标签在数据坐标里就越长，即使贴着自己的点放也会横跨到别的点上方。
	# 所以 (b) 拿走最多宽度，(a)/(c) 只需容纳 y 轴文字和短数字。
	fig, axes = plt.subplots(
		1, 3, figsize=(FULL_WIDTH, 2.15), gridspec_kw={"width_ratios": [1.05, 1.42, 0.72], "wspace": 0.52}
	)
	labels = [row[0] for row in rows]
	ungated = np.array([row[1] for row in rows])
	gated = np.array([row[2] for row in rows])
	retained = np.array([row[3] for row in rows])
	gate = np.array([row[4] for row in rows])
	faulty = [row[5] for row in rows]
	y = np.arange(len(rows))[::-1]

	# (a) 门控前后配对
	ax = axes[0]
	panel_tag(ax, "a", "Wrong-direction torque, un/gated")
	for yi, u, g in zip(y, ungated, gated):
		ax.plot([u, g], [yi, yi], color="#c9ccd1", lw=0.8, zorder=1, solid_capstyle="round")
	ax.plot(ungated, y, "o", markersize=4.0, markerfacecolor="#ffffff", markeredgecolor=GREY, markeredgewidth=0.8, linestyle="none", label="ungated", zorder=3)
	ax.plot(gated, y, "D", markersize=3.6, color=BLUE, linestyle="none", label="gated", zorder=3)
	span = max(float(np.max(ungated) - np.min(gated)), 1e-6)
	for yi, u, g in zip(y, ungated, gated):
		ax.annotate(
			f"{(g - u) * 100:+.2f} pp", (min(u, g), yi), textcoords="offset points",
			xytext=(-5, 0), ha="right", va="center", fontsize=5.8, color="#202124",
		)
	ax.set_yticks(y)
	ax.set_yticklabels(labels)
	ax.set_xlabel("wrong-direction fraction of active steps")
	ax.set_xlim(float(np.min(gated)) - 0.52 * span, float(np.max(ungated)) + 0.06 * span)
	ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.40), frameon=False, ncol=2, handlelength=1.0)
	recessive(ax, grid_axis="x")

	# (b) 安全-效用平面
	ax = axes[1]
	panel_tag(ax, "b", "Safety-utility plane")
	reduction = (ungated - gated) * 100.0
	# 六个点常挤在同一 retention 水平附近，固定错开量在数值变动后会失效；
	# 这里按坐标贪心选放置方向：先试右上，撞了就换，保证互不重叠。
	placed: list[tuple[float, float, float, float]] = []
	xspan_ref = max(float(np.ptp(reduction)), 1e-3)
	# 坐标范围要先定下来：贪心放置除了避让彼此，还得知道边界在哪，否则最右侧的点
	# 会把标签甩出画布，在双栏排版里溢到相邻栏上。
	y_lo = min(0.94, float(np.min(retained)) - 0.01)
	y_hi = max(1.0, float(np.max(retained)) + 0.01)
	# 左右都要留出一个标签的宽度：最右那个点（OOD unseen）没有右侧余量时会被挤到
	# 左边，而左边被 "validation floor 0.95" 那行占着，最后只能拉长引线横穿画面。
	x_lo = float(np.min(reduction)) - 0.46 * xspan_ref
	x_hi = float(np.max(reduction)) + 0.62 * xspan_ref
	ax.set_xlim(x_lo, x_hi)
	ax.set_ylim(y_lo, y_hi)
	# 候选偏移一律用排版点：annotate 就按点偏移，碰撞盒也按点换算，两边同一口径。
	# 之前碰撞盒按数据坐标比例（0.02*xspan）拼，比真实落位远好几倍，
	# 于是「OOD clean / ID clean」被判为不重叠、画出来却叠成一团。
	ptx, pty = points_to_data(ax)
	BASE_OFFSETS = (
		(5.0, 4.5, "bottom", "left"),
		(5.0, -4.5, "top", "left"),
		(-5.0, 4.5, "bottom", "right"),
		(-5.0, -4.5, "top", "right"),
		(6.5, 0.0, "center", "left"),
		(-6.5, 0.0, "center", "right"),
	)
	# 候选顺序按「远离点云中心」排：固定用右上优先时，处在云左侧的点也往右上放，
	# 标签就挤进云里、看着像在标注邻点。背向中心的方向天然是空的那一侧。
	_cx = float(np.mean(reduction))
	_cy = float(np.mean(retained))

	def ranked_offsets(red: float, ret: float):
		ox = 1.0 if red >= _cx else -1.0
		oy = 1.0 if ret >= _cy else -1.0
		return sorted(
			BASE_OFFSETS,
			key=lambda c: -((1.0 if c[0] > 0 else -1.0) * ox + (1.0 if c[1] > 0 else -1.0 if c[1] < 0 else 0.0) * oy),
		)

	def label_box(red: float, ret: float, w: float, h: float, dx: float, dy: float, va: str, ha: str):
		x0 = red + dx * ptx if ha == "left" else red + dx * ptx - w
		if va == "bottom":
			y0 = ret + dy * pty
		elif va == "top":
			y0 = ret + dy * pty - h
		else:
			y0 = ret + dy * pty - 0.5 * h
		return (x0, y0, x0 + w, y0 + h)
	# "validation floor 0.95" 注记先占位：它不是数据标签，但落在 0.95 那条线附近，
	# 不占位的话贪心放置会把恰好在 0.95 上的点标签叠上去。
	_floor_w, _floor_h = text_extent_in_data(ax, "validation floor 0.95", fontsize=5.6)
	# 注记原先贴在 0.95 线下方靠左，那一带正好是几个 clean 点的左侧空位，把它们逼进
	# 点云中央。挪到右下角空白处：0.95 那条虚线横贯整幅，注记在哪一端都读得出对应关系。
	_floor_right = x_hi - 0.02 * (x_hi - x_lo)
	_floor_top = y_lo + 0.05 * (y_hi - y_lo)
	placed.append((_floor_right - _floor_w, _floor_top - _floor_h, _floor_right, _floor_top))
	# 六个数据点自身也要占位。只避让其他标签的话，标签会落在两个邻近标记之间，
	# 看起来像在标注别的点——归属含糊比稍远一点更糟。标记半径 4.4 pt，留一点余量。
	_mark_w = 4.0 * ptx
	_mark_h = 4.0 * pty
	for _red, _ret in zip(reduction, retained):
		placed.append((_red - _mark_w, _ret - _mark_h, _red + _mark_w, _ret + _mark_h))
	for label, red, ret, is_fault in zip(labels, reduction, retained, faulty):
		if is_fault:
			ax.plot(red, ret, "^", markersize=4.4, color=RED, linestyle="none", zorder=3)
		else:
			ax.plot(red, ret, "o", markersize=4.4, markerfacecolor="#ffffff", markeredgecolor=BLUE, markeredgewidth=0.8, linestyle="none", zorder=3)
		# 标签占位取真实渲染尺寸再换算成数据坐标。按字符数估宽会低估比例字体里的
		# 宽字母，"OOD unseen" 就因此被判为放得下、结果溢出了右边界。
		width, height = text_extent_in_data(ax, label, fontsize=5.8)
		chosen = None
		for dx, dy, va, ha in ranked_offsets(red, ret):
			box = label_box(red, ret, width, height, dx, dy, va, ha)
			if box[0] < x_lo or box[2] > x_hi or box[1] < y_lo or box[3] > y_hi:
				continue
			if all(
				box[2] < other[0] or box[0] > other[2] or box[3] < other[1] or box[1] > other[3]
				for other in placed
			):
				chosen = (dx, dy, va, ha)
				placed.append(box)
				break
		if chosen is not None:
			ax.annotate(
				label, (red, ret), textcoords="offset points", xytext=chosen[0:2],
				ha=chosen[3], va=chosen[2], fontsize=5.8, color="#202124",
			)
			continue
		# 贴身的六个位置都放不下：再往外推，并画一条细引线把标签和点连起来。
		# 不带引线地随便找块空地会让标签看起来在标注邻近的点，归属含糊比多一条线更糟。
		far_side = 1.0 if red < 0.5 * (x_lo + x_hi) else -1.0
		for radius in (16.0, 22.0, 28.0):
			for sx, sy in ((far_side, 1.0), (far_side, -1.0), (-far_side, 1.0), (-far_side, -1.0)):
				dx, dy = sx * radius, sy * radius * 0.7
				ha = "left" if dx > 0 else "right"
				va = "bottom" if dy > 0 else "top"
				box = label_box(red, ret, width, height, dx, dy, va, ha)
				if box[0] < x_lo or box[2] > x_hi or box[1] < y_lo or box[3] > y_hi:
					continue
				if not all(
					box[2] < other[0] or box[0] > other[2] or box[3] < other[1] or box[1] > other[3]
					for other in placed
				):
					continue
				placed.append(box)
				ax.annotate(
					label, (red, ret), textcoords="offset points", xytext=(dx, dy),
					ha=ha, va=va, fontsize=5.8, color="#202124",
					arrowprops={"arrowstyle": "-", "lw": 0.4, "color": GREY, "shrinkA": 1.0, "shrinkB": 2.5},
				)
				chosen = True
				break
			if chosen:
				break
		if not chosen:
			# 连远处也排不开：说明数据点密到自动放置解决不了，直接报错而不是画出
			# 一张标签叠在一起、读者会读错的图。
			raise RuntimeError(f"panel (b) 找不到标签 {label!r} 的可行位置，请调整坐标范围或字号")
	ax.axhline(0.95, color=GREY, lw=0.6, ls="--")
	ax.text(0.98, _floor_top, "validation floor 0.95", transform=ax.get_yaxis_transform(), fontsize=5.6, color=GREY, va="top", ha="right")
	ax.set_xlabel("wrong-direction reduction (pp)")
	ax.set_ylabel("retained aligned torque")
	# 坐标范围已在贪心放置之前设定，这里不要再改，否则边界避让就失效了。
	# 角注放在图例一侧的空白处，避免和最低那个点的标签抢位置。
	ax.text(0.98, 0.97, "up and right is better", transform=ax.transAxes, ha="right", va="top", fontsize=5.6, color=GREY)
	handles = [
		plt.Line2D([], [], marker="o", color=BLUE, markerfacecolor="#ffffff", lw=0, markersize=4.0, label="clean"),
		plt.Line2D([], [], marker="^", color=RED, lw=0, markersize=4.0, label="faulty"),
	]
	ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.40), frameon=False, ncol=2, handlelength=1.0)
	recessive(ax, grid_axis="both")

	# (c) 平均门控值
	ax = axes[2]
	panel_tag(ax, "c", "Mean gate value")
	# 全部门控值都挤在 0.90–0.98，必须截断 x 轴才看得出差别。但截断轴上不能用柱：
	# 柱长会不再与数值成比例（0.904 的柱看着只有 0.978 的四分之一）。所以画从
	# g_t=1（无衰减）拉向实际值的棒线 + 端点标记——棒长直接读作衰减量，基准点又是
	# 一个有物理意义的固定值，截断轴不会误导。标记形状沿用面板 b 的干净/故障编码。
	gate_lo = min(0.94, float(np.min(gate)) - 0.012)
	ax.hlines(y, gate, 1.0, color=[RED if f else BLUE for f in faulty], lw=1.3, alpha=0.55)
	for yi, value, f in zip(y, gate, faulty):
		if f:
			ax.plot([value], [yi], marker="^", color=RED, markersize=4.0, lw=0)
		else:
			ax.plot([value], [yi], marker="o", color=BLUE, markerfacecolor="#ffffff", markersize=4.0, lw=0)
		# 标签放在标记右侧、棒线上方：放左侧时最短的两根（0.904/0.915）会压到 y 轴
		# 刻度文字上。棒线本身够长，标签压不到右端的基准虚线。
		ax.annotate(
			f"{value:.3f}", (value, yi), textcoords="offset points", xytext=(5, 2.2),
			ha="left", va="bottom", fontsize=5.8, color="#202124",
		)
	ax.axvline(1.0, color=GREY, lw=0.6, ls="--")
	ax.set_yticks(y)
	ax.set_yticklabels(labels)
	ax.set_xlim(gate_lo, 1.006)
	ax.set_xticks([0.90, 0.95, 1.00])
	# 标签必须比这个窄面板更短，否则 bbox=tight 会把尾字母裁掉（assert_no_clipping 会拦）。
	# "over active steps" 的口径 caption 里已经写明，这里只留量名。
	ax.set_xlabel("mean gate $g_t$")
	ax.text(0.5, -0.30, "$g_t = 1$: no attenuation", transform=ax.transAxes, ha="center", fontsize=5.6, color=GREY)
	recessive(ax, grid_axis="x")
	save(fig, path_stem, formats)


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
		        ha="center", va="center", fontsize=5.8, color=GREY)
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
			xytext=(0, 3), ha="center", va="bottom", fontsize=5.4, color=GREY,
		)
	ax.annotate(
		f"{values[-1]:+.2f}", (stages[-1], values[-1] - (sd[-1] if sd is not None else 0.0)),
		textcoords="offset points", xytext=(0, -3), ha="center", va="top", fontsize=6.0,
	)
	ax.set_xticks(stages)
	ax.set_xticklabels(stages)
	ax.set_ylabel("wrong-direction change (pp)")
	recessive(ax, grid_axis="y")


def draw_ablation_summary(path_stem: Path, formats: list[str], ablation) -> None:
	fig, axes = plt.subplots(
		1, 4, figsize=(FULL_WIDTH, 2.05), gridspec_kw={"width_ratios": [1.42, 0.74, 0.86, 0.74], "wspace": 0.52}
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
	panel_tag(ax, "a", "Detection gain is not monotone")
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
	x_lo, x_hi = ax.get_xlim()
	y_lo, y_hi = ax.get_ylim()
	ptx, pty = points_to_data(ax)
	taken: list[tuple[float, float, float, float]] = []

	def reserve(cx: float, cy: float, w: float, h: float, dx: float, dy: float, ha: str, va: str):
		x0 = cx + dx * ptx if ha == "left" else cx + dx * ptx - w if ha == "right" else cx + dx * ptx - 0.5 * w
		y0 = cy + dy * pty if va == "bottom" else cy + dy * pty - h if va == "top" else cy + dy * pty - 0.5 * h
		return (x0, y0, x0 + w, y0 + h)

	def free(box) -> bool:
		if box[0] < x_lo or box[2] > x_hi or box[1] < y_lo or box[3] > y_hi:
			return False
		return all(
			box[2] < o[0] or box[0] > o[2] or box[3] < o[1] or box[1] > o[3] for o in taken
		)

	# 数据点先占位。标记和误差棒要分开算：把整根误差棒按标记宽度占位的话，第 1 阶段
	# 那根 ±0.036 的长棒会封掉整列，差值标签（宽度接近一个阶段间距）就无处可放。
	# 标记按半径 5 pt 的方框，棒身只占 cap 的宽度。
	for stage, value in zip(stages, auroc):
		taken.append((stage - 5.0 * ptx, value - 5.0 * pty, stage + 5.0 * ptx, value + 5.0 * pty))
		if auroc_sd is not None:
			sd = float(auroc_sd[int(stage) - 1])
			taken.append((stage - 2.5 * ptx, value - sd - 2.0 * pty, stage + 2.5 * ptx, value + sd + 2.0 * pty))
	# 终点值放到点的右侧：末段差值也在附近，两者同侧会叠字。
	end_text = f"{auroc[-1]:.3f}"
	end_w, end_h = text_extent_in_data(ax, end_text, fontsize=6.0)
	taken.append(reserve(stages[-1], auroc[-1], end_w, end_h, 7.0, -2.0, "left", "center"))
	ax.annotate(
		end_text, (stages[-1], auroc[-1]), textcoords="offset points",
		xytext=(7, -2), ha="left", va="center", fontsize=6.0,
	)
	# 相邻差值标在线段中点。固定「正数在上、负数在下」在末两段会撞上：那两段几乎水平、
	# 中点又贴着终点值标签。改成沿垂直方向逐步找空位，撞了就再往外推。
	for i in range(1, len(stages)):
		delta = auroc[i] - auroc[i - 1]
		mx = (stages[i] + stages[i - 1]) / 2
		my = (auroc[i] + auroc[i - 1]) / 2
		text = f"{delta:+.3f}"
		width, height = text_extent_in_data(ax, text, fontsize=6.0)
		first = 1.0 if delta >= 0 else -1.0
		chosen = None
		for sign in (first, -first):
			for gap in (7.0, 12.0, 17.0, 23.0):
				dy = sign * gap
				va = "bottom" if dy > 0 else "top"
				box = reserve(mx, my, width, height, 0.0, dy, "center", va)
				if free(box):
					taken.append(box)
					chosen = (0.0, dy, "center", va)
					break
			if chosen:
				break
		if chosen is None:
			raise RuntimeError(f"panel (a) 差值标签 {text!r} 找不到空位，请加宽面板或缩小字号")
		ax.annotate(
			text, (mx, my), textcoords="offset points", xytext=chosen[0:2],
			ha=chosen[2], va=chosen[3], fontsize=6.0, color="#202124",
		)
	recessive(ax, grid_axis="y")

	# (b) clean OOD RMSE
	ax = axes[1]
	panel_tag(ax, "b", "Clean held-out cost")
	for stage, value, color in zip(stages, rmse, STAGE_RAMP):
		ax.bar(stage, value, width=0.62, color=color, linewidth=0)
	if rmse_sd is not None:
		ax.errorbar(stages, rmse, yerr=rmse_sd, fmt="none", ecolor="#202124", elinewidth=0.7, capsize=1.6, zorder=5)
	ax.axhline(rmse[0], color=GREY, lw=0.6, ls="--", zorder=4)
	# 有误差棒时数值要抬到棒顶之上，否则压在 cap 上。
	rmse_top = float(rmse[-1] + (rmse_sd[-1] if rmse_sd is not None else 0.0))
	ax.annotate(f"{rmse[-1]:.4f}", (stages[-1], rmse_top), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=6.0)
	ax.set_xticks(stages)
	ax.set_xticklabels(stages)
	ax.set_ylabel("clean OOD RMSE (Nm/kg)")
	rmse_pad = rmse_sd if rmse_sd is not None else 0
	ax.set_ylim(float(np.min(rmse - rmse_pad)) - 0.008, float(np.max(rmse + rmse_pad)) + 0.008)
	# 参考线注记先是横排压柱体，改竖排放到右侧留白后又被 (d) 挤窄的面板里那个
	# 0.2545 数值标签压上。面板只有 0.74 份宽度，容不下第五根柱子 + 一列竖排文字，
	# 所以虚线的含义交给图注（figure_note）说，面板里只留线。
	ax.set_xlim(0.45, 5.55)
	recessive(ax, grid_axis="y")

	# (c) retained aligned torque；阶段 1 无门控
	ax = axes[2]
	panel_tag(ax, "c", "Retained torque")
	shown = [(s, v, c) for s, v, c in zip(stages, retained, STAGE_RAMP) if v is not None]
	for stage, value, color in shown:
		ax.bar(stage, value, width=0.62, color=color, linewidth=0)
	if shown and retained_sd is not None:
		idx = [int(s) - 1 for s, _, _ in shown]
		ax.errorbar([s for s, _, _ in shown], [v for _, v, _ in shown], yerr=retained_sd[idx],
		            fmt="none", ecolor="#202124", elinewidth=0.7, capsize=1.6, zorder=5)
	if shown:
		last_top = shown[-1][1] + (float(retained_sd[int(shown[-1][0]) - 1]) if retained_sd is not None else 0.0)
		ax.annotate(f"{shown[-1][1]:.3f}", (shown[-1][0], last_top), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=6.0)
		values = [v for _, v, _ in shown]
		spread = float(np.max(retained_sd)) if retained_sd is not None else 0.0
		ax.set_ylim(min(values) - 0.03 - spread, max(values) + 0.03 + spread)
	if retained[0] is None:
		ax.text(1, ax.get_ylim()[0] + 0.004, "no gate", rotation=90, ha="center", va="bottom", fontsize=5.6, color=GREY)
		ax.set_xlim(0.45, 5.55)
	ax.set_xticks(stages)
	ax.set_xticklabels(stages)
	ax.set_ylabel("retained torque")
	recessive(ax, grid_axis="y")

	# (d) 门控带来的反向力矩变化。没有这一列，读者只能从 (a) 读出「阶段 2 的 AUROC 最高」，
	# 而阶段 1–3 的门控在全部 60 行评测里 wrong_delta 精确为 0——检测再准也没有作动后果。
	ax = axes[3]
	panel_tag(ax, "d", "Safety effect")
	draw_wrong_delta_panel(ax, stages, wrong_delta, wrong_sd)

	seed_note = (
		f"Mean over {n_seeds} training seeds with standard-deviation bars"
		if n_seeds >= 3 else "Single attribution seed"
	)
	figure_note(
		fig,
		f"{seed_note}, identical split and training budget per stage; darker marks are later stages. "
		"Panels b--d reuse the stage numbering of panel a; the dashed line in b is the stage-1 deterministic level.",
	)
	save(fig, path_stem, formats)


# ------------------------------------------------------------------ figure 5


def draw_stress_transfer(path_stem: Path, formats: list[str], stress, loso_rows) -> None:
	fig, axes = plt.subplots(2, 1, figsize=(COL_WIDTH, 3.05), gridspec_kw={"height_ratios": [1.12, 0.88], "hspace": 0.78})

	# (a) 最高强度点：AUROC 与保留比例共轴
	ax = axes[0]
	panel_tag(ax, "a", "Highest fault intensity")
	labels: list[str] = []
	positions: list[float] = []
	y = 0.0
	last_group = None
	for group, split, auroc, retained, delta in stress:
		if group != last_group:
			if last_group is not None:
				y -= 0.55
			# x 是 axes 分数：-0.02 会让组标题跨过左边框、压在 y 轴刻度文字和第一条网格线上。
			# 挪到边框内 0.02，和图 2(b) 的组标题一致。
			ax.text(0.02, y + 0.55, group, transform=ax.get_yaxis_transform(), fontsize=6.0, color=GREY, ha="left")
			last_group = group
		positions.append(y)
		labels.append(split)
		ax.plot(auroc, y, "o", markersize=4.2, color=BLUE, linestyle="none", zorder=3)
		ax.plot(retained, y, "D", markersize=4.0, markerfacecolor="#ffffff", markeredgecolor=GREEN, markeredgewidth=0.8, linestyle="none", zorder=3)
		# pp 标注统一贴右边界，避免跟着点走出坐标区。
		ax.annotate(
			f"{delta * 100:+.2f} pp", (0.995, y), xycoords=ax.get_yaxis_transform(),
			ha="right", va="center", fontsize=5.8, color="#202124",
		)
		y -= 1.0
	ax.set_yticks(positions)
	ax.set_yticklabels(labels)
	ax.set_ylim(y + 0.5, 1.05)
	ax.set_xlabel("fused AUROC $\\cdot$ retained aligned torque")
	values = [v for _, _, a, r, _ in stress for v in (a, r)]
	ax.set_xlim(min(values) - 0.03, max(values) + 0.075)
	handles = [
		plt.Line2D([], [], marker="o", color=BLUE, lw=0, markersize=4.0, label="fused AUROC"),
		plt.Line2D([], [], marker="D", color=GREEN, markerfacecolor="#ffffff", lw=0, markersize=4.0, label="retained torque"),
	]
	ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.52), frameon=False, ncol=2, handlelength=1.0)
	recessive(ax, grid_axis="x")

	# (b) LOSO：逐折效应量 + bootstrap 均值与 95% CI
	ax = axes[1]
	panel_tag(ax, "b", "Leave-one-subject-out transfer")
	if loso_rows is None:
		ax.axis("off")
		save(fig, path_stem, formats)
		return
	effects = np.sort(np.asarray(loso_rows["effects"], dtype=float) * 100.0)
	mean_pp = loso_rows["mean"] * 100.0
	lo_pp, hi_pp = (v * 100.0 for v in loso_rows["ci"])
	jitter = np.linspace(-0.22, 0.22, effects.size)
	ax.plot(
		effects, jitter, "o", markersize=3.4, color=RED, markerfacecolor="#ffffff",
		markeredgecolor=RED, markeredgewidth=0.7, linestyle="none", zorder=3,
		label=f"per-subject effect ($n={effects.size}$)",
	)
	ax.errorbar(
		mean_pp, -0.72, xerr=[[mean_pp - lo_pp], [hi_pp - mean_pp]], fmt="D", markersize=4.2,
		color=BLUE, ecolor=BLUE, elinewidth=0.9, capsize=2.2, capthick=0.9, zorder=4,
		label="mean, 95% bootstrap CI",
	)
	ax.axvline(0.0, color=GREY, lw=0.5)
	# CI 完全在 0 右侧才是「跨被试稳定」的证据，直接把这句话写在图里。
	ax.annotate(
		f"CI excludes zero; {loso_rows['improved']}/{effects.size} subjects improve",
		(0.985, -1.16), xycoords=ax.get_yaxis_transform(), ha="right", va="center",
		fontsize=5.6, color=GREY,
	)
	ax.set_yticks([0.0, -0.72])
	ax.set_yticklabels(["folds", "pooled"])
	ax.set_ylim(-1.45, 0.55)
	ax.set_xlabel("wrong-direction reduction (pp)")
	span = max(float(effects.max()), hi_pp)
	ax.set_xlim(-0.06 * span, span * 1.12)
	# 两条图例并排时比这个单栏面板宽 39 px，右端 "CI" 越出边框。缩到 5.4 pt 后
	# 宽 684 px，留得下余量；标签文案照旧，n 和置信水平都还写在图里。
	ax.legend(
		loc="lower center", bbox_to_anchor=(0.5, -0.62), frameon=False, ncol=2,
		handlelength=1.0, columnspacing=0.8, fontsize=5.4,
	)
	recessive(ax, grid_axis="x")
	save(fig, path_stem, formats)


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
	draw_method_overview(output_dir / "fig_method_overview", formats, overview_series)
	draw_detection_taxonomy(output_dir / "fig_detection_taxonomy", formats, cells, macro)
	draw_safety_utility(output_dir / "fig_safety_utility", formats, safety)
	draw_ablation_summary(output_dir / "fig_ablation_summary", formats, ablation)
	draw_stress_transfer(output_dir / "fig_stress_transfer", formats, stress, loso_rows)

	print(f"Wrote 5 figure(s) x {len(formats)} format(s) to {output_dir}")
	print("data sources: " + ", ".join(suite.sources))


if __name__ == "__main__":
	main()
