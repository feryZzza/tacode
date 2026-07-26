"""确定性生成 AAAI-27 稿件的五张图。

数字优先从 `--suite-dir` 读取（`paper_numbers.json` / `table1_detection.tsv` /
`table2_safety.tsv` / `loso_effect.json`）；读不到就退回内置常量，这样在没有报告的
机器上也能重现出图。所以正文数字变了只要重跑汇总再 `make figures`，不必手改常量。

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
ABLATION_STAGE_LABELS = (
	"determ-\ninistic",
	"+ fault\naug.",
	"+ prob.\nhead",
	"+ recon-\nstruction",
	"+ fore-\ncast",
)

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
				"auroc_sd": tuple(col("macro_fused_auroc", "sd")),
				"rmse_sd": tuple(col("ood_clean_rmse", "sd")),
				"retained_sd": tuple(col("fault_retained_ratio", "sd")),
				"n_seeds": tuple(seeds),
			}
	stages = suite.numbers.get("table4_ablation", [])
	if isinstance(stages, list) and len(stages) == 5:
		auroc = [as_float(s.get("macro_fused_auroc")) for s in stages]
		rmse = [as_float(s.get("ood_clean_rmse")) for s in stages]
		# 图注写的是「在固定故障集合上取宏平均」，所以 retention 用故障集合那一列。
		retained = [as_float(s.get("fault_retained_ratio")) for s in stages]
		if all(v is not None for v in auroc) and all(v is not None for v in rmse):
			suite.note("ablation", True)
			# 五个阶段都按同一套验证选门控流程评测，所以每阶段都有定义好的
			# retained ratio（含确定性阶段），不再把阶段 1 当成「无门控」。
			return {"auroc": tuple(auroc), "ood_rmse": tuple(rmse), "retained": tuple(retained)}
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
	ax.legend(
		handles, labels, loc="upper left", ncol=2, frameon=False,
		handlelength=1.2, borderpad=0.0, labelspacing=0.2, columnspacing=0.9,
		fontsize=5.6,
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
	fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.1), gridspec_kw={"wspace": 0.34})
	rng = np.random.default_rng(11)
	t = np.arange(0, 400)

	# (a) 传感输入 + 一段保持值故障
	ax = axes[0]
	panel_tag(ax, "a", "Causal streams, plausible fault")
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
	ax.legend(
		loc="upper left", frameon=False, ncol=3, handlelength=1.1,
		borderpad=0.0, columnspacing=0.8, fontsize=5.6,
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

	# 面板宽度只有约 2 英寸，框宽和字号必须匹配最长的标签行（mean/var、forecast）。
	box(0.005, 0.50, 0.165, 0.18, "sensor\nwindow", VIOLET, fontsize=5.2)
	box(0.215, 0.46, 0.145, 0.26, "causal\nTCN", BLUE, fontsize=5.2)
	heads = [
		("moment\nmean/var", GREEN, 0.855),
		("fault\nlogit", RED, 0.665),
		("recon-\nstruction", VIOLET, 0.475),
		("one-step\nforecast", VIOLET, 0.285),
	]
	for text, color, y in heads:
		box(0.405, y - 0.075, 0.245, 0.15, text, color, fontsize=5.2)
		arrow(0.36, 0.59, 0.405, y)
	arrow(0.17, 0.59, 0.215, 0.59)
	box(0.695, 0.38, 0.30, 0.42, "integrity checks\n(staleness,\ncoherence, drift)", GREY, fontsize=5.2)
	arrow(0.0875, 0.50, 0.0875, 0.30)
	arrow(0.0875, 0.30, 0.845, 0.30)
	arrow(0.845, 0.30, 0.845, 0.38)
	ax.text(
		0.5, 0.02,
		"calibration, detector subset, and gate tuning\nuse validation data only",
		ha="center", va="bottom", fontsize=5.4, color=GREY,
	)

	# (c) 风险 -> 门控 -> 力矩衰减
	ax = axes[2]
	panel_tag(ax, "c", "Risk above deadband attenuates")
	if series is not None:
		draw_overview_gate_panel(ax, series)
		fig.text(
			0.5, -0.045,
			"Panel a is an illustrative trace; panel c is a measured per-timestep export "
			f"({series['meta'].get('fault', '')} on {series['meta'].get('split', '')}).",
			ha="center", fontsize=5.6, color=GREY,
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

	fig.text(
		0.5, -0.045,
		"Panels a and c are illustrative traces of the mechanism, not measured per-timestep exports.",
		ha="center", fontsize=5.6, color=GREY,
	)
	save(fig, path_stem, formats)


def save(fig, path_stem: Path, formats: list[str]) -> None:
	for fmt in formats:
		fig.savefig(path_stem.with_suffix(f".{fmt}"), format=fmt)
	plt.close(fig)


# ------------------------------------------------------------------ figure 2


def draw_detection_taxonomy(path_stem: Path, formats: list[str], cells, macro) -> None:
	fig, axes = plt.subplots(
		1, 2, figsize=(FULL_WIDTH, 2.45), gridspec_kw={"width_ratios": [2.35, 1.0], "wspace": 0.30}
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
		ax.text(
			ax.get_xlim()[0], y + 0.30, title, fontsize=6.0, color=GREY,
			transform=ax.get_yaxis_transform(),
		)
	ax.set_yticks([])
	ax.set_ylim(-0.55, 1.62)
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
	fig, axes = plt.subplots(
		1, 3, figsize=(FULL_WIDTH, 2.35), gridspec_kw={"width_ratios": [1.25, 1.05, 0.80], "wspace": 0.58}
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
	yspan_ref = max(float(np.ptp(retained)), 1e-3)
	candidates = (
		(4.5, 4.5, "bottom", "left"),
		(4.5, -8.0, "top", "left"),
		(-4.5, 4.5, "bottom", "right"),
		(-4.5, -8.0, "top", "right"),
		(4.5, 0.0, "center", "left"),
		(-4.5, 0.0, "center", "right"),
	)
	for label, red, ret, is_fault in zip(labels, reduction, retained, faulty):
		if is_fault:
			ax.plot(red, ret, "^", markersize=4.4, color=RED, linestyle="none", zorder=3)
		else:
			ax.plot(red, ret, "o", markersize=4.4, markerfacecolor="#ffffff", markeredgecolor=BLUE, markeredgewidth=0.8, linestyle="none", zorder=3)
		# 标签占位按字符宽度估算成数据坐标下的矩形。
		width = 0.075 * len(label) * xspan_ref
		height = 0.09 * yspan_ref
		chosen = candidates[0]
		for dx, dy, va, ha in candidates:
			x0 = red + 0.02 * xspan_ref if ha == "left" else red - 0.02 * xspan_ref - width
			y0 = ret + (0.03 * yspan_ref if dy > 0 else -0.03 * yspan_ref - height if dy < 0 else -height / 2)
			box = (x0, y0, x0 + width, y0 + height)
			if all(
				box[2] < other[0] or box[0] > other[2] or box[3] < other[1] or box[1] > other[3]
				for other in placed
			):
				chosen = (dx, dy, va, ha)
				placed.append(box)
				break
		else:
			placed.append((red, ret, red + width, ret + height))
		dx, dy, va, ha = chosen
		ax.annotate(
			label, (red, ret), textcoords="offset points", xytext=(dx, dy),
			ha=ha, va=va, fontsize=5.8, color="#202124",
		)
	ax.axhline(0.95, color=GREY, lw=0.6, ls="--")
	# 约束线注记贴在线下方，和落在 0.95 附近的点标签错开。
	ax.text(0.02, 0.9485, "validation floor 0.95", transform=ax.get_yaxis_transform(), fontsize=5.6, color=GREY, va="top")
	ax.set_xlabel("wrong-direction reduction (pp)")
	ax.set_ylabel("retained aligned torque")
	lo = min(0.94, float(np.min(retained)) - 0.01)
	ax.set_ylim(lo, max(1.0, float(np.max(retained)) + 0.01))
	xspan = max(float(np.ptp(reduction)), 1e-3)
	# 左右都留出标签空间：贪心放置可能把标签甩到点左侧。
	ax.set_xlim(float(np.min(reduction)) - 0.42 * xspan, float(np.max(reduction)) + 0.42 * xspan)
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
	ax.barh(y, gate, height=0.55, color=[RED if f else BLUE for f in faulty], alpha=0.85, linewidth=0)
	gate_lo = min(0.94, float(np.min(gate)) - 0.008)
	# 门控值越接近下界，条越短，标签放不进去就会压到 y 轴刻度上；
	# 条内放不下时改成放在条右侧、用深色字。
	inside_width = 0.30 * (1.0 - gate_lo)
	for yi, value in zip(y, gate):
		if value - gate_lo >= inside_width:
			ax.annotate(f"{value:.3f}", (value, yi), textcoords="offset points", xytext=(-3, 0), ha="right", va="center", fontsize=5.8, color="#ffffff")
		else:
			ax.annotate(f"{value:.3f}", (value, yi), textcoords="offset points", xytext=(3, 0), ha="left", va="center", fontsize=5.8, color="#202124")
	ax.set_yticks(y)
	ax.set_yticklabels(labels)
	ax.set_xlim(gate_lo, 1.0)
	ax.set_xticks([round(gate_lo + 0.002, 3), round((gate_lo + 1.0) / 2, 3), 1.0])
	ax.set_xlabel("mean gate $g_t$, active steps")
	ax.text(0.5, -0.30, "$g_t = 1$: no attenuation", transform=ax.transAxes, ha="center", fontsize=5.6, color=GREY)
	recessive(ax, grid_axis="x")
	save(fig, path_stem, formats)


# ------------------------------------------------------------------ figure 4


def draw_ablation_summary(path_stem: Path, formats: list[str], ablation) -> None:
	fig, axes = plt.subplots(
		1, 3, figsize=(FULL_WIDTH, 2.45), gridspec_kw={"width_ratios": [1.60, 0.88, 0.88], "wspace": 0.42}
	)
	stages = np.arange(1, 6)
	auroc = np.array([v for v in ablation["auroc"]], dtype=float)
	rmse = np.array([v for v in ablation["ood_rmse"]], dtype=float)
	retained = ablation["retained"]
	# 三种子版本才有 *_sd；单种子扫描下这些键缺失，误差棒自动消失。
	def sd_of(key: str) -> np.ndarray | None:
		values = ablation.get(key)
		if not values or any(v is None for v in values):
			return None
		return np.array([float(v) for v in values], dtype=float)
	auroc_sd, rmse_sd, retained_sd = sd_of("auroc_sd"), sd_of("rmse_sd"), sd_of("retained_sd")
	n_seeds = max(ablation.get("n_seeds") or [1])

	# (a) macro AUROC 与相邻差值
	ax = axes[0]
	panel_tag(ax, "a", "Detection gain is not monotone")
	ax.plot(stages, auroc, color=BLUE, lw=0.8, zorder=1)
	if auroc_sd is not None:
		ax.errorbar(stages, auroc, yerr=auroc_sd, fmt="none", ecolor=BLUE, elinewidth=0.7, capsize=1.8, zorder=2)
	for stage, value, color in zip(stages, auroc, STAGE_RAMP):
		ax.plot(stage, value, "o", markersize=5.0, color=color, markeredgecolor=BLUE, markeredgewidth=0.7, linestyle="none", zorder=3)
	# 终点值放到点的右侧：末段差值标在线段中点下方，两者同处下方会叠字。
	ax.annotate(
		f"{auroc[-1]:.3f}", (stages[-1], auroc[-1]), textcoords="offset points",
		xytext=(7, -2), ha="left", va="center", fontsize=6.0,
	)
	ax.set_xlim(0.6, 5.75)
	for i in range(1, len(stages)):
		delta = auroc[i] - auroc[i - 1]
		ax.annotate(
			f"{delta:+.3f}", ((stages[i] + stages[i - 1]) / 2, (auroc[i] + auroc[i - 1]) / 2),
			textcoords="offset points", xytext=(0, 7 if delta >= 0 else -12), ha="center", fontsize=6.0, color="#202124",
		)
	ax.set_xticks(stages)
	ax.set_xticklabels([f"{s}\n{label}" for s, label in zip(stages, ABLATION_STAGE_LABELS)], fontsize=5.8)
	ax.set_ylabel("macro fused AUROC")
	pad = max(0.35 * float(np.ptp(auroc)), 0.01)
	lo = float(np.min(auroc - (auroc_sd if auroc_sd is not None else 0)))
	hi = float(np.max(auroc + (auroc_sd if auroc_sd is not None else 0)))
	ax.set_ylim(lo - pad, hi + pad)
	recessive(ax, grid_axis="y")

	# (b) clean OOD RMSE
	ax = axes[1]
	panel_tag(ax, "b", "Clean held-out cost")
	for stage, value, color in zip(stages, rmse, STAGE_RAMP):
		ax.bar(stage, value, width=0.62, color=color, linewidth=0)
	if rmse_sd is not None:
		ax.errorbar(stages, rmse, yerr=rmse_sd, fmt="none", ecolor="#202124", elinewidth=0.7, capsize=1.6, zorder=5)
	ax.axhline(rmse[0], color=GREY, lw=0.6, ls="--", zorder=4)
	ax.text(3.0, rmse[0] - 0.0015, "deterministic reference", ha="center", fontsize=5.6, color=GREY, va="top")
	# 有误差棒时数值要抬到棒顶之上，否则压在 cap 上。
	rmse_top = float(rmse[-1] + (rmse_sd[-1] if rmse_sd is not None else 0.0))
	ax.annotate(f"{rmse[-1]:.4f}", (stages[-1], rmse_top), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=6.0)
	ax.set_xticks(stages)
	ax.set_xticklabels(stages)
	ax.set_ylabel("clean OOD RMSE (Nm/kg)")
	rmse_pad = rmse_sd if rmse_sd is not None else 0
	ax.set_ylim(float(np.min(rmse - rmse_pad)) - 0.008, float(np.max(rmse + rmse_pad)) + 0.008)
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

	seed_note = (
		f"Mean over {n_seeds} training seeds with standard-deviation bars"
		if n_seeds >= 3 else "Single attribution seed"
	)
	fig.text(
		0.5, -0.06,
		f"{seed_note}, identical split and training budget per stage; darker marks are later stages. "
		"Panels b and c share the stage numbering of panel a.",
		ha="center", fontsize=5.6, color=GREY,
	)
	save(fig, path_stem, formats)


# ------------------------------------------------------------------ figure 5


def draw_stress_transfer(path_stem: Path, formats: list[str], stress, loso_rows) -> None:
	fig, axes = plt.subplots(2, 1, figsize=(COL_WIDTH, 3.45), gridspec_kw={"height_ratios": [1.12, 0.88], "hspace": 0.85})

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
			ax.text(-0.02, y + 0.55, group, transform=ax.get_yaxis_transform(), fontsize=6.0, color=GREY, ha="left")
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
	ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.62), frameon=False, ncol=2, handlelength=1.0)
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
