#!/usr/bin/env python3
"""EXPERIMENTS_TODO §6：leave-one-component-out 反向消融的汇总与 Pareto 结论。

§6 的要求是「7 格 × ≥3 seed，结论基于 Pareto 而非单指标」。这里做三件事：

1. 把 `reports/v2_reverse_<mode>_seed<seed>/` 与已有的 full-stack 主 run 抽平成一张长表
   （mode × seed × task_split × fault_operator × 指标）。
2. 按 (mode, task_split, fault_operator) 跨 seed 汇总，同时报标准差与符号一致性——
   3 个 seed 里有 1 个反向时，「去掉这个组件更差」不能当结论。
3. 在 (clean 侧代价, 故障侧 $X_{\\mathrm{opp}}$) 平面上做 Pareto 判定：只有在两个轴上
   都不被任何其他 mode 支配的配置才算「被保留的组件」。单看 $X_{\\mathrm{opp}}$ 会把
   「靠多关门换来的下降」误判成组件贡献。

用法：
    python scripts/aggregate_reverse_ablation.py --reports-dir reports \
        --output-dir reports/aaai27_reverse_ablation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

#: 汇总的指标：X_opp（安全）、retention/RMSE（clean 代价）、torque rate（平滑）、关断统计。
#:
#: 每项给一串候选键，按顺序取第一个存在的。早于术语重命名的归档 run 里，$X_{opp}$ 叫
#: `wrong_energy`、torque rate 叫 `mean_abs_jerk`；`reliability/metrics.py` 现在同时写
#: 新旧两个键（同一个数值，见 metrics.py:645-654），所以旧 run 可以按新名字读回来，
#: 不用重跑。顺序是「新名优先」，这样新 run 不会误命中别名。
METRIC_KEYS = {
	"x_opp": ("gated_wrong_torque_product_integral", "gated_wrong_energy"),
	"ungated_x_opp": ("baseline_wrong_torque_product_integral", "baseline_wrong_energy"),
	"aligned_command_retention": ("gate_retention_ratio",),
	"aligned_command_adequacy": ("gated_aligned_command_adequacy", "gated_retained_aligned_torque"),
	"ungated_aligned_command_adequacy": (
		"baseline_aligned_command_adequacy", "baseline_retained_aligned_torque",
	),
	"capped_overlap_retention": ("capped_overlap_retention_ratio",),
	"capped_overlap_adequacy": ("gated_capped_overlap_adequacy", "gated_capped_overlap"),
	"ungated_capped_overlap_adequacy": (
		"baseline_capped_overlap_adequacy", "baseline_capped_overlap",
	),
	"tracking_rmse": ("gated_tracking_rmse",),
	"mean_abs_torque_rate": ("gated_mean_abs_torque_rate", "gated_mean_abs_jerk"),
	"full_shutdown_fraction": ("full_shutdown_fraction",),
	"mean_gate": ("mean_gate",),
	"rmse": ("rmse",),
	"r2": ("r2",),
	"fault_auroc": ("fault_auroc",),
	"coverage_ece": ("coverage_ece",),
}

#: mode 名 -> 去掉了哪个组件（用于论文表格的可读标签）。
MODE_LABELS = {
	"full": "full stack (all components)",
	"det_aug_recon_fc": "− probabilistic head (deterministic)",
	"prob_aug_fc": "− reconstruction head",
	"prob_aug_recon_fc_nofault": "− fault-supervision head",
	"prob_aug_recon": "− forecast head",
	"prob_recon_fc": "− fault augmentation",
	"det_noaug": "− probabilistic head & augmentation",
	"minus_staleness": "− staleness gate channel (eval-side)",
	"selected_subset_gate": "full stack, validation-selected gate subset",
	"all_gate_channels": "full stack, all six gate channels",
}

RUN_PATTERN = re.compile(r"^v2_reverse_(?P<mode>.+)_seed(?P<seed>\d+)$")

#: 统一门控协议的重评产物（run_reverse_cells_reeval.sh）。归档的 v2_reverse_* eval 用的
#: 门控选择协议彼此不同：det_aug_recon_fc 三个 seed 与 prob_aug_fc seed7 用旧目标
#: dimensionless_relative_wrong_v2（无速率网格），prob_aug_fc seed13/23 与 nofault 用
#: x_opp 但只有 6 个候选。主重选和另外几格是 216 候选带速率限幅。混在一张 Pareto 表里
#: 会把「组件差异」和「门控搜索空间差异」叠在一起。优先读重评，缺失才回落到归档 run。
REEVAL_PATTERN = re.compile(r"^(?P<mode>.+)_seed(?P<seed>\d+)$")

#: §6 的七格里，「minus forecasting」不需要新训练：ordered ablation 的
#: `prob_aug_recon`（无 forecast 头）就是这一格，三个 seed 都已训好。复用而不重训。
#: 但那批老 eval 没有算 x_opp，Pareto 需要它，所以 run_reverse_forecast_cell_reeval.sh
#: 用同一批 checkpoint、同一 split、x_opp 目标重跑了一遍（纯 eval，不重训）。优先读
#: 重评产物；只有缺失时才回落到老 ablation 报告。
REUSED_CELLS = {
	"prob_aug_recon": (
		"aaai27_reverse_forecast/seed*",
		"v2_ablation_prob_aug_recon_seed*",
	)
}

#: 另两格（minus staleness、validation-selected-subset）是纯评测侧的门控通道差异：
#: 同一批 full-stack checkpoint、同一批窗口，只改门控读哪几路。它们由
#: `run_gate_baselines.py` 的 `loo_minus_staleness` / `selected_subset` /
#: `all_gate_channels` 行提供，在这里映射进同一张表，口径与训练侧的格子一致。
GATE_CELLS = {
	"loo_minus_staleness": "minus_staleness",
	"selected_subset": "selected_subset_gate",
	"all_gate_channels": "all_gate_channels",
}

#: gate-baselines TSV 的列名 -> 本表的指标名。缺的（rmse/r2/AUROC/ECE）留空：
#: 那几项是模型级指标，与门控通道无关，取 full-stack 的值会造成「同一格里
#: 一半数字来自另一个实验」的错觉。
GATE_TSV_COLUMNS = {
	"x_opp": "x_opp",
	"ungated_x_opp": "ungated_x_opp",
	"aligned_command_retention": "aligned_command_retention",
	"aligned_command_adequacy": "aligned_command_adequacy",
	"ungated_aligned_command_adequacy": "ungated_aligned_command_adequacy",
	"capped_overlap_retention": "capped_overlap_retention",
	"capped_overlap_adequacy": "capped_overlap_adequacy",
	"ungated_capped_overlap_adequacy": "ungated_capped_overlap_adequacy",
	"tracking_rmse": "tracking_rmse",
	"mean_abs_torque_rate": "mean_abs_torque_rate",
	"full_shutdown_fraction": "full_shutdown_fraction",
	"mean_gate": "mean_gate",
}


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--reports-dir", default="reports")
	p.add_argument("--output-dir", default="reports/aaai27_reverse_ablation")
	p.add_argument("--full-stack-glob", default="v2_fc_main_seed*",
		help="full-stack 参照 run（第 7 格）的目录名模式")
	p.add_argument("--clean-cost-metric", default="aligned_command_retention",
		choices=["aligned_command_retention", "tracking_rmse", "capped_overlap_retention"])
	p.add_argument("--pareto-split", default="test_id")
	p.add_argument("--gate-baselines-dir", default="aaai27_gate_baselines",
		help="§2 的产出目录，供 minus-staleness / selected-subset 两格（纯评测侧）")
	p.add_argument("--reeval-dir", default="aaai27_reverse_cells",
		help="统一门控协议的训练侧格子重评目录（优先于归档的 v2_reverse_*）")
	return p.parse_args()


def write_tsv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	if not rows:
		path.write_text("", encoding="utf-8")
		return
	columns: List[str] = []
	for row in rows:
		for key in row:
			if key not in columns:
				columns.append(key)
	lines = ["\t".join(columns)]
	for row in rows:
		lines.append("\t".join(_fmt(row.get(col)) for col in columns))
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: object) -> str:
	if value is None:
		return ""
	if isinstance(value, bool):
		return "1" if value else "0"
	if isinstance(value, float):
		return "nan" if value != value else f"{value:.6g}"
	return str(value)


def _clean(value: object) -> Optional[float]:
	"""转成有限浮点，否则 None。JSON 给的是数值，TSV 给的是字符串，两边都要接。

	bool 显式排除：`True` 会被 `float()` 变成 1.0，把布尔标志混进数值列。
	"""
	if isinstance(value, bool) or value is None:
		return None
	if not isinstance(value, (int, float)):
		try:
			value = float(str(value).strip())
		except (TypeError, ValueError):
			return None
	value = float(value)
	return None if math.isnan(value) or math.isinf(value) else value


def gate_protocol(directory: Path) -> Optional[Dict[str, object]]:
	"""读一个 run 的门控选择协议指纹：目标函数 + 候选网格规模。

	同一张 Pareto 表里的格子必须共享这个指纹，否则「组件差异」会和「门控搜索空间
	差异」混在一起——旧归档 run 只搜 6 个 softness，新重选搜 216 个含速率限幅的候选。
	"""
	report_path = directory / "reliability_report.json"
	if not report_path.is_file():
		return None
	try:
		report = json.loads(report_path.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError):
		return None
	policy = report.get("results", {}).get("_gate_policy", {})
	if not isinstance(policy, dict):
		return None
	return {
		"selection_objective": policy.get("selection_objective"),
		"n_candidates": len(policy.get("candidates") or []),
	}


def heldout_tasks(directory: Path) -> Optional[frozenset]:
	"""读取一个 run 的留出任务集合；读不到就返回 None。"""
	report_path = directory / "reliability_report.json"
	if not report_path.is_file():
		return None
	try:
		report = json.loads(report_path.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError):
		return None
	raw = report.get("args", {}).get("heldout_tasks")
	if isinstance(raw, str):
		return frozenset(item.strip() for item in raw.split(",") if item.strip())
	if isinstance(raw, (list, tuple)):
		return frozenset(str(item).strip() for item in raw if str(item).strip())
	return None


def extract_rows(directory: Path, mode: str, seed: str) -> List[Dict[str, object]]:
	report_path = directory / "reliability_report.json"
	if not report_path.is_file():
		return []
	report = json.loads(report_path.read_text(encoding="utf-8"))
	policy = report.get("results", {}).get("_gate_policy", {})
	rows: List[Dict[str, object]] = []
	for key, payload in report.get("results", {}).items():
		if key.startswith("_") or not isinstance(payload, dict) or "/" not in key:
			continue
		split, _, fault = key.partition("/")
		if split not in ("test_id", "test_ood", "val"):
			continue
		row: Dict[str, object] = {
			"mode": mode,
			"mode_label": MODE_LABELS.get(mode, mode),
			"seed": seed,
			"run": directory.name,
			"task_split": split,
			"fault_operator": fault,
			"gate_softness": policy.get("softness"),
			"gate_deadband": policy.get("deadband"),
		}
		for out_key, candidates in METRIC_KEYS.items():
			row[out_key] = _first_present(payload, candidates)
		if row.get("aligned_command_retention") is None:
			gated = _clean(row.get("aligned_command_adequacy"))
			ungated = _clean(row.get("ungated_aligned_command_adequacy"))
			row["aligned_command_retention"] = (
				gated / ungated
				if gated is not None and ungated is not None and abs(ungated) > 1e-12
				else None
			)
		if row.get("capped_overlap_retention") is None:
			gated = _clean(row.get("capped_overlap_adequacy"))
			ungated = _clean(row.get("ungated_capped_overlap_adequacy"))
			row["capped_overlap_retention"] = (
				gated / ungated
				if gated is not None and ungated is not None and abs(ungated) > 1e-12
				else None
			)
		rows.append(row)
	return rows


def _first_present(payload: Dict[str, object], candidates: Sequence[str]) -> Optional[float]:
	"""按候选键顺序取第一个可用数值（新键优先，旧归档 run 回落到别名）。"""
	for key in candidates:
		value = _clean(payload.get(key))
		if value is not None:
			return value
	return None


def extract_gate_rows(baselines_dir: Path) -> List[Dict[str, object]]:
	"""把 gate-baselines 的三行门控变体读成 §6 的评测侧格子。

	只带得动的指标进来（见 `GATE_TSV_COLUMNS`）；模型级指标（RMSE/R²/AUROC/ECE）留空，
	因为这三格用的是同一个 full-stack checkpoint，抄过来会把「组件差异」和
	「模型差异」混成一列。
	"""
	rows: List[Dict[str, object]] = []
	for path in sorted(baselines_dir.glob("gate_baselines_*.tsv")):
		with path.open(encoding="utf-8") as handle:
			for record in csv.DictReader(handle, delimiter="\t"):
				mode = GATE_CELLS.get(str(record.get("gate")))
				if mode is None:
					continue
				split = str(record.get("task_split"))
				if split not in ("test_id", "test_ood", "val"):
					continue
				row: Dict[str, object] = {
					"mode": mode,
					"mode_label": MODE_LABELS.get(mode, mode),
					"seed": str(record.get("seed")),
					"run": f"{record.get('run')}::{record.get('gate')}",
					"task_split": split,
					"fault_operator": str(record.get("fault_operator")),
					"gate_softness": _clean(record.get("gate_softness")),
					"gate_deadband": _clean(record.get("gate_deadband")),
					"source": "gate_baselines",
				}
				for out_key in METRIC_KEYS:
					column = GATE_TSV_COLUMNS.get(out_key)
					row[out_key] = _clean(record.get(column)) if column else None
				rows.append(row)
	return rows


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
	buckets: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
	for row in rows:
		buckets[(str(row["mode"]), str(row["task_split"]), str(row["fault_operator"]))].append(row)
	out: List[Dict[str, object]] = []
	for (mode, split, fault), group in sorted(buckets.items()):
		record: Dict[str, object] = {
			"mode": mode,
			"mode_label": MODE_LABELS.get(mode, mode),
			"task_split": split,
			"fault_operator": fault,
			"n_seeds": len({str(r["seed"]) for r in group}),
			"seeds": ",".join(sorted({str(r["seed"]) for r in group})),
		}
		for metric in METRIC_KEYS:
			values = [v for v in (_clean(r.get(metric)) for r in group) if v is not None]
			if not values:
				continue
			record[f"{metric}_mean"] = statistics.fmean(values)
			record[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
			record[f"{metric}_min"] = min(values)
			record[f"{metric}_max"] = max(values)
		out.append(record)
	return out


def pareto(
	summary: Sequence[Dict[str, object]],
	split: str,
	cost_metric: str,
) -> List[Dict[str, object]]:
	"""在 (故障侧 X_opp ↓, clean 侧代价) 平面上判 Pareto 前沿。

	故障侧取所有非 clean 算子的均值，clean 侧取 clean 那一行。retention/overlap 是
	「越大越好」，RMSE 是「越小越好」，两种方向都要处理，否则 Pareto 判定会反。
	"""
	higher_is_better = cost_metric != "tracking_rmse"
	per_mode: Dict[str, Dict[str, object]] = {}
	fault_by_mode: Dict[str, Dict[str, float]] = defaultdict(dict)
	for row in summary:
		if row["task_split"] != split:
			continue
		mode = str(row["mode"])
		if row["fault_operator"] == "clean":
			per_mode.setdefault(mode, {})["clean_cost"] = _clean(row.get(f"{cost_metric}_mean"))
			per_mode[mode]["clean_x_opp"] = _clean(row.get("x_opp_mean"))
			per_mode[mode]["mode_label"] = row["mode_label"]
			per_mode[mode]["n_seeds"] = row["n_seeds"]
		else:
			value = _clean(row.get("x_opp_mean"))
			if value is not None:
				fault_by_mode[mode][str(row["fault_operator"])] = value

	# 各 mode 的 fault-scenario 列表不一定相同（ordered ablation 与 gate-baselines 用了
	# 不同的算子集）。若直接对「各自有的算子」求均值，Pareto 比较的其实是不同故障混合。
	# 取所有 mode 的交集，并把用到的算子写进每一行，让口径可核。
	shared: Optional[set] = None
	for mode in per_mode:
		operators = set(fault_by_mode.get(mode, {}))
		shared = operators if shared is None else (shared & operators)
	common = sorted(shared or set())

	points: List[Dict[str, object]] = []
	for mode, record in per_mode.items():
		available = fault_by_mode.get(mode, {})
		values = [available[op] for op in common if op in available]
		if not values or record.get("clean_cost") is None:
			continue
		points.append(
			{
				"mode": mode,
				"mode_label": record.get("mode_label", mode),
				"n_seeds": record.get("n_seeds"),
				"task_split": split,
				"clean_cost_metric": cost_metric,
				"clean_cost": record["clean_cost"],
				"clean_x_opp": record.get("clean_x_opp"),
				"fault_x_opp_mean": statistics.fmean(values),
				"fault_x_opp_worst": max(values),
				"n_fault_operators": len(values),
				"fault_operators": ",".join(common),
				"n_operators_available": len(available),
			}
		)

	for point in points:
		dominated_by: List[str] = []
		for other in points:
			if other["mode"] == point["mode"]:
				continue
			safer = other["fault_x_opp_mean"] <= point["fault_x_opp_mean"]
			cheaper = (
				other["clean_cost"] >= point["clean_cost"]
				if higher_is_better
				else other["clean_cost"] <= point["clean_cost"]
			)
			strictly = (
				other["fault_x_opp_mean"] < point["fault_x_opp_mean"]
				or (other["clean_cost"] != point["clean_cost"])
			)
			if safer and cheaper and strictly:
				dominated_by.append(str(other["mode"]))
		point["dominated_by"] = ",".join(sorted(dominated_by))
		point["on_pareto_front"] = not dominated_by
	points.sort(key=lambda r: r["fault_x_opp_mean"])
	return points


def main() -> None:
	cli = parse_args()
	reports = Path(cli.reports_dir)
	rows: List[Dict[str, object]] = []
	incomplete: List[str] = []
	split_mismatch: List[str] = []

	# split-lock：训练侧格子的留出任务必须和 full-stack 参照完全一致。留出任务不同会
	# 改变 test_id/test_ood 的任务成员，那样的 Pareto 比较是在两套不同的评测集上做的，
	# 必须剔除而不是静默混入（否则错误批次会一路进正文）。
	reference_split: Optional[frozenset] = None
	for directory in sorted(reports.glob(cli.full_stack_glob)):
		reference_split = heldout_tasks(directory)
		if reference_split:
			break

	gate_protocols: Dict[str, Dict[str, object]] = {}

	def accept(directory: Path, mode: str, seed: str) -> List[Dict[str, object]]:
		if reference_split is not None:
			actual = heldout_tasks(directory)
			if actual is not None and actual != reference_split:
				split_mismatch.append(
					f"{directory.name}: heldout_tasks={sorted(actual)} != "
					f"{sorted(reference_split)}"
				)
				return []
		protocol = gate_protocol(directory)
		if protocol is not None:
			gate_protocols[f"{mode}/seed{seed}"] = protocol
		return extract_rows(directory, mode, seed)

	# 训练侧格子：先收统一协议的重评，再用归档 run 补它没覆盖到的 (mode, seed)。
	# 同一格里不允许混两种门控协议，所以按 mode 整体取舍，而不是按 seed 逐个回落。
	reeval_root = reports / cli.reeval_dir
	reeval_rows: Dict[str, List[Dict[str, object]]] = defaultdict(list)
	for directory in sorted(reeval_root.glob("*_seed*")):
		match = REEVAL_PATTERN.match(directory.name)
		if not match:
			continue
		extracted = accept(directory, match.group("mode"), match.group("seed"))
		if extracted:
			reeval_rows[match.group("mode")].extend(extracted)

	for directory in sorted(reports.glob("v2_reverse_*")):
		match = RUN_PATTERN.match(directory.name)
		if not match:
			continue
		mode = match.group("mode")
		if mode in reeval_rows:
			continue
		extracted = accept(directory, mode, match.group("seed"))
		if extracted:
			rows.extend(extracted)
		elif directory.name not in " ".join(split_mismatch):
			incomplete.append(directory.name)
	for mode_rows in reeval_rows.values():
		rows.extend(mode_rows)

	# 复用已训好的 checkpoint 补「minus forecasting」这一格：按优先级取第一个能产出
	# 完整 x_opp 的来源，避免同一格混入两批不同口径的 eval。
	for mode, patterns in REUSED_CELLS.items():
		for pattern in patterns:
			candidate: List[Dict[str, object]] = []
			for directory in sorted(reports.glob(pattern)):
				seed_match = re.search(r"seed(\d+)$", directory.name)
				if not seed_match:
					continue
				candidate.extend(accept(directory, mode, seed_match.group(1)))
			# 这一格必须带 x_opp，否则 Pareto 的算子交集会归零、前沿被清空。
			if candidate and any(
				_clean(row.get("x_opp")) is not None
				for row in candidate
				if row.get("fault_operator") != "clean"
			):
				rows.extend(candidate)
				break

	# 评测侧的三格：minus staleness / validation-selected subset / all channels。
	gate_rows = extract_gate_rows(reports / cli.gate_baselines_dir)
	if gate_rows:
		rows.extend(gate_rows)
	else:
		incomplete.append(f"{cli.gate_baselines_dir}: 无 gate_baselines_*.tsv，缺 3 个评测侧格子")
		# 正式汇总由 all_gate_channels 提供 full-stack 参照。只有缺少统一的
		# gate-baselines 重评时才回退到主 report，避免重复一格并混入另一套故障抽样。
		for directory in sorted(reports.glob(cli.full_stack_glob)):
			seed_match = re.search(r"seed(\d+)$", directory.name)
			if not seed_match:
				continue
			rows.extend(extract_rows(directory, "full", seed_match.group(1)))

	summary = summarize(rows)
	front = pareto(summary, cli.pareto_split, cli.clean_cost_metric)

	out_dir = Path(cli.output_dir)
	write_tsv(out_dir / "reverse_ablation_runs.tsv", rows)
	write_tsv(out_dir / "aggregate.tsv", summary)
	write_tsv(out_dir / "pareto.tsv", front)

	# §9 的 run_manifest：每格来自哪个 run 目录、哪个 seed、哪种来源（新训练／复用
	# checkpoint／纯评测侧门控变体）。§6 的七格来源不同，混在一张表里必须能追回去。
	seen: Dict[Tuple[str, str], Dict[str, object]] = {}
	for row in rows:
		key = (str(row["mode"]), str(row["seed"]))
		record = seen.setdefault(
			key,
			{
				"mode": row["mode"],
				"mode_label": row["mode_label"],
				"seed": row["seed"],
				"run": row["run"],
				"source": row.get("source", "trained_report"),
				"n_rows": 0,
				"task_splits": set(),
			},
		)
		record["n_rows"] = int(record["n_rows"]) + 1
		record["task_splits"].add(str(row["task_split"]))
	manifest = [
		{**record, "task_splits": ",".join(sorted(record["task_splits"]))}
		for _, record in sorted(seen.items())
	]
	write_tsv(out_dir / "run_manifest_reverse.tsv", manifest)

	modes = sorted({str(r["mode"]) for r in rows})
	seeds_per_mode = {
		mode: sorted({str(r["seed"]) for r in rows if r["mode"] == mode}) for mode in modes
	}
	(out_dir / "provenance_reverse.json").write_text(
		json.dumps(
			{
				"reports_dir": str(reports),
				"modes": modes,
				"seeds_per_mode": seeds_per_mode,
				"n_cells": len(modes),
				"cells_required": 7,
				"incomplete_runs": incomplete,
				"reference_heldout_tasks": sorted(reference_split) if reference_split else [],
				"excluded_split_mismatch": split_mismatch,
				"gate_protocols": gate_protocols,
				"gate_protocol_uniform": len(
					{
						(p.get("selection_objective"), p.get("n_candidates"))
						for p in gate_protocols.values()
					}
				)
				<= 1,
				"clean_cost_metric": cli.clean_cost_metric,
				"pareto_split": cli.pareto_split,
				"pareto_rule": "dominated iff another mode has <= fault X_opp and no worse clean cost, with a strict improvement",
				"pareto_fault_operators": front[0].get("fault_operators") if front else "",
				"cell_sources": {
					"trained_new": sorted(
						{str(r["mode"]) for r in rows if str(r.get("run", "")).startswith("v2_reverse_")}
					),
					"reused_checkpoints": sorted(REUSED_CELLS),
					"eval_side_gate_variants": sorted(set(GATE_CELLS.values())),
					"full_stack_reference": cli.full_stack_glob,
				},
				"cli": vars(cli),
			},
			indent=2,
			ensure_ascii=False,
		),
		encoding="utf-8",
	)

	print(f"{len(modes)} 格 / 需要 7：{', '.join(modes)}")
	for mode, seeds in seeds_per_mode.items():
		flag = "" if len(seeds) >= 3 else "  <-- seed 不足 3"
		print(f"  {mode:34s} seeds={','.join(seeds)}{flag}")
	if incomplete:
		print(f"未完成（无 reliability_report.json）：{', '.join(incomplete)}")
	if split_mismatch:
		print(f"\n因 split 不一致被剔除（留出任务={sorted(reference_split or [])}）：")
		for item in split_mismatch:
			print(f"  - {item}")
	fingerprints = {
		(p.get("selection_objective"), p.get("n_candidates")) for p in gate_protocols.values()
	}
	if len(fingerprints) > 1:
		print("\n门控协议不一致（同一张 Pareto 表里混了不同搜索空间）：")
		for name, protocol in sorted(gate_protocols.items()):
			print(
				f"  {name:44s} obj={protocol.get('selection_objective')} "
				f"n_candidates={protocol.get('n_candidates')}"
			)
	print(f"\nPareto ({cli.pareto_split}, cost={cli.clean_cost_metric}):")
	for point in front:
		mark = "*" if point["on_pareto_front"] else " "
		print(
			f" {mark} {point['mode']:34s} fault_x_opp={point['fault_x_opp_mean']:.5g} "
			f"clean_cost={point['clean_cost']:.5g} seeds={point['n_seeds']}"
		)


if __name__ == "__main__":
	main()
