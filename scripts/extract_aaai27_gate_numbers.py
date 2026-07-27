#!/usr/bin/env python3
"""把 §2–§6 的汇总表压成稿件直接引用的数字（`aaai27_gate_numbers.json`）。

正文里能写的句子是有限的几句，但支撑它们的表有上千行。这个脚本只做一件事：把
「论文会引用的那些数」显式挑出来并落成一个扁平 JSON，让 `check_paper_numbers.py`
式的逐位核对可以覆盖新章节，而不是靠人从 TSV 里读。

挑数规则都写在各个 `collect_*` 的 docstring 里。三条通用原则：

* 跨 seed 一律报 mean(sd)，sd 是样本标准差（3 个 seed，n-1）；
* 任何「更好」的陈述都要带上它的代价列，不允许只取安全那一侧；
* 符号跨 seed 不一致时输出 `sign_consistent=false`，正文就不能当结论写。

用法：
    python scripts/extract_aaai27_gate_numbers.py --reports-dir reports
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

FAULT_FAMILIES = {
	"seen": ("insole_missing", "encoder_dropout", "imu_bias", "packet_loss", "stuck_imu", "sensor_delay"),
	"heldout_op": ("packet_loss_burst", "packet_loss_partial", "sensor_delay_jitter"),
}


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--reports-dir", default="reports")
	p.add_argument("--output", default="reports/aaai27_gate_numbers.json")
	return p.parse_args()


def read_tsv(path: Path) -> List[Dict[str, str]]:
	if not path.is_file():
		return []
	with path.open(encoding="utf-8") as handle:
		return [row for row in csv.DictReader(handle, delimiter="\t")]


def num(row: Dict[str, str], key: str) -> Optional[float]:
	try:
		value = float(row.get(key, ""))
	except (TypeError, ValueError):
		return None
	return None if math.isnan(value) else value


def mean_sd(values: Sequence[float]) -> Dict[str, object]:
	clean = [v for v in values if v is not None]
	if not clean:
		return {"mean": None, "sd": None, "n": 0}
	return {
		"mean": statistics.fmean(clean),
		"sd": statistics.stdev(clean) if len(clean) > 1 else 0.0,
		"n": len(clean),
		"min": min(clean),
		"max": max(clean),
		"sign_consistent": all(v > 0 for v in clean) or all(v < 0 for v in clean),
	}


def collect_reselection(directory: Path) -> Dict[str, object]:
	"""§1：选出的工作点与它在 val 上的六项 clean 约束值，逐 seed 加跨 seed 汇总。"""
	rows = read_tsv(directory / "aggregate.tsv")
	if not rows:
		return {}
	fields = [
		"val_clean_x_opp", "val_fault_x_opp", "val_clean_tracking_rmse",
		"val_clean_capped_overlap_retention", "val_clean_full_shutdown_fraction",
		"val_clean_shutdowns_per_minute", "val_clean_mean_abs_torque_rate",
		"selected_gate_softness", "selected_gate_deadband",
		"selected_gate_max_fall_per_step", "selected_gate_max_rise_per_step",
	]
	out: Dict[str, object] = {
		"per_seed": {
			str(row.get("seed")): {f: num(row, f) for f in fields} for row in rows
		},
		"objective": rows[0].get("selection_objective"),
		"n_candidates": num(rows[0], "n_candidates"),
	}
	for field in fields:
		out[field] = mean_sd([v for v in (num(row, field) for row in rows) if v is not None])
	# 选出的 rate limit 是否在所有 seed 上一致——不一致就不能在正文写成单一工作点。
	fall = {row.get("selected_gate_max_fall_per_step") for row in rows}
	rise = {row.get("selected_gate_max_rise_per_step") for row in rows}
	out["rate_limit_identical_across_seeds"] = len(fall) == 1 and len(rise) == 1
	return out


def collect_baselines(directory: Path) -> Dict[str, object]:
	"""§2：九格门控在 (split × 故障族) 上的均值，以及随机化检验的最坏情形。

	最坏情形（percentile 最高的那个故障算子）比宏平均更该进正文：宏平均会把
	「有几个算子上门控与随机不可区分」这一负面结果稀释掉。
	"""
	rows = [r for r in read_tsv(directory / "aggregate.tsv") if r.get("gate")]
	random_rows = [
		r
		for name in ("randomization_v2_fc_main_seed7.tsv", "randomization_v2_fc_main_seed13.tsv",
			"randomization_v2_fc_main_seed23.tsv")
		for r in read_tsv(directory / name)
	]
	if not rows:
		return {}

	metrics = ("x_opp", "aligned_command_retention", "tracking_rmse",
		"mean_abs_torque_rate", "shutdowns_per_minute", "full_shutdown_fraction")
	by_gate: Dict[str, Dict[str, object]] = {}
	buckets: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
	for row in rows:
		fault = str(row.get("fault_operator"))
		family = "clean" if fault == "clean" else next(
			(name for name, members in FAULT_FAMILIES.items() if fault in members), "other"
		)
		buckets[(str(row.get("gate")), str(row.get("task_split")), family)].append(row)
	for (gate, split, family), group in buckets.items():
		# 先在故障算子上平均（每个 seed 一个数），再跨 seed 求 mean(sd)。
		per_seed: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
		for row in group:
			for metric in metrics:
				value = num(row, metric)
				if value is not None:
					per_seed[str(row.get("seed"))][metric].append(value)
		record: Dict[str, object] = {}
		for metric in metrics:
			seed_means = [
				statistics.fmean(values[metric]) for values in per_seed.values() if values.get(metric)
			]
			record[metric] = mean_sd(seed_means)
		by_gate.setdefault(gate, {})[f"{split}/{family}"] = record

	# 随机化：报告与随机不可区分的格子（percentile ≥ 0.5），逐 control 列全名。
	indistinguishable: Dict[str, List[str]] = defaultdict(list)
	worst: Dict[str, Dict[str, object]] = {}
	for row in random_rows:
		control = str(row.get("gate"))
		pct = num(row, "actual_x_opp_percentile")
		if pct is None:
			continue
		key = f"{row.get('task_split')}/{row.get('fault_operator')}"
		if pct >= 0.5:
			indistinguishable[control].append(f"{key}@seed{row.get('seed')}")
		current = worst.get(control)
		if current is None or pct > float(current["percentile"]):
			worst[control] = {"cell": key, "seed": row.get("seed"), "percentile": pct,
				"p_value": num(row, "one_sided_p_value")}
	return {
		"by_gate": by_gate,
		"randomization_worst": worst,
		"randomization_indistinguishable_cells": {k: sorted(v) for k, v in indistinguishable.items()},
		"randomization_n_cells": len(random_rows),
		"draws": num(random_rows[0], "draws") if random_rows else None,
	}


def collect_dynamics(directory: Path) -> Dict[str, object]:
	"""§3：五类必报行 + trade-off 判定（降 X_opp 但抬 torque rate 的配置）。"""
	rows = read_tsv(directory / "aggregate.tsv")
	if not rows:
		return {}
	families: Dict[str, Dict[str, object]] = {}
	trade_offs: List[Dict[str, object]] = []

	# 以 ungated 的 torque rate 为参照判 trade-off，逐 (split, scenario, seed) 取。
	ungated: Dict[Tuple[str, str, str], Dict[str, Optional[float]]] = {}
	for row in rows:
		if row.get("gate_family") == "ungated":
			ungated[(str(row.get("task_split")), str(row.get("scenario")), str(row.get("seed")))] = {
				"x_opp": num(row, "x_opp"), "rate": num(row, "mean_abs_torque_rate"),
			}

	grouped: Dict[Tuple[str, str, str, str, str], List[Dict[str, str]]] = defaultdict(list)
	for row in rows:
		grouped[(
			str(row.get("gate_family")), str(row.get("task_split")), str(row.get("scenario")),
			str(row.get("max_fall_per_step", "")), str(row.get("max_rise_per_step", "")),
		)].append(row)

	for key, group in grouped.items():
		family, split, scenario, fall, rise = key
		record: Dict[str, object] = {"family": family, "task_split": split, "scenario": scenario,
			"max_fall_per_step": fall, "max_rise_per_step": rise}
		for metric in ("x_opp", "aligned_command_retention", "tracking_rmse",
			"mean_abs_torque_rate", "shutdowns_per_minute", "p95_shutdown_duration_ms"):
			record[metric] = mean_sd([v for v in (num(r, metric) for r in group) if v is not None])
		families[f"{family}|{split}|{scenario}|{fall}|{rise}"] = record

		if family == "ungated":
			continue
		safer = 0
		rougher = 0
		for row in group:
			base = ungated.get((split, scenario, str(row.get("seed"))))
			x = num(row, "x_opp")
			rate = num(row, "mean_abs_torque_rate")
			if not base or x is None or rate is None or base["x_opp"] is None or base["rate"] is None:
				continue
			if x < base["x_opp"]:
				safer += 1
			if rate > base["rate"]:
				rougher += 1
		if safer and rougher:
			trade_offs.append(
				{
					"family": family, "task_split": split, "scenario": scenario,
					"max_fall_per_step": fall, "max_rise_per_step": rise,
					"n_seeds_safer": safer, "n_seeds_rougher": rougher,
					"note": "lowers X_opp but raises torque rate relative to ungated: trade-off, not strictly safer",
				}
			)
	return {"rows": families, "trade_offs": trade_offs, "n_rows": len(rows)}


def _duration_label(row: Dict[str, str]) -> str:
	"""故障时长的分桶名。

	`duration_ms` 为 nan 的行不是坏数据：它们是「不恢复」条件——故障从 onset 一直持续到
	窗口末尾，所以没有有限时长。这类行的 onset 延迟仍然有意义（故障确实在 onset 发生），
	但恢复延迟必然缺失。单独命名成 `persistent`，避免在表里出现一个叫 "nan" 的桶，
	也避免它和真的缺值混在一起。
	"""
	raw = str(row.get("duration_ms", "")).strip()
	if not raw or raw.lower() == "nan":
		return "persistent"
	return raw


def collect_transients(directory: Path) -> Dict[str, object]:
	"""§4：按故障时长汇总延迟/漏检，以及 clean-pause 负对照的误报率。"""
	rows = read_tsv(directory / "aggregate.tsv")
	if not rows:
		return {}
	by_duration: Dict[str, Dict[str, object]] = defaultdict(dict)
	control: Dict[str, object] = {}
	metrics = ("onset_to_half_ms", "onset_to_half_rate", "onset_to_zero_ms",
		"recovery_to_half_ms", "recovery_to_full_ms", "fault_miss_rate",
		"false_shutdowns_per_minute", "alarm_chatter_per_minute")

	buckets: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
	for row in rows:
		if row.get("fault_operator") == "clean_pause" or row.get("is_negative_control") == "1":
			buckets[("clean_pause", str(row.get("task_split")))].append(row)
		else:
			buckets[(_duration_label(row), str(row.get("task_split")))].append(row)
	for (bucket, split), group in buckets.items():
		record = {
			metric: mean_sd([v for v in (num(r, metric) for r in group) if v is not None])
			for metric in metrics
		}
		record["n_rows"] = len(group)
		if bucket == "clean_pause":
			control[split] = record
		else:
			by_duration[split][bucket] = record
	return {"by_duration": dict(by_duration), "clean_pause_control": control, "n_rows": len(rows)}


def collect_paired(directory: Path) -> Dict[str, object]:
	"""§5：每个指标上跨故障算子显著的格子数，以及 X_opp 的宏观区间。"""
	rows = read_tsv(directory / "paired_summary.tsv")
	seed_rows = read_tsv(directory / "seed_sensitivity.tsv")
	if not rows:
		return {}
	by_metric: Dict[str, Dict[str, object]] = defaultdict(dict)
	for row in rows:
		metric = str(row.get("metric"))
		key = f"{row.get('task_split')}/{row.get('fault_operator')}/seed{row.get('seed')}"
		n_participants = num(row, "n_participants")
		by_metric[metric][key] = {
			"delta": num(row, "delta_fault_specific_mean"),
			"ci95": [num(row, "ci95_low"), num(row, "ci95_high")],
			"p": num(row, "bootstrap_p_two_sided"),
			"n_participants": n_participants,
			"n_pairs": num(row, "n_pairs"),
			"significant": row.get("significant_at_95") == "1",
			"ci_excludes_zero": row.get("ci_excludes_zero") == "1",
			"underpowered": row.get("underpowered") == "1",
		}
	summary = {
		metric: {
			"n_cells": len(cells),
			"n_significant": sum(1 for c in cells.values() if c["significant"]),
			"n_ci_excludes_zero": sum(1 for c in cells.values() if c["ci_excludes_zero"]),
			"n_underpowered": sum(1 for c in cells.values() if c["underpowered"]),
			"n_negative": sum(1 for c in cells.values() if (c["delta"] or 0) < 0),
			"min_participants": min(
				(c["n_participants"] for c in cells.values() if c["n_participants"] is not None),
				default=None,
			),
		}
		for metric, cells in by_metric.items()
	}
	# 被试数是这一节的硬上限：held-out 只有 2 个被试时，被试级 bootstrap 的重采样空间
	# 只有 3 种均值，区间窄和 p=0 都是退化产物。这个标记必须传到正文，否则会写出
	# 「190/216 个格子显著」这种由 n=2 造出来的结论。
	underpowered_total = sum(s["n_underpowered"] for s in summary.values())
	inference_note = (
		"participant-level bootstrap is degenerate at this participant count: "
		"window-level pairing removes within-participant variance but cannot create "
		"between-participant degrees of freedom"
		if underpowered_total
		else ""
	)
	seed_consistency = {
		f"{r.get('task_split')}/{r.get('fault_operator')}/{r.get('metric')}": {
			"n_seeds": num(r, "n_seeds"),
			"mean": num(r, "across_seed_mean"),
			"sd": num(r, "across_seed_sd"),
			"sign_consistent": r.get("sign_consistent") == "1",
		}
		for r in seed_rows
	}
	return {"cells": {k: dict(v) for k, v in by_metric.items()}, "summary": summary,
		"seed_sensitivity": seed_consistency,
		"n_underpowered_cells": underpowered_total,
		"n_cells_total": sum(s["n_cells"] for s in summary.values()),
		"inference_limitation": inference_note}


def collect_reverse(directory: Path) -> Dict[str, object]:
	"""§6：Pareto 前沿与每格的 seed 数。seed<3 的格子标出来，不能写成结论。"""
	pareto = read_tsv(directory / "pareto.tsv")
	if not pareto:
		return {}
	return {
		"pareto": [
			{
				"mode": r.get("mode"), "label": r.get("mode_label"),
				"fault_x_opp": num(r, "fault_x_opp_mean"),
				"clean_cost": num(r, "clean_cost"),
				"clean_cost_metric": r.get("clean_cost_metric"),
				"on_front": r.get("on_pareto_front") == "1",
				"dominated_by": r.get("dominated_by"),
				"n_seeds": num(r, "n_seeds"),
			}
			for r in pareto
		],
		"n_cells": len(pareto),
		"cells_required": 7,
		"cells_with_three_seeds": sum(1 for r in pareto if (num(r, "n_seeds") or 0) >= 3),
	}


def main() -> None:
	cli = parse_args()
	reports = Path(cli.reports_dir)
	payload = {
		"gate_reselection": collect_reselection(reports / "aaai27_gate_reselection"),
		"gate_baselines": collect_baselines(reports / "aaai27_gate_baselines"),
		"gate_dynamics": collect_dynamics(reports / "aaai27_gate_dynamics"),
		"transient_faults": collect_transients(reports / "aaai27_transient_faults"),
		"paired_analysis": collect_paired(reports / "aaai27_paired_analysis"),
		"reverse_ablation": collect_reverse(reports / "aaai27_reverse_ablation"),
	}
	missing = [k for k, v in payload.items() if not v]
	payload["_provenance"] = {
		"reports_dir": str(reports),
		"missing_sections": missing,
		"fault_families": {k: list(v) for k, v in FAULT_FAMILIES.items()},
	}
	out = Path(cli.output)
	out.parent.mkdir(parents=True, exist_ok=True)
	out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
	for key, value in payload.items():
		if key.startswith("_"):
			continue
		print(f"{key:20s} {'ok' if value else 'MISSING'}")
	if missing:
		print(f"\n缺 {len(missing)} 节：{', '.join(missing)}")
	print(f"Wrote {out}")


if __name__ == "__main__":
	main()
