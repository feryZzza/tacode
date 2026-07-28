#!/usr/bin/env python3
"""EXPERIMENTS_TODO §9：把七个产出目录汇总成 `aggregate.tsv` 并做交付自检。

每个 stage 的逐 run 明细已经由各自的 driver 落盘，这里做两件事：

1. 把同一目录下的逐 seed 表纵向拼成 `aggregate.tsv`，并按 §9 要求的分组键（task split
   × fault operator × gate/条件）补一张跨 seed 的 `aggregate_by_seed.tsv`：均值、标准差、
   种子数、以及**符号一致性**。跨 seed 只报均值会掩盖「某个 seed 结论相反」。
2. 检查每个目录该有的东西是否齐（逐 run JSON/TSV、run_manifest、provenance、日志），
   缺什么明确列出来。这是交付清单，不是装饰——缺 provenance 的数字不能进论文。

用法：
    python scripts/aggregate_aaai27_suite.py --reports-dir reports
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

#: 目录 -> (明细表 glob, 跨 seed 分组键, 汇总的数值列前缀过滤)
STAGES: Dict[str, Dict[str, object]] = {
	"aaai27_gate_reselection": {
		"glob": "seed*/reliability_report.json",
		"kind": "report",
	},
	"aaai27_gate_baselines": {
		"glob": "gate_baselines_*.tsv",
		"group": ("task_split", "fault_operator", "gate"),
	},
	"aaai27_gate_dynamics": {
		"glob": "gate_dynamics_*.tsv",
		"group": ("task_split", "scenario", "gate_family", "max_fall_per_step", "max_rise_per_step"),
	},
	"aaai27_transient_faults": {
		"glob": "transient_faults_*.tsv",
		"group": ("task_split", "fault_operator", "onset_step", "duration_ms", "with_recovery"),
	},
	"aaai27_paired_analysis": {
		"glob": "paired_summary.tsv",
		"group": ("task_split", "fault_operator", "metric"),
	},
	"aaai27_reverse_ablation": {
		"glob": "reverse_*.tsv",
		"group": ("mode", "task_split", "fault_operator"),
	},
}

#: §1 报告里必须出现的字段（§1.3 的验收列表）。
RESELECTION_FIELDS = (
	"selection_objective",
	"val_clean_x_opp",
	"val_fault_x_opp",
	"val_clean_tracking_rmse",
	"val_clean_aligned_command_adequacy",
	"val_clean_aligned_command_retention",
	"val_clean_capped_overlap_adequacy",
	"val_clean_capped_overlap_retention",
	"val_clean_full_shutdown_fraction",
	"val_clean_shutdowns_per_minute",
	"val_clean_mean_abs_torque_rate",
	"selected_gate_deadband",
	"selected_gate_softness",
	"selected_gate_max_fall_per_step",
	"selected_gate_max_rise_per_step",
)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--reports-dir", default="reports")
	p.add_argument("--log-dir", default="reports/aaai27_logs")
	return p.parse_args()


def read_tsv(path: Path) -> List[Dict[str, str]]:
	with path.open(encoding="utf-8") as handle:
		return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
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


def _as_float(text: object) -> Optional[float]:
	try:
		value = float(str(text))
	except (TypeError, ValueError):
		return None
	return None if math.isnan(value) else value


def summarize_across_seeds(
	rows: Sequence[Dict[str, str]],
	group_keys: Sequence[str],
) -> List[Dict[str, object]]:
	"""按分组键跨 seed 汇总每个数值列：均值/标准差/极值/符号一致性。"""
	present = [k for k in group_keys if rows and k in rows[0]]
	if not present:
		return []
	numeric: List[str] = []
	for key in rows[0]:
		if key in present or key in ("run", "seed"):
			continue
		if any(_as_float(row.get(key)) is not None for row in rows):
			numeric.append(key)

	buckets: Dict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)
	for row in rows:
		buckets[tuple(str(row.get(k, "")) for k in present)].append(row)

	out: List[Dict[str, object]] = []
	for key, group in sorted(buckets.items()):
		record: Dict[str, object] = dict(zip(present, key))
		record["n_seeds"] = len({row.get("seed", "") for row in group})
		record["n_rows"] = len(group)
		for column in numeric:
			values = [v for v in (_as_float(row.get(column)) for row in group) if v is not None]
			if not values:
				continue
			record[f"{column}_mean"] = statistics.fmean(values)
			record[f"{column}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
			record[f"{column}_min"] = min(values)
			record[f"{column}_max"] = max(values)
			if len(values) > 1:
				# 符号一致性：跨 seed 是否都同向。不一致的格子在论文里不能当结论。
				record[f"{column}_sign_consistent"] = all(v > 0 for v in values) or all(v < 0 for v in values)
		out.append(record)
	return out


def collect_reselection(directory: Path) -> Tuple[List[Dict[str, object]], List[str]]:
	"""§1 的 report 不是 TSV，单独抽平字段并检查 §1.3 要求的键是否齐。"""
	rows: List[Dict[str, object]] = []
	problems: List[str] = []
	for report_path in sorted(directory.glob("seed*/reliability_report.json")):
		report = json.loads(report_path.read_text(encoding="utf-8"))
		policy = report.get("results", {}).get("_gate_policy", {})
		row: Dict[str, object] = {
			"run": report_path.parent.name,
			"seed": report.get("args", {}).get("seed", ""),
		}
		for field in RESELECTION_FIELDS:
			row[field] = policy.get(field)
			if field not in policy:
				problems.append(f"{report_path.parent.name}: _gate_policy 缺 {field}")
		row["n_candidates"] = len(policy.get("candidates", []) or [])
		row["n_feasible"] = sum(
			1 for c in (policy.get("candidates", []) or []) if c.get("constraints_met")
		)
		rows.append(row)
	return rows, problems


def main() -> None:
	cli = parse_args()
	reports = Path(cli.reports_dir)
	log_dir = Path(cli.log_dir)
	manifest: List[Dict[str, object]] = []
	problems: List[str] = []

	for name, spec in STAGES.items():
		directory = reports / name
		if not directory.is_dir():
			problems.append(f"{name}: 目录不存在（stage 未跑或未回传）")
			manifest.append({"stage": name, "status": "missing", "n_rows": 0, "n_runs": 0})
			continue

		if spec.get("kind") == "report":
			rows, stage_problems = collect_reselection(directory)
			problems.extend(stage_problems)
			write_tsv(directory / "aggregate.tsv", rows)
			manifest.append(
				{
					"stage": name, "status": "ok" if rows else "empty",
					"n_rows": len(rows), "n_runs": len(rows),
					"aggregate": str(directory / "aggregate.tsv"),
				}
			)
			continue

		files = sorted(directory.glob(str(spec["glob"])))
		merged: List[Dict[str, str]] = []
		for path in files:
			merged.extend(read_tsv(path))
		write_tsv(directory / "aggregate.tsv", merged)
		by_seed = summarize_across_seeds(merged, tuple(spec.get("group", ())))
		write_tsv(directory / "aggregate_by_seed.tsv", by_seed)

		seeds = sorted({row.get("seed", "") for row in merged})
		if len(seeds) < 3 and name != "aaai27_paired_analysis":
			problems.append(f"{name}: 只有 {len(seeds)} 个 seed（{','.join(map(str, seeds))}），§9 要求 3")
		if not (directory / f"run_manifest_{seeds[0] if seeds else ''}").exists():
			if not list(directory.glob("run_manifest*.tsv")):
				problems.append(f"{name}: 缺 run_manifest*.tsv")
		if not list(directory.glob("provenance*.json")):
			problems.append(f"{name}: 缺 provenance*.json（无 checkpoint 哈希/split/seed 记录）")
		manifest.append(
			{
				"stage": name, "status": "ok" if merged else "empty",
				"n_rows": len(merged), "n_runs": len(files),
				"seeds": ",".join(str(s) for s in seeds),
				"aggregate": str(directory / "aggregate.tsv"),
				"by_seed": str(directory / "aggregate_by_seed.tsv"),
			}
		)

	if log_dir.is_dir():
		logs = sorted(p.name for p in log_dir.glob("*.log"))
	else:
		logs = []
		problems.append(f"{log_dir}: 日志目录不存在")

	out = reports / "aaai27_suite_manifest.tsv"
	write_tsv(out, manifest)
	(reports / "aaai27_suite_status.json").write_text(
		json.dumps(
			{"stages": manifest, "logs": logs, "problems": problems},
			indent=2,
			ensure_ascii=False,
		),
		encoding="utf-8",
	)
	for row in manifest:
		print(f"{row['stage']:32s} {row['status']:8s} rows={row['n_rows']:<7} runs={row['n_runs']}")
	if problems:
		print(f"\n{len(problems)} 项待办：")
		for item in problems:
			print(f"  - {item}")
	else:
		print("\n七个目录齐全。")


if __name__ == "__main__":
	main()
