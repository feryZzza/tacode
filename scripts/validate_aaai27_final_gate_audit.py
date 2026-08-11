#!/usr/bin/env python3
"""Validate the final AAAI-27 all-channel gate audit before manuscript use.

This intentionally checks policy and metric semantics that a row-count-only
aggregator cannot detect:

* the actuation gate uses all six causal command-time channels;
* the Detector-gate AUROC subset remains a separate diagnostic policy;
* every downstream stage reads the reselected reports;
* the random attenuation control matches the paper-defined retention ratio;
* retention and aligned-command adequacy are stored as distinct quantities.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set


EXPECTED_SIGNALS = {
	"logit",
	"aleatoric",
	"residual",
	"forecast",
	"epistemic",
	"staleness",
}
EXPECTED_SEEDS = {"7", "13", "23"}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--reports-dir", default="reports")
	parser.add_argument("--retention-tolerance", type=float, default=1e-5)
	return parser.parse_args()


def read_json(path: Path, errors: List[str]) -> Dict[str, object]:
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except FileNotFoundError:
		errors.append(f"missing {path}")
	except (OSError, json.JSONDecodeError) as exc:
		errors.append(f"cannot read {path}: {exc}")
	return {}


def parse_signals(value: object) -> Set[str]:
	if isinstance(value, str):
		return {item.strip() for item in value.split(",") if item.strip()}
	if isinstance(value, Sequence):
		return {str(item).strip() for item in value if str(item).strip()}
	return set()


def check_all_channel_policy(
	policy: object,
	label: str,
	errors: List[str],
	field: str = "signals",
) -> None:
	if not isinstance(policy, dict):
		errors.append(f"{label}: missing gate policy")
		return
	actual = parse_signals(policy.get(field))
	if actual != EXPECTED_SIGNALS:
		errors.append(
			f"{label}: {field}={sorted(actual)}, expected {sorted(EXPECTED_SIGNALS)}"
		)


def load_tsv(path: Path, errors: List[str]) -> List[Dict[str, str]]:
	try:
		with path.open(encoding="utf-8") as handle:
			return list(csv.DictReader(handle, delimiter="\t"))
	except OSError as exc:
		errors.append(f"cannot read {path}: {exc}")
		return []


def require_columns(
	path: Path,
	rows: Sequence[Dict[str, str]],
	columns: Iterable[str],
	errors: List[str],
) -> None:
	if not rows:
		errors.append(f"{path}: empty table")
		return
	missing = set(columns) - set(rows[0])
	if missing:
		errors.append(f"{path}: missing columns {sorted(missing)}")


def finite_float(value: object) -> float | None:
	try:
		number = float(str(value))
	except (TypeError, ValueError):
		return None
	return number if math.isfinite(number) else None


def normalized_csv(value: object) -> Set[str]:
	if isinstance(value, str):
		return {item.strip() for item in value.split(",") if item.strip()}
	if isinstance(value, Sequence):
		return {str(item).strip() for item in value if str(item).strip()}
	return set()


def same_scalar(left: object, right: object) -> bool:
	left_number = finite_float(left)
	right_number = finite_float(right)
	if left_number is not None and right_number is not None:
		return abs(left_number - right_number) <= 1e-12
	return str(left) == str(right)


def check_reselection(reports: Path, errors: List[str]) -> None:
	paths = sorted((reports / "aaai27_gate_reselection").glob("seed*/reliability_report.json"))
	if len(paths) != 3:
		errors.append(f"gate reselection: found {len(paths)} reports, expected 3")
	seen: Set[str] = set()
	for path in paths:
		report = read_json(path, errors)
		args = report.get("args", {}) if isinstance(report, dict) else {}
		results = report.get("results", {}) if isinstance(report, dict) else {}
		if not isinstance(args, dict) or not isinstance(results, dict):
			errors.append(f"{path}: malformed report")
			continue
		seed = str(args.get("seed", ""))
		seen.add(seed)
		if int(args.get("limit_trials", 0) or 0) != 0:
			errors.append(f"{path}: limit_trials must be 0")
		source_path = reports / f"v2_fc_main_seed{seed}" / "reliability_report.json"
		source = read_json(source_path, errors)
		source_args = source.get("args", {}) if isinstance(source, dict) else {}
		if isinstance(source_args, dict):
			for field in (
				"window_size",
				"stride",
				"input_profile",
				"side",
				"training_mode",
				"limit_trials",
				"val_count",
				"test_count",
				"val_participants",
				"test_participants",
				"max_windows_per_trial",
				"min_valid_fraction",
				"eval_ignore_history",
				"batch_size",
				"amp",
			):
				if not same_scalar(args.get(field), source_args.get(field)):
					errors.append(
						f"{path}: {field}={args.get(field)!r} differs from "
						f"{source_path} ({source_args.get(field)!r})"
					)
			if normalized_csv(args.get("heldout_tasks")) != normalized_csv(
				source_args.get("heldout_tasks")
			):
				errors.append(
					f"{path}: heldout_tasks={sorted(normalized_csv(args.get('heldout_tasks')))} "
					f"differs from source "
					f"{sorted(normalized_csv(source_args.get('heldout_tasks')))}"
				)
		gate_policy = results.get("_gate_policy", {})
		check_all_channel_policy(gate_policy, str(path), errors, field="gate_signals")
		if not isinstance(gate_policy, dict):
			continue
		if gate_policy.get("selection_objective") != "x_opp":
			errors.append(f"{path}: selection_objective is not x_opp")
		for field in (
			"val_clean_aligned_command_retention",
			"val_clean_aligned_command_adequacy",
			"val_clean_capped_overlap_retention",
			"val_clean_capped_overlap_adequacy",
		):
			if finite_float(gate_policy.get(field)) is None:
				errors.append(f"{path}: missing finite {field}")
		candidates = gate_policy.get("candidates", [])
		if not isinstance(candidates, list) or not any(
			isinstance(candidate, dict) and candidate.get("constraints_met")
			for candidate in candidates
		):
			errors.append(f"{path}: no validation-feasible gate candidate")
		detector = results.get("_detector_policy", {})
		if not isinstance(detector, dict):
			errors.append(f"{path}: missing _detector_policy")
			continue
		detector_signals = parse_signals(detector.get("online_signals"))
		if not detector_signals or not detector_signals <= EXPECTED_SIGNALS:
			errors.append(
				f"{path}: invalid Detector-gate subset {sorted(detector_signals)}"
			)
	if seen != EXPECTED_SEEDS:
		errors.append(f"gate reselection seeds={sorted(seen)}, expected {sorted(EXPECTED_SEEDS)}")


def check_stage_provenance(reports: Path, errors: List[str]) -> None:
	for stage in ("aaai27_gate_baselines", "aaai27_gate_dynamics", "aaai27_transient_faults"):
		paths = sorted((reports / stage).glob("provenance_*.json"))
		if len(paths) != 3:
			errors.append(f"{stage}: found {len(paths)} provenance files, expected 3")
		seeds: Set[str] = set()
		for path in paths:
			provenance = read_json(path, errors)
			if not provenance:
				continue
			seeds.add(str(provenance.get("training_seed", "")))
			check_all_channel_policy(
				provenance.get("gate_policy"), str(path), errors
			)
			report_path = str(provenance.get("report", ""))
			if "aaai27_gate_reselection/seed" not in report_path:
				errors.append(f"{path}: sourced non-reselected report {report_path!r}")
			if stage == "aaai27_gate_baselines":
				if provenance.get("reference_gate") != "all_gate_channels":
					errors.append(f"{path}: reference_gate is not all_gate_channels")
				gate_policy = provenance.get("gate_policy", {})
				if isinstance(gate_policy, dict):
					detector = parse_signals(gate_policy.get("detector_signals"))
					if not detector or not detector <= EXPECTED_SIGNALS:
						errors.append(f"{path}: invalid detector_signals {sorted(detector)}")
		if seeds != EXPECTED_SEEDS:
			errors.append(f"{stage}: seeds={sorted(seeds)}, expected {sorted(EXPECTED_SEEDS)}")

	paired_path = reports / "aaai27_paired_analysis" / "provenance_paired.json"
	paired = read_json(paired_path, errors)
	runs = paired.get("runs", []) if isinstance(paired, dict) else []
	if not isinstance(runs, list) or len(runs) != 3:
		errors.append(f"{paired_path}: expected 3 run provenance entries")
		return
	seeds = set()
	for index, run in enumerate(runs):
		if not isinstance(run, dict):
			errors.append(f"{paired_path}: malformed run {index}")
			continue
		seeds.add(str(run.get("training_seed", "")))
		check_all_channel_policy(run.get("gate_policy"), f"{paired_path} run {index}", errors)
		if "aaai27_gate_reselection/seed" not in str(run.get("report", "")):
			errors.append(f"{paired_path} run {index}: source is not gate reselection")
	if seeds != EXPECTED_SEEDS:
		errors.append(f"paired analysis: seeds={sorted(seeds)}, expected {sorted(EXPECTED_SEEDS)}")
	if paired.get("bootstrap_draws") != 10000:
		errors.append(f"{paired_path}: bootstrap_draws must be 10000")


def check_tables(reports: Path, tolerance: float, errors: List[str]) -> None:
	common = {"aligned_command_retention", "aligned_command_adequacy"}
	capped = {"capped_overlap_retention", "capped_overlap_adequacy"}
	baseline_dir = reports / "aaai27_gate_baselines"
	baseline_paths = sorted(baseline_dir.glob("gate_baselines_*.tsv"))
	random_paths = sorted(baseline_dir.glob("randomization_*.tsv"))
	if len(baseline_paths) != 3 or len(random_paths) != 3:
		errors.append(
			f"gate baselines: found {len(baseline_paths)} metric and "
			f"{len(random_paths)} randomization tables, expected 3 each"
		)
	for path in baseline_paths:
		rows = load_tsv(path, errors)
		require_columns(path, rows, common | capped, errors)
		gates = {row.get("gate", "") for row in rows}
		for required in ("all_gate_channels", "selected_subset", "oracle_error"):
			if required not in gates:
				errors.append(f"{path}: missing gate {required}")
	for path in random_paths:
		rows = load_tsv(path, errors)
		require_columns(
			path,
			rows,
			{
				"gate",
				"matched_quantity",
				"random_retention_mean",
				"actual_retention",
				"random_adequacy_mean",
				"actual_adequacy",
			},
			errors,
		)
		attempts = 0
		for row in rows:
			if row.get("gate") != "random_attenuation":
				continue
			attempts += 1
			if row.get("matched_quantity") != "aligned_command_retention":
				errors.append(f"{path}: random attenuation matches the wrong quantity")
			random_value = finite_float(row.get("random_retention_mean"))
			actual_value = finite_float(row.get("actual_retention"))
			if random_value is None or actual_value is None:
				errors.append(f"{path}: non-finite retention match")
			elif abs(random_value - actual_value) > tolerance:
				errors.append(
					f"{path}: retention mismatch {random_value} vs {actual_value} "
					f"(tolerance {tolerance})"
				)
		if attempts == 0:
			errors.append(f"{path}: no random_attenuation rows")

	for stage, pattern in (
		("aaai27_gate_dynamics", "gate_dynamics_*.tsv"),
		("aaai27_transient_faults", "transient_faults_*.tsv"),
	):
		paths = sorted((reports / stage).glob(pattern))
		if len(paths) != 3:
			errors.append(f"{stage}: found {len(paths)} metric tables, expected 3")
		for path in paths:
			rows = load_tsv(path, errors)
			require_columns(
				path,
				rows,
				common | (capped if stage == "aaai27_gate_dynamics" else set()),
				errors,
			)
			if stage == "aaai27_transient_faults" and rows:
				operators = {row.get("fault_operator", "") for row in rows}
				if "clean_pause" not in operators:
					errors.append(f"{path}: missing clean_pause control")
				if not any("+" in operator for operator in operators):
					errors.append(f"{path}: missing simultaneous-fault conditions")
				splits = {row.get("task_split", "") for row in rows}
				if not {"test_id", "test_ood"} <= splits:
					errors.append(f"{path}: missing test_id or test_ood")

	paired_rows = load_tsv(
		reports / "aaai27_paired_analysis" / "paired_summary.tsv", errors
	)
	metrics = {row.get("metric", "") for row in paired_rows}
	if not {"aligned_command_retention", "aligned_command_adequacy"} <= metrics:
		errors.append(
			"paired_summary.tsv: retention and adequacy are not both present"
		)


def check_aggregate_status(reports: Path, errors: List[str]) -> None:
	status_path = reports / "aaai27_suite_status.json"
	status = read_json(status_path, errors)
	problems = status.get("problems") if isinstance(status, dict) else None
	if problems:
		errors.append(f"{status_path}: aggregate reports problems: {problems}")
	reverse_path = reports / "aaai27_reverse_ablation" / "reverse_ablation_runs.tsv"
	rows = load_tsv(reverse_path, errors)
	modes = {row.get("mode", "") for row in rows}
	if "full" in modes and "all_gate_channels" in modes:
		errors.append(
			f"{reverse_path}: duplicate full references (full and all_gate_channels)"
		)
	# split-lock：所有七格必须在同一 split 上评测。一个留出任务不同的批次会改变
	# test_id/test_ood 的任务成员，其数字无法与其余格比较，行数校验查不出来。
	provenance_path = reports / "aaai27_reverse_ablation" / "provenance_reverse.json"
	provenance = read_json(provenance_path, errors)
	if isinstance(provenance, dict):
		if int(provenance.get("n_cells", 0) or 0) != int(
			provenance.get("cells_required", 7) or 7
		):
			errors.append(
				f"{provenance_path}: n_cells={provenance.get('n_cells')} != "
				f"cells_required={provenance.get('cells_required')}"
			)
		if not provenance.get("reference_heldout_tasks"):
			errors.append(f"{provenance_path}: missing reference_heldout_tasks (no split lock)")
		# 门控协议锁：同一张 Pareto 表里的训练侧格子必须共享选择目标与候选网格规模，
		# 否则「去掉某组件更差」里混着「那一格搜的候选更少」。
		protocols = provenance.get("gate_protocols")
		if not isinstance(protocols, dict) or not protocols:
			errors.append(f"{provenance_path}: missing gate_protocols (no gate-protocol lock)")
		elif not provenance.get("gate_protocol_uniform"):
			seen = sorted(
				{
					f"{p.get('selection_objective')}/{p.get('n_candidates')}"
					for p in protocols.values()
					if isinstance(p, dict)
				}
			)
			errors.append(
				f"{provenance_path}: gate protocols differ across cells: {seen}"
			)
		front = load_tsv(
			reports / "aaai27_reverse_ablation" / "pareto.tsv", errors
		)
		if not front:
			errors.append(
				f"{reverse_path}: empty Pareto front (a cell is missing x_opp)"
			)


def main() -> int:
	cli = parse_args()
	reports = Path(cli.reports_dir)
	errors: List[str] = []
	check_reselection(reports, errors)
	check_stage_provenance(reports, errors)
	check_tables(reports, cli.retention_tolerance, errors)
	check_aggregate_status(reports, errors)
	if errors:
		print(f"FINAL GATE AUDIT FAILED ({len(errors)} issue(s))")
		for error in errors:
			print(f"  - {error}")
		return 1
	print("FINAL GATE AUDIT PASSED")
	print("  seeds: 7, 13, 23")
	print("  actuation policy: all six command-time channels")
	print(f"  retention-match tolerance: {cli.retention_tolerance:g}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
