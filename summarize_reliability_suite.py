"""Build paper-review tables from one or more reliability reports.

The single-run summary is useful while iterating. This script is stricter: it
expects several reports, aggregates mean/std across seeds or ablations, and
emits a gap report that points to evidence still missing for a strong paper
submission.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable


CORE_SCENARIOS = [
	"test_id/clean",
	"test_ood/clean",
	"test_id/insole_missing",
	"test_id/encoder_dropout",
	"test_id/imu_bias",
	"test_id/packet_loss",
	"test_id/packet_loss_burst",
	"test_id/packet_loss_partial",
	"test_id/sensor_delay",
	"test_id/sensor_delay_jitter",
	"test_ood/insole_missing",
	"test_ood/encoder_dropout",
	"test_ood/imu_bias",
	"test_ood/packet_loss",
	"test_ood/packet_loss_burst",
	"test_ood/packet_loss_partial",
	"test_ood/sensor_delay",
	"test_ood/sensor_delay_jitter",
]

RUN_FIELDS = [
	"our_rmse",
	"nature_rmse",
	"rmse_delta",
	"our_r2",
	"nature_r2",
	"r2_delta",
	"our_wrong",
	"our_ungated_wrong",
	"gate_wrong_delta",
	"nature_wrong",
	"wrong_delta",
	"our_retained",
	"our_ungated_retained",
	"gate_retained_delta",
	"gate_retained_ratio",
	"nature_retained",
	"retained_delta",
	"ungated_retained_delta",
	"mean_gate",
	"cl_net_moment_rms_reduction_ratio",
	"cl_abs_work_reduction_ratio",
	"cl_fight_fraction",
]

DETECTOR_SIGNAL_NAMES = ["logit", "aleatoric", "residual", "forecast", "epistemic", "staleness", "coherence", "drift"]
DETECTOR_FUSION_GAP_WARN = 0.10

DETECTOR_FIELDS = [
	"auroc",
	"best_signal_auroc",
	"auroc_logit",
	"auroc_aleatoric",
	"auroc_residual",
	"auroc_forecast",
	"auroc_epistemic",
	"auroc_staleness",
	"auroc_coherence",
	"auroc_drift",
]

FAIR_BASELINE_FIELDS = [
	"main_rmse",
	"baseline_rmse",
	"rmse_delta",
	"main_r2",
	"baseline_r2",
	"r2_delta",
	"main_ungated_retained",
	"baseline_ungated_retained",
	"ungated_retained_delta",
]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("reports", nargs="+", help="Report paths, report directories, or glob patterns.")
	parser.add_argument("--output-dir", default="reports/reliability_paper_suite")
	parser.add_argument("--primary-report", default="", help="Report used for the automatic gap narrative.")
	parser.add_argument("--scenarios", default=",".join(CORE_SCENARIOS))
	parser.add_argument("--min-seed-reports", type=int, default=3)
	parser.add_argument("--ood-rmse-gap-warn", type=float, default=0.02)
	parser.add_argument("--fair-baseline-rmse-gap-warn", type=float, default=0.01)
	parser.add_argument("--retained-gap-warn", type=float, default=-0.20)
	parser.add_argument(
		"--gate-retention-ratio-warn",
		type=float,
		default=0.90,
		help="Warn that the gate itself is conservative below this gated/ungated retained-torque ratio.",
	)
	parser.add_argument("--detector-auroc-warn", type=float, default=0.60)
	parser.add_argument("--gate-wrong-margin", type=float, default=0.01)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	report_paths = expand_reports(args.reports)
	if not report_paths:
		raise FileNotFoundError("No reliability_report.json files matched the requested inputs.")

	reports = [load_report(path) for path in report_paths]
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]
	primary = choose_primary_report(reports, args.primary_report)
	primary_family = classify_run_family(primary)

	run_rows = build_run_rows(reports, scenarios)
	aggregate_rows = aggregate_rows_by(run_rows, ["run_family", "training_mode", "profile", "scenario"], RUN_FIELDS)
	profile_aggregate_rows = aggregate_rows_by(run_rows, ["profile", "scenario"], RUN_FIELDS)
	detector_rows = build_detector_rows(reports)
	detector_aggregate_rows = aggregate_rows_by(detector_rows, ["run_family", "training_mode", "profile", "detector"], DETECTOR_FIELDS)
	detector_profile_aggregate_rows = aggregate_rows_by(detector_rows, ["profile", "detector"], DETECTOR_FIELDS)
	gate_rows = build_gate_policy_rows(reports, args.gate_wrong_margin)
	stress_rows = build_stress_rows(reports)
	stress_aggregate_rows = aggregate_rows_by(stress_rows, ["run_family", "training_mode", "profile", "scenario"], RUN_FIELDS)
	main_run_rows = [row for row in run_rows if row.get("run_family") == primary_family]
	main_detector_rows = [row for row in detector_rows if row.get("run_family") == primary_family]
	main_gate_rows = [row for row in gate_rows if row.get("run_family") == primary_family]
	main_aggregate_rows = aggregate_rows_by(main_run_rows, ["training_mode", "profile", "scenario"], RUN_FIELDS)
	main_detector_aggregate_rows = aggregate_rows_by(main_detector_rows, ["training_mode", "profile", "detector"], DETECTOR_FIELDS)
	fair_baseline_rows = build_fair_baseline_rows(reports, scenarios)
	fair_baseline_aggregate_rows = aggregate_rows_by(
		fair_baseline_rows,
		["main_family", "profile", "scenario"],
		FAIR_BASELINE_FIELDS,
	)

	write_tsv(output_dir / "core_runs.tsv", run_rows)
	write_tsv(output_dir / "core_aggregate.tsv", aggregate_rows)
	write_tsv(output_dir / "core_aggregate_by_profile.tsv", profile_aggregate_rows)
	write_tsv(output_dir / "main_aggregate.tsv", main_aggregate_rows)
	write_tsv(output_dir / "detection_runs.tsv", detector_rows)
	write_tsv(output_dir / "detection_aggregate.tsv", detector_aggregate_rows)
	write_tsv(output_dir / "detection_aggregate_by_profile.tsv", detector_profile_aggregate_rows)
	write_tsv(output_dir / "main_detection_aggregate.tsv", main_detector_aggregate_rows)
	write_tsv(output_dir / "gate_policy.tsv", gate_rows)
	write_tsv(output_dir / "main_gate_policy.tsv", main_gate_rows)
	write_tsv(output_dir / "fair_baseline_runs.tsv", fair_baseline_rows)
	write_tsv(output_dir / "fair_baseline_aggregate.tsv", fair_baseline_aggregate_rows)
	write_tsv(output_dir / "stress_runs.tsv", stress_rows)
	write_tsv(output_dir / "stress_aggregate.tsv", stress_aggregate_rows)

	(output_dir / "paper_gap_report.md").write_text(
		render_gap_report(
			reports,
			primary,
			args,
			core_rows=run_rows,
			detector_rows=detector_rows,
			gate_rows=gate_rows,
			stress_rows=stress_rows,
			fair_baseline_rows=fair_baseline_rows,
		),
		encoding="utf-8",
	)
	print(f"Wrote {output_dir / 'paper_gap_report.md'}")
	print(f"Wrote {output_dir / 'core_aggregate.tsv'}")
	print(f"Wrote {output_dir / 'main_aggregate.tsv'}")
	print(f"Wrote {output_dir / 'detection_aggregate.tsv'}")
	print(f"Wrote {output_dir / 'main_detection_aggregate.tsv'}")
	print(f"Wrote {output_dir / 'gate_policy.tsv'}")
	print(f"Wrote {output_dir / 'main_gate_policy.tsv'}")
	print(f"Wrote {output_dir / 'fair_baseline_aggregate.tsv'}")
	if stress_rows:
		print(f"Wrote {output_dir / 'stress_aggregate.tsv'}")


def expand_reports(items: Iterable[str]) -> list[Path]:
	paths: list[Path] = []
	for item in items:
		matches = glob.glob(item)
		if not matches:
			matches = [item]
		for match in matches:
			path = Path(match)
			if path.is_dir():
				path = path / "reliability_report.json"
			if path.name != "reliability_report.json" and path.exists():
				continue
			if path.exists():
				paths.append(path)
	unique = sorted({path.resolve(): path for path in paths}.values())
	return unique


def load_report(path: Path) -> dict[str, Any]:
	report = json.loads(path.read_text(encoding="utf-8"))
	report["_path"] = str(path)
	report["_run"] = path.parent.name
	return report


def report_metadata(report: dict[str, Any]) -> dict[str, Any]:
	args = report.get("args", {})
	policy = report.get("results", {}).get("_gate_policy", {})
	return {
		"run": report["_run"],
		"report": report["_path"],
		"run_family": classify_run_family(report),
		"profile": args.get("input_profile", ""),
		"training_mode": args.get("training_mode", ""),
		"mode": args.get("mode", ""),
		"seed": args.get("seed", ""),
		"selected_gate_softness": policy.get("softness", args.get("gate_softness")),
		"selected_gate_deadband": policy.get("deadband", args.get("gate_deadband")),
	}


def classify_run_family(report: dict[str, Any]) -> str:
	run = report.get("_run", "")
	if run.startswith("v2_refreshed_main_seed"):
		return "main_refreshed"
	if run.startswith("v2_refreshed_stress_seed"):
		return "stress_refreshed"
	if run.startswith("v3_main_seed"):
		return "candidate"
	if run.startswith("v2_main_seed") or run.startswith("reliability_human_nature_capacity"):
		return "main"
	if run.startswith("v2_loso_"):
		return "loso"
	if run.startswith("v2_ablation_"):
		return "ablation"
	if run.startswith("v2_baseline_"):
		return "baseline"
	if run.startswith("v2_action_") or run.startswith("reliability_ablation_"):
		return "action"
	if run.startswith("v2_stress") or run.startswith("reliability_human_stress"):
		return "stress"
	if run.startswith("v2_ensemble"):
		return "ensemble"
	return "other"


def is_main_report(report: dict[str, Any]) -> bool:
	return classify_run_family(report) in {"main", "main_refreshed", "candidate"} and report.get("args", {}).get("input_profile") == "human"


def build_run_rows(reports: list[dict[str, Any]], scenarios: list[str]) -> list[dict[str, Any]]:
	rows = []
	for report in reports:
		results = report.get("results", {})
		for scenario in scenarios:
			ours = results.get(scenario)
			if not ours:
				continue
			rows.append(make_comparison_row(report, scenario, ours, results.get(f"nature_baseline/{scenario}", {})))
	return rows


def build_fair_baseline_rows(reports: list[dict[str, Any]], scenarios: list[str]) -> list[dict[str, Any]]:
	"""Pair main and deterministic runs only when seed and evaluation split match."""
	baseline_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
	# Prefer the dedicated E6 baseline; det_noaug ablations are a compatible fallback.
	for family in ("ablation", "baseline"):
		for report in reports:
			if classify_run_family(report) != family or report.get("args", {}).get("training_mode") != "det_noaug":
				continue
			key = (
				report.get("args", {}).get("input_profile"),
				report.get("args", {}).get("seed"),
				report_split_signature(report),
			)
			baseline_by_key[key] = report

	rows = []
	for report in reports:
		if not is_main_report(report) or report.get("args", {}).get("mode") not in ("train", "smoke", "eval"):
			continue
		key = (
			report.get("args", {}).get("input_profile"),
			report.get("args", {}).get("seed"),
			report_split_signature(report),
		)
		baseline_report = baseline_by_key.get(key)
		if baseline_report is None:
			continue
		main_results = report.get("results", {})
		baseline_results = baseline_report.get("results", {})
		for scenario in scenarios:
			main_metrics = main_results.get(scenario)
			baseline_metrics = baseline_results.get(scenario)
			if not main_metrics or not baseline_metrics:
				continue
			rows.append(
				{
					"main_run": report["_run"],
					"main_report": report["_path"],
					"main_family": classify_run_family(report),
					"baseline_run": baseline_report["_run"],
					"baseline_report": baseline_report["_path"],
					"profile": report.get("args", {}).get("input_profile", ""),
					"seed": report.get("args", {}).get("seed", ""),
					"scenario": scenario,
					"main_rmse": main_metrics.get("rmse"),
					"baseline_rmse": baseline_metrics.get("rmse"),
					"rmse_delta": delta(main_metrics.get("rmse"), baseline_metrics.get("rmse")),
					"main_r2": main_metrics.get("r2"),
					"baseline_r2": baseline_metrics.get("r2"),
					"r2_delta": delta(main_metrics.get("r2"), baseline_metrics.get("r2")),
					"main_ungated_retained": main_metrics.get("baseline_retained_aligned_torque"),
					"baseline_ungated_retained": baseline_metrics.get("baseline_retained_aligned_torque"),
					"ungated_retained_delta": delta(
						main_metrics.get("baseline_retained_aligned_torque"),
						baseline_metrics.get("baseline_retained_aligned_torque"),
					),
				}
			)
	return rows


def report_split_signature(report: dict[str, Any]) -> tuple[Any, ...]:
	parts = []
	for split_name in ("test_id", "test_ood"):
		split = report.get("splits", {}).get(split_name, {})
		parts.append(
			(
				tuple(sorted(split.get("participants", []))),
				tuple(sorted(split.get("tasks", []))),
			)
		)
	return tuple(parts)


def build_stress_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
	rows = []
	for report in reports:
		results = report.get("results", {})
		for scenario, ours in sorted(results.items()):
			if "@" not in scenario or not is_model_scenario(scenario):
				continue
			rows.append(make_comparison_row(report, scenario, ours, results.get(f"nature_baseline/{scenario}", {})))
	return rows


def make_comparison_row(
	report: dict[str, Any],
	scenario: str,
	ours: dict[str, Any],
	baseline: dict[str, Any],
) -> dict[str, Any]:
	ungated_wrong = ours.get("baseline_wrong_direction_ratio")
	gated_wrong = ours.get("gated_wrong_direction_ratio")
	ungated_retained = ours.get("baseline_retained_aligned_torque")
	gated_retained = ours.get("gated_retained_aligned_torque")
	nature_retained = baseline.get("baseline_retained_aligned_torque")
	return {
		**report_metadata(report),
		"scenario": scenario,
		"our_rmse": ours.get("rmse"),
		"nature_rmse": baseline.get("rmse"),
		"rmse_delta": delta(ours.get("rmse"), baseline.get("rmse")),
		"our_r2": ours.get("r2"),
		"nature_r2": baseline.get("r2"),
		"r2_delta": delta(ours.get("r2"), baseline.get("r2")),
		"our_wrong": gated_wrong,
		"our_ungated_wrong": ungated_wrong,
		"gate_wrong_delta": delta(gated_wrong, ungated_wrong),
		"nature_wrong": baseline.get("baseline_wrong_direction_ratio"),
		"wrong_delta": delta(gated_wrong, baseline.get("baseline_wrong_direction_ratio")),
		"our_retained": gated_retained,
		"our_ungated_retained": ungated_retained,
		"gate_retained_delta": delta(gated_retained, ungated_retained),
		"gate_retained_ratio": ratio(gated_retained, ungated_retained),
		"nature_retained": nature_retained,
		"retained_delta": delta(gated_retained, nature_retained),
		"ungated_retained_delta": delta(ungated_retained, nature_retained),
		"mean_gate": ours.get("mean_gate"),
		"cl_net_moment_rms_reduction_ratio": ours.get("cl_net_moment_rms_reduction_ratio"),
		"cl_abs_work_reduction_ratio": ours.get("cl_abs_work_reduction_ratio"),
		"cl_fight_fraction": ours.get("cl_fight_fraction"),
	}


def build_detector_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
	rows = []
	for report in reports:
		for key, metrics in sorted(report.get("results", {}).items()):
			if key.startswith("fault_detection/"):
				auroc = metrics.get("fault_auroc")
			elif key == "ood_detection/clean":
				auroc = metrics.get("risk_auroc")
			else:
				continue
			best_signal, best_signal_auroc = detector_best_signal(metrics)
			rows.append(
				{
					**report_metadata(report),
					"detector": key,
					"auroc": auroc,
					"best_signal": metrics.get("best_signal", best_signal),
					"best_signal_auroc": metrics.get("best_signal_auroc", best_signal_auroc),
					# 每路信号 AUROC（E1/E2 对比：监督 logit vs 重构残差 vs 预测残差 vs 认知不确定性）。
					"auroc_logit": metrics.get("auroc_logit"),
					"auroc_aleatoric": metrics.get("auroc_aleatoric"),
					"auroc_residual": metrics.get("auroc_residual"),
					"auroc_forecast": metrics.get("auroc_forecast"),
					"auroc_epistemic": metrics.get("auroc_epistemic"),
					"auroc_staleness": metrics.get("auroc_staleness"),
					"auroc_coherence": metrics.get("auroc_coherence"),
					"auroc_drift": metrics.get("auroc_drift"),
				}
			)
	return rows


def detector_best_signal(metrics: dict[str, Any]) -> tuple[str, float | None]:
	candidates = []
	for name in DETECTOR_SIGNAL_NAMES:
		value = metrics.get(f"auroc_{name}")
		if is_number(value):
			candidates.append((name, float(value)))
	if not candidates:
		return "", None
	return max(candidates, key=lambda item: item[1])


def build_gate_policy_rows(reports: list[dict[str, Any]], wrong_margin: float) -> list[dict[str, Any]]:
	rows = []
	for report in reports:
		results = report.get("results", {})
		policy = results.get("_gate_policy", {})
		actual_scenarios = [
			scenario
			for scenario in CORE_SCENARIOS
			if scenario in results
		]
		if actual_scenarios:
			for scenario in actual_scenarios:
				ours = results.get(scenario, {})
				baseline = results.get(f"nature_baseline/{scenario}", {})
				rows.append(
					make_gate_policy_row(
						report,
						scenario,
						ours,
						baseline,
						wrong_margin,
						policy_source="selected_val_gate",
						selected_softness=policy.get("softness", report.get("args", {}).get("gate_softness")),
						selected_deadband=policy.get("deadband", report.get("args", {}).get("gate_deadband")),
					)
				)
			continue

		# Backward-compatible fallback for reports that only contain gate_sweep rows.
		by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
		for key, metrics in sorted(results.items()):
			if not key.startswith("gate_sweep/"):
				continue
			parsed = parse_gate_key(key)
			if parsed is None:
				continue
			scenario, softness = parsed
			by_scenario[scenario].append({"softness": softness, **metrics})
		for scenario, candidates in sorted(by_scenario.items()):
			baseline = results.get(f"nature_baseline/{scenario}", {})
			nature_wrong = baseline.get("baseline_wrong_direction_ratio")
			wrong_limit = nature_wrong + wrong_margin if is_number(nature_wrong) else None
			valid = [
				row
				for row in candidates
				if wrong_limit is not None and is_number(row.get("gated_wrong_direction_ratio")) and row["gated_wrong_direction_ratio"] <= wrong_limit
			]
			met_budget = bool(valid)
			if valid:
				best = max(valid, key=lambda row: safe_number(row.get("gated_retained_aligned_torque"), -math.inf))
			else:
				best = min(
					candidates,
					key=lambda row: (
						safe_number(row.get("gated_wrong_direction_ratio"), math.inf),
						-safe_number(row.get("gated_retained_aligned_torque"), -math.inf),
					),
				)
			rows.append(
				{
					**report_metadata(report),
					"scenario": scenario,
					"policy_source": "gate_sweep_fallback",
					"selected_softness": best.get("softness"),
					"selected_deadband": best.get("gate_deadband", report.get("args", {}).get("gate_deadband")),
					"met_wrong_budget": met_budget,
						"wrong_limit": wrong_limit,
						"gated_wrong": best.get("gated_wrong_direction_ratio"),
						"ungated_wrong": None,
						"gate_wrong_delta": None,
						"gated_retained": best.get("gated_retained_aligned_torque"),
						"ungated_retained": None,
						"gate_retained_delta": None,
						"gate_retained_ratio": None,
						"mean_gate": best.get("mean_gate"),
					"nature_wrong": nature_wrong,
					"nature_retained": baseline.get("baseline_retained_aligned_torque"),
					"retained_delta": delta(best.get("gated_retained_aligned_torque"), baseline.get("baseline_retained_aligned_torque")),
				}
			)
	return rows


def make_gate_policy_row(
	report: dict[str, Any],
	scenario: str,
	ours: dict[str, Any],
	baseline: dict[str, Any],
	wrong_margin: float,
	policy_source: str,
	selected_softness: Any,
	selected_deadband: Any,
) -> dict[str, Any]:
	nature_wrong = baseline.get("baseline_wrong_direction_ratio")
	wrong_limit = nature_wrong + wrong_margin if is_number(nature_wrong) else None
	gated_wrong = ours.get("gated_wrong_direction_ratio")
	ungated_wrong = ours.get("baseline_wrong_direction_ratio")
	gated_retained = ours.get("gated_retained_aligned_torque")
	ungated_retained = ours.get("baseline_retained_aligned_torque")
	met_budget = bool(wrong_limit is not None and is_number(gated_wrong) and gated_wrong <= wrong_limit)
	return {
		**report_metadata(report),
		"scenario": scenario,
		"policy_source": policy_source,
		"selected_softness": selected_softness,
		"selected_deadband": selected_deadband,
		"met_wrong_budget": met_budget,
		"wrong_limit": wrong_limit,
		"gated_wrong": gated_wrong,
		"ungated_wrong": ungated_wrong,
		"gate_wrong_delta": delta(gated_wrong, ungated_wrong),
		"gated_retained": gated_retained,
		"ungated_retained": ungated_retained,
		"gate_retained_delta": delta(gated_retained, ungated_retained),
		"gate_retained_ratio": ratio(gated_retained, ungated_retained),
		"mean_gate": ours.get("mean_gate"),
		"nature_wrong": nature_wrong,
		"nature_retained": baseline.get("baseline_retained_aligned_torque"),
		"retained_delta": delta(ours.get("gated_retained_aligned_torque"), baseline.get("baseline_retained_aligned_torque")),
	}


def aggregate_rows_by(rows: list[dict[str, Any]], keys: list[str], fields: list[str]) -> list[dict[str, Any]]:
	grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
	for row in rows:
		grouped[tuple(row.get(key, "") for key in keys)].append(row)
	aggregated = []
	for group_key, group_rows in sorted(grouped.items()):
		out = {key: value for key, value in zip(keys, group_key)}
		out["n"] = len(group_rows)
		for field in fields:
			values = [float(row[field]) for row in group_rows if is_number(row.get(field))]
			out[f"{field}_mean"] = mean(values) if values else None
			out[f"{field}_std"] = stdev(values) if len(values) > 1 else 0.0 if values else None
		aggregated.append(out)
	return aggregated


def render_gap_report(
	reports: list[dict[str, Any]],
	primary: dict[str, Any],
	args: argparse.Namespace,
	core_rows: list[dict[str, Any]],
	detector_rows: list[dict[str, Any]],
	gate_rows: list[dict[str, Any]],
	stress_rows: list[dict[str, Any]],
	fair_baseline_rows: list[dict[str, Any]],
) -> str:
	lines = [
		"# Paper-Readiness Gap Report",
		"",
		f"- Reports: {len(reports)}",
		f"- Primary report: `{primary['_path']}`",
		"",
		"## Evidence Gaps",
		"",
	]
	lines.extend(f"- {item}" for item in evidence_gaps(reports, args, stress_rows, fair_baseline_rows))
	lines.extend(["", "## Primary Performance Gaps", ""])
	lines.extend(f"- {item}" for item in performance_gaps(primary, args, fair_baseline_rows))
	lines.extend(["", "## Recommended Next Runs", ""])
	lines.extend(
		f"- {item}"
		for item in recommended_next_runs(
			reports,
			primary,
			core_rows,
			detector_rows,
			gate_rows,
			stress_rows,
			fair_baseline_rows,
		)
	)
	lines.extend(["", "## Primary Best Validation", ""])
	best = best_history(primary.get("history", []))
	if best:
		lines.append(
			"- "
			+ ", ".join(
				[
					f"epoch={best.get('epoch')}",
					f"val_rmse={format_value(best.get('val_rmse'))}",
					f"val_r2={format_value(best.get('val_r2'))}",
					f"val_ece={format_value(best.get('val_coverage_ece'))}",
					f"uncertainty_error_corr={format_value(best.get('val_uncertainty_error_corr'))}",
				]
			)
		)
	else:
		lines.append("- No validation history found.")
	lines.extend(["", "## Primary Gate Policy", ""])
	policy = primary.get("results", {}).get("_gate_policy", {})
	if policy:
		lines.append(f"- softness={format_value(policy.get('softness'))}, deadband={format_value(policy.get('deadband'))}")
	else:
		lines.append("- No frozen gate policy found.")
	lines.extend(["", "## Primary Detector Policy", ""])
	detector_policy = primary.get("results", {}).get("_detector_policy", {})
	if detector_policy:
		signals = ",".join(detector_policy.get("signals", [])) or "all"
		validation = detector_policy.get("validation_aurocs", {})
		validation_text = ", ".join(f"{name}={format_value(value)}" for name, value in validation.items())
		lines.append(f"- signals={signals}; source={detector_policy.get('policy_source', 'unknown')}")
		if validation_text:
			lines.append(f"- validation AUROC: {validation_text}")
	else:
		lines.append("- No frozen detector policy found.")
	return "\n".join(lines) + "\n"


def evidence_gaps(
	reports: list[dict[str, Any]],
	args: argparse.Namespace,
	stress_rows: list[dict[str, Any]],
	fair_baseline_rows: list[dict[str, Any]],
) -> list[str]:
	gaps = []
	human_seed_reports = {
		(report.get("args", {}).get("seed"), report["_path"])
		for report in reports
		if is_main_report(report) and report.get("args", {}).get("mode") in ("train", "smoke", "eval")
	}
	if len(human_seed_reports) < args.min_seed_reports:
		gaps.append(f"Multi-seed evidence is weak: found {len(human_seed_reports)} human train/smoke reports; target at least {args.min_seed_reports}.")
	paired_baseline_seeds = {
		row.get("seed") for row in fair_baseline_rows if row.get("scenario") == "test_id/clean"
	}
	if len(paired_baseline_seeds) < args.min_seed_reports:
		gaps.append(
			f"Same-split deterministic baseline has only {len(paired_baseline_seeds)} paired seed(s); "
			f"target at least {args.min_seed_reports}."
		)
	profiles = {report.get("args", {}).get("input_profile") for report in reports}
	missing_profiles = [profile for profile in ["human_desired", "human_measured", "human_execution", "human_interaction"] if profile not in profiles]
	if missing_profiles:
		gaps.append("Action-input ablations are incomplete: missing " + ", ".join(missing_profiles) + ".")
	if not stress_rows:
		gaps.append("Fault stress curves are missing: run scenarios such as packet_loss@0.05, packet_loss@0.30, sensor_delay@5, sensor_delay@20.")
	else:
		stress_seeds = {row.get("seed") for row in stress_rows if row.get("seed") not in (None, "")}
		if len(stress_seeds) < args.min_seed_reports:
			gaps.append(
				f"Fault stress curves have only {len(stress_seeds)} seed(s); target at least {args.min_seed_reports} "
				"before making variance or monotonicity claims."
			)
	if not any(report.get("args", {}).get("limit_trials") == 0 for report in reports):
		gaps.append("At least one full-data aligned evaluation should be included; quick limited runs are useful only for debugging.")
	return gaps or ["Core evidence coverage looks reasonable for this report set."]


def performance_gaps(
	report: dict[str, Any],
	args: argparse.Namespace,
	fair_baseline_rows: list[dict[str, Any]],
) -> list[str]:
	results = report.get("results", {})
	gaps = []
	paired_rows = {
		row["scenario"]: row for row in fair_baseline_rows if row.get("main_run") == report.get("_run")
	}
	for scenario in ("test_id/clean", "test_ood/clean"):
		row = paired_rows.get(scenario)
		if row and is_number(row.get("rmse_delta")) and row["rmse_delta"] > args.fair_baseline_rmse_gap_warn:
			gaps.append(
				f"Same-split deterministic baseline is better on {scenario}: "
				f"{row['main_rmse']:.4f} vs {row['baseline_rmse']:.4f} (delta {row['rmse_delta']:+.4f})."
			)
	ood_clean = make_comparison_row(report, "test_ood/clean", results.get("test_ood/clean", {}), results.get("nature_baseline/test_ood/clean", {}))
	if is_number(ood_clean.get("rmse_delta")) and ood_clean["rmse_delta"] > args.ood_rmse_gap_warn:
		gaps.append(
			f"Secondary official-checkpoint comparison: OOD clean RMSE is worse by {ood_clean['rmse_delta']:.4f}; "
			"this checkpoint is not split-matched and must not be used as the primary baseline."
		)
	for scenario in ["test_id/clean", "test_ood/clean", "test_ood/sensor_delay"]:
		row = make_comparison_row(report, scenario, results.get(scenario, {}), results.get(f"nature_baseline/{scenario}", {}))
		if is_number(row.get("retained_delta")) and row["retained_delta"] < args.retained_gap_warn:
			if is_number(row.get("gate_retained_ratio")) and row["gate_retained_ratio"] < args.gate_retention_ratio_warn:
				gaps.append(
					f"Gate is conservative on {scenario}: it keeps {row['gate_retained_ratio']:.1%} of the model's ungated torque "
					f"(gate delta {row['gate_retained_delta']:.4f}; total delta vs Nature {row['retained_delta']:.4f})."
				)
			else:
				gaps.append(
					f"Base predictor under-retains torque on {scenario}: ungated delta vs Nature is "
					f"{row['ungated_retained_delta']:.4f}; the gate itself keeps {row['gate_retained_ratio']:.1%}."
				)
	core_gate_reductions = []
	for scenario, metrics in results.items():
		if not scenario.startswith(("test_id/", "test_ood/")) or scenario.endswith("/clean") or "@" in scenario:
			continue
		ungated_wrong = metrics.get("baseline_wrong_direction_ratio") if isinstance(metrics, dict) else None
		gated_wrong = metrics.get("gated_wrong_direction_ratio") if isinstance(metrics, dict) else None
		if is_number(ungated_wrong) and is_number(gated_wrong):
			core_gate_reductions.append(float(ungated_wrong) - float(gated_wrong))
	if core_gate_reductions and max(core_gate_reductions) < args.gate_wrong_margin:
		gaps.append(
			"Gate impact is small at the core fault settings: maximum absolute wrong-direction reduction is "
			f"{max(core_gate_reductions):.4f}; any strong safety claim must be supported by multi-seed severity curves."
		)
	low_detectors = []
	weak_fusions = []
	for key, metrics in sorted(results.items()):
		if key.startswith("fault_detection/"):
			_, best_signal_auroc = detector_best_signal(metrics)
			fused_auroc = metrics.get("fault_auroc")
			if is_number(best_signal_auroc) and best_signal_auroc < args.detector_auroc_warn:
				low_detectors.append(f"{key}={best_signal_auroc:.3f}")
			elif (
				is_number(best_signal_auroc)
				and is_number(fused_auroc)
				and best_signal_auroc - fused_auroc >= DETECTOR_FUSION_GAP_WARN
			):
				weak_fusions.append(f"{key}: fused={fused_auroc:.3f}, best={best_signal_auroc:.3f}")
	if low_detectors:
		gaps.append("Weak fault detectors: " + ", ".join(low_detectors[:8]) + (" ..." if len(low_detectors) > 8 else ""))
	if weak_fusions:
		gaps.append("Detector fusion masks useful signals: " + ", ".join(weak_fusions[:8]) + (" ..." if len(weak_fusions) > 8 else ""))
	best = best_history(report.get("history", []))
	if best and is_number(best.get("val_coverage_ece")) and best["val_coverage_ece"] > 0.02:
		gaps.append(f"Calibration is not yet paper-strong: best val coverage ECE is {best['val_coverage_ece']:.4f}.")
	return gaps or ["No major primary-performance gap was triggered by the configured thresholds."]


def recommended_next_runs(
	reports: list[dict[str, Any]],
	primary: dict[str, Any],
	core_rows: list[dict[str, Any]],
	detector_rows: list[dict[str, Any]],
	gate_rows: list[dict[str, Any]],
	stress_rows: list[dict[str, Any]],
	fair_baseline_rows: list[dict[str, Any]],
) -> list[str]:
	recs = []
	primary_family = classify_run_family(primary)
	primary_detector_rows = [row for row in detector_rows if row.get("run_family") == primary_family]
	if len(reports) < 3:
		recs.append("Run seed repeats with the same split and hyperparameters, then cite core_aggregate.tsv instead of a single best run.")
	if not stress_rows:
		recs.append("Run the stress task in scripts/reliability_paper_experiments.sh to diagnose packet_loss and sensor_delay across severity.")
	main_seeds = {
		report.get("args", {}).get("seed")
		for report in reports
		if is_main_report(report) and report.get("args", {}).get("mode") in ("train", "smoke", "eval")
	}
	paired_baseline_seeds = {
		row.get("seed") for row in fair_baseline_rows if row.get("scenario") == "test_id/clean"
	}
	missing_baseline_seeds = sorted(seed for seed in main_seeds - paired_baseline_seeds if seed is not None)
	if missing_baseline_seeds:
		recs.append(
			"Train same-split deterministic baselines for missing seeds: "
			+ ", ".join(str(seed) for seed in missing_baseline_seeds)
			+ "."
		)
	if gate_rows:
		if any(row.get("policy_source") != "selected_val_gate" for row in gate_rows):
			recs.append("Freeze one validation-derived gate policy before reporting test metrics.")
	else:
		recs.append("Enable --gate-softness-grid so the gate threshold is selected systematically rather than by hand.")
	if not any(row.get("profile") and row.get("profile") != "human" for row in core_rows):
		recs.append("Run action-input ablations to prove the contribution is reliability-aware estimation, not hidden action leakage.")
	weak = [row for row in primary_detector_rows if is_number(row.get("best_signal_auroc")) and row["best_signal_auroc"] < 0.6]
	if weak:
		recs.append(
			"Treat near-chance detectors as scoped empirical limitations unless a mechanism-based signal improves validation; "
			"do not tune detector candidates against test AUROC."
		)
	weak_fusion = [
		row
		for row in primary_detector_rows
		if is_number(row.get("auroc"))
		and is_number(row.get("best_signal_auroc"))
		and row["best_signal_auroc"] - row["auroc"] >= DETECTOR_FUSION_GAP_WARN
	]
	if weak_fusion:
		recs.append(
			"Keep the fixed fault-by-signal matrix as primary evidence. The validation-selected ECDF candidate is rejected "
			"because it harms held-out burst detection; no further fusion tuning is prioritized without a new mechanism."
		)
	stress_seeds = {row.get("seed") for row in stress_rows if row.get("seed") not in (None, "")}
	if stress_rows and len(stress_seeds) < 3:
		recs.append("Refresh stress curves for seeds 7, 13, and 23 before reporting severity trends as stable evidence.")
	return recs or ["The next improvement should target model-level OOD generalization, since the current evidence scaffold is present."]


def detector_row_score(row: dict[str, Any]) -> float | None:
	best = row.get("best_signal_auroc")
	if is_number(best):
		return float(best)
	auroc = row.get("auroc")
	return float(auroc) if is_number(auroc) else None


def choose_primary_report(reports: list[dict[str, Any]], primary_report: str) -> dict[str, Any]:
	if primary_report:
		target = str(Path(primary_report).resolve())
		for report in reports:
			if str(Path(report["_path"]).resolve()) == target:
				return report
		raise ValueError(f"--primary-report was not included in reports: {primary_report}")
	for report in reports:
		if report.get("_run") == "reliability_human_nature_capacity":
			return report
	for report in reports:
		args = report.get("args", {})
		if args.get("input_profile") == "human" and args.get("mode") == "train":
			return report
	return reports[0]


def parse_gate_key(key: str) -> tuple[str, float] | None:
	parts = key.split("/")
	if len(parts) != 4:
		return None
	softness_text = parts[3].replace("softness_", "").replace("p", ".").replace("m", "-")
	try:
		softness = float(softness_text)
	except ValueError:
		return None
	return f"{parts[1]}/{parts[2]}", softness


def best_history(history: list[dict[str, Any]]) -> dict[str, Any] | None:
	candidates = [row for row in history if is_number(row.get("val_rmse"))]
	return min(candidates, key=lambda row: row["val_rmse"]) if candidates else None


def is_model_scenario(key: str) -> bool:
	if key.startswith(("nature_baseline/", "fault_detection/", "gate_sweep/", "ood_detection/")):
		return False
	return "/" in key


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
	if not rows:
		path.write_text("", encoding="utf-8")
		return
	headers = list(rows[0])
	lines = ["\t".join(headers)]
	for row in rows:
		lines.append("\t".join(format_value(row.get(header)) for header in headers))
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def delta(a: Any, b: Any) -> float | None:
	if not is_number(a) or not is_number(b):
		return None
	return float(a) - float(b)


def ratio(numerator: Any, denominator: Any) -> float | None:
	if not is_number(numerator) or not is_number(denominator) or abs(float(denominator)) < 1e-12:
		return None
	return float(numerator) / float(denominator)


def is_number(value: Any) -> bool:
	return isinstance(value, (int, float)) and math.isfinite(float(value))


def safe_number(value: Any, default: float) -> float:
	return float(value) if is_number(value) else default


def format_value(value: Any) -> str:
	if value is None:
		return ""
	if is_number(value):
		return f"{float(value):.4f}"
	return str(value)


if __name__ == "__main__":
	main()
