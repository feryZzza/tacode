"""Summarize reliability experiment JSON reports into paper-ready tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


DEFAULT_COMPARISON_SCENARIOS = [
	"test_id/clean",
	"test_ood/clean",
	"test_id/insole_missing",
	"test_id/encoder_dropout",
	"test_id/imu_bias",
	"test_id/packet_loss",
	"test_id/sensor_delay",
	"test_ood/insole_missing",
	"test_ood/encoder_dropout",
	"test_ood/sensor_delay",
]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("report", help="Path to reliability_report.json")
	parser.add_argument("--output-dir", default="", help="Defaults to the report directory.")
	parser.add_argument("--scenarios", default=",".join(DEFAULT_COMPARISON_SCENARIOS))
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	report_path = Path(args.report)
	output_dir = Path(args.output_dir) if args.output_dir else report_path.parent
	output_dir.mkdir(parents=True, exist_ok=True)

	report = json.loads(report_path.read_text(encoding="utf-8"))
	results = report.get("results", {})
	scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]

	comparison_rows = build_comparison_rows(results, scenarios)
	detector_rows = build_detector_rows(results)
	gate_rows = build_gate_sweep_rows(results)

	(output_dir / "comparison.tsv").write_text(to_tsv(comparison_rows), encoding="utf-8")
	if gate_rows:
		(output_dir / "gate_sweep.tsv").write_text(to_tsv(gate_rows), encoding="utf-8")
	(output_dir / "summary.md").write_text(
		render_markdown(report_path, report, comparison_rows, detector_rows, gate_rows),
		encoding="utf-8",
	)
	print(f"Wrote {output_dir / 'summary.md'}")
	print(f"Wrote {output_dir / 'comparison.tsv'}")
	if gate_rows:
		print(f"Wrote {output_dir / 'gate_sweep.tsv'}")


def build_comparison_rows(results: dict[str, Any], scenarios: Iterable[str]) -> list[dict[str, Any]]:
	rows = []
	for scenario in scenarios:
		ours = results.get(scenario)
		baseline = results.get(f"nature_baseline/{scenario}")
		if ours is None or baseline is None:
			continue
		row = {
			"scenario": scenario,
			"our_rmse": ours.get("rmse"),
			"nature_rmse": baseline.get("rmse"),
			"rmse_delta": delta(ours.get("rmse"), baseline.get("rmse")),
			"our_r2": ours.get("r2"),
			"nature_r2": baseline.get("r2"),
			"r2_delta": delta(ours.get("r2"), baseline.get("r2")),
			"our_gated_wrong": ours.get("gated_wrong_direction_ratio"),
			"nature_wrong": baseline.get("baseline_wrong_direction_ratio"),
			"wrong_delta": delta(ours.get("gated_wrong_direction_ratio"), baseline.get("baseline_wrong_direction_ratio")),
			"our_gated_retained": ours.get("gated_retained_aligned_torque"),
			"nature_retained": baseline.get("baseline_retained_aligned_torque"),
			"retained_delta": delta(ours.get("gated_retained_aligned_torque"), baseline.get("baseline_retained_aligned_torque")),
			"our_mean_gate": ours.get("mean_gate"),
		}
		rows.append(row)
	return rows


def build_detector_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
	rows = []
	for key, metrics in sorted(results.items()):
		if key.startswith("fault_detection/"):
			rows.append({"detector": key, "auroc": metrics.get("fault_auroc")})
		elif key == "ood_detection/clean":
			rows.append({"detector": key, "auroc": metrics.get("risk_auroc")})
	return rows


def build_gate_sweep_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
	rows = []
	for key, metrics in sorted(results.items()):
		if not key.startswith("gate_sweep/"):
			continue
		parts = key.split("/")
		if len(parts) != 4:
			continue
		rows.append(
			{
				"scenario": f"{parts[1]}/{parts[2]}",
				"softness": parts[3].replace("softness_", "").replace("p", "."),
				"gated_wrong": metrics.get("gated_wrong_direction_ratio"),
				"gated_peak_wrong": metrics.get("gated_peak_wrong_torque"),
				"gated_retained": metrics.get("gated_retained_aligned_torque"),
				"mean_gate": metrics.get("mean_gate"),
				"mean_risk": metrics.get("mean_risk"),
			}
		)
	return rows


def render_markdown(
	report_path: Path,
	report: dict[str, Any],
	comparison_rows: list[dict[str, Any]],
	detector_rows: list[dict[str, Any]],
	gate_rows: list[dict[str, Any]],
) -> str:
	args = report.get("args", {})
	lines = [
		"# Reliability Experiment Summary",
		"",
		f"- Report: `{report_path}`",
		f"- Mode: `{args.get('mode')}`",
		f"- Input profile: `{args.get('input_profile')}`",
		f"- Epochs: `{args.get('epochs')}`",
		f"- Eval ignore history: `{args.get('eval_ignore_history', 0)}`",
		f"- Gate softness: `{args.get('gate_softness', 0.5)}`",
		"",
	]
	best = best_history(report.get("history", []))
	if best:
		lines.extend(
			[
				"## Best Validation",
				"",
				md_table(
					["epoch", "val_rmse", "val_r2", "val_coverage_ece", "val_uncertainty_error_corr"],
					[
						[
							best.get("epoch"),
							best.get("val_rmse"),
							best.get("val_r2"),
							best.get("val_coverage_ece"),
							best.get("val_uncertainty_error_corr"),
						]
					],
				),
				"",
			]
		)

	lines.extend(["## Splits", ""])
	split_rows = []
	for name, split in report.get("splits", {}).items():
		split_rows.append([name, split.get("num_trials"), split.get("num_participants"), ",".join(split.get("participants", []))])
	lines.extend([md_table(["split", "trials", "participants", "ids"], split_rows), ""])

	if comparison_rows:
		lines.extend(
			[
				"## Nature Baseline Comparison",
				"",
				md_table(
					[
						"scenario",
						"our_rmse",
						"nature_rmse",
						"rmse_delta",
						"our_gated_wrong",
						"nature_wrong",
						"wrong_delta",
						"our_gated_retained",
						"nature_retained",
						"retained_delta",
					],
					[[row.get(key) for key in [
						"scenario",
						"our_rmse",
						"nature_rmse",
						"rmse_delta",
						"our_gated_wrong",
						"nature_wrong",
						"wrong_delta",
						"our_gated_retained",
						"nature_retained",
						"retained_delta",
					]] for row in comparison_rows],
				),
				"",
			]
		)

	if detector_rows:
		lines.extend(["## Detection", "", md_table(["detector", "auroc"], [[row["detector"], row["auroc"]] for row in detector_rows]), ""])

	if gate_rows:
		lines.extend(
			[
				"## Gate Sweep",
				"",
				md_table(
					["scenario", "softness", "gated_wrong", "gated_peak_wrong", "gated_retained", "mean_gate"],
					[[row.get(key) for key in ["scenario", "softness", "gated_wrong", "gated_peak_wrong", "gated_retained", "mean_gate"]] for row in gate_rows],
				),
				"",
			]
		)
	return "\n".join(lines)


def best_history(history: list[dict[str, Any]]) -> dict[str, Any] | None:
	candidates = [row for row in history if is_number(row.get("val_rmse"))]
	return min(candidates, key=lambda row: row["val_rmse"]) if candidates else None


def to_tsv(rows: list[dict[str, Any]]) -> str:
	if not rows:
		return ""
	headers = list(rows[0])
	lines = ["\t".join(headers)]
	for row in rows:
		lines.append("\t".join(format_value(row.get(header)) for header in headers))
	return "\n".join(lines) + "\n"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
	lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
	for row in rows:
		lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
	return "\n".join(lines)


def delta(a: Any, b: Any) -> float | None:
	if not is_number(a) or not is_number(b):
		return None
	return float(a) - float(b)


def is_number(value: Any) -> bool:
	return isinstance(value, (int, float)) and math.isfinite(float(value))


def format_value(value: Any) -> str:
	if value is None:
		return ""
	if is_number(value):
		return f"{float(value):.4f}"
	return str(value)


if __name__ == "__main__":
	main()
