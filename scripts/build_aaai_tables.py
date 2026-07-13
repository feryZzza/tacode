"""Build the two core AAAI evidence tables from the refreshed suite.

The column set is fixed by the paper's detector taxonomy and safety claims.
No test result is used to choose which rows or channels are reported.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


FAULTS = (
	"insole_missing",
	"encoder_dropout",
	"imu_bias",
	"packet_loss",
	"packet_loss_burst",
	"packet_loss_partial",
	"sensor_delay",
	"sensor_delay_jitter",
)
CHALLENGE_FAULTS = {"packet_loss_burst", "packet_loss_partial", "sensor_delay_jitter"}
DISPLAY_SIGNALS = ("logit", "residual", "forecast", "staleness", "coherence", "drift")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--suite-dir", default="reports/v2_refreshed_suite")
	parser.add_argument("--output-dir", default="", help="Defaults to --suite-dir.")
	return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
	with path.open(encoding="utf-8", newline="") as handle:
		return list(csv.DictReader(handle, delimiter="\t"))


def number(row: dict[str, str], key: str) -> float | None:
	value = row.get(key, "")
	if value in ("", None):
		return None
	try:
		return float(value)
	except ValueError:
		return None


def mean_std(row: dict[str, str], stem: str, digits: int = 3) -> str:
	mean = number(row, f"{stem}_mean")
	std = number(row, f"{stem}_std")
	if mean is None:
		return "NA"
	if std is None:
		return f"{mean:.{digits}f}"
	return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def fault_label(fault: str) -> str:
	label = fault.replace("_", " ")
	return f"{label} [held-out]" if fault in CHALLENGE_FAULTS else label


def write_tsv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
	with path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
		writer.writeheader()
		writer.writerows(rows)


def build_detection_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
	by_detector = {row["detector"]: row for row in rows}
	result = []
	for split in ("test_id", "test_ood"):
		for fault in FAULTS:
			key = f"fault_detection/{split}/{fault}"
			row = by_detector.get(key)
			if row is None:
				raise KeyError(f"Missing detector aggregate row: {key}")
			out: dict[str, object] = {
				"split": split,
				"fault": fault,
				"held_out_generator": fault in CHALLENGE_FAULTS,
				"n": int(float(row["n"])),
				"fused_mean": number(row, "auroc_mean"),
				"fused_std": number(row, "auroc_std"),
			}
			for signal in DISPLAY_SIGNALS:
				out[f"{signal}_mean"] = number(row, f"auroc_{signal}_mean")
				out[f"{signal}_std"] = number(row, f"auroc_{signal}_std")
			result.append(out)
	return result


def render_detection_markdown(source_rows: list[dict[str, str]]) -> str:
	by_detector = {row["detector"]: row for row in source_rows}
	lines = [
		"# Table I. Fault Detection AUROC",
		"",
		"Mean +/- std across three seeds. Held-out rows use fault generators excluded from training. "
		"Channels are fixed by the detectability taxonomy; Fused is the frozen q90-max policy and coherence is causal.",
	]
	for split, title in (("test_id", "ID"), ("test_ood", "OOD")):
		lines.extend(
			[
				"",
				f"## {title}",
				"",
				"| Fault | Fused (q90-max) | Logit | Residual | Forecast | Staleness | Coherence | Drift |",
				"|---|---:|---:|---:|---:|---:|---:|---:|",
			]
		)
		for fault in FAULTS:
			row = by_detector[f"fault_detection/{split}/{fault}"]
			cells = [
				fault_label(fault),
				mean_std(row, "auroc"),
				*(mean_std(row, f"auroc_{signal}") for signal in DISPLAY_SIGNALS),
			]
			lines.append("| " + " | ".join(cells) + " |")
	return "\n".join(lines) + "\n"


def build_safety_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
	by_scenario = {row["scenario"]: row for row in rows}
	result = []
	for split in ("test_id", "test_ood"):
		for fault in ("clean", *FAULTS):
			key = f"{split}/{fault}"
			row = by_scenario.get(key)
			if row is None:
				raise KeyError(f"Missing safety aggregate row: {key}")
			result.append(
				{
					"split": split,
					"fault": fault,
					"held_out_generator": fault in CHALLENGE_FAULTS,
					"n": int(float(row["n"])),
					"ungated_wrong_mean": number(row, "our_ungated_wrong_mean"),
					"ungated_wrong_std": number(row, "our_ungated_wrong_std"),
					"gated_wrong_mean": number(row, "our_wrong_mean"),
					"gated_wrong_std": number(row, "our_wrong_std"),
					"wrong_delta_mean": number(row, "gate_wrong_delta_mean"),
					"wrong_delta_std": number(row, "gate_wrong_delta_std"),
					"ungated_retained_mean": number(row, "our_ungated_retained_mean"),
					"ungated_retained_std": number(row, "our_ungated_retained_std"),
					"gated_retained_mean": number(row, "our_retained_mean"),
					"gated_retained_std": number(row, "our_retained_std"),
					"retained_ratio_mean": number(row, "gate_retained_ratio_mean"),
					"retained_ratio_std": number(row, "gate_retained_ratio_std"),
					"mean_gate_mean": number(row, "mean_gate_mean"),
					"mean_gate_std": number(row, "mean_gate_std"),
				}
			)
	return result


def render_safety_markdown(rows: list[dict[str, str]]) -> str:
	by_scenario = {row["scenario"]: row for row in rows}
	lines = [
		"# Table II. Utility-Aware Gate Safety Trade-off",
		"",
		"Mean +/- std across three seeds. Wrong delta is gated minus ungated (negative is safer); "
		"retained ratio is gated divided by ungated aligned torque.",
	]
	for split, title in (("test_id", "ID"), ("test_ood", "OOD")):
		lines.extend(
			[
				"",
				f"## {title}",
				"",
				"| Scenario | Wrong ungated | Wrong gated | Wrong delta | Retained ungated | Retained gated | Retained ratio | Mean gate |",
				"|---|---:|---:|---:|---:|---:|---:|---:|",
			]
		)
		for fault in ("clean", *FAULTS):
			row = by_scenario[f"{split}/{fault}"]
			cells = [
				fault_label(fault),
				mean_std(row, "our_ungated_wrong"),
				mean_std(row, "our_wrong"),
				mean_std(row, "gate_wrong_delta"),
				mean_std(row, "our_ungated_retained"),
				mean_std(row, "our_retained"),
				mean_std(row, "gate_retained_ratio"),
				mean_std(row, "mean_gate"),
			]
			lines.append("| " + " | ".join(cells) + " |")
	return "\n".join(lines) + "\n"


def main() -> None:
	args = parse_args()
	suite_dir = Path(args.suite_dir)
	output_dir = Path(args.output_dir) if args.output_dir else suite_dir
	output_dir.mkdir(parents=True, exist_ok=True)
	detection_source = read_tsv(suite_dir / "main_detection_aggregate.tsv")
	safety_source = read_tsv(suite_dir / "main_aggregate.tsv")

	detection_rows = build_detection_rows(detection_source)
	detection_fields = list(detection_rows[0])
	write_tsv(output_dir / "table1_detection.tsv", detection_rows, detection_fields)
	(output_dir / "table1_detection.md").write_text(
		render_detection_markdown(detection_source), encoding="utf-8"
	)

	safety_rows = build_safety_rows(safety_source)
	safety_fields = list(safety_rows[0])
	write_tsv(output_dir / "table2_safety.tsv", safety_rows, safety_fields)
	(output_dir / "table2_safety.md").write_text(render_safety_markdown(safety_source), encoding="utf-8")

	print(f"Wrote {output_dir / 'table1_detection.tsv'}")
	print(f"Wrote {output_dir / 'table1_detection.md'}")
	print(f"Wrote {output_dir / 'table2_safety.tsv'}")
	print(f"Wrote {output_dir / 'table2_safety.md'}")


if __name__ == "__main__":
	main()
