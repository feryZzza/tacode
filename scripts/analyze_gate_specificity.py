#!/usr/bin/env python3
"""Compute clean-adjusted gate effects from the archived paper-suite aggregates.

The synthetic-fault scenarios reuse the clean evaluation corpus within each
seed and task split.  This script therefore subtracts the clean gate effect
from every fault-scenario effect before macro-averaging.  It is an
aggregate-level, seed- and split-matched contrast; it is not a window-level
paired confidence interval.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


SEEN_FAULTS = (
	"insole_missing",
	"encoder_dropout",
	"imu_bias",
	"packet_loss",
	"stuck_imu",
	"sensor_delay",
)
HELD_OUT_OPERATORS = (
	"packet_loss_burst",
	"packet_loss_partial",
	"sensor_delay_jitter",
)
FAULTS = (*SEEN_FAULTS, *HELD_OUT_OPERATORS)
SPLITS = ("test_id", "test_ood")
METRICS = {
	"wrong_fraction": "gate_wrong_delta",
	"opposition_integral": "gate_wrong_energy_delta",
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--input",
		type=Path,
		default=Path("reports/v2_fc_paper_suite/core_runs.tsv"),
	)
	parser.add_argument(
		"--output-json",
		type=Path,
		default=Path("reports/v2_fc_paper_suite/gate_specificity.json"),
	)
	parser.add_argument(
		"--output-md",
		type=Path,
		default=Path("reports/v2_fc_paper_suite/gate_specificity.md"),
	)
	return parser.parse_args()


def load_main_rows(path: Path) -> dict[tuple[int, str, str], dict[str, float]]:
	rows: dict[tuple[int, str, str], dict[str, float]] = {}
	with path.open(encoding="utf-8", newline="") as handle:
		for row in csv.DictReader(handle, delimiter="\t"):
			if not (
				row["run_family"] == "main"
				and row["profile"] == "human"
				and row["training_mode"] == "prob_aug_recon_fc"
				and row["mode"] == "eval"
			):
				continue
			seed = int(float(row["seed"]))
			split, scenario = row["scenario"].split("/", 1)
			if split not in SPLITS or scenario not in ("clean", *FAULTS):
				continue
			key = (seed, split, scenario)
			if key in rows:
				raise ValueError(f"Duplicate aggregate row: {key}")
			rows[key] = {name: float(row[column]) for name, column in METRICS.items()}
	return rows


def summarize(values: list[float]) -> dict[str, object]:
	return {
		"mean": mean(values),
		"sd": stdev(values) if len(values) > 1 else 0.0,
		"n_seeds": len(values),
		"per_seed": values,
	}


def analyze(rows: dict[tuple[int, str, str], dict[str, float]]) -> dict[str, object]:
	seeds = sorted({seed for seed, _, _ in rows})
	expected = {(seed, split, scenario) for seed in seeds for split in SPLITS for scenario in ("clean", *FAULTS)}
	missing = sorted(expected - set(rows))
	if missing:
		raise ValueError(f"Missing {len(missing)} required rows; first: {missing[0]}")

	result: dict[str, object] = {
		"design": {
			"seeds": seeds,
			"splits": list(SPLITS),
			"seen_faults": list(SEEN_FAULTS),
			"held_out_operators": list(HELD_OUT_OPERATORS),
			"matching": "within seed and task split",
			"unit": "scenario-level aggregate",
		}
	}
	for metric in METRICS:
		per_seed: defaultdict[str, list[float]] = defaultdict(list)
		for seed in seeds:
			fault_effects: list[float] = []
			clean_effects: list[float] = []
			adjusted: list[float] = []
			for split in SPLITS:
				clean = rows[(seed, split, "clean")][metric]
				for fault in FAULTS:
					fault_value = rows[(seed, split, fault)][metric]
					fault_effects.append(fault_value)
					clean_effects.append(clean)
					adjusted.append(fault_value - clean)
			per_seed["fault_effect"].append(mean(fault_effects))
			per_seed["matched_clean_effect"].append(mean(clean_effects))
			per_seed["clean_adjusted_effect"].append(mean(adjusted))

		fault_mean = mean(per_seed["fault_effect"])
		adjusted_mean = mean(per_seed["clean_adjusted_effect"])
		excess_share = adjusted_mean / fault_mean if fault_mean else None
		result[metric] = {
			"fault_effect": summarize(per_seed["fault_effect"]),
			"matched_clean_effect": summarize(per_seed["matched_clean_effect"]),
			"clean_adjusted_effect": summarize(per_seed["clean_adjusted_effect"]),
			"fault_specific_share_of_total_effect": excess_share,
		}
	return result


def render_markdown(result: dict[str, object]) -> str:
	lines = [
		"# Clean-adjusted gate effect",
		"",
		"Scenario-level aggregate contrast matched within training seed and task split.",
		"It is not a window-level paired confidence interval.",
		"",
		"| Metric | Fault effect | Matched clean effect | Clean-adjusted effect | Share beyond clean |",
		"|---|---:|---:|---:|---:|",
	]
	for metric, unit_scale, unit in (
		("wrong_fraction", 100.0, "pp"),
		("opposition_integral", 1.0, "$(Nm/kg)^2 s$"),
	):
		node = result[metric]
		fault = node["fault_effect"]
		clean = node["matched_clean_effect"]
		adjusted = node["clean_adjusted_effect"]
		share = node["fault_specific_share_of_total_effect"]
		lines.append(
			f"| {metric.replace('_', ' ')} "
			f"| {fault['mean'] * unit_scale:+.3f} ± {fault['sd'] * unit_scale:.3f} {unit} "
			f"| {clean['mean'] * unit_scale:+.3f} ± {clean['sd'] * unit_scale:.3f} {unit} "
			f"| {adjusted['mean'] * unit_scale:+.3f} ± {adjusted['sd'] * unit_scale:.3f} {unit} "
			f"| {share:.1%} |"
		)
	return "\n".join(lines) + "\n"


def main() -> None:
	args = parse_args()
	result = analyze(load_main_rows(args.input))
	args.output_json.parent.mkdir(parents=True, exist_ok=True)
	args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
	args.output_md.write_text(render_markdown(result), encoding="utf-8")
	print(f"Wrote {args.output_json}")
	print(f"Wrote {args.output_md}")


if __name__ == "__main__":
	main()
