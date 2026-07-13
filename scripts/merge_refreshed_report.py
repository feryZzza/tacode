"""Attach training provenance to an eval-only refreshed reliability report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--source", required=True, help="Original training report.")
	parser.add_argument("--refreshed", required=True, help="Eval-only report with current metrics.")
	parser.add_argument("--output", default="", help="Output path; defaults to replacing --refreshed.")
	return parser.parse_args()


def report_hash(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
	args = parse_args()
	source_path = Path(args.source)
	refreshed_path = Path(args.refreshed)
	output_path = Path(args.output) if args.output else refreshed_path
	source = json.loads(source_path.read_text(encoding="utf-8"))
	refreshed = json.loads(refreshed_path.read_text(encoding="utf-8"))

	for key in ("input_names", "label_names", "splits"):
		if source.get(key) != refreshed.get(key):
			raise ValueError(f"Source and refreshed reports differ in '{key}'.")
	if source.get("args", {}).get("seed") != refreshed.get("args", {}).get("seed"):
		raise ValueError("Source and refreshed report seeds do not match.")

	merged_args = dict(source.get("args", {}))
	for key in (
		"selected_gate_softness",
		"selected_gate_deadband",
		"selected_detector_signals",
		"detector_validation_faults",
		"detector_max_signals",
	):
		if key in refreshed.get("args", {}):
			merged_args[key] = refreshed["args"][key]
	detector_policy = refreshed.get("results", {}).get("_detector_policy", {})
	if detector_policy.get("signals"):
		merged_args["selected_detector_signals"] = ",".join(detector_policy["signals"])
	refreshed["args"] = merged_args
	refreshed["history"] = source.get("history", [])
	for key, value in source.get("results", {}).items():
		if key.startswith("nature_baseline/") and key not in refreshed.get("results", {}):
			refreshed["results"][key] = value
	refreshed["provenance"] = {
		"source_report": str(source_path),
		"source_report_sha256": report_hash(source_path),
		"evaluation_revision": "local_staleness_window25_plus_causal_imu_coherence_window32",
		"detector_fusion_revision": "validation_fault_subset_search_max3",
	}
	output_path.write_text(json.dumps(refreshed, indent=2) + "\n", encoding="utf-8")
	print(f"Merged training provenance into {output_path}")


if __name__ == "__main__":
	main()
