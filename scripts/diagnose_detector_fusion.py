"""Compare validation-only detector fusion rules on an existing checkpoint.

The production fusion divides each detector channel by its clean-validation
90th percentile and takes a maximum. This diagnostic compares that rule with
empirical-CDF calibration while keeping model weights and test labels out of
policy selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_reliability_experiment as R
from reliability.features import feature_groups, input_names_for_profile, label_names
from reliability.model import load_checkpoint
from scripts.diag_imu_bias import build_datasets, parse_csv, restore_report_args, sha256_file


SIGNAL_ORDER = ("logit", "aleatoric", "residual", "forecast", "epistemic", "staleness", "coherence", "drift")
FUSION_METHODS = ("q90_max", "ecdf_max", "ecdf_mean", "ecdf_fisher")


def parse_cli() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--report", default="reports/v2_refreshed_main_seed7/reliability_report.json")
	parser.add_argument("--checkpoint", default="reports/v2_main_seed7/reliability_tcn_best.pt")
	parser.add_argument("--output-dir", default="reports/diagnostics/detector_fusion_seed7")
	parser.add_argument("--device", default="cpu")
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--num-workers", type=int, default=8)
	parser.add_argument("--max-eval-batches", type=int, default=0)
	parser.add_argument("--max-signals", type=int, default=3)
	parser.add_argument("--selection-faults", default="insole_missing,encoder_dropout,imu_bias,packet_loss,sensor_delay")
	parser.add_argument(
		"--test-faults",
		default="insole_missing,encoder_dropout,imu_bias,packet_loss,packet_loss_burst,packet_loss_partial,sensor_delay,sensor_delay_jitter",
	)
	return parser.parse_args()


def stable_condition_seed(base_seed: int, split: str, fault: str) -> int:
	digest = hashlib.sha256(f"{split}/{fault}".encode("utf-8")).digest()
	return base_seed + int.from_bytes(digest[:4], "little") % 1_000_000


def collect_scores(
	model,
	dataset,
	groups: Dict[str, list[int]],
	args: argparse.Namespace,
	device: torch.device,
	split: str,
	fault: str,
) -> Dict[str, torch.Tensor]:
	R.set_seed(stable_condition_seed(args.seed, split, fault))
	print(f"collecting {split}/{fault} windows={len(dataset)}", flush=True)
	return R.collect_detector_scores(model, dataset, groups, args, device, fault)


def fit_calibration(clean: Dict[str, torch.Tensor], available: Sequence[str]) -> dict:
	q90 = {
		name: float(torch.quantile(clean[name], 0.9).clamp_min(1e-6))
		for name in available
	}
	# Match estimate_risk_refs() so q90_max is a faithful production baseline.
	if "staleness" in q90:
		q90["staleness"] = max(q90["staleness"], 0.02)
	if "drift" in q90:
		q90["drift"] = max(q90["drift"], 1e-3)
	return {
		"q90": q90,
		"sorted_clean": {
			name: torch.sort(clean[name].float().flatten()).values
			for name in available
		},
	}


def empirical_cdf(values: torch.Tensor, sorted_clean: torch.Tensor) -> torch.Tensor:
	rank = torch.searchsorted(sorted_clean, values.float().contiguous(), right=True).float()
	return rank / float(sorted_clean.numel() + 1)


def fused_score(
	scores: Dict[str, torch.Tensor],
	calibration: dict,
	signals: Sequence[str],
	method: str,
) -> torch.Tensor:
	if method == "q90_max":
		parts = [scores[name].float() / calibration["q90"][name] for name in signals]
		return torch.stack(parts, dim=1).amax(dim=1)
	percentiles = [empirical_cdf(scores[name], calibration["sorted_clean"][name]) for name in signals]
	stacked = torch.stack(percentiles, dim=1)
	if method == "ecdf_max":
		return stacked.amax(dim=1)
	if method == "ecdf_mean":
		return stacked.mean(dim=1)
	if method == "ecdf_fisher":
		upper = 1.0 - 1.0 / (next(iter(calibration["sorted_clean"].values())).numel() + 1.0)
		return -torch.log1p(-stacked.clamp(max=upper)).sum(dim=1)
	raise ValueError(f"Unknown fusion method: {method}")


def auroc(clean: torch.Tensor, faulty: torch.Tensor) -> float:
	values = torch.cat([clean, faulty])
	labels = torch.cat([torch.zeros_like(clean), torch.ones_like(faulty)])
	return R.binary_auroc(values, labels)


def finite(values: Iterable[float]) -> list[float]:
	return [float(value) for value in values if math.isfinite(float(value))]


def select_policies(
	clean: Dict[str, torch.Tensor],
	faults: Dict[str, Dict[str, torch.Tensor]],
	calibration: dict,
	available: Sequence[str],
	max_signals: int,
) -> list[dict]:
	policies = []
	for method in FUSION_METHODS:
		for size in range(1, min(max_signals, len(available)) + 1):
			for signals in combinations(available, size):
				clean_fused = fused_score(clean, calibration, signals, method)
				aurocs = {
					fault: auroc(clean_fused, fused_score(scores, calibration, signals, method))
					for fault, scores in faults.items()
				}
				valid = finite(aurocs.values())
				if not valid:
					continue
				mean_auc = sum(valid) / len(valid)
				worst_auc = min(valid)
				score = mean_auc + 0.25 * worst_auc - 0.002 * (size - 1)
				policies.append(
					{
						"method": method,
						"signals": list(signals),
						"validation_aurocs": aurocs,
						"mean_validation_auc": mean_auc,
						"worst_validation_auc": worst_auc,
						"selection_score": score,
					}
				)
	return sorted(policies, key=lambda row: row["selection_score"], reverse=True)


def evaluate_policy(
	policy: dict,
	clean: Dict[str, torch.Tensor],
	faults: Dict[str, Dict[str, torch.Tensor]],
	calibration: dict,
) -> Dict[str, float]:
	clean_fused = fused_score(clean, calibration, policy["signals"], policy["method"])
	return {
		fault: auroc(clean_fused, fused_score(scores, calibration, policy["signals"], policy["method"]))
		for fault, scores in faults.items()
	}


def best_signal_aurocs(
	clean: Dict[str, torch.Tensor],
	faults: Dict[str, Dict[str, torch.Tensor]],
	available: Sequence[str],
) -> Dict[str, dict]:
	result = {}
	for fault, scores in faults.items():
		candidates = [(name, auroc(clean[name], scores[name])) for name in available]
		name, value = max(candidates, key=lambda item: item[1])
		result[fault] = {"signal": name, "auroc": value}
	return result


def write_metrics(path: Path, rows: Sequence[dict]) -> None:
	fields = ["split", "fault", "selected_auc", "current_auc", "best_signal", "best_signal_auc"]
	with path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
		writer.writeheader()
		writer.writerows(rows)


def main() -> None:
	cli = parse_cli()
	report_path = Path(cli.report)
	checkpoint_path = Path(cli.checkpoint)
	report = json.loads(report_path.read_text(encoding="utf-8"))
	args = restore_report_args(
		report,
		checkpoint_path,
		cli.device,
		cli.batch_size,
		cli.num_workers,
		cli.max_eval_batches,
	)
	args.mc_samples = 0
	args.extra_models = []
	input_names, target_names, datasets = build_datasets(report, args, ("val", "test_id", "test_ood"))
	groups = feature_groups(input_names)
	device = torch.device(args.device)
	model, payload = load_checkpoint(checkpoint_path, map_location=device)
	R.validate_checkpoint_inputs(payload, input_names, target_names)
	model = model.to(device).eval()

	selection_faults = parse_csv(cli.selection_faults)
	test_faults = parse_csv(cli.test_faults)
	validation_clean = collect_scores(model, datasets["val"], groups, args, device, "val", "clean")
	validation_faults = {
		fault: collect_scores(model, datasets["val"], groups, args, device, "val", fault)
		for fault in selection_faults
	}
	available = [
		name for name in SIGNAL_ORDER
		if name in validation_clean and all(name in scores for scores in validation_faults.values())
	]
	calibration = fit_calibration(validation_clean, available)
	policies = select_policies(
		validation_clean,
		validation_faults,
		calibration,
		available,
		max(cli.max_signals, 1),
	)
	if not policies:
		raise RuntimeError("No detector fusion policy could be selected.")
	selected = policies[0]
	current_signals = report.get("results", {}).get("_detector_policy", {}).get("signals", available[:1])
	current = {"method": "q90_max", "signals": [name for name in current_signals if name in available]}

	print("\nTop validation policies")
	for rank, policy in enumerate(policies[:10], start=1):
		print(
			f"{rank:>2}. score={policy['selection_score']:.4f} mean={policy['mean_validation_auc']:.4f} "
			f"worst={policy['worst_validation_auc']:.4f} method={policy['method']} signals={','.join(policy['signals'])}"
		)

	all_metrics = {}
	rows = []
	for split in ("test_id", "test_ood"):
		clean = collect_scores(model, datasets[split], groups, args, device, split, "clean")
		fault_scores = {
			fault: collect_scores(model, datasets[split], groups, args, device, split, fault)
			for fault in test_faults
		}
		selected_aurocs = evaluate_policy(selected, clean, fault_scores, calibration)
		current_aurocs = evaluate_policy(current, clean, fault_scores, calibration)
		best = best_signal_aurocs(clean, fault_scores, available)
		all_metrics[split] = {
			"selected": selected_aurocs,
			"current": current_aurocs,
			"best_signal": best,
		}
		for fault in test_faults:
			rows.append(
				{
					"split": split,
					"fault": fault,
					"selected_auc": selected_aurocs[fault],
					"current_auc": current_aurocs[fault],
					"best_signal": best[fault]["signal"],
					"best_signal_auc": best[fault]["auroc"],
				}
			)

	print("\nFrozen test comparison")
	print(f"{'split/fault':<38} {'selected':>9} {'current':>9} {'best':>9}  signal")
	for row in rows:
		print(
			f"{row['split'] + '/' + row['fault']:<38} {row['selected_auc']:>9.3f} "
			f"{row['current_auc']:>9.3f} {row['best_signal_auc']:>9.3f}  {row['best_signal']}"
		)

	output_dir = Path(cli.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	metrics_path = output_dir / "detector_fusion_metrics.tsv"
	result_path = output_dir / "detector_fusion_diagnostic.json"
	write_metrics(metrics_path, rows)
	result = {
		"source_report": str(report_path),
		"source_checkpoint": str(checkpoint_path),
		"source_checkpoint_sha256": sha256_file(checkpoint_path),
		"selection_source": "validation_faults_only",
		"selection_faults": selection_faults,
		"available_signals": available,
		"calibration": {
			"q90": calibration["q90"],
			"clean_window_count": {name: values.numel() for name, values in calibration["sorted_clean"].items()},
		},
		"selected_policy": selected,
		"current_policy": current,
		"top_validation_policies": policies[:25],
		"test_metrics": all_metrics,
	}
	result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
	print(f"\nWrote {metrics_path}")
	print(f"Wrote {result_path}")


if __name__ == "__main__":
	main()
