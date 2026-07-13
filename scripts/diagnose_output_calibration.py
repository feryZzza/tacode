"""Evaluate validation-fitted output calibration on an existing checkpoint.

The script fits scale-only and affine calibrators on validation participants,
then reports frozen performance on held-out test participants. It is diagnostic
only: it does not modify the checkpoint or the saved reliability report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_reliability_experiment as R
from reliability.features import feature_groups, input_names_for_profile, label_names
from reliability.metrics import interval_calibration, masked_gaussian_nll, regression_metrics, torque_replay_metrics
from reliability.model import load_checkpoint
from reliability.nature_dataset import discover_trials, filter_records


def parse_cli() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--report", default="reports/v2_main_seed7/reliability_report.json")
	parser.add_argument("--checkpoint", default="reports/v2_main_seed7/reliability_tcn_best.pt")
	parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--num-workers", type=int, default=8)
	parser.add_argument("--max-eval-batches", type=int, default=0)
	parser.add_argument("--output-dir", default="reports/diagnostics/output_calibration_seed7")
	return parser.parse_args()


def experiment_args(report: dict, cli: argparse.Namespace) -> argparse.Namespace:
	saved_argv = sys.argv
	try:
		sys.argv = ["diagnose_output_calibration"]
		args = R.parse_args()
	finally:
		sys.argv = saved_argv
	for key, value in report.get("args", {}).items():
		if hasattr(args, key):
			setattr(args, key, value)
	args.device = cli.device
	args.batch_size = cli.batch_size
	args.num_workers = cli.num_workers
	args.mc_samples = 0
	args.max_eval_batches = cli.max_eval_batches
	return args


@torch.no_grad()
def collect_predictions(model, dataset, args, device: torch.device) -> Dict[str, torch.Tensor]:
	means, logvars, targets, masks = [], [], [], []
	for batch_index, batch in enumerate(R.make_loader(dataset, args, device, shuffle=False)):
		if args.max_eval_batches and batch_index >= args.max_eval_batches:
			break
		x = batch["x"].to(device, non_blocking=True)
		output = model(x)
		means.append(output["mean"].cpu())
		logvars.append(output["logvar"].cpu())
		targets.append(batch["y"].cpu())
		masks.append(R.apply_ignore_history(batch["mask"], args.eval_ignore_history).cpu())
	return {
		"mean": torch.cat(means),
		"logvar": torch.cat(logvars),
		"target": torch.cat(targets),
		"mask": torch.cat(masks).bool(),
	}


def concatenate(parts: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
	return {key: torch.cat([part[key] for part in parts]) for key in ("mean", "logvar", "target", "mask")}


def fit_calibrator(data: Dict[str, torch.Tensor], affine: bool) -> tuple[torch.Tensor, torch.Tensor]:
	prediction, target, mask = data["mean"], data["target"], data["mask"]
	scales, biases = [], []
	for channel in range(prediction.shape[1]):
		valid = mask[:, channel, :]
		x = prediction[:, channel, :][valid].double()
		y = target[:, channel, :][valid].double()
		if affine:
			x_centered = x - x.mean()
			y_centered = y - y.mean()
			scale = (x_centered * y_centered).sum() / x_centered.square().sum().clamp_min(1e-12)
			bias = y.mean() - scale * x.mean()
		else:
			scale = (x * y).sum() / x.square().sum().clamp_min(1e-12)
			bias = torch.zeros((), dtype=torch.double)
		scales.append(scale.float())
		biases.append(bias.float())
	return torch.stack(scales), torch.stack(biases)


def evaluate(data: Dict[str, torch.Tensor], scale: torch.Tensor, bias: torch.Tensor) -> tuple[float, float]:
	prediction = data["mean"] * scale.view(1, -1, 1) + bias.view(1, -1, 1)
	regression = regression_metrics(prediction, data["target"], data["mask"])
	torque = torque_replay_metrics(
		prediction,
		data["target"],
		torch.full_like(prediction, -4.0),
		torch.zeros(prediction.shape[0], 1, prediction.shape[-1]),
		data["mask"],
		uncertainty_ref=1.0,
		gate_softness=1e6,
		deadband=1e6,
	)
	return regression["rmse"], torque["baseline_retained_aligned_torque"]


def fit_logvar_offset(data: Dict[str, torch.Tensor], per_channel: bool) -> torch.Tensor:
	"""Fit the Gaussian NLL-optimal variance multiplier on validation data."""
	normalized_squared_error = (data["target"] - data["mean"]).square() * torch.exp(-data["logvar"])
	if not per_channel:
		value = normalized_squared_error[data["mask"]].double().mean().clamp_min(1e-8)
		return value.log().float().view(1)
	offsets = []
	for channel in range(data["mean"].shape[1]):
		valid = data["mask"][:, channel, :]
		value = normalized_squared_error[:, channel, :][valid].double().mean().clamp_min(1e-8)
		offsets.append(value.log().float())
	return torch.stack(offsets)


def apply_logvar_offset(logvar: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
	if offset.numel() == 1:
		return logvar + offset.item()
	return logvar + offset.view(1, -1, 1)


def variance_metrics(data: Dict[str, torch.Tensor], offset: torch.Tensor) -> Dict[str, float]:
	calibrated_logvar = apply_logvar_offset(data["logvar"], offset)
	result = {
		"nll": float(masked_gaussian_nll(data["mean"], calibrated_logvar, data["target"], data["mask"])),
	}
	result.update(interval_calibration(data["mean"], calibrated_logvar, data["target"], data["mask"]))
	return result


def mean_finite(values: Iterable[float]) -> float:
	finite = [float(value) for value in values if math.isfinite(float(value))]
	return sum(finite) / len(finite) if finite else float("inf")


def build_variance_calibrators(fit_sets: Dict[str, Dict[str, torch.Tensor]], channels: int) -> list[dict]:
	calibrators = [{"method": "identity", "fit": "none", "offset": torch.zeros(1)}]
	for fit_name, fit_data in fit_sets.items():
		calibrators.append(
			{
				"method": "global_temperature",
				"fit": fit_name,
				"offset": fit_logvar_offset(fit_data, per_channel=False),
			}
		)
		calibrators.append(
			{
				"method": "per_channel_temperature",
				"fit": fit_name,
				"offset": fit_logvar_offset(fit_data, per_channel=True),
			}
		)
	for calibrator in calibrators:
		if calibrator["offset"].numel() not in (1, channels):
			raise RuntimeError("Variance calibrator has an incompatible channel count.")
	return calibrators


def main() -> None:
	cli = parse_cli()
	report = json.loads(Path(cli.report).read_text(encoding="utf-8"))
	args = experiment_args(report, cli)

	input_names = input_names_for_profile(args.input_profile, args.side)
	target_names = label_names(args.side)
	groups = feature_groups(input_names)
	records = discover_trials(args.data_root)
	heldout = R._parse_csv(args.heldout_tasks)
	_, val_participants, test_participants = R.resolve_split(records, args)
	split_records = {
		"val": filter_records(records, participants=val_participants, exclude_tasks=heldout),
		"val_ood": filter_records(records, participants=val_participants, include_tasks=heldout),
		"test_id": filter_records(records, participants=test_participants, exclude_tasks=heldout),
		"test_ood": filter_records(records, participants=test_participants, include_tasks=heldout),
	}
	datasets = {
		name: R.make_dataset(records_for_split, input_names, target_names, args)
		for name, records_for_split in split_records.items()
		if records_for_split
	}

	device = torch.device(args.device)
	model, _ = load_checkpoint(cli.checkpoint, map_location=device)
	model = model.to(device).eval()
	print(f"checkpoint={cli.checkpoint} device={device} labels={target_names}")

	outputs = {}
	for name in ("val", "val_ood", "test_id", "test_ood"):
		if name in datasets:
			print(f"collecting {name}: windows={len(datasets[name])}")
			outputs[name] = collect_predictions(model, datasets[name], args, device)

	fit_sets = {
		"val": outputs["val"],
		"val_ood": outputs["val_ood"],
		"val_all": concatenate([outputs["val"], outputs["val_ood"]]),
	}

	variance_calibrators = build_variance_calibrators(fit_sets, len(target_names))
	variance_rows = []
	for calibrator in variance_calibrators:
		metrics = {
			split_name: variance_metrics(split_data, calibrator["offset"])
			for split_name, split_data in outputs.items()
		}
		selection_ece = mean_finite([metrics["val"]["coverage_ece"], metrics["val_ood"]["coverage_ece"]])
		variance_rows.append(
			{
				"method": calibrator["method"],
				"fit": calibrator["fit"],
				"logvar_offset": calibrator["offset"].tolist(),
				"sigma_scale": torch.exp(0.5 * calibrator["offset"]).tolist(),
				"selection_ece": selection_ece,
				"metrics": metrics,
			}
		)
	selected_variance = min(variance_rows, key=lambda row: (row["selection_ece"], row["method"], row["fit"]))

	print("\nVariance temperature scaling (selection uses val + val_ood ECE only)")
	print("method                    fit       sigma_scale          val ECE   valOOD ECE   testID ECE   testOOD ECE   testID NLL   testOOD NLL")
	for row in variance_rows:
		metrics = row["metrics"]
		marker = "*" if row is selected_variance else " "
		print(
			f"{marker}{row['method']:<25} {row['fit']:<9} "
			f"{str([round(value, 4) for value in row['sigma_scale']]):<20} "
			f"{metrics['val']['coverage_ece']:.4f}    {metrics['val_ood']['coverage_ece']:.4f}       "
			f"{metrics['test_id']['coverage_ece']:.4f}        {metrics['test_ood']['coverage_ece']:.4f}        "
			f"{metrics['test_id']['nll']:.4f}       {metrics['test_ood']['nll']:.4f}"
		)

	print("\nmethod       fit       scale                 bias                  test_id rmse/retain  test_ood rmse/retain")
	identity_scale = torch.ones(len(target_names))
	identity_bias = torch.zeros(len(target_names))
	calibrators = [("identity", "none", identity_scale, identity_bias)]
	for fit_name, fit_data in fit_sets.items():
		for affine in (False, True):
			scale, bias = fit_calibrator(fit_data, affine=affine)
			calibrators.append(("affine" if affine else "scale", fit_name, scale, bias))
	for method, fit_name, scale, bias in calibrators:
		id_rmse, id_retained = evaluate(outputs["test_id"], scale, bias)
		ood_rmse, ood_retained = evaluate(outputs["test_ood"], scale, bias)
		print(
			f"{method:<12} {fit_name:<9} "
			f"{str([round(v, 4) for v in scale.tolist()]):<21} "
			f"{str([round(v, 4) for v in bias.tolist()]):<21} "
			f"{id_rmse:.4f}/{id_retained:.4f}       {ood_rmse:.4f}/{ood_retained:.4f}"
		)

	output_dir = Path(cli.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	result = {
		"source_report": cli.report,
		"source_checkpoint": cli.checkpoint,
		"selection_source": "mean_coverage_ece_on_val_and_val_ood",
		"selected_variance_calibrator": {
			key: value for key, value in selected_variance.items() if key != "metrics"
		},
		"variance_calibrators": variance_rows,
	}
	output_path = output_dir / "output_calibration_diagnostic.json"
	output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
	print(f"\nWrote {output_path}")


if __name__ == "__main__":
	main()
