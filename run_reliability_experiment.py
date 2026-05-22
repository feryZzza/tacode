"""Train and evaluate reliability-aware moment estimation on Nature 2024 data.

This script is intentionally separate from the original reproduction files. It
adds the paper-route components we discussed: probabilistic moment estimation,
synthetic sensor-fault supervision, OOD splits, and offline torque-risk replay.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from reliability.faults import apply_fault, parse_fault_spec, random_training_fault
from reliability.features import feature_groups, input_names_for_profile, label_names, participant_masses
from reliability.metrics import (
	binary_auroc,
	interval_calibration,
	masked_gaussian_nll,
	regression_metrics,
	torque_replay_metrics,
	uncertainty_error_correlation,
)
from reliability.model import ReliabilityTCN, save_checkpoint
from reliability.nature_dataset import (
	ParsedWindowDataset,
	discover_trials,
	filter_records,
	participant_split,
	summarize_records,
)
from tcn import TCN


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-root", default="/home/zfy/dataset/tcn/Parsed")
	parser.add_argument("--output-dir", default="reports/reliability")
	parser.add_argument("--checkpoint", default="")
	parser.add_argument("--nature-baseline-checkpoint", default="models/trained_tcn.tar")
	parser.add_argument("--mode", choices=["train", "eval", "smoke"], default="smoke")
	parser.add_argument("--device", default="cpu")
	parser.add_argument("--side", choices=["r", "l"], default="r")
	parser.add_argument("--input-profile", default="human", choices=["human", "human_desired", "human_measured", "human_execution", "human_interaction", "full"])
	parser.add_argument("--heldout-tasks", default="jump,cutting,lift_weight,lunges")
	parser.add_argument("--val-count", type=int, default=2)
	parser.add_argument("--test-count", type=int, default=2)
	parser.add_argument("--window-size", type=int, default=768)
	parser.add_argument("--stride", type=int, default=512)
	parser.add_argument("--min-valid-fraction", type=float, default=0.5)
	parser.add_argument("--max-windows-per-trial", type=int, default=2)
	parser.add_argument("--limit-trials", type=int, default=0, help="Per split. 0 means no limit.")
	parser.add_argument("--epochs", type=int, default=3)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--num-channels", default="32,32,32,32")
	parser.add_argument("--kernel-size", type=int, default=4)
	parser.add_argument("--dropout", type=float, default=0.1)
	parser.add_argument("--fault-loss-weight", type=float, default=0.1)
	parser.add_argument("--stats-batches", type=int, default=24)
	parser.add_argument("--max-eval-batches", type=int, default=0)
	parser.add_argument("--fault-scenarios", default="clean,insole_missing,encoder_dropout,imu_bias,packet_loss,sensor_delay")
	parser.add_argument("--seed", type=int, default=7)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.mode == "smoke":
		args.epochs = 1
		args.limit_trials = args.limit_trials or 8
		args.max_windows_per_trial = 1
		args.max_eval_batches = args.max_eval_batches or 2

	set_seed(args.seed)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	input_names = input_names_for_profile(args.input_profile, args.side)
	target_names = label_names(args.side)
	groups = feature_groups(input_names)
	records = discover_trials(args.data_root)
	heldout_tasks = _parse_csv(args.heldout_tasks)
	train_participants, val_participants, test_participants = participant_split(records, args.val_count, args.test_count)

	splits = {
		"train": filter_records(records, participants=train_participants, exclude_tasks=heldout_tasks),
		"val": filter_records(records, participants=val_participants, exclude_tasks=heldout_tasks),
		"test_id": filter_records(records, participants=test_participants, exclude_tasks=heldout_tasks),
		"test_ood": filter_records(records, participants=test_participants, include_tasks=heldout_tasks),
	}
	if args.limit_trials:
		splits = {name: split[: args.limit_trials] for name, split in splits.items()}

	print(json.dumps({name: summarize_records(split) for name, split in splits.items()}, indent=2))

	datasets = {}
	for name, split in splits.items():
		if not split:
			continue
		dataset = make_dataset(split, input_names, target_names, args)
		if len(dataset) > 0:
			datasets[name] = dataset
		else:
			print(f"Warning: split '{name}' has no windows after valid-label filtering.")
	if "train" not in datasets or len(datasets["train"]) == 0:
		raise RuntimeError("Training split has no windows. Check heldout tasks or dataset root.")

	device = torch.device(args.device)
	model = build_model(args, len(input_names), len(target_names), device)
	if args.checkpoint:
		payload = torch.load(args.checkpoint, map_location=device)
		model.load_state_dict(payload["state_dict"])
		print(f"Loaded checkpoint: {args.checkpoint}")
	else:
		center, scale = estimate_normalization(datasets["train"], args.batch_size, args.stats_batches, device)
		model.set_normalization(center, scale)

	history: List[Dict[str, float]] = []
	if args.mode in ("train", "smoke"):
		history = train(model, datasets["train"], datasets.get("val"), groups, args, device)
		checkpoint_path = output_dir / "reliability_tcn_best.pt"
		save_checkpoint(
			str(checkpoint_path),
			model,
			{
				"input_names": input_names,
				"label_names": target_names,
				"input_profile": args.input_profile,
				"num_channels": _parse_channels(args.num_channels),
				"ksize": args.kernel_size,
				"dropout": args.dropout,
				"history": history,
				"data_root": args.data_root,
				"heldout_tasks": heldout_tasks,
			},
		)
		print(f"Saved checkpoint: {checkpoint_path}")

	results = evaluate_all(model, datasets, groups, args, device)
	if args.input_profile == "human" and args.nature_baseline_checkpoint:
		baseline_path = Path(args.nature_baseline_checkpoint)
		if baseline_path.exists():
			results.update(evaluate_nature_baseline(str(baseline_path), datasets, groups, args, device))
		else:
			print(f"Warning: Nature baseline checkpoint not found: {baseline_path}")
	report = {
		"args": vars(args),
		"input_names": input_names,
		"label_names": target_names,
		"splits": {name: summarize_records(split) for name, split in splits.items()},
		"history": history,
		"results": results,
	}
	report_path = output_dir / "reliability_report.json"
	report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
	print(json.dumps(results, indent=2))
	print(f"Wrote report: {report_path}")


def make_dataset(records, input_names, target_names, args) -> ParsedWindowDataset:
	return ParsedWindowDataset(
		records,
		input_names=input_names,
		label_names=target_names,
		side=args.side,
		participant_masses=participant_masses,
		window_size=args.window_size,
		stride=args.stride,
		max_windows_per_trial=args.max_windows_per_trial,
		min_valid_fraction=args.min_valid_fraction,
	)


def build_model(args, input_size: int, output_size: int, device: torch.device) -> ReliabilityTCN:
	return ReliabilityTCN(
		input_size=input_size,
		output_size=output_size,
		num_channels=_parse_channels(args.num_channels),
		ksize=args.kernel_size,
		dropout=args.dropout,
	).to(device)


def train(
	model: ReliabilityTCN,
	train_dataset: ParsedWindowDataset,
	val_dataset: Optional[ParsedWindowDataset],
	groups: Dict[str, List[int]],
	args: argparse.Namespace,
	device: torch.device,
) -> List[Dict[str, float]]:
	loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
	history = []
	best_score = math.inf
	best_state = None
	for epoch in range(1, args.epochs + 1):
		model.train()
		running = []
		for batch in loader:
			x = batch["x"].to(device)
			y = batch["y"].to(device)
			mask = batch["mask"].to(device)
			spec = random_training_fault()
			x_fault, fault_target = apply_fault(x, groups, spec)
			out = model(x_fault)
			nll = masked_gaussian_nll(out["mean"], out["logvar"], y, mask)
			fault_loss = F.binary_cross_entropy_with_logits(out["fault_logit"], fault_target)
			loss = nll + args.fault_loss_weight * fault_loss
			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
			optimizer.step()
			running.append(float(loss.detach().cpu()))

		epoch_info = {
			"epoch": epoch,
			"train_loss": sum(running) / max(len(running), 1),
		}
		if val_dataset is not None and len(val_dataset) > 0:
			val_metrics = evaluate_dataset(model, val_dataset, groups, args, device, fault_name="clean")
			epoch_info.update({f"val_{key}": value for key, value in val_metrics.items() if isinstance(value, float)})
			rmse = val_metrics.get("rmse", math.inf)
			score = rmse + 0.02 * val_metrics.get("coverage_ece", 0.0) if math.isfinite(rmse) else math.inf
			if score < best_score:
				best_score = score
				best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
		print(json.dumps(epoch_info, indent=2))
		history.append(epoch_info)

	if best_state is not None:
		model.load_state_dict(best_state)
	return history


def evaluate_all(
	model: ReliabilityTCN,
	datasets: Dict[str, ParsedWindowDataset],
	groups: Dict[str, List[int]],
	args: argparse.Namespace,
	device: torch.device,
) -> Dict[str, Dict[str, float]]:
	results: Dict[str, Dict[str, float]] = {}
	ref_dataset = datasets.get("val", datasets.get("train"))
	uncertainty_ref = collect_uncertainty_ref(model, ref_dataset, groups, args, device) if ref_dataset is not None else 1.0
	for split_name, dataset in datasets.items():
		for fault_name in _parse_csv(args.fault_scenarios):
			if split_name == "train" and fault_name != "clean":
				continue
			key = f"{split_name}/{fault_name}"
			results[key] = evaluate_dataset(
				model,
				dataset,
				groups,
				args,
				device,
				fault_name=fault_name,
				uncertainty_ref=uncertainty_ref,
			)
		for fault_name in _parse_csv(args.fault_scenarios):
			if fault_name == "clean":
				continue
			clean_scores = collect_fault_scores(model, dataset, groups, args, device, "clean")
			fault_scores = collect_fault_scores(model, dataset, groups, args, device, fault_name)
			if clean_scores.numel() and fault_scores.numel():
				scores = torch.cat([clean_scores, fault_scores])
				labels = torch.cat([torch.zeros_like(clean_scores), torch.ones_like(fault_scores)])
				results[f"fault_detection/{split_name}/{fault_name}"] = {"fault_auroc": binary_auroc(scores, labels)}

	if "test_id" in datasets and "test_ood" in datasets and len(datasets["test_ood"]) > 0:
		id_scores = collect_risk_scores(model, datasets["test_id"], groups, args, device, "clean")
		ood_scores = collect_risk_scores(model, datasets["test_ood"], groups, args, device, "clean")
		scores = torch.cat([id_scores, ood_scores])
		labels = torch.cat([torch.zeros_like(id_scores), torch.ones_like(ood_scores)])
		results["ood_detection/clean"] = {"risk_auroc": binary_auroc(scores, labels)}
	return results


def evaluate_nature_baseline(
	checkpoint_path: str,
	datasets: Dict[str, ParsedWindowDataset],
	groups: Dict[str, List[int]],
	args: argparse.Namespace,
	device: torch.device,
) -> Dict[str, Dict[str, float]]:
	model = load_nature_tcn(checkpoint_path, device)
	model.eval()
	results = {}
	for split_name, dataset in datasets.items():
		if split_name == "train":
			continue
		for fault_name in _parse_csv(args.fault_scenarios):
			key = f"nature_baseline/{split_name}/{fault_name}"
			results[key] = evaluate_point_estimator(
				model,
				dataset,
				groups,
				args,
				device,
				fault_name=fault_name,
				ignore_history=model.get_effective_history(),
			)
	return results


def load_nature_tcn(checkpoint_path: str, device: torch.device) -> TCN:
	payload = torch.load(checkpoint_path, map_location=device)
	state_dict = payload["state_dict"]
	model_info = {key: value for key, value in payload.items() if key != "state_dict"}
	model = TCN(**model_info).to(device)
	model.load_state_dict(state_dict)
	return model


def evaluate_point_estimator(
	model: torch.nn.Module,
	dataset: ParsedWindowDataset,
	groups: Dict[str, List[int]],
	args: argparse.Namespace,
	device: torch.device,
	fault_name: str = "clean",
	ignore_history: int = 0,
) -> Dict[str, float]:
	loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
	spec = parse_fault_spec(fault_name)
	means, ys, masks = [], [], []
	with torch.no_grad():
		for batch_index, batch in enumerate(loader):
			if args.max_eval_batches and batch_index >= args.max_eval_batches:
				break
			x = batch["x"].to(device)
			y = batch["y"].to(device)
			mask = batch["mask"].to(device)
			x_eval, _ = apply_fault(x, groups, spec)
			mean = model(x_eval)
			if ignore_history > 0:
				mask = mask.clone()
				usable_history = min(ignore_history, max(mask.shape[-1] - 1, 0))
				mask[:, :, :usable_history] = False
			means.append(mean.cpu())
			ys.append(y.cpu())
			masks.append(mask.cpu())
	if not means:
		return {}
	mean = torch.cat(means)
	y = torch.cat(ys)
	mask = torch.cat(masks).bool()
	metrics = regression_metrics(mean, y, mask)
	logvar = torch.zeros_like(mean) - 4.0
	fault_logit = torch.zeros((mean.shape[0], 1, mean.shape[-1]))
	metrics.update(torque_replay_metrics(mean, y, logvar, fault_logit, mask, uncertainty_ref=1.0))
	return metrics


def evaluate_dataset(
	model: ReliabilityTCN,
	dataset: ParsedWindowDataset,
	groups: Dict[str, List[int]],
	args: argparse.Namespace,
	device: torch.device,
	fault_name: str = "clean",
	uncertainty_ref: Optional[float] = None,
) -> Dict[str, float]:
	loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
	spec = parse_fault_spec(fault_name)
	model.eval()
	means, logvars, ys, masks, fault_logits, fault_targets = [], [], [], [], [], []
	with torch.no_grad():
		for batch_index, batch in enumerate(loader):
			if args.max_eval_batches and batch_index >= args.max_eval_batches:
				break
			x = batch["x"].to(device)
			y = batch["y"].to(device)
			mask = batch["mask"].to(device)
			x_eval, fault_target = apply_fault(x, groups, spec)
			out = model(x_eval)
			means.append(out["mean"].cpu())
			logvars.append(out["logvar"].cpu())
			ys.append(y.cpu())
			masks.append(mask.cpu())
			fault_logits.append(out["fault_logit"].cpu())
			fault_targets.append(fault_target.cpu())

	if not means:
		return {}

	mean = torch.cat(means)
	logvar = torch.cat(logvars)
	y = torch.cat(ys)
	mask = torch.cat(masks).bool()
	fault_logit = torch.cat(fault_logits)
	fault_target = torch.cat(fault_targets)
	metrics = regression_metrics(mean, y, mask)
	metrics["nll"] = float(masked_gaussian_nll(mean, logvar, y, mask).detach().cpu())
	metrics["uncertainty_error_corr"] = uncertainty_error_correlation(mean, logvar, y, mask)
	metrics.update(interval_calibration(mean, logvar, y, mask))
	metrics["fault_auroc"] = binary_auroc(torch.sigmoid(fault_logit), fault_target)
	if uncertainty_ref is None:
		uncertainty_ref = estimate_uncertainty_ref(logvar, mask)
	metrics.update(torque_replay_metrics(mean, y, logvar, fault_logit, mask, uncertainty_ref))
	return metrics


def collect_risk_scores(
	model: ReliabilityTCN,
	dataset: ParsedWindowDataset,
	groups: Dict[str, List[int]],
	args: argparse.Namespace,
	device: torch.device,
	fault_name: str,
) -> torch.Tensor:
	loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
	spec = parse_fault_spec(fault_name)
	model.eval()
	scores = []
	with torch.no_grad():
		for batch_index, batch in enumerate(loader):
			if args.max_eval_batches and batch_index >= args.max_eval_batches:
				break
			x = batch["x"].to(device)
			x_eval, _ = apply_fault(x, groups, spec)
			out = model(x_eval)
			sigma = torch.exp(0.5 * out["logvar"]).mean(dim=(1, 2))
			fault = torch.sigmoid(out["fault_logit"]).mean(dim=(1, 2))
			scores.append(torch.maximum(sigma, fault).cpu())
	return torch.cat(scores) if scores else torch.empty(0)


def collect_fault_scores(
	model: ReliabilityTCN,
	dataset: ParsedWindowDataset,
	groups: Dict[str, List[int]],
	args: argparse.Namespace,
	device: torch.device,
	fault_name: str,
) -> torch.Tensor:
	loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
	spec = parse_fault_spec(fault_name)
	model.eval()
	scores = []
	with torch.no_grad():
		for batch_index, batch in enumerate(loader):
			if args.max_eval_batches and batch_index >= args.max_eval_batches:
				break
			x = batch["x"].to(device)
			x_eval, _ = apply_fault(x, groups, spec)
			out = model(x_eval)
			scores.append(torch.sigmoid(out["fault_logit"]).mean(dim=(1, 2)).cpu())
	return torch.cat(scores) if scores else torch.empty(0)


def collect_uncertainty_ref(
	model: ReliabilityTCN,
	dataset: ParsedWindowDataset,
	groups: Dict[str, List[int]],
	args: argparse.Namespace,
	device: torch.device,
) -> float:
	loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
	model.eval()
	values = []
	with torch.no_grad():
		for batch_index, batch in enumerate(loader):
			if args.max_eval_batches and batch_index >= args.max_eval_batches:
				break
			x = batch["x"].to(device)
			mask = batch["mask"].to(device)
			out = model(x)
			sigma = torch.exp(0.5 * out["logvar"])
			values.append(sigma[mask].detach().cpu())
	if not values:
		return 1.0
	valid = torch.cat(values)
	if valid.numel() == 0:
		return 1.0
	return float(torch.quantile(valid, 0.75).clamp_min(1e-6))


def estimate_normalization(
	dataset: ParsedWindowDataset,
	batch_size: int,
	stats_batches: int,
	device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
	total = None
	total_sq = None
	count = 0
	for batch_index, batch in enumerate(loader):
		if batch_index >= stats_batches:
			break
		x = batch["x"].to(device)
		if total is None:
			total = torch.zeros(x.shape[1], device=device)
			total_sq = torch.zeros(x.shape[1], device=device)
		total += x.sum(dim=(0, 2))
		total_sq += x.pow(2).sum(dim=(0, 2))
		count += x.shape[0] * x.shape[2]
	if total is None or total_sq is None or count == 0:
		raise RuntimeError("Could not estimate normalization statistics.")
	mean = total / count
	var = (total_sq / count - mean.pow(2)).clamp_min(1e-6)
	return mean.detach(), torch.sqrt(var).detach()


def estimate_uncertainty_ref(logvar: torch.Tensor, mask: torch.Tensor) -> float:
	sigma = torch.exp(0.5 * logvar)
	valid = sigma[mask]
	if valid.numel() == 0:
		return 1.0
	return float(torch.quantile(valid, 0.75).clamp_min(1e-6))


def set_seed(seed: int) -> None:
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def _parse_csv(value: str) -> List[str]:
	return [item.strip() for item in value.split(",") if item.strip()]


def _parse_channels(value: str) -> List[int]:
	return [int(item) for item in _parse_csv(value)]


if __name__ == "__main__":
	main()
