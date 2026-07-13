"""Diagnose prefix-safe cross-modal detector candidates for sensor delay.

All lag/coherence statistics are computed after ``eval_ignore_history`` plus a
guard equal to the largest injected delay.  Clean validation data calibrates
feature centers, scales, score direction, and the operating threshold.  Fixed
delay and jittered-delay validation faults jointly select one candidate before
the policy is frozen for ID/OOD test data.  Full-window lag features are kept
as an offline upper bound; only ``causal_*`` rolling features are eligible for
integration into real-time gating.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_reliability_experiment as R
from reliability.faults import apply_fault, parse_fault_spec
from reliability.features import feature_groups, label_names
from reliability.model import load_checkpoint
from scripts.diag_imu_bias import (
	auroc,
	build_datasets,
	evaluate_policies,
	finite_mean,
	finite_min,
	parse_csv,
	parse_floats,
	resolve_checkpoint,
	restore_report_args,
	sha256_file,
)


DEFAULT_REPORT = "reports/v2_main_seed7/reliability_report.json"
DEFAULT_OUTPUT_DIR = "reports/diagnostics/sensor_delay_seed7"
DELAY_MULTIPLIERS = {
	"foot_imu": 1.0,
	"shank_imu": 1.6,
	"thigh_imu": 0.6,
	"insole": 2.0,
	"encoder": 1.2,
}


def parse_cli() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--report", default=DEFAULT_REPORT)
	parser.add_argument("--checkpoint", default="")
	parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--num-workers", type=int, default=8)
	parser.add_argument("--max-eval-batches", type=int, default=0)
	parser.add_argument("--splits", default="val,val_ood,test_id,test_ood")
	parser.add_argument("--delay-steps", default="5,10,20")
	parser.add_argument("--selection-conditions", default="canonical@10,jitter@10")
	parser.add_argument("--max-lag", type=int, default=64)
	parser.add_argument("--causal-windows", default="32,64,128,256")
	parser.add_argument("--clean-quantile", type=float, default=0.95)
	parser.add_argument("--seed", type=int, default=7)
	parser.add_argument("--log-interval", type=int, default=2)
	return parser.parse_args()


def condition_name(mode: str, delay_steps: float) -> str:
	return f"{mode}@{delay_steps:g}"


def standardize(values: torch.Tensor) -> torch.Tensor:
	centered = values - values.mean(dim=-1, keepdim=True)
	scale = centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-4)
	return centered / scale


def lagged_correlation(
	left: torch.Tensor,
	right: torch.Tensor,
	max_lag: int,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return normalized cross-correlation and lags for paired channels."""
	if left.shape != right.shape:
		raise ValueError(f"Paired signals must have the same shape: {left.shape} != {right.shape}")
	time_steps = left.shape[-1]
	max_lag = min(max(int(max_lag), 1), max(time_steps // 3, 1))
	nfft = 1 << (2 * time_steps - 1).bit_length()
	left_fft = torch.fft.rfft(standardize(left), n=nfft, dim=-1)
	right_fft = torch.fft.rfft(standardize(right), n=nfft, dim=-1)
	circular = torch.fft.irfft(left_fft * torch.conj(right_fft), n=nfft, dim=-1)
	negative = circular[..., nfft - max_lag :]
	positive = circular[..., : max_lag + 1]
	lags = torch.arange(-max_lag, max_lag + 1, device=left.device)
	overlap = (time_steps - lags.abs()).to(left.dtype).clamp_min(1.0)
	correlation = torch.cat([negative, positive], dim=-1) / overlap.view(1, 1, -1)
	return correlation, lags


def pair_features(
	left: torch.Tensor,
	right: torch.Tensor,
	max_lag: int,
	expected_pattern: torch.Tensor,
) -> Dict[str, torch.Tensor]:
	correlation, lags = lagged_correlation(left, right, max_lag)
	abs_correlation = correlation.abs()
	peak, peak_index = abs_correlation.max(dim=-1)
	best_lag = lags[peak_index].to(left.dtype)
	zero = abs_correlation[..., lags.numel() // 2]
	gain = peak - zero
	pattern = expected_pattern.to(device=left.device, dtype=left.dtype).view(1, -1)
	projection = (best_lag * pattern).sum(dim=1) / pattern.square().sum(dim=1).clamp_min(1e-6)
	return {
		"lag": best_lag,
		"zero": zero,
		"gain": gain,
		"peak": peak,
		"pattern_projection": projection,
	}


def imu_pairs(z: torch.Tensor, groups: Dict[str, List[int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	pairs = (("foot_imu", "shank_imu"), ("foot_imu", "thigh_imu"), ("shank_imu", "thigh_imu"))
	left_parts, right_parts, pattern = [], [], []
	for left_name, right_name in pairs:
		left_indices = groups.get(left_name, [])
		right_indices = groups.get(right_name, [])
		if not left_indices or len(left_indices) != len(right_indices):
			continue
		left_parts.append(z[:, left_indices, :])
		right_parts.append(z[:, right_indices, :])
		pattern.extend([DELAY_MULTIPLIERS[left_name] - DELAY_MULTIPLIERS[right_name]] * len(left_indices))
	if not left_parts:
		raise RuntimeError("No corresponding IMU channel pairs are available.")
	return torch.cat(left_parts, dim=1), torch.cat(right_parts, dim=1), torch.tensor(pattern)


def activity_pairs(z: torch.Tensor, groups: Dict[str, List[int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	group_names = [name for name in DELAY_MULTIPLIERS if groups.get(name)]
	activities = {}
	for name in group_names:
		indices = groups[name]
		difference = z[:, indices, 1:] - z[:, indices, :-1]
		activity = difference.square().mean(dim=1, keepdim=True).clamp_min(1e-8).sqrt()
		activities[name] = F.avg_pool1d(activity, kernel_size=9, stride=1, padding=4)
	left_parts, right_parts, pattern = [], [], []
	for left_name, right_name in combinations(group_names, 2):
		left_parts.append(activities[left_name])
		right_parts.append(activities[right_name])
		pattern.append(DELAY_MULTIPLIERS[left_name] - DELAY_MULTIPLIERS[right_name])
	return torch.cat(left_parts, dim=1), torch.cat(right_parts, dim=1), torch.tensor(pattern)


def causal_difference_coherence(
	input_norm: torch.Tensor,
	groups: Dict[str, List[int]],
	window: int,
) -> torch.Tensor:
	"""Causal rolling zero-lag correlation loss for corresponding IMU axes."""
	left, right, _ = imu_pairs(input_norm, groups)
	left = torch.cat([torch.zeros_like(left[..., :1]), left[..., 1:] - left[..., :-1]], dim=-1)
	right = torch.cat([torch.zeros_like(right[..., :1]), right[..., 1:] - right[..., :-1]], dim=-1)
	window = max(int(window), 2)

	def rolling_mean(values: torch.Tensor) -> torch.Tensor:
		return F.avg_pool1d(F.pad(values, (window - 1, 0)), kernel_size=window, stride=1)

	left_mean = rolling_mean(left)
	right_mean = rolling_mean(right)
	covariance = rolling_mean(left * right) - left_mean * right_mean
	left_variance = (rolling_mean(left.square()) - left_mean.square()).clamp_min(1e-8)
	right_variance = (rolling_mean(right.square()) - right_mean.square()).clamp_min(1e-8)
	correlation = covariance / (left_variance * right_variance).sqrt()
	return 1.0 - correlation.abs().clamp_max(1.0).mean(dim=1, keepdim=True)


def temporal_features(
	input_norm: torch.Tensor,
	groups: Dict[str, List[int]],
	start: int,
	max_lag: int,
	causal_windows: Sequence[int],
) -> Dict[str, torch.Tensor]:
	start = min(max(int(start), 0), max(input_norm.shape[-1] - 16, 0))
	z = input_norm[..., start:]
	imu_left, imu_right, imu_pattern = imu_pairs(z, groups)
	imu_raw = pair_features(imu_left, imu_right, max_lag, imu_pattern)
	imu_diff = pair_features(
		imu_left[..., 1:] - imu_left[..., :-1],
		imu_right[..., 1:] - imu_right[..., :-1],
		max_lag,
		imu_pattern,
	)
	activity_left, activity_right, activity_pattern = activity_pairs(z, groups)
	activity = pair_features(activity_left, activity_right, max_lag, activity_pattern)

	features: Dict[str, torch.Tensor] = {}
	for family, values in (("imu_raw", imu_raw), ("imu_diff", imu_diff), ("activity", activity)):
		lag = values["lag"]
		zero = values["zero"]
		gain = values["gain"]
		features[f"{family}_lag_abs_mean"] = lag.abs().mean(dim=1)
		features[f"{family}_lag_dispersion"] = lag.std(dim=1, unbiased=False)
		features[f"{family}_zero_corr_loss"] = 1.0 - zero.mean(dim=1)
		features[f"{family}_lag_gain_mean"] = gain.mean(dim=1)
		features[f"{family}_pattern_projection_abs"] = values["pattern_projection"].abs()
		features[f"_{family}_lag_vector"] = lag
		features[f"_{family}_zero_vector"] = zero
		features[f"_{family}_gain_vector"] = gain
		features[f"_{family}_lag_zero_vector"] = torch.cat([lag, zero], dim=1)
	for window in causal_windows:
		coherence = causal_difference_coherence(input_norm, groups, window)
		causal_start = max(start, window - 1)
		features[f"causal_imu_diff_coherence_w{window}"] = coherence[..., causal_start:].mean(dim=(1, 2))
	return features


def max_delay_guard(delay_steps: Sequence[float]) -> int:
	# Jitter samples up to 1.5x each modality's nominal delay.
	return int(math.ceil(max(delay_steps) * max(DELAY_MULTIPLIERS.values()) * 1.5))


def make_generator(device: torch.device, seed: int) -> torch.Generator:
	generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
	generator.manual_seed(seed)
	return generator


def collect_raw_split(
	model: torch.nn.Module,
	dataset,
	groups: Dict[str, List[int]],
	args: argparse.Namespace,
	device: torch.device,
	delay_steps: Sequence[float],
	feature_start: int,
	max_lag: int,
	causal_windows: Sequence[int],
	seed: int,
	log_interval: int,
) -> dict:
	loader = R.make_loader(dataset, args, device, shuffle=False)
	clean_parts: Dict[str, List[torch.Tensor]] = {}
	fault_parts = {
		condition_name(mode, steps): {}
		for steps in delay_steps
		for mode in ("canonical", "jitter")
	}
	metadata = {"participant": [], "trial": [], "task": [], "start": []}
	model.eval()
	with torch.no_grad():
		for batch_index, batch in enumerate(loader):
			if args.max_eval_batches and batch_index >= args.max_eval_batches:
				break
			x = batch["x"].to(device, non_blocking=device.type == "cuda")
			_append(clean_parts, temporal_features(model.normalize(x), groups, feature_start, max_lag, causal_windows))
			for steps in delay_steps:
				for mode in ("canonical", "jitter"):
					fault_name = "sensor_delay" if mode == "canonical" else "sensor_delay_jitter"
					generator = make_generator(device, seed + 1009 * batch_index + int(steps * 7919))
					corrupted, _ = apply_fault(x, groups, parse_fault_spec(f"{fault_name}@{steps:g}"), generator)
					features = temporal_features(
						model.normalize(corrupted),
						groups,
						feature_start,
						max_lag,
						causal_windows,
					)
					_append(fault_parts[condition_name(mode, steps)], features)
			metadata["participant"].extend(list(batch["participant"]))
			metadata["trial"].extend(list(batch["trial"]))
			metadata["task"].extend(list(batch["task"]))
			metadata["start"].extend(int(value) for value in batch["start"])
			if log_interval > 0 and (batch_index + 1) % log_interval == 0:
				print(f"  batches={batch_index + 1}/{len(loader)} windows={len(metadata['participant'])}", flush=True)
	return {
		"clean": _concat(clean_parts),
		"faults": {name: _concat(parts) for name, parts in fault_parts.items()},
		"metadata": metadata,
	}


def _append(parts: Dict[str, List[torch.Tensor]], values: Dict[str, torch.Tensor]) -> None:
	for name, value in values.items():
		parts.setdefault(name, []).append(value.detach().float().cpu())


def _concat(parts: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
	return {name: torch.cat(values) for name, values in parts.items() if values}


def calibrate_vector_distances(raw_scores: Dict[str, dict]) -> tuple[Dict[str, dict], dict]:
	validation_clean = raw_scores["val"]["clean"]
	vector_names = sorted(name for name, value in validation_clean.items() if name.startswith("_") and value.ndim == 2)
	calibrators = {}
	for name in vector_names:
		values = validation_clean[name]
		center = values.mean(dim=0)
		scale = values.std(dim=0, unbiased=False)
		if "lag_zero" in name:
			minimum_scale = torch.full_like(scale, 0.02)
			minimum_scale[: scale.numel() // 2] = 1.0
		elif "lag" in name:
			minimum_scale = torch.ones_like(scale)
		else:
			minimum_scale = torch.full_like(scale, 0.02)
		scale = torch.maximum(scale, minimum_scale)
		calibrators[name] = {"center": center, "scale": scale}

	calibrated = {}
	for split_name, split_scores in raw_scores.items():
		clean = _calibrate_score_set(split_scores["clean"], calibrators)
		faults = {
			condition: _calibrate_score_set(values, calibrators)
			for condition, values in split_scores["faults"].items()
		}
		calibrated[split_name] = {"clean": clean, "faults": faults, "metadata": split_scores["metadata"]}
	serializable = {
		name: {
			"center": values["center"].tolist(),
			"scale": values["scale"].tolist(),
		}
		for name, values in calibrators.items()
	}
	return calibrated, serializable


def _calibrate_score_set(values: Dict[str, torch.Tensor], calibrators: dict) -> Dict[str, torch.Tensor]:
	scores = {name: value for name, value in values.items() if not name.startswith("_")}
	for name, calibration in calibrators.items():
		z = (values[name] - calibration["center"]) / calibration["scale"]
		scores[f"{name[1:]}_zdist"] = z.square().mean(dim=1).sqrt()
	return scores


def select_policy(
	validation: dict,
	selection_conditions: Sequence[str],
	clean_quantile: float,
) -> tuple[str, Dict[str, dict]]:
	for condition in selection_conditions:
		if condition not in validation["faults"]:
			raise ValueError(f"Selection condition '{condition}' was not evaluated.")
	policies = {}
	for candidate, clean in validation["clean"].items():
		fault_sets = [validation["faults"][condition][candidate] for condition in selection_conditions]
		combined_fault = torch.cat(fault_sets)
		raw_combined_auc = auroc(clean, combined_fault)
		polarity = 1.0 if raw_combined_auc >= 0.5 else -1.0
		condition_aurocs = {
			condition: auroc(clean * polarity, validation["faults"][condition][candidate] * polarity)
			for condition in selection_conditions
		}
		mean_auc = finite_mean(condition_aurocs.values())
		worst_auc = finite_min(condition_aurocs.values())
		selection_score = (mean_auc or 0.0) + 0.25 * (worst_auc or 0.0)
		policies[candidate] = {
			"polarity": polarity,
			"threshold": float(torch.quantile(clean * polarity, clean_quantile)),
			"validation_auc": raw_combined_auc if polarity > 0 else 1.0 - raw_combined_auc,
			"validation_condition_aurocs": condition_aurocs,
			"selection_score": selection_score,
		}
	eligible = [name for name in policies if name.startswith("causal_")]
	if not eligible:
		raise RuntimeError("No causal detector candidates were evaluated.")
	selected = max(eligible, key=lambda name: (policies[name]["selection_score"], -len(name), name))
	for name, policy in policies.items():
		policy["eligible_for_integration"] = name in eligible
	return selected, policies


def write_metrics(path: Path, rows: Sequence[dict]) -> None:
	fields = [
		"candidate",
		"split",
		"mode",
		"severity",
		"auc",
		"clean_fpr",
		"fault_tpr",
		"participant_auc_mean",
		"participant_auc_min",
		"task_auc_mean",
		"n_clean",
		"n_fault",
	]
	with path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
		writer.writeheader()
		writer.writerows(rows)


def print_summary(selected: str, rows: Sequence[dict]) -> None:
	print(f"\nSelected delay candidate: {selected}")
	print(f"{'split':<10} {'mode':<10} {'steps':>5} {'auc':>7} {'FPR':>7} {'TPR':>7} {'pAUC':>7}")
	for row in rows:
		if row["candidate"] != selected:
			continue
		participant_auc = row["participant_auc_mean"]
		participant_text = f"{participant_auc:.3f}" if participant_auc is not None else "-"
		print(
			f"{row['split']:<10} {row['mode']:<10} {row['severity']:>5.0f} "
			f"{row['auc']:>7.3f} {row['clean_fpr']:>7.3f} {row['fault_tpr']:>7.3f} {participant_text:>7}"
		)


def main() -> None:
	cli = parse_cli()
	report_path = Path(cli.report)
	report = json.loads(report_path.read_text(encoding="utf-8"))
	checkpoint = resolve_checkpoint(report_path, cli.checkpoint)
	args = restore_report_args(
		report,
		checkpoint,
		cli.device,
		cli.batch_size,
		cli.num_workers,
		cli.max_eval_batches,
	)
	split_names = parse_csv(cli.splits)
	if "val" not in split_names:
		raise ValueError("--splits must include val.")
	delay_steps = sorted(set(parse_floats(cli.delay_steps)))
	causal_windows = sorted(set(int(value) for value in parse_floats(cli.causal_windows)))
	selection_conditions = parse_csv(cli.selection_conditions)
	if not delay_steps or any(steps <= 0 for steps in delay_steps):
		raise ValueError("--delay-steps must contain positive values.")
	if not causal_windows or any(window < 2 for window in causal_windows):
		raise ValueError("--causal-windows must contain integers >= 2.")
	if not 0.5 < cli.clean_quantile < 1.0:
		raise ValueError("--clean-quantile must be between 0.5 and 1.0.")

	input_names, _, datasets = build_datasets(report, args, split_names)
	groups = feature_groups(input_names)
	device = torch.device(args.device)
	model, payload = load_checkpoint(checkpoint, map_location=device)
	R.validate_checkpoint_inputs(payload, input_names, label_names(args.side))
	model = model.to(device).eval()
	guard = max_delay_guard(delay_steps)
	feature_start = args.eval_ignore_history + guard
	if feature_start + 2 * cli.max_lag >= args.window_size:
		raise ValueError("Delay guard and --max-lag leave too little usable history.")
	print(
		f"Loaded report={report_path} checkpoint={checkpoint} device={device} "
		f"eval_ignore_history={args.eval_ignore_history} delay_guard={guard} "
		f"feature_start={feature_start} max_lag={cli.max_lag}",
		flush=True,
	)

	raw_scores = {}
	for split_name, dataset in datasets.items():
		print(f"\n[{split_name}] windows={len(dataset)}", flush=True)
		raw_scores[split_name] = collect_raw_split(
			model,
			dataset,
			groups,
			args,
			device,
			delay_steps,
			feature_start,
			cli.max_lag,
			causal_windows,
			cli.seed,
			cli.log_interval,
		)

	scores, vector_calibration = calibrate_vector_distances(raw_scores)
	selected, policies = select_policy(scores["val"], selection_conditions, cli.clean_quantile)
	rows = evaluate_policies(scores, policies)
	print_summary(selected, rows)

	output_dir = Path(cli.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	metrics_path = output_dir / "sensor_delay_candidate_metrics.tsv"
	json_path = output_dir / "sensor_delay_diagnostic.json"
	write_metrics(metrics_path, rows)
	result = {
		"source_report": str(report_path),
		"source_checkpoint": str(checkpoint),
		"source_checkpoint_sha256": sha256_file(checkpoint),
		"restored_data_args": {
			"data_root": args.data_root,
			"heldout_tasks": args.heldout_tasks,
			"window_size": args.window_size,
			"stride": args.stride,
			"max_windows_per_trial": args.max_windows_per_trial,
			"eval_ignore_history": args.eval_ignore_history,
		},
		"run_args": {
			"device": str(device),
			"batch_size": args.batch_size,
			"num_workers": args.num_workers,
			"max_eval_batches": args.max_eval_batches,
			"splits": split_names,
			"delay_steps": delay_steps,
			"selection_conditions": selection_conditions,
			"max_lag": cli.max_lag,
			"causal_windows": causal_windows,
			"delay_guard": guard,
			"feature_start": feature_start,
			"clean_quantile": cli.clean_quantile,
			"seed": cli.seed,
		},
		"selection_source": [f"val/{condition}" for condition in selection_conditions],
		"selected_candidate": selected,
		"candidate_policies": policies,
		"vector_calibration": vector_calibration,
		"metrics": rows,
	}
	json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
	print(f"\nWrote {metrics_path}")
	print(f"Wrote {json_path}")


if __name__ == "__main__":
	main()
