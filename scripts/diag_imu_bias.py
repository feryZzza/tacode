"""Validation-calibrated diagnostic for IMU bias detector candidates.

The diagnostic restores the exact data settings from an existing reliability
report, selects score direction and threshold on validation data only, and
then freezes that policy for ID/OOD test splits.  Candidate statistics only
use samples after ``eval_ignore_history`` so the synthetic window prefix cannot
leak into the reported detector performance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_reliability_experiment as R
from reliability.features import feature_groups, input_names_for_profile, label_names
from reliability.model import load_checkpoint
from reliability.nature_dataset import discover_trials, filter_records


DEFAULT_REPORT = "reports/v2_main_seed7/reliability_report.json"
DEFAULT_OUTPUT_DIR = "reports/diagnostics/imu_bias_seed7"
FAULT_MODES = ("canonical", "negative", "random_sign")


def parse_cli() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--report", default=DEFAULT_REPORT)
	parser.add_argument("--checkpoint", default="", help="Defaults to reliability_tcn_best.pt beside --report.")
	parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--num-workers", type=int, default=8)
	parser.add_argument("--max-eval-batches", type=int, default=0, help="0 evaluates every window; use 1 for a smoke run.")
	parser.add_argument("--splits", default="val,val_ood,test_id,test_ood")
	parser.add_argument("--severities", default="0.10,0.25,0.50")
	parser.add_argument("--selection-severity", type=float, default=0.25)
	parser.add_argument("--clean-quantile", type=float, default=0.95)
	parser.add_argument("--seed", type=int, default=7, help="Seed for the random-sign robustness audit.")
	parser.add_argument("--log-interval", type=int, default=5)
	return parser.parse_args()


def restore_report_args(
	report: dict,
	checkpoint: Path,
	device: str,
	batch_size: int,
	num_workers: int,
	max_eval_batches: int,
) -> argparse.Namespace:
	saved_argv = sys.argv
	sys.argv = ["diag_imu_bias", "--checkpoint", str(checkpoint), "--mode", "eval"]
	try:
		args = R.parse_args()
	finally:
		sys.argv = saved_argv
	for key, value in report.get("args", {}).items():
		if hasattr(args, key):
			setattr(args, key, value)
	args.checkpoint = str(checkpoint)
	args.mode = "eval"
	args.device = device
	args.batch_size = batch_size
	args.num_workers = num_workers
	args.max_eval_batches = max_eval_batches
	args.mc_samples = 0
	return args


def resolve_checkpoint(report_path: Path, checkpoint_arg: str) -> Path:
	checkpoint = Path(checkpoint_arg) if checkpoint_arg else report_path.parent / "reliability_tcn_best.pt"
	if not checkpoint.is_file():
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
	return checkpoint


def sha256_file(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for block in iter(lambda: handle.read(1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()


def parse_csv(value: str) -> List[str]:
	return [item.strip() for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> List[float]:
	return [float(item.strip()) for item in value.split(",") if item.strip()]


def format_severity(value: float) -> str:
	return f"{value:g}"


def condition_name(mode: str, severity: float) -> str:
	return f"{mode}@{format_severity(severity)}"


def build_datasets(report: dict, args: argparse.Namespace, split_names: Sequence[str]):
	input_names = input_names_for_profile(args.input_profile, args.side)
	target_names = label_names(args.side)
	records = discover_trials(args.data_root)
	heldout = R._parse_csv(args.heldout_tasks)
	_, val_participants, test_participants = R.resolve_split(records, args)
	record_splits = {
		"val": filter_records(records, participants=val_participants, exclude_tasks=heldout),
		"val_ood": filter_records(records, participants=val_participants, include_tasks=heldout),
		"test_id": filter_records(records, participants=test_participants, exclude_tasks=heldout),
		"test_ood": filter_records(records, participants=test_participants, include_tasks=heldout),
	}
	datasets = {}
	for split_name in split_names:
		if split_name not in record_splits:
			raise ValueError(f"Unsupported split '{split_name}'. Expected one of {sorted(record_splits)}.")
		split = record_splits[split_name]
		_validate_split_against_report(split_name, split, report)
		dataset = R.make_dataset(split, input_names, target_names, args)
		if len(dataset) == 0:
			raise RuntimeError(f"Split '{split_name}' has no usable windows.")
		datasets[split_name] = dataset
	return input_names, target_names, datasets


def _validate_split_against_report(split_name: str, records: Sequence[object], report: dict) -> None:
	expected = report.get("splits", {}).get(split_name)
	if not expected:
		raise ValueError(f"Source report does not contain split metadata for '{split_name}'.")
	participants = sorted({record.participant for record in records})
	expected_participants = sorted(expected.get("participants", []))
	if participants != expected_participants:
		raise ValueError(
			f"Reconstructed participants for {split_name} differ from the report: "
			f"{participants} != {expected_participants}"
		)
	if len(records) != int(expected.get("num_trials", -1)):
		raise ValueError(
			f"Reconstructed trial count for {split_name} differs from the report: "
			f"{len(records)} != {expected.get('num_trials')}"
		)


def inject_bias_drift(
	x: torch.Tensor,
	shank_indices: Sequence[int],
	severity: float,
	mode: str,
	batch_index: int,
	seed: int,
) -> torch.Tensor:
	"""Apply the canonical bias drift or a sign-robustness variant."""
	corrupted = x.clone()
	group = corrupted[:, shank_indices, :]
	scale = group.std(dim=-1, keepdim=True).clamp_min(1e-3)
	ramp = torch.linspace(0.0, 1.0, x.shape[-1], device=x.device, dtype=x.dtype).view(1, 1, -1)
	if mode == "canonical":
		sign = torch.ones((x.shape[0], len(shank_indices), 1), device=x.device, dtype=x.dtype)
	elif mode == "negative":
		sign = -torch.ones((x.shape[0], len(shank_indices), 1), device=x.device, dtype=x.dtype)
	elif mode == "random_sign":
		generator = torch.Generator(device="cpu")
		generator.manual_seed(seed + 104729 * batch_index)
		sign = torch.randint(0, 2, (x.shape[0], len(shank_indices), 1), generator=generator)
		sign = sign.to(device=x.device, dtype=x.dtype).mul_(2.0).sub_(1.0)
	else:
		raise ValueError(f"Unknown bias mode: {mode}")
	corrupted[:, shank_indices, :] = group + severity * scale * sign * ramp
	return corrupted


def normalized_slope(values: torch.Tensor) -> torch.Tensor:
	"""Least-squares slope over a normalized [-1, 1] time axis, per channel."""
	time_steps = values.shape[-1]
	if time_steps < 4:
		return torch.zeros(values.shape[:2], device=values.device, dtype=values.dtype)
	t = torch.linspace(-1.0, 1.0, time_steps, device=values.device, dtype=values.dtype)
	denom = t.pow(2).mean().clamp_min(1e-6)
	return (values * t.view(1, 1, -1)).mean(dim=-1) / denom


def half_shift(values: torch.Tensor) -> torch.Tensor:
	half = values.shape[-1] // 2
	return values[..., half:].mean(dim=-1) - values[..., :half].mean(dim=-1)


def endpoint_shift(values: torch.Tensor) -> torch.Tensor:
	width = max(values.shape[-1] // 10, 1)
	return values[..., -width:].mean(dim=-1) - values[..., :width].mean(dim=-1)


def candidate_scores(input_norm: torch.Tensor, groups: Dict[str, List[int]], ignore_history: int) -> Dict[str, torch.Tensor]:
	start = min(max(int(ignore_history), 0), max(input_norm.shape[-1] - 4, 0))
	z = input_norm[..., start:]
	shank = groups.get("shank_imu", [])
	foot = groups.get("foot_imu", [])
	thigh = groups.get("thigh_imu", [])
	if not shank:
		raise RuntimeError("The selected input profile has no shank_imu channels.")

	shank_slope = normalized_slope(z[:, shank, :])
	shank_half = half_shift(z[:, shank, :])
	shank_end = endpoint_shift(z[:, shank, :])
	scores = {
		"shank_slope_abs_mean": shank_slope.abs().mean(dim=1),
		"shank_slope_abs_max": shank_slope.abs().amax(dim=1),
		"shank_slope_common_abs": shank_slope.mean(dim=1).abs(),
		"shank_half_abs_mean": shank_half.abs().mean(dim=1),
		"shank_half_common_abs": shank_half.mean(dim=1).abs(),
		"shank_endpoint_abs_mean": shank_end.abs().mean(dim=1),
		"shank_endpoint_common_abs": shank_end.mean(dim=1).abs(),
	}

	if len(foot) == len(shank) == len(thigh):
		neighbor_slope = 0.5 * (normalized_slope(z[:, foot, :]) + normalized_slope(z[:, thigh, :]))
		relative_slope = shank_slope - neighbor_slope
		neighbor_half = 0.5 * (half_shift(z[:, foot, :]) + half_shift(z[:, thigh, :]))
		relative_half = shank_half - neighbor_half
		scores.update(
			{
				"cross_modal_slope_rms": relative_slope.pow(2).mean(dim=1).sqrt(),
				"cross_modal_slope_common_abs": relative_slope.mean(dim=1).abs(),
				"cross_modal_half_rms": relative_half.pow(2).mean(dim=1).sqrt(),
				"cross_modal_half_common_abs": relative_half.mean(dim=1).abs(),
			}
		)
	return scores


def collect_split_scores(
	model: torch.nn.Module,
	dataset,
	groups: Dict[str, List[int]],
	args: argparse.Namespace,
	device: torch.device,
	severities: Sequence[float],
	seed: int,
	log_interval: int,
) -> dict:
	loader = R.make_loader(dataset, args, device, shuffle=False)
	clean_parts: Dict[str, List[torch.Tensor]] = {}
	fault_parts: Dict[str, Dict[str, List[torch.Tensor]]] = {
		condition_name(mode, severity): {}
		for severity in severities
		for mode in FAULT_MODES
	}
	metadata = {"participant": [], "trial": [], "task": [], "start": []}
	shank = groups.get("shank_imu", [])
	model.eval()
	with torch.no_grad():
		for batch_index, batch in enumerate(loader):
			if args.max_eval_batches and batch_index >= args.max_eval_batches:
				break
			x = batch["x"].to(device, non_blocking=device.type == "cuda")
			clean = candidate_scores(model.normalize(x), groups, args.eval_ignore_history)
			_append_scores(clean_parts, clean)
			for severity in severities:
				for mode in FAULT_MODES:
					corrupted = inject_bias_drift(x, shank, severity, mode, batch_index, seed)
					scores = candidate_scores(model.normalize(corrupted), groups, args.eval_ignore_history)
					_append_scores(fault_parts[condition_name(mode, severity)], scores)
			metadata["participant"].extend(list(batch["participant"]))
			metadata["trial"].extend(list(batch["trial"]))
			metadata["task"].extend(list(batch["task"]))
			metadata["start"].extend(int(value) for value in batch["start"])
			if log_interval > 0 and (batch_index + 1) % log_interval == 0:
				print(f"  batches={batch_index + 1}/{len(loader)} windows={len(metadata['participant'])}", flush=True)
	return {
		"clean": _concat_scores(clean_parts),
		"faults": {name: _concat_scores(parts) for name, parts in fault_parts.items()},
		"metadata": metadata,
	}


def _append_scores(parts: Dict[str, List[torch.Tensor]], scores: Dict[str, torch.Tensor]) -> None:
	for name, values in scores.items():
		parts.setdefault(name, []).append(values.detach().float().cpu())


def _concat_scores(parts: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
	return {name: torch.cat(values) for name, values in parts.items() if values}


def auroc(clean: torch.Tensor, faulty: torch.Tensor) -> float:
	scores = torch.cat([clean, faulty])
	labels = torch.cat([torch.zeros_like(clean), torch.ones_like(faulty)])
	return R.binary_auroc(scores, labels)


def grouped_aurocs(
	clean: torch.Tensor,
	faulty: torch.Tensor,
	labels: Sequence[str],
	polarity: float,
) -> Dict[str, float]:
	result = {}
	for label in sorted(set(labels)):
		mask = torch.tensor([value == label for value in labels], dtype=torch.bool)
		if mask.any():
			result[label] = auroc(clean[mask] * polarity, faulty[mask] * polarity)
	return result


def finite_mean(values: Iterable[float]) -> float | None:
	finite = [value for value in values if math.isfinite(value)]
	return sum(finite) / len(finite) if finite else None


def finite_min(values: Iterable[float]) -> float | None:
	finite = [value for value in values if math.isfinite(value)]
	return min(finite) if finite else None


def select_validation_policy(
	validation_scores: dict,
	selection_condition: str,
	clean_quantile: float,
) -> tuple[str, Dict[str, dict]]:
	clean_scores = validation_scores["clean"]
	fault_scores = validation_scores["faults"][selection_condition]
	policies = {}
	for candidate in sorted(clean_scores):
		raw_auc = auroc(clean_scores[candidate], fault_scores[candidate])
		polarity = 1.0 if raw_auc >= 0.5 else -1.0
		oriented_clean = clean_scores[candidate] * polarity
		threshold = float(torch.quantile(oriented_clean, clean_quantile))
		policies[candidate] = {
			"raw_validation_auc": raw_auc,
			"validation_auc": auroc(oriented_clean, fault_scores[candidate] * polarity),
			"polarity": polarity,
			"threshold": threshold,
		}
	selected = max(
		policies,
		key=lambda name: (policies[name]["validation_auc"], -len(name), name),
	)
	return selected, policies


def evaluate_policies(all_scores: Dict[str, dict], policies: Dict[str, dict]) -> List[dict]:
	rows = []
	for candidate, policy in policies.items():
		polarity = float(policy["polarity"])
		threshold = float(policy["threshold"])
		for split_name, split_scores in all_scores.items():
			clean = split_scores["clean"][candidate] * polarity
			metadata = split_scores["metadata"]
			for condition, condition_scores in split_scores["faults"].items():
				faulty = condition_scores[candidate] * polarity
				participant_aucs = grouped_aurocs(clean, faulty, metadata["participant"], 1.0)
				task_aucs = grouped_aurocs(clean, faulty, metadata["task"], 1.0)
				mode, severity_text = condition.split("@", 1)
				rows.append(
					{
						"candidate": candidate,
						"split": split_name,
						"mode": mode,
						"severity": float(severity_text),
						"auc": auroc(clean, faulty),
						"clean_fpr": float((clean > threshold).float().mean()),
						"fault_tpr": float((faulty > threshold).float().mean()),
						"participant_auc_mean": finite_mean(participant_aucs.values()),
						"participant_auc_min": finite_min(participant_aucs.values()),
						"task_auc_mean": finite_mean(task_aucs.values()),
						"n_clean": clean.numel(),
						"n_fault": faulty.numel(),
						"participant_aucs": participant_aucs,
					}
				)
	return rows


def write_metrics_tsv(path: Path, rows: Sequence[dict]) -> None:
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


def print_selected_summary(selected: str, rows: Sequence[dict], selection_severity: float) -> None:
	print(f"\nSelected on val/canonical@{format_severity(selection_severity)}: {selected}")
	print(f"{'split':<10} {'mode':<12} {'sev':>5} {'auc':>7} {'FPR':>7} {'TPR':>7} {'pAUC':>7}")
	for row in rows:
		if row["candidate"] != selected or not math.isclose(row["severity"], selection_severity):
			continue
		participant_auc = row["participant_auc_mean"]
		participant_text = f"{participant_auc:.3f}" if participant_auc is not None else "-"
		print(
			f"{row['split']:<10} {row['mode']:<12} {row['severity']:>5.2f} "
			f"{row['auc']:>7.3f} {row['clean_fpr']:>7.3f} {row['fault_tpr']:>7.3f} {participant_text:>7}"
		)


def main() -> None:
	cli = parse_cli()
	report_path = Path(cli.report)
	if not report_path.is_file():
		raise FileNotFoundError(f"Report not found: {report_path}")
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
	if args.eval_ignore_history <= 0:
		raise ValueError("The source report has no positive eval_ignore_history; refusing a prefix-sensitive diagnostic.")

	split_names = parse_csv(cli.splits)
	if "val" not in split_names:
		raise ValueError("--splits must include val because detector policy selection is validation-only.")
	severities = parse_floats(cli.severities)
	if not any(math.isclose(value, cli.selection_severity) for value in severities):
		severities.append(cli.selection_severity)
	severities = sorted(set(severities))
	if not 0.5 < cli.clean_quantile < 1.0:
		raise ValueError("--clean-quantile must be between 0.5 and 1.0.")

	input_names, _, datasets = build_datasets(report, args, split_names)
	groups = feature_groups(input_names)
	device = torch.device(args.device)
	model, payload = load_checkpoint(checkpoint, map_location=device)
	R.validate_checkpoint_inputs(payload, input_names, label_names(args.side))
	model = model.to(device).eval()

	print(
		f"Loaded report={report_path} checkpoint={checkpoint} device={device} "
		f"window={args.window_size} stride={args.stride} max_windows_per_trial={args.max_windows_per_trial} "
		f"eval_ignore_history={args.eval_ignore_history}",
		flush=True,
	)
	all_scores = {}
	for split_name, dataset in datasets.items():
		print(f"\n[{split_name}] windows={len(dataset)}", flush=True)
		all_scores[split_name] = collect_split_scores(
			model,
			dataset,
			groups,
			args,
			device,
			severities,
			cli.seed,
			cli.log_interval,
		)

	selection_condition = condition_name("canonical", cli.selection_severity)
	selected, policies = select_validation_policy(all_scores["val"], selection_condition, cli.clean_quantile)
	rows = evaluate_policies(all_scores, policies)
	print_selected_summary(selected, rows, cli.selection_severity)

	output_dir = Path(cli.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	metrics_path = output_dir / "imu_bias_candidate_metrics.tsv"
	json_path = output_dir / "imu_bias_diagnostic.json"
	write_metrics_tsv(metrics_path, rows)
	result = {
		"source_report": str(report_path),
		"source_checkpoint": str(checkpoint),
		"source_checkpoint_sha256": sha256_file(checkpoint),
		"restored_data_args": {
			"data_root": args.data_root,
			"heldout_tasks": args.heldout_tasks,
			"val_count": args.val_count,
			"test_count": args.test_count,
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
			"severities": severities,
			"selection_severity": cli.selection_severity,
			"clean_quantile": cli.clean_quantile,
			"seed": cli.seed,
		},
		"selection_source": f"val/{selection_condition}",
		"selected_candidate": selected,
		"candidate_policies": policies,
		"metrics": rows,
	}
	json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
	print(f"\nWrote {metrics_path}")
	print(f"Wrote {json_path}")


if __name__ == "__main__":
	main()
