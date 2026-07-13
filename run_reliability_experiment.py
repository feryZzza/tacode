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
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from reliability.closed_loop import closed_loop_metrics
from reliability.faults import apply_fault, canonical_fault_name, fault_classes_from_names, parse_fault_spec
from reliability.features import feature_groups, input_names_for_profile, joint_velocity_indices, label_names, participant_masses
from reliability.gate_selection import gate_selection_score
from reliability.metrics import (
	binary_auroc,
	calibrate_risk,
	cross_modal_coherence_score,
	drift_score,
	forecast_residual,
	interval_calibration,
	masked_gaussian_nll,
	masked_mean,
	masked_mse,
	reconstruction_residual,
	regression_metrics,
	staleness_score,
	torque_replay_metrics,
	uncertainty_error_correlation,
)
from reliability.model import ReliabilityTCN, load_checkpoint, save_checkpoint
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
	parser.add_argument("--val-participants", default="", help="Explicit val participant ids (CSV). Overrides --val-count. Used for LOSO.")
	parser.add_argument("--test-participants", default="", help="Explicit test participant ids (CSV). Overrides --test-count. Used for LOSO.")
	parser.add_argument("--window-size", type=int, default=768)
	parser.add_argument("--stride", type=int, default=512)
	parser.add_argument("--min-valid-fraction", type=float, default=0.5)
	parser.add_argument("--max-windows-per-trial", type=int, default=2)
	parser.add_argument("--limit-trials", type=int, default=0, help="Per split. 0 means no limit.")
	parser.add_argument("--epochs", type=int, default=3)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--num-workers", type=int, default=8, help="DataLoader 子进程数。0 = 主进程内加载（GPU 易饿）。")
	parser.add_argument("--prefetch-factor", type=int, default=4, help="每个 worker 预取的 batch 数（num-workers>0 时生效）。")
	parser.add_argument("--amp", choices=["off", "bf16", "fp16"], default="bf16", help="训练前向/反向的混合精度。CPU 上自动关闭。A100 默认 bf16。")
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--num-channels", default="32,32,32,32")
	parser.add_argument("--kernel-size", type=int, default=4)
	parser.add_argument("--dropout", type=float, default=0.1)
	parser.add_argument("--spatial-dropout", action="store_true")
	parser.add_argument("--activation", default="ReLU")
	parser.add_argument("--norm", default="weight_norm")
	parser.add_argument("--fault-loss-weight", type=float, default=0.1)
	parser.add_argument("--fault-positive-weight", type=float, default=0.0,
		help="Positive-class weight for per-fault BCE. <=0 uses the number of fault heads.")
	parser.add_argument("--fault-head-mode", choices=["binary", "per_fault"], default="binary",
		help="Use the validated binary fault head or an experimental per-fault head bank.")
	parser.add_argument("--clean-sample-prob", type=float, default=-1.0,
		help="Clean-view probability during augmentation. Negative keeps legacy uniform sampling over --training-faults.")
	parser.add_argument("--recon-loss-weight", type=float, default=0.1, help="Weight for input-integrity reconstruction loss.")
	parser.add_argument("--recon-detach", dest="recon_detach", action="store_true", default=False,
		help="Experimental: stop reconstruction gradients at the shared trunk.")
	parser.add_argument("--no-recon-detach", dest="recon_detach", action="store_false",
		help="Let reconstruction gradients update the shared trunk.")
	parser.add_argument("--forecast-loss-weight", type=float, default=0.1, help="Weight for one-step input forecasting loss (temporal-integrity head).")
	parser.add_argument("--forecast-horizon", type=int, default=1, help="Steps ahead the forecast head predicts; residual uses the same horizon.")
	parser.add_argument("--forecast-detach", dest="forecast_detach", action="store_true", default=True,
		help="Stop-gradient the forecast head from the shared trunk so its multi-task loss cannot degrade the mean head (default).")
	parser.add_argument("--no-forecast-detach", dest="forecast_detach", action="store_false",
		help="Let forecast gradients flow into the trunk (round-2 behaviour; tends to cost ~0.01-0.02 clean RMSE).")
	parser.add_argument("--training-mode", default="prob_aug_recon_fc",
		choices=["det_noaug", "det_aug", "prob_aug", "prob_aug_recon", "prob_aug_recon_fc"],
		help="Attribution ablation: deterministic/probabilistic x with/without fault augmentation, reconstruction head, and forecast head.")
	parser.add_argument("--mc-samples", type=int, default=0, help="MC-dropout samples at eval for epistemic uncertainty. 0 disables.")
	parser.add_argument("--ensemble-checkpoints", default="", help="CSV of extra checkpoints to average for a deep ensemble at eval.")
	parser.add_argument("--training-faults", default="clean,insole_missing,encoder_dropout,imu_bias,packet_loss,stuck_imu,sensor_delay")
	parser.add_argument("--train-ignore-history", type=int, default=0)
	parser.add_argument("--eval-ignore-history", type=int, default=0)
	parser.add_argument("--gate-softness", type=float, default=0.5)
	parser.add_argument("--gate-deadband", type=float, default=0.0, help="Risk margin within which the gate stays at 1.0 (no clean-condition penalty).")
	parser.add_argument("--select-gate-on-val", action="store_true", help="Pick gate softness/deadband on val clean, then freeze for test.")
	parser.add_argument("--val-wrong-margin", type=float, default=0.01, help="Allowed wrong-direction increase over ungated when selecting the gate on val.")
	parser.add_argument("--gate-validation-faults", default="insole_missing,encoder_dropout,packet_loss,sensor_delay",
		help="Validation fault scenarios used to select a utility-aware gate. Empty keeps clean-only selection.")
	parser.add_argument("--gate-clean-retained-floor", type=float, default=0.95,
		help="Minimum val/clean retained aligned torque as a fraction of the ungated model when selecting the gate.")
	parser.add_argument("--gate-fault-wrong-margin", type=float, default=0.0,
		help="Allowed wrong-direction increase over ungated on validation faults when selecting the gate.")
	parser.add_argument("--gate-fault-retained-weight", type=float, default=0.25,
		help="Selection score weight for keeping useful torque under faults after wrong-direction constraints are met.")
	parser.add_argument("--gate-fault-safety-weight", type=float, default=2.0,
		help="Selection score weight for the relative reduction in wrong-direction torque on validation faults.")
	parser.add_argument("--ood-deadband-quantile", type=float, default=0.9, help="Quantile of clean OOD-val risk used to set a domain-shift deadband (gate stays ~1 on clean OOD). <=0 disables (uses --gate-deadband).")
	parser.add_argument("--gate-max-deadband", type=float, default=2.0,
		help="Upper bound for validation-derived deadband. Prevents OOD clean calibration from disabling the gate.")
	parser.add_argument("--gate-softness-grid", default="")
	parser.add_argument("--gate-sweep-scenarios", default="test_id/clean,test_ood/clean,test_id/insole_missing,test_id/encoder_dropout,test_id/sensor_delay,test_ood/insole_missing,test_ood/encoder_dropout,test_ood/sensor_delay")
	parser.add_argument("--stats-batches", type=int, default=24)
	parser.add_argument("--max-eval-batches", type=int, default=0)
	parser.add_argument("--fault-scenarios", default="clean,insole_missing,encoder_dropout,imu_bias,packet_loss,sensor_delay")
	parser.add_argument(
		"--detector-validation-faults",
		default="insole_missing,encoder_dropout,imu_bias,packet_loss,sensor_delay",
		help="Validation-only fault set used to select the compact detector fusion.",
	)
	parser.add_argument("--detector-max-signals", type=int, default=3)
	parser.add_argument("--log-interval", type=int, default=25, help="Batch interval for line-mode progress. 0 disables live progress.")
	parser.add_argument("--progress-style", choices=["auto", "bar", "line", "off"], default="auto", help="Live training progress display.")
	parser.add_argument("--print-json-results", action="store_true", help="Print the full results JSON to stdout.")
	parser.add_argument("--seed", type=int, default=7)
	return parser.parse_args()


def main() -> None:
	configure_terminal_warnings()
	args = parse_args()
	if args.mode == "smoke":
		args.epochs = 1
		args.limit_trials = args.limit_trials or 8
		args.max_windows_per_trial = 1
		args.max_eval_batches = args.max_eval_batches or 2

	set_seed(args.seed)
	configure_backend(args)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	input_names = input_names_for_profile(args.input_profile, args.side)
	target_names = label_names(args.side)
	groups = feature_groups(input_names)
	args.velocity_indices = joint_velocity_indices(input_names, target_names)
	records = discover_trials(args.data_root)
	heldout_tasks = _parse_csv(args.heldout_tasks)
	train_participants, val_participants, test_participants = resolve_split(records, args)

	splits = {
		"train": filter_records(records, participants=train_participants, exclude_tasks=heldout_tasks),
		"val": filter_records(records, participants=val_participants, exclude_tasks=heldout_tasks),
		# val_ood: 验证被试 × heldout 任务，合法的域移标定集（无 test 泄漏），用于选 OOD deadband。
		"val_ood": filter_records(records, participants=val_participants, include_tasks=heldout_tasks),
		"test_id": filter_records(records, participants=test_participants, exclude_tasks=heldout_tasks),
		"test_ood": filter_records(records, participants=test_participants, include_tasks=heldout_tasks),
	}
	if args.limit_trials:
		splits = {name: split[: args.limit_trials] for name, split in splits.items()}

	datasets = {}
	for name, split in splits.items():
		if not split:
			continue
		dataset = make_dataset(split, input_names, target_names, args)
		if len(dataset) > 0:
			datasets[name] = dataset
		else:
			print(f"WARNING split '{name}' has no windows after valid-label filtering.")
	if "train" not in datasets or len(datasets["train"]) == 0:
		raise RuntimeError("Training split has no windows. Check heldout tasks or dataset root.")

	print_run_summary(args, input_names, target_names, splits, datasets)

	device = torch.device(args.device)
	if args.checkpoint:
		model, payload = load_checkpoint(args.checkpoint, map_location=device)
		validate_checkpoint_inputs(payload, input_names, target_names)
		model = model.to(device)
		print(f"Loaded checkpoint: {args.checkpoint}")
	else:
		model = build_model(args, len(input_names), len(target_names), device)
		print("Estimating input normalization statistics...")
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
				"spatial_dropout": args.spatial_dropout,
				"activation": args.activation,
				"norm": args.norm,
				"use_recon": mode_flags(args.training_mode)["use_recon"],
				"recon_detach": args.recon_detach,
				"use_forecast": mode_flags(args.training_mode)["use_forecast"],
				"forecast_horizon": args.forecast_horizon,
				"forecast_detach": args.forecast_detach,
				"training_mode": args.training_mode,
				"history": history,
				"data_root": args.data_root,
				"heldout_tasks": heldout_tasks,
				"training_faults": _parse_csv(args.training_faults),
				"train_ignore_history": args.train_ignore_history,
				"eval_ignore_history": args.eval_ignore_history,
			},
		)
		print(f"\nSaved checkpoint: {checkpoint_path}")

	print("\nEvaluating reliability scenarios...")
	args.extra_models = load_ensemble_models(args, input_names, target_names, device)
	if args.extra_models:
		print(f"Deep ensemble: +{len(args.extra_models)} models for epistemic uncertainty.")
	results = evaluate_all(model, datasets, groups, args, device)
	if args.input_profile == "human" and args.nature_baseline_checkpoint:
		baseline_path = Path(args.nature_baseline_checkpoint)
		if baseline_path.exists():
			print("Evaluating Nature baseline checkpoint...")
			results.update(evaluate_nature_baseline(str(baseline_path), datasets, groups, args, device))
		else:
			print(f"WARNING Nature baseline checkpoint not found: {baseline_path}")
	report = {
		"args": _serializable_args(args),
		"input_names": input_names,
		"label_names": target_names,
		"splits": {name: summarize_records(split) for name, split in splits.items()},
		"history": history,
		"results": results,
	}
	report_path = output_dir / "reliability_report.json"
	report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
	print_result_summary(results, args)
	if args.print_json_results:
		print(json.dumps(results, indent=2))
	print(f"\nWrote report: {report_path}")


def resolve_split(records, args):
	"""按显式被试（LOSO）或计数划分 train/val/test。"""
	explicit_val = _parse_csv(args.val_participants)
	explicit_test = _parse_csv(args.test_participants)
	if explicit_test or explicit_val:
		all_participants = sorted({record.participant for record in records})
		test = explicit_test
		val = explicit_val
		held = set(test) | set(val)
		train = [p for p in all_participants if p not in held]
		if not train:
			raise RuntimeError("Explicit split left no training participants.")
		return train, val, test
	return participant_split(records, args.val_count, args.test_count)


def mode_flags(training_mode: str) -> dict:
	"""把 training-mode 字符串展开成 (probabilistic, augment, use_recon, use_forecast) 开关。"""
	return {
		"det_noaug": {"probabilistic": False, "augment": False, "use_recon": False, "use_forecast": False},
		"det_aug": {"probabilistic": False, "augment": True, "use_recon": False, "use_forecast": False},
		"prob_aug": {"probabilistic": True, "augment": True, "use_recon": False, "use_forecast": False},
		"prob_aug_recon": {"probabilistic": True, "augment": True, "use_recon": True, "use_forecast": False},
		"prob_aug_recon_fc": {"probabilistic": True, "augment": True, "use_recon": True, "use_forecast": True},
	}[training_mode]


def sample_training_fault_name(nonclean_faults: List[str], clean_probability: float) -> str:
	clean_probability = min(max(float(clean_probability), 0.0), 1.0)
	if not nonclean_faults or random.random() < clean_probability:
		return "clean"
	return random.choice(nonclean_faults)


def multihead_fault_target(
	binary_target: torch.Tensor,
	fault_name: str,
	fault_classes: List[str],
) -> torch.Tensor:
	if len(fault_classes) == 1 and fault_classes[0] == "fault":
		return binary_target
	target = torch.zeros(
		(binary_target.shape[0], len(fault_classes), binary_target.shape[-1]),
		device=binary_target.device,
		dtype=binary_target.dtype,
	)
	fault_class = canonical_fault_name(fault_name)
	if fault_class == "clean":
		return target
	if fault_class not in fault_classes:
		raise ValueError(f"Fault class '{fault_class}' is not configured in {fault_classes}.")
	index = fault_classes.index(fault_class)
	target[:, index : index + 1, :] = binary_target
	return target


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
	flags = mode_flags(args.training_mode)
	return ReliabilityTCN(
		input_size=input_size,
		output_size=output_size,
		num_channels=_parse_channels(args.num_channels),
		ksize=args.kernel_size,
		dropout=args.dropout,
		spatial_dropout=args.spatial_dropout,
		activation=args.activation,
		norm=args.norm,
		use_recon=flags["use_recon"],
		recon_detach=args.recon_detach,
		use_forecast=flags["use_forecast"],
		forecast_horizon=args.forecast_horizon,
		forecast_detach=args.forecast_detach,
		fault_classes=(
			fault_classes_from_names(_parse_csv(args.training_faults))
			if flags["augment"] and args.fault_head_mode == "per_fault"
			else ["fault"]
		),
	).to(device)


def validate_checkpoint_inputs(payload: dict, input_names: List[str], target_names: List[str]) -> None:
	checkpoint_inputs = payload.get("input_names")
	checkpoint_labels = payload.get("label_names")
	if checkpoint_inputs is not None and list(checkpoint_inputs) != list(input_names):
		raise ValueError("Checkpoint input_names do not match the selected --input-profile/--side.")
	if checkpoint_labels is not None and list(checkpoint_labels) != list(target_names):
		raise ValueError("Checkpoint label_names do not match the selected --side.")


def train(
	model: ReliabilityTCN,
	train_dataset: ParsedWindowDataset,
	val_dataset: Optional[ParsedWindowDataset],
	groups: Dict[str, List[int]],
	args: argparse.Namespace,
	device: torch.device,
) -> List[Dict[str, float]]:
	loader = make_loader(train_dataset, args, device, shuffle=True)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
	use_amp, amp_dtype = amp_settings(args, device)
	scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
	history = []
	best_score = math.inf
	best_state = None
	training_faults = _parse_csv(args.training_faults)
	nonclean_training_faults = [name for name in training_faults if canonical_fault_name(name) != "clean"]
	flags = mode_flags(args.training_mode)
	print_training_header(args)
	for epoch in range(1, args.epochs + 1):
		model.train()
		running = []
		progress = TrainingProgress(epoch, args.epochs, len(loader), args, device)
		for batch_index, batch in enumerate(loader, start=1):
			x = batch["x"].to(device, non_blocking=True)
			y = batch["y"].to(device, non_blocking=True)
			mask = apply_ignore_history(batch["mask"].to(device, non_blocking=True), args.train_ignore_history)
			if flags["augment"]:
				if args.clean_sample_prob < 0:
					fault_name = random.choice(training_faults)
				else:
					fault_name = sample_training_fault_name(nonclean_training_faults, args.clean_sample_prob)
				spec = parse_fault_spec(fault_name)
				x_fault, fault_target = apply_fault(x, groups, spec)
			else:
				x_fault, fault_target = x, torch.zeros((x.shape[0], 1, x.shape[2]), device=device, dtype=x.dtype)
			with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
				out = model(x_fault)
				# 异方差 NLL（概率模式）或纯 MSE（确定性模式）。
				if flags["probabilistic"]:
					estimate_loss = masked_gaussian_nll(out["mean"], out["logvar"], y, mask)
				else:
					estimate_loss = masked_mse(out["mean"], y, mask)
				loss = estimate_loss
				if flags["augment"]:
					fault_logits = out.get("fault_logits", out["fault_logit"])
					fault_targets = multihead_fault_target(
						fault_target,
						fault_name,
						getattr(model, "fault_classes", ["fault"]),
					)
					if args.fault_positive_weight > 0:
						positive_weight = args.fault_positive_weight
					elif fault_logits.shape[1] == 1:
						# Preserve the validated v2 binary-BCE objective.
						positive_weight = 1.0
					else:
						if args.clean_sample_prob < 0:
							positive_rate = 1.0 / max(len(training_faults), 1)
						else:
							positive_rate = (1.0 - min(max(args.clean_sample_prob, 0.0), 1.0)) / fault_logits.shape[1]
						positive_weight = (1.0 - positive_rate) / max(positive_rate, 1e-6)
					pos_weight = torch.full(
						(1, fault_logits.shape[1], 1),
						float(positive_weight),
						device=fault_logits.device,
						dtype=fault_logits.dtype,
					)
					fault_loss_raw = F.binary_cross_entropy_with_logits(
						fault_logits,
						fault_targets,
						pos_weight=pos_weight,
						reduction="none",
					)
					fault_mask = mask.any(dim=1, keepdim=True).expand_as(fault_loss_raw)
					fault_loss = masked_mean(fault_loss_raw, fault_mask)
					loss = loss + args.fault_loss_weight * fault_loss
				if flags["use_recon"] and "recon" in out:
					# 重构头学习从（可能受损的）输入恢复干净的归一化输入，
					# 使重构残差在受损输入上偏大，成为输入完整性信号。
					clean_norm = model.normalize(x).detach()
					recon_loss = masked_mean((out["recon"] - clean_norm).pow(2), mask.any(dim=1, keepdim=True))
					loss = loss + args.recon_loss_weight * recon_loss
				if flags["use_forecast"] and "forecast" in out:
					# 预测头学习从干净输入一步预测未来：forecast[t] ≈ clean_norm[t+h]。
					# 在干净动态上训练，故障(丢包/延迟/漂移)破坏可预测性使残差升高。
					h = max(int(args.forecast_horizon), 1)
					clean_norm = model.normalize(x).detach()
					if clean_norm.shape[-1] > h:
						fc_pred = out["forecast"][..., :-h]
						fc_target = clean_norm[..., h:]
						fc_mask = mask.any(dim=1, keepdim=True)[..., h:]
						forecast_loss = masked_mean((fc_pred - fc_target).pow(2), fc_mask)
						loss = loss + args.forecast_loss_weight * forecast_loss
			optimizer.zero_grad(set_to_none=True)
			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
			scaler.step(optimizer)
			scaler.update()
			running.append(float(loss.detach().cpu()))
			progress.update(batch_index, running[-1], sum(running) / max(len(running), 1))
		progress.close()

		epoch_info = {
			"epoch": epoch,
			"train_loss": sum(running) / max(len(running), 1),
		}
		is_best = False
		if val_dataset is not None and len(val_dataset) > 0:
			# 每轮 val 评估关闭 MC-dropout 以加速（epistemic 只在最终评估时算）。
			saved_mc = args.mc_samples
			args.mc_samples = 0
			val_metrics = evaluate_dataset(model, val_dataset, groups, args, device, fault_name="clean")
			args.mc_samples = saved_mc
			epoch_info.update({f"val_{key}": value for key, value in val_metrics.items() if isinstance(value, float)})
			rmse = val_metrics.get("rmse", math.inf)
			score = rmse + 0.02 * val_metrics.get("coverage_ece", 0.0) if math.isfinite(rmse) else math.inf
			if score < best_score:
				best_score = score
				best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
				is_best = True
		print_epoch_row(epoch_info, args, device, is_best)
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
	# 在干净 val 上估计每路风险信号的参考分位，供 calibrate_risk 与检测分数归一使用。
	risk_refs = estimate_risk_refs(model, ref_dataset, groups, args, device) if ref_dataset is not None else None
	# 门控超参：仅在 val 上选定后冻结（E4）。
	gate_softness, gate_deadband, gate_policy_info = select_gate_on_val(model, datasets, groups, args, device, uncertainty_ref, risk_refs)
	args.selected_gate_softness = gate_softness
	args.selected_gate_deadband = gate_deadband
	results["_gate_policy"] = {"softness": gate_softness, "deadband": gate_deadband, **gate_policy_info}
	detector_signals, detector_policy_info = select_detector_signals_on_val(model, datasets, groups, args, device, risk_refs)
	args.selected_detector_signals = ",".join(detector_signals)
	results["_detector_policy"] = detector_policy_info
	for split_name, dataset in datasets.items():
		# val_ood 仅用于 deadband 标定，不作为评估场景报告（避免泄漏式自评 + 报告膨胀）。
		if split_name == "val_ood":
			continue
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
				risk_refs=risk_refs,
				gate_softness=gate_softness,
				gate_deadband=gate_deadband,
			)
		# 每路检测信号分别报 AUROC（E1/E2），并报融合分数。
		for fault_name in _parse_csv(args.fault_scenarios):
			if fault_name == "clean":
				continue
			clean = collect_detector_scores(model, dataset, groups, args, device, "clean")
			faulty = collect_detector_scores(model, dataset, groups, args, device, fault_name)
			detector_metrics = detector_aurocs(clean, faulty, risk_refs, signal_names=detector_signals)
			if detector_metrics:
				results[f"fault_detection/{split_name}/{fault_name}"] = detector_metrics

	if "test_id" in datasets and "test_ood" in datasets and len(datasets["test_ood"]) > 0:
		id_scores = collect_detector_scores(model, datasets["test_id"], groups, args, device, "clean")
		ood_scores = collect_detector_scores(model, datasets["test_ood"], groups, args, device, "clean")
		results["ood_detection/clean"] = detector_aurocs(id_scores, ood_scores, risk_refs, primary_key="risk_auroc")
	if _parse_csv(args.gate_softness_grid):
		results.update(evaluate_gate_sweep(model, datasets, groups, args, device, uncertainty_ref, risk_refs))
	return results


def select_detector_signals_on_val(
	model: ReliabilityTCN,
	datasets: Dict[str, ParsedWindowDataset],
	groups: Dict[str, List[int]],
	args: argparse.Namespace,
	device: torch.device,
	refs: Optional[Dict[str, float]],
) -> tuple[List[str], Dict[str, object]]:
	if "val" not in datasets:
		signals = ["logit", "aleatoric", "residual", "forecast", "epistemic", "staleness", "coherence", "drift"]
		return signals, {"policy_source": "all_available_no_val_split", "signals": signals}
	saved_mc_samples = args.mc_samples
	args.mc_samples = 0
	try:
		clean = collect_detector_scores(model, datasets["val"], groups, args, device, "clean")
		faults = {
			fault_name: collect_detector_scores(model, datasets["val"], groups, args, device, fault_name)
			for fault_name in _parse_csv(args.detector_validation_faults)
			if fault_name != "clean"
		}
	finally:
		args.mc_samples = saved_mc_samples
	signals, info = select_detector_signal_subset(clean, faults, refs, max_signals=max(args.detector_max_signals, 1))
	info["selection_mc_samples"] = 0
	return signals, info


def detector_aurocs(
	clean: Dict[str, torch.Tensor],
	faulty: Dict[str, torch.Tensor],
	refs: Optional[Dict[str, float]],
	primary_key: str = "fault_auroc",
	signal_names: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
	"""对每路信号和融合分数分别计算 clean-vs-fault AUROC。"""
	metrics: Dict[str, float] = {}
	for name in ("logit", "aleatoric", "residual", "forecast", "epistemic", "staleness", "coherence", "drift"):
		if name in clean and name in faulty and clean[name].numel() and faulty[name].numel():
			scores = torch.cat([clean[name], faulty[name]])
			labels = torch.cat([torch.zeros_like(clean[name]), torch.ones_like(faulty[name])])
			metrics[f"auroc_{name}"] = binary_auroc(scores, labels)
	signal_scores = [
		(name, value)
		for name, value in ((name, metrics.get(f"auroc_{name}")) for name in ("logit", "aleatoric", "residual", "forecast", "epistemic", "staleness", "coherence", "drift"))
		if isinstance(value, float) and math.isfinite(value)
	]
	if signal_scores:
		best_signal, best_score = max(signal_scores, key=lambda item: item[1])
		metrics["best_signal"] = best_signal
		metrics["best_signal_auroc"] = best_score
	combined_clean = combined_detector_score(clean, refs, signal_names)
	combined_fault = combined_detector_score(faulty, refs, signal_names)
	if combined_clean.numel() and combined_fault.numel():
		scores = torch.cat([combined_clean, combined_fault])
		labels = torch.cat([torch.zeros_like(combined_clean), torch.ones_like(combined_fault)])
		metrics[primary_key] = binary_auroc(scores, labels)
	return metrics


def estimate_risk_refs(
	model: ReliabilityTCN,
	dataset: ParsedWindowDataset,
	groups: Dict[str, List[int]],
	args: argparse.Namespace,
	device: torch.device,
) -> Dict[str, float]:
	"""在干净 val 上估计每路风险信号的高分位参考（默认 0.9）。"""
	scores = collect_detector_scores(model, dataset, groups, args, device, "clean")
	refs: Dict[str, float] = {}
	for name, value in scores.items():
		if value.numel():
			refs[name] = float(torch.quantile(value, 0.9).clamp_min(1e-6))
	# staleness/drift 在干净信号上接近 0，0.9 分位可能也是 0；若用 ~1e-6 归一，clean 上的
	# 微小噪声会被放大成巨大风险污染融合分数。改用绝对下限：staleness 以"5% 冻结帧"为名义阈，
	# drift 以 clean 斜率分布的高分位且不低于 1e-3 为阈，保证 clean→低分、故障才触发。
	if "staleness" in scores and scores["staleness"].numel():
		refs["staleness"] = max(refs.get("staleness", 0.0), 0.02)
	if "drift" in scores and scores["drift"].numel():
		refs["drift"] = max(refs.get("drift", 0.0), 1e-3)
	# calibrate_risk 用的键名对齐。
	refs.setdefault("fault", refs.get("logit", 0.5))
	return refs


def select_gate_on_val(
	model: ReliabilityTCN,
	datasets: Dict[str, ParsedWindowDataset],
	groups: Dict[str, List[int]],
	args: argparse.Namespace,
	device: torch.device,
	uncertainty_ref: float,
	risk_refs: Optional[Dict[str, float]],
) -> tuple[float, float, Dict[str, object]]:
	"""在 val 上选门控超参，使 clean 上不误伤、故障下抑制错误方向，然后冻结。

	策略：softness 在网格里挑能让 val/clean 的 retained torque 保持接近 ungated、
	且 validation faults 的 wrong-direction 不比 ungated 更差，并尽量降低 wrong torque。
	deadband 默认用 args.gate_deadband；若提供 val_ood（验证被试×
	heldout 任务）且 --ood-deadband-quantile>0，则用 clean OOD-val 风险分位设一个"域移 deadband"，
	让门控的名义带宽吸收域移引起的残差/不确定性升高——这样 OOD-clean 不被误伤，而真故障(风险更高)
	仍越过带宽被抑制。这是把 deadband 标定从 ID-clean 扩展到 OOD-clean 的关键修复。
	"""
	if not args.select_gate_on_val or "val" not in datasets:
		return args.gate_softness, args.gate_deadband, {"policy_source": "fixed_cli"}
	grid = _parse_floats(args.gate_softness_grid) or [0.25, 0.5, 1.0, 2.0]
	# 域移 deadband：用 val_ood 的 clean 融合风险分位（threshold = 1 + deadband）。
	deadband = args.gate_deadband
	if getattr(args, "ood_deadband_quantile", 0.0) > 0.0 and "val_ood" in datasets:
		ood_out = collect_reliability_outputs(model, datasets["val_ood"], groups, args, device, "clean")
		if ood_out is not None:
			ood_risk = calibrate_risk(
				aleatoric_sigma=torch.exp(0.5 * ood_out["logvar"]),
				fault_prob=torch.sigmoid(ood_out["fault_logit"]),
				residual=ood_out.get("residual"),
				epistemic_sigma=ood_out.get("epistemic"),
				forecast=ood_out.get("forecast"),
				staleness=ood_out.get("staleness"),
				drift=None,
				refs=risk_refs,
			)
			ood_tm = ood_out["mask"].any(dim=1, keepdim=True)
			if ood_tm.any():
				q = float(torch.quantile(ood_risk[ood_tm], args.ood_deadband_quantile))
				# deadband = 把 clean OOD 名义风险移到带宽内所需的余量，下限 args.gate_deadband。
				deadband = max(args.gate_deadband, q - 1.0)
				if args.gate_max_deadband > 0:
					deadband = min(deadband, args.gate_max_deadband)
	outputs = collect_reliability_outputs(model, datasets["val"], groups, args, device, "clean")
	if outputs is None:
		return args.gate_softness, deadband, {"policy_source": "fixed_cli_no_val_outputs"}

	clean_risk = risk_from_outputs(outputs, risk_refs)
	clean_ungated = gate_metrics_for_outputs(outputs, clean_risk, uncertainty_ref, softness=1e6, deadband=1e6)
	clean_wrong_limit = clean_ungated.get("gated_wrong_direction_ratio", 1.0) + args.val_wrong_margin
	clean_ungated_retained = clean_ungated.get("gated_retained_aligned_torque", 0.0)
	clean_retained_floor = clean_ungated_retained * max(args.gate_clean_retained_floor, 0.0)

	fault_cases = []
	for fault_name in _parse_csv(args.gate_validation_faults):
		if fault_name == "clean":
			continue
		fault_outputs = collect_reliability_outputs(model, datasets["val"], groups, args, device, fault_name)
		if fault_outputs is None:
			continue
		fault_risk = risk_from_outputs(fault_outputs, risk_refs)
		fault_ungated = gate_metrics_for_outputs(fault_outputs, fault_risk, uncertainty_ref, softness=1e6, deadband=1e6)
		fault_cases.append(
			{
				"name": fault_name,
				"outputs": fault_outputs,
				"risk": fault_risk,
				"ungated_wrong": fault_ungated.get("gated_wrong_direction_ratio", 1.0),
				"ungated_retained": fault_ungated.get("gated_retained_aligned_torque", 0.0),
				"wrong_limit": fault_ungated.get("gated_wrong_direction_ratio", 1.0) + args.gate_fault_wrong_margin,
			}
		)

	best = (args.gate_softness, deadband)
	best_score = -math.inf
	best_info: Dict[str, object] = {
		"policy_source": "val_utility_aware",
		"selection_objective": "dimensionless_relative_wrong_v2",
		"constraints_met": False,
	}
	for softness in grid:
		clean_metrics = gate_metrics_for_outputs(outputs, clean_risk, uncertainty_ref, softness=softness, deadband=deadband)
		clean_wrong = clean_metrics.get("gated_wrong_direction_ratio")
		clean_retained = clean_metrics.get("gated_retained_aligned_torque")
		if clean_wrong is None or clean_retained is None:
			continue
		clean_ok = clean_wrong <= clean_wrong_limit and clean_retained >= clean_retained_floor
		clean_retained_ratio = clean_retained / max(clean_ungated_retained, 1e-6)

		fault_ok = True
		fault_wrong_reductions = []
		fault_wrong_reduction_ratios = []
		fault_retained_ratios = []
		fault_names = []
		for case in fault_cases:
			metrics = gate_metrics_for_outputs(case["outputs"], case["risk"], uncertainty_ref, softness=softness, deadband=deadband)
			wrong = metrics.get("gated_wrong_direction_ratio")
			retained = metrics.get("gated_retained_aligned_torque")
			if wrong is None or retained is None:
				continue
			fault_names.append(case["name"])
			fault_ok = fault_ok and wrong <= case["wrong_limit"]
			wrong_reduction = max(case["ungated_wrong"] - wrong, 0.0)
			fault_wrong_reductions.append(wrong_reduction)
			fault_wrong_reduction_ratios.append(
				min(wrong_reduction / max(case["ungated_wrong"], 1e-6), 1.0)
			)
			fault_retained_ratios.append(retained / max(case["ungated_retained"], 1e-6))
		mean_fault_wrong_reduction = sum(fault_wrong_reductions) / max(len(fault_wrong_reductions), 1)
		mean_fault_wrong_reduction_ratio = sum(fault_wrong_reduction_ratios) / max(len(fault_wrong_reduction_ratios), 1)
		mean_fault_retained_ratio = sum(fault_retained_ratios) / max(len(fault_retained_ratios), 1)

		constraints_met = clean_ok and fault_ok
		score = gate_selection_score(
			clean_retained_ratio=clean_retained_ratio,
			mean_fault_wrong_reduction_ratio=mean_fault_wrong_reduction_ratio,
			mean_fault_retained_ratio=mean_fault_retained_ratio,
			fault_safety_weight=args.gate_fault_safety_weight,
			fault_retained_weight=args.gate_fault_retained_weight,
			constraints_met=constraints_met,
		)
		if score > best_score:
			best_score = score
			best = (softness, deadband)
			best_info = {
				"policy_source": "val_utility_aware",
				"selection_objective": "dimensionless_relative_wrong_v2",
				"constraints_met": constraints_met,
				"selection_score": score,
				"clean_wrong_limit": clean_wrong_limit,
				"clean_wrong": clean_wrong,
				"clean_ungated_retained": clean_ungated_retained,
				"clean_retained_floor": clean_retained_floor,
				"clean_retained": clean_retained,
				"clean_retained_ratio": clean_retained_ratio,
				"validation_faults": ",".join(fault_names),
				"mean_fault_wrong_reduction": mean_fault_wrong_reduction,
				"mean_fault_wrong_reduction_ratio": mean_fault_wrong_reduction_ratio,
				"mean_fault_retained_ratio": mean_fault_retained_ratio,
			}
	return best[0], best[1], best_info


def risk_from_outputs(outputs: Dict[str, torch.Tensor], risk_refs: Optional[Dict[str, float]]) -> torch.Tensor:
	return calibrate_risk(
		aleatoric_sigma=torch.exp(0.5 * outputs["logvar"]),
		fault_prob=torch.sigmoid(outputs["fault_logit"]),
		residual=outputs.get("residual"),
		epistemic_sigma=outputs.get("epistemic"),
		forecast=outputs.get("forecast"),
		staleness=outputs.get("staleness"),
		drift=None,  # drift 只作诊断通道，不进门控，避免 clean 上误抑制。
		refs=risk_refs,
	)


def gate_metrics_for_outputs(
	outputs: Dict[str, torch.Tensor],
	risk: torch.Tensor,
	uncertainty_ref: float,
	softness: float,
	deadband: float,
) -> Dict[str, float]:
	return torque_replay_metrics(
		outputs["mean"],
		outputs["y"],
		outputs["logvar"],
		outputs["fault_logit"],
		outputs["mask"],
		uncertainty_ref,
		gate_softness=softness,
		risk=risk,
		deadband=deadband,
	)


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
				ignore_history=max(model.get_effective_history(), args.eval_ignore_history),
			)
	return results


def load_ensemble_models(
	args: argparse.Namespace,
	input_names: List[str],
	target_names: List[str],
	device: torch.device,
) -> List[ReliabilityTCN]:
	"""加载 --ensemble-checkpoints 指定的额外 ReliabilityTCN，用于 deep ensemble epistemic。

	跨独立训练模型的预测方差通常比单模型 MC-dropout 更能反映 OOD。
	输入/标签不匹配或缺失的 checkpoint 会被跳过并告警。
	"""
	paths = _parse_csv(getattr(args, "ensemble_checkpoints", "") or "")
	models: List[ReliabilityTCN] = []
	for path in paths:
		if not Path(path).exists():
			print(f"WARNING ensemble checkpoint not found, skipping: {path}")
			continue
		try:
			extra, payload = load_checkpoint(path, map_location=device)
			validate_checkpoint_inputs(payload, input_names, target_names)
		except Exception as exc:  # noqa: BLE001 - 跳过任何不兼容 checkpoint
			print(f"WARNING failed to load ensemble checkpoint {path}: {exc}")
			continue
		models.append(extra.to(device).eval())
	return models


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
	loader = make_loader(dataset, args, device, shuffle=False)
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
			mask = apply_ignore_history(mask, ignore_history)
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
	risk_refs: Optional[Dict[str, float]] = None,
	gate_softness: Optional[float] = None,
	gate_deadband: Optional[float] = None,
) -> Dict[str, float]:
	outputs = collect_reliability_outputs(model, dataset, groups, args, device, fault_name)
	if outputs is None:
		return {}
	if uncertainty_ref is None:
		uncertainty_ref = estimate_uncertainty_ref(outputs["logvar"], outputs["mask"])
	return reliability_metrics_from_outputs(
		outputs,
		uncertainty_ref,
		gate_softness if gate_softness is not None else args.gate_softness,
		risk_refs=risk_refs,
		gate_deadband=gate_deadband if gate_deadband is not None else args.gate_deadband,
		velocity_indices=getattr(args, "velocity_indices", []),
	)


def collect_reliability_outputs(
	model: ReliabilityTCN,
	dataset: ParsedWindowDataset,
	groups: Dict[str, List[int]],
	args: argparse.Namespace,
	device: torch.device,
	fault_name: str = "clean",
) -> Optional[Dict[str, torch.Tensor]]:
	loader = make_loader(dataset, args, device, shuffle=False)
	spec = parse_fault_spec(fault_name)
	model.eval()
	means, logvars, ys, masks, fault_logits, fault_targets = [], [], [], [], [], []
	residuals, epistemics, velocities, forecasts = [], [], [], []
	stalenesses, drifts = [], []
	vel_idx = [i for i in getattr(args, "velocity_indices", []) if i >= 0]
	use_recon = getattr(model, "recon_head", None) is not None
	use_forecast = getattr(model, "forecast_head", None) is not None
	horizon = getattr(model, "forecast_horizon", 1)
	extra_models = getattr(args, "extra_models", None) or []
	with torch.no_grad():
		for batch_index, batch in enumerate(loader):
			if args.max_eval_batches and batch_index >= args.max_eval_batches:
				break
			x = batch["x"].to(device)
			y = batch["y"].to(device)
			mask = apply_ignore_history(batch["mask"].to(device), args.eval_ignore_history)
			x_eval, fault_target = apply_fault(x, groups, spec)
			out = model(x_eval)
			means.append(out["mean"].cpu())
			logvars.append(out["logvar"].cpu())
			ys.append(y.cpu())
			masks.append(mask.cpu())
			fault_logits.append(out["fault_logit"].cpu())
			fault_targets.append(fault_target.cpu())
			if use_recon and "recon" in out:
				residuals.append(reconstruction_residual(out["recon"], out["input_norm"], groups).cpu())
			if use_forecast and "forecast" in out:
				forecasts.append(forecast_residual(out["forecast"], out["input_norm"], horizon, groups).cpu())
			# 显式时序统计风险通道（逐帧 [B,1,T]，供门控按时刻抑制力矩）。
			stalenesses.append(staleness_score(out["input_norm"], groups).cpu())
			drifts.append(drift_score(out["input_norm"], groups).cpu())
			# 认知不确定性：优先用 deep ensemble 跨模型方差（更可靠），否则回退 MC-dropout。
			if extra_models:
				preds = [out["mean"]] + [em(x_eval)["mean"] for em in extra_models]
				epi = torch.stack(preds, dim=0).std(dim=0).mean(dim=1, keepdim=True)
				epistemics.append(epi.cpu())
			elif args.mc_samples and args.mc_samples > 1:
				mc = model.predict_uncertainty(x_eval, samples=args.mc_samples)
				epistemics.append(mc["epistemic_sigma"].mean(dim=1, keepdim=True).cpu())
			# 关节角速度（取自受损前的真实输入），用于 closed-loop 功率。
			if vel_idx:
				velocities.append(x[:, vel_idx, :].cpu())

	if not means:
		return None

	collected = {
		"mean": torch.cat(means),
		"logvar": torch.cat(logvars),
		"y": torch.cat(ys),
		"mask": torch.cat(masks).bool(),
		"fault_logit": torch.cat(fault_logits),
		"fault_target": torch.cat(fault_targets),
	}
	if residuals:
		collected["residual"] = torch.cat(residuals)
	if forecasts:
		collected["forecast"] = torch.cat(forecasts)
	if stalenesses:
		collected["staleness"] = torch.cat(stalenesses)
	if drifts:
		collected["drift"] = torch.cat(drifts)
	if epistemics:
		collected["epistemic"] = torch.cat(epistemics)
	if velocities:
		collected["velocity"] = torch.cat(velocities)
	return collected


def reliability_metrics_from_outputs(
	outputs: Dict[str, torch.Tensor],
	uncertainty_ref: float,
	gate_softness: float,
	risk_refs: Optional[Dict[str, float]] = None,
	gate_deadband: float = 0.0,
	velocity_indices: Optional[List[int]] = None,
) -> Dict[str, float]:
	mean = outputs["mean"]
	logvar = outputs["logvar"]
	y = outputs["y"]
	mask = outputs["mask"]
	fault_logit = outputs["fault_logit"]
	fault_target = outputs["fault_target"]
	residual = outputs.get("residual")
	epistemic = outputs.get("epistemic")
	forecast = outputs.get("forecast")
	staleness = outputs.get("staleness")
	drift = outputs.get("drift")
	metrics = regression_metrics(mean, y, mask)
	metrics["nll"] = float(masked_gaussian_nll(mean, logvar, y, mask).detach().cpu())
	metrics["uncertainty_error_corr"] = uncertainty_error_correlation(mean, logvar, y, mask)
	metrics.update(interval_calibration(mean, logvar, y, mask))
	time_mask = mask.any(dim=1, keepdim=True)
	metrics["fault_auroc"] = binary_auroc(torch.sigmoid(fault_logit)[time_mask], fault_target[time_mask])
	# 标定的多信号融合风险。staleness 接入门控（clean=0 无误伤，对 packet_loss/sensor_delay
	# 大幅抑制）；drift 不接入门控——它检测 imu_bias 的能力仅 ~0.51（漂移在分布内），接入只会
	# 在 clean 上无谓拉低 mean_gate，故仅作检测通道如实报告 imu_bias 限制。
	risk = calibrate_risk(
		aleatoric_sigma=torch.exp(0.5 * logvar),
		fault_prob=torch.sigmoid(fault_logit),
		residual=residual,
		epistemic_sigma=epistemic,
		forecast=forecast,
		staleness=staleness,
		drift=None,
		refs=risk_refs,
	)
	if risk_refs is not None:
		metrics["mean_calibrated_risk"] = float(risk[time_mask].mean().detach().cpu()) if time_mask.any() else float("nan")
	metrics.update(
		torque_replay_metrics(
			mean, y, logvar, fault_logit, mask, uncertainty_ref,
			gate_softness=gate_softness,
			risk=risk if risk_refs is not None else None,
			deadband=gate_deadband,
		)
	)
	# 轻量动力学在环代理收益（门控后的指令施加到真实生物力矩上）。
	vel_idx = [i for i in (velocity_indices or []) if i >= 0]
	if "velocity" in outputs and len(vel_idx) == mean.shape[1]:
		from reliability.metrics import moment_to_torque, reliability_gate
		gate = reliability_gate(risk, softness=gate_softness, deadband=gate_deadband)
		command = moment_to_torque(mean) * gate
		metrics.update(
			closed_loop_metrics(
				true_moment=y,
				command_torque=command,
				joint_velocity=outputs["velocity"],
				mask=mask,
			)
		)
	return metrics


def evaluate_gate_sweep(
	model: ReliabilityTCN,
	datasets: Dict[str, ParsedWindowDataset],
	groups: Dict[str, List[int]],
	args: argparse.Namespace,
	device: torch.device,
	uncertainty_ref: float,
	risk_refs: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
	results: Dict[str, Dict[str, float]] = {}
	softness_values = _parse_floats(args.gate_softness_grid)
	sweep_deadband = getattr(args, "selected_gate_deadband", args.gate_deadband)
	for scenario in _parse_csv(args.gate_sweep_scenarios):
		if "/" not in scenario:
			raise ValueError(f"Gate sweep scenario must be split/fault, got: {scenario}")
		split_name, fault_name = scenario.split("/", 1)
		dataset = datasets.get(split_name)
		if dataset is None:
			continue
		outputs = collect_reliability_outputs(model, dataset, groups, args, device, fault_name)
		if outputs is None:
			continue
		risk = calibrate_risk(
			aleatoric_sigma=torch.exp(0.5 * outputs["logvar"]),
			fault_prob=torch.sigmoid(outputs["fault_logit"]),
			residual=outputs.get("residual"),
			epistemic_sigma=outputs.get("epistemic"),
			forecast=outputs.get("forecast"),
			staleness=outputs.get("staleness"),
			drift=None,  # drift 不进门控（见 reliability_metrics_from_outputs 注释）。
			refs=risk_refs,
		) if risk_refs is not None else None
		for softness in softness_values:
			key = f"gate_sweep/{split_name}/{fault_name}/softness_{_format_float(softness)}"
			metrics = torque_replay_metrics(
				outputs["mean"],
				outputs["y"],
				outputs["logvar"],
				outputs["fault_logit"],
				outputs["mask"],
				uncertainty_ref,
				gate_softness=softness,
				risk=risk,
				deadband=sweep_deadband,
			)
			metrics["gate_deadband"] = sweep_deadband
			results[key] = metrics
	return results


def collect_detector_scores(
	model: ReliabilityTCN,
	dataset: ParsedWindowDataset,
	groups: Dict[str, List[int]],
	args: argparse.Namespace,
	device: torch.device,
	fault_name: str,
) -> Dict[str, torch.Tensor]:
	"""每个窗口返回多路检测分数：监督 logit、aleatoric σ、重构残差、预测残差、epistemic σ。

	这支撑 E1/E2：直接对比"监督故障头"与"重构残差/预测残差/认知不确定性"的可检测性。
	预测残差(forecast)是本轮新增，针对 packet_loss/sensor_delay/imu_bias 这类时序结构故障。
	"""
	loader = make_loader(dataset, args, device, shuffle=False)
	spec = parse_fault_spec(fault_name)
	model.eval()
	use_recon = getattr(model, "recon_head", None) is not None
	use_forecast = getattr(model, "forecast_head", None) is not None
	horizon = getattr(model, "forecast_horizon", 1)
	extra_models = getattr(args, "extra_models", None) or []
	channels: Dict[str, List[torch.Tensor]] = {"logit": [], "aleatoric": [], "residual": [], "forecast": [], "epistemic": [], "staleness": [], "coherence": [], "drift": []}
	with torch.no_grad():
		for batch_index, batch in enumerate(loader):
			if args.max_eval_batches and batch_index >= args.max_eval_batches:
				break
			x = batch["x"].to(device)
			mask = apply_ignore_history(batch["mask"].to(device), args.eval_ignore_history).all(dim=1, keepdim=True)
			x_eval, _ = apply_fault(x, groups, spec)
			out = model(x_eval)
			channels["logit"].append(masked_time_mean(torch.sigmoid(out["fault_logit"]), mask).cpu())
			channels["aleatoric"].append(masked_time_mean(torch.exp(0.5 * out["logvar"]).mean(dim=1, keepdim=True), mask).cpu())
			# 显式统计检测通道：不依赖训练头，直接看归一化输入的时序结构。
			# staleness 抓 packet loss，coherence 抓跨 IMU 不同步；drift 仅保留为 IMU-bias 诊断。
			channels["staleness"].append(masked_time_mean(staleness_score(out["input_norm"], groups), mask).cpu())
			channels["coherence"].append(masked_time_mean(cross_modal_coherence_score(out["input_norm"], groups), mask).cpu())
			channels["drift"].append(masked_time_mean(drift_score(out["input_norm"], groups), mask).cpu())
			if use_recon and "recon" in out:
				residual = reconstruction_residual(out["recon"], out["input_norm"], groups)
				channels["residual"].append(masked_time_mean(residual, mask).cpu())
			if use_forecast and "forecast" in out:
				fc = forecast_residual(out["forecast"], out["input_norm"], horizon, groups)
				channels["forecast"].append(masked_time_mean(fc, mask).cpu())
			if extra_models:
				preds = [out["mean"]] + [em(x_eval)["mean"] for em in extra_models]
				epi = torch.stack(preds, dim=0).std(dim=0).mean(dim=1, keepdim=True)
				channels["epistemic"].append(masked_time_mean(epi, mask).cpu())
			elif args.mc_samples and args.mc_samples > 1:
				mc = model.predict_uncertainty(x_eval, samples=args.mc_samples)
				channels["epistemic"].append(masked_time_mean(mc["epistemic_sigma"].mean(dim=1, keepdim=True), mask).cpu())
	return {name: torch.cat(parts) for name, parts in channels.items() if parts}


def combined_detector_score(
	scores: Dict[str, torch.Tensor],
	refs: Optional[Dict[str, float]] = None,
	signal_names: Optional[Iterable[str]] = None,
) -> torch.Tensor:
	"""把可用检测分数按参考归一后取最大，作为统一风险分数。"""
	refs = refs or {}
	selected = set(signal_names) if signal_names is not None else None
	parts = []
	for name, value in scores.items():
		if selected is not None and name not in selected:
			continue
		ref = refs.get(name)
		if ref is None:
			ref = float(value.median()) if value.numel() else 1.0
		parts.append(value / max(ref, 1e-6))
	if not parts:
		return torch.empty(0)
	combined = parts[0]
	for other in parts[1:]:
		combined = torch.maximum(combined, other)
	return combined


def select_detector_signal_subset(
	clean: Dict[str, torch.Tensor],
	faults: Dict[str, Dict[str, torch.Tensor]],
	refs: Optional[Dict[str, float]],
	max_signals: int = 3,
) -> tuple[List[str], Dict[str, object]]:
	"""Select a compact detector fusion using validation clean/fault labels only."""
	available = [
		name
		for name in ("logit", "aleatoric", "residual", "forecast", "epistemic", "staleness", "coherence", "drift")
		if name in clean and all(name in scores for scores in faults.values())
	]
	if not available or not faults:
		return available, {"policy_source": "all_available_no_validation_faults"}

	best_signals = available
	best_score = -math.inf
	best_aurocs: Dict[str, float] = {}
	for size in range(1, min(max_signals, len(available)) + 1):
		for candidate in combinations(available, size):
			clean_score = combined_detector_score(clean, refs, candidate)
			aurocs = {}
			for fault_name, fault_scores in faults.items():
				fault_score = combined_detector_score(fault_scores, refs, candidate)
				scores = torch.cat([clean_score, fault_score])
				labels = torch.cat([torch.zeros_like(clean_score), torch.ones_like(fault_score)])
				aurocs[fault_name] = binary_auroc(scores, labels)
			finite = [value for value in aurocs.values() if math.isfinite(value)]
			if not finite:
				continue
			mean_auroc = sum(finite) / len(finite)
			worst_auroc = min(finite)
			# Favor broad fault coverage and compact policies; the test set is never consulted.
			score = mean_auroc + 0.25 * worst_auroc - 0.002 * (size - 1)
			if score > best_score:
				best_score = score
				best_signals = list(candidate)
				best_aurocs = aurocs
	return best_signals, {
		"policy_source": "validation_fault_subset_search",
		"signals": best_signals,
		"selection_score": best_score,
		"validation_aurocs": best_aurocs,
		"max_signals": max_signals,
	}


def collect_risk_scores(
	model: ReliabilityTCN,
	dataset: ParsedWindowDataset,
	groups: Dict[str, List[int]],
	args: argparse.Namespace,
	device: torch.device,
	fault_name: str,
) -> torch.Tensor:
	scores = collect_detector_scores(model, dataset, groups, args, device, fault_name)
	return combined_detector_score(scores)


def collect_fault_scores(
	model: ReliabilityTCN,
	dataset: ParsedWindowDataset,
	groups: Dict[str, List[int]],
	args: argparse.Namespace,
	device: torch.device,
	fault_name: str,
) -> torch.Tensor:
	"""向后兼容：仅返回监督 fault logit 分数（用于消融对照）。"""
	scores = collect_detector_scores(model, dataset, groups, args, device, fault_name)
	return scores.get("logit", torch.empty(0))


def collect_uncertainty_ref(
	model: ReliabilityTCN,
	dataset: ParsedWindowDataset,
	groups: Dict[str, List[int]],
	args: argparse.Namespace,
	device: torch.device,
) -> float:
	loader = make_loader(dataset, args, device, shuffle=False)
	model.eval()
	values = []
	with torch.no_grad():
		for batch_index, batch in enumerate(loader):
			if args.max_eval_batches and batch_index >= args.max_eval_batches:
				break
			x = batch["x"].to(device)
			mask = apply_ignore_history(batch["mask"].to(device), args.eval_ignore_history)
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


def apply_ignore_history(mask: torch.Tensor, ignore_history: int) -> torch.Tensor:
	if ignore_history <= 0:
		return mask
	mask = mask.clone()
	usable_history = min(ignore_history, max(mask.shape[-1] - 1, 0))
	mask[..., :usable_history] = False
	return mask


def masked_time_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	mask = mask.to(dtype=values.dtype)
	denom = mask.sum(dim=(1, 2)).clamp_min(1.0)
	return (values * mask).sum(dim=(1, 2)) / denom


def print_run_summary(
	args: argparse.Namespace,
	input_names: List[str],
	target_names: List[str],
	splits: Dict[str, list],
	datasets: Dict[str, ParsedWindowDataset],
) -> None:
	print("")
	print("Reliability-TCN")
	print("-" * 72)
	print(
		" ".join(
			[
				f"mode={args.mode}",
				f"device={args.device}",
				f"profile={args.input_profile}",
				f"epochs={args.epochs}",
				f"batch={args.batch_size}",
				f"window={args.window_size}",
				f"stride={args.stride}",
			]
		)
	)
	print(
		" ".join(
			[
				f"channels={args.num_channels}",
				f"kernel={args.kernel_size}",
				f"dropout={args.dropout:g}",
				f"spatial_dropout={args.spatial_dropout}",
			]
		)
	)
	print(f"data={args.data_root}")
	print(f"outputs={args.output_dir}")
	print(f"inputs={len(input_names)} labels={len(target_names)} heldout={args.heldout_tasks}")
	if args.training_faults:
		print(f"training_faults={args.training_faults}")
		print(
			f"clean_sample_prob={args.clean_sample_prob:g} fault_positive_weight="
			f"{'auto' if args.fault_positive_weight <= 0 else f'{args.fault_positive_weight:g}'}"
		)
	print(f"aux_detach=recon:{args.recon_detach},forecast:{args.forecast_detach}")
	if args.fault_scenarios:
		print(f"eval_faults={args.fault_scenarios}")
	print("")
	print_table(
		["split", "trials", "windows", "subjects", "tasks", "participants"],
		[
			[
				name,
				len(split),
				len(datasets[name]) if name in datasets else 0,
				len(summary["participants"]),
				len(summary["tasks"]),
				_short_list(summary["participants"], limit=5),
			]
			for name, split in splits.items()
			for summary in [summarize_records(split)]
		],
	)
	sys.stdout.flush()


def print_training_header(args: argparse.Namespace) -> None:
	print("")
	print(f"Training  [mode={args.training_mode}  recon={mode_flags(args.training_mode)['use_recon']}  mc={args.mc_samples}]")
	print("-" * 112)
	print(
		_format_row(
			[
				"Epoch",
				"GPU_mem",
				"loss",
				"val_rmse",
				"val_R2",
				"val_ECE",
				"err_corr",
				"cl_net↓",
				"gate",
				"risk",
				"best",
			],
			[10, 9, 11, 10, 9, 9, 9, 9, 8, 8, 6],
		)
	)
	sys.stdout.flush()


class TrainingProgress:
	"""YOLO-style live progress without adding a tqdm dependency."""

	def __init__(
		self,
		epoch: int,
		epochs: int,
		total_batches: int,
		args: argparse.Namespace,
		device: torch.device,
	) -> None:
		self.epoch = epoch
		self.epochs = epochs
		self.total_batches = max(total_batches, 1)
		self.device = device
		self.enabled = args.progress_style != "off" and args.log_interval != 0 and total_batches > 0
		self.style = self._resolve_style(args.progress_style)
		self.line_interval = max(args.log_interval, 1)
		self.start_time = time.monotonic()
		self.last_print_time = 0.0
		self.closed = False

	def update(self, batch_index: int, loss: float, avg_loss: float) -> None:
		if not self.enabled:
			return
		final = batch_index >= self.total_batches
		now = time.monotonic()
		if self.style == "line" and not final and batch_index % self.line_interval != 0:
			return
		if self.style == "bar" and not final and now - self.last_print_time < 0.10:
			return
		self.last_print_time = now
		message = self._render(batch_index, loss, avg_loss, now)
		if self.style == "bar":
			print("\r" + message, end="", flush=True)
		else:
			print(message, flush=True)

	def close(self) -> None:
		if self.enabled and self.style == "bar" and not self.closed:
			print("", flush=True)
		self.closed = True

	def _resolve_style(self, requested: str) -> str:
		if requested == "auto":
			return "bar" if sys.stdout.isatty() else "line"
		return requested

	def _render(self, batch_index: int, loss: float, avg_loss: float, now: float) -> str:
		elapsed = max(now - self.start_time, 1e-6)
		progress = min(max(batch_index / self.total_batches, 0.0), 1.0)
		speed = batch_index / elapsed
		remaining = (self.total_batches - batch_index) / speed if speed > 0 else math.inf
		prefix = f"{self.epoch:>3}/{self.epochs:<3}"
		body = (
			f"{batch_index:>4}/{self.total_batches:<4} "
			f"{progress * 100:>5.1f}% "
			f"GPU_mem {gpu_memory(self.device):>5} "
			f"loss {_fmt_float(loss, 4):>9} "
			f"avg {_fmt_float(avg_loss, 4):>9} "
			f"ETA {_format_duration(remaining):>8}"
		)
		if self.style == "bar":
			return f"{prefix} [{self._bar(progress)}] {body}"
		return f"{prefix} {body}"

	@staticmethod
	def _bar(progress: float, width: int = 24) -> str:
		filled = min(width, max(0, int(round(progress * width))))
		return "#" * filled + "." * (width - filled)


def print_epoch_row(epoch_info: Dict[str, float], args: argparse.Namespace, device: torch.device, is_best: bool) -> None:
	row = [
		f"{int(epoch_info['epoch'])}/{args.epochs}",
		gpu_memory(device),
		_fmt_float(epoch_info.get("train_loss"), 4),
		_fmt_float(epoch_info.get("val_rmse"), 4),
		_fmt_float(epoch_info.get("val_r2"), 3),
		_fmt_float(epoch_info.get("val_coverage_ece"), 3),
		_fmt_float(epoch_info.get("val_uncertainty_error_corr"), 3),
		_fmt_float(epoch_info.get("val_cl_net_moment_rms_reduction_ratio"), 3),
		_fmt_float(epoch_info.get("val_mean_gate"), 3),
		_fmt_float(epoch_info.get("val_mean_risk"), 3),
		"*" if is_best else "",
	]
	print(_format_row(row, [10, 9, 11, 10, 9, 9, 9, 9, 8, 8, 6]))


def print_result_summary(results: Dict[str, Dict[str, float]], args: argparse.Namespace) -> None:
	print("")
	print("Results")
	print("-" * 100)
	scenarios = [
		"test_id/clean",
		"test_ood/clean",
		"test_id/insole_missing",
		"test_id/encoder_dropout",
		"test_id/packet_loss",
		"test_id/packet_loss_burst",
		"test_id/packet_loss_partial",
		"test_id/sensor_delay",
		"test_id/sensor_delay_jitter",
		"test_ood/insole_missing",
		"test_ood/encoder_dropout",
		"test_ood/packet_loss",
		"test_ood/packet_loss_burst",
		"test_ood/packet_loss_partial",
		"test_ood/sensor_delay",
		"test_ood/sensor_delay_jitter",
	]
	rows = []
	for scenario in scenarios:
		metrics = results.get(scenario)
		if not metrics:
			continue
		baseline = results.get(f"nature_baseline/{scenario}", {})
		rows.append(
			[
				scenario,
				_fmt_float(metrics.get("rmse"), 4),
				_fmt_float(metrics.get("r2"), 3),
				_fmt_float(metrics.get("gated_wrong_direction_ratio"), 3),
				_fmt_float(metrics.get("gated_retained_aligned_torque"), 3),
				_fmt_float(baseline.get("rmse"), 4),
				_fmt_delta(metrics.get("rmse"), baseline.get("rmse"), 4),
			]
		)
	if rows:
		print_table(["scenario", "rmse", "R2", "wrong", "retained", "nature", "d_rmse"], rows)

	detection_rows = []
	for key in [
		"fault_detection/test_id/insole_missing",
		"fault_detection/test_id/encoder_dropout",
		"fault_detection/test_id/imu_bias",
		"fault_detection/test_id/packet_loss",
		"fault_detection/test_id/packet_loss_burst",
		"fault_detection/test_id/packet_loss_partial",
		"fault_detection/test_id/sensor_delay",
		"fault_detection/test_id/sensor_delay_jitter",
		"ood_detection/clean",
	]:
		metrics = results.get(key)
		if not metrics:
			continue
		# 每路信号分别报，凸显 residual/forecast/epistemic 相对 supervised logit 的增益（E1/E2）。
		detection_rows.append([
			key.replace("fault_detection/", "").replace("ood_detection/", "OOD:"),
			_fmt_float(metrics.get("auroc_logit"), 3),
			_fmt_float(metrics.get("auroc_aleatoric"), 3),
			_fmt_float(metrics.get("auroc_residual"), 3),
			_fmt_float(metrics.get("auroc_forecast"), 3),
			_fmt_float(metrics.get("auroc_epistemic"), 3),
			_fmt_float(metrics.get("auroc_coherence"), 3),
			_fmt_float(metrics.get("fault_auroc", metrics.get("risk_auroc")), 3),
		])
	if detection_rows:
		print("")
		print_table(["detector", "logit", "aleat", "resid", "fcast", "epist", "coher", "fused"], detection_rows)

	cl_rows = []
	for scenario in ["test_id/clean", "test_ood/clean", "test_id/encoder_dropout", "test_ood/sensor_delay"]:
		metrics = results.get(scenario)
		if not metrics or "cl_net_moment_rms_reduction_ratio" not in metrics:
			continue
		cl_rows.append([
			scenario,
			_fmt_float(metrics.get("cl_net_moment_rms_reduction_ratio"), 3),
			_fmt_float(metrics.get("cl_abs_work_reduction_ratio"), 3),
			_fmt_float(metrics.get("cl_fight_fraction"), 3),
			_fmt_float(metrics.get("cl_peak_net_moment_assisted"), 3),
		])
	if cl_rows:
		print("")
		print("Closed-loop proxy (net-moment↓ / work↓ are benefit, fight is harm)")
		print_table(["scenario", "net_rms↓", "work↓", "fight", "peak_net"], cl_rows)

	gate_rows = []
	for key, metrics in sorted(results.items()):
		if key.startswith("gate_sweep/"):
			parts = key.split("/")
			if len(parts) == 4:
				gate_rows.append(
					[
						f"{parts[1]}/{parts[2]}",
						parts[3].replace("softness_", "").replace("p", "."),
						_fmt_float(metrics.get("gated_wrong_direction_ratio"), 3),
						_fmt_float(metrics.get("gated_retained_aligned_torque"), 3),
						_fmt_float(metrics.get("mean_gate"), 3),
					]
				)
	if gate_rows:
		print("")
		print_table(["gate_scenario", "soft", "wrong", "retained", "gate"], gate_rows[:16])
		if len(gate_rows) > 16:
			print(f"... {len(gate_rows) - 16} more gate-sweep rows saved in the JSON report.")


def print_table(headers: List[str], rows: List[List[object]]) -> None:
	widths = [len(header) for header in headers]
	for row in rows:
		for idx, value in enumerate(row):
			widths[idx] = max(widths[idx], len(str(value)))
	print(_format_row(headers, widths))
	print(_format_row(["-" * width for width in widths], widths))
	for row in rows:
		print(_format_row(row, widths))


def _format_row(values: List[object], widths: List[int]) -> str:
	return "  ".join(str(value).rjust(width) for value, width in zip(values, widths))


def _short_list(values: List[str], limit: int = 5) -> str:
	if len(values) <= limit:
		return ",".join(values)
	return ",".join(values[:limit]) + f",+{len(values) - limit}"


def _fmt_float(value: object, digits: int = 3) -> str:
	if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
		return "-"
	return f"{float(value):.{digits}f}"


def _fmt_delta(value: object, baseline: object, digits: int = 3) -> str:
	if not isinstance(value, (float, int)) or not isinstance(baseline, (float, int)):
		return "-"
	if not math.isfinite(float(value)) or not math.isfinite(float(baseline)):
		return "-"
	return f"{float(value) - float(baseline):+.{digits}f}"


def _format_duration(seconds: float) -> str:
	if not math.isfinite(seconds):
		return "--:--"
	seconds = max(int(seconds), 0)
	hours, rem = divmod(seconds, 3600)
	minutes, secs = divmod(rem, 60)
	if hours:
		return f"{hours:d}:{minutes:02d}:{secs:02d}"
	return f"{minutes:02d}:{secs:02d}"


def gpu_memory(device: torch.device) -> str:
	if device.type != "cuda" or not torch.cuda.is_available():
		return "-"
	index = device.index if device.index is not None else torch.cuda.current_device()
	memory_gb = torch.cuda.max_memory_reserved(index) / (1024 ** 3)
	return f"{memory_gb:.1f}G"


def configure_terminal_warnings() -> None:
	warnings.filterwarnings("ignore", message="`torch.nn.utils.weight_norm` is deprecated.*")
	warnings.filterwarnings("ignore", message="dropout2d: Received a 3D input.*")


def set_seed(seed: int) -> None:
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def configure_backend(args: argparse.Namespace) -> None:
	"""开启 A100 上的 TF32 matmul/cudnn，并根据设备校正 AMP 设置。"""
	if torch.cuda.is_available() and torch.device(args.device).type == "cuda":
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True
		torch.backends.cudnn.benchmark = True
	else:
		# CPU 上没有 AMP 收益，强制关闭以免 autocast 报错。
		args.amp = "off"


def amp_settings(args: argparse.Namespace, device: torch.device) -> tuple[bool, "torch.dtype | None"]:
	"""返回 (是否启用 autocast, autocast dtype)。"""
	if device.type != "cuda" or getattr(args, "amp", "off") == "off":
		return False, None
	if args.amp == "fp16":
		return True, torch.float16
	return True, torch.bfloat16


def make_loader(
	dataset: ParsedWindowDataset,
	args: argparse.Namespace,
	device: torch.device,
	shuffle: bool,
) -> DataLoader:
	"""统一构造 DataLoader：多 worker + pin_memory + 持久 worker，喂饱 GPU。"""
	num_workers = max(int(getattr(args, "num_workers", 0)), 0)
	kwargs: dict = {
		"batch_size": args.batch_size,
		"shuffle": shuffle,
		"drop_last": False,
		"num_workers": num_workers,
		"pin_memory": device.type == "cuda",
	}
	if num_workers > 0:
		kwargs["persistent_workers"] = True
		kwargs["prefetch_factor"] = max(int(getattr(args, "prefetch_factor", 2)), 1)
	return DataLoader(dataset, **kwargs)


def _serializable_args(args: argparse.Namespace) -> dict:
	"""把 args 转成可 JSON 序列化的字典，剔除运行期注入的非数据字段（如加载的集成模型）。"""
	skip = {"extra_models"}
	result = {}
	for key, value in vars(args).items():
		if key in skip:
			continue
		if isinstance(value, (str, int, float, bool, type(None), list, dict)):
			result[key] = value
		else:
			result[key] = str(value)
	return result


def _parse_csv(value: str) -> List[str]:
	return [item.strip() for item in value.split(",") if item.strip()]


def _parse_channels(value: str) -> List[int]:
	return [int(item) for item in _parse_csv(value)]


def _parse_floats(value: str) -> List[float]:
	return [float(item) for item in _parse_csv(value)]


def _format_float(value: float) -> str:
	return f"{value:g}".replace("-", "m").replace(".", "p")


if __name__ == "__main__":
	main()
