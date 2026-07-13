"""验证显式时序检测通道——在已有 checkpoint 上离线评估，不重训。

直接复用 run_reliability_experiment 的数据/模型/打分函数，对 test_id、test_ood 的
packet_loss、sensor_delay 与 imu_bias 计算逐通道 AUROC。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_reliability_experiment as R
from reliability.features import feature_groups, input_names_for_profile, joint_velocity_indices, label_names
from reliability.model import load_checkpoint
from reliability.nature_dataset import discover_trials, filter_records


def build_args(report_path: str, checkpoint: str, device: str, batch_size: int) -> argparse.Namespace:
	saved = sys.argv
	sys.argv = ["validate", "--checkpoint", checkpoint, "--mode", "eval"]
	try:
		args = R.parse_args()
	finally:
		sys.argv = saved
	report = json.loads(Path(report_path).read_text(encoding="utf-8"))
	for key, value in report.get("args", {}).items():
		if hasattr(args, key):
			setattr(args, key, value)
	args.checkpoint = checkpoint
	args.device = device
	args.batch_size = batch_size
	args.mc_samples = 0
	args.max_eval_batches = 0
	return args


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--report", default="reports/v2_main_seed7/reliability_report.json")
	ap.add_argument("--checkpoint", default="reports/v2_main_seed7/reliability_tcn_best.pt")
	ap.add_argument("--faults", default="packet_loss,packet_loss_partial,imu_bias,sensor_delay,sensor_delay_jitter,encoder_dropout,insole_missing")
	ap.add_argument("--selection-faults", default="insole_missing,encoder_dropout,imu_bias,packet_loss,sensor_delay")
	ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	ap.add_argument("--batch-size", type=int, default=128)
	cli = ap.parse_args()

	args = build_args(cli.report, cli.checkpoint, cli.device, cli.batch_size)
	input_names = input_names_for_profile(args.input_profile, args.side)
	target_names = label_names(args.side)
	groups = feature_groups(input_names)
	args.velocity_indices = joint_velocity_indices(input_names, target_names)

	records = discover_trials(args.data_root)
	heldout = R._parse_csv(args.heldout_tasks)
	train_p, val_p, test_p = R.resolve_split(records, args)
	splits = {
		"val": filter_records(records, participants=val_p, exclude_tasks=heldout),
		"test_id": filter_records(records, participants=test_p, exclude_tasks=heldout),
		"test_ood": filter_records(records, participants=test_p, include_tasks=heldout),
	}
	datasets = {}
	for name, split in splits.items():
		if not split:
			continue
		ds = R.make_dataset(split, input_names, target_names, args)
		if len(ds) > 0:
			datasets[name] = ds

	device = torch.device(args.device)
	model, payload = load_checkpoint(args.checkpoint, map_location=device)
	model = model.to(device)
	args.extra_models = []
	print(f"Loaded {args.checkpoint} | device={device}")
	print(f"use_recon={getattr(model,'recon_head',None) is not None} use_forecast={getattr(model,'forecast_head',None) is not None}")

	# 干净 val 上的参考分位，用于 combined score 归一化。
	refs = R.estimate_risk_refs(model, datasets["val"], groups, args, device)
	print("risk_refs:", {k: round(v, 4) for k, v in refs.items()})
	selection_faults = [f.strip() for f in cli.selection_faults.split(",") if f.strip()]
	validation_clean = R.collect_detector_scores(model, datasets["val"], groups, args, device, "clean")
	validation_faults = {
		fault: R.collect_detector_scores(model, datasets["val"], groups, args, device, fault)
		for fault in selection_faults
	}
	selected_signals, selection_info = R.select_detector_signal_subset(validation_clean, validation_faults, refs)
	print("selected_signals:", selected_signals)
	print("selection_validation_aurocs:", {k: round(v, 4) for k, v in selection_info.get("validation_aurocs", {}).items()})

	faults = [f.strip() for f in cli.faults.split(",") if f.strip()]
	chans = ("residual", "forecast", "staleness", "coherence", "drift", "logit", "aleatoric")
	for split_name in ("test_id", "test_ood"):
		if split_name not in datasets:
			continue
		print(f"\n===== {split_name} =====")
		clean = R.collect_detector_scores(model, datasets[split_name], groups, args, device, "clean")
		header = f"{'fault':<18}" + "".join(f"{c[:6]:>9}" for c in chans) + f"{'COMB':>9}"
		print(header)
		for fault in faults:
			faulty = R.collect_detector_scores(model, datasets[split_name], groups, args, device, fault)
			m = R.detector_aurocs(clean, faulty, refs, signal_names=selected_signals)
			row = f"{fault:<18}"
			for c in chans:
				v = m.get(f"auroc_{c}")
				row += f"{v:>9.3f}" if v is not None else f"{'-':>9}"
			comb = m.get("fault_auroc")
			row += f"{comb:>9.3f}" if comb is not None else f"{'-':>9}"
			print(row)


if __name__ == "__main__":
	main()
