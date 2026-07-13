"""验证 staleness/drift 接入门控后的效果：对比"接入 vs 不接入"的门控指标。

关键检查：
 (a) clean 上 mean_gate 不被新通道拉低（无误伤）。
 (b) packet_loss/sensor_delay 上 gated wrong-direction 下降、mean_gate 下降（门控现在能看见并抑制）。
全程 eval-only，复用已有 checkpoint，不重训。
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_reliability_experiment as R
from reliability.features import feature_groups, input_names_for_profile, joint_velocity_indices, label_names
from reliability.model import load_checkpoint
from reliability.nature_dataset import discover_trials, filter_records
from reliability.metrics import calibrate_risk, torque_replay_metrics


def gate_metrics(outputs, refs, unc_ref, softness, deadband, use_temporal):
	risk = calibrate_risk(
		aleatoric_sigma=torch.exp(0.5 * outputs["logvar"]),
		fault_prob=torch.sigmoid(outputs["fault_logit"]),
		residual=outputs.get("residual"),
		epistemic_sigma=outputs.get("epistemic"),
		forecast=outputs.get("forecast"),
		staleness=outputs.get("staleness") if use_temporal else None,
		drift=None,  # drift 不接入门控（仅检测通道）。
		refs=refs,
	)
	m = torque_replay_metrics(
		outputs["mean"], outputs["y"], outputs["logvar"], outputs["fault_logit"],
		outputs["mask"], unc_ref, gate_softness=softness, risk=risk, deadband=deadband,
	)
	return m


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--checkpoint", default="reports/v2_main_seed7/reliability_tcn_best.pt")
	ap.add_argument("--faults", default="clean,packet_loss,sensor_delay,imu_bias,encoder_dropout,insole_missing")
	cli = ap.parse_args()

	saved = sys.argv
	sys.argv = ["v", "--checkpoint", cli.checkpoint, "--mode", "eval"]
	args = R.parse_args()
	sys.argv = saved
	args.device = "cuda" if torch.cuda.is_available() else "cpu"
	args.max_eval_batches = 0

	input_names = input_names_for_profile(args.input_profile, args.side)
	target_names = label_names(args.side)
	groups = feature_groups(input_names)
	args.velocity_indices = joint_velocity_indices(input_names, target_names)
	records = discover_trials(args.data_root)
	heldout = R._parse_csv(args.heldout_tasks)
	_, val_p, test_p = R.resolve_split(records, args)
	val = R.make_dataset(filter_records(records, participants=val_p, exclude_tasks=heldout), input_names, target_names, args)
	tid = R.make_dataset(filter_records(records, participants=test_p, exclude_tasks=heldout), input_names, target_names, args)

	device = torch.device(args.device)
	model, _ = load_checkpoint(args.checkpoint, map_location=device)
	model = model.to(device).eval()
	args.extra_models = []

	refs = R.estimate_risk_refs(model, val, groups, args, device)
	unc_ref = R.collect_uncertainty_ref(model, val, groups, args, device)
	softness, deadband = R.select_gate_on_val(model, {"val": val}, groups, args, device, unc_ref, refs)
	print(f"selected gate: softness={softness} deadband={deadband}")
	print(f"refs staleness={refs.get('staleness')} drift={refs.get('drift')}")

	faults = [f.strip() for f in cli.faults.split(",") if f.strip()]
	print(f"\n{'fault':<16}{'gate_OFF':>22}{'gate_ON(+temporal)':>26}")
	print(f"{'':<16}{'wrong / mean_gate / retain':>22}{'wrong / mean_gate / retain':>26}")
	for fault in faults:
		out = R.collect_reliability_outputs(model, tid, groups, args, device, fault)
		if out is None:
			continue
		off = gate_metrics(out, refs, unc_ref, softness, deadband, use_temporal=False)
		on = gate_metrics(out, refs, unc_ref, softness, deadband, use_temporal=True)
		def fmt(m):
			return f"{m['gated_wrong_direction_ratio']:.3f}/{m['mean_gate']:.3f}/{m['gated_retained_aligned_torque']:.3f}"
		print(f"{fault:<16}{fmt(off):>22}{fmt(on):>26}")


if __name__ == "__main__":
	main()
