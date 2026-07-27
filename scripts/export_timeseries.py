#!/usr/bin/env python3
"""任务 I：导出单个试验的逐 timestep 时序（prediction / risk / gate / commanded torque）。

`fig_method_overview` 的面板 (c) 目前是示意曲线。本脚本从已有 checkpoint 复算一段真实
窗口的时序并写 `.npz`，供出图脚本替换示意数据。

设计约束（与主评测保持同口径，避免图与表互相矛盾）：
  * 只做 eval，不训练，不改任何已有报告；
  * 风险参考分位 (`risk_refs`) 与不确定性参考仍在干净 val 上估计；
  * 门控 softness / deadband 默认从对应的 `reliability_report.json` 里读回**已冻结**的
    `_gate_policy`，而不是重新在 val 上选，这样导出的曲线就是论文报告的那个门控；
  * 融合风险的信号集合同样从报告的 `_detector_policy` 读回。

用法：
    python scripts/export_timeseries.py \
        --checkpoint reports/v2_main_seed7/reliability_tcn_best.pt \
        --report reports/v2_fc_main_seed7/reliability_report.json \
        --split test_id --fault sensor_delay \
        --output aaai27/figures/data/timeseries_test_id_sensor_delay.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_reliability_experiment as rre
from reliability.faults import apply_fault, parse_fault_spec
from reliability.features import (
	feature_groups,
	input_names_for_profile,
	joint_velocity_indices,
	label_names,
)
from reliability.metrics import (
	drift_score,
	forecast_residual,
	moment_to_torque,
	reconstruction_residual,
	reliability_gate,
	staleness_score,
)
from reliability.model import load_checkpoint
from reliability.nature_dataset import discover_trials, filter_records


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--checkpoint", required=True)
	parser.add_argument("--report", required=True, help="同一模型的 reliability_report.json，用于读回冻结的门控与检测器策略")
	parser.add_argument("--data-root", default="data/Parsed")
	parser.add_argument("--split", default="test_id", choices=["train", "val", "test_id", "test_ood"])
	parser.add_argument("--fault", default="sensor_delay", help="场景名，可带强度，例如 sensor_delay@20")
	parser.add_argument("--trial-index", type=int, default=0, help="该 split 里第几个试验")
	parser.add_argument("--window-index", type=int, default=0, help="该试验里第几个窗口")
	parser.add_argument(
		"--scan-trials",
		type=int,
		default=0,
		help=">0 时扫描前 N 个试验的所有窗口，选门控活动最强的一个导出（参考分位只估一次）",
	)
	parser.add_argument("--device", default="cuda")
	parser.add_argument("--output", required=True)
	return parser.parse_args()


def build_eval_args(report_args: Dict, cli: argparse.Namespace) -> argparse.Namespace:
	"""用报告里的 args 复原评测配置，只覆盖本次导出需要改的字段。"""
	ns = argparse.Namespace(**report_args)
	ns.data_root = cli.data_root
	ns.device = cli.device
	ns.mode = "eval"
	ns.checkpoint = cli.checkpoint
	ns.fault_scenarios = "clean"
	ns.batch_size = 8
	ns.num_workers = 2
	ns.prefetch_factor = 2
	ns.max_eval_batches = 0
	ns.extra_models = []
	# MC-dropout 只影响 epistemic 通道；导出单窗口时保留报告里的设置。
	ns.mc_samples = int(report_args.get("mc_samples") or 0)
	return ns


def main() -> None:
	cli = parse_args()
	report = json.loads(Path(cli.report).read_text(encoding="utf-8"))
	results = report.get("results", {})
	gate_policy = results.get("_gate_policy", {})
	detector_policy = results.get("_detector_policy", {})
	raw_signals = detector_policy.get("signals", [])
	if isinstance(raw_signals, str):
		raw_signals = [s.strip() for s in raw_signals.split(",")]
	frozen_signals: List[str] = [s for s in raw_signals if s]

	args = build_eval_args(report["args"], cli)
	input_names = input_names_for_profile(args.input_profile, args.side)
	target_names = label_names(args.side)
	groups = feature_groups(input_names)
	args.velocity_indices = joint_velocity_indices(input_names, target_names)

	records = discover_trials(args.data_root)
	heldout = rre._parse_csv(args.heldout_tasks)
	train_p, val_p, test_p = rre.resolve_split(records, args)
	split_records = {
		"train": filter_records(records, participants=train_p, exclude_tasks=heldout),
		"val": filter_records(records, participants=val_p, exclude_tasks=heldout),
		"test_id": filter_records(records, participants=test_p, exclude_tasks=heldout),
		"test_ood": filter_records(records, participants=test_p, include_tasks=heldout),
	}

	device = torch.device(args.device)
	model, payload = load_checkpoint(cli.checkpoint, map_location=device)
	rre.validate_checkpoint_inputs(payload, input_names, target_names)
	model = model.to(device).eval()

	# 参考分位必须来自干净 val，和主评测一致；否则风险尺度不可比。
	val_dataset = rre.make_dataset(split_records["val"], input_names, target_names, args)
	uncertainty_ref = rre.collect_uncertainty_ref(model, val_dataset, groups, args, device)
	risk_refs = rre.estimate_risk_refs(model, val_dataset, groups, args, device)

	target_records = split_records[cli.split]
	if not target_records:
		raise SystemExit(f"split {cli.split} 无试验")

	softness = float(gate_policy.get("softness", args.gate_softness))
	deadband = float(gate_policy.get("deadband", args.gate_deadband))
	spec = parse_fault_spec(cli.fault)
	horizon = getattr(model, "forecast_horizon", 1)

	def compute(window: Dict) -> Dict[str, torch.Tensor]:
		x = window["x"].unsqueeze(0).to(device)
		y = window["y"].unsqueeze(0).to(device)
		mask = rre.apply_ignore_history(window["mask"].unsqueeze(0).to(device), args.eval_ignore_history)
		x_eval, fault_target = apply_fault(x, groups, spec)
		with torch.no_grad():
			out = model(x_eval)
			residual = (
				reconstruction_residual(out["recon"], out["input_norm"], groups) if "recon" in out else None
			)
			forecast = (
				forecast_residual(out["forecast"], out["input_norm"], horizon, groups)
				if "forecast" in out
				else None
			)
			staleness = staleness_score(out["input_norm"], groups)
			drift = drift_score(out["input_norm"], groups)
			epistemic = None
			if args.mc_samples and args.mc_samples > 1:
				mc = model.predict_uncertainty(x_eval, samples=args.mc_samples)
				epistemic = mc["epistemic_sigma"].mean(dim=1, keepdim=True)
			# drift=None 与主评测一致：它只作检测通道，不进门控。
			risk = rre.calibrate_risk(
				aleatoric_sigma=torch.exp(0.5 * out["logvar"]),
				fault_prob=torch.sigmoid(out["fault_logit"]),
				residual=residual,
				epistemic_sigma=epistemic,
				forecast=forecast,
				staleness=staleness,
				drift=None,
				refs=risk_refs,
			)
			gate = reliability_gate(risk, softness=softness, deadband=deadband)
			baseline_torque = moment_to_torque(out["mean"])
		tensors = {
			"mean": out["mean"],
			"sigma": torch.exp(0.5 * out["logvar"]),
			"truth": y,
			"mask": mask,
			"risk": risk,
			"gate": gate,
			"baseline_torque": baseline_torque,
			"commanded_torque": baseline_torque * gate,
			"fault_target": fault_target,
			"staleness": staleness,
			"drift": drift,
		}
		if residual is not None:
			tensors["residual"] = residual
		if forecast is not None:
			tensors["forecast"] = forecast
		if epistemic is not None:
			tensors["epistemic"] = epistemic
		return tensors

	def gate_activity(tensors: Dict[str, torch.Tensor]) -> float:
		"""门控活动度：有效时刻的平均衰减量。用于挑一个能看出机制的窗口。"""
		gate = tensors["gate"][0, 0, args.eval_ignore_history :]
		return float((1.0 - gate).mean())

	# 扫描模式：参考分位已估好，逐窗口复算只花前向的时间。
	best = None
	if cli.scan_trials > 0:
		for trial_index, record in enumerate(target_records[: cli.scan_trials]):
			dataset = rre.make_dataset([record], input_names, target_names, args)
			for window_index in range(len(dataset)):
				tensors = compute(dataset[window_index])
				score = gate_activity(tensors)
				if best is None or score > best[0]:
					best = (score, trial_index, window_index, record, tensors)
		if best is None:
			raise SystemExit("扫描范围内没有有效窗口")
		score, trial_index, window_index, record, tensors = best
		print(f"scan: 选中 trial={trial_index} window={window_index} gate_activity={score:.4f}")
	else:
		trial_index = cli.trial_index
		record = target_records[trial_index]
		dataset = rre.make_dataset([record], input_names, target_names, args)
		if len(dataset) == 0:
			raise SystemExit("该试验在有效标签过滤后没有窗口")
		window_index = min(cli.window_index, len(dataset) - 1)
		tensors = compute(dataset[window_index])

	payload_out: Dict[str, np.ndarray] = {}
	for key, tensor in tensors.items():
		array = tensor.squeeze(0).detach().cpu().numpy()
		payload_out[key] = array.astype(np.bool_) if key == "mask" else array

	meta = {
		"checkpoint": cli.checkpoint,
		"report": cli.report,
		"split": cli.split,
		"fault": cli.fault,
		"participant": record.participant,
		"trial": record.trial_name,
		"task": record.task_family,
		"trial_index": trial_index,
		"window_index": window_index,
		"gate_activity": gate_activity(tensors),
		"gate_softness": softness,
		"gate_deadband": deadband,
		"gate_policy_source": gate_policy.get("policy_source", ""),
		"detector_signals": frozen_signals,
		"label_names": target_names,
		"uncertainty_ref": uncertainty_ref,
		"eval_ignore_history": args.eval_ignore_history,
	}
	out_path = Path(cli.output)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(out_path, meta=json.dumps(meta, ensure_ascii=False), **payload_out)
	print(f"Wrote {out_path}")
	print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
