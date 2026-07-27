#!/usr/bin/env python3
"""EXPERIMENTS_TODO §3：rate-limit 网格上的 Pareto 前沿与平滑代价。

横轴给 clean 侧代价（aligned retention 与 tracking RMSE 两种口径都出），纵轴同时给
$X_{\\mathrm{opp}}$ 与 mean absolute torque rate——因为「更安全」和「更平滑」不是同一
件事：任何降 $X_{\\mathrm{opp}}$ 但抬 torque rate 的门控在汇总里都被标成 trade-off，
不能用一句「更安全」盖过去。

必报的五类行（§3）：
  no_smoothing        原始无 rate limit 的软门控
  rate_limited        因果 rate-limit 网格（fall × rise 全格）
  hysteretic          因果滞回门控（双阈值 + 最短保持时长）
  hard_threshold      硬阈值门控
  ungated             无门控的基础估计器

滞回门控的实现是因果的：风险越过 `--hysteresis-high` 才关，落回 `--hysteresis-low`
以下且已保持 `--hysteresis-hold-ms` 才允许恢复。这样「抖动一下就恢复」被结构性禁止，
而不是靠 rate limit 把毛刺磨平。

用法：
    python scripts/run_gate_dynamics.py \
        --report reports/v2_fc_main_seed7/reliability_report.json \
        --checkpoint reports/v2_main_seed7/reliability_tcn_best.pt \
        --output-dir reports/aaai27_gate_dynamics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_reliability_experiment as rre
from reliability.features import (
	feature_groups,
	input_names_for_profile,
	joint_velocity_indices,
	label_names,
)
from reliability.gate_lab import (
	GatePolicy,
	WindowBatch,
	channel_subset,
	collect_windows,
	hard_threshold_gate,
	load_report,
	write_tsv,
)
from reliability.metrics import (
	causal_gate_rate_limit,
	metrics_for_given_gate,
	reliability_gate,
	risk_from_channels,
)
from reliability.model import load_checkpoint
from reliability.nature_dataset import discover_trials, filter_records


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--report", required=True)
	p.add_argument("--checkpoint", required=True)
	p.add_argument("--data-root", default="data/Parsed")
	p.add_argument("--output-dir", required=True)
	p.add_argument("--splits", default="test_id,test_ood")
	p.add_argument("--clean-scenario", default="clean", help="clean 侧代价用哪个场景衡量")
	p.add_argument("--fault-scenarios", default="insole_missing,encoder_dropout,packet_loss,sensor_delay,stuck_imu")
	p.add_argument("--fall-grid", default="0,0.0025,0.005,0.01,0.02,0.05")
	p.add_argument("--rise-grid", default="0,0.001,0.0025,0.005,0.01,0.02")
	p.add_argument("--hysteresis-high", type=float, default=1.0, help="关断阈（标定风险）")
	p.add_argument("--hysteresis-low", type=float, default=0.6, help="恢复阈，必须 < high")
	p.add_argument("--hysteresis-hold-ms", type=float, default=200.0, help="最短关断保持时长")
	p.add_argument("--seed", type=int, default=0)
	p.add_argument("--device", default="cuda")
	p.add_argument("--batch-size", type=int, default=32)
	p.add_argument("--num-workers", type=int, default=8)
	p.add_argument("--max-windows", type=int, default=0)
	p.add_argument("--tag", default="")
	return p.parse_args()


def causal_hysteretic_gate(
	risk: torch.Tensor,
	high: float,
	low: float,
	hold_steps: int,
) -> torch.Tensor:
	"""因果双阈值滞回门控，返回 {0,1} 序列。

	状态机逐步推进（只看当前与过去，可在线实现）：`risk > high` 进入关断并至少保持
	`hold_steps`；保持期满后要等 `risk < low` 才恢复。`low < high` 的间隙就是滞回带，
	它把「风险在阈值附近抖动」变成一次关断而不是一串开关。
	"""
	if risk.dim() == 3:
		flat = risk.mean(dim=1)
	else:
		flat = risk
	n, t = flat.shape
	gate = torch.ones((n, t), dtype=torch.float32)
	closed = torch.zeros(n, dtype=torch.bool)
	held = torch.zeros(n, dtype=torch.long)
	for step in range(t):
		r = flat[:, step].detach().cpu()
		# 已关断：保持期未满强制继续关；期满后风险落回 low 以下才恢复。
		can_open = closed & (held >= hold_steps) & (r < low)
		closed = torch.where(can_open, torch.zeros_like(closed), closed)
		held = torch.where(can_open, torch.zeros_like(held), held)
		trigger = (~closed) & (r > high)
		closed = closed | trigger
		held = torch.where(trigger, torch.zeros_like(held), held)
		gate[:, step] = (~closed).to(torch.float32)
		held = torch.where(closed, held + 1, held)
	return gate.unsqueeze(1)


def build_eval_args(report_args: dict, cli: argparse.Namespace) -> argparse.Namespace:
	ns = argparse.Namespace(**report_args)
	ns.data_root = cli.data_root
	ns.device = cli.device
	ns.mode = "eval"
	ns.checkpoint = cli.checkpoint
	ns.batch_size = cli.batch_size
	ns.num_workers = cli.num_workers
	ns.prefetch_factor = 2
	ns.max_eval_batches = 0
	ns.extra_models = []
	ns.mc_samples = int(report_args.get("mc_samples") or 0)
	return ns


def summarize(gate: torch.Tensor, batch: WindowBatch) -> Dict[str, float]:
	m = metrics_for_given_gate(gate, batch.mean, batch.y, batch.mask)
	return {
		"x_opp": m.get("gated_wrong_torque_product_integral"),
		"wrong_direction_fraction": m.get("gated_wrong_direction_ratio"),
		"aligned_command_retention": m.get("gated_retained_aligned_torque"),
		"capped_overlap_retention": m.get("gated_capped_overlap"),
		"tracking_rmse": m.get("gated_tracking_rmse"),
		"mean_abs_torque_rate": m.get("gated_mean_abs_torque_rate"),
		"full_shutdown_fraction": m.get("full_shutdown_fraction"),
		"shutdowns_per_minute": m.get("shutdowns_per_minute"),
		"mean_shutdown_duration_ms": m.get("mean_shutdown_duration_ms"),
		"p95_shutdown_duration_ms": m.get("p95_shutdown_duration_ms"),
		"gate_transitions_per_minute": m.get("gate_transitions_per_minute"),
		"mean_gate": m.get("mean_gate"),
		"ungated_x_opp": m.get("ungated_wrong_torque_product_integral"),
		"ungated_retention": m.get("ungated_retained_aligned_torque"),
		"ungated_tracking_rmse": m.get("ungated_tracking_rmse"),
		"ungated_mean_abs_torque_rate": m.get("ungated_mean_abs_torque_rate"),
	}


def main() -> None:
	cli = parse_args()
	if cli.hysteresis_low >= cli.hysteresis_high:
		raise SystemExit("--hysteresis-low 必须小于 --hysteresis-high，否则没有滞回带")
	report = load_report(cli.report)
	policy = GatePolicy.from_report(report)
	args = build_eval_args(report["args"], cli)
	tag = cli.tag or Path(cli.report).parent.name
	seed_label = report["args"].get("seed", "")

	input_names = input_names_for_profile(args.input_profile, args.side)
	target_names = label_names(args.side)
	groups = feature_groups(input_names)
	args.velocity_indices = joint_velocity_indices(input_names, target_names)

	records = discover_trials(args.data_root)
	heldout = rre._parse_csv(args.heldout_tasks)
	train_p, val_p, test_p = rre.resolve_split(records, args)
	split_records = {
		"val": filter_records(records, participants=val_p, exclude_tasks=heldout),
		"test_id": filter_records(records, participants=test_p, exclude_tasks=heldout),
		"test_ood": filter_records(records, participants=test_p, include_tasks=heldout),
	}

	device = torch.device(args.device)
	model, payload = load_checkpoint(cli.checkpoint, map_location=device)
	rre.validate_checkpoint_inputs(payload, input_names, target_names)
	model = model.to(device).eval()

	val_dataset = rre.make_dataset(split_records["val"], input_names, target_names, args)
	risk_refs = rre.estimate_risk_refs(model, val_dataset, groups, args, device)

	fall_grid = rre._parse_floats(cli.fall_grid)
	rise_grid = rre._parse_floats(cli.rise_grid)
	hold_steps = max(int(round(cli.hysteresis_hold_ms / 1000.0 * 200.0)), 1)
	max_windows = cli.max_windows or None

	rows: List[Dict[str, object]] = []
	manifest: List[Dict[str, object]] = []
	started = time.time()

	scenarios = [cli.clean_scenario] + rre._parse_csv(cli.fault_scenarios)
	for split in rre._parse_csv(cli.splits):
		if not split_records.get(split):
			continue
		dataset = rre.make_dataset(split_records[split], input_names, target_names, args)
		for scenario in scenarios:
			batch = collect_windows(
				model, dataset, groups, args, device, scenario,
				generator_seed=cli.seed, max_windows=max_windows,
			)
			if batch is None:
				continue
			channels = channel_subset(batch, policy.signals)
			risk = risk_from_channels(channels, risk_refs) if channels else None
			if risk is None:
				continue
			raw_gate = reliability_gate(risk, softness=policy.softness, deadband=policy.deadband)
			valid = batch.valid_time()

			def emit(family: str, gate: torch.Tensor, **extra) -> None:
				row: Dict[str, object] = {
					"run": tag,
					"seed": seed_label,
					"task_split": split,
					"scenario": scenario,
					"is_clean": scenario == cli.clean_scenario,
					"gate_family": family,
					"n_windows": len(batch),
					"gate_softness": policy.softness,
					"gate_deadband": policy.deadband,
				}
				row.update(extra)
				row.update(summarize(gate, batch))
				rows.append(row)

			emit("no_smoothing", raw_gate, max_fall_per_step=0.0, max_rise_per_step=0.0)
			for fall in fall_grid:
				for rise in rise_grid:
					if fall == 0.0 and rise == 0.0:
						continue  # 与 no_smoothing 完全同一条，不重复计。
					emit(
						"rate_limited",
						causal_gate_rate_limit(raw_gate, max_fall_per_step=fall, max_rise_per_step=rise),
						max_fall_per_step=fall,
						max_rise_per_step=rise,
					)
			emit(
				"hysteretic",
				causal_hysteretic_gate(risk, cli.hysteresis_high, cli.hysteresis_low, hold_steps),
				hysteresis_high=cli.hysteresis_high,
				hysteresis_low=cli.hysteresis_low,
				hysteresis_hold_ms=cli.hysteresis_hold_ms,
			)
			hard = hard_threshold_gate(batch, policy.signals, risk_refs)
			if hard is not None:
				emit("hard_threshold", hard, hard_threshold=1.0)
			emit("ungated", torch.ones_like(valid, dtype=batch.mean.dtype))
			manifest.append(
				{
					"run": tag, "task_split": split, "scenario": scenario,
					"n_windows": len(batch),
					"participants": ",".join(sorted(set(batch.participant))),
				}
			)
			print(f"[{time.time() - started:7.1f}s] {split}/{scenario}: {len(fall_grid) * len(rise_grid) + 3} points")
			del batch, risk, raw_gate

	out_dir = Path(cli.output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	write_tsv(out_dir / f"gate_dynamics_{tag}.tsv", rows)
	write_tsv(out_dir / f"run_manifest_{tag}.tsv", manifest)
	(out_dir / f"provenance_{tag}.json").write_text(
		json.dumps(
			{
				"report": str(cli.report),
				"checkpoint": str(cli.checkpoint),
				"data_root": cli.data_root,
				"fall_grid": fall_grid,
				"rise_grid": rise_grid,
				"hysteresis": {
					"high": cli.hysteresis_high, "low": cli.hysteresis_low,
					"hold_ms": cli.hysteresis_hold_ms, "hold_steps": hold_steps,
				},
				"gate_policy": {
					"softness": policy.softness, "deadband": policy.deadband,
					"signals": list(policy.signals), "source": policy.source,
				},
				"risk_refs": risk_refs,
				"training_seed": seed_label,
				"participants": {"val": sorted(val_p), "test": sorted(test_p)},
				"cli": vars(cli),
				"elapsed_seconds": time.time() - started,
			},
			indent=2,
			ensure_ascii=False,
		),
		encoding="utf-8",
	)
	print(f"Wrote {out_dir}/gate_dynamics_{tag}.tsv ({len(rows)} rows)")


if __name__ == "__main__":
	main()
