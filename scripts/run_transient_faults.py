#!/usr/bin/env python3
"""EXPERIMENTS_TODO §4：部分窗口瞬态故障与门控的时间响应。

整窗损坏（`apply_fault`）测不出「故障发生后多久门控落下」「恢复后多久放开」，因为
窗口里没有干净的对照段。这里把损坏限制在窗口内的一个区间：

  * onset ∈ {20,40,60,80}% 窗长；
  * duration ∈ {25,50,100,250,500} ms；
  * with recovery（区间结束后恢复干净）与 without recovery（持续到窗尾）；
  * 单故障与两个同时发生的故障；
  * clean pseudo-onset 负对照：完全不注入故障或暂停，只复用 onset 网格，用来量
    false shutdowns 与 alarm chatter，
    否则「响应快」可能只是「一直在关」。

报告的时间量（全部以 onset/recovery 为零点，单位 ms）：
  onset → 首次 g<0.5 / 首次 g=0；recovery → 首次 g>0.5 / 首次 g=1；
  false shutdowns per minute（故障区间外的关断）；alarm chatter（跨 0.5 次数/min）；
  短故障漏检率（整段故障期内 g 从未低于 0.5 的窗口占比）；clean pseudo-onset 误报率。

`--dump-timesteps` 打开时按 §4 落逐 timestep 明细（timestamp/participant/trial/
task_split/fault_operator/onset/offset/fault_active/各异常通道/归一风险/gate/
ungated torque/gated torque/ideal torque/active mask）到 parquet 或压缩 npz。

用法：
    python scripts/run_transient_faults.py \
        --report reports/v2_fc_main_seed7/reliability_report.json \
        --checkpoint reports/v2_main_seed7/reliability_tcn_best.pt \
        --output-dir reports/aaai27_transient_faults --dump-timesteps
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
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
	ACTIVE_TORQUE_FLOOR,
	GatePolicy,
	WindowBatch,
	channel_subset,
	collect_windows,
	load_report,
	write_tsv,
)
from reliability.metrics import (
	causal_gate_rate_limit,
	metrics_for_given_gate,
	moment_to_torque,
	reliability_gate,
	risk_from_channels,
)
from reliability.model import load_checkpoint
from reliability.nature_dataset import discover_trials, filter_records

SAMPLE_RATE = 200.0

#: §4 要求的 ≥9 个故障算子。stuck_imu 与 packet_loss_partial 都在内。
DEFAULT_OPERATORS = (
	"insole_missing,encoder_dropout,imu_bias,packet_loss,stuck_imu,"
	"sensor_delay,packet_loss_burst,packet_loss_partial,sensor_delay_jitter"
)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--report", required=True)
	p.add_argument("--checkpoint", required=True)
	p.add_argument("--data-root", default="data/Parsed")
	p.add_argument("--output-dir", required=True)
	p.add_argument("--splits", default="test_id,test_ood")
	p.add_argument("--operators", default=DEFAULT_OPERATORS)
	p.add_argument("--onset-fractions", default="0.2,0.4,0.6,0.8")
	p.add_argument("--durations-ms", default="25,50,100,250,500")
	p.add_argument("--simultaneous", default="packet_loss+imu_bias,insole_missing+sensor_delay",
		help="两故障同时发生的组合，用 + 连接")
	p.add_argument("--dump-timesteps", action="store_true", help="落逐 timestep 明细（npz）")
	p.add_argument("--dump-max-windows", type=int, default=8, help="每个条件最多落几个窗口的明细")
	p.add_argument("--seed", type=int, default=0)
	p.add_argument("--device", default="cuda")
	p.add_argument("--batch-size", type=int, default=32)
	p.add_argument("--num-workers", type=int, default=8)
	p.add_argument("--max-windows", type=int, default=0)
	p.add_argument("--tag", default="")
	return p.parse_args()


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


def _first_crossing_ms(
	flags: torch.Tensor,
	origin: int,
	limit: Optional[int],
	valid: torch.Tensor,
) -> Tuple[float, float]:
	"""从 origin 起首次满足 flags 的延迟（ms）与达成率。

	窗口里没有达成的样本不计入延迟均值——把它们当成「延迟 = 窗口剩余长度」会把延迟
	压向一个与故障无关的常数。达成率单独报告，两个数一起看才完整。
	"""
	n, t = flags.shape
	stop = t if limit is None else min(limit, t)
	if origin >= stop:
		return float("nan"), float("nan")
	segment = flags[:, origin:stop] & valid[:, origin:stop]
	rows_valid = valid[:, origin:stop].any(dim=1)
	if not rows_valid.any():
		return float("nan"), float("nan")
	hit = segment.any(dim=1)
	idx = segment.float().argmax(dim=1)
	achieved = hit & rows_valid
	rate = float(achieved.sum()) / float(rows_valid.sum())
	if not achieved.any():
		return float("nan"), rate
	delay = idx[achieved].to(torch.float64).mean().item() * 1000.0 / SAMPLE_RATE
	return delay, rate


def transient_metrics(
	gate: torch.Tensor,
	batch: WindowBatch,
	onset: int,
	offset: Optional[int],
	negative_control: bool = False,
) -> Dict[str, float]:
	"""onset/recovery 前后的门控时间响应。

	`negative_control=True`（clean pseudo-onset）时没有故障或暂停区间，所有以 onset/recovery 为零点的
	延迟一律留 NaN——那种情况下「首次 g>0.5」只是「门控本来就开着」，报出来会被误读成
	恢复很快。整窗都算区间外，于是 false shutdown / chatter 就是纯误报率。
	"""
	g = gate.mean(dim=1) if gate.dim() == 3 and gate.shape[1] > 1 else gate.reshape(gate.shape[0], -1)
	valid = batch.valid_time().reshape(gate.shape[0], -1)
	t = g.shape[1]
	below = g < 0.5
	zero = g <= 1e-6
	stop = t if offset is None else offset

	if negative_control:
		drop_ms = drop_rate = shut_ms = shut_rate = float("nan")
		release_ms = release_rate = full_ms = full_rate = float("nan")
	else:
		drop_ms, drop_rate = _first_crossing_ms(below, onset, stop, valid)
		shut_ms, shut_rate = _first_crossing_ms(zero, onset, stop, valid)
		if offset is not None and offset < t:
			release_ms, release_rate = _first_crossing_ms(g > 0.5, offset, None, valid)
			full_ms, full_rate = _first_crossing_ms(g >= 1.0 - 1e-6, offset, None, valid)
		else:
			release_ms = release_rate = full_ms = full_rate = float("nan")

	# 故障区间以外的关断都是误报。区间外的有效时长做分母，得到 per-minute 率。
	outside = torch.ones_like(valid)
	if not negative_control:
		outside[:, onset:stop] = False
	outside = outside & valid
	outside_minutes = float(outside.sum()) / SAMPLE_RATE / 60.0
	fp_episodes = _count_episodes(zero & outside)
	chatter = int(((below[:, 1:] != below[:, :-1]) & valid[:, 1:] & valid[:, :-1]).sum())
	total_minutes = float(valid.sum()) / SAMPLE_RATE / 60.0

	# 漏检：整段故障期内 g 从未低于 0.5 的窗口。短故障上这个数才是关键结论。
	if negative_control:
		miss_rate = float("nan")
	else:
		inside = torch.zeros_like(valid)
		inside[:, onset:stop] = True
		inside = inside & valid
		rows_with_fault = inside.any(dim=1)
		detected = (below & inside).any(dim=1)
		miss_rate = (
			float((rows_with_fault & ~detected).sum()) / float(rows_with_fault.sum())
			if rows_with_fault.any()
			else float("nan")
		)
	return {
		"onset_to_half_ms": drop_ms,
		"onset_to_half_rate": drop_rate,
		"onset_to_zero_ms": shut_ms,
		"onset_to_zero_rate": shut_rate,
		"recovery_to_half_ms": release_ms,
		"recovery_to_half_rate": release_rate,
		"recovery_to_full_ms": full_ms,
		"recovery_to_full_rate": full_rate,
		"false_shutdowns_per_minute": fp_episodes / outside_minutes if outside_minutes > 0 else float("nan"),
		"alarm_chatter_per_minute": chatter / total_minutes if total_minutes > 0 else float("nan"),
		"fault_miss_rate": miss_rate,
	}


def _count_episodes(flag: torch.Tensor) -> int:
	pad = torch.zeros((flag.shape[0], 1), dtype=torch.bool)
	flat = torch.cat([pad, flag.cpu(), pad], dim=1).reshape(-1)
	edges = flat[1:].to(torch.int8) - flat[:-1].to(torch.int8)
	return int((edges == 1).sum())


def dump_timesteps(
	path: Path,
	batch: WindowBatch,
	gate: torch.Tensor,
	risk: torch.Tensor,
	split: str,
	operator: str,
	onset: int,
	offset: Optional[int],
	limit: int,
) -> None:
	"""逐 timestep 明细。每行一个 (window, t)，列按 §4 列表给全。"""
	n = min(len(batch), limit)
	ideal = moment_to_torque(batch.y[:n])
	ungated = moment_to_torque(batch.mean[:n])
	g = gate[:n]
	if g.dim() == 3 and g.shape[1] > 1:
		g = g.mean(dim=1, keepdim=True)
	valid = batch.mask[:n].all(dim=1, keepdim=True)
	active = valid.expand_as(ideal) & (ideal.abs() > ACTIVE_TORQUE_FLOOR)
	payload: Dict[str, np.ndarray] = {
		"sample_index": np.arange(batch.mean.shape[-1], dtype=np.int32),
		"timestamp_s": (np.arange(batch.mean.shape[-1], dtype=np.float32) / SAMPLE_RATE),
		"participant": np.array(batch.participant[:n], dtype=object),
		"trial": np.array(batch.trial[:n], dtype=object),
		"task": np.array(batch.task[:n], dtype=object),
		"window_start": np.array(batch.start[:n], dtype=np.int64),
		"task_split": np.array([split] * n, dtype=object),
		"fault_operator": np.array([operator] * n, dtype=object),
		"fault_onset": np.full(n, onset, dtype=np.int32),
		"fault_offset": np.full(n, -1 if offset is None else offset, dtype=np.int32),
		"fault_active": (batch.fault_target[:n, 0] > 0.5).numpy(),
		"normalized_risk": risk[:n].reshape(n, -1).numpy().astype(np.float32),
		"gate": g.reshape(n, -1).numpy().astype(np.float32),
		"ungated_torque": ungated.numpy().astype(np.float32),
		"gated_torque": (ungated * g).numpy().astype(np.float32),
		"ideal_torque": ideal.numpy().astype(np.float32),
		"active_mask": active.numpy(),
	}
	for name, tensor in batch.channels.items():
		payload[f"anomaly_{name}"] = tensor[:n].reshape(n, -1).numpy().astype(np.float32)
	path.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(path, **payload)


def main() -> None:
	cli = parse_args()
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

	window = int(args.window_size)
	onsets = sorted({max(int(round(frac * window)), 1) for frac in rre._parse_floats(cli.onset_fractions)})
	durations = [int(round(ms / 1000.0 * SAMPLE_RATE)) for ms in rre._parse_floats(cli.durations_ms)]
	operators: List[Tuple[str, List[str]]] = [(name, []) for name in rre._parse_csv(cli.operators)]
	for combo in rre._parse_csv(cli.simultaneous):
		parts = [p for p in combo.split("+") if p]
		if len(parts) >= 2:
			operators.append((parts[0], parts[1:]))
	max_windows = cli.max_windows or None

	rows: List[Dict[str, object]] = []
	manifest: List[Dict[str, object]] = []
	out_dir = Path(cli.output_dir)
	started = time.time()
	dumped = 0

	for split in rre._parse_csv(cli.splits):
		if not split_records.get(split):
			continue
		dataset = rre.make_dataset(split_records[split], input_names, target_names, args)

		def evaluate(
			operator_label: str,
			primary: str,
			extras: Sequence[str],
			onset: int,
			duration_steps: Optional[int],
			recovery: bool,
			dump: bool = False,
		) -> None:
			nonlocal dumped
			offset = None if not recovery or duration_steps is None else min(onset + duration_steps, window)
			interval = None if primary == "clean" else (onset, offset)
			batch = collect_windows(
				model, dataset, groups, args, device, primary,
				interval=interval, extra_specs=extras,
				generator_seed=cli.seed, max_windows=max_windows,
			)
			if batch is None:
				return
			channels = channel_subset(batch, policy.signals)
			if not channels:
				return
			risk = risk_from_channels(channels, risk_refs)
			gate = causal_gate_rate_limit(
				reliability_gate(risk, softness=policy.softness, deadband=policy.deadband),
				max_fall_per_step=policy.max_fall_per_step,
				max_rise_per_step=policy.max_rise_per_step,
			)
			is_control = primary == "clean"
			timing = transient_metrics(gate, batch, onset, offset, negative_control=is_control)
			exposure = metrics_for_given_gate(gate, batch.mean, batch.y, batch.mask)
			row: Dict[str, object] = {
				"run": tag,
				"seed": seed_label,
				"task_split": split,
				"fault_operator": operator_label,
				"n_simultaneous": 1 + len(extras),
				"onset_step": onset,
				"onset_fraction": onset / window,
				"duration_ms": float("nan") if duration_steps is None else duration_steps * 1000.0 / SAMPLE_RATE,
				"with_recovery": recovery,
				"is_negative_control": is_control,
				"n_windows": len(batch),
			}
			row.update(timing)
			row["x_opp"] = exposure.get("gated_wrong_torque_product_integral")
			row["ungated_x_opp"] = exposure.get("ungated_wrong_torque_product_integral")
			row["aligned_command_retention"] = exposure.get("gate_retention_ratio")
			row["aligned_command_adequacy"] = exposure.get("gated_aligned_command_adequacy")
			row["tracking_rmse"] = exposure.get("gated_tracking_rmse")
			row["mean_abs_torque_rate"] = exposure.get("gated_mean_abs_torque_rate")
			row["full_shutdown_fraction"] = exposure.get("full_shutdown_fraction")
			row["mean_gate"] = exposure.get("mean_gate")
			rows.append(row)
			if dump and cli.dump_timesteps:
				name = f"timesteps_{tag}_{split}_{operator_label.replace('+', '-')}_on{onset}_d{duration_steps}.npz"
				dump_timesteps(
					out_dir / "timesteps" / name, batch, gate, risk,
					split, operator_label, onset, offset, cli.dump_max_windows,
				)
				dumped += 1
			manifest.append(
				{
					"run": tag, "task_split": split, "fault_operator": operator_label,
					"onset_step": onset, "duration_ms": row["duration_ms"],
					"with_recovery": recovery, "n_windows": len(batch),
					"participants": ",".join(sorted(set(batch.participant))),
				}
			)
			del batch, risk, gate

		for primary, extras in operators:
			label = "+".join([primary, *extras])
			for onset in onsets:
				for steps in durations:
					evaluate(label, primary, extras, onset, steps, recovery=True,
						dump=(onset == onsets[len(onsets) // 2]))
				evaluate(label, primary, extras, onset, None, recovery=False)
			print(f"[{time.time() - started:7.1f}s] {split}/{label} 完成 {len(onsets) * (len(durations) + 1)} 条")
		# clean pseudo-onset 负对照：同样的 onset 网格，但不注入任何故障或暂停。
		for onset in onsets:
			evaluate("clean_pause", "clean", [], onset, None, recovery=False)
		print(f"[{time.time() - started:7.1f}s] {split}/clean_pause 负对照完成")

	out_dir.mkdir(parents=True, exist_ok=True)
	write_tsv(out_dir / f"transient_faults_{tag}.tsv", rows)
	write_tsv(out_dir / f"run_manifest_{tag}.tsv", manifest)
	(out_dir / f"provenance_{tag}.json").write_text(
		json.dumps(
			{
				"report": str(cli.report),
				"checkpoint": str(cli.checkpoint),
				"data_root": cli.data_root,
				"onset_steps": onsets,
				"durations_steps": durations,
				"operators": [list((p, e)) for p, e in operators],
				"gate_policy": {
					"softness": policy.softness, "deadband": policy.deadband,
					"max_fall_per_step": policy.max_fall_per_step,
					"max_rise_per_step": policy.max_rise_per_step,
					"signals": list(policy.signals), "source": policy.source,
				},
				"risk_refs": risk_refs,
				"sample_rate": SAMPLE_RATE,
				"window_size": window,
				"training_seed": seed_label,
				"participants": {"val": sorted(val_p), "test": sorted(test_p)},
				"timestep_dumps": dumped,
				"cli": vars(cli),
				"elapsed_seconds": time.time() - started,
			},
			indent=2,
			ensure_ascii=False,
		),
		encoding="utf-8",
	)
	print(f"Wrote {out_dir}/transient_faults_{tag}.tsv ({len(rows)} rows, {dumped} timestep dumps)")


if __name__ == "__main__":
	main()
