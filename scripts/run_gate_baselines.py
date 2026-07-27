#!/usr/bin/env python3
"""EXPERIMENTS_TODO §2：九种门控的同口径对照（含 matched-random 随机化检验）。

九格（全部在同一批模型输出、同一组窗口、同一个 active mask、固定种子下评测）：

  1. fault_only            仅 fault-head logit
  2. reconstruction_only   仅重构残差
  3. staleness_only        仅 staleness
  4. selected_subset       验证集选出的冻结子集（论文报告的门控）
  5. all_gate_channels     全部 K_gate 六路
  6. hard_threshold        风险过阈直接断力矩（无软化、无 rate limit）
  7. oracle_error          用真实预测误差当风险（不可实现的上界）
  8. random_shutdown       $P(g{=}0)$-matched 随机关断
  9. random_attenuation    clean-retention-matched 随机衰减（门控值随机置换）

第 8、9 格各做 `--draws` 次（默认 1000）固定种子抽样，报告随机化分布的
均值/标准差/分位以及实际门控在该分布里的百分位——「比随机好」必须是可检验的陈述，
不是并列两行数字。

逐 seed × task split × fault operator 落盘，不只报 macro 平均：单一平均会把
「某个故障算子上门控完全没反应」藏起来。

用法：
    python scripts/run_gate_baselines.py \
        --report reports/v2_fc_main_seed7/reliability_report.json \
        --checkpoint reports/v2_main_seed7/reliability_tcn_best.pt \
        --data-root /home/zfy/dataset/tcn/Parsed \
        --output-dir reports/aaai27_gate_baselines
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
	ExposureBasis,
	GatePolicy,
	K_GATE,
	WindowBatch,
	collect_windows,
	hard_threshold_gate,
	load_report,
	oracle_error_gate,
	permuted_gate,
	random_shutdown_gate,
	soft_gate,
	write_tsv,
)
from reliability.metrics import metrics_for_given_gate, moment_to_torque
from reliability.model import load_checkpoint
from reliability.nature_dataset import discover_trials, filter_records

#: §2 要求的 13 项逐门控指标。键名对应 `metrics_for_given_gate` 的输出。
REQUIRED_METRICS = (
	("x_opp", "gated_wrong_torque_product_integral"),
	("wrong_direction_fraction", "gated_wrong_direction_ratio"),
	("peak_wrong_torque", "gated_peak_wrong_torque"),
	("aligned_command_retention", "gated_retained_aligned_torque"),
	("capped_overlap_retention", "gated_capped_overlap"),
	("over_assistance_fraction", "gated_over_assistance_fraction"),
	("tracking_rmse", "gated_tracking_rmse"),
	("mean_abs_torque_rate", "gated_mean_abs_torque_rate"),
	("full_shutdown_fraction", "full_shutdown_fraction"),
	("gate_below_half_fraction", "gate_below_half_fraction"),
	("shutdowns_per_minute", "shutdowns_per_minute"),
	("mean_shutdown_duration_ms", "mean_shutdown_duration_ms"),
	("p95_shutdown_duration_ms", "p95_shutdown_duration_ms"),
)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--report", required=True, help="读回冻结门控工作点与检测子集的 reliability_report.json")
	p.add_argument("--checkpoint", required=True)
	p.add_argument("--data-root", default="data/Parsed")
	p.add_argument("--output-dir", required=True)
	p.add_argument("--splits", default="test_id,test_ood")
	p.add_argument(
		"--faults",
		default="clean,insole_missing,encoder_dropout,imu_bias,packet_loss,packet_loss_burst,"
		"packet_loss_partial,stuck_imu,sensor_delay,sensor_delay_jitter",
	)
	p.add_argument("--draws", type=int, default=1000, help="随机化检验的抽样次数（§2 要求 ≥1000）")
	p.add_argument("--seed", type=int, default=0, help="随机化检验的固定种子")
	p.add_argument("--device", default="cuda")
	p.add_argument("--batch-size", type=int, default=32)
	p.add_argument("--num-workers", type=int, default=8)
	p.add_argument("--max-windows", type=int, default=0, help=">0 时只取前 N 个窗口（仅供 smoke）")
	p.add_argument("--tag", default="", help="写进每行的 run 标签，默认取 report 的目录名")
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


def gate_variants(
	batch: WindowBatch,
	policy: GatePolicy,
	risk_refs: Optional[Dict[str, float]],
	error_ref: float,
) -> Dict[str, Dict[str, object]]:
	"""构造前七种确定性门控。返回 name -> {gate, description, channels}。"""
	out: Dict[str, Dict[str, object]] = {}

	def add(name: str, gate: Optional[torch.Tensor], channels: Sequence[str], desc: str) -> None:
		if gate is not None:
			out[name] = {"gate": gate, "channels": ",".join(channels), "description": desc}

	add("fault_only", soft_gate(batch, ["logit"], policy, risk_refs), ["logit"],
		"fault-head logit only")
	add("reconstruction_only", soft_gate(batch, ["residual"], policy, risk_refs), ["residual"],
		"reconstruction residual only")
	add("staleness_only", soft_gate(batch, ["staleness"], policy, risk_refs), ["staleness"],
		"staleness channel only")
	add("selected_subset", soft_gate(batch, policy.signals, policy, risk_refs), policy.signals,
		"validation-selected frozen subset (the reported gate)")
	add("all_gate_channels", soft_gate(batch, K_GATE, policy, risk_refs), K_GATE,
		"all six command-time channels")
	add("hard_threshold", hard_threshold_gate(batch, policy.signals, risk_refs), policy.signals,
		"hard shutdown when calibrated risk exceeds 1.0")
	add("oracle_error", oracle_error_gate(batch, policy, error_ref), ["oracle"],
		"oracle: true prediction error as risk (unachievable upper bound)")

	# 逐通道留一：K_gate 去掉一路。这同时供 §6 的「minus staleness channel」与
	# 「validation-selected-subset vs all channels」两格——它们是纯评测侧的差异，
	# 不需要重训，所以在这里出比另开一个 stage 更省一次前向。
	for dropped in K_GATE:
		kept = [name for name in K_GATE if name != dropped]
		add(
			f"loo_minus_{dropped}",
			soft_gate(batch, kept, policy, risk_refs),
			kept,
			f"all gate channels except {dropped}",
		)
	return out


def randomization_rows(
	basis: ExposureBasis,
	valid: torch.Tensor,
	reference_gate: torch.Tensor,
	reference: Dict[str, float],
	draws: int,
	seed: int,
	dtype: torch.dtype,
) -> List[Dict[str, object]]:
	"""§2 第 8、9 格：两种 matched-random 门控的抽样分布 + 实际门控的百分位。"""
	rows: List[Dict[str, object]] = []
	shutdown_fraction = float(reference.get("full_shutdown_fraction") or 0.0)
	actual_x_opp = float(reference["x_opp"])
	actual_retention = float(reference["aligned_command_retention"])

	for name, sampler in (
		("random_shutdown", "shutdown"),
		("random_attenuation", "permute"),
	):
		generator = torch.Generator()
		generator.manual_seed(seed)
		x_opps: List[float] = []
		retentions: List[float] = []
		for _ in range(draws):
			if sampler == "shutdown":
				gate = random_shutdown_gate(
					tuple(reference_gate.shape), valid, shutdown_fraction, generator, dtype=dtype
				)
			else:
				gate = permuted_gate(reference_gate, valid, generator)
			x_opps.append(basis.x_opp(gate))
			retentions.append(basis.retention(gate))
		rows.append(
			{
				"gate": name,
				"draws": draws,
				"randomization_seed": seed,
				"matched_quantity": "full_shutdown_fraction" if sampler == "shutdown" else "gate value distribution",
				"matched_value": shutdown_fraction if sampler == "shutdown" else float("nan"),
				"random_x_opp_mean": statistics.fmean(x_opps),
				"random_x_opp_sd": statistics.pstdev(x_opps) if len(x_opps) > 1 else 0.0,
				"random_x_opp_p05": _quantile(x_opps, 0.05),
				"random_x_opp_p50": _quantile(x_opps, 0.50),
				"random_x_opp_p95": _quantile(x_opps, 0.95),
				"random_retention_mean": statistics.fmean(retentions),
				"actual_x_opp": actual_x_opp,
				"actual_retention": actual_retention,
				# 实际门控在随机分布里的百分位：越接近 0 越说明「时机」本身有贡献。
				"actual_x_opp_percentile": _percentile_of(x_opps, actual_x_opp),
				"actual_retention_percentile": _percentile_of(retentions, actual_retention),
				"one_sided_p_value": (sum(1 for v in x_opps if v <= actual_x_opp) + 1) / (draws + 1),
			}
		)
	return rows


def _quantile(values: Sequence[float], q: float) -> float:
	ordered = sorted(values)
	if not ordered:
		return float("nan")
	idx = min(max(int(round(q * (len(ordered) - 1))), 0), len(ordered) - 1)
	return ordered[idx]


def _percentile_of(values: Sequence[float], target: float) -> float:
	if not values:
		return float("nan")
	return sum(1 for v in values if v <= target) / len(values)


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

	# 参考分位一律来自干净 val，与主评测同口径；test 不参与任何标定。
	val_dataset = rre.make_dataset(split_records["val"], input_names, target_names, args)
	risk_refs = rre.estimate_risk_refs(model, val_dataset, groups, args, device)
	val_batch = collect_windows(model, val_dataset, groups, args, device, "clean")
	if val_batch is None:
		raise SystemExit("val split 没有窗口，无法标定 oracle 门控的误差参考")
	val_error = (moment_to_torque(val_batch.mean) - moment_to_torque(val_batch.y)).abs().mean(dim=1, keepdim=True)
	val_valid = val_batch.valid_time()
	error_ref = float(torch.quantile(val_error[val_valid], 0.9).clamp_min(1e-6))
	del val_batch

	max_windows = cli.max_windows or None
	rows: List[Dict[str, object]] = []
	random_rows: List[Dict[str, object]] = []
	manifest: List[Dict[str, object]] = []
	started = time.time()

	for split in rre._parse_csv(cli.splits):
		if not split_records.get(split):
			print(f"WARNING split {split} 无试验，跳过")
			continue
		dataset = rre.make_dataset(split_records[split], input_names, target_names, args)
		for fault in rre._parse_csv(cli.faults):
			batch = collect_windows(
				model, dataset, groups, args, device, fault,
				generator_seed=cli.seed, max_windows=max_windows,
			)
			if batch is None:
				continue
			valid = batch.valid_time()
			basis = ExposureBasis.build(batch)
			variants = gate_variants(batch, policy, risk_refs, error_ref)
			ungated = torch.ones_like(valid, dtype=batch.mean.dtype)
			variants["ungated"] = {
				"gate": ungated, "channels": "", "description": "no gate (base estimator)",
			}
			reference: Optional[Dict[str, float]] = None
			for name, spec in variants.items():
				gate = spec["gate"]
				metrics = metrics_for_given_gate(gate, batch.mean, batch.y, batch.mask)
				row: Dict[str, object] = {
					"run": tag,
					"seed": seed_label,
					"task_split": split,
					"fault_operator": fault,
					"gate": name,
					"gate_channels": spec["channels"],
					"description": spec["description"],
					"n_windows": len(batch),
					"gate_softness": policy.softness,
					"gate_deadband": policy.deadband,
					"gate_max_fall_per_step": policy.max_fall_per_step,
					"gate_max_rise_per_step": policy.max_rise_per_step,
					"mean_gate": metrics.get("mean_gate"),
				}
				for out_key, src_key in REQUIRED_METRICS:
					row[out_key] = metrics.get(src_key)
				row["ungated_x_opp"] = metrics.get("ungated_wrong_torque_product_integral")
				row["ungated_retention"] = metrics.get("ungated_retained_aligned_torque")
				row["ungated_tracking_rmse"] = metrics.get("ungated_tracking_rmse")
				row["ungated_mean_abs_torque_rate"] = metrics.get("ungated_mean_abs_torque_rate")
				rows.append(row)
				if name == "selected_subset":
					reference = {k: row[k] for k, _ in REQUIRED_METRICS}
					reference_gate = gate
			if reference is not None:
				for extra in randomization_rows(
					basis, valid, reference_gate, reference, cli.draws, cli.seed, batch.mean.dtype
				):
					extra.update({"run": tag, "seed": seed_label, "task_split": split, "fault_operator": fault})
					random_rows.append(extra)
			manifest.append(
				{
					"run": tag,
					"task_split": split,
					"fault_operator": fault,
					"n_windows": len(batch),
					"participants": ",".join(sorted(set(batch.participant))),
					"trials": len(set(batch.trial)),
				}
			)
			print(f"[{time.time() - started:7.1f}s] {split}/{fault}: {len(variants)} gates, {len(batch)} windows")
			del batch, basis, variants

	out_dir = Path(cli.output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	write_tsv(out_dir / f"gate_baselines_{tag}.tsv", rows)
	write_tsv(out_dir / f"randomization_{tag}.tsv", random_rows)
	write_tsv(out_dir / f"run_manifest_{tag}.tsv", manifest)
	(out_dir / f"provenance_{tag}.json").write_text(
		json.dumps(
			{
				"report": str(cli.report),
				"checkpoint": str(cli.checkpoint),
				"checkpoint_sha256": rre_sha256(cli.checkpoint),
				"data_root": cli.data_root,
				"splits": cli.splits,
				"faults": cli.faults,
				"draws": cli.draws,
				"randomization_seed": cli.seed,
				"training_seed": seed_label,
				"gate_policy": {
					"softness": policy.softness,
					"deadband": policy.deadband,
					"max_fall_per_step": policy.max_fall_per_step,
					"max_rise_per_step": policy.max_rise_per_step,
					"signals": list(policy.signals),
					"source": policy.source,
				},
				"oracle_error_ref_q90_val_clean": error_ref,
				"risk_refs": risk_refs,
				"participants": {
					"val": sorted(val_p), "test": sorted(test_p), "train": sorted(train_p),
				},
				"cli": vars(cli),
				"elapsed_seconds": time.time() - started,
			},
			indent=2,
			ensure_ascii=False,
		),
		encoding="utf-8",
	)
	print(f"Wrote {out_dir}/gate_baselines_{tag}.tsv ({len(rows)} rows)")


def rre_sha256(path: str) -> str:
	import hashlib

	digest = hashlib.sha256()
	with open(path, "rb") as handle:
		for chunk in iter(lambda: handle.read(1 << 20), b""):
			digest.update(chunk)
	return digest.hexdigest()


if __name__ == "__main__":
	main()
