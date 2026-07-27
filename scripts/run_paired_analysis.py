#!/usr/bin/env python3
"""EXPERIMENTS_TODO §5：同窗配对的 fault-specific 效应与被试级 bootstrap CI。

要回答的问题不是「加门控后指标变好了吗」，而是「门控的收益是否**特定于故障**」。
后者需要一个双重差分，且两个差都必须在**同一个窗口**上取：

    Δ_fault-specific = (M^f_gated − M^f_ungated) − (M^clean_gated − M^clean_ungated)

减掉 clean 侧那一项，是因为门控在干净数据上也会误关一部分——那部分 X_opp 的下降不是
「检测到故障」的功劳。跨窗平均后再相减做不到这件事：窗口本身的难度差异会整体平移两项。

聚合顺序按 §5 严格执行：窗口内配对 → 被试内平均 → 被试级 bootstrap 95% CI。被试是
唯一的独立重复单位，窗口不是（同一被试的窗口高度相关，按窗口 bootstrap 会把 CI 压窄
好几倍）。`--seeds` 给多个训练种子时额外落一张种子敏感性表。

用法：
    python scripts/run_paired_analysis.py \
        --report reports/v2_fc_main_seed7/reliability_report.json \
        --checkpoint reports/v2_main_seed7/reliability_tcn_best.pt \
        --output-dir reports/aaai27_paired_analysis
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
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
	moment_to_torque,
	reliability_gate,
	risk_from_channels,
)
from reliability.model import load_checkpoint
from reliability.nature_dataset import discover_trials, filter_records

SAMPLE_RATE = 200.0

#: 逐窗配对的指标。每个都必须能在单窗口上算出来（否则没法配对）。
WINDOW_METRICS = ("x_opp", "aligned_command_retention", "tracking_rmse", "mean_abs_torque_rate")


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--report", required=True, help="主 seed 的 report，用于读回冻结门控")
	p.add_argument("--checkpoint", required=True)
	p.add_argument("--extra-runs", default="", help="种子敏感性用的额外 report:checkpoint 对，逗号分隔")
	p.add_argument("--data-root", default="data/Parsed")
	p.add_argument("--output-dir", required=True)
	p.add_argument("--splits", default="test_id,test_ood")
	p.add_argument(
		"--faults",
		default="insole_missing,encoder_dropout,imu_bias,packet_loss,packet_loss_burst,"
		"packet_loss_partial,stuck_imu,sensor_delay,sensor_delay_jitter",
	)
	p.add_argument("--bootstrap", type=int, default=10000, help="被试级 bootstrap 重采样次数")
	p.add_argument("--seed", type=int, default=0)
	p.add_argument("--device", default="cuda")
	p.add_argument("--batch-size", type=int, default=32)
	p.add_argument("--num-workers", type=int, default=8)
	p.add_argument("--max-windows", type=int, default=0)
	return p.parse_args()


def build_eval_args(report_args: dict, cli: argparse.Namespace) -> argparse.Namespace:
	ns = argparse.Namespace(**report_args)
	ns.data_root = cli.data_root
	ns.device = cli.device
	ns.mode = "eval"
	ns.batch_size = cli.batch_size
	ns.num_workers = cli.num_workers
	ns.prefetch_factor = 2
	ns.max_eval_batches = 0
	ns.extra_models = []
	ns.mc_samples = int(report_args.get("mc_samples") or 0)
	return ns


def per_window_metrics(
	batch: WindowBatch,
	gate: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
	"""逐窗口指标，返回每个指标的 [N] 张量。gate=None 表示 ungated。

	与 `metrics_for_given_gate` 的差别只在归约范围：这里**不跨窗口**求和，因为配对分析
	需要每个窗口自己的值。active mask 与 floor 与全局版本保持一致，否则两套数字不可比。
	"""
	ideal = moment_to_torque(batch.y)
	command = moment_to_torque(batch.mean)
	if gate is not None:
		g = gate
		if g.dim() == 3 and g.shape[1] == 1:
			g = g.expand_as(command)
		command = command * g
	valid = batch.valid_time().expand_as(ideal)
	active = valid & (ideal.abs() > ACTIVE_TORQUE_FLOOR)
	dims = (1, 2)
	n_active = active.sum(dim=dims).clamp_min(1).to(torch.float64)

	opposition = torch.clamp(-(command * ideal), min=0.0) * active
	aligned = torch.clamp(command * torch.sign(ideal), min=0.0) * active
	denom = (ideal.abs() * active).sum(dim=dims).to(torch.float64)
	err = ((command - ideal) ** 2 * active).sum(dim=dims).to(torch.float64)
	rate = (command[..., 1:] - command[..., :-1]).abs() * active[..., 1:]
	n_rate = active[..., 1:].sum(dim=dims).clamp_min(1).to(torch.float64)

	return {
		"x_opp": opposition.sum(dim=dims).to(torch.float64) / SAMPLE_RATE,
		"aligned_command_retention": torch.where(
			denom > 1e-9, aligned.sum(dim=dims).to(torch.float64) / denom.clamp_min(1e-9),
			torch.full_like(denom, float("nan")),
		),
		"tracking_rmse": torch.sqrt(err / n_active),
		"mean_abs_torque_rate": rate.sum(dim=dims).to(torch.float64) / n_rate * SAMPLE_RATE,
		"n_active": n_active,
	}


def bootstrap_ci(
	values: Sequence[float],
	draws: int,
	rng: random.Random,
	alpha: float = 0.05,
) -> Tuple[float, float, float, float]:
	"""被试级 bootstrap：返回 (mean, lo, hi, p_two_sided_vs_zero)。

	被试数只有个位数，正态近似不可靠，所以走 percentile bootstrap。双侧 p 用
	「重采样均值跨过 0 的比例」的两倍近似——它与 CI 是否含 0 一致，避免两个口径打架。
	"""
	clean = [v for v in values if v == v]
	if len(clean) < 2:
		return (clean[0] if clean else float("nan")), float("nan"), float("nan"), float("nan")
	n = len(clean)
	means: List[float] = []
	for _ in range(draws):
		means.append(sum(clean[rng.randrange(n)] for _ in range(n)) / n)
	means.sort()
	lo = means[max(int(round(alpha / 2 * (draws - 1))), 0)]
	hi = means[min(int(round((1 - alpha / 2) * (draws - 1))), draws - 1)]
	point = sum(clean) / n
	tail = sum(1 for m in means if (m >= 0.0 if point < 0 else m <= 0.0)) / draws
	return point, lo, hi, min(2.0 * tail, 1.0)


#: 被试级 bootstrap 要想给出有意义的区间，最少需要这么多被试。
#: n=2 时重采样只有 3 种可能的均值（aa, ab, bb），两个被试同号就必然 p=0 且区间不含 0——
#: 这是重采样空间退化，不是证据。低于这个数的格子一律标 underpowered，正文不能当显著性用。
MIN_PARTICIPANTS_FOR_INFERENCE = 4


def is_underpowered(n_participants: int) -> bool:
	return n_participants < MIN_PARTICIPANTS_FOR_INFERENCE


def gate_for(batch: WindowBatch, policy: GatePolicy, risk_refs) -> Optional[torch.Tensor]:
	channels = channel_subset(batch, policy.signals)
	if not channels:
		return None
	risk = risk_from_channels(channels, risk_refs)
	return causal_gate_rate_limit(
		reliability_gate(risk, softness=policy.softness, deadband=policy.deadband),
		max_fall_per_step=policy.max_fall_per_step,
		max_rise_per_step=policy.max_rise_per_step,
	)


def analyse_run(
	report_path: str,
	checkpoint: str,
	cli: argparse.Namespace,
	rng: random.Random,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
	report = load_report(report_path)
	policy = GatePolicy.from_report(report)
	args = build_eval_args(report["args"], cli)
	args.checkpoint = checkpoint
	tag = Path(report_path).parent.name
	seed_label = report["args"].get("seed", "")

	input_names = input_names_for_profile(args.input_profile, args.side)
	target_names = label_names(args.side)
	groups = feature_groups(input_names)
	args.velocity_indices = joint_velocity_indices(input_names, target_names)

	records = discover_trials(args.data_root)
	heldout = rre._parse_csv(args.heldout_tasks)
	_, val_p, test_p = rre.resolve_split(records, args)
	split_records = {
		"val": filter_records(records, participants=val_p, exclude_tasks=heldout),
		"test_id": filter_records(records, participants=test_p, exclude_tasks=heldout),
		"test_ood": filter_records(records, participants=test_p, include_tasks=heldout),
	}

	device = torch.device(args.device)
	model, payload = load_checkpoint(checkpoint, map_location=device)
	rre.validate_checkpoint_inputs(payload, input_names, target_names)
	model = model.to(device).eval()

	val_dataset = rre.make_dataset(split_records["val"], input_names, target_names, args)
	risk_refs = rre.estimate_risk_refs(model, val_dataset, groups, args, device)
	max_windows = cli.max_windows or None

	summary_rows: List[Dict[str, object]] = []
	window_rows: List[Dict[str, object]] = []
	manifest_rows: List[Dict[str, object]] = []

	for split in rre._parse_csv(cli.splits):
		if not split_records.get(split):
			continue
		dataset = rre.make_dataset(split_records[split], input_names, target_names, args)
		clean = collect_windows(
			model, dataset, groups, args, device, "clean",
			generator_seed=cli.seed, max_windows=max_windows,
		)
		if clean is None:
			continue
		clean_gate = gate_for(clean, policy, risk_refs)
		if clean_gate is None:
			continue
		clean_gated = per_window_metrics(clean, clean_gate)
		clean_ungated = per_window_metrics(clean, None)
		clean_keys = clean.window_keys()
		clean_index = {key: i for i, key in enumerate(clean_keys)}

		for fault in rre._parse_csv(cli.faults):
			batch = collect_windows(
				model, dataset, groups, args, device, fault,
				generator_seed=cli.seed, max_windows=max_windows,
			)
			if batch is None:
				continue
			gate = gate_for(batch, policy, risk_refs)
			if gate is None:
				continue
			f_gated = per_window_metrics(batch, gate)
			f_ungated = per_window_metrics(batch, None)

			# 配对：同 (participant, trial, window start) 才配。取不到就丢，不用近似匹配。
			pairs: List[Tuple[int, int, str]] = []
			for j, key in enumerate(batch.window_keys()):
				i = clean_index.get(key)
				if i is not None:
					pairs.append((i, j, batch.participant[j]))

			per_participant: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
			for i, j, participant in pairs:
				row: Dict[str, object] = {
					"run": tag, "seed": seed_label, "task_split": split,
					"fault_operator": fault, "participant": participant,
					"trial": batch.trial[j], "window_start": batch.start[j],
				}
				for metric in WINDOW_METRICS:
					fault_effect = float(f_gated[metric][j] - f_ungated[metric][j])
					clean_effect = float(clean_gated[metric][i] - clean_ungated[metric][i])
					delta = fault_effect - clean_effect
					row[f"{metric}_fault_gated"] = float(f_gated[metric][j])
					row[f"{metric}_fault_ungated"] = float(f_ungated[metric][j])
					row[f"{metric}_clean_gated"] = float(clean_gated[metric][i])
					row[f"{metric}_clean_ungated"] = float(clean_ungated[metric][i])
					row[f"{metric}_delta_gate_fault"] = fault_effect
					row[f"{metric}_delta_gate_clean"] = clean_effect
					row[f"{metric}_delta_fault_specific"] = delta
					if delta == delta:
						per_participant[participant][metric].append(delta)
				window_rows.append(row)

			for metric in WINDOW_METRICS:
				# 被试内先平均（每个被试一个数），再对被试做 bootstrap。
				participant_means = [
					sum(v) / len(v) for v in
					(per_participant[p][metric] for p in sorted(per_participant))
					if v
				]
				point, lo, hi, pval = bootstrap_ci(participant_means, cli.bootstrap, rng)
				summary_rows.append(
					{
						"run": tag, "seed": seed_label, "task_split": split,
						"fault_operator": fault, "metric": metric,
						"n_pairs": len(pairs),
						"n_participants": len(participant_means),
						"delta_fault_specific_mean": point,
						"ci95_low": lo, "ci95_high": hi,
						"bootstrap_p_two_sided": pval,
						"significant_at_95": (
							lo == lo and hi == hi and (lo > 0.0 or hi < 0.0)
							and not is_underpowered(len(participant_means))
						),
						# 区间本身照样落盘（它描述了被试间的散布），但 n 太小时不允许它
						# 被读成显著性结论。两列分开，免得下游只看到一个布尔。
						"ci_excludes_zero": (lo == lo and hi == hi and (lo > 0.0 or hi < 0.0)),
						"underpowered": is_underpowered(len(participant_means)),
						"min_participants_for_inference": MIN_PARTICIPANTS_FOR_INFERENCE,
						"participant_values": ";".join(f"{v:.6g}" for v in participant_means),
					}
				)
			# §9 的交付清单：每个 (split, fault) 实际配上了多少窗口、丢了多少、
			# 落在哪些被试/试验上。丢弃数必须显式记，否则「配对窗口 337」看不出
			# 是全配上了还是丢了一半。
			manifest_rows.append(
				{
					"run": tag,
					"seed": seed_label,
					"task_split": split,
					"fault_operator": fault,
					"n_fault_windows": len(batch),
					"n_clean_windows": len(clean),
					"n_paired": len(pairs),
					"n_unmatched_dropped": len(batch) - len(pairs),
					"participants": ",".join(sorted(per_participant)),
					"n_participants": len(per_participant),
					"trials": len({batch.trial[j] for _, j, _ in pairs}),
				}
			)
			print(f"  {split}/{fault}: {len(pairs)} 配对窗口, {len(per_participant)} 被试")
			del batch, gate, f_gated, f_ungated
		del clean, clean_gate, clean_gated, clean_ungated

	provenance = {
		"report": report_path,
		"checkpoint": checkpoint,
		"training_seed": seed_label,
		"gate_policy": {
			"softness": policy.softness, "deadband": policy.deadband,
			"max_fall_per_step": policy.max_fall_per_step,
			"max_rise_per_step": policy.max_rise_per_step,
			"signals": list(policy.signals), "source": policy.source,
		},
		"risk_refs": risk_refs,
		"participants": {"val": sorted(val_p), "test": sorted(test_p)},
	}
	return summary_rows, window_rows, manifest_rows, provenance


def main() -> None:
	cli = parse_args()
	rng = random.Random(cli.seed)
	runs: List[Tuple[str, str]] = [(cli.report, cli.checkpoint)]
	for item in rre._parse_csv(cli.extra_runs):
		if ":" in item:
			rep, ckpt = item.split(":", 1)
			runs.append((rep, ckpt))

	all_summary: List[Dict[str, object]] = []
	all_windows: List[Dict[str, object]] = []
	all_manifest: List[Dict[str, object]] = []
	provenances: List[Dict[str, object]] = []
	started = time.time()
	for report_path, checkpoint in runs:
		print(f"[{time.time() - started:7.1f}s] {report_path}")
		summary, windows, manifest, prov = analyse_run(report_path, checkpoint, cli, rng)
		all_summary.extend(summary)
		all_windows.extend(windows)
		all_manifest.extend(manifest)
		provenances.append(prov)

	# 种子敏感性：同 (split, fault, metric) across seeds 的散布。
	grouped: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
	for row in all_summary:
		value = row["delta_fault_specific_mean"]
		if isinstance(value, float) and value == value:
			grouped[(str(row["task_split"]), str(row["fault_operator"]), str(row["metric"]))].append(value)
	seed_rows = [
		{
			"task_split": split, "fault_operator": fault, "metric": metric,
			"n_seeds": len(values),
			"across_seed_mean": sum(values) / len(values),
			"across_seed_sd": (
				math.sqrt(sum((v - sum(values) / len(values)) ** 2 for v in values) / (len(values) - 1))
				if len(values) > 1 else 0.0
			),
			"across_seed_min": min(values), "across_seed_max": max(values),
			"sign_consistent": all(v > 0 for v in values) or all(v < 0 for v in values),
		}
		for (split, fault, metric), values in sorted(grouped.items())
	]

	out_dir = Path(cli.output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	write_tsv(out_dir / "run_manifest_paired.tsv", all_manifest)
	write_tsv(out_dir / "paired_summary.tsv", all_summary)
	write_tsv(out_dir / "paired_windows.tsv", all_windows)
	write_tsv(out_dir / "seed_sensitivity.tsv", seed_rows)
	(out_dir / "provenance_paired.json").write_text(
		json.dumps(
			{
				"runs": provenances,
				"estimator": "delta_fault_specific = (M_f_gated - M_f_ungated) - (M_clean_gated - M_clean_ungated)",
				"aggregation": "pair within window -> mean within participant -> participant-level percentile bootstrap",
				"bootstrap_draws": cli.bootstrap,
				"bootstrap_seed": cli.seed,
				"metrics": list(WINDOW_METRICS),
				"data_root": cli.data_root,
				"cli": vars(cli),
				"elapsed_seconds": time.time() - started,
			},
			indent=2,
			ensure_ascii=False,
		),
		encoding="utf-8",
	)
	print(f"Wrote {out_dir}/paired_summary.tsv ({len(all_summary)} rows, {len(all_windows)} paired windows)")


if __name__ == "__main__":
	main()
