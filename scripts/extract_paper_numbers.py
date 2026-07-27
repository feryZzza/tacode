"""按 DATA_REQUIREMENTS.md §5 的固定口径，抽出可直接填进 main.tex 的数字。

汇总 TSV 不含 NLL / coverage ECE，且 macro 必须「先在 seed 内对 fault 等权平均，
再跨 seed 统计」——直接平均 TSV 的 std 是错的。所以这里从原始 JSON 重算。

约定：
- seen faults = 六个训练期故障（含 stuck_imu）；unseen = 三个未见生成器。
- clean retention = gated_retained_aligned_torque / baseline_retained_aligned_torque。
- wrong-direction reduction = baseline_wrong_direction_ratio - gated_wrong_direction_ratio。
  这里的 baseline_* 是同一 proposed estimator 的 ungated command，不是确定性 TCN。
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any

SEEN_FAULTS = (
	"insole_missing",
	"encoder_dropout",
	"imu_bias",
	"packet_loss",
	"stuck_imu",
	"sensor_delay",
)
UNSEEN_FAULTS = ("packet_loss_burst", "packet_loss_partial", "sensor_delay_jitter")
ALL_FAULTS = SEEN_FAULTS + UNSEEN_FAULTS
SPLITS = ("test_id", "test_ood")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--main-glob", default="reports/v2_fc_main_seed*/reliability_report.json")
	parser.add_argument("--baseline-glob", default="reports/v2_fc_baseline_det_seed*/reliability_report.json")
	parser.add_argument("--stress-glob", default="reports/v2_fc_stress_seed*/reliability_report.json")
	parser.add_argument("--ablation-glob", default="reports/v2_fc_ablation_*_seed7/reliability_report.json")
	parser.add_argument(
		"--ablation-seeds-glob",
		default="reports/v2_fc_ablation_*_seed*/reliability_report.json",
		help="任务 G：三个种子的消融梯度。按 training_mode 分组、种子内先做 macro 再跨种子统计。",
	)
	parser.add_argument("--action-glob", default="reports/v2_fc_action_*_seed7/reliability_report.json")
	parser.add_argument("--output", default="")
	return parser.parse_args()


def load(pattern: str) -> list[tuple[str, dict[str, Any]]]:
	out = []
	for path in sorted(glob.glob(pattern)):
		payload = json.loads(Path(path).read_text(encoding="utf-8"))
		out.append((Path(path).parent.name, payload))
	return out


def mean_sd(values: list[float]) -> dict[str, Any]:
	if not values:
		return {"mean": None, "sd": None, "n": 0}
	return {
		"mean": statistics.mean(values),
		"sd": statistics.stdev(values) if len(values) > 1 else 0.0,
		"n": len(values),
	}


def scenario_macro(results: dict[str, Any], field: str, splits, faults) -> float | None:
	"""先在给定 split×fault 集合上等权平均（seed 内 macro）。"""
	values = []
	for split in splits:
		for fault in faults:
			metrics = results.get(f"{split}/{fault}")
			if isinstance(metrics, dict) and isinstance(metrics.get(field), (int, float)):
				values.append(float(metrics[field]))
	return statistics.mean(values) if values else None


def detector_macro(results: dict[str, Any], field: str, splits, faults) -> float | None:
	values = []
	for split in splits:
		for fault in faults:
			metrics = results.get(f"fault_detection/{split}/{fault}")
			if isinstance(metrics, dict) and isinstance(metrics.get(field), (int, float)):
				values.append(float(metrics[field]))
	return statistics.mean(values) if values else None


def retention_ratio_macro(results: dict[str, Any], splits, scenarios) -> float | None:
	"""retained ratio = gated / ungated，先在 split×scenario 上等权平均。"""
	ratios = []
	for split in splits:
		for scenario in scenarios:
			metrics = results.get(f"{split}/{scenario}")
			if not isinstance(metrics, dict):
				continue
			gated = metrics.get("gated_retained_aligned_torque")
			ungated = metrics.get("baseline_retained_aligned_torque")
			if isinstance(gated, (int, float)) and isinstance(ungated, (int, float)) and ungated:
				ratios.append(float(gated) / float(ungated))
	return statistics.mean(ratios) if ratios else None


def retention_macro(results: dict[str, Any], splits) -> float | None:
	return retention_ratio_macro(results, splits, ("clean",))


def wrong_reduction_macro(results: dict[str, Any], splits, faults) -> float | None:
	deltas = []
	for split in splits:
		for fault in faults:
			metrics = results.get(f"{split}/{fault}")
			if not isinstance(metrics, dict):
				continue
			ungated = metrics.get("baseline_wrong_direction_ratio")
			gated = metrics.get("gated_wrong_direction_ratio")
			if isinstance(ungated, (int, float)) and isinstance(gated, (int, float)):
				deltas.append(float(ungated) - float(gated))
	return statistics.mean(deltas) if deltas else None


def collect(values_by_seed: dict[Any, float | None]) -> dict[str, Any]:
	clean = [v for v in values_by_seed.values() if v is not None]
	stats = mean_sd(clean)
	stats["per_seed"] = {str(k): v for k, v in sorted(values_by_seed.items(), key=lambda kv: str(kv[0]))}
	return stats


def main() -> None:
	args = parse_args()
	mains = load(args.main_glob)
	baselines = {p["args"].get("seed"): p for _, p in load(args.baseline_glob)}
	out: dict[str, Any] = {}

	if not mains:
		raise SystemExit(f"No main reports matched {args.main_glob}")

	# ---- Abstract ----
	wrong_by_seed, ret_by_seed, rmse_delta_by_seed = {}, {}, {}
	nll_by_seed, ece_by_seed = {}, {}
	for _, report in mains:
		seed = report["args"].get("seed")
		results = report["results"]
		wrong_by_seed[seed] = wrong_reduction_macro(results, SPLITS, ALL_FAULTS)
		ret_by_seed[seed] = retention_macro(results, SPLITS)
		nll_by_seed[seed] = scenario_macro(results, "nll", SPLITS, ("clean",))
		ece_by_seed[seed] = scenario_macro(results, "coverage_ece", SPLITS, ("clean",))
		base = baselines.get(seed)
		if base is not None:
			main_rmse = scenario_macro(results, "rmse", SPLITS, ("clean",))
			base_rmse = scenario_macro(base["results"], "rmse", SPLITS, ("clean",))
			if main_rmse is not None and base_rmse is not None:
				rmse_delta_by_seed[seed] = main_rmse - base_rmse

	out["abstract"] = {
		"wrong_direction_reduction_absolute": collect(wrong_by_seed),
		"clean_aligned_torque_retained_ratio": collect(ret_by_seed),
		"clean_rmse_change_vs_paired_baseline": collect(rmse_delta_by_seed),
	}

	# ---- Table 1: accuracy / calibration ----
	table1: dict[str, Any] = {"proposed_nll_clean_macro": collect(nll_by_seed), "proposed_ece_clean_macro": collect(ece_by_seed)}
	for split in SPLITS:
		for field in ("rmse", "r2", "nll", "coverage_ece"):
			main_vals = {}
			base_vals = {}
			for _, report in mains:
				seed = report["args"].get("seed")
				metrics = report["results"].get(f"{split}/clean", {})
				if isinstance(metrics.get(field), (int, float)):
					main_vals[seed] = float(metrics[field])
				base = baselines.get(seed)
				if base:
					bm = base["results"].get(f"{split}/clean", {})
					if isinstance(bm.get(field), (int, float)):
						base_vals[seed] = float(bm[field])
			table1[f"{split}/clean.{field}.proposed"] = collect(main_vals)
			if base_vals:
				table1[f"{split}/clean.{field}.paired_baseline"] = collect(base_vals)
	out["table1_accuracy"] = table1

	# ---- Table 2: detection macro AUROC ----
	detection: dict[str, Any] = {}
	for label, faults in (("seen", SEEN_FAULTS), ("unseen", UNSEEN_FAULTS), ("all", ALL_FAULTS)):
		for split in SPLITS:
			by_seed = {}
			for _, report in mains:
				by_seed[report["args"].get("seed")] = detector_macro(report["results"], "fault_auroc", (split,), faults)
			detection[f"macro_fused_auroc.{split}.{label}"] = collect(by_seed)
			# Detector-online：候选集限制在 K_gate 六路（门控实际可用）。摘要与主结论用这一组。
			online_by_seed = {}
			for _, report in mains:
				online_by_seed[report["args"].get("seed")] = detector_macro(
					report["results"], "fault_auroc_online", (split,), faults
				)
			if any(value is not None for value in online_by_seed.values()):
				detection[f"macro_online_auroc.{split}.{label}"] = collect(online_by_seed)
	# 逐 fault 的 fused AUROC 与最强单通道，供 Discussion 里的 hardest fault 用。
	per_fault: dict[str, Any] = {}
	for split in SPLITS:
		for fault in ALL_FAULTS:
			by_seed = {}
			online_by_seed = {}
			signals: dict[str, list[float]] = {}
			for _, report in mains:
				metrics = report["results"].get(f"fault_detection/{split}/{fault}", {})
				if isinstance(metrics.get("fault_auroc"), (int, float)):
					by_seed[report["args"].get("seed")] = float(metrics["fault_auroc"])
				if isinstance(metrics.get("fault_auroc_online"), (int, float)):
					online_by_seed[report["args"].get("seed")] = float(metrics["fault_auroc_online"])
				for key, value in metrics.items():
					# auroc_online 是融合口径，不是单通道，不能混进 signal_means。
					if key == "auroc_online":
						continue
					if key.startswith("auroc_") and isinstance(value, (int, float)):
						signals.setdefault(key[len("auroc_"):], []).append(float(value))
			entry = collect(by_seed)
			if online_by_seed:
				entry["online"] = collect(online_by_seed)
			entry["signal_means"] = {k: statistics.mean(v) for k, v in sorted(signals.items())}
			if entry["signal_means"]:
				entry["strongest_signal"] = max(entry["signal_means"], key=entry["signal_means"].get)
			per_fault[f"{split}/{fault}"] = entry
	detection["per_fault"] = per_fault
	out["table2_detection"] = detection

	# ---- Table 3: safety / utility ----
	safety: dict[str, Any] = {}
	rows = [("clean", ("clean",)), ("seen_faults", SEEN_FAULTS), ("unseen_faults", UNSEEN_FAULTS)]
	for split in SPLITS:
		for label, faults in rows:
			for field, alias in (
				("baseline_wrong_direction_ratio", "wrong_ungated"),
				("gated_wrong_direction_ratio", "wrong_gated"),
				# E_wrong 能量口径 + 峰值反向力矩 + 指令 jerk：正文 Safety--Utility 那句
				# 「Peak wrong torque and command jerk follow the same comparison」要有数。
				("baseline_wrong_energy", "wrong_energy_ungated"),
				("gated_wrong_energy", "wrong_energy_gated"),
				("baseline_peak_wrong_torque", "peak_wrong_ungated"),
				("gated_peak_wrong_torque", "peak_wrong_gated"),
				("baseline_mean_abs_jerk", "jerk_ungated"),
				("gated_mean_abs_jerk", "jerk_gated"),
				("mean_gate", "mean_gate"),
			):
				by_seed = {}
				for _, report in mains:
					by_seed[report["args"].get("seed")] = scenario_macro(report["results"], field, (split,), faults)
				safety[f"{split}.{label}.{alias}"] = collect(by_seed)
			delta_by_seed, ratio_by_seed = {}, {}
			for _, report in mains:
				seed = report["args"].get("seed")
				results = report["results"]
				ung = scenario_macro(results, "baseline_wrong_direction_ratio", (split,), faults)
				gat = scenario_macro(results, "gated_wrong_direction_ratio", (split,), faults)
				delta_by_seed[seed] = None if ung is None or gat is None else gat - ung
				ratios = []
				for fault in faults:
					metrics = results.get(f"{split}/{fault}")
					if not isinstance(metrics, dict):
						continue
					g = metrics.get("gated_retained_aligned_torque")
					u = metrics.get("baseline_retained_aligned_torque")
					if isinstance(g, (int, float)) and isinstance(u, (int, float)) and u:
						ratios.append(float(g) / float(u))
				ratio_by_seed[seed] = statistics.mean(ratios) if ratios else None
			safety[f"{split}.{label}.wrong_delta_gated_minus_ungated"] = collect(delta_by_seed)
			safety[f"{split}.{label}.retained_ratio"] = collect(ratio_by_seed)
	out["table3_safety"] = safety

	# ---- Table 4: ablation ladder ----
	order = ["det_noaug", "det_aug", "prob_aug", "prob_aug_recon", "prob_aug_recon_fc"]
	ablation_rows: dict[str, Any] = {}
	for run_name, report in load(args.ablation_glob):
		mode = report["args"].get("training_mode")
		results = report["results"]
		ood_clean = results.get("test_ood/clean", {})
		ablation_rows[mode] = {
			"run": run_name,
			"ood_clean_rmse": ood_clean.get("rmse"),
			"macro_fused_auroc": detector_macro(results, "fault_auroc", SPLITS, ALL_FAULTS),
			"wrong_delta_gated_minus_ungated": (
				lambda ung, gat: None if ung is None or gat is None else gat - ung
			)(
				scenario_macro(results, "baseline_wrong_direction_ratio", SPLITS, ALL_FAULTS),
				scenario_macro(results, "gated_wrong_direction_ratio", SPLITS, ALL_FAULTS),
			),
			"clean_retained_ratio": retention_macro(results, SPLITS),
			# 正文消融段的 0.731→0.798 序列是故障集合上的 retained ratio macro，
			# 不是 clean 那一列；两者都留着，避免下次又猜口径。
			"fault_retained_ratio": retention_ratio_macro(results, SPLITS, ALL_FAULTS),
		}
	stages = []
	for index, mode in enumerate(order):
		row = ablation_rows.get(mode)
		if row is None:
			continue
		entry = {"stage": index + 1, "mode": mode, **row}
		if index > 0 and order[index - 1] in ablation_rows:
			prev = ablation_rows[order[index - 1]]
			entry["delta_vs_previous"] = {
				key: (None if row.get(key) is None or prev.get(key) is None else row[key] - prev[key])
				for key in (
					"ood_clean_rmse",
					"macro_fused_auroc",
					"clean_retained_ratio",
					"fault_retained_ratio",
				)
			}
		stages.append(entry)
	out["table4_ablation"] = stages

	# ---- 任务 G：同一梯度的三种子版本（seed 内 macro，再跨 seed mean/sd）----
	FIELDS = ("ood_clean_rmse", "macro_fused_auroc", "wrong_delta_gated_minus_ungated",
	          "clean_retained_ratio", "fault_retained_ratio")
	by_mode: dict[str, dict[int, dict[str, Any]]] = {}
	for run_name, report in load(args.ablation_seeds_glob):
		mode = report["args"].get("training_mode")
		seed = report["args"].get("seed")
		if mode is None or seed is None:
			continue
		results = report["results"]
		ung = scenario_macro(results, "baseline_wrong_direction_ratio", SPLITS, ALL_FAULTS)
		gat = scenario_macro(results, "gated_wrong_direction_ratio", SPLITS, ALL_FAULTS)
		by_mode.setdefault(mode, {})[int(seed)] = {
			"run": run_name,
			"ood_clean_rmse": results.get("test_ood/clean", {}).get("rmse"),
			"macro_fused_auroc": detector_macro(results, "fault_auroc", SPLITS, ALL_FAULTS),
			"wrong_delta_gated_minus_ungated": None if ung is None or gat is None else gat - ung,
			"clean_retained_ratio": retention_macro(results, SPLITS),
			"fault_retained_ratio": retention_ratio_macro(results, SPLITS, ALL_FAULTS),
		}
	multi = []
	for index, mode in enumerate(order):
		per_seed = by_mode.get(mode)
		if not per_seed:
			continue
		entry: dict[str, Any] = {
			"stage": index + 1,
			"mode": mode,
			"seeds": sorted(per_seed),
			"runs": [per_seed[s]["run"] for s in sorted(per_seed)],
		}
		for field in FIELDS:
			vals = [per_seed[s][field] for s in sorted(per_seed)
			        if isinstance(per_seed[s].get(field), (int, float))]
			entry[field] = mean_sd(vals)
		multi.append(entry)
	for index in range(1, len(multi)):
		prev, cur = multi[index - 1], multi[index]
		# 配对差值：只在两阶段都有的种子上做差，再跨种子统计。
		shared = sorted(set(by_mode[prev["mode"]]) & set(by_mode[cur["mode"]]))
		cur["delta_vs_previous"] = {}
		for field in FIELDS:
			diffs = [
				by_mode[cur["mode"]][s][field] - by_mode[prev["mode"]][s][field]
				for s in shared
				if isinstance(by_mode[cur["mode"]][s].get(field), (int, float))
				and isinstance(by_mode[prev["mode"]][s].get(field), (int, float))
			]
			cur["delta_vs_previous"][field] = mean_sd(diffs)
	out["table4_ablation_multiseed"] = multi

	# ---- Stress: full severity curve (fig_stress_transfer) + 最大强度点 ----
	stress_reports = load(args.stress_glob)
	stress_scenarios: list[str] = []
	for _, report in stress_reports:
		for key in report["results"]:
			if key.startswith("test_id/") and "@" in key:
				name = key.split("/", 1)[1]
				if name not in stress_scenarios:
					stress_scenarios.append(name)
	curve: dict[str, Any] = {}
	for scenario in stress_scenarios:
		for split in SPLITS:
			auroc, ret, wrong = {}, {}, {}
			for _, report in stress_reports:
				seed = report["args"].get("seed")
				results = report["results"]
				det = results.get(f"fault_detection/{split}/{scenario}", {})
				if isinstance(det.get("fault_auroc"), (int, float)):
					auroc[seed] = float(det["fault_auroc"])
				metrics = results.get(f"{split}/{scenario}", {})
				g = metrics.get("gated_retained_aligned_torque")
				u = metrics.get("baseline_retained_aligned_torque")
				if isinstance(g, (int, float)) and isinstance(u, (int, float)) and u:
					ret[seed] = float(g) / float(u)
				ung = metrics.get("baseline_wrong_direction_ratio")
				gat = metrics.get("gated_wrong_direction_ratio")
				if isinstance(ung, (int, float)) and isinstance(gat, (int, float)):
					wrong[seed] = gat - ung
			curve[f"{split}/{scenario}"] = {
				"fused_auroc": collect(auroc),
				"retained_ratio": collect(ret),
				"wrong_delta": collect(wrong),
			}
	out["stress_curve"] = curve

	stress: dict[str, Any] = {}
	for scenario in ("packet_loss@0.30", "sensor_delay@20"):
		for split in SPLITS:
			auroc, online, ret, wrong = {}, {}, {}, {}
			for _, report in stress_reports:
				seed = report["args"].get("seed")
				results = report["results"]
				det = results.get(f"fault_detection/{split}/{scenario}", {})
				if isinstance(det.get("fault_auroc"), (int, float)):
					auroc[seed] = float(det["fault_auroc"])
				# 极值点在两个融合口径下差异最大（延迟类），正文两个都报。
				if isinstance(det.get("fault_auroc_online"), (int, float)):
					online[seed] = float(det["fault_auroc_online"])
				metrics = results.get(f"{split}/{scenario}", {})
				g = metrics.get("gated_retained_aligned_torque")
				u = metrics.get("baseline_retained_aligned_torque")
				if isinstance(g, (int, float)) and isinstance(u, (int, float)) and u:
					ret[seed] = float(g) / float(u)
				ung = metrics.get("baseline_wrong_direction_ratio")
				gat = metrics.get("gated_wrong_direction_ratio")
				if isinstance(ung, (int, float)) and isinstance(gat, (int, float)):
					wrong[seed] = gat - ung
			stress[f"{split}/{scenario}.fused_auroc"] = collect(auroc)
			stress[f"{split}/{scenario}.online_auroc"] = collect(online)
			stress[f"{split}/{scenario}.retained_ratio"] = collect(ret)
			stress[f"{split}/{scenario}.wrong_delta"] = collect(wrong)
	out["stress_max_severity"] = stress

	# ---- Action-input ablation (single seed; qualitative) ----
	action: dict[str, Any] = {}
	action_reports = load(args.action_glob)
	reference = mains[0][1] if mains else None
	for _, report in mains:
		if report["args"].get("seed") == 7:
			reference = report
			break
	if reference is not None:
		action["human (reference, 25ch)"] = {
			"macro_fused_auroc": detector_macro(reference["results"], "fault_auroc", SPLITS, ALL_FAULTS),
			"ood_clean_rmse": reference["results"].get("test_ood/clean", {}).get("rmse"),
			"clean_retained_ratio": retention_macro(reference["results"], SPLITS),
		}
	for run_name, report in action_reports:
		action[report["args"].get("input_profile", run_name)] = {
			"macro_fused_auroc": detector_macro(report["results"], "fault_auroc", SPLITS, ALL_FAULTS),
			"ood_clean_rmse": report["results"].get("test_ood/clean", {}).get("rmse"),
			"clean_retained_ratio": retention_macro(report["results"], SPLITS),
			# 加动作通道若真带来「设备专用捷径」，会表现为 ID 检测涨、OOD 不涨。
			"macro_fused_auroc_id": detector_macro(
				report["results"], "fault_auroc", ("test_id",), ALL_FAULTS
			),
			"macro_fused_auroc_ood": detector_macro(
				report["results"], "fault_auroc", ("test_ood",), ALL_FAULTS
			),
			"n_input_channels": len(report.get("input_names") or ()) or None,
		}
	if "human (reference, 25ch)" in action and reference is not None:
		action["human (reference, 25ch)"].update(
			{
				"macro_fused_auroc_id": detector_macro(
					reference["results"], "fault_auroc", ("test_id",), ALL_FAULTS
				),
				"macro_fused_auroc_ood": detector_macro(
					reference["results"], "fault_auroc", ("test_ood",), ALL_FAULTS
				),
				"n_input_channels": len(reference.get("input_names") or ()) or None,
			}
		)
	out["action_input_ablation"] = action

	# ---- Gate policy range ----
	# 正文那句「平均门控保持在 x–y 之间」说的是安全表六行的取值范围
	# （clean / seen / unseen × ID / OOD），不是只看 clean，所以直接从 table3 取。
	gate_row_means = [
		value["mean"]
		for key, value in safety.items()
		if key.endswith(".mean_gate") and isinstance(value, dict) and value.get("mean") is not None
	]
	out["gate_mean_range_safety_rows"] = {
		"min": min(gate_row_means) if gate_row_means else None,
		"max": max(gate_row_means) if gate_row_means else None,
		"n_rows": len(gate_row_means),
	}
	gate_means = []
	for _, report in mains:
		value = scenario_macro(report["results"], "mean_gate", SPLITS, ("clean",))
		if value is not None:
			gate_means.append(value)
	out["gate_mean_clean_range"] = {"min": min(gate_means) if gate_means else None, "max": max(gate_means) if gate_means else None}
	out["_provenance"] = {
		"main_runs": [name for name, _ in mains],
		"paired_baseline_seeds": sorted(str(s) for s in baselines),
		"seen_faults": list(SEEN_FAULTS),
		"unseen_faults": list(UNSEEN_FAULTS),
	}

	text = json.dumps(out, indent=2)
	print(text)
	if args.output:
		Path(args.output).write_text(text + "\n", encoding="utf-8")
		print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
	main()
