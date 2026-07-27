#!/usr/bin/env python3
"""把稿件引用的数字从新汇总里一次性打印出来，供逐位核对。

只读 reports/v2_fc_paper_suite/paper_numbers.json，不改任何文件。
每行给出「正文位置 / 稿件值 / suite 值」，稿件值是人工填的期望，
不一致就打 DIFF——用来替代逐个 grep 表格。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def get(obj: Any, dotted: str) -> Any:
	"""按点路径取值，但键本身可能含点（suite 用扁平键如
	`macro_fused_auroc.test_id.seen`），所以每一层都先试最长匹配。"""
	cur = obj
	rest = dotted
	while rest:
		if not isinstance(cur, dict):
			return None
		if rest in cur:
			return cur[rest]
		parts = rest.split(".")
		for cut in range(len(parts) - 1, 0, -1):
			head = ".".join(parts[:cut])
			if head in cur:
				cur = cur[head]
				rest = ".".join(parts[cut:])
				break
		else:
			return None
	return cur


def fmt(value: Any) -> str:
	if value is None:
		return "缺失"
	if isinstance(value, dict):
		# collect() 用的是 sd，不是 std。
		mean = value.get("mean")
		std = value.get("sd", value.get("std"))
		n = value.get("n")
		if mean is None:
			return json.dumps(value, ensure_ascii=False)
		std_s = "n/a" if std is None else f"{std:.4f}"
		return f"{mean:.4f} ± {std_s} (n={n})"
	if isinstance(value, float):
		return f"{value:.4f}"
	return str(value)


#: AAAI-27 新增章节（EXPERIMENTS_TODO §1–§6）在正文里引用的数字。
#: 与上面的旧 suite 分开：那些键来自 `paper_numbers.json`，这些来自
#: `aaai27_gate_numbers.json`（由 extract_aaai27_gate_numbers.py 生成）。
GATE_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
	(
		"§1 工作点重选（val 侧六项 clean 约束；跨 3 seed）",
		[
			("gate_reselection.objective", "x_opp"),
			("gate_reselection.val_clean_x_opp.mean", "0.100"),
			("gate_reselection.val_fault_x_opp.mean", "0.113"),
			("gate_reselection.val_clean_tracking_rmse.mean", "0.0456"),
			("gate_reselection.val_clean_capped_overlap_retention.mean", "0.631"),
			("gate_reselection.val_clean_full_shutdown_fraction.mean", "0.0028"),
			("gate_reselection.val_clean_shutdowns_per_minute.mean", "0.508"),
			("gate_reselection.val_clean_mean_abs_torque_rate.mean", "0.258"),
			("gate_reselection.rate_limit_identical_across_seeds", "False"),
		],
	),
	(
		"§2 随机化检验（最坏格子必须进正文）",
		[
			("gate_baselines.draws", "1000"),
			("gate_baselines.randomization_n_cells", "120"),
			("gate_baselines.randomization_worst.random_attenuation.percentile", "1.000"),
			("gate_baselines.randomization_worst.random_shutdown.percentile", "0.272"),
		],
	),
	(
		"§4 瞬态（负对照 + 短故障漏检）",
		[
			("transient_faults.n_rows", "1608"),
		],
	),
	(
		"§5 配对双重差分（被试数是硬上限）",
		[
			("paired_analysis.n_underpowered_cells", "216"),
		],
	),
	(
		"§6 反向消融（七格 × 3 seed；另加 all-gate-channels 参照格，故为 8）",
		[
			# §6 要七格，这里是 8：多出来的 `all_gate_channels` 是评测侧参照
			# （full stack + 六路门控全开），用来把「去掉某一路」的效应量与
			# 「什么都不去掉」对齐。它不是第八个消融，但要进 Pareto 才能比。
			("reverse_ablation.n_cells", "8"),
			("reverse_ablation.cells_with_three_seeds", "8"),
		],
	),
]


def main() -> int:
	parser = argparse.ArgumentParser()
	parser.add_argument("--suite-dir", default="reports/v2_fc_paper_suite")
	parser.add_argument("--gate-numbers", default="reports/aaai27_gate_numbers.json",
		help="extract_aaai27_gate_numbers.py 的产物；缺失时只跳过 §1–§6 那几组")
	args = parser.parse_args()

	suite = Path(args.suite_dir)
	numbers = json.loads((suite / "paper_numbers.json").read_text())

	groups: list[tuple[str, list[tuple[str, str]]]] = [
		(
			"检测宏平均（Detector-gate / all；suite 兼容键仍为 online）",
			[
				("table2_detection.macro_online_auroc.test_id.seen", "0.815"),
				("table2_detection.macro_online_auroc.test_ood.seen", "0.740"),
				("table2_detection.macro_online_auroc.test_id.unseen", "0.812"),
				("table2_detection.macro_online_auroc.test_ood.unseen", "0.715"),
				("table2_detection.macro_fused_auroc.test_id.seen", "0.849"),
				("table2_detection.macro_fused_auroc.test_ood.seen", "0.773"),
				("table2_detection.macro_fused_auroc.test_id.unseen", "0.842"),
				("table2_detection.macro_fused_auroc.test_ood.unseen", "0.778"),
			],
		),
		(
			"强度极值 packet_loss@0.30",
			[
				("stress_max_severity.test_id/packet_loss@0.30.fused_auroc", "0.961"),
				("stress_max_severity.test_id/packet_loss@0.30.online_auroc", "0.961"),
				("stress_max_severity.test_ood/packet_loss@0.30.fused_auroc", "0.914"),
				("stress_max_severity.test_ood/packet_loss@0.30.online_auroc", "0.914"),
				("stress_max_severity.test_id/packet_loss@0.30.retained_ratio", "0.939"),
				("stress_max_severity.test_ood/packet_loss@0.30.retained_ratio", "0.938"),
				("stress_max_severity.test_id/packet_loss@0.30.wrong_delta", "-0.0007"),
				("stress_max_severity.test_ood/packet_loss@0.30.wrong_delta", "-0.0080"),
			],
		),
		(
			"强度极值 sensor_delay@20",
			[
				("stress_max_severity.test_id/sensor_delay@20.fused_auroc", "0.899"),
				("stress_max_severity.test_id/sensor_delay@20.online_auroc", "0.743"),
				("stress_max_severity.test_ood/sensor_delay@20.fused_auroc", "0.788"),
				("stress_max_severity.test_ood/sensor_delay@20.online_auroc", "0.588"),
				("stress_max_severity.test_id/sensor_delay@20.retained_ratio", "0.751"),
				("stress_max_severity.test_ood/sensor_delay@20.retained_ratio", "0.933"),
				("stress_max_severity.test_id/sensor_delay@20.wrong_delta", "-0.0132"),
				("stress_max_severity.test_ood/sensor_delay@20.wrong_delta", "-0.0230"),
			],
		),
		(
			"安全力矩乘积积分与峰值/力矩变化率（留出任务；suite 使用历史键名）",
			[
				("table3_safety.test_ood.seen_faults.wrong_energy_ungated", "0.0439"),
				("table3_safety.test_ood.seen_faults.wrong_energy_gated", "0.0298"),
				("table3_safety.test_ood.unseen_faults.wrong_energy_ungated", "0.0412"),
				("table3_safety.test_ood.unseen_faults.wrong_energy_gated", "0.0270"),
				("table3_safety.test_ood.seen_faults.peak_wrong_ungated", "0.175"),
				("table3_safety.test_ood.seen_faults.peak_wrong_gated", "0.137"),
				("table3_safety.test_ood.seen_faults.jerk_ungated", "0.237"),
				("table3_safety.test_ood.seen_faults.jerk_gated", "0.279"),
			],
		),
		(
			"摘要三项与门控区间",
			[
				("abstract.clean_aligned_torque_retained_ratio", "0.961"),
				("abstract.wrong_direction_reduction_absolute", "0.0074"),
				("abstract.clean_rmse_change_vs_paired_baseline", "0.016"),
				("gate_mean_range_safety_rows.min", "0.904"),
				("gate_mean_range_safety_rows.max", "0.978"),
			],
		),
	]

	loso_path = suite / "loso_effect.json"
	if loso_path.exists():
		loso = json.loads(loso_path.read_text())
		print("=== LOSO（loso_effect.json）")
		for key in sorted(loso):
			value = loso[key]
			if isinstance(value, (int, float, str)):
				print(f"  {key:<40} {value}")
		print()

	bad = 0
	for title, keys in groups:
		print(f"=== {title}")
		for key, expected in keys:
			value = get(numbers, key)
			mark = ""
			if value is None:
				mark = "  <== 缺失"
				bad += 1
			print(f"  {key:<62} 稿件={expected:<9} suite={fmt(value)}{mark}")
		print()

	gate_path = Path(args.gate_numbers)
	if gate_path.is_file():
		gate_numbers = json.loads(gate_path.read_text())
		for title, keys in GATE_GROUPS:
			print(f"=== {title}")
			for key, expected in keys:
				value = get(gate_numbers, key)
				mark = ""
				if value is None:
					mark = "  <== 缺失"
					bad += 1
				print(f"  {key:<62} 稿件={expected:<9} suite={fmt(value)}{mark}")
			print()
	else:
		print(f"=== §1–§6 跳过：{gate_path} 不存在（先跑 extract_aaai27_gate_numbers.py）\n")

	print(f"缺失键数量: {bad}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
