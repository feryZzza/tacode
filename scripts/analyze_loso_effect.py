"""LOSO 主安全效应的被试间置信区间。

主表的 3-seed SD 衡量训练随机性；这里的 15 折 LOSO 衡量被试间变异，两者不可混用。
效应量定义（与 DATA_REQUIREMENTS.md 5.6 一致）：每个 test participant 上固定 fault set 的
macro `ungated_wrong - gated_wrong`，正值表示门控减少了反向力矩。同时报告 clean retention，
防止「全面关闭助力」换来的表面安全提升。
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import statistics
from pathlib import Path

# 六个训练期故障 + 三个未见生成器；缺失的场景跳过而不是补 0。
FAULTS = (
	"insole_missing",
	"encoder_dropout",
	"imu_bias",
	"packet_loss",
	"stuck_imu",
	"sensor_delay",
	"packet_loss_burst",
	"packet_loss_partial",
	"sensor_delay_jitter",
)
SPLITS = ("test_id", "test_ood")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--glob", default="reports/v2_fc_loso_*/reliability_report.json")
	parser.add_argument("--min-folds", type=int, default=10)
	parser.add_argument("--bootstrap", type=int, default=10000)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--output", default="", help="Optional JSON output path.")
	return parser.parse_args()


def bootstrap_ci(values: list[float], draws: int, seed: int) -> tuple[float, float]:
	rng = random.Random(seed)
	means = sorted(
		statistics.mean(rng.choices(values, k=len(values))) for _ in range(draws)
	)
	lo = means[int(0.025 * draws)]
	hi = means[min(int(0.975 * draws), draws - 1)]
	return lo, hi


def main() -> None:
	args = parse_args()
	paths = sorted(Path(p) for p in glob.glob(args.glob))
	if not paths:
		raise SystemExit(f"No reports matched {args.glob}")

	folds: list[dict[str, object]] = []
	# 逐 split×fault 收集跨折的门控前后差值，供 fig_stress_transfer(b) 用。
	per_scenario: dict[str, list[float]] = {}
	per_scenario_retained: dict[str, list[float]] = {}
	for path in paths:
		results = json.loads(path.read_text(encoding="utf-8"))["results"]
		effects = []
		missing = []
		for split in SPLITS:
			for fault in FAULTS:
				key = f"{split}/{fault}"
				metrics = results.get(key)
				if not isinstance(metrics, dict):
					missing.append(key)
					continue
				ungated = metrics.get("baseline_wrong_direction_ratio")
				gated = metrics.get("gated_wrong_direction_ratio")
				if ungated is None or gated is None:
					missing.append(key)
					continue
				effects.append(float(ungated) - float(gated))
				# 图里画的是 gated - ungated（负值更安全），符号与 effect 相反。
				per_scenario.setdefault(key, []).append(float(gated) - float(ungated))
				g = metrics.get("gated_retained_aligned_torque")
				u = metrics.get("baseline_retained_aligned_torque")
				if isinstance(g, (int, float)) and isinstance(u, (int, float)) and u:
					per_scenario_retained.setdefault(key, []).append(float(g) / float(u))
		retentions = []
		for split in SPLITS:
			metrics = results.get(f"{split}/clean")
			if not isinstance(metrics, dict):
				continue
			gated = metrics.get("gated_retained_aligned_torque")
			ungated = metrics.get("baseline_retained_aligned_torque")
			if gated is None or not ungated:
				continue
			retentions.append(float(gated) / float(ungated))
		if not effects or not retentions:
			print(f"{path.parent.name:22s} SKIP (missing clean or all fault scenarios)")
			continue
		folds.append(
			{
				"fold": path.parent.name,
				"n_scenarios": len(effects),
				"effect": statistics.mean(effects),
				"clean_retention": statistics.mean(retentions),
				"missing": missing,
			}
		)
		flag = f"  [missing {len(missing)}]" if missing else ""
		print(
			f"{path.parent.name:22s} effect={folds[-1]['effect']:+.5f} "
			f"clean_ret={folds[-1]['clean_retention']:.3f} n={len(effects)}{flag}"
		)

	effects = [float(f["effect"]) for f in folds]
	if len(effects) < args.min_folds:
		raise SystemExit(
			f"Only {len(effects)} usable fold(s); CI is not credible below {args.min_folds}."
		)
	retentions = [float(f["clean_retention"]) for f in folds]
	lo, hi = bootstrap_ci(effects, args.bootstrap, args.seed)
	improved = sum(1 for value in effects if value > 0)

	summary = {
		"n_folds": len(effects),
		"effect_mean": statistics.mean(effects),
		"effect_median": statistics.median(effects),
		"effect_sd": statistics.stdev(effects) if len(effects) > 1 else 0.0,
		"effect_min": min(effects),
		"effect_max": max(effects),
		"ci95_low": lo,
		"ci95_high": hi,
		"improved_folds": improved,
		"clean_retention_mean": statistics.mean(retentions),
		"clean_retention_min": min(retentions),
		"bootstrap_draws": args.bootstrap,
		"bootstrap_seed": args.seed,
		"folds": folds,
		"per_scenario_wrong_delta": {
			key: {
				"mean": statistics.mean(values),
				"sd": statistics.stdev(values) if len(values) > 1 else 0.0,
				"n_folds": len(values),
				"retained_ratio_mean": (
					statistics.mean(per_scenario_retained[key])
					if per_scenario_retained.get(key)
					else None
				),
			}
			for key, values in sorted(per_scenario.items())
		},
	}

	print()
	print(f"n={summary['n_folds']}  mean={summary['effect_mean']:+.5f}  median={summary['effect_median']:+.5f}  sd={summary['effect_sd']:.5f}")
	print(f"95% bootstrap CI = [{lo:+.5f}, {hi:+.5f}]")
	print(f"range = [{summary['effect_min']:+.5f}, {summary['effect_max']:+.5f}]")
	print(f"improved folds = {improved}/{summary['n_folds']}")
	print(f"clean retention mean = {summary['clean_retention_mean']:.3f} (min {summary['clean_retention_min']:.3f})")
	print()
	print("In percentage points: mean="
		f"{summary['effect_mean'] * 100:+.3f} pp, CI=[{lo * 100:+.3f}, {hi * 100:+.3f}] pp")

	if args.output:
		Path(args.output).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
		print(f"\nWrote {args.output}")


if __name__ == "__main__":
	main()
