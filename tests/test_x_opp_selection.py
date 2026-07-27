"""EXPERIMENTS_TODO §1 的选择规则单测：X_opp 目标 + 六项 clean-side 硬约束。"""

import unittest

import torch

from reliability.gate_selection import (
	CLEAN_CONSTRAINT_NAMES,
	evaluate_clean_constraints,
	x_opp_selection_score,
)
from reliability.metrics import gate_dynamics_metrics


UNGATED = {
	"gated_retained_aligned_torque": 1.0,
	"gated_capped_overlap": 1.0,
	"gated_tracking_rmse": 0.10,
	"gated_mean_abs_torque_rate": 2.0,
	"full_shutdown_fraction": 0.0,
	"shutdowns_per_minute": 0.0,
}
LIMITS = dict(
	retained_floor_ratio=0.95,
	overlap_floor_ratio=0.95,
	rmse_ceiling_ratio=1.05,
	max_full_shutdown_fraction=0.02,
	torque_rate_ceiling_ratio=1.10,
	max_shutdowns_per_minute=6.0,
)


def _clean(**overrides):
	base = {
		"gated_retained_aligned_torque": 0.98,
		"gated_capped_overlap": 0.99,
		"gated_tracking_rmse": 0.101,
		"gated_mean_abs_torque_rate": 2.05,
		"full_shutdown_fraction": 0.0,
		"shutdowns_per_minute": 1.0,
	}
	base.update(overrides)
	return base


class XOppSelectionScoreTest(unittest.TestCase):
	def test_amplitude_sensitive_objective_prefers_larger_x_opp_drop(self) -> None:
		strong = x_opp_selection_score(
			fault_x_opp_reduction_ratio=0.60,
			clean_retained_ratio=0.97,
			clean_torque_rate_ratio=1.0,
			fault_retained_weight=0.25,
			mean_fault_retained_ratio=0.85,
			torque_rate_weight=0.5,
			constraints_met=True,
		)
		weak = x_opp_selection_score(
			fault_x_opp_reduction_ratio=0.10,
			clean_retained_ratio=0.98,
			clean_torque_rate_ratio=1.0,
			fault_retained_weight=0.25,
			mean_fault_retained_ratio=0.90,
			torque_rate_weight=0.5,
			constraints_met=True,
		)
		self.assertGreater(strong, weak)

	def test_torque_rate_inflation_is_penalised_not_ignored(self) -> None:
		"""抖着把 X_opp 压下去不该被当成纯收益（§3 的 trade-off 要求）。"""
		smooth = x_opp_selection_score(0.50, 0.97, 1.00, 0.25, 0.85, 0.5, True)
		chattering = x_opp_selection_score(0.55, 0.97, 1.30, 0.25, 0.85, 0.5, True)
		self.assertGreater(smooth, chattering)

	def test_rate_ratio_below_one_is_not_a_bonus(self) -> None:
		neutral = x_opp_selection_score(0.5, 0.97, 1.0, 0.25, 0.85, 0.5, True)
		smoother = x_opp_selection_score(0.5, 0.97, 0.7, 0.25, 0.85, 0.5, True)
		self.assertEqual(neutral, smoother)

	def test_infeasible_candidate_loses_to_any_feasible_one(self) -> None:
		feasible = x_opp_selection_score(0.0, 0.95, 1.0, 0.25, 0.80, 0.5, True)
		infeasible = x_opp_selection_score(1.0, 1.00, 1.0, 0.25, 1.00, 0.5, False)
		self.assertGreater(feasible, infeasible)


class CleanConstraintTest(unittest.TestCase):
	def test_all_six_names_are_reported(self) -> None:
		ok, checks, limits = evaluate_clean_constraints(_clean(), UNGATED, **LIMITS)
		self.assertTrue(ok)
		self.assertEqual(tuple(sorted(checks)), tuple(sorted(CLEAN_CONSTRAINT_NAMES)))
		self.assertEqual(tuple(sorted(limits)), tuple(sorted(CLEAN_CONSTRAINT_NAMES)))

	def test_each_constraint_can_fail_on_its_own(self) -> None:
		cases = {
			"aligned_retention": {"gated_retained_aligned_torque": 0.80},
			"capped_overlap_retention": {"gated_capped_overlap": 0.80},
			"tracking_rmse": {"gated_tracking_rmse": 0.30},
			"full_shutdown_fraction": {"full_shutdown_fraction": 0.10},
			"mean_abs_torque_rate": {"gated_mean_abs_torque_rate": 4.0},
			"shutdowns_per_minute": {"shutdowns_per_minute": 60.0},
		}
		for name, override in cases.items():
			with self.subTest(constraint=name):
				ok, checks, _ = evaluate_clean_constraints(_clean(**override), UNGATED, **LIMITS)
				self.assertFalse(ok)
				self.assertFalse(checks[name])
				others = {k: v for k, v in checks.items() if k != name}
				self.assertTrue(all(others.values()), others)

	def test_missing_or_nan_metric_counts_as_violation(self) -> None:
		missing = _clean()
		missing.pop("shutdowns_per_minute")
		ok, checks, _ = evaluate_clean_constraints(missing, UNGATED, **LIMITS)
		self.assertFalse(ok)
		self.assertFalse(checks["shutdowns_per_minute"])

		nan_clean = _clean(gated_tracking_rmse=float("nan"))
		ok, checks, _ = evaluate_clean_constraints(nan_clean, UNGATED, **LIMITS)
		self.assertFalse(ok)
		self.assertFalse(checks["tracking_rmse"])

	def test_full_shutdown_uses_absolute_limit(self) -> None:
		"""ungated 的 P(g=0) 恒为 0，比值无定义，所以该约束必须是绝对阈。"""
		_, _, limits = evaluate_clean_constraints(_clean(), UNGATED, **LIMITS)
		self.assertEqual(limits["full_shutdown_fraction"], LIMITS["max_full_shutdown_fraction"])
		self.assertEqual(limits["shutdowns_per_minute"], LIMITS["max_shutdowns_per_minute"])


class GateDynamicsMetricsTest(unittest.TestCase):
	def test_chatter_and_single_long_shutdown_differ_at_equal_mean_gate(self) -> None:
		"""平均门控相同、佩戴体验完全不同的两种情形必须被区分开。"""
		n = 400
		valid = torch.ones(1, n, dtype=torch.bool)
		chatter = torch.ones(1, n)
		for start in range(0, n, 20):
			chatter[0, start : start + 2] = 0.0
		single = torch.ones(1, n)
		single[0, 100:140] = 0.0
		self.assertAlmostEqual(float(chatter.mean()), float(single.mean()), places=6)

		a = gate_dynamics_metrics(chatter, valid, sample_rate=200.0)
		b = gate_dynamics_metrics(single, valid, sample_rate=200.0)
		self.assertGreater(a["shutdowns_per_minute"], 10 * b["shutdowns_per_minute"])
		self.assertLess(a["mean_shutdown_duration_ms"], b["mean_shutdown_duration_ms"])

	def test_no_shutdown_reports_zero_rate_and_zero_duration(self) -> None:
		valid = torch.ones(2, 100, dtype=torch.bool)
		out = gate_dynamics_metrics(torch.ones(2, 100), valid, sample_rate=200.0)
		self.assertEqual(out["shutdowns_per_minute"], 0.0)
		self.assertEqual(out["mean_shutdown_duration_ms"], 0.0)
		self.assertEqual(out["p95_shutdown_duration_ms"], 0.0)

	def test_all_invalid_reports_nan(self) -> None:
		valid = torch.zeros(2, 100, dtype=torch.bool)
		out = gate_dynamics_metrics(torch.ones(2, 100), valid, sample_rate=200.0)
		for key, value in out.items():
			self.assertNotEqual(value, value, msg=f"{key} should be NaN when nothing is valid")

	def test_rate_is_per_valid_minute(self) -> None:
		valid = torch.ones(1, 200, dtype=torch.bool)  # 200 样本 @200 Hz = 1 s
		gate = torch.ones(1, 200)
		gate[0, 10:15] = 0.0
		gate[0, 100:105] = 0.0
		out = gate_dynamics_metrics(gate, valid, sample_rate=200.0)
		self.assertAlmostEqual(out["shutdowns_per_minute"], 120.0, places=3)
		self.assertAlmostEqual(out["mean_shutdown_duration_ms"], 25.0, places=3)

	def test_padding_prevents_episodes_bleeding_across_windows(self) -> None:
		"""相邻窗口首尾都是 0 时，不能被拼成一个跨窗口段。"""
		valid = torch.ones(2, 10, dtype=torch.bool)
		gate = torch.ones(2, 10)
		gate[0, 8:] = 0.0
		gate[1, :2] = 0.0
		out = gate_dynamics_metrics(gate, valid, sample_rate=200.0)
		self.assertAlmostEqual(out["mean_shutdown_duration_ms"], 10.0, places=3)


if __name__ == "__main__":
	unittest.main()
