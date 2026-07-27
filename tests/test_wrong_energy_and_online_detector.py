"""力矩暴露量与因果检测候选集的回归测试。

X_opp 必须区分「一步大反向」与「多步小反向」；它是力矩乘积时间积分，不是机械能。
Detector-gate 的候选集必须限制在当前门控实现的 K_gate 内，coherence/drift 不参与门控。
"""

import unittest

import torch

import run_reliability_experiment as R
from reliability.metrics import _torque_metrics_for_command, causal_gate_rate_limit


SAMPLE_RATE = 200.0
K_GATE = ("logit", "aleatoric", "residual", "forecast", "epistemic", "staleness")
ALL_SIGNALS = (*K_GATE, "coherence", "drift")


def metrics(command, ideal):
	cmd = torch.tensor(command, dtype=torch.float32).reshape(1, 1, -1)
	ref = torch.tensor(ideal, dtype=torch.float32).reshape(1, 1, -1)
	valid_time = torch.ones(1, 1, cmd.shape[-1], dtype=torch.bool)
	return _torque_metrics_for_command("x", cmd, ref, valid_time, SAMPLE_RATE)


class WrongTorqueProductIntegralTest(unittest.TestCase):
	def test_matches_closed_form(self):
		# 一步 -0.5 对上 +1.0 的理想力矩：X_opp = 0.5 * 1.0 * (1/200)。
		out = metrics([-0.5, 0.0, 0.3], [1.0, 1.0, 1.0])
		self.assertAlmostEqual(out["x_wrong_torque_product_integral"], 0.5 / SAMPLE_RATE, places=9)
		self.assertEqual(out["x_wrong_energy"], out["x_wrong_torque_product_integral"])

	def test_only_active_wrong_steps_contribute(self):
		# |ideal| <= 0.02 的步不在 A 里，方向对的步贡献为 0。
		out = metrics([-1.0, 0.5], [0.01, 1.0])
		self.assertEqual(out["x_wrong_torque_product_integral"], 0.0)

	def test_separates_one_large_reversal_from_many_small_ones(self):
		# 一步 -0.45：比例只有 1/100，但积分为 0.45/200 = 2.25e-3。
		one_large = metrics([-0.45] + [0.1] * 99, [1.0] * 100)
		# 二十步 -0.01：比例 20/100（差 20 倍），积分只有 20*0.01/200 = 1.0e-3。
		many_small = metrics([-0.01] * 20 + [0.1] * 80, [1.0] * 100)
		self.assertLess(one_large["x_wrong_direction_ratio"], many_small["x_wrong_direction_ratio"])
		# 两个口径给出相反的排序，这正是同时报告 X_opp 的理由。
		self.assertGreater(
			one_large["x_wrong_torque_product_integral"],
			many_small["x_wrong_torque_product_integral"],
		)

	def test_gate_cannot_increase_opposition_integral(self):
		ideal = [1.0] * 8
		ungated = [-0.3, 0.4, -0.2, 0.5, -0.1, 0.6, -0.4, 0.2]
		gated = [value * 0.0 if value < 0 else value for value in ungated]
		self.assertLess(
			metrics(gated, ideal)["x_wrong_torque_product_integral"],
			metrics(ungated, ideal)["x_wrong_torque_product_integral"],
		)

	def test_torque_rate_has_backward_compatible_alias(self):
		out = metrics([0.0, 0.5], [1.0, 1.0])
		self.assertEqual(out["x_mean_abs_torque_rate"], out["x_mean_abs_jerk"])

	def test_capped_overlap_does_not_reward_same_direction_overshoot(self):
		out = metrics([2.0, 0.5], [1.0, 1.0])
		self.assertAlmostEqual(out["x_capped_overlap"], 0.75)
		self.assertAlmostEqual(out["x_retained_aligned_torque"], 1.25)
		self.assertAlmostEqual(out["x_over_assistance_fraction"], 0.5)

	def test_nonnegative_gate_cannot_create_wrong_direction(self):
		ideal = [1.0, 1.0, -1.0, -1.0]
		baseline = [0.3, -0.2, -0.4, 0.5]
		gated = [0.0, -0.1, -0.2, 0.0]
		self.assertLessEqual(
			metrics(gated, ideal)["x_wrong_direction_ratio"],
			metrics(baseline, ideal)["x_wrong_direction_ratio"],
		)


class GateRateLimitTest(unittest.TestCase):
	def test_recovery_limit_is_causal_and_direction_specific(self):
		gate = torch.tensor([[[1.0, 0.0, 1.0, 1.0]]])
		out = causal_gate_rate_limit(gate, max_rise_per_step=0.25)
		expected = torch.tensor([[[1.0, 0.0, 0.25, 0.50]]])
		torch.testing.assert_close(out, expected)

	def test_fall_limit_bounds_shutdown_step(self):
		gate = torch.tensor([[[1.0, 0.0, 0.0]]])
		out = causal_gate_rate_limit(gate, max_fall_per_step=0.4)
		expected = torch.tensor([[[1.0, 0.6, 0.2]]])
		torch.testing.assert_close(out, expected)


class GateDetectorPoolTest(unittest.TestCase):
	def _scores(self, strong):
		torch.manual_seed(0)
		clean = {name: torch.rand(256) * 0.5 for name in ALL_SIGNALS}
		faulty = {
			name: torch.rand(256) * 0.5 + (2.0 if name in strong else 0.02)
			for name in ALL_SIGNALS
		}
		refs = {name: 0.45 for name in ALL_SIGNALS}
		return clean, {"fault_a": faulty}, refs

	def test_online_pool_excludes_coherence_and_drift(self):
		# coherence 是最强通道时，Detector-all 会选它，Detector-gate 不能。
		clean, faults, refs = self._scores(strong={"coherence"})
		all_signals, _ = R.select_detector_signal_subset(clean, faults, refs)
		online, info = R.select_detector_signal_subset(clean, faults, refs, candidate_pool=list(K_GATE))
		self.assertIn("coherence", all_signals)
		self.assertNotIn("coherence", online)
		self.assertNotIn("drift", online)
		self.assertEqual(info["candidate_pool"], sorted(K_GATE))

	def test_both_fusions_are_reported(self):
		clean, faults, refs = self._scores(strong={"coherence", "residual"})
		all_signals, _ = R.select_detector_signal_subset(clean, faults, refs)
		online, _ = R.select_detector_signal_subset(clean, faults, refs, candidate_pool=list(K_GATE))
		out = R.detector_aurocs(
			clean, faults["fault_a"], refs, signal_names=all_signals, online_signal_names=online
		)
		self.assertIn("fault_auroc", out)
		self.assertIn("fault_auroc_online", out)

	def test_online_key_absent_when_not_requested(self):
		clean, faults, refs = self._scores(strong={"residual"})
		out = R.detector_aurocs(clean, faults["fault_a"], refs, signal_names=["residual"])
		self.assertNotIn("fault_auroc_online", out)


if __name__ == "__main__":
	unittest.main()
