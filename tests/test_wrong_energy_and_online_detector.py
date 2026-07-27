"""S1/S2 两项审稿修正的回归测试。

S1：E_wrong 能量口径必须区分「一步大反向」与「多步小反向」——比例口径分不出来。
S2：Detector-online 的候选集必须限制在 K_gate 内，coherence/drift 不参与门控，
    所以也不能出现在 online 子集里。
"""

import unittest

import torch

import run_reliability_experiment as R
from reliability.metrics import _torque_metrics_for_command


SAMPLE_RATE = 200.0
K_GATE = ("logit", "aleatoric", "residual", "forecast", "epistemic", "staleness")
ALL_SIGNALS = (*K_GATE, "coherence", "drift")


def metrics(command, ideal):
	cmd = torch.tensor(command, dtype=torch.float32).reshape(1, 1, -1)
	ref = torch.tensor(ideal, dtype=torch.float32).reshape(1, 1, -1)
	valid_time = torch.ones(1, 1, cmd.shape[-1], dtype=torch.bool)
	return _torque_metrics_for_command("x", cmd, ref, valid_time, SAMPLE_RATE)


class WrongEnergyTest(unittest.TestCase):
	def test_matches_closed_form(self):
		# 一步 -0.5 对上 +1.0 的理想力矩：E = 0.5 * 1.0 * (1/200)。
		out = metrics([-0.5, 0.0, 0.3], [1.0, 1.0, 1.0])
		self.assertAlmostEqual(out["x_wrong_energy"], 0.5 / SAMPLE_RATE, places=9)

	def test_only_active_wrong_steps_contribute(self):
		# |ideal| <= 0.02 的步不在 A 里，方向对的步贡献为 0。
		out = metrics([-1.0, 0.5], [0.01, 1.0])
		self.assertEqual(out["x_wrong_energy"], 0.0)

	def test_separates_one_large_reversal_from_many_small_ones(self):
		# 一步 -0.45：比例只有 1/100，但能量 0.45/200 = 2.25e-3。
		one_large = metrics([-0.45] + [0.1] * 99, [1.0] * 100)
		# 二十步 -0.01：比例 20/100（差 20 倍），能量只有 20*0.01/200 = 1.0e-3。
		many_small = metrics([-0.01] * 20 + [0.1] * 80, [1.0] * 100)
		self.assertLess(one_large["x_wrong_direction_ratio"], many_small["x_wrong_direction_ratio"])
		# 两个口径给出相反的排序，这正是加 E_wrong 的理由。
		self.assertGreater(one_large["x_wrong_energy"], many_small["x_wrong_energy"])

	def test_gate_can_only_reduce_energy(self):
		ideal = [1.0] * 8
		ungated = [-0.3, 0.4, -0.2, 0.5, -0.1, 0.6, -0.4, 0.2]
		gated = [value * 0.0 if value < 0 else value for value in ungated]
		self.assertLess(metrics(gated, ideal)["x_wrong_energy"], metrics(ungated, ideal)["x_wrong_energy"])


class OnlineDetectorPoolTest(unittest.TestCase):
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
		# coherence 是最强通道时，Detector-all 会选它，Detector-online 不能。
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
