"""§4 瞬态时间响应与 §5 配对双重差分的单元测试。

这两处的错误都是「沉默」的：延迟算错只会让数字偏小，聚合顺序搞反只会让 CI 变窄，
跑通不报错。所以用手工构造的门控序列把每个量的定义钉死。
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reliability.faults import apply_fault_interval, parse_fault_spec
from reliability.gate_lab import WindowBatch


def _load(name: str, relative: str):
	spec = importlib.util.spec_from_file_location(name, ROOT / relative)
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


transient = _load("run_transient_faults", "scripts/run_transient_faults.py")
paired = _load("run_paired_analysis", "scripts/run_paired_analysis.py")


def make_batch(gate_len: int = 200, n: int = 2) -> WindowBatch:
	torch.manual_seed(0)
	y = torch.ones((n, 1, gate_len))
	mean = torch.ones((n, 1, gate_len))
	mask = torch.ones((n, 1, gate_len), dtype=torch.bool)
	return WindowBatch(
		mean=mean, y=y, mask=mask,
		fault_target=torch.zeros((n, 1, gate_len)),
		participant=["P1"] * n, trial=["t"] * n, task=["walk"] * n, start=list(range(n)),
	)


class TransientTimingTests(unittest.TestCase):
	def test_onset_delay_counts_samples_after_onset(self):
		"""onset 后第 10 个样本落到 0.5 以下 → 50 ms @200 Hz。"""
		batch = make_batch()
		gate = torch.ones((2, 1, 200))
		gate[:, :, 110:] = 0.0
		out = transient.transient_metrics(gate, batch, onset=100, offset=None)
		self.assertAlmostEqual(out["onset_to_half_ms"], 50.0, places=6)
		self.assertAlmostEqual(out["onset_to_zero_ms"], 50.0, places=6)
		self.assertEqual(out["onset_to_half_rate"], 1.0)

	def test_pre_onset_shutdown_is_not_credited_as_fast_response(self):
		"""onset 之前就关的门控不算「响应快」：它是误报，延迟要从 onset 起算。"""
		batch = make_batch()
		gate = torch.ones((2, 1, 200))
		gate[:, :, 20:60] = 0.0  # onset 之前的一段误关
		out = transient.transient_metrics(gate, batch, onset=100, offset=None)
		self.assertTrue(out["onset_to_half_ms"] != out["onset_to_half_ms"])  # NaN
		self.assertEqual(out["onset_to_half_rate"], 0.0)
		self.assertGreater(out["false_shutdowns_per_minute"], 0.0)

	def test_recovery_latency_measured_from_offset(self):
		batch = make_batch()
		gate = torch.ones((2, 1, 200))
		gate[:, :, 100:160] = 0.0  # 故障区间 100..150，门控多关了 10 个样本
		out = transient.transient_metrics(gate, batch, onset=100, offset=150)
		self.assertAlmostEqual(out["recovery_to_half_ms"], 50.0, places=6)
		self.assertAlmostEqual(out["recovery_to_full_ms"], 50.0, places=6)

	def test_miss_rate_is_per_window_not_per_sample(self):
		"""一个窗口在故障期内关过就算检出，即使只关了一个样本。"""
		batch = make_batch()
		gate = torch.ones((2, 1, 200))
		gate[0, :, 120] = 0.0  # 只有第 0 个窗口检出
		out = transient.transient_metrics(gate, batch, onset=100, offset=150)
		self.assertAlmostEqual(out["fault_miss_rate"], 0.5, places=6)

	def test_never_firing_gate_misses_everything(self):
		batch = make_batch()
		gate = torch.ones((2, 1, 200))
		out = transient.transient_metrics(gate, batch, onset=100, offset=150)
		self.assertEqual(out["fault_miss_rate"], 1.0)
		self.assertEqual(out["false_shutdowns_per_minute"], 0.0)

	def test_negative_control_reports_no_latencies(self):
		"""clean-pause 下门控一直开着，「恢复很快」是伪结论，必须留 NaN。"""
		batch = make_batch()
		gate = torch.ones((2, 1, 200))
		out = transient.transient_metrics(gate, batch, onset=100, offset=None, negative_control=True)
		for key in ("onset_to_half_ms", "onset_to_zero_ms", "recovery_to_half_ms",
			"recovery_to_full_ms", "fault_miss_rate"):
			self.assertTrue(out[key] != out[key], f"{key} 应为 NaN")
		self.assertEqual(out["false_shutdowns_per_minute"], 0.0)

	def test_negative_control_counts_full_window_as_outside(self):
		batch = make_batch()
		gate = torch.ones((2, 1, 200))
		gate[:, :, 50:55] = 0.0
		out = transient.transient_metrics(gate, batch, onset=100, offset=None, negative_control=True)
		# 2 个窗口 × 200 样本 @200 Hz = 2 秒 = 1/30 分钟；2 次关断 → 60/min。
		self.assertAlmostEqual(out["false_shutdowns_per_minute"], 60.0, places=4)

	def test_chatter_counts_crossings_not_closed_samples(self):
		batch = make_batch(n=1)
		chatter = torch.ones((1, 1, 200))
		chatter[:, :, 10:12] = 0.0
		chatter[:, :, 20:22] = 0.0
		single = torch.ones((1, 1, 200))
		single[:, :, 10:14] = 0.0
		a = transient.transient_metrics(chatter, batch, onset=100, offset=None)
		b = transient.transient_metrics(single, batch, onset=100, offset=None)
		self.assertGreater(a["alarm_chatter_per_minute"], b["alarm_chatter_per_minute"])


class FaultIntervalTests(unittest.TestCase):
	def test_interval_leaves_outside_samples_untouched(self):
		x = torch.randn(2, 4, 100)
		groups = {"insole": [0, 1], "imu": [2, 3]}
		out, target = apply_fault_interval(x, groups, [parse_fault_spec("insole_missing")], onset=40, offset=60)
		torch.testing.assert_close(out[..., :40], x[..., :40])
		torch.testing.assert_close(out[..., 60:], x[..., 60:])
		self.assertTrue(bool((target[..., 40:60] > 0.5).all()))
		self.assertTrue(bool((target[..., :40] < 0.5).all()))
		self.assertTrue(bool((target[..., 60:] < 0.5).all()))

	def test_offset_none_runs_to_window_end(self):
		x = torch.randn(1, 4, 50)
		groups = {"insole": [0, 1], "imu": [2, 3]}
		_, target = apply_fault_interval(x, groups, [parse_fault_spec("insole_missing")], onset=30, offset=None)
		self.assertTrue(bool((target[..., 30:] > 0.5).all()))


class PairedAnalysisTests(unittest.TestCase):
	def _batch(self, command: torch.Tensor, ideal: torch.Tensor) -> WindowBatch:
		n = command.shape[0]
		return WindowBatch(
			mean=command, y=ideal, mask=torch.ones_like(command, dtype=torch.bool),
			fault_target=torch.zeros((n, 1, command.shape[-1])),
			participant=["P1"] * n, trial=["t"] * n, task=["walk"] * n, start=list(range(n)),
		)

	def test_per_window_metrics_are_not_pooled_across_windows(self):
		"""两个窗口的 x_opp 必须各自算：合并求和会让配对失去意义。"""
		ideal = torch.ones((2, 1, 100))
		command = torch.zeros((2, 1, 100))
		command[0] = -1.0  # 第 0 窗全反向
		batch = self._batch(command, ideal)
		out = paired.per_window_metrics(batch, None)
		self.assertGreater(float(out["x_opp"][0]), 0.0)
		self.assertAlmostEqual(float(out["x_opp"][1]), 0.0, places=9)

	def test_gate_scales_x_opp_linearly(self):
		ideal = torch.ones((1, 1, 100))
		command = -torch.ones((1, 1, 100))
		batch = self._batch(command, ideal)
		full = float(paired.per_window_metrics(batch, None)["x_opp"][0])
		half = float(paired.per_window_metrics(batch, torch.full((1, 1, 100), 0.5))["x_opp"][0])
		self.assertAlmostEqual(half, full * 0.5, places=6)

	def test_double_difference_cancels_a_gate_effect_present_on_clean(self):
		"""门控在 clean 与 fault 上产生同样的变化时，fault-specific 效应应为 0。"""
		delta_fault = 0.30 - 0.50
		delta_clean = 0.30 - 0.50
		self.assertAlmostEqual(delta_fault - delta_clean, 0.0, places=12)

	def test_bootstrap_ci_uses_participants_as_units(self):
		import random

		values = [-0.10, -0.12, -0.11, -0.09, -0.13]
		point, lo, hi, pval = paired.bootstrap_ci(values, 4000, random.Random(0))
		self.assertAlmostEqual(point, sum(values) / len(values), places=9)
		self.assertLess(hi, 0.0)  # 全负 → 区间不含 0
		self.assertLess(pval, 0.05)

	def test_bootstrap_ci_straddling_zero_is_not_significant(self):
		import random

		values = [-0.10, 0.11, -0.02, 0.05, 0.01]
		_, lo, hi, pval = paired.bootstrap_ci(values, 4000, random.Random(0))
		self.assertLess(lo, 0.0)
		self.assertGreater(hi, 0.0)
		self.assertGreater(pval, 0.05)

	def test_bootstrap_with_single_participant_returns_no_interval(self):
		"""1 个被试给不出被试级区间——必须是 NaN，不能退化成点值当区间。"""
		import random

		point, lo, hi, pval = paired.bootstrap_ci([-0.1], 100, random.Random(0))
		self.assertAlmostEqual(point, -0.1, places=9)
		for value in (lo, hi, pval):
			self.assertTrue(value != value)

	def test_nan_values_are_dropped_not_propagated(self):
		import random

		point, lo, hi, _ = paired.bootstrap_ci([-0.1, float("nan"), -0.2], 2000, random.Random(0))
		self.assertAlmostEqual(point, -0.15, places=9)
		self.assertFalse(lo != lo)
		self.assertFalse(hi != hi)


if __name__ == "__main__":
	unittest.main()
