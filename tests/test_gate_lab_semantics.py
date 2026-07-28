"""Regression tests for gate-policy and retention semantics."""

from __future__ import annotations

import unittest

import torch

from reliability.gate_lab import (
	ExposureBasis,
	GatePolicy,
	K_GATE,
	WindowBatch,
	retention_matched_random_attenuation_gate,
)
from reliability.metrics import metrics_for_given_gate


class GatePolicySemanticsTests(unittest.TestCase):
	def test_detector_subset_does_not_replace_actual_gate_channels(self):
		report = {
			"args": {"detector_online_signals": ",".join(K_GATE)},
			"results": {
				"_gate_policy": {"softness": 1.0, "deadband": 0.2},
				"_detector_policy": {"online_signals": "residual,staleness"},
			},
		}
		policy = GatePolicy.from_report(report)
		self.assertEqual(policy.signals, K_GATE)
		self.assertEqual(policy.detector_signals, ("residual", "staleness"))


class RetentionSemanticsTests(unittest.TestCase):
	def _batch(self) -> WindowBatch:
		mean = torch.full((2, 1, 32), 0.5)
		ideal = torch.ones((2, 1, 32))
		mask = torch.ones_like(mean, dtype=torch.bool)
		return WindowBatch(mean=mean, y=ideal, mask=mask)

	def test_metrics_distinguish_adequacy_from_gate_retention(self):
		batch = self._batch()
		gate = torch.full((2, 1, 32), 0.5)
		out = metrics_for_given_gate(gate, batch.mean, batch.y, batch.mask)
		self.assertAlmostEqual(out["ungated_aligned_command_adequacy"], 0.5, places=7)
		self.assertAlmostEqual(out["gated_aligned_command_adequacy"], 0.25, places=7)
		self.assertAlmostEqual(out["gate_retention_ratio"], 0.5, places=7)
		self.assertAlmostEqual(out["ungated_capped_overlap_adequacy"], 0.5, places=7)
		self.assertAlmostEqual(out["gated_capped_overlap_adequacy"], 0.25, places=7)
		self.assertAlmostEqual(out["capped_overlap_retention_ratio"], 0.5, places=7)

	def test_random_attenuation_matches_weighted_retention(self):
		batch = self._batch()
		basis = ExposureBasis.build(batch)
		shape = (2, 1, 32)
		valid = batch.valid_time()
		reference = torch.full(shape, 0.63)
		target = basis.aligned_command_adequacy(reference)
		generator = torch.Generator().manual_seed(7)
		random_gate = retention_matched_random_attenuation_gate(
			shape, valid, basis, target, generator
		)
		self.assertAlmostEqual(basis.aligned_command_adequacy(random_gate), target, places=6)
		self.assertGreater(float(random_gate.std()), 0.0)


if __name__ == "__main__":
	unittest.main()
