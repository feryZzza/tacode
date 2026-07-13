import unittest

from reliability.gate_selection import gate_selection_score


class GateSelectionScoreTest(unittest.TestCase):
	def test_relative_safety_can_outweigh_small_retention_gain(self) -> None:
		safety_oriented = gate_selection_score(
			clean_retained_ratio=0.96,
			mean_fault_wrong_reduction_ratio=0.05,
			mean_fault_retained_ratio=0.85,
			fault_safety_weight=2.0,
			fault_retained_weight=0.25,
			constraints_met=True,
		)
		soft_gate = gate_selection_score(
			clean_retained_ratio=0.98,
			mean_fault_wrong_reduction_ratio=0.01,
			mean_fault_retained_ratio=0.90,
			fault_safety_weight=2.0,
			fault_retained_weight=0.25,
			constraints_met=True,
		)

		self.assertGreater(safety_oriented, soft_gate)

	def test_constraint_violation_dominates_score(self) -> None:
		feasible = gate_selection_score(0.95, 0.0, 0.80, 2.0, 0.25, True)
		infeasible = gate_selection_score(1.00, 1.0, 1.00, 2.0, 0.25, False)

		self.assertGreater(feasible, infeasible)


if __name__ == "__main__":
	unittest.main()
