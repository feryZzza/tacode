"""Validation-only objective helpers for the reliability gate."""


def gate_selection_score(
	clean_retained_ratio: float,
	mean_fault_wrong_reduction_ratio: float,
	mean_fault_retained_ratio: float,
	fault_safety_weight: float,
	fault_retained_weight: float,
	constraints_met: bool,
) -> float:
	"""Score a gate using dimensionless safety and retained-utility terms."""
	penalty = 0.0 if constraints_met else 100.0
	return (
		clean_retained_ratio
		+ fault_safety_weight * mean_fault_wrong_reduction_ratio
		+ fault_retained_weight * mean_fault_retained_ratio
		- penalty
	)
