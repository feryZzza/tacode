"""Validation-only objective helpers for the reliability gate."""

from typing import Dict, Optional, Tuple

# 六项 clean-side 约束的名字，与 EXPERIMENTS_TODO §1.1 列表一一对应。
# 顺序固定，用于报告里稳定的列名。
CLEAN_CONSTRAINT_NAMES: Tuple[str, ...] = (
	"aligned_retention",
	"capped_overlap_retention",
	"tracking_rmse",
	"full_shutdown_fraction",
	"mean_abs_torque_rate",
	"shutdowns_per_minute",
)


def x_opp_selection_score(
	fault_x_opp_reduction_ratio: float,
	clean_retained_ratio: float,
	clean_torque_rate_ratio: float,
	fault_retained_weight: float,
	mean_fault_retained_ratio: float,
	torque_rate_weight: float,
	constraints_met: bool,
	infeasible_penalty: float = 100.0,
) -> float:
	"""按 $X_{\\mathrm{opp}}$ 打分的门控选择目标（EXPERIMENTS_TODO §1.1）。

	主目标是验证故障上 $X_{\\mathrm{opp}}$ 相对 ungated 的下降比例——幅值敏感，
	因此「一步 0.45 Nm/kg 的反向」不会和「一百步 0.01 的反向」记同样的分。
	wrong-direction fraction 不再进目标也不进约束：对非负门控恒有 $w(gu)\\le w(u)$，
	把它当安全约束等于什么都没约束（该恒等式的证明见论文 §Metrics）。它只留作
	hard-shutdown 诊断量。

	clean 侧的六项要求走硬约束（`constraints_met`），不折进标量目标——否则大权重的
    安全项能把 clean 侧的退化「买」回来。这里只保留两个 clean 侧的软偏好：
	retention 越高越好，torque rate 相对 ungated 的放大越小越好。torque rate 项是
	为了不把「更安全」和「更抖」混为一谈：门控降 $X_{\\mathrm{opp}}$ 但抬高 torque
	rate 时，它会在目标里显式扣分，而不是被当成纯收益。
	"""
	penalty = 0.0 if constraints_met else infeasible_penalty
	return (
		fault_x_opp_reduction_ratio
		+ clean_retained_ratio
		+ fault_retained_weight * mean_fault_retained_ratio
		- torque_rate_weight * max(clean_torque_rate_ratio - 1.0, 0.0)
		- penalty
	)


def evaluate_clean_constraints(
	clean: Dict[str, float],
	ungated: Dict[str, float],
	retained_floor_ratio: float,
	overlap_floor_ratio: float,
	rmse_ceiling_ratio: float,
	max_full_shutdown_fraction: float,
	torque_rate_ceiling_ratio: float,
	max_shutdowns_per_minute: float,
) -> Tuple[bool, Dict[str, bool], Dict[str, float]]:
	"""检查 §1.1 的六项 clean-side 约束，返回 (全部通过, 逐项通过, 逐项阈值)。

	四项相对 ungated 定阈（retention / overlap / RMSE / torque rate），两项绝对定阈
	（full-shutdown fraction / shutdowns per minute），因为后两者的「可接受量」由佩戴
	体验决定，与 ungated 基线无关——ungated 的 $P(g{=}0)$ 恒为 0，比值无定义。

	缺失或 NaN 的量按**不通过**处理：选择器宁可判一个工作点不可行，也不能把读不到的
	约束当成满足。
	"""

	def _ok(value: Optional[float], limit: float, upper: bool) -> bool:
		if value is None or value != value or limit != limit:
			return False
		return value <= limit if upper else value >= limit

	def _base(key: str) -> float:
		value = ungated.get(key)
		return float("nan") if value is None else value

	limits = {
		"aligned_retention": _base("gated_retained_aligned_torque") * max(retained_floor_ratio, 0.0),
		"capped_overlap_retention": _base("gated_capped_overlap") * max(overlap_floor_ratio, 0.0),
		"tracking_rmse": _base("gated_tracking_rmse") * max(rmse_ceiling_ratio, 0.0),
		"full_shutdown_fraction": max_full_shutdown_fraction,
		"mean_abs_torque_rate": _base("gated_mean_abs_torque_rate") * max(torque_rate_ceiling_ratio, 0.0),
		"shutdowns_per_minute": max_shutdowns_per_minute,
	}
	checks = {
		"aligned_retention": _ok(clean.get("gated_retained_aligned_torque"), limits["aligned_retention"], upper=False),
		"capped_overlap_retention": _ok(clean.get("gated_capped_overlap"), limits["capped_overlap_retention"], upper=False),
		"tracking_rmse": _ok(clean.get("gated_tracking_rmse"), limits["tracking_rmse"], upper=True),
		"full_shutdown_fraction": _ok(clean.get("full_shutdown_fraction"), limits["full_shutdown_fraction"], upper=True),
		"mean_abs_torque_rate": _ok(clean.get("gated_mean_abs_torque_rate"), limits["mean_abs_torque_rate"], upper=True),
		"shutdowns_per_minute": _ok(clean.get("shutdowns_per_minute"), limits["shutdowns_per_minute"], upper=True),
	}
	return all(checks.values()), checks, limits


def gate_selection_score(
	clean_retained_ratio: float,
	mean_fault_wrong_reduction_ratio: float,
	mean_fault_retained_ratio: float,
	fault_safety_weight: float,
	fault_retained_weight: float,
	constraints_met: bool,
) -> float:
	"""Legacy v2 objective, kept so archived operating points remain recomputable.

	Superseded by :func:`x_opp_selection_score`; new runs must not use it as the
	primary objective because its safety term is a wrong-direction *fraction*,
	which a nonnegative gate can never increase.
	"""
	penalty = 0.0 if constraints_met else 100.0
	return (
		clean_retained_ratio
		+ fault_safety_weight * mean_fault_wrong_reduction_ratio
		+ fault_retained_weight * mean_fault_retained_ratio
		- penalty
	)
