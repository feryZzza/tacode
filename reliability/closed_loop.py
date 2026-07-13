"""轻量动力学在环代理：在无真机条件下，把估计力矩翻译成潜在用户收益。

设计假设（论文 limitation 中明确声明，仅做相对比较，不做绝对代谢声明）：
- 被动人体近似：关节角速度取自数据集，辅助不改变运动学（一阶近似）。
- 执行器：一阶低通响应 + 力矩饱和，复现真实作动器带宽与上限。
- 收益代理：人体净关节力矩 RMS 与净关节功的下降；左右对称、单关节。

关键 framing：错误的力矩估计会产生错误的指令，经作动器（滞后+饱和）施加后，
作用在**真实**生物力矩 M_bio 上。净人体负担 = M_bio − τ_applied。
一个方向正确的估计会降低净负担；方向错误的辅助会让人"对抗"外骨骼、净负担反升。
这把"估计误差"翻译成"部署安全性与潜在收益"，是 RMSE 之外的部署价值信号。
"""

from __future__ import annotations

from typing import Dict

import torch


def actuator_response(
	command: torch.Tensor,
	alpha: float = 0.25,
	max_torque_nm_per_kg: float = 0.45,
) -> torch.Tensor:
	"""一阶作动器响应 + 力矩饱和。command/return 形状 [B, J, T]。"""
	command = command.clamp(min=-max_torque_nm_per_kg, max=max_torque_nm_per_kg)
	applied = command.clone()
	for t in range(1, command.shape[-1]):
		applied[..., t] = applied[..., t - 1] + alpha * (command[..., t] - applied[..., t - 1])
	return applied


def closed_loop_metrics(
	true_moment: torch.Tensor,
	command_torque: torch.Tensor,
	joint_velocity: torch.Tensor,
	mask: torch.Tensor,
	prefix: str = "cl",
	alpha: float = 0.25,
	max_torque_nm_per_kg: float = 0.45,
	sample_rate: float = 200.0,
) -> Dict[str, float]:
	"""比较未辅助与辅助下的人体净关节负担。

	Args:
		true_moment: 真实生物关节力矩 M_bio，[B, J, T]，单位 Nm/kg。
		command_torque: 外骨骼力矩指令（门控后），[B, J, T]。
		joint_velocity: 关节角速度，[B, J, T]，单位 rad/s，与 moment 关节对齐。
		mask: 有效标签掩码，[B, J, T] 或可广播到该形状。
	"""
	dt = 1.0 / sample_rate
	applied = actuator_response(command_torque, alpha=alpha, max_torque_nm_per_kg=max_torque_nm_per_kg)
	net_moment = true_moment - applied

	valid = _broadcast_mask(mask, true_moment).to(dtype=true_moment.dtype)
	denom = valid.sum().clamp_min(1.0)

	# 净力矩 RMS（与速度符号无关，作为主指标）。
	rms_unassisted = torch.sqrt((true_moment.pow(2) * valid).sum() / denom)
	rms_assisted = torch.sqrt((net_moment.pow(2) * valid).sum() / denom)

	# 关节功率与功：P = moment * velocity。
	power_unassisted = true_moment * joint_velocity
	power_assisted = net_moment * joint_velocity

	# 绝对关节功（符号约定鲁棒）。
	abs_work_unassisted = (power_unassisted.abs() * valid).sum() * dt
	abs_work_assisted = (power_assisted.abs() * valid).sum() * dt
	# 正功（人体做功代理；依赖速度符号，作为次要指标）。
	pos_work_unassisted = (power_unassisted.clamp(min=0.0) * valid).sum() * dt
	pos_work_assisted = (power_assisted.clamp(min=0.0) * valid).sum() * dt

	# 外骨骼"帮倒忙"占比：辅助后净力矩反而比未辅助更大。
	active = valid.bool() & (true_moment.abs() > 0.02)
	worse = active & (net_moment.abs() > true_moment.abs())
	fight_fraction = _safe_ratio(worse.float().sum(), active.float().sum())

	peak_net_assisted = net_moment.abs()[valid.bool()].max() if valid.any() else torch.tensor(0.0)

	return {
		f"{prefix}_net_moment_rms_unassisted": _f(rms_unassisted),
		f"{prefix}_net_moment_rms_assisted": _f(rms_assisted),
		f"{prefix}_net_moment_rms_reduction": _f(rms_unassisted - rms_assisted),
		f"{prefix}_net_moment_rms_reduction_ratio": _safe_ratio(rms_unassisted - rms_assisted, rms_unassisted),
		f"{prefix}_abs_work_reduction": _f(abs_work_unassisted - abs_work_assisted),
		f"{prefix}_abs_work_reduction_ratio": _safe_ratio(abs_work_unassisted - abs_work_assisted, abs_work_unassisted),
		f"{prefix}_pos_work_reduction_ratio": _safe_ratio(pos_work_unassisted - pos_work_assisted, pos_work_unassisted),
		f"{prefix}_fight_fraction": fight_fraction,
		f"{prefix}_peak_net_moment_assisted": _f(peak_net_assisted),
	}


def _broadcast_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
	if mask.shape == reference.shape:
		return mask
	if mask.dim() == reference.dim() and mask.shape[1] == 1:
		return mask.expand_as(reference)
	return mask.reshape(reference.shape[0], -1, reference.shape[-1])[:, :1, :].expand_as(reference)


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
	denom = denominator if torch.is_tensor(denominator) else torch.tensor(float(denominator))
	if float(denom.abs()) < 1e-8:
		return float("nan")
	return _f(numerator / denom)


def _f(value: torch.Tensor) -> float:
	return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
