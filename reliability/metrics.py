"""可靠性估计和离线 torque replay 指标。"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F


def masked_gaussian_nll(
	mean: torch.Tensor,
	logvar: torch.Tensor,
	target: torch.Tensor,
	mask: torch.Tensor,
) -> torch.Tensor:
	"""只在有效标签上计算概率回归损失。"""
	loss = 0.5 * (logvar + (target - mean).pow(2) * torch.exp(-logvar))
	return masked_mean(loss, mask)


def masked_mse(mean: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	return masked_mean((mean - target).pow(2), mask)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	mask = mask.to(dtype=value.dtype)
	denom = mask.sum().clamp_min(1.0)
	return (value * mask).sum() / denom


def regression_metrics(mean: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
	mask_f = mask.to(dtype=mean.dtype)
	valid_count = float(mask_f.sum().detach().cpu())
	if valid_count < 1:
		return {
			"rmse": float("nan"),
			"mae": float("nan"),
			"r2": float("nan"),
			"valid_fraction": 0.0,
		}
	err = (mean - target) * mask_f
	mse = err.pow(2).sum() / mask_f.sum().clamp_min(1.0)
	mae = err.abs().sum() / mask_f.sum().clamp_min(1.0)
	rmse = torch.sqrt(mse)
	target_valid = target * mask_f
	target_mean = target_valid.sum() / mask_f.sum().clamp_min(1.0)
	ss_res = err.pow(2).sum()
	ss_tot = ((target - target_mean).pow(2) * mask_f).sum().clamp_min(1e-8)
	return {
		"rmse": float(rmse.detach().cpu()),
		"mae": float(mae.detach().cpu()),
		"r2": float((1.0 - ss_res / ss_tot).detach().cpu()),
		"valid_fraction": valid_count / float(mask.numel()),
	}


def uncertainty_error_correlation(
	mean: torch.Tensor,
	logvar: torch.Tensor,
	target: torch.Tensor,
	mask: torch.Tensor,
) -> float:
	error = (mean - target).abs()[mask]
	sigma = torch.exp(0.5 * logvar)[mask]
	if error.numel() < 3:
		return float("nan")
	error = error - error.mean()
	sigma = sigma - sigma.mean()
	denom = torch.sqrt((error.pow(2).sum() * sigma.pow(2).sum()).clamp_min(1e-12))
	return float((error * sigma).sum().detach().cpu() / denom.cpu())


def interval_calibration(
	mean: torch.Tensor,
	logvar: torch.Tensor,
	target: torch.Tensor,
	mask: torch.Tensor,
	levels: Iterable[float] = (0.5, 0.68, 0.9, 0.95),
) -> Dict[str, float]:
	"""检查预测区间覆盖率是否接近期望置信度。"""
	z_values = {0.5: 0.674, 0.68: 1.0, 0.9: 1.645, 0.95: 1.96}
	sigma = torch.exp(0.5 * logvar)
	results = {}
	errors = []
	for level in levels:
		z = z_values.get(level)
		if z is None:
			continue
		covered = ((target - mean).abs() <= z * sigma)[mask].float().mean()
		coverage = float(covered.detach().cpu()) if covered.numel() else float("nan")
		results[f"coverage_{int(level * 100)}"] = coverage
		errors.append(abs(coverage - level))
	results["coverage_ece"] = float(sum(errors) / max(len(errors), 1))
	return results


def binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
	scores = scores.detach().flatten().cpu()
	labels = labels.detach().flatten().bool().cpu()
	pos = labels.sum().item()
	neg = labels.numel() - pos
	if pos == 0 or neg == 0:
		return float("nan")
	order = torch.argsort(scores)
	ranks = torch.empty_like(order, dtype=torch.float32)
	ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
	pos_rank_sum = ranks[labels].sum()
	return float(((pos_rank_sum - pos * (pos + 1) / 2) / (pos * neg)).item())


def lowpass(signal: torch.Tensor, sample_rate: float = 200.0, cutoff_hz: float = 6.0) -> torch.Tensor:
	if cutoff_hz <= 0:
		return signal
	dt = 1.0 / sample_rate
	rc = 1.0 / (2.0 * torch.pi * torch.tensor(cutoff_hz, device=signal.device, dtype=signal.dtype))
	alpha = dt / (rc + dt)
	out = signal.clone()
	for t in range(1, signal.shape[-1]):
		out[..., t] = out[..., t - 1] + alpha * (signal[..., t] - out[..., t - 1])
	return out


def moment_to_torque(
	moment: torch.Tensor,
	scales: Tuple[float, float] = (0.20, 0.15),
	delays: Tuple[int, int] = (10, 0),
	clamp_nm_per_kg: float = 0.45,
	sample_rate: float = 200.0,
	cutoff_hz: float = 6.0,
) -> torch.Tensor:
	"""复现 Nature-style scale/delay/low-pass/clamp 映射。"""
	scale = torch.tensor(scales, device=moment.device, dtype=moment.dtype).view(1, -1, 1)
	tau = moment * scale
	for channel, delay in enumerate(delays):
		if delay > 0:
			shifted = torch.roll(tau[:, channel : channel + 1, :], shifts=delay, dims=-1)
			shifted[:, :, :delay] = tau[:, channel : channel + 1, :1]
			tau[:, channel : channel + 1, :] = shifted
	tau = lowpass(tau, sample_rate=sample_rate, cutoff_hz=cutoff_hz)
	return tau.clamp(min=-clamp_nm_per_kg, max=clamp_nm_per_kg)


def reliability_gate(
	logvar: torch.Tensor,
	fault_logit: torch.Tensor,
	uncertainty_ref: float,
	softness: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""高风险时降低辅助增益。"""
	sigma = torch.exp(0.5 * logvar).mean(dim=1, keepdim=True)
	uncertainty_risk = sigma / max(uncertainty_ref, 1e-6)
	fault_risk = torch.sigmoid(fault_logit)
	risk = torch.maximum(uncertainty_risk, fault_risk)
	gate = (1.0 - (risk - 1.0) / max(softness, 1e-6)).clamp(0.0, 1.0)
	return gate, risk


def torque_replay_metrics(
	predicted_moment: torch.Tensor,
	true_moment: torch.Tensor,
	logvar: torch.Tensor,
	fault_logit: torch.Tensor,
	mask: torch.Tensor,
	uncertainty_ref: float,
	sample_rate: float = 200.0,
) -> Dict[str, float]:
	"""离线比较固定辅助与门控辅助的错误风险。"""
	valid_time = mask.all(dim=1, keepdim=True)
	ideal_tau = moment_to_torque(true_moment)
	baseline_tau = moment_to_torque(predicted_moment)
	gate, risk = reliability_gate(logvar, fault_logit, uncertainty_ref)
	gated_tau = baseline_tau * gate
	return {
		**_torque_metrics_for_command("baseline", baseline_tau, ideal_tau, valid_time, sample_rate),
		**_torque_metrics_for_command("gated", gated_tau, ideal_tau, valid_time, sample_rate),
		"mean_gate": float(gate[valid_time].mean().detach().cpu()) if valid_time.any() else float("nan"),
		"mean_risk": float(risk[valid_time].mean().detach().cpu()) if valid_time.any() else float("nan"),
	}


def _torque_metrics_for_command(
	prefix: str,
	command: torch.Tensor,
	ideal: torch.Tensor,
	valid_time: torch.Tensor,
	sample_rate: float,
) -> Dict[str, float]:
	valid = valid_time.expand_as(command)
	ideal_abs = ideal.abs()
	active = valid & (ideal_abs > 0.02)
	wrong = active & (command * ideal < 0.0)
	wrong_values = command.abs()[wrong]
	aligned = torch.clamp(command * torch.sign(ideal), min=0.0)
	retained = aligned[active].sum() / ideal_abs[active].sum().clamp_min(1e-6)
	jerk = torch.diff(command, dim=-1) * sample_rate
	jerk_valid = valid[..., 1:]
	return {
		f"{prefix}_wrong_direction_ratio": _safe_mean(wrong.float(), active),
		f"{prefix}_peak_wrong_torque": float(wrong_values.max().detach().cpu()) if wrong_values.numel() else 0.0,
		f"{prefix}_retained_aligned_torque": float(retained.detach().cpu()) if active.any() else float("nan"),
		f"{prefix}_mean_abs_jerk": float(jerk.abs()[jerk_valid].mean().detach().cpu()) if jerk_valid.any() else float("nan"),
	}


def _safe_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
	if not mask.any():
		return float("nan")
	return float(value[mask].mean().detach().cpu())
