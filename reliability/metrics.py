"""可靠性估计和离线 torque replay 指标。"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F


def reconstruction_residual(
	recon: torch.Tensor,
	input_norm: torch.Tensor,
	feature_groups: Optional[Dict[str, Iterable[int]]] = None,
) -> torch.Tensor:
	"""输入完整性残差：归一化空间中重构与观测之差的能量，[B,1,T]。

	packet_loss/sensor_delay/imu_bias 对单帧统计几乎无损，但破坏了输入的
	时序可预测性，因此在重构残差域可分。可选按模态聚合以增强对单组故障的敏感度。
	"""
	per_channel = (recon - input_norm).pow(2)
	if feature_groups:
		group_scores = []
		for indices in feature_groups.values():
			idx = list(indices)
			if idx:
				group_scores.append(per_channel[:, idx, :].mean(dim=1, keepdim=True))
		if group_scores:
			# 取最敏感模态（max）以放大单组失效。
			return torch.cat(group_scores, dim=1).max(dim=1, keepdim=True).values
	return per_channel.mean(dim=1, keepdim=True)


def forecast_residual(
	forecast: torch.Tensor,
	input_norm: torch.Tensor,
	horizon: int = 1,
	feature_groups: Optional[Dict[str, Iterable[int]]] = None,
) -> torch.Tensor:
	"""一步(或 h 步)预测残差，在目标到达时刻 t+h 发布，[B,1,T]。

	因果 TCN 下同时刻重构平凡，无法察觉 packet_loss(前值保持)/sensor_delay
	(相位错位)/imu_bias(累积漂移)——这些故障值仍合理，只破坏时间可预测性。
	一步预测在干净输入上学真实动态，故障会使预测残差升高，与 reconstruction_residual
	形成互补检测信号。

	因果性：`forecast[t]` 预测 `input_norm[t+h]`，该残差只有等到 `t+h` 观测到达后
	才可计算，因此写在时间轴的 `t+h` 位置。在线门控在时刻 u 只会读到由
	`forecast[u-h]` 与 `input_norm[u]` 构成的残差，不含任何未来信息。开头 h 帧
	没有对应预测(需要 `forecast[-h]`)，置 0；`eval_ignore_history` 远大于 h，
	这些帧本就被掩掉。
	"""
	horizon = max(int(horizon), 1)
	if forecast.shape[-1] <= horizon:
		return torch.zeros(forecast.shape[0], 1, forecast.shape[-1], device=forecast.device, dtype=forecast.dtype)
	pred = forecast[..., :-horizon]
	target = input_norm[..., horizon:]
	per_channel = (pred - target).pow(2)
	if feature_groups:
		group_scores = []
		for indices in feature_groups.values():
			idx = list(indices)
			if idx:
				group_scores.append(per_channel[:, idx, :].mean(dim=1, keepdim=True))
		if group_scores:
			residual = torch.cat(group_scores, dim=1).max(dim=1, keepdim=True).values
		else:
			residual = per_channel.mean(dim=1, keepdim=True)
	else:
		residual = per_channel.mean(dim=1, keepdim=True)
	# 开头 horizon 帧无可用预测：置 0 右移，使残差在其目标到达的时刻发布。
	pad = torch.zeros(
		residual.shape[0], residual.shape[1], horizon,
		device=residual.device, dtype=residual.dtype,
	)
	return torch.cat([pad, residual], dim=-1)


def staleness_score(
	input_norm: torch.Tensor,
	feature_groups: Optional[Dict[str, Iterable[int]]] = None,
	eps: float = 1e-3,
	window: int = 25,
) -> torch.Tensor:
	"""丢包检测：局部窗口内冻结通道的占比，[B,1,T]。

	前值保持会令受影响通道满足 z[t]==z[t-1]。逐通道统计而不是要求所有通道同时冻结，
	因此 full/partial packet loss 都可见。只在最近 window 帧做因果平均，避免 delay 仿真在
	窗口开头的首值填充被累计到整段，从而产生跨过 ignore-history 的边界伪信号。
	"""
	if input_norm.shape[-1] < 2:
		return torch.zeros(input_norm.shape[0], 1, input_norm.shape[-1], device=input_norm.device, dtype=input_norm.dtype)
	diff = (input_norm[..., 1:] - input_norm[..., :-1]).abs()
	if feature_groups:
		idx = [i for indices in feature_groups.values() for i in indices]
		if idx:
			diff = diff[:, idx, :]
	# 每帧冻结通道比例：整包丢失接近 1，部分通道独立丢失接近其丢包率。
	frozen = (diff < eps).to(input_norm.dtype).mean(dim=1, keepdim=True)
	# 首帧无前值，记 0；随后做固定长度的因果局部平均。
	frozen = torch.cat([torch.zeros_like(frozen[..., :1]), frozen], dim=-1)
	window = max(int(window), 1)
	padded = F.pad(frozen, (window - 1, 0))
	return F.avg_pool1d(padded, kernel_size=window, stride=1)


def cross_modal_coherence_score(
	input_norm: torch.Tensor,
	feature_groups: Optional[Dict[str, Iterable[int]]] = None,
	window: int = 32,
) -> torch.Tensor:
	"""Detect asynchronous IMU channels with causal rolling coherence loss.

	Corresponding foot/shank/thigh IMU axes are differentiated, then compared
	with a rolling zero-lag Pearson correlation. Sensor-delay faults shift each
	modality by a different amount, so their derivative coherence drops. Every
	output at time ``t`` uses only samples at or before ``t``.
	"""
	batch_size, _, time_steps = input_norm.shape
	if time_steps < 2 or not feature_groups:
		return torch.zeros(batch_size, 1, time_steps, device=input_norm.device, dtype=input_norm.dtype)
	pair_names = (("foot_imu", "shank_imu"), ("foot_imu", "thigh_imu"), ("shank_imu", "thigh_imu"))
	left_parts = []
	right_parts = []
	for left_name, right_name in pair_names:
		left_indices = list(feature_groups.get(left_name, []))
		right_indices = list(feature_groups.get(right_name, []))
		if left_indices and len(left_indices) == len(right_indices):
			left_parts.append(input_norm[:, left_indices, :])
			right_parts.append(input_norm[:, right_indices, :])
	if not left_parts:
		return torch.zeros(batch_size, 1, time_steps, device=input_norm.device, dtype=input_norm.dtype)

	left = torch.cat(left_parts, dim=1)
	right = torch.cat(right_parts, dim=1)
	left = torch.cat([torch.zeros_like(left[..., :1]), left[..., 1:] - left[..., :-1]], dim=-1)
	right = torch.cat([torch.zeros_like(right[..., :1]), right[..., 1:] - right[..., :-1]], dim=-1)
	window = min(max(int(window), 2), time_steps)

	def rolling_sum(values: torch.Tensor) -> torch.Tensor:
		cumulative = F.pad(values.cumsum(dim=-1), (1, 0))
		end = torch.arange(1, time_steps + 1, device=values.device)
		start = (end - window).clamp_min(0)
		return cumulative.index_select(-1, end) - cumulative.index_select(-1, start)

	count = torch.arange(1, time_steps + 1, device=input_norm.device, dtype=input_norm.dtype).clamp_max(window)
	count = count.view(1, 1, -1)
	left_mean = rolling_sum(left) / count
	right_mean = rolling_sum(right) / count
	covariance = rolling_sum(left * right) / count - left_mean * right_mean
	left_variance = (rolling_sum(left.square()) / count - left_mean.square()).clamp_min(1e-8)
	right_variance = (rolling_sum(right.square()) / count - right_mean.square()).clamp_min(1e-8)
	correlation = covariance / (left_variance * right_variance).sqrt()
	score = 1.0 - correlation.abs().clamp_max(1.0).mean(dim=1, keepdim=True)
	score[..., : window - 1] = 0.0
	return score


def drift_score(
	input_norm: torch.Tensor,
	feature_groups: Optional[Dict[str, Iterable[int]]] = None,
) -> torch.Tensor:
	"""慢偏置检测：每通道线性趋势(斜率)幅度，[B,1,T]。

	imu_bias 是单模态上的线性慢漂移(bias_drift = severity·std·ramp)，量级在归一化 std 内、
	且平滑，故 forecast/recon 这类差分式信号天然失明——逐帧动态没变，只是整体被缓慢线性抬升。
	对"叠加线性 ramp"最匹配的统计量是窗口内对时间轴的最小二乘斜率 cov(z,t)/var(t)：漂移让斜率
	系统性偏离 0，而干净步态信号去趋势后斜率接近 0。按模态分组取通道均值再跨组取最大，以在
	不知道哪一组失效的前提下放大单组漂移。窗口级标量沿时间广播成 [B,1,T] 与其它通道对齐。
	"""
	T = input_norm.shape[-1]
	if T < 4:
		return torch.zeros(input_norm.shape[0], 1, T, device=input_norm.device, dtype=input_norm.dtype)
	t = torch.arange(T, device=input_norm.device, dtype=input_norm.dtype)
	t = t - t.mean()
	denom = t.pow(2).mean().clamp_min(1e-6)
	slope = (input_norm * t.view(1, 1, -1)).mean(dim=-1) / denom  # [B, C] 最小二乘斜率
	slope = slope.abs()
	if feature_groups:
		group_scores = []
		for indices in feature_groups.values():
			idx = list(indices)
			if idx:
				group_scores.append(slope[:, idx].mean(dim=1, keepdim=True))
		if group_scores:
			window_score = torch.cat(group_scores, dim=1).max(dim=1, keepdim=True).values
		else:
			window_score = slope.mean(dim=1, keepdim=True)
	else:
		window_score = slope.mean(dim=1, keepdim=True)
	return window_score.unsqueeze(-1).expand(window_score.shape[0], 1, T)


def calibrate_risk(
	aleatoric_sigma: torch.Tensor,
	fault_prob: torch.Tensor,
	residual: Optional[torch.Tensor] = None,
	epistemic_sigma: Optional[torch.Tensor] = None,
	forecast: Optional[torch.Tensor] = None,
	staleness: Optional[torch.Tensor] = None,
	drift: Optional[torch.Tensor] = None,
	refs: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
	"""把多路可靠性信号融合成 [0,∞) 的标定风险，1.0 为名义阈值。

	每路信号除以其 validation 参考分位（refs，由训练后在干净 val 上估计），
	再取最大值。这样任一信号显著超出 clean 分布即触发风险，且阈值由 val 决定。
	所有输入张量在时间维上压成 [B,1,T] 或 [B,T]。staleness/drift 是显式时序统计通道，
	专抓 packet_loss/sensor_delay（冻结）与 imu_bias（漂移）——这些故障使预测/重构残差失明，
	不进融合则门控对它们视而不见。
	"""
	refs = refs or {}
	risks = [fault_prob / max(refs.get("fault", 0.5), 1e-6)]
	risks.append(_reduce_joint(aleatoric_sigma) / max(refs.get("aleatoric", 1.0), 1e-6))
	if residual is not None:
		risks.append(_reduce_joint(residual) / max(refs.get("residual", 1.0), 1e-6))
	if epistemic_sigma is not None:
		risks.append(_reduce_joint(epistemic_sigma) / max(refs.get("epistemic", 1.0), 1e-6))
	if forecast is not None:
		risks.append(_reduce_joint(forecast) / max(refs.get("forecast", 1.0), 1e-6))
	if staleness is not None:
		risks.append(_reduce_joint(staleness) / max(refs.get("staleness", 1.0), 1e-6))
	if drift is not None:
		risks.append(_reduce_joint(drift) / max(refs.get("drift", 1.0), 1e-6))
	risk = risks[0]
	for other in risks[1:]:
		risk = torch.maximum(risk, other)
	return risk


#: `calibrate_risk` 里每路信号的默认参考值（refs 缺失时用）。fault 是概率，名义阈值
#: 0.5；其余通道都已按 val 分位归一，参考值 1.0。
DEFAULT_RISK_REFS: Dict[str, float] = {
	"fault": 0.5,
	"aleatoric": 1.0,
	"residual": 1.0,
	"epistemic": 1.0,
	"forecast": 1.0,
	"staleness": 1.0,
	"drift": 1.0,
}


def risk_from_channels(
	channels: Dict[str, torch.Tensor],
	refs: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
	"""任意通道子集的标定风险 $\\max_k s_k / r_k$（EXPERIMENTS_TODO §2 的单通道门控）。

	`calibrate_risk` 永远包含 fault 与 aleatoric 两路，所以没法用来构造「只用重构残差」
	或「只用 staleness」这类单通道门控。这里接受显式的通道字典，融合规则与
	`calibrate_risk` 完全一致（同样的 refs、同样的关节维求均值、同样取最大值），
	区别只在候选集——这正是消融要隔离的变量。
	"""
	if not channels:
		raise ValueError("risk_from_channels needs at least one channel")
	refs = refs or {}
	risk: Optional[torch.Tensor] = None
	for name, value in channels.items():
		ref = refs.get(name, DEFAULT_RISK_REFS.get(name, 1.0))
		scaled = _reduce_joint(value) / max(ref, 1e-6)
		risk = scaled if risk is None else torch.maximum(risk, scaled)
	return risk


def _reduce_joint(value: torch.Tensor) -> torch.Tensor:
	"""把可能的多关节信号 [B,J,T] 在关节维取均值成 [B,1,T]。"""
	if value.dim() >= 2 and value.shape[1] > 1:
		return value.mean(dim=1, keepdim=True)
	return value


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
	risk: torch.Tensor,
	softness: float = 0.5,
	deadband: float = 0.0,
) -> torch.Tensor:
	"""把标定风险映射成辅助增益 ∈ [0,1]。

	deadband 之内（risk ≤ 1+deadband）门控保持 1，不削弱 clean 条件下的有用辅助；
	超出后随 risk 线性降级，softness 控制降级速率。
	"""
	threshold = 1.0 + max(deadband, 0.0)
	return (1.0 - (risk - threshold) / max(softness, 1e-6)).clamp(0.0, 1.0)


def causal_gate_rate_limit(
	gate: torch.Tensor,
	max_fall_per_step: float = 0.0,
	max_rise_per_step: float = 0.0,
) -> torch.Tensor:
	"""Causally limit gate changes; nonpositive limits leave that direction free.

	A fast fall can be retained for fault response while recovery is slowed by
	setting only ``max_rise_per_step``.  The operation uses no future sample.
	"""
	if gate.shape[-1] < 2 or (max_fall_per_step <= 0 and max_rise_per_step <= 0):
		return gate
	out = gate.clone()
	for t in range(1, gate.shape[-1]):
		lower = out[..., t - 1] - max(max_fall_per_step, 0.0)
		upper = out[..., t - 1] + max(max_rise_per_step, 0.0)
		current = gate[..., t]
		if max_fall_per_step > 0:
			current = torch.maximum(current, lower)
		if max_rise_per_step > 0:
			current = torch.minimum(current, upper)
		out[..., t] = current.clamp(0.0, 1.0)
	return out


def gate_dynamics_metrics(
	gate: torch.Tensor,
	valid_time: torch.Tensor,
	sample_rate: float = 200.0,
	prefix: str = "",
	zero_tol: float = 1e-6,
) -> Dict[str, float]:
	"""关闭事件的时间结构：频率、持续时长分布与开关抖动。

	平均门控值无法区分「一次 200 ms 的关闭」和「二十次 10 ms 的抖动」，但佩戴者感受到的
	是后者。这里把 gate 序列切成 episode 来度量：

	- shutdown episode：`g <= zero_tol` 的极大连续段（完全断力矩）；
	- attenuation episode：`g < 0.5` 的极大连续段（力矩被砍半以上）；
	- transition：跨过 0.5 的次数，作为 alarm chatter 的代理量。

	每个窗口独立计段。跨窗口边界的 episode 会被截断，因此这里的持续时长是**窗口内**
	时长的下界；窗口 768 sample @200 Hz = 3.84 s，远长于关心的 25–500 ms 故障尺度，
	截断只影响极少数贴边事件。频率类指标（per-minute）不受截断影响，因为分母用的是
	同一批有效样本的总时长。
	"""
	if gate.dim() == 3 and gate.shape[1] > 1:
		gate = gate.mean(dim=1, keepdim=True)
	if valid_time.dim() == 3 and valid_time.shape[1] > 1:
		valid_time = valid_time.all(dim=1, keepdim=True)
	gate = gate.reshape(gate.shape[0], -1)
	valid = valid_time.reshape(valid_time.shape[0], -1).to(dtype=torch.bool)
	if valid.shape != gate.shape:
		valid = valid.expand_as(gate)
	keys = (
		"shutdowns_per_minute",
		"mean_shutdown_duration_ms",
		"p95_shutdown_duration_ms",
		"max_shutdown_duration_ms",
		"attenuations_per_minute",
		"mean_attenuation_duration_ms",
		"gate_transitions_per_minute",
	)
	if not valid.any():
		return {f"{prefix}{k}": float("nan") for k in keys}
	valid_minutes = float(valid.sum().detach().cpu()) / sample_rate / 60.0
	step_ms = 1000.0 / sample_rate

	def _episodes(flag: torch.Tensor) -> torch.Tensor:
		"""每行的连续 True 段长度，向量化实现（选择器要在网格上调用数百次）。

		在每行两端各插一列 False 后展平，段边界就是展平序列上的上升/下降沿；哨兵列
		保证不同窗口的段不会跨行粘连。
		"""
		flag = (flag & valid).detach().cpu()
		pad = torch.zeros((flag.shape[0], 1), dtype=torch.bool)
		flat = torch.cat([pad, flag, pad], dim=1).reshape(-1)
		edges = flat[1:].to(torch.int8) - flat[:-1].to(torch.int8)
		starts = (edges == 1).nonzero(as_tuple=True)[0]
		ends = (edges == -1).nonzero(as_tuple=True)[0]
		return (ends - starts).to(torch.float64)

	shutdown = _episodes(gate <= zero_tol)
	attenuation = _episodes(gate < 0.5)
	below = (gate < 0.5) & valid
	transitions = (below[:, 1:] != below[:, :-1]) & valid[:, 1:] & valid[:, :-1]
	transition_count = float(transitions.sum().detach().cpu())

	def _duration_stats(lengths: torch.Tensor) -> tuple:
		if lengths.numel() == 0:
			return 0.0, 0.0, 0.0, 0.0
		durations = lengths * step_ms
		p95 = float(torch.quantile(durations, 0.95))
		return float(durations.mean()), p95, float(durations.max()), float(durations.numel())

	shut_mean, shut_p95, shut_max, shut_n = _duration_stats(shutdown)
	att_mean, _, _, att_n = _duration_stats(attenuation)
	per_minute = lambda count: count / max(valid_minutes, 1e-9)
	return {
		f"{prefix}shutdowns_per_minute": per_minute(shut_n),
		f"{prefix}mean_shutdown_duration_ms": shut_mean,
		f"{prefix}p95_shutdown_duration_ms": shut_p95,
		f"{prefix}max_shutdown_duration_ms": shut_max,
		f"{prefix}attenuations_per_minute": per_minute(att_n),
		f"{prefix}mean_attenuation_duration_ms": att_mean,
		f"{prefix}gate_transitions_per_minute": per_minute(transition_count),
	}


def risk_from_logvar_fault(
	logvar: torch.Tensor,
	fault_logit: torch.Tensor,
	uncertainty_ref: float,
) -> torch.Tensor:
	"""旧版风险：max(σ/σ_ref, p_fault)，作为消融对照保留。"""
	sigma = torch.exp(0.5 * logvar).mean(dim=1, keepdim=True)
	return torch.maximum(sigma / max(uncertainty_ref, 1e-6), torch.sigmoid(fault_logit))


def torque_replay_metrics(
	predicted_moment: torch.Tensor,
	true_moment: torch.Tensor,
	logvar: torch.Tensor,
	fault_logit: torch.Tensor,
	mask: torch.Tensor,
	uncertainty_ref: float,
	gate_softness: float = 0.5,
	sample_rate: float = 200.0,
	risk: Optional[torch.Tensor] = None,
	deadband: float = 0.0,
	gate_max_fall_per_step: float = 0.0,
	gate_max_rise_per_step: float = 0.0,
) -> Dict[str, float]:
	"""离线比较固定辅助与门控辅助的错误风险。

	risk 若给定（已标定的多信号融合风险），直接使用；否则回退到旧版
	max(σ/σ_ref, p_fault)，以保持向后兼容。
	"""
	valid_time = mask.all(dim=1, keepdim=True)
	ideal_tau = moment_to_torque(true_moment)
	baseline_tau = moment_to_torque(predicted_moment)
	if risk is None:
		risk = risk_from_logvar_fault(logvar, fault_logit, uncertainty_ref)
	gate = reliability_gate(risk, softness=gate_softness, deadband=deadband)
	gate = causal_gate_rate_limit(
		gate,
		max_fall_per_step=gate_max_fall_per_step,
		max_rise_per_step=gate_max_rise_per_step,
	)
	gated_tau = baseline_tau * gate
	metrics = {
		**_torque_metrics_for_command("baseline", baseline_tau, ideal_tau, valid_time, sample_rate),
		**_torque_metrics_for_command("gated", gated_tau, ideal_tau, valid_time, sample_rate),
		"mean_gate": float(gate[valid_time].mean().detach().cpu()) if valid_time.any() else float("nan"),
		"mean_risk": float(risk[valid_time].mean().detach().cpu()) if valid_time.any() else float("nan"),
		"full_shutdown_fraction": _safe_mean((gate <= 1e-6).float(), valid_time),
		"gate_below_half_fraction": _safe_mean((gate < 0.5).float(), valid_time),
		**gate_dynamics_metrics(gate, valid_time, sample_rate=sample_rate),
	}
	return _add_retention_semantics(metrics, "baseline", "gated")


def metrics_for_given_gate(
	gate: torch.Tensor,
	predicted_moment: torch.Tensor,
	true_moment: torch.Tensor,
	mask: torch.Tensor,
	sample_rate: float = 200.0,
) -> Dict[str, float]:
	"""对**任意**给定的门控序列算 EXPERIMENTS_TODO §2 要求的 13 项指标。

	`torque_replay_metrics` 把门控的构造（risk → sigmoid → rate limit）和指标计算绑在
	一起，因此无法用来评「随机关断」「硬阈值」「oracle 预测误差」这些不是由 risk
	生成的门控。这个函数只接受成品 gate，保证九种门控走完全相同的指标代码、同一批
	模型输出、同一个 active mask——否则跨门控的比较会混入实现差异。

	gate 形状可以是 [B,1,T] 或 [B,T]；广播到关节维后与力矩逐点相乘。
	"""
	valid_time = mask.all(dim=1, keepdim=True)
	if gate.dim() == 2:
		gate = gate.unsqueeze(1)
	ideal_tau = moment_to_torque(true_moment)
	baseline_tau = moment_to_torque(predicted_moment)
	gated_tau = baseline_tau * gate
	metrics = {
		**_torque_metrics_for_command("ungated", baseline_tau, ideal_tau, valid_time, sample_rate),
		**_torque_metrics_for_command("gated", gated_tau, ideal_tau, valid_time, sample_rate),
		"mean_gate": _safe_mean(gate, valid_time.expand_as(gate)),
		"full_shutdown_fraction": _safe_mean((gate <= 1e-6).float(), valid_time.expand_as(gate)),
		"gate_below_half_fraction": _safe_mean((gate < 0.5).float(), valid_time.expand_as(gate)),
		**gate_dynamics_metrics(gate, valid_time, sample_rate=sample_rate),
	}
	return _add_retention_semantics(metrics, "ungated", "gated")


def _add_retention_semantics(
	metrics: Dict[str, float],
	ungated_prefix: str,
	gated_prefix: str,
) -> Dict[str, float]:
	"""Expose adequacy and retention as two different quantities.

	The historical ``*_retained_aligned_torque`` field is normalized by the
	*ideal* aligned command.  It is therefore an aligned-command adequacy, not
	the fraction of the ungated estimator that survives gating.  The paper's
	``kappa`` is the latter ratio.  ``*_capped_overlap`` has the same ideal
	denominator and is likewise an adequacy; its retention is gated / ungated.
	Keep the historical fields for archived reports, but write explicit names
	so new experiments cannot silently mix the two denominators.
	"""
	ungated = metrics.get(f"{ungated_prefix}_retained_aligned_torque")
	gated = metrics.get(f"{gated_prefix}_retained_aligned_torque")
	metrics[f"{ungated_prefix}_aligned_command_adequacy"] = ungated
	metrics[f"{gated_prefix}_aligned_command_adequacy"] = gated
	if (
		ungated is None
		or gated is None
		or ungated != ungated
		or gated != gated
		or abs(ungated) < 1e-12
	):
		metrics["gate_retention_ratio"] = float("nan")
	else:
		metrics["gate_retention_ratio"] = gated / ungated

	ungated_overlap = metrics.get(f"{ungated_prefix}_capped_overlap")
	gated_overlap = metrics.get(f"{gated_prefix}_capped_overlap")
	metrics[f"{ungated_prefix}_capped_overlap_adequacy"] = ungated_overlap
	metrics[f"{gated_prefix}_capped_overlap_adequacy"] = gated_overlap
	if (
		ungated_overlap is None
		or gated_overlap is None
		or ungated_overlap != ungated_overlap
		or gated_overlap != gated_overlap
		or abs(ungated_overlap) < 1e-12
	):
		metrics["capped_overlap_retention_ratio"] = float("nan")
	else:
		metrics["capped_overlap_retention_ratio"] = gated_overlap / ungated_overlap
	return metrics


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
	# X_opp = Σ_{(j,t)∈A} max(0, −τ^cmd·τ*) · Δt，Δt = 1/sample_rate。
	# 量纲为 (Nm/kg)^2 s；没有角速度项，因此它是反向力矩乘积暴露量，不是机械能。
	wrong_torque_product_integral = torch.clamp(-(command * ideal), min=0.0)[active].sum() / sample_rate
	aligned = torch.clamp(command * torch.sign(ideal), min=0.0)
	retained = aligned[active].sum() / ideal_abs[active].sum().clamp_min(1e-6)
	same_direction = active & (command * ideal > 0.0)
	overlap = torch.minimum(command.abs(), ideal_abs)
	capped_overlap = overlap[same_direction].sum() / ideal_abs[active].sum().clamp_min(1e-6)
	over_assistance = same_direction & (command.abs() > ideal_abs)
	tracking_rmse = torch.sqrt((command[active] - ideal[active]).square().mean()) if active.any() else None
	torque_rate = torch.diff(command, dim=-1) * sample_rate
	rate_valid = valid[..., 1:]
	wrong_integral_value = (
		float(wrong_torque_product_integral.detach().cpu()) if active.any() else 0.0
	)
	torque_rate_value = (
		float(torque_rate.abs()[rate_valid].mean().detach().cpu()) if rate_valid.any() else float("nan")
	)
	return {
		f"{prefix}_wrong_direction_ratio": _safe_mean(wrong.float(), active),
		f"{prefix}_wrong_torque_product_integral": wrong_integral_value,
		# Deprecated aliases retained so archived reports and aggregation scripts remain readable.
		f"{prefix}_wrong_energy": wrong_integral_value,
		f"{prefix}_peak_wrong_torque": float(wrong_values.max().detach().cpu()) if wrong_values.numel() else 0.0,
		f"{prefix}_retained_aligned_torque": float(retained.detach().cpu()) if active.any() else float("nan"),
		f"{prefix}_capped_overlap": float(capped_overlap.detach().cpu()) if active.any() else float("nan"),
		f"{prefix}_over_assistance_fraction": _safe_mean(over_assistance.float(), active),
		f"{prefix}_tracking_rmse": float(tracking_rmse.detach().cpu()) if tracking_rmse is not None else float("nan"),
		f"{prefix}_mean_abs_torque_rate": torque_rate_value,
		f"{prefix}_mean_abs_jerk": torque_rate_value,
	}


def _safe_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
	if not mask.any():
		return float("nan")
	return float(value[mask].mean().detach().cpu())
