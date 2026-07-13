"""传感器故障注入，用于可靠性评估。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class FaultSpec:
	name: str
	group: str
	severity: float = 1.0
	probability: float = 1.0
	delay_steps: int = 8


DEFAULT_FAULTS = {
	"clean": FaultSpec("clean", "all", 0.0, 0.0),
	"insole_missing": FaultSpec("zero_group", "insole", 1.0, 1.0),
	"encoder_dropout": FaultSpec("zero_group", "encoder", 1.0, 1.0),
	"imu_bias": FaultSpec("bias_drift", "shank_imu", 0.25, 1.0),
	"imu_axis_scale": FaultSpec("scale_error", "thigh_imu", 0.35, 1.0),
	"packet_loss": FaultSpec("packet_loss", "all", 0.15, 1.0),
	"packet_loss_burst": FaultSpec("packet_loss_burst", "all", 0.15, 1.0),
	"packet_loss_partial": FaultSpec("packet_loss_partial", "all", 0.15, 1.0),
	"stuck_imu": FaultSpec("stuck_signal", "foot_imu", 1.0, 1.0),
	"sensor_delay": FaultSpec("delay", "all", 1.0, 1.0, delay_steps=10),
	"sensor_delay_jitter": FaultSpec("delay_jitter", "all", 1.0, 1.0, delay_steps=10),
}


_FAULT_CLASS_ALIASES = {
	"packet_loss_burst": "packet_loss",
	"packet_loss_partial": "packet_loss",
	"sensor_delay_jitter": "sensor_delay",
}


def canonical_fault_name(name: str) -> str:
	"""Map severity specs and held-out variants to their supervised fault head."""
	base_name, _ = _split_fault_name(name)
	return _FAULT_CLASS_ALIASES.get(base_name, base_name)


def fault_classes_from_names(names: Sequence[str]) -> List[str]:
	classes = []
	for name in names:
		fault_class = canonical_fault_name(name)
		if fault_class != "clean" and fault_class not in classes:
			classes.append(fault_class)
	return classes


def parse_fault_spec(name: str) -> FaultSpec:
	base_name, modifier = _split_fault_name(name)
	if base_name not in DEFAULT_FAULTS:
		valid = ", ".join(sorted(DEFAULT_FAULTS))
		raise ValueError(f"Unknown fault scenario '{name}'. Valid scenarios: {valid}")
	spec = DEFAULT_FAULTS[base_name]
	if modifier is None:
		return spec
	value = _parse_fault_modifier(name, modifier)
	if spec.name in ("delay", "delay_jitter"):
		return FaultSpec(spec.name, spec.group, spec.severity, spec.probability, delay_steps=max(1, int(round(value))))
	return FaultSpec(spec.name, spec.group, severity=max(value, 0.0), probability=spec.probability, delay_steps=spec.delay_steps)


def random_training_fault(fault_names: Optional[Sequence[str]] = None) -> FaultSpec:
	"""训练时随机扰动，让 fault head 见过异常。"""
	names = list(fault_names) if fault_names is not None else [
		"clean",
		"insole_missing",
		"encoder_dropout",
		"imu_bias",
		"packet_loss",
		"stuck_imu",
		"sensor_delay",
	]
	choices = [parse_fault_spec(name) for name in names]
	return random.choice(choices)


def _split_fault_name(name: str) -> Tuple[str, Optional[str]]:
	"""Allow stress-test specs such as packet_loss@0.30 or sensor_delay@20."""
	clean = name.strip()
	if "@" not in clean:
		return clean, None
	base_name, modifier = clean.split("@", 1)
	return base_name.strip(), modifier.strip() or None


def _parse_fault_modifier(full_name: str, modifier: str) -> float:
	value = modifier.lower().replace("steps", "").replace("step", "").strip()
	try:
		return float(value)
	except ValueError as exc:
		raise ValueError(f"Fault modifier must be numeric in '{full_name}'.") from exc


def apply_fault(
	x: torch.Tensor,
	feature_groups: Dict[str, List[int]],
	spec: FaultSpec,
	generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""注入一种故障，并返回故障标签。

	Args:
		x: Tensor with shape [batch, channels, time].
		feature_groups: Mapping from semantic group name to channel indices.
		spec: Fault specification.
		generator: Optional torch random generator.

	Returns:
		(corrupted_x, fault_target), where fault_target has shape [batch, 1, time].
	"""
	if spec.name == "clean" or spec.probability <= 0:
		return x, torch.zeros((x.shape[0], 1, x.shape[2]), device=x.device, dtype=x.dtype)

	corrupted = x.clone()
	indices = _indices_for_group(feature_groups, spec.group, x.shape[1])
	if not indices:
		return corrupted, torch.zeros((x.shape[0], 1, x.shape[2]), device=x.device, dtype=x.dtype)

	if spec.name == "zero_group":
		corrupted[:, indices, :] = 0.0
	elif spec.name == "bias_drift":
		_apply_bias_drift(corrupted, indices, spec.severity)
	elif spec.name == "scale_error":
		corrupted[:, indices, :] = corrupted[:, indices, :] * (1.0 + spec.severity)
	elif spec.name == "stuck_signal":
		corrupted[:, indices, :] = corrupted[:, indices, :1]
	elif spec.name == "delay":
		_apply_jittered_delay(corrupted, feature_groups, indices, spec.delay_steps)
	elif spec.name == "delay_jitter":
		_apply_random_jittered_delay(corrupted, feature_groups, indices, spec.delay_steps, generator)
	elif spec.name == "packet_loss":
		_apply_packet_loss(corrupted, indices, spec.severity, generator)
	elif spec.name == "packet_loss_burst":
		_apply_packet_loss_burst(corrupted, indices, spec.severity, generator)
	elif spec.name == "packet_loss_partial":
		_apply_packet_loss_partial(corrupted, indices, spec.severity, generator)
	else:
		raise ValueError(f"Unsupported fault type: {spec.name}")

	fault_target = torch.ones((x.shape[0], 1, x.shape[2]), device=x.device, dtype=x.dtype)
	return corrupted, fault_target


def _indices_for_group(feature_groups: Dict[str, List[int]], group: str, num_channels: int) -> List[int]:
	if group == "all":
		return list(range(num_channels))
	return list(feature_groups.get(group, []))


# 各模态相对 base 延迟的固定倍率，模拟真实总线/传感器传输延迟不一致。
# 固定倍率保证 sensor_delay@N 应力曲线随 N 单调，便于跨强度比较。
_DELAY_GROUP_MULTIPLIERS = {
	"foot_imu": 1.0,
	"shank_imu": 1.6,
	"thigh_imu": 0.6,
	"insole": 2.0,
	"encoder": 1.2,
	"action": 0.4,
}


def _apply_jittered_delay(
	x: torch.Tensor,
	feature_groups: Dict[str, List[int]],
	indices: Sequence[int],
	base_steps: int,
) -> None:
	"""逐组抖动延迟：各模态按不同步数时移，制造跨通道相位不一致。

	全局均匀时移使整段信号自洽、从输入侧几乎不可辨；让各模态延迟步数不同后，
	通道间相位错位破坏了输入的时序结构，预测残差与跨通道一致性都能察觉。
	未被任何已知模态覆盖的通道按 base_steps 兜底时移。
	"""
	base = max(int(base_steps), 1)
	assigned = set()
	for group_name, channels in feature_groups.items():
		group_idx = [i for i in channels if i in set(indices)]
		if not group_idx:
			continue
		multiplier = _DELAY_GROUP_MULTIPLIERS.get(group_name, 1.0)
		steps = max(int(round(base * multiplier)), 1)
		_roll_hold(x, group_idx, steps)
		assigned.update(group_idx)
	leftover = [i for i in indices if i not in assigned]
	if leftover:
		_roll_hold(x, leftover, base)


def _roll_hold(x: torch.Tensor, channels: Sequence[int], steps: int) -> None:
	"""把给定通道整体后移 steps，并用首帧保持填充开头。"""
	steps = max(int(steps), 1)
	delayed = torch.roll(x[:, channels, :], shifts=steps, dims=-1)
	delayed[:, :, :steps] = x[:, channels, :1]
	x[:, channels, :] = delayed


def _apply_random_jittered_delay(
	x: torch.Tensor,
	feature_groups: Dict[str, List[int]],
	indices: Sequence[int],
	base_steps: int,
	generator: Optional[torch.Generator],
) -> None:
	"""逐 batch/模态随机延迟，作为未见过的 sensor-delay 泛化测试。

	训练默认只见固定倍率的 sensor_delay；这个 held-out 版本给每个样本和模态
	抽不同 delay，使检测器不能只依赖固定相位错位模式。
	"""
	base = max(int(base_steps), 1)
	idx_set = set(indices)
	assigned = set()
	for group_name, channels in feature_groups.items():
		group_idx = [i for i in channels if i in idx_set]
		if not group_idx:
			continue
		multiplier = _DELAY_GROUP_MULTIPLIERS.get(group_name, 1.0)
		nominal = max(int(round(base * multiplier)), 1)
		_roll_hold_random_per_batch(x, group_idx, nominal, generator)
		assigned.update(group_idx)
	leftover = [i for i in indices if i not in assigned]
	if leftover:
		_roll_hold_random_per_batch(x, leftover, base, generator)


def _roll_hold_random_per_batch(
	x: torch.Tensor,
	channels: Sequence[int],
	nominal_steps: int,
	generator: Optional[torch.Generator],
) -> None:
	low = max(int(round(nominal_steps * 0.5)), 1)
	high = max(int(round(nominal_steps * 1.5)), low + 1)
	for batch_index in range(x.shape[0]):
		steps = int(torch.randint(low, high + 1, (1,), device=x.device, generator=generator).item())
		delayed = torch.roll(x[batch_index : batch_index + 1, channels, :], shifts=steps, dims=-1)
		delayed[:, :, :steps] = x[batch_index : batch_index + 1, channels, :1]
		x[batch_index : batch_index + 1, channels, :] = delayed


def _apply_bias_drift(x: torch.Tensor, indices: Sequence[int], severity: float) -> None:
	"""用线性漂移模拟 IMU 慢偏置。"""
	group = x[:, indices, :]
	scale = group.std(dim=-1, keepdim=True).clamp_min(1e-3)
	ramp = torch.linspace(0.0, 1.0, x.shape[-1], device=x.device, dtype=x.dtype).view(1, 1, -1)
	x[:, indices, :] = group + severity * scale * ramp


def _apply_packet_loss(
	x: torch.Tensor,
	indices: Sequence[int],
	severity: float,
	generator: Optional[torch.Generator],
) -> None:
	"""用前值保持模拟丢包。"""
	probability = min(max(severity, 0.0), 0.8)
	mask = torch.rand(
		(x.shape[0], 1, x.shape[2]),
		device=x.device,
		dtype=x.dtype,
		generator=generator,
	) < probability
	group = x[:, indices, :]
	held = group.clone()
	for t in range(1, group.shape[-1]):
		held[:, :, t] = torch.where(mask[:, :, t], held[:, :, t - 1], group[:, :, t])
	x[:, indices, :] = held


def _apply_packet_loss_burst(
	x: torch.Tensor,
	indices: Sequence[int],
	severity: float,
	generator: Optional[torch.Generator],
) -> None:
	"""突发丢包：连续若干帧前值保持，作为 packet_loss 的 held-out 变体。"""
	probability = min(max(severity, 0.0), 0.8)
	group = x[:, indices, :]
	mask = torch.zeros((x.shape[0], 1, x.shape[2]), device=x.device, dtype=torch.bool)
	burst_len = max(2, int(round(2 + 24 * probability)))
	start_prob = min(probability / burst_len, 0.5)
	remaining = torch.zeros((x.shape[0], 1), device=x.device, dtype=torch.long)
	for t in range(1, x.shape[-1]):
		start = torch.rand((x.shape[0], 1), device=x.device, generator=generator) < start_prob
		remaining = torch.where((remaining <= 0) & start, torch.full_like(remaining, burst_len), remaining)
		active = remaining > 0
		mask[:, :, t] = active
		remaining = torch.clamp_min(remaining - 1, 0)
	_forward_fill_where(x, indices, group, mask)


def _apply_packet_loss_partial(
	x: torch.Tensor,
	indices: Sequence[int],
	severity: float,
	generator: Optional[torch.Generator],
) -> None:
	"""部分通道独立丢包，避免 detector 只抓"全通道同时冻结"。"""
	probability = min(max(severity, 0.0), 0.8)
	group = x[:, indices, :]
	mask = torch.rand(
		(x.shape[0], len(indices), x.shape[2]),
		device=x.device,
		dtype=x.dtype,
		generator=generator,
	) < probability
	_forward_fill_where(x, indices, group, mask)


def _forward_fill_where(
	x: torch.Tensor,
	indices: Sequence[int],
	group: torch.Tensor,
	mask: torch.Tensor,
) -> None:
	held = group.clone()
	for t in range(1, group.shape[-1]):
		held[:, :, t] = torch.where(mask[:, :, t], held[:, :, t - 1], group[:, :, t])
	x[:, indices, :] = held
