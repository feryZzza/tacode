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
	"stuck_imu": FaultSpec("stuck_signal", "foot_imu", 1.0, 1.0),
	"sensor_delay": FaultSpec("delay", "all", 1.0, 1.0, delay_steps=10),
}


def parse_fault_spec(name: str) -> FaultSpec:
	if name not in DEFAULT_FAULTS:
		valid = ", ".join(sorted(DEFAULT_FAULTS))
		raise ValueError(f"Unknown fault scenario '{name}'. Valid scenarios: {valid}")
	return DEFAULT_FAULTS[name]


def random_training_fault() -> FaultSpec:
	"""训练时随机扰动，让 fault head 见过异常。"""
	choices = [
		DEFAULT_FAULTS["clean"],
		DEFAULT_FAULTS["insole_missing"],
		DEFAULT_FAULTS["encoder_dropout"],
		DEFAULT_FAULTS["imu_bias"],
		DEFAULT_FAULTS["packet_loss"],
		DEFAULT_FAULTS["stuck_imu"],
	]
	return random.choice(choices)


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
		steps = max(int(spec.delay_steps), 1)
		delayed = torch.roll(corrupted[:, indices, :], shifts=steps, dims=-1)
		delayed[:, :, :steps] = corrupted[:, indices, :1]
		corrupted[:, indices, :] = delayed
	elif spec.name == "packet_loss":
		_apply_packet_loss(corrupted, indices, spec.severity, generator)
	else:
		raise ValueError(f"Unsupported fault type: {spec.name}")

	fault_target = torch.ones((x.shape[0], 1, x.shape[2]), device=x.device, dtype=x.dtype)
	return corrupted, fault_target


def _indices_for_group(feature_groups: Dict[str, List[int]], group: str, num_channels: int) -> List[int]:
	if group == "all":
		return list(range(num_channels))
	return list(feature_groups.get(group, []))


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
