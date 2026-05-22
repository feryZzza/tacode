"""输出力矩、不确定性和故障概率的 TCN。"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from tcn import TemporalConvNet


class ReliabilityTCN(nn.Module):
	"""共享时序编码器，分头预测 mean/logvar/fault。"""

	def __init__(
		self,
		input_size: int,
		output_size: int = 2,
		num_channels: List[int] | None = None,
		ksize: int = 4,
		dropout: float = 0.1,
		spatial_dropout: bool = False,
		activation: str = "ReLU",
		norm: str = "weight_norm",
		center: torch.Tensor | float = 0.0,
		scale: torch.Tensor | float = 1.0,
	):
		super().__init__()
		num_channels = num_channels or [32, 32, 32, 32]
		dropout_type = "Dropout2d" if spatial_dropout else "Dropout"
		self.tcn = TemporalConvNet(
			input_size,
			num_channels,
			kernel_size=ksize,
			dropout=dropout,
			dropout_type=dropout_type,
			activation=activation,
			norm=norm,
		)
		hidden_size = num_channels[-1]
		# 三个 head 对应估计值、置信度和传感器健康度。
		self.mean_head = nn.Conv1d(hidden_size, output_size, kernel_size=1)
		self.logvar_head = nn.Conv1d(hidden_size, output_size, kernel_size=1)
		self.fault_head = nn.Conv1d(hidden_size, 1, kernel_size=1)
		self.register_buffer("center", _as_feature_buffer(center, input_size))
		self.register_buffer("scale", _as_feature_buffer(scale, input_size))
		self.output_size = output_size
		self.input_size = input_size
		self.init_heads()

	def init_heads(self) -> None:
		for head in (self.mean_head, self.logvar_head, self.fault_head):
			nn.init.normal_(head.weight, mean=0.0, std=0.01)
			nn.init.zeros_(head.bias)
		nn.init.constant_(self.logvar_head.bias, -2.0)

	def set_normalization(self, center: torch.Tensor, scale: torch.Tensor) -> None:
		self.center = _as_feature_buffer(center, self.input_size).to(self.center.device)
		self.scale = _as_feature_buffer(scale, self.input_size).to(self.scale.device)

	def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
		"""输入 [B,C,T]，输出逐时刻可靠性信息。"""
		z = (x - self.center) / self.scale.clamp_min(1e-6)
		hidden = self.tcn(z)
		logvar = self.logvar_head(hidden).clamp(min=-8.0, max=5.0)
		return {
			"mean": self.mean_head(hidden),
			"logvar": logvar,
			"fault_logit": self.fault_head(hidden),
		}


def _as_feature_buffer(value: torch.Tensor | float, input_size: int) -> torch.Tensor:
	if not torch.is_tensor(value):
		value = torch.full((input_size,), float(value), dtype=torch.float32)
	value = value.detach().float()
	if value.ndim == 1:
		value = value.view(1, -1, 1)
	if value.shape[1] != input_size:
		raise ValueError(f"Expected {input_size} normalization values, got {value.shape[1]}")
	return value


def save_checkpoint(path: str, model: ReliabilityTCN, extra: dict) -> None:
	payload = {
		"state_dict": model.state_dict(),
		"model_kwargs": {
			"input_size": model.input_size,
			"output_size": model.output_size,
		},
	}
	payload.update(extra)
	torch.save(payload, path)


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> tuple[ReliabilityTCN, dict]:
	payload = torch.load(path, map_location=map_location)
	model_kwargs = dict(payload["model_kwargs"])
	for key in ("num_channels", "ksize", "dropout", "spatial_dropout"):
		if key in payload:
			model_kwargs[key] = payload[key]
	model = ReliabilityTCN(**model_kwargs)
	model.load_state_dict(payload["state_dict"])
	return model, payload
