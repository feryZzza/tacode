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
		use_recon: bool = True,
		recon_detach: bool = False,
		use_forecast: bool = True,
		forecast_horizon: int = 1,
		forecast_detach: bool = True,
		fault_classes: List[str] | None = None,
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
		self.fault_classes = list(fault_classes or ["fault"])
		self.fault_head = nn.Conv1d(hidden_size, len(self.fault_classes), kernel_size=1)
		# 输入完整性头：去噪重构归一化输入，残差能量度量输入是否偏离训练分布。
		self.use_recon = use_recon
		self.recon_detach = recon_detach
		self.recon_head = nn.Conv1d(hidden_size, input_size, kernel_size=1) if use_recon else None
		# 时序预测头：用 hidden[t] 预测 z[t+h]。因果 TCN 下同时刻重构是平凡的，
		# 但一步预测对丢包(前值保持)/延迟(相位错位)/缓漂(累积偏移)等
		# 时间结构故障敏感——这些故障破坏可预测性，使预测残差升高。
		# forecast_detach=True 时切断预测头到共享 trunk 的梯度：预测头退化为
		# 纯时序探针，其多任务梯度不再污染安全关键的 mean 头（修正干净精度回退）。
		self.use_forecast = use_forecast
		self.forecast_horizon = max(int(forecast_horizon), 1)
		self.forecast_detach = forecast_detach
		self.forecast_head = nn.Conv1d(hidden_size, input_size, kernel_size=1) if use_forecast else None
		self.register_buffer("center", _as_feature_buffer(center, input_size))
		self.register_buffer("scale", _as_feature_buffer(scale, input_size))
		self.output_size = output_size
		self.input_size = input_size
		self.init_heads()

	def init_heads(self) -> None:
		heads = [self.mean_head, self.logvar_head, self.fault_head]
		if self.recon_head is not None:
			heads.append(self.recon_head)
		if self.forecast_head is not None:
			heads.append(self.forecast_head)
		for head in heads:
			nn.init.normal_(head.weight, mean=0.0, std=0.01)
			nn.init.zeros_(head.bias)
		nn.init.constant_(self.logvar_head.bias, -2.0)

	def set_normalization(self, center: torch.Tensor, scale: torch.Tensor) -> None:
		self.center = _as_feature_buffer(center, self.input_size).to(self.center.device)
		self.scale = _as_feature_buffer(scale, self.input_size).to(self.scale.device)

	def normalize(self, x: torch.Tensor) -> torch.Tensor:
		"""把原始输入映射到模型内部使用的归一化空间。"""
		return (x - self.center) / self.scale.clamp_min(1e-6)

	def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
		"""输入 [B,C,T]，输出逐时刻可靠性信息。"""
		z = self.normalize(x)
		hidden = self.tcn(z)
		logvar = self.logvar_head(hidden).clamp(min=-8.0, max=5.0)
		fault_logits = self.fault_head(hidden)
		out = {
			"mean": self.mean_head(hidden),
			"logvar": logvar,
			"fault_logits": fault_logits,
			"fault_logit": fault_logits.max(dim=1, keepdim=True).values,
			"input_norm": z,
		}
		if self.recon_head is not None:
			# 在归一化空间重构"干净"输入；与观测 z 的差即输入完整性残差。
			recon_hidden = hidden.detach() if self.recon_detach else hidden
			out["recon"] = self.recon_head(recon_hidden)
		if self.forecast_head is not None:
			# hidden[t] 预测 z[t+horizon]；评估端按 horizon 错位求残差。
			# 默认 detach：预测头从 trunk 取特征但不回传梯度，避免拖累主任务精度。
			fc_hidden = hidden.detach() if self.forecast_detach else hidden
			out["forecast"] = self.forecast_head(fc_hidden)
		return out

	def enable_mc_dropout(self) -> None:
		"""保持网络处于 eval，但打开 dropout 以做 MC-dropout 认知不确定性采样。"""
		self.eval()
		for module in self.modules():
			if isinstance(module, (nn.Dropout, nn.Dropout2d)):
				module.train()

	@torch.no_grad()
	def predict_uncertainty(self, x: torch.Tensor, samples: int = 10) -> dict[str, torch.Tensor]:
		"""MC-dropout 推理，分离 aleatoric 与 epistemic 不确定性。

		返回 mean（采样均值）、aleatoric_sigma（异方差头均值）、
		epistemic_sigma（采样间均值预测的标准差）。
		"""
		if samples <= 1 or not _has_dropout(self):
			out = self.forward(x)
			return {
				"mean": out["mean"],
				"aleatoric_sigma": torch.exp(0.5 * out["logvar"]),
				"epistemic_sigma": torch.zeros_like(out["mean"]),
			}
		self.enable_mc_dropout()
		means = []
		aleatorics = []
		for _ in range(samples):
			out = self.forward(x)
			means.append(out["mean"])
			aleatorics.append(torch.exp(0.5 * out["logvar"]))
		self.eval()
		stacked = torch.stack(means, dim=0)
		return {
			"mean": stacked.mean(dim=0),
			"aleatoric_sigma": torch.stack(aleatorics, dim=0).mean(dim=0),
			"epistemic_sigma": stacked.std(dim=0),
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


def _has_dropout(model: nn.Module) -> bool:
	return any(isinstance(module, (nn.Dropout, nn.Dropout2d)) for module in model.modules())


def save_checkpoint(path: str, model: ReliabilityTCN, extra: dict) -> None:
	payload = {
		"state_dict": model.state_dict(),
		"model_kwargs": {
			"input_size": model.input_size,
			"output_size": model.output_size,
			"use_recon": model.use_recon,
			"recon_detach": model.recon_detach,
			"use_forecast": model.use_forecast,
			"forecast_horizon": model.forecast_horizon,
			"forecast_detach": model.forecast_detach,
			"fault_classes": model.fault_classes,
		},
	}
	payload.update(extra)
	torch.save(payload, path)


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> tuple[ReliabilityTCN, dict]:
	payload = torch.load(path, map_location=map_location)
	model_kwargs = dict(payload["model_kwargs"])
	for key in ("num_channels", "ksize", "dropout", "spatial_dropout", "activation", "norm", "use_recon", "recon_detach", "use_forecast", "forecast_horizon", "forecast_detach", "fault_classes"):
		if key in payload:
			model_kwargs[key] = payload[key]
	# 兼容旧 checkpoint（无 recon / forecast 头）。
	if "use_recon" not in model_kwargs:
		model_kwargs["use_recon"] = any(key.startswith("recon_head") for key in payload["state_dict"])
	if "use_forecast" not in model_kwargs:
		model_kwargs["use_forecast"] = any(key.startswith("forecast_head") for key in payload["state_dict"])
	# 旧 checkpoint 的 reconstruction loss 会回传共享主干，加载时保持原行为。
	if "recon_detach" not in model_kwargs:
		model_kwargs["recon_detach"] = False
	# 旧 checkpoint 无 forecast_detach：默认按未切断梯度（与训练时一致）。
	if "forecast_detach" not in model_kwargs:
		model_kwargs["forecast_detach"] = False
	if "fault_classes" not in model_kwargs:
		fault_channels = payload["state_dict"]["fault_head.weight"].shape[0]
		model_kwargs["fault_classes"] = ["fault"] if fault_channels == 1 else [f"fault_{i}" for i in range(fault_channels)]
	model = ReliabilityTCN(**model_kwargs)
	model.load_state_dict(payload["state_dict"])
	return model, payload
