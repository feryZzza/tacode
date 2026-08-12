"""Mask-aware temporal model with optional domain-adversarial alignment."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = strength
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.strength * gradient, None


def gradient_reverse(value: torch.Tensor, strength: float) -> torch.Tensor:
    return _GradientReverse.apply(value, float(strength))


class CausalResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.padding = padding
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation))
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def _causal(self, layer: nn.Module, value: torch.Tensor) -> torch.Tensor:
        output = layer(value)
        return output[..., :-self.padding] if self.padding else output

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.skip(value)
        output = self.dropout(F.gelu(self._causal(self.conv1, value)))
        output = self.dropout(F.gelu(self._causal(self.conv2, output)))
        return F.gelu(output + residual)


class MaskAwareMomentTCN(nn.Module):
    """Causal moment estimator that treats sensor availability as data.

    Missing samples are zeroed after normalization and a binary availability
    channel is concatenated for every physical feature. This permits one shared
    network to consume the different sensor suites in the three datasets.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        output_size: int = 2,
        channels: Sequence[int] = (64, 64, 64, 64),
        kernel_size: int = 5,
        dropout: float = 0.1,
        num_domains: int = 3,
    ):
        super().__init__()
        self.feature_names = tuple(feature_names)
        self.output_size = int(output_size)
        self.channels = tuple(int(channel) for channel in channels)
        self.kernel_size = int(kernel_size)
        self.dropout_rate = float(dropout)
        self.num_domains = int(num_domains)
        feature_count = len(self.feature_names)
        self.register_buffer("center", torch.zeros(1, feature_count, 1))
        self.register_buffer("scale", torch.ones(1, feature_count, 1))
        blocks = []
        in_channels = feature_count * 2
        for level, out_channels in enumerate(self.channels):
            blocks.append(CausalResidualBlock(in_channels, out_channels, kernel_size, 2**level, dropout))
            in_channels = out_channels
        self.encoder = nn.Sequential(*blocks)
        self.moment_head = nn.Conv1d(self.channels[-1], self.output_size, 1)
        self.domain_head = nn.Sequential(
            nn.Linear(self.channels[-1], self.channels[-1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.channels[-1], self.num_domains),
        )

    @property
    def effective_history(self) -> int:
        return 1 + 2 * (self.kernel_size - 1) * sum(2**level for level in range(len(self.channels)))

    def set_normalization(self, center: torch.Tensor, scale: torch.Tensor) -> None:
        center = center.detach().reshape(1, -1, 1).to(self.center)
        scale = scale.detach().reshape(1, -1, 1).clamp_min(1e-5).to(self.scale)
        if center.shape != self.center.shape:
            raise ValueError(f"Expected normalization shape {self.center.shape}, got {center.shape}")
        self.center.copy_(center)
        self.scale.copy_(scale)

    def forward(
        self,
        x: torch.Tensor,
        feature_mask: torch.Tensor,
        time_mask: torch.Tensor | None = None,
        domain_lambda: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        mask = feature_mask.to(dtype=x.dtype)
        normalized = ((x - self.center) / self.scale) * mask
        hidden = self.encoder(torch.cat((normalized, mask), dim=1))
        moments = self.moment_head(hidden)
        if time_mask is None:
            pooled = hidden.mean(dim=-1)
        else:
            valid = time_mask.to(hidden.dtype).unsqueeze(1)
            pooled = (hidden * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)
        domain_features = gradient_reverse(pooled, domain_lambda) if domain_lambda else pooled
        return {
            "moments": moments,
            "domain_logits": self.domain_head(domain_features),
            "embedding": pooled,
        }

    def checkpoint_config(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "output_size": self.output_size,
            "channels": list(self.channels),
            "kernel_size": self.kernel_size,
            "dropout": self.dropout_rate,
            "num_domains": self.num_domains,
        }
