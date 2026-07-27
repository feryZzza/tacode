"""AAAI-27 门控实验的共享评测底座（EXPERIMENTS_TODO §2/§3/§4/§5）。

这些实验的共同硬性要求是「同一批模型输出、同一组窗口、同一个 active mask、固定
种子」。把收集与门控构造放在一个模块里，是为了让九种门控、rate-limit 网格、瞬态
故障与配对分析走**同一条**代码路径——否则跨门控的差异里会混进实现差异。

三个设计决定值得记下来：

1. 收集时保留 provenance（participant / trial / task / window start），因为 §4 要逐
   timestep 落盘、§5 要把每个 clean 窗口和它的九个受损副本配上对，而
   `run_reliability_experiment.collect_reliability_outputs` 会把这些字段丢掉。
2. 门控一律返回成品张量，指标一律走 `metrics_for_given_gate`。随机门控与 oracle
   门控不是由 risk 生成的，走不了 `torque_replay_metrics`。
3. $X_{\\mathrm{opp}}$ 与 aligned retention 对**非负**门控是逐点线性的
   （$\\max(0,-(g\\,\\tau^{cmd}\\tau^*)) = g\\max(0,-\\tau^{cmd}\\tau^*)$），所以 1000 次随机
   抽样只是 1000 个点积，不需要重跑指标代码。`ExposureBasis` 就是这个预计算。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from reliability.faults import apply_fault, apply_fault_interval, parse_fault_spec
from reliability.metrics import (
	causal_gate_rate_limit,
	drift_score,
	forecast_residual,
	moment_to_torque,
	reconstruction_residual,
	reliability_gate,
	risk_from_channels,
	staleness_score,
)

#: 命令时刻可用的六路风险通道（Detector-gate 的 K_gate）。coherence/drift 只作诊断。
K_GATE: Tuple[str, ...] = ("logit", "aleatoric", "residual", "forecast", "epistemic", "staleness")

#: 通道名 -> `calibrate_risk`/`estimate_risk_refs` 里的 refs 键名。
_REF_KEY = {"logit": "fault", "aleatoric": "aleatoric", "residual": "residual",
	"forecast": "forecast", "epistemic": "epistemic", "staleness": "staleness", "drift": "drift"}

#: |ideal torque| 低于该值的时刻不计入 active mask，与 `_torque_metrics_for_command` 一致。
ACTIVE_TORQUE_FLOOR = 0.02


@dataclass
class WindowBatch:
	"""一个 (split, fault) 条件下的全部窗口输出 + provenance。"""

	mean: torch.Tensor          # [N,J,T] 预测力矩（moment）
	y: torch.Tensor             # [N,J,T] 真实力矩
	mask: torch.Tensor          # [N,J,T] 标签有效
	channels: Dict[str, torch.Tensor] = field(default_factory=dict)  # 每路 [N,1,T]
	fault_target: Optional[torch.Tensor] = None                      # [N,1,T]
	participant: List[str] = field(default_factory=list)
	trial: List[str] = field(default_factory=list)
	task: List[str] = field(default_factory=list)
	start: List[int] = field(default_factory=list)

	def __len__(self) -> int:
		return int(self.mean.shape[0])

	def valid_time(self) -> torch.Tensor:
		return self.mask.all(dim=1, keepdim=True)

	def window_keys(self) -> List[Tuple[str, str, int]]:
		return list(zip(self.participant, self.trial, self.start))


def collect_windows(
	model,
	dataset,
	groups: Dict[str, List[int]],
	args: argparse.Namespace,
	device: torch.device,
	fault_name: str = "clean",
	interval: Optional[Tuple[int, Optional[int]]] = None,
	extra_specs: Sequence[str] = (),
	generator_seed: Optional[int] = None,
	max_windows: Optional[int] = None,
) -> Optional[WindowBatch]:
	"""前向一遍并保留 provenance 与全部风险通道。

	`interval=(onset, offset)` 时走 `apply_fault_interval`，只损坏窗口的一段（§4）；
	`offset=None` 表示故障持续到窗口末尾。`extra_specs` 用于同时注入第二个故障算子。
	`generator_seed` 固定随机类故障（packet loss / delay jitter）的抽样，保证跨门控、
	跨 stage 复现同一份损坏输入。
	"""
	import run_reliability_experiment as rre

	loader = rre.make_loader(dataset, args, device, shuffle=False)
	specs = [parse_fault_spec(fault_name)] + [parse_fault_spec(name) for name in extra_specs]
	horizon = getattr(model, "forecast_horizon", 1)
	use_recon = getattr(model, "recon_head", None) is not None
	use_forecast = getattr(model, "forecast_head", None) is not None
	generator = None
	if generator_seed is not None:
		generator = torch.Generator(device=device)
		generator.manual_seed(int(generator_seed))

	acc: Dict[str, List[torch.Tensor]] = {}
	meta: Dict[str, List] = {"participant": [], "trial": [], "task": [], "start": []}

	def push(key: str, value: torch.Tensor) -> None:
		acc.setdefault(key, []).append(value.detach().cpu())

	model.eval()
	seen = 0
	with torch.no_grad():
		for batch in loader:
			x = batch["x"].to(device)
			y = batch["y"].to(device)
			mask = rre.apply_ignore_history(batch["mask"].to(device), args.eval_ignore_history)
			if interval is None:
				x_eval, fault_target = apply_fault(x, groups, specs[0], generator=generator)
				for spec in specs[1:]:
					x_eval, _ = apply_fault(x_eval, groups, spec, generator=generator)
			else:
				x_eval, fault_target = apply_fault_interval(
					x, groups, specs, onset=interval[0], offset=interval[1], generator=generator
				)
			out = model(x_eval)
			push("mean", out["mean"])
			push("y", y)
			push("mask", mask)
			push("fault_target", fault_target)
			push("logit", torch.sigmoid(out["fault_logit"]))
			push("aleatoric", torch.exp(0.5 * out["logvar"]).mean(dim=1, keepdim=True))
			if use_recon and "recon" in out:
				push("residual", reconstruction_residual(out["recon"], out["input_norm"], groups))
			if use_forecast and "forecast" in out:
				push("forecast", forecast_residual(out["forecast"], out["input_norm"], horizon, groups))
			push("staleness", staleness_score(out["input_norm"], groups))
			push("drift", drift_score(out["input_norm"], groups))
			if getattr(args, "mc_samples", 0) and args.mc_samples > 1:
				mc = model.predict_uncertainty(x_eval, samples=args.mc_samples)
				push("epistemic", mc["epistemic_sigma"].mean(dim=1, keepdim=True))
			for key in meta:
				value = batch.get(key)
				if value is None:
					meta[key].extend([""] * x.shape[0])
				elif torch.is_tensor(value):
					meta[key].extend([int(v) for v in value.tolist()])
				else:
					meta[key].extend(list(value))
			seen += int(x.shape[0])
			if max_windows and seen >= max_windows:
				break

	if "mean" not in acc:
		return None
	joined = {key: torch.cat(values) for key, values in acc.items()}
	limit = max_windows or joined["mean"].shape[0]
	batch_out = WindowBatch(
		mean=joined["mean"][:limit],
		y=joined["y"][:limit],
		mask=joined["mask"][:limit].bool(),
		fault_target=joined["fault_target"][:limit],
		channels={
			key: value[:limit]
			for key, value in joined.items()
			if key not in ("mean", "y", "mask", "fault_target")
		},
		participant=meta["participant"][:limit],
		trial=meta["trial"][:limit],
		task=meta["task"][:limit],
		start=[int(s) for s in meta["start"][:limit]],
	)
	return batch_out


# ---------------------------------------------------------------- 门控构造


@dataclass
class GatePolicy:
	"""冻结的门控工作点（从 `_gate_policy` 读回，不重新在 val 上选）。"""

	softness: float
	deadband: float
	max_fall_per_step: float = 0.0
	max_rise_per_step: float = 0.0
	signals: Tuple[str, ...] = K_GATE
	source: str = ""

	@classmethod
	def from_report(cls, report: dict) -> "GatePolicy":
		results = report.get("results", {})
		policy = results.get("_gate_policy", {})
		args = report.get("args", {})
		raw = results.get("_detector_policy", {}).get("online_signals", "")
		if isinstance(raw, str):
			signals = tuple(s.strip() for s in raw.split(",") if s.strip())
		else:
			signals = tuple(raw)
		return cls(
			softness=float(policy.get("softness", args.get("gate_softness", 0.5))),
			deadband=float(policy.get("deadband", args.get("gate_deadband", 0.0))),
			max_fall_per_step=float(
				policy.get("selected_gate_max_fall_per_step", args.get("gate_max_fall_per_step", 0.0) or 0.0)
			),
			max_rise_per_step=float(
				policy.get("selected_gate_max_rise_per_step", args.get("gate_max_rise_per_step", 0.0) or 0.0)
			),
			signals=signals or K_GATE,
			source=str(policy.get("policy_source", "")),
		)


def channel_subset(batch: WindowBatch, names: Sequence[str]) -> Dict[str, torch.Tensor]:
	"""按通道名取子集，映射到 `risk_from_channels` 的 refs 键名。缺失通道静默跳过。"""
	out: Dict[str, torch.Tensor] = {}
	for name in names:
		tensor = batch.channels.get(name)
		if tensor is not None:
			out[_REF_KEY.get(name, name)] = tensor
	return out


def soft_gate(
	batch: WindowBatch,
	names: Sequence[str],
	policy: GatePolicy,
	risk_refs: Optional[Dict[str, float]],
	rate_limited: bool = True,
) -> Optional[torch.Tensor]:
	"""指定通道子集 → 标定风险 → sigmoid 门控 → （可选）因果 rate limit。"""
	channels = channel_subset(batch, names)
	if not channels:
		return None
	risk = risk_from_channels(channels, risk_refs)
	gate = reliability_gate(risk, softness=policy.softness, deadband=policy.deadband)
	if rate_limited:
		gate = causal_gate_rate_limit(
			gate,
			max_fall_per_step=policy.max_fall_per_step,
			max_rise_per_step=policy.max_rise_per_step,
		)
	return gate


def hard_threshold_gate(
	batch: WindowBatch,
	names: Sequence[str],
	risk_refs: Optional[Dict[str, float]],
	threshold: float = 1.0,
) -> Optional[torch.Tensor]:
	"""硬阈值门控：risk 超过标定阈值就完全断力矩，无软化、无 rate limit。

	这是「不平滑」的一端，用来说明软门控换来的是什么（§3 的 trade-off 行）。
	"""
	channels = channel_subset(batch, names)
	if not channels:
		return None
	risk = risk_from_channels(channels, risk_refs)
	return (risk <= threshold).to(dtype=batch.mean.dtype)


def oracle_error_gate(
	batch: WindowBatch,
	policy: GatePolicy,
	error_ref: float,
) -> torch.Tensor:
	"""oracle 门控：用真实预测误差（|τ̂-τ*| 在关节维取均值）当风险。

	它不可实现，作用是给出「如果检测器完美」时门控能达到的上界；实际门控与它的
	差距就是检测误差造成的损失，而不是门控形式造成的。误差用 `error_ref` 归一，
	与其他通道同样以 1.0 为名义阈值。
	"""
	error = (moment_to_torque(batch.mean) - moment_to_torque(batch.y)).abs().mean(dim=1, keepdim=True)
	risk = error / max(error_ref, 1e-6)
	gate = reliability_gate(risk, softness=policy.softness, deadband=policy.deadband)
	return causal_gate_rate_limit(
		gate, max_fall_per_step=policy.max_fall_per_step, max_rise_per_step=policy.max_rise_per_step
	)


def random_shutdown_gate(
	shape: Tuple[int, ...],
	valid: torch.Tensor,
	shutdown_fraction: float,
	generator: torch.Generator,
	dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
	"""$P(g{=}0)$-matched 随机关断：与实际门控同样的关断占比，但时机随机。

	它回答的是「收益是否只来自『少输出力矩』」：如果随机关断同样的时间比例就能拿到
	相同的 $X_{\\mathrm{opp}}$ 下降，那门控的时机选择没有贡献。
	"""
	draw = torch.rand(shape, generator=generator, dtype=torch.float32)
	gate = (draw >= shutdown_fraction).to(dtype=dtype)
	gate = torch.where(valid.expand_as(gate), gate, torch.ones_like(gate))
	return gate


def permuted_gate(
	gate: torch.Tensor,
	valid: torch.Tensor,
	generator: torch.Generator,
) -> torch.Tensor:
	"""把实际门控值在有效时刻之间随机置换。

	置换保持门控值的**边缘分布**完全不变（因此 clean retention、$P(g{=}0)$、平均衰减
	都自动匹配），只销毁「什么时候衰减」的信息。这是比「同均值常数门控」更严格的
	对照：任何剩余的 $X_{\\mathrm{opp}}$ 优势只能来自时机。
	"""
	out = gate.clone()
	flat_valid = valid.expand_as(gate).reshape(-1)
	values = gate.reshape(-1)[flat_valid]
	if values.numel() == 0:
		return out
	perm = torch.randperm(values.numel(), generator=generator)
	flat = out.reshape(-1)
	flat[flat_valid] = values[perm]
	return flat.reshape(gate.shape)


# ---------------------------------------------------------------- 快速暴露量


@dataclass
class ExposureBasis:
	"""$X_{\\mathrm{opp}}$ / retention 对非负门控的线性基（随机化检验用）。

	对 $g\\ge 0$：$\\max(0,-(g\\tau^{cmd}\\tau^*)) = g\\max(0,-\\tau^{cmd}\\tau^*)$，
	aligned retention 的分子同理逐点线性。所以只要预存这两个逐点系数，任意门控的
	暴露量就是一次带 mask 的点积——1000 次抽样从「重跑指标」变成「1000 个点积」。
	"""

	opposition: torch.Tensor   # [N,J,T] max(0, -(τ̂·τ*))，非 active 处置零
	aligned: torch.Tensor      # [N,J,T] max(0, τ̂·sign(τ*))，非 active 处置零
	denom: float               # Σ|τ*| over active
	sample_rate: float
	active: torch.Tensor       # [N,J,T] bool

	@classmethod
	def build(cls, batch: WindowBatch, sample_rate: float = 200.0) -> "ExposureBasis":
		ideal = moment_to_torque(batch.y)
		command = moment_to_torque(batch.mean)
		valid = batch.valid_time().expand_as(command)
		active = valid & (ideal.abs() > ACTIVE_TORQUE_FLOOR)
		opposition = torch.clamp(-(command * ideal), min=0.0) * active
		aligned = torch.clamp(command * torch.sign(ideal), min=0.0) * active
		denom = float(ideal.abs()[active].sum()) if active.any() else 0.0
		return cls(opposition, aligned, denom, sample_rate, active)

	def x_opp(self, gate: torch.Tensor) -> float:
		return float((self.opposition * gate).sum()) / self.sample_rate

	def retention(self, gate: torch.Tensor) -> float:
		if self.denom <= 1e-6:
			return float("nan")
		return float((self.aligned * gate).sum()) / self.denom


# ---------------------------------------------------------------- 报告工具


def load_report(path: str | Path) -> dict:
	return json.loads(Path(path).read_text(encoding="utf-8"))


def write_tsv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
	"""列名取所有行键的并集，缺失写空——不同 stage 的行宽不同但要能同表读入。"""
	path.parent.mkdir(parents=True, exist_ok=True)
	if not rows:
		path.write_text("", encoding="utf-8")
		return
	columns: List[str] = []
	for row in rows:
		for key in row:
			if key not in columns:
				columns.append(key)
	lines = ["\t".join(columns)]
	for row in rows:
		lines.append("\t".join(_fmt(row.get(col)) for col in columns))
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: object) -> str:
	if value is None:
		return ""
	if isinstance(value, bool):
		return "1" if value else "0"
	if isinstance(value, float):
		if value != value:
			return "nan"
		return f"{value:.6g}"
	return str(value)
