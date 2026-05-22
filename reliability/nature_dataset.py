"""Nature 2024 Parsed 数据的窗口化读取器。"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


TASK_PREFIXES = [
	"ball_toss",
	"curb_down",
	"curb_up",
	"cutting",
	"dynamic_walk",
	"incline_walk",
	"jump",
	"lift_weight",
	"lunges",
	"meander",
	"normal_walk",
	"obstacle_walk",
	"poses",
	"push",
	"side_shuffle",
	"sit_to_stand",
	"squats",
	"stairs",
	"start_stop",
	"step_ups",
	"tire_run",
	"tug_of_war",
	"turn_and_step",
	"walk_backward",
	"weighted_walk",
]


@dataclass(frozen=True)
class TrialRecord:
	participant: str
	trial_name: str
	task_family: str
	assistance: str
	trial_dir: Path
	exo_path: Path
	moment_path: Path


@dataclass(frozen=True)
class WindowRecord:
	trial_index: int
	start: int
	stop: int


def parse_task_family(trial_name: str) -> str:
	"""从 trial 名提取任务大类。"""
	for prefix in TASK_PREFIXES:
		if trial_name.startswith(prefix):
			return prefix
	return trial_name.split("_")[0]


def parse_assistance(trial_name: str) -> str:
	if trial_name.endswith("_on"):
		return "on"
	if trial_name.endswith("_off"):
		return "off"
	return "unknown"


def discover_trials(root: str | Path, require_bio: bool = True) -> List[TrialRecord]:
	"""扫描 Parsed 目录，匹配 exo 与 biological moment 文件。"""
	root = Path(root).expanduser()
	if not root.exists():
		raise FileNotFoundError(f"Dataset root does not exist: {root}")

	records: List[TrialRecord] = []
	for participant_dir in sorted(path for path in root.iterdir() if path.is_dir()):
		participant = participant_dir.name
		for trial_dir in sorted(path for path in participant_dir.iterdir() if path.is_dir()):
			exo_paths = sorted(trial_dir.glob("*_exo.csv"))
			if not exo_paths:
				continue
			moment_paths = sorted(trial_dir.glob("*_moment_filt_bio.csv"))
			if not moment_paths and not require_bio:
				moment_paths = sorted(trial_dir.glob("*_moment_filt.csv"))
			if not moment_paths:
				continue
			trial_name = trial_dir.name
			records.append(
				TrialRecord(
					participant=participant,
					trial_name=trial_name,
					task_family=parse_task_family(trial_name),
					assistance=parse_assistance(trial_name),
					trial_dir=trial_dir,
					exo_path=exo_paths[0],
					moment_path=moment_paths[0],
				)
			)
	return records


def filter_records(
	records: Sequence[TrialRecord],
	participants: Optional[Iterable[str]] = None,
	include_tasks: Optional[Iterable[str]] = None,
	exclude_tasks: Optional[Iterable[str]] = None,
	assistance: Optional[str] = None,
) -> List[TrialRecord]:
	participant_set = set(participants) if participants else None
	include_set = set(include_tasks) if include_tasks else None
	exclude_set = set(exclude_tasks) if exclude_tasks else None
	filtered = []
	for record in records:
		if participant_set is not None and record.participant not in participant_set:
			continue
		if include_set is not None and record.task_family not in include_set:
			continue
		if exclude_set is not None and record.task_family in exclude_set:
			continue
		if assistance is not None and record.assistance != assistance:
			continue
		filtered.append(record)
	return filtered


def participant_split(
	records: Sequence[TrialRecord],
	val_count: int = 2,
	test_count: int = 2,
) -> Tuple[List[str], List[str], List[str]]:
	"""按被试划分，减少同人泄漏。"""
	participants = sorted({record.participant for record in records})
	if len(participants) < val_count + test_count + 1:
		raise ValueError("Not enough participants for requested train/val/test split.")
	test = participants[-test_count:] if test_count else []
	val = participants[-(test_count + val_count) : -test_count] if val_count else []
	train = [participant for participant in participants if participant not in set(val + test)]
	return train, val, test


class ParsedWindowDataset(Dataset):
	"""把长 trial 切成固定长度窗口。"""

	def __init__(
		self,
		records: Sequence[TrialRecord],
		input_names: Sequence[str],
		label_names: Sequence[str],
		side: str = "r",
		participant_masses: Optional[Dict[str, float]] = None,
		window_size: int = 512,
		stride: int = 256,
		min_valid_fraction: float = 0.5,
		max_windows_per_trial: Optional[int] = None,
		cache_size: int = 6,
	):
		self.records = list(records)
		self.input_names = list(input_names)
		self.label_names = list(label_names)
		self.side = side
		self.participant_masses = participant_masses or {}
		self.window_size = window_size
		self.stride = stride
		self.min_valid_fraction = min_valid_fraction
		self.max_windows_per_trial = max_windows_per_trial
		self.cache_size = cache_size
		self._cache: OrderedDict[int, Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
		self.windows = self._build_windows()

	def __len__(self) -> int:
		return len(self.windows)

	def __getitem__(self, index: int) -> dict:
		window = self.windows[index]
		record = self.records[window.trial_index]
		x, y = self._load_trial(window.trial_index)
		xw = x[:, window.start : window.stop].clone()
		yw = y[:, window.start : window.stop].clone()
		mask = torch.isfinite(yw)
		xw = torch.nan_to_num(xw, nan=0.0, posinf=0.0, neginf=0.0)
		yw = torch.nan_to_num(yw, nan=0.0, posinf=0.0, neginf=0.0)
		return {
			"x": xw,
			"y": yw,
			"mask": mask,
			"participant": record.participant,
			"trial": record.trial_name,
			"task": record.task_family,
			"assistance": record.assistance,
			"start": window.start,
		}

	def _build_windows(self) -> List[WindowRecord]:
		"""只保留标签足够完整的窗口。"""
		windows: List[WindowRecord] = []
		for trial_index, record in enumerate(self.records):
			length = self._count_rows(record.exo_path)
			if length < self.window_size:
				continue
			valid = self._label_valid_mask(record, length)
			starts = list(range(0, length - self.window_size + 1, self.stride))
			kept_for_trial = 0
			for start in starts:
				if self.min_valid_fraction > 0:
					valid_fraction = valid[start : start + self.window_size].float().mean().item()
					if valid_fraction < self.min_valid_fraction:
						continue
				windows.append(WindowRecord(trial_index, start, start + self.window_size))
				kept_for_trial += 1
				if self.max_windows_per_trial is not None and kept_for_trial >= self.max_windows_per_trial:
					break
		return windows

	@staticmethod
	def _count_rows(csv_path: Path) -> int:
		with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
			return max(sum(1 for _ in handle) - 1, 0)

	def _load_trial(self, trial_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
		if trial_index in self._cache:
			x, y = self._cache.pop(trial_index)
			self._cache[trial_index] = (x, y)
			return x, y

		record = self.records[trial_index]
		exo_df = pd.read_csv(record.exo_path)
		moment_df = pd.read_csv(record.moment_path)
		exo_df = self._prepare_exo_dataframe(exo_df, record.participant)
		x = torch.tensor(exo_df[self.input_names].values, dtype=torch.float32).transpose(0, 1)
		y = torch.tensor(moment_df[self.label_names].values, dtype=torch.float32).transpose(0, 1)

		if x.shape[-1] != y.shape[-1]:
			length = min(x.shape[-1], y.shape[-1])
			x = x[:, :length]
			y = y[:, :length]

		self._cache[trial_index] = (x, y)
		while len(self._cache) > self.cache_size:
			self._cache.popitem(last=False)
		return x, y

	def _label_valid_mask(self, record: TrialRecord, exo_length: int) -> torch.Tensor:
		moment_df = pd.read_csv(record.moment_path, usecols=self.label_names)
		missing = [name for name in self.label_names if name not in moment_df.columns]
		if missing:
			raise KeyError(f"Missing label columns in parsed moment file: {missing}")
		valid = torch.tensor(moment_df[self.label_names].notna().all(axis=1).values, dtype=torch.bool)
		if valid.numel() != exo_length:
			length = min(valid.numel(), exo_length)
			padded = torch.zeros(exo_length, dtype=torch.bool)
			padded[:length] = valid[:length]
			return padded
		return valid

	def _prepare_exo_dataframe(self, df: pd.DataFrame, participant: str) -> pd.DataFrame:
		df = df.copy()
		body_mass = self.participant_masses.get(participant, 1.0)
		# 鞋垫力按体重归一化，对齐 Nature baseline。
		for column in ("insole_l_force_y", "insole_r_force_y"):
			if column in df.columns:
				df.loc[:, column] = df[column] / body_mass

		if self.side == "l":
			# 左腿镜像到右腿坐标，复用同一套模型输入定义。
			mirror_columns = [
				"foot_imu_l_gyro_x",
				"foot_imu_l_gyro_y",
				"foot_imu_l_accel_z",
				"shank_imu_l_gyro_x",
				"shank_imu_l_gyro_y",
				"shank_imu_l_accel_z",
				"thigh_imu_l_gyro_x",
				"thigh_imu_l_gyro_y",
				"thigh_imu_l_accel_z",
				"insole_l_cop_z",
			]
			for column in mirror_columns:
				if column in df.columns:
					df.loc[:, column] = -df[column]

		missing = [name for name in self.input_names if name not in df.columns]
		if missing:
			raise KeyError(f"Missing input columns in parsed exo file: {missing[:8]}")
		return df


def summarize_records(records: Sequence[TrialRecord]) -> dict:
	participants = sorted({record.participant for record in records})
	tasks = sorted({record.task_family for record in records})
	assist = {state: sum(record.assistance == state for record in records) for state in ("on", "off", "unknown")}
	return {
		"num_trials": len(records),
		"num_participants": len(participants),
		"participants": participants,
		"tasks": tasks,
		"assistance_counts": assist,
	}
