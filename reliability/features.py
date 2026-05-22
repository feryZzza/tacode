"""Nature 2024 数据的输入特征定义。"""

from __future__ import annotations

from typing import Iterable, List

from configs.default_config import input_names as nature_human_inputs
from configs.default_config import label_names as nature_label_names
from configs.default_config import participant_masses


ACTION_FEATURE_GROUPS = {
	"desired": ["hip_angle_*_torque_desired", "knee_angle_*_torque_desired"],
	"measured": ["hip_angle_*_torque_measured", "knee_angle_*_torque_measured"],
	"estimated": ["hip_angle_*_torque_estimated", "knee_angle_*_torque_estimated"],
	"interaction": ["hip_angle_*_torque_interaction", "knee_angle_*_torque_interaction"],
}


# 统一控制 action 消融入口，避免在训练脚本里手写列名。
INPUT_PROFILES = {
	"human": [],
	"human_desired": ["desired"],
	"human_measured": ["measured"],
	"human_execution": ["desired", "measured"],
	"human_interaction": ["desired", "measured", "interaction"],
	"full": ["desired", "measured", "estimated", "interaction"],
}


def resolve_side(names: Iterable[str], side: str) -> List[str]:
	return [name.replace("*", side) for name in names]


def input_names_for_profile(profile: str, side: str = "r") -> List[str]:
	if profile not in INPUT_PROFILES:
		valid = ", ".join(sorted(INPUT_PROFILES))
		raise ValueError(f"Unknown input profile '{profile}'. Valid profiles: {valid}")

	names = list(nature_human_inputs)
	for group_name in INPUT_PROFILES[profile]:
		names.extend(ACTION_FEATURE_GROUPS[group_name])
	return resolve_side(names, side)


def label_names(side: str = "r") -> List[str]:
	return resolve_side(nature_label_names, side)


def feature_groups(feature_names: List[str]) -> dict[str, List[int]]:
	"""按传感器模态分组，用于故障注入。"""
	groups = {
		"foot_imu": [],
		"shank_imu": [],
		"thigh_imu": [],
		"insole": [],
		"encoder": [],
		"action": [],
	}
	for idx, name in enumerate(feature_names):
		if "foot_imu" in name:
			groups["foot_imu"].append(idx)
		elif "shank_imu" in name:
			groups["shank_imu"].append(idx)
		elif "thigh_imu" in name:
			groups["thigh_imu"].append(idx)
		elif "insole" in name:
			groups["insole"].append(idx)
		elif "torque" in name:
			groups["action"].append(idx)
		elif "hip_angle" in name or "knee_angle" in name:
			groups["encoder"].append(idx)
	return {name: indices for name, indices in groups.items() if indices}
