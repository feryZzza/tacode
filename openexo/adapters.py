"""Dataset discovery and canonicalization for Nature, Dryad and Camargo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import math
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .schema import CANONICAL_TARGETS


GRAVITY = 9.80665
DEG2RAD = math.pi / 180.0

NATURE_TASKS = (
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
)

LOCOMOTION_NATURE_TASKS = {"normal_walk", "incline_walk", "stairs"}
LOCOMOTION_DRYAD_MODES = {"LG", "RA", "RD", "SA", "SD"}

# Participant metadata shipped with the public releases.  Keeping these values
# beside the adapters makes every force/moment normalization auditable and
# avoids silently falling back to an incorrect 1 kg body mass.
NATURE_MASSES_KG = {
    "BT01": 80.59, "BT02": 72.24, "BT03": 95.29, "BT04": 98.23,
    "BT06": 79.33, "BT07": 64.49, "BT08": 69.13, "BT09": 82.31,
    "BT10": 93.45, "BT11": 50.39, "BT12": 78.15, "BT13": 89.85,
    "BT14": 67.30, "BT15": 58.40, "BT16": 64.33, "BT17": 60.03,
    "BT18": 67.96, "BT19": 69.95, "BT20": 55.44, "BT21": 58.85,
    "BT22": 76.79, "BT23": 67.23, "BT24": 77.79,
}

CAMARGO_MASSES_KG = {
    "AB06": 74.84, "AB07": 55.34, "AB08": 72.57, "AB09": 63.50,
    "AB10": 83.91, "AB11": 77.11, "AB12": 86.18, "AB13": 58.97,
    "AB14": 58.41, "AB15": 96.16, "AB16": 55.79, "AB17": 61.23,
    "AB18": 60.13, "AB19": 68.04, "AB20": 68.04, "AB21": 58.06,
    "AB23": 76.82, "AB24": 72.57, "AB25": 52.16, "AB27": 68.04,
    "AB28": 62.14, "AB30": 77.03,
}


@dataclass(frozen=True)
class TrialSpec:
    """One leg of one time-synchronized trial."""

    dataset: str
    participant: str
    trial_id: str
    side: str
    mode: str
    sensor_path: Path
    target_path: Path
    kinematic_path: Path | None = None
    body_mass: float = 1.0
    sample_rate: float = 200.0

    @property
    def subject_key(self) -> str:
        return f"{self.dataset}:{self.participant}"

    @property
    def key(self) -> str:
        return f"{self.dataset}:{self.participant}:{self.trial_id}:{self.side}"


@dataclass(frozen=True)
class TrialArrays:
    """Canonical arrays returned by each adapter."""

    x: np.ndarray
    feature_mask: np.ndarray
    y: np.ndarray
    target_mask: np.ndarray
    time: np.ndarray
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...] = CANONICAL_TARGETS


def _nature_task(trial_name: str) -> str:
    for task in NATURE_TASKS:
        if trial_name.startswith(task):
            return task
    return trial_name.split("_")[0]


def _nature_mode(trial_name: str) -> str:
    task = _nature_task(trial_name)
    if task == "normal_walk":
        return "level_ground"
    if task == "incline_walk":
        if "_down" in trial_name:
            return "ramp_descent"
        if "_up" in trial_name:
            return "ramp_ascent"
        return "ramp"
    if task == "stairs":
        if "_down" in trial_name:
            return "stair_descent"
        if "_up" in trial_name:
            return "stair_ascent"
        return "stair"
    return task


def discover_nature(root: str | Path, task_scope: str = "locomotion", sides: Sequence[str] = ("l", "r")) -> list[TrialSpec]:
    """Discover the two official Nature Parsed releases.

    Returning participants BT01/BT02/BT13 keep the same participant id across
    phases so subject-level splitting cannot leak their Phase 1/2 data into a
    Phase 3 test fold.
    """
    root = Path(root).expanduser()
    phase_dirs = [root / "Phase1And2_Parsed", root / "Phase3_Parsed"]
    if root.name in {"Phase1And2_Parsed", "Phase3_Parsed"}:
        phase_dirs = [root]
    records: list[TrialSpec] = []
    for phase_dir in phase_dirs:
        if not phase_dir.is_dir():
            continue
        phase = phase_dir.name.replace("_Parsed", "")
        for participant_dir in sorted(path for path in phase_dir.iterdir() if path.is_dir()):
            participant = participant_dir.name
            for trial_dir in sorted(path for path in participant_dir.iterdir() if path.is_dir()):
                task = _nature_task(trial_dir.name)
                if task_scope == "locomotion" and task not in LOCOMOTION_NATURE_TASKS:
                    continue
                prefix = f"{participant}_{trial_dir.name}"
                sensor_path = trial_dir / f"{prefix}_exo.csv"
                target_path = trial_dir / f"{prefix}_moment_filt_bio.csv"
                if not sensor_path.is_file() or not target_path.is_file():
                    continue
                for side in sides:
                    records.append(
                        TrialSpec(
                            dataset="nature",
                            participant=participant,
                            trial_id=f"{phase}/{trial_dir.name}",
                            side=side,
                            mode=_nature_mode(trial_dir.name),
                            sensor_path=sensor_path,
                            target_path=target_path,
                            body_mass=NATURE_MASSES_KG.get(participant, 1.0),
                        )
                    )
    return records


def _dryad_mode(trial_name: str) -> str:
    code = trial_name.split("_", 1)[0]
    return {
        "LG": "level_ground",
        "RA": "ramp_ascent",
        "RD": "ramp_descent",
        "SA": "stair_ascent",
        "SD": "stair_descent",
        "ST": "standing",
        "TR": "transition",
        "TRA": "transition",
        "TRB": "transition",
    }.get(code, code.lower())


def discover_dryad(root: str | Path, task_scope: str = "locomotion", sides: Sequence[str] = ("l", "r")) -> list[TrialSpec]:
    root = Path(root).expanduser()
    records: list[TrialSpec] = []
    for participant_dir in sorted(path for path in root.glob("AB*") if path.is_dir()):
        for trial_dir in sorted(path for path in participant_dir.iterdir() if path.is_dir()):
            mode_code = trial_dir.name.split("_", 1)[0]
            if task_scope == "locomotion" and mode_code not in LOCOMOTION_DRYAD_MODES:
                continue
            sensor_path = trial_dir / "exo.csv"
            target_path = trial_dir / "moment.csv"
            if not sensor_path.is_file() or not target_path.is_file():
                continue
            for side in sides:
                records.append(
                    TrialSpec(
                        dataset="dryad",
                        participant=participant_dir.name,
                        trial_id=trial_dir.name,
                        side=side,
                        mode=_dryad_mode(trial_dir.name),
                        sensor_path=sensor_path,
                        target_path=target_path,
                    )
                )
    return records


def _osim_mass(subject_dir: Path) -> float:
    models = list((subject_dir / "osimxml").glob("*.osim"))
    if not models:
        return 1.0
    try:
        root = ET.parse(models[0]).getroot()
        masses = []
        for body in root.iter("Body"):
            element = body.find("mass")
            if element is not None and element.text:
                masses.append(float(element.text))
        return sum(masses) if masses else 1.0
    except (ET.ParseError, ValueError, OSError):
        return 1.0


def discover_camargo(root: str | Path, task_scope: str = "locomotion") -> list[TrialSpec]:
    """Discover right-side Camargo wearable trials.

    Camargo's four IMUs and goniometers were recorded on the right side only,
    so creating artificial left-side examples would duplicate inputs with an
    unobserved target and is intentionally avoided.
    """
    del task_scope  # all four Camargo modes are locomotion modes
    root = Path(root).expanduser()
    records: list[TrialSpec] = []
    for participant_dir in sorted(path for path in root.glob("AB*") if path.is_dir()):
        body_mass = CAMARGO_MASSES_KG.get(participant_dir.name, _osim_mass(participant_dir))
        for date_dir in sorted(path for path in participant_dir.iterdir() if path.is_dir() and path.name != "osimxml"):
            for mode_dir in sorted(path for path in date_dir.iterdir() if path.is_dir()):
                imu_dir = mode_dir / "imu"
                id_dir = mode_dir / "id"
                gon_dir = mode_dir / "gon"
                if not imu_dir.is_dir() or not id_dir.is_dir() or not gon_dir.is_dir():
                    continue
                for sensor_path in sorted(imu_dir.glob("*.mat")):
                    target_path = id_dir / sensor_path.name
                    kinematic_path = gon_dir / sensor_path.name
                    if not target_path.is_file() or not kinematic_path.is_file():
                        continue
                    records.append(
                        TrialSpec(
                            dataset="camargo",
                            participant=participant_dir.name,
                            trial_id=f"{date_dir.name}/{mode_dir.name}/{sensor_path.stem}",
                            side="r",
                            mode=mode_dir.name,
                            sensor_path=sensor_path,
                            target_path=target_path,
                            kinematic_path=kinematic_path,
                            body_mass=body_mass,
                        )
                    )
    return records


def discover_all(
    roots: Mapping[str, str | Path],
    datasets: Iterable[str] = ("nature", "dryad", "camargo"),
    task_scope: str = "locomotion",
) -> list[TrialSpec]:
    """Discover selected datasets using one consistent task scope."""
    records: list[TrialSpec] = []
    for dataset in datasets:
        if dataset not in roots:
            raise KeyError(f"Missing root for dataset {dataset!r}")
        if dataset == "nature":
            records.extend(discover_nature(roots[dataset], task_scope=task_scope))
        elif dataset == "dryad":
            records.extend(discover_dryad(roots[dataset], task_scope=task_scope))
        elif dataset == "camargo":
            records.extend(discover_camargo(roots[dataset], task_scope=task_scope))
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
    return records


def _empty_channels(feature_names: Sequence[str], length: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((len(feature_names), length), dtype=np.float32)
    mask = np.zeros_like(x, dtype=bool)
    return x, mask


def _assign(
    x: np.ndarray,
    mask: np.ndarray,
    feature_names: Sequence[str],
    name: str,
    values: np.ndarray | pd.Series,
    scale: float = 1.0,
) -> None:
    if name not in feature_names:
        return
    index = feature_names.index(name)
    array = np.asarray(values, dtype=np.float64) * scale
    valid = np.isfinite(array)
    x[index, valid] = array[valid].astype(np.float32)
    mask[index] = valid


def _check_time(time: np.ndarray, expected_length: int, path: Path) -> None:
    if time.ndim != 1 or len(time) != expected_length:
        raise ValueError(f"Invalid time vector in {path}")
    if len(time) > 1 and np.any(np.diff(time) <= 0):
        raise ValueError(f"Non-monotonic time vector in {path}")


def _load_nature(record: TrialSpec, feature_names: tuple[str, ...]) -> TrialArrays:
    sensor = pd.read_csv(record.sensor_path)
    target = pd.read_csv(record.target_path)
    if len(sensor) != len(target):
        raise ValueError(f"Nature input/target length mismatch: {record.key}")
    time = sensor["time"].to_numpy(dtype=np.float64)
    if not np.array_equal(time, target["time"].to_numpy(dtype=np.float64)):
        raise ValueError(f"Nature input/target time mismatch: {record.key}")
    _check_time(time, len(sensor), record.sensor_path)
    x, feature_mask = _empty_channels(feature_names, len(sensor))
    side = record.side
    _assign(x, feature_mask, feature_names, "hip_angle", sensor[f"hip_angle_{side}"], DEG2RAD)
    _assign(x, feature_mask, feature_names, "hip_velocity", sensor[f"hip_angle_{side}_velocity_filt"], DEG2RAD)
    for segment in ("thigh", "shank", "foot"):
        for signal, scale in (("accel", 1.0), ("gyro", DEG2RAD)):
            for axis in "xyz":
                # Reflect left local IMUs into the right-leg sensor frame used
                # by the released controller (same convention as its original
                # inference loader).
                mirror = -1.0 if record.side == "l" and (
                    (signal == "gyro" and axis in "xy") or (signal == "accel" and axis == "z")
                ) else 1.0
                _assign(
                    x,
                    feature_mask,
                    feature_names,
                    f"{segment}_{signal}_{axis}",
                    sensor[f"{segment}_imu_{side}_{signal}_{axis}"],
                    scale * mirror,
                )
    _assign(x, feature_mask, feature_names, "foot_force", sensor[f"insole_{side}_force_y"], 1.0 / max(record.body_mass, 1.0))
    _assign(x, feature_mask, feature_names, "foot_cop_x", sensor[f"insole_{side}_cop_x"])
    _assign(x, feature_mask, feature_names, "foot_cop_z", sensor[f"insole_{side}_cop_z"])

    # Nature's biological convention is hip-flexion positive but knee-extension
    # positive. Convert only the hip channel to the canonical extension-positive
    # convention.
    y = np.stack(
        (
            -target[f"hip_flexion_{side}_moment"].to_numpy(dtype=np.float32),
            target[f"knee_angle_{side}_moment"].to_numpy(dtype=np.float32),
        )
    )
    target_mask = np.isfinite(y)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return TrialArrays(x, feature_mask, y, target_mask, time, feature_names)


def _load_dryad(record: TrialSpec, feature_names: tuple[str, ...]) -> TrialArrays:
    sensor = pd.read_csv(record.sensor_path)
    target = pd.read_csv(record.target_path)
    if len(sensor) != len(target):
        raise ValueError(f"Dryad input/target length mismatch: {record.key}")
    time = sensor["time"].to_numpy(dtype=np.float64)
    if not np.array_equal(time, target["time"].to_numpy(dtype=np.float64)):
        raise ValueError(f"Dryad input/target time mismatch: {record.key}")
    _check_time(time, len(sensor), record.sensor_path)
    x, feature_mask = _empty_channels(feature_names, len(sensor))
    side = record.side
    _assign(x, feature_mask, feature_names, "hip_angle", sensor[f"enc_angle_{side}"])
    _assign(x, feature_mask, feature_names, "hip_velocity", sensor[f"enc_velo_{side}"])
    for segment in ("thigh",):
        for signal in ("accel", "gyro"):
            for axis in "xyz":
                mirror = -1.0 if record.side == "l" and (
                    (signal == "gyro" and axis in "xy") or (signal == "accel" and axis == "z")
                ) else 1.0
                _assign(
                    x,
                    feature_mask,
                    feature_names,
                    f"{segment}_{signal}_{axis}",
                    sensor[f"{segment}_{signal}_{axis}_{side}"],
                    mirror,
                )
    for signal in ("accel", "gyro"):
        for axis in "xyz":
            # Dryad documents pelvis axes in the ground frame (x lateral,
            # y vertical, z posterior). Reflecting a left-leg sample changes
            # vector accel-x and pseudovector gyro-y/z signs.
            mirror = -1.0 if record.side == "l" and (
                (signal == "accel" and axis == "x") or (signal == "gyro" and axis in "yz")
            ) else 1.0
            _assign(
                x,
                feature_mask,
                feature_names,
                f"pelvis_{signal}_{axis}",
                sensor[f"pelvis_{signal}_{axis}"],
                mirror,
            )
    y = np.stack(
        (
            target[f"hip_moment_{side}"].to_numpy(dtype=np.float32),
            target[f"knee_moment_{side}"].to_numpy(dtype=np.float32),
        )
    )
    target_mask = np.isfinite(y)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return TrialArrays(x, feature_mask, y, target_mask, time, feature_names)


def _load_mat_table(path: Path) -> pd.DataFrame:
    try:
        from matio import load_from_mat
    except ImportError as exc:
        raise ImportError(
            "Camargo uses MATLAB MCOS table objects. Install the optional dependency "
            "with `pip install 'mat-io>=1.0.0'`. Older mat-io releases do not decode "
            "these specific files."
        ) from exc
    payload = load_from_mat(path)
    tables = [value for value in payload.values() if isinstance(value, pd.DataFrame)]
    if len(tables) != 1:
        raise ValueError(f"Expected one MATLAB table in {path}, found {len(tables)}")
    return tables[0]


def _interp(time: np.ndarray, source_time: np.ndarray, values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(source_time) & np.isfinite(values)
    if valid.sum() < 2:
        return np.full_like(time, np.nan, dtype=np.float64)
    return np.interp(time, source_time[valid], values[valid], left=np.nan, right=np.nan)


def _load_camargo(record: TrialSpec, feature_names: tuple[str, ...]) -> TrialArrays:
    if record.kinematic_path is None:
        raise ValueError(f"Camargo record lacks goniometer path: {record.key}")
    imu = _load_mat_table(record.sensor_path)
    target = _load_mat_table(record.target_path)
    gon = _load_mat_table(record.kinematic_path)
    time = imu["Header"].to_numpy(dtype=np.float64)
    target_time = target["Header"].to_numpy(dtype=np.float64)
    if len(time) != len(target_time) or not np.allclose(time, target_time, rtol=0.0, atol=1e-9):
        raise ValueError(f"Camargo IMU/ID time mismatch: {record.key}")
    _check_time(time, len(imu), record.sensor_path)
    x, feature_mask = _empty_channels(feature_names, len(imu))

    gon_time = gon["Header"].to_numpy(dtype=np.float64)
    # Camargo goniometer hip flexion is positive. Negate it to match the
    # extension-positive exoskeleton encoder convention, then differentiate on
    # the synchronized 200 Hz grid.
    hip_angle = -_interp(time, gon_time, gon["hip_sagittal"].to_numpy(dtype=np.float64)) * DEG2RAD
    hip_velocity = np.gradient(hip_angle, time) if len(time) > 1 else np.zeros_like(hip_angle)
    _assign(x, feature_mask, feature_names, "hip_angle", hip_angle)
    _assign(x, feature_mask, feature_names, "hip_velocity", hip_velocity)

    segment_map = {"trunk": "pelvis", "thigh": "thigh", "shank": "shank", "foot": "foot"}
    for source_segment, canonical_segment in segment_map.items():
        for signal, scale in (("Accel", GRAVITY), ("Gyro", 1.0)):
            for axis in "XYZ":
                source = f"{source_segment}_{signal}_{axis}"
                name = f"{canonical_segment}_{signal.lower()}_{axis.lower()}"
                _assign(x, feature_mask, feature_names, name, imu[source], scale)

    mass = max(float(record.body_mass), 1.0)
    # Raw OpenSim generalized forces follow each coordinate: hip_flexion is
    # flexion-positive, whereas knee_angle is extension-positive (flexed knee
    # angles are negative in the supplied IK tables). Convert only the hip and
    # normalize Nm by subject mass.
    y = np.stack(
        (
            -target["hip_flexion_r_moment"].to_numpy(dtype=np.float32) / mass,
            target["knee_angle_r_moment"].to_numpy(dtype=np.float32) / mass,
        )
    )
    target_mask = np.isfinite(y)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return TrialArrays(x, feature_mask, y, target_mask, time, feature_names)


def load_trial(record: TrialSpec, feature_names: Sequence[str]) -> TrialArrays:
    """Load and convert one trial into the canonical schema."""
    names = tuple(feature_names)
    if record.dataset == "nature":
        return _load_nature(record, names)
    if record.dataset == "dryad":
        return _load_dryad(record, names)
    if record.dataset == "camargo":
        return _load_camargo(record, names)
    raise ValueError(f"Unsupported dataset: {record.dataset}")
