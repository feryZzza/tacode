"""Canonical sensor and target schema shared by all three public datasets.

All adapters convert into SI units:

* joint angle: rad
* joint angular velocity and gyroscope: rad/s
* acceleration: m/s^2
* force normalized by body mass: N/kg
* joint moment: Nm/kg

The target convention is extension-positive for hip and knee.  Dataset-specific
sign changes are explicit in :mod:`openexo.adapters` instead of being hidden in
the training code.
"""

from __future__ import annotations

from collections import OrderedDict


CANONICAL_FEATURES = (
    "hip_angle",
    "hip_velocity",
    "pelvis_accel_x",
    "pelvis_accel_y",
    "pelvis_accel_z",
    "pelvis_gyro_x",
    "pelvis_gyro_y",
    "pelvis_gyro_z",
    "thigh_accel_x",
    "thigh_accel_y",
    "thigh_accel_z",
    "thigh_gyro_x",
    "thigh_gyro_y",
    "thigh_gyro_z",
    "shank_accel_x",
    "shank_accel_y",
    "shank_accel_z",
    "shank_gyro_x",
    "shank_gyro_y",
    "shank_gyro_z",
    "foot_accel_x",
    "foot_accel_y",
    "foot_accel_z",
    "foot_gyro_x",
    "foot_gyro_y",
    "foot_gyro_z",
    "foot_force",
    "foot_cop_x",
    "foot_cop_z",
)

CANONICAL_TARGETS = ("hip_extension_moment", "knee_extension_moment")

FEATURE_GROUPS = OrderedDict(
    (
        ("kinematic", ("hip_angle", "hip_velocity")),
        ("pelvis", tuple(name for name in CANONICAL_FEATURES if name.startswith("pelvis_"))),
        ("thigh", tuple(name for name in CANONICAL_FEATURES if name.startswith("thigh_"))),
        ("shank", tuple(name for name in CANONICAL_FEATURES if name.startswith("shank_"))),
        ("foot_imu", tuple(name for name in CANONICAL_FEATURES if name.startswith("foot_a") or name.startswith("foot_g"))),
        ("foot_contact", ("foot_force", "foot_cop_x", "foot_cop_z")),
    )
)

FEATURE_PROFILES = {
    # These eight channels are genuinely available in Nature, Dryad and
    # Camargo (Camargo uses its wearable hip goniometer for hip_angle).
    "minimal": (
        "hip_angle",
        "hip_velocity",
        "thigh_accel_x",
        "thigh_accel_y",
        "thigh_accel_z",
        "thigh_gyro_x",
        "thigh_gyro_y",
        "thigh_gyro_z",
    ),
    "imu": tuple(name for name in CANONICAL_FEATURES if "accel" in name or "gyro" in name),
    "hip_imu": tuple(name for name in CANONICAL_FEATURES if name in {"hip_angle", "hip_velocity"} or "accel" in name or "gyro" in name),
    "all": CANONICAL_FEATURES,
}


def features_for_profile(profile: str) -> tuple[str, ...]:
    """Return canonical feature names for a named sensor profile."""
    try:
        return FEATURE_PROFILES[profile]
    except KeyError as exc:
        choices = ", ".join(sorted(FEATURE_PROFILES))
        raise ValueError(f"Unknown feature profile {profile!r}; choose one of: {choices}") from exc


def feature_group_indices(feature_names: tuple[str, ...] | list[str]) -> dict[str, list[int]]:
    """Map available feature groups to column indices for sensor dropout."""
    lookup = {name: index for index, name in enumerate(feature_names)}
    groups: dict[str, list[int]] = {}
    for group, names in FEATURE_GROUPS.items():
        indices = [lookup[name] for name in names if name in lookup]
        if indices:
            groups[group] = indices
    return groups
