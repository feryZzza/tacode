from pathlib import Path
import unittest

import numpy as np
import torch

from openexo.adapters import TrialArrays, TrialSpec
from openexo.dataset import TrialWindowDataset
from openexo.engine import apply_sensor_dropout, masked_smooth_l1
from openexo.model import MaskAwareMomentTCN
from openexo.schema import CANONICAL_TARGETS, features_for_profile
from openexo.splits import assert_subject_disjoint, leave_one_dataset_out_split, pooled_subject_split


FEATURES = features_for_profile("minimal")


def _record(dataset: str, participant: str, trial: str = "trial") -> TrialSpec:
    path = Path("unused")
    return TrialSpec(dataset, participant, trial, "r", "level_ground", path, path)


def _fake_loader(record: TrialSpec, feature_names: tuple[str, ...]) -> TrialArrays:
    del record
    length = 80
    x = np.arange(len(feature_names) * length, dtype=np.float32).reshape(len(feature_names), length)
    feature_mask = np.ones_like(x, dtype=bool)
    y = np.ones((2, length), dtype=np.float32)
    target_mask = np.ones_like(y, dtype=bool)
    target_mask[1, :40] = False
    return TrialArrays(x, feature_mask, y, target_mask, np.arange(length) / 200, tuple(feature_names))


def _check_window_dataset_preserves_per_target_masks_and_is_deterministic():
    record = _record("nature", "S1")
    dataset = TrialWindowDataset(
        [record], FEATURES, {"nature": 0}, window_size=32, windows_per_trial=2, seed=11, loader=_fake_loader
    )
    first = dataset[0]
    repeated = dataset[0]
    assert torch.equal(first["x"], repeated["x"])
    assert first["target_mask"].shape == (2, 32)
    assert first["target_mask"][0].all()
    assert not torch.equal(first["target_mask"][0], first["target_mask"][1])


def _check_subject_splits_do_not_leak_people():
    records = [
        _record(dataset, f"S{subject}", f"T{trial}")
        for dataset in ("nature", "dryad", "camargo")
        for subject in range(7)
        for trial in range(2)
    ]
    pooled = pooled_subject_split(records, val_participants_per_dataset=1, test_participants_per_dataset=1)
    assert_subject_disjoint(pooled)
    lodo = leave_one_dataset_out_split(records, "camargo", adaptation_participants=2, target_val_participants=1)
    assert_subject_disjoint(lodo)
    assert len({record.participant for record in lodo.train if record.dataset == "camargo"}) == 2
    assert len({record.participant for record in lodo.val if record.dataset == "camargo"}) == 1


def _check_masked_loss_ignores_history_and_missing_joint():
    prediction = torch.zeros(1, 2, 10, requires_grad=True)
    target = torch.ones_like(prediction)
    valid = torch.ones_like(prediction, dtype=torch.bool)
    valid[:, 1] = False
    loss, count = masked_smooth_l1(prediction, target, valid, ignore_history=3)
    assert count == 7
    assert torch.isclose(loss, torch.tensor(0.5))
    loss.backward()
    assert prediction.grad[:, :, :3].abs().sum() == 0


def _check_model_forward_backward_and_sensor_dropout():
    model = MaskAwareMomentTCN(FEATURES, channels=(8, 8), num_domains=3)
    x = torch.randn(4, len(FEATURES), 64)
    mask = torch.ones_like(x, dtype=torch.bool)
    torch.manual_seed(1)
    dropped = apply_sensor_dropout(mask, FEATURES, probability=0.5)
    assert dropped.shape == mask.shape
    output = model(x, dropped, torch.ones(4, 64, dtype=torch.bool), domain_lambda=0.2)
    assert output["moments"].shape == (4, len(CANONICAL_TARGETS), 64)
    assert output["domain_logits"].shape == (4, 3)
    (output["moments"].mean() + output["domain_logits"].mean()).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


class OpenExoCoreTests(unittest.TestCase):
    def test_window_dataset_preserves_per_target_masks_and_is_deterministic(self):
        _check_window_dataset_preserves_per_target_masks_and_is_deterministic()

    def test_subject_splits_do_not_leak_people(self):
        _check_subject_splits_do_not_leak_people()

    def test_masked_loss_ignores_history_and_missing_joint(self):
        _check_masked_loss_ignores_history_and_missing_joint()

    def test_model_forward_backward_and_sensor_dropout(self):
        _check_model_forward_backward_and_sensor_dropout()
