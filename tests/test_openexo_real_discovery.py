"""Read-only integration checks, skipped when the public datasets are absent."""

from collections import Counter
from pathlib import Path
import unittest

from openexo.adapters import discover_all, load_trial
from openexo.schema import features_for_profile


ROOT = Path("/media/fery/新加卷/Dataset")
ROOTS = {
    "nature": ROOT / "Task-Agnostic",
    "dryad": ROOT / "Dryad",
    "camargo": ROOT / "Camargo",
}


def _check_all_downloads_discover_expected_participants_and_trials():
    records = discover_all(ROOTS, task_scope="locomotion")
    counts = Counter(record.dataset for record in records)
    participants = {
        dataset: {record.participant for record in records if record.dataset == dataset}
        for dataset in ROOTS
    }
    assert counts == {"nature": 1500, "dryad": 10256, "camargo": 3147}
    assert {name: len(value) for name, value in participants.items()} == {
        "nature": 22,
        "dryad": 34,
        "camargo": 22,
    }


def _check_one_trial_per_csv_dataset_loads_canonical_minimal_profile():
    records = discover_all(ROOTS, datasets=("nature", "dryad"), task_scope="locomotion")
    feature_names = features_for_profile("minimal")
    for dataset in ("nature", "dryad"):
        candidates = (record for record in records if record.dataset == dataset)
        arrays = None
        for record in candidates:
            candidate = load_trial(record, feature_names)
            if candidate.target_mask.any():
                arrays = candidate
                break
        assert arrays is not None
        assert arrays.x.shape[0] == len(feature_names)
        assert arrays.x.shape == arrays.feature_mask.shape
        assert arrays.y.shape == arrays.target_mask.shape
        assert arrays.y.shape[0] == 2


@unittest.skipUnless(all(path.is_dir() for path in ROOTS.values()), "public datasets are not mounted")
class OpenExoRealDiscoveryTests(unittest.TestCase):
    def test_all_downloads_discover_expected_participants_and_trials(self):
        _check_all_downloads_discover_expected_participants_and_trials()

    def test_one_trial_per_csv_dataset_loads_canonical_minimal_profile(self):
        _check_one_trial_per_csv_dataset_loads_canonical_minimal_profile()
