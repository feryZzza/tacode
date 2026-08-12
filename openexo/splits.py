"""Subject-disjoint protocols for pooled and leave-one-dataset-out studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
import hashlib

from .adapters import TrialSpec


@dataclass(frozen=True)
class DataSplits:
    train: tuple[TrialSpec, ...]
    val: tuple[TrialSpec, ...]
    test: tuple[TrialSpec, ...]
    protocol: str
    target_dataset: str | None = None

    def as_dict(self) -> dict[str, tuple[TrialSpec, ...]]:
        return {"train": self.train, "val": self.val, "test": self.test}


def _rank(dataset: str, participant: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{dataset}:{participant}".encode()).hexdigest()


def _participants_by_dataset(records: Sequence[TrialSpec], seed: int) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = {}
    for record in records:
        grouped.setdefault(record.dataset, set()).add(record.participant)
    return {
        dataset: sorted(participants, key=lambda participant: _rank(dataset, participant, seed))
        for dataset, participants in grouped.items()
    }


def _select(records: Iterable[TrialSpec], subjects: set[str]) -> tuple[TrialSpec, ...]:
    return tuple(record for record in records if record.subject_key in subjects)


def pooled_subject_split(
    records: Sequence[TrialSpec],
    val_participants_per_dataset: int = 2,
    test_participants_per_dataset: int = 2,
    seed: int = 7,
) -> DataSplits:
    """Hold out validation and test people independently in every dataset."""
    participants = _participants_by_dataset(records, seed)
    train_subjects: set[str] = set()
    val_subjects: set[str] = set()
    test_subjects: set[str] = set()
    for dataset, people in participants.items():
        required = val_participants_per_dataset + test_participants_per_dataset + 1
        if len(people) < required:
            raise ValueError(f"{dataset} has {len(people)} participants; pooled split requires at least {required}")
        test = people[:test_participants_per_dataset]
        val = people[test_participants_per_dataset:test_participants_per_dataset + val_participants_per_dataset]
        train = people[test_participants_per_dataset + val_participants_per_dataset:]
        train_subjects.update(f"{dataset}:{person}" for person in train)
        val_subjects.update(f"{dataset}:{person}" for person in val)
        test_subjects.update(f"{dataset}:{person}" for person in test)
    return DataSplits(
        train=_select(records, train_subjects),
        val=_select(records, val_subjects),
        test=_select(records, test_subjects),
        protocol="pooled_subject",
    )


def leave_one_dataset_out_split(
    records: Sequence[TrialSpec],
    target_dataset: str,
    adaptation_participants: int = 0,
    source_val_participants: int = 1,
    target_val_participants: int = 1,
    seed: int = 7,
) -> DataSplits:
    """Evaluate zero-shot or few-participant adaptation on a target dataset.

    In the zero-shot case no target samples, including validation labels, are
    used: all target participants are test participants and hyperparameters are
    selected on source validation subjects. In the few-shot case, target train,
    validation and test participants are strictly disjoint.
    """
    participants = _participants_by_dataset(records, seed)
    if target_dataset not in participants:
        raise ValueError(f"Target dataset {target_dataset!r} is not present")
    target_people = participants[target_dataset]
    if adaptation_participants < 0:
        raise ValueError("adaptation_participants cannot be negative")
    if adaptation_participants > 0 and len(target_people) <= adaptation_participants + target_val_participants:
        raise ValueError("Not enough target participants for disjoint adaptation/val/test groups")

    train_subjects: set[str] = set()
    val_subjects: set[str] = set()
    test_subjects: set[str] = set()
    for dataset, people in participants.items():
        if dataset == target_dataset:
            if adaptation_participants == 0:
                test_subjects.update(f"{dataset}:{person}" for person in people)
            else:
                adapt = people[:adaptation_participants]
                target_val = people[adaptation_participants:adaptation_participants + target_val_participants]
                target_test = people[adaptation_participants + target_val_participants:]
                train_subjects.update(f"{dataset}:{person}" for person in adapt)
                val_subjects.update(f"{dataset}:{person}" for person in target_val)
                test_subjects.update(f"{dataset}:{person}" for person in target_test)
            continue
        if len(people) <= source_val_participants:
            raise ValueError(f"Not enough source participants in {dataset}")
        val = people[:source_val_participants]
        train = people[source_val_participants:]
        train_subjects.update(f"{dataset}:{person}" for person in train)
        val_subjects.update(f"{dataset}:{person}" for person in val)

    return DataSplits(
        train=_select(records, train_subjects),
        val=_select(records, val_subjects),
        test=_select(records, test_subjects),
        protocol="leave_one_dataset_out",
        target_dataset=target_dataset,
    )


def assert_subject_disjoint(splits: DataSplits) -> None:
    """Raise if any human appears in more than one split."""
    groups = {
        name: {record.subject_key for record in records}
        for name, records in splits.as_dict().items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = groups[left] & groups[right]
        if overlap:
            raise AssertionError(f"Subject leakage between {left} and {right}: {sorted(overlap)}")


def split_summary(splits: DataSplits) -> dict:
    result = {"protocol": splits.protocol, "target_dataset": splits.target_dataset}
    for split, records in splits.as_dict().items():
        by_dataset: dict[str, dict[str, object]] = {}
        for dataset in sorted({record.dataset for record in records}):
            selected = [record for record in records if record.dataset == dataset]
            by_dataset[dataset] = {
                "trials": len(selected),
                "participants": sorted({record.participant for record in selected}),
            }
        result[split] = by_dataset
    return result
