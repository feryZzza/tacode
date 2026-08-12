"""Deterministic, mask-aware trial window sampling."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .adapters import TrialArrays, TrialSpec, load_trial


def _stable_seed(text: str, seed: int) -> int:
    digest = hashlib.blake2b(f"{seed}:{text}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


class TrialWindowDataset(Dataset):
    """Sample a small, deterministic set of valid windows per trial.

    Unlike the previous loader, this class does not scan every multi-gigabyte
    CSV before training can start. Each trial is loaded only when an item from
    it is requested, and a bounded LRU cache prevents repeated disk reads.
    Label validity is retained per target instead of requiring both joints to
    be valid, which is important for the documented unilateral invalid trials.
    """

    def __init__(
        self,
        records: Sequence[TrialSpec],
        feature_names: Sequence[str],
        domain_to_id: dict[str, int],
        window_size: int = 256,
        windows_per_trial: int = 2,
        min_valid_fraction: float = 0.25,
        candidates: int = 24,
        cache_size: int = 4,
        seed: int = 7,
        loader: Callable[[TrialSpec, Sequence[str]], TrialArrays] = load_trial,
    ):
        if window_size <= 0 or windows_per_trial <= 0:
            raise ValueError("window_size and windows_per_trial must be positive")
        if not 0.0 <= min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be in [0, 1]")
        self.records = list(records)
        self.feature_names = tuple(feature_names)
        self.domain_to_id = dict(domain_to_id)
        self.window_size = int(window_size)
        self.windows_per_trial = int(windows_per_trial)
        self.min_valid_fraction = float(min_valid_fraction)
        self.candidates = max(int(candidates), 1)
        self.cache_size = max(int(cache_size), 0)
        self.seed = int(seed)
        self.loader = loader
        self._cache: OrderedDict[int, TrialArrays] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records) * self.windows_per_trial

    @property
    def item_domain_ids(self) -> list[int]:
        return [self.domain_to_id[record.dataset] for record in self.records for _ in range(self.windows_per_trial)]

    def _load(self, trial_index: int) -> TrialArrays:
        if trial_index in self._cache:
            arrays = self._cache.pop(trial_index)
            self._cache[trial_index] = arrays
            return arrays
        arrays = self.loader(self.records[trial_index], self.feature_names)
        if arrays.x.shape[0] != len(self.feature_names):
            raise ValueError(f"Feature count mismatch for {self.records[trial_index].key}")
        if arrays.x.shape != arrays.feature_mask.shape:
            raise ValueError(f"Feature mask shape mismatch for {self.records[trial_index].key}")
        if arrays.y.shape != arrays.target_mask.shape or arrays.y.shape[0] != 2:
            raise ValueError(f"Target mask shape mismatch for {self.records[trial_index].key}")
        if arrays.x.shape[-1] != arrays.y.shape[-1]:
            raise ValueError(f"Input/target length mismatch for {self.records[trial_index].key}")
        if self.cache_size:
            self._cache[trial_index] = arrays
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return arrays

    def _choose_start(self, arrays: TrialArrays, record: TrialSpec, repeat: int) -> int:
        length = arrays.y.shape[-1]
        if length <= self.window_size:
            return 0
        max_start = length - self.window_size
        rng = np.random.default_rng(_stable_seed(f"{record.key}:{repeat}", self.seed))
        starts = rng.integers(0, max_start + 1, size=self.candidates)
        # Include boundary candidates so short valid islands at trial edges are
        # not systematically missed.
        starts = np.unique(np.concatenate(([0, max_start], starts)))
        best_start = int(starts[0])
        best_fraction = -1.0
        for start in starts:
            stop = int(start) + self.window_size
            fraction = float(arrays.target_mask[:, int(start):stop].mean())
            if fraction > best_fraction:
                best_start, best_fraction = int(start), fraction
            if fraction >= self.min_valid_fraction:
                return int(start)
        return best_start

    @staticmethod
    def _pad(array: np.ndarray, size: int, value: float | bool) -> np.ndarray:
        if array.shape[-1] >= size:
            return array[..., :size]
        padding = size - array.shape[-1]
        return np.pad(array, [(0, 0)] * (array.ndim - 1) + [(0, padding)], constant_values=value)

    def __getitem__(self, index: int) -> dict:
        trial_index, repeat = divmod(index, self.windows_per_trial)
        record = self.records[trial_index]
        arrays = self._load(trial_index)
        start = self._choose_start(arrays, record, repeat)
        stop = start + self.window_size
        x = self._pad(arrays.x[:, start:stop], self.window_size, 0.0)
        feature_mask = self._pad(arrays.feature_mask[:, start:stop], self.window_size, False)
        y = self._pad(arrays.y[:, start:stop], self.window_size, 0.0)
        target_mask = self._pad(arrays.target_mask[:, start:stop], self.window_size, False)
        time_mask = np.zeros(self.window_size, dtype=bool)
        real_length = min(max(arrays.x.shape[-1] - start, 0), self.window_size)
        time_mask[:real_length] = True
        return {
            "x": torch.from_numpy(np.ascontiguousarray(x)).float(),
            "feature_mask": torch.from_numpy(np.ascontiguousarray(feature_mask)),
            "y": torch.from_numpy(np.ascontiguousarray(y)).float(),
            "target_mask": torch.from_numpy(np.ascontiguousarray(target_mask)),
            "time_mask": torch.from_numpy(time_mask),
            "domain": torch.tensor(self.domain_to_id[record.dataset], dtype=torch.long),
            "dataset": record.dataset,
            "participant": record.participant,
            "trial": record.trial_id,
            "side": record.side,
            "mode": record.mode,
            "start": start,
        }
