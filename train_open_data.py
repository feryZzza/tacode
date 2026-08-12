#!/usr/bin/env python3
"""Train/evaluate the cross-dataset public-data moment estimator."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
import math
import os
from pathlib import Path
import random
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from openexo.adapters import TrialSpec, discover_all
from openexo.dataset import TrialWindowDataset
from openexo.engine import evaluate, estimate_normalization, load_checkpoint, save_checkpoint, train_one_epoch
from openexo.model import MaskAwareMomentTCN
from openexo.schema import FEATURE_PROFILES, features_for_profile
from openexo.splits import (
    DataSplits,
    assert_subject_disjoint,
    leave_one_dataset_out_split,
    pooled_subject_split,
    split_summary,
)


DEFAULT_DATA_PARENT = Path("/media/fery/新加卷/Dataset")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mask-aware cross-dataset hip/knee moment estimation on Nature, Dryad and Camargo."
    )
    parser.add_argument("--mode", choices=("train", "smoke", "eval"), default="train")
    parser.add_argument("--nature-root", default=os.environ.get("NATURE_ROOT", str(DEFAULT_DATA_PARENT / "Task-Agnostic")))
    parser.add_argument("--dryad-root", default=os.environ.get("DRYAD_ROOT", str(DEFAULT_DATA_PARENT / "Dryad")))
    parser.add_argument("--camargo-root", default=os.environ.get("CAMARGO_ROOT", str(DEFAULT_DATA_PARENT / "Camargo")))
    parser.add_argument("--datasets", default="nature,dryad,camargo", help="Comma-separated dataset names.")
    parser.add_argument("--task-scope", choices=("locomotion", "all"), default="locomotion")
    parser.add_argument("--protocol", choices=("pooled", "lodo"), default="pooled")
    parser.add_argument("--target-dataset", choices=("nature", "dryad", "camargo"), default="camargo")
    parser.add_argument("--adaptation-participants", type=int, default=0)
    parser.add_argument(
        "--target-supervision",
        choices=("labels", "none"),
        default="labels",
        help="For LODO adaptation people: use their labels (few-shot) or features only (subject-disjoint UDA).",
    )
    parser.add_argument("--val-participants", type=int, default=2)
    parser.add_argument("--test-participants", type=int, default=2)
    parser.add_argument("--source-val-participants", type=int, default=1)
    parser.add_argument("--target-val-participants", type=int, default=1)
    parser.add_argument("--feature-profile", choices=tuple(FEATURE_PROFILES), default="minimal")
    parser.add_argument("--alignment", choices=("none", "dann"), default="dann")
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--windows-per-trial", type=int, default=2)
    parser.add_argument("--min-valid-fraction", type=float, default=0.25)
    parser.add_argument("--cache-size", type=int, default=4)
    parser.add_argument("--max-trials-per-dataset", type=int, default=0, help="Per split; 0 keeps all trials.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--channels", default="64,64,64,64")
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--domain-loss-weight", type=float, default=0.1)
    parser.add_argument("--domain-lambda", type=float, default=0.2)
    parser.add_argument("--sensor-dropout", type=float, default=0.15)
    parser.add_argument("--normalization-batches", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0, help="Debug limit per epoch/evaluation; 0 is unlimited.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="reports/open_data")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--checkpoint", default="", help="Required for --mode eval; optional output path for training.")
    return parser


def _parse_names(value: str, allowed: set[str]) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(names) - allowed
    if not names or unknown:
        raise ValueError(f"Invalid datasets {sorted(unknown)}; choose from {sorted(allowed)}")
    if len(names) != len(set(names)):
        raise ValueError("Dataset names must not be repeated")
    return names


def _parse_channels(value: str) -> tuple[int, ...]:
    channels = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not channels or any(channel <= 0 for channel in channels):
        raise ValueError("--channels must be a comma-separated list of positive integers")
    return channels


def _limit_records(records: Sequence[TrialSpec], limit_per_dataset: int) -> tuple[TrialSpec, ...]:
    """Round-robin participants so debug caps do not collapse to one person."""
    if limit_per_dataset <= 0:
        return tuple(records)
    by_dataset_subject: dict[str, dict[str, deque[TrialSpec]]] = defaultdict(lambda: defaultdict(deque))
    for record in records:
        by_dataset_subject[record.dataset][record.participant].append(record)
    selected: list[TrialSpec] = []
    for dataset in sorted(by_dataset_subject):
        subjects = by_dataset_subject[dataset]
        ordered = sorted(subjects)
        count = 0
        while count < limit_per_dataset and any(subjects[subject] for subject in ordered):
            for subject in ordered:
                if subjects[subject] and count < limit_per_dataset:
                    selected.append(subjects[subject].popleft())
                    count += 1
    return tuple(selected)


def _limit_splits(splits: DataSplits, limit: int) -> DataSplits:
    return DataSplits(
        train=_limit_records(splits.train, limit),
        val=_limit_records(splits.val, limit),
        test=_limit_records(splits.test, limit),
        protocol=splits.protocol,
        target_dataset=splits.target_dataset,
    )


def _loader(
    dataset: TrialWindowDataset,
    batch_size: int,
    workers: int,
    seed: int,
    balanced_domains: bool,
) -> DataLoader:
    sampler = None
    shuffle = False
    if balanced_domains and len(dataset):
        counts = Counter(dataset.item_domain_ids)
        weights = torch.tensor([1.0 / counts[item] for item in dataset.item_domain_ids], dtype=torch.double)
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _macro_rmse(metrics: dict) -> float:
    value = metrics.get("overall", {}).get("macro_rmse")
    return float(value) if value is not None else math.inf


def main() -> None:
    args = _parser().parse_args()
    datasets = _parse_names(args.datasets, {"nature", "dryad", "camargo"})
    channels = _parse_channels(args.channels)
    if args.protocol == "lodo" and args.target_dataset not in datasets:
        raise ValueError("--target-dataset must be included in --datasets")
    if args.target_supervision == "none" and args.adaptation_participants == 0:
        print("Note: no target adaptation participants were requested; this is source-only zero-shot evaluation.")
    if args.mode == "smoke":
        args.epochs = 1
        args.max_batches = args.max_batches or 2
        args.normalization_batches = args.normalization_batches or 2
        args.max_trials_per_dataset = args.max_trials_per_dataset or 6
        args.windows_per_trial = 1
        if args.channels == "64,64,64,64":
            channels = (16, 16)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    roots = {"nature": args.nature_root, "dryad": args.dryad_root, "camargo": args.camargo_root}
    records = discover_all(roots, datasets=datasets, task_scope=args.task_scope)
    if not records:
        raise FileNotFoundError("No trials discovered; verify the three dataset root arguments")
    print("Discovered trial legs:", dict(sorted(Counter(record.dataset for record in records).items())))

    if args.protocol == "pooled":
        splits = pooled_subject_split(
            records,
            val_participants_per_dataset=args.val_participants,
            test_participants_per_dataset=args.test_participants,
            seed=args.seed,
        )
    else:
        target_val = 0 if args.target_supervision == "none" else args.target_val_participants
        splits = leave_one_dataset_out_split(
            records,
            target_dataset=args.target_dataset,
            adaptation_participants=args.adaptation_participants,
            source_val_participants=args.source_val_participants,
            target_val_participants=target_val,
            seed=args.seed,
        )
    assert_subject_disjoint(splits)
    full_split_summary = split_summary(splits)
    splits = _limit_splits(splits, args.max_trials_per_dataset)
    if not splits.train or not splits.val or not splits.test:
        raise ValueError("The selected protocol produced an empty train, validation or test split")
    limited_split_summary = split_summary(splits)
    print("Working trial legs:", {name: len(value) for name, value in splits.as_dict().items()})

    feature_names = features_for_profile(args.feature_profile)
    domain_to_id = {name: index for index, name in enumerate(sorted(datasets))}
    dataset_kwargs = {
        "feature_names": feature_names,
        "domain_to_id": domain_to_id,
        "window_size": args.window_size,
        "windows_per_trial": args.windows_per_trial,
        "min_valid_fraction": args.min_valid_fraction,
        "cache_size": args.cache_size,
        "seed": args.seed,
    }
    train_dataset = TrialWindowDataset(splits.train, **dataset_kwargs)
    val_dataset = TrialWindowDataset(splits.val, **dataset_kwargs)
    test_dataset = TrialWindowDataset(splits.test, **dataset_kwargs)
    stats_loader = _loader(train_dataset, args.batch_size, args.workers, args.seed, balanced_domains=False)
    train_loader = _loader(train_dataset, args.batch_size, args.workers, args.seed, balanced_domains=True)
    val_loader = _loader(val_dataset, args.batch_size, args.workers, args.seed, balanced_domains=False)
    test_loader = _loader(test_dataset, args.batch_size, args.workers, args.seed, balanced_domains=False)

    run_name = args.run_name or (
        "smoke" if args.mode == "smoke" else f"{args.protocol}_{args.target_dataset if args.protocol == 'lodo' else 'all'}"
    )
    run_dir = Path(args.output_dir) / run_name
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "best.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args).copy()
    run_config.update({
        "datasets_parsed": list(datasets),
        "channels_parsed": list(channels),
        "feature_names": list(feature_names),
        "domain_to_id": domain_to_id,
        "device_resolved": str(device),
    })
    _write_json(run_dir / "config.json", run_config)
    _write_json(run_dir / "splits.json", {"full": full_split_summary, "working": limited_split_summary})

    if args.mode == "eval":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required with --mode eval")
        payload = load_checkpoint(checkpoint_path, device)
        model = MaskAwareMomentTCN(**payload["model_config"]).to(device)
        model.load_state_dict(payload["state_dict"])
        if tuple(model.feature_names) != feature_names:
            raise ValueError("Checkpoint feature schema does not match --feature-profile")
        metrics = evaluate(model, test_loader, device, model.effective_history, max_batches=args.max_batches)
        _write_json(run_dir / "test_metrics.json", metrics)
        print(json.dumps(metrics["overall"], indent=2))
        return

    model = MaskAwareMomentTCN(
        feature_names=feature_names,
        channels=channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        num_domains=len(domain_to_id),
    ).to(device)
    if model.effective_history >= args.window_size:
        raise ValueError(
            f"Model history ({model.effective_history}) must be smaller than window size ({args.window_size}); "
            "reduce --channels depth/kernel size or increase --window-size"
        )
    center, scale = estimate_normalization(
        stats_loader, len(feature_names), max_batches=args.normalization_batches
    )
    model.set_normalization(center, scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    supervised_domain_ids = set(domain_to_id.values())
    if args.protocol == "lodo" and args.target_supervision == "none":
        supervised_domain_ids.remove(domain_to_id[args.target_dataset])

    history: list[dict] = []
    best_score = math.inf
    for epoch in range(1, args.epochs + 1):
        # A one-epoch smoke run still exercises gradient reversal; longer runs
        # ramp it smoothly from a small value toward the requested maximum.
        progress = epoch / max(args.epochs, 1)
        if args.alignment == "dann":
            domain_lambda = args.domain_lambda * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
            domain_weight = args.domain_loss_weight
        else:
            domain_lambda = 0.0
            domain_weight = 0.0
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            domain_lambda=domain_lambda,
            domain_loss_weight=domain_weight,
            sensor_dropout=args.sensor_dropout,
            ignore_history=model.effective_history,
            supervised_domain_ids=supervised_domain_ids,
            max_batches=args.max_batches,
        )
        val_metrics = evaluate(model, val_loader, device, model.effective_history, max_batches=args.max_batches)
        epoch_record = {
            "epoch": epoch,
            "domain_lambda": domain_lambda,
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(epoch_record)
        score = _macro_rmse(val_metrics)
        print(
            f"epoch={epoch:03d} train={train_metrics['regression_loss']:.5f} "
            f"val_macro_rmse={score:.5f} domain_lambda={domain_lambda:.3f}"
        )
        if score < best_score:
            best_score = score
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                args.feature_profile,
                domain_to_id,
                metadata={"validation_macro_rmse": score, "protocol": args.protocol},
            )
        _write_json(run_dir / "history.json", history)
    if not checkpoint_path.is_file():
        raise RuntimeError("No checkpoint was saved because validation contained no valid labels")
    best = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(best["state_dict"])
    test_metrics = evaluate(model, test_loader, device, model.effective_history, max_batches=args.max_batches)
    _write_json(run_dir / "test_metrics.json", test_metrics)
    print("Best epoch:", best["epoch"])
    print(json.dumps(test_metrics["overall"], indent=2))


if __name__ == "__main__":
    main()
