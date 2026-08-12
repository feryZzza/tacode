"""Training, normalization and masked regression evaluation utilities."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .schema import CANONICAL_TARGETS, feature_group_indices


def masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    ignore_history: int = 0,
) -> tuple[torch.Tensor, int]:
    """Smooth-L1 over only finite, labelled samples and outside TCN warm-up."""
    mask = valid.bool() & torch.isfinite(target) & torch.isfinite(prediction)
    if ignore_history:
        mask = mask.clone()
        mask[..., :ignore_history] = False
    count = int(mask.sum().item())
    if not count:
        # Retain a valid autograd graph for an unusually sparse batch.
        return prediction.sum() * 0.0, 0
    losses = F.smooth_l1_loss(prediction, target, reduction="none")
    return losses[mask].mean(), count


def apply_sensor_dropout(
    feature_mask: torch.Tensor,
    feature_names: Sequence[str],
    probability: float,
) -> torch.Tensor:
    """Drop complete physical sensor groups independently for each example."""
    if probability <= 0.0:
        return feature_mask
    if probability >= 1.0:
        raise ValueError("sensor dropout probability must be below 1")
    output = feature_mask.clone()
    batch_size = output.shape[0]
    for indices in feature_group_indices(tuple(feature_names)).values():
        dropped = torch.rand(batch_size, device=output.device) < probability
        if dropped.any():
            batch_indices = dropped.nonzero(as_tuple=False).squeeze(1)
            for feature_index in indices:
                output[batch_indices, feature_index, :] = False
    return output


@torch.no_grad()
def estimate_normalization(loader: Iterable[Mapping], feature_count: int, max_batches: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate masked channel means/scales from training data only."""
    total = torch.zeros(feature_count, dtype=torch.float64)
    total_sq = torch.zeros(feature_count, dtype=torch.float64)
    count = torch.zeros(feature_count, dtype=torch.float64)
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        values = batch["x"].to(torch.float64)
        valid = batch["feature_mask"].bool() & torch.isfinite(values)
        numeric = torch.where(valid, values, torch.zeros_like(values))
        total += numeric.sum(dim=(0, 2))
        total_sq += numeric.square().sum(dim=(0, 2))
        count += valid.sum(dim=(0, 2))
    if not torch.any(count):
        raise ValueError("No valid training features were found while estimating normalization")
    safe_count = count.clamp_min(1.0)
    center = total / safe_count
    variance = (total_sq / safe_count - center.square()).clamp_min(0.0)
    scale = variance.sqrt()
    center = torch.where(count > 0, center, torch.zeros_like(center))
    scale = torch.where((count > 1) & (scale > 1e-5), scale, torch.ones_like(scale))
    return center.float(), scale.float()


def train_one_epoch(
    model: torch.nn.Module,
    loader: Iterable[Mapping],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    domain_lambda: float,
    domain_loss_weight: float,
    sensor_dropout: float,
    ignore_history: int,
    supervised_domain_ids: set[int] | None = None,
    max_batches: int = 0,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_regression = 0.0
    total_domain = 0.0
    labelled_samples = 0
    batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        x = batch["x"].to(device)
        feature_mask = batch["feature_mask"].to(device)
        target = batch["y"].to(device)
        target_mask = batch["target_mask"].to(device)
        time_mask = batch["time_mask"].to(device)
        domain = batch["domain"].to(device)
        if supervised_domain_ids is not None:
            supervised = torch.tensor(
                [int(item) in supervised_domain_ids for item in domain.tolist()],
                dtype=torch.bool,
                device=device,
            )
            target_mask = target_mask & supervised[:, None, None]
        feature_mask = apply_sensor_dropout(feature_mask, model.feature_names, sensor_dropout)

        optimizer.zero_grad(set_to_none=True)
        output = model(x, feature_mask, time_mask=time_mask, domain_lambda=domain_lambda)
        regression_loss, valid_count = masked_smooth_l1(
            output["moments"], target, target_mask, ignore_history=ignore_history
        )
        if domain_loss_weight:
            domain_loss = F.cross_entropy(output["domain_logits"], domain)
        else:
            domain_loss = regression_loss.detach() * 0.0
        loss = regression_loss + domain_loss_weight * domain_loss
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.detach())
        total_regression += float(regression_loss.detach())
        total_domain += float(domain_loss.detach())
        labelled_samples += valid_count
        batches += 1
    if not batches:
        raise ValueError("Training loader produced no batches")
    return {
        "loss": total_loss / batches,
        "regression_loss": total_regression / batches,
        "domain_loss": total_domain / batches,
        "labelled_samples": labelled_samples,
        "batches": batches,
    }


class _RegressionSums:
    def __init__(self, target_names: Sequence[str]):
        count = len(target_names)
        self.target_names = tuple(target_names)
        self.n = np.zeros(count, dtype=np.int64)
        self.abs_error = np.zeros(count, dtype=np.float64)
        self.sq_error = np.zeros(count, dtype=np.float64)
        self.y_sum = np.zeros(count, dtype=np.float64)
        self.y_sq_sum = np.zeros(count, dtype=np.float64)

    def update(self, prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> None:
        for target_index in range(len(self.target_names)):
            mask = valid[target_index] & np.isfinite(prediction[target_index]) & np.isfinite(target[target_index])
            if not np.any(mask):
                continue
            estimate = prediction[target_index, mask].astype(np.float64)
            truth = target[target_index, mask].astype(np.float64)
            error = estimate - truth
            self.n[target_index] += int(mask.sum())
            self.abs_error[target_index] += np.abs(error).sum()
            self.sq_error[target_index] += np.square(error).sum()
            self.y_sum[target_index] += truth.sum()
            self.y_sq_sum[target_index] += np.square(truth).sum()

    def metrics(self) -> dict:
        result: dict[str, object] = {}
        valid_rmse: list[float] = []
        for index, name in enumerate(self.target_names):
            count = int(self.n[index])
            if not count:
                result[name] = {"count": 0, "rmse": None, "mae": None, "r2": None}
                continue
            rmse = float(np.sqrt(self.sq_error[index] / count))
            mae = float(self.abs_error[index] / count)
            denominator = self.y_sq_sum[index] - self.y_sum[index] ** 2 / count
            r2 = None if denominator <= 0 else float(1.0 - self.sq_error[index] / denominator)
            result[name] = {"count": count, "rmse": rmse, "mae": mae, "r2": r2}
            valid_rmse.append(rmse)
        result["macro_rmse"] = float(np.mean(valid_rmse)) if valid_rmse else None
        return result


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Iterable[Mapping],
    device: torch.device,
    ignore_history: int,
    max_batches: int = 0,
) -> dict:
    """Report sample-weighted target metrics overall and by dataset/mode."""
    model.eval()
    overall = _RegressionSums(CANONICAL_TARGETS)
    by_dataset: dict[str, _RegressionSums] = defaultdict(lambda: _RegressionSums(CANONICAL_TARGETS))
    by_mode: dict[str, _RegressionSums] = defaultdict(lambda: _RegressionSums(CANONICAL_TARGETS))
    by_subject: dict[str, _RegressionSums] = defaultdict(lambda: _RegressionSums(CANONICAL_TARGETS))
    correct_domains = 0
    domain_examples = 0
    batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        output = model(
            batch["x"].to(device),
            batch["feature_mask"].to(device),
            time_mask=batch["time_mask"].to(device),
            domain_lambda=0.0,
        )
        prediction = output["moments"].cpu().numpy()
        target = batch["y"].numpy()
        valid = batch["target_mask"].numpy().astype(bool)
        if ignore_history:
            valid[..., :ignore_history] = False
        domain_prediction = output["domain_logits"].argmax(dim=1).cpu()
        correct_domains += int((domain_prediction == batch["domain"]).sum())
        domain_examples += len(domain_prediction)
        for index, (dataset, participant, mode) in enumerate(
            zip(batch["dataset"], batch["participant"], batch["mode"])
        ):
            overall.update(prediction[index], target[index], valid[index])
            by_dataset[str(dataset)].update(prediction[index], target[index], valid[index])
            by_mode[str(mode)].update(prediction[index], target[index], valid[index])
            by_subject[f"{dataset}:{participant}"].update(prediction[index], target[index], valid[index])
        batches += 1
    return {
        "overall": overall.metrics(),
        "by_dataset": {name: sums.metrics() for name, sums in sorted(by_dataset.items())},
        "by_mode": {name: sums.metrics() for name, sums in sorted(by_mode.items())},
        "by_subject": {name: sums.metrics() for name, sums in sorted(by_subject.items())},
        "domain_accuracy": correct_domains / domain_examples if domain_examples else None,
        "batches": batches,
    }


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    feature_profile: str,
    domain_to_id: Mapping[str, int],
    metadata: Mapping | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "epoch": int(epoch),
        "model_config": model.checkpoint_config(),
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "feature_profile": feature_profile,
        "domain_to_id": dict(domain_to_id),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    return torch.load(Path(path), map_location=device, weights_only=False)
