"""Durable per-epoch training history (contract §6).

``history.csv`` is rewritten atomically (temp file + flush + fsync + replace)
immediately after every completed epoch, with the full required column set.
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any

HISTORY_COLUMNS: list[str] = [
    "run_id", "fold", "epoch", "training_phase", "eligible_for_best", "is_best_epoch",
    "best_epoch_so_far", "learning_rate", "learning_rate_min", "learning_rate_max",
    "weight_decay", "lambda_norm", "train_loss", "train_recon_loss",
    "train_recon_fixation", "train_recon_transition", "train_recon_temporal",
    "train_norm_loss", "train_spread_loss", "val_loss", "val_recon_loss",
    "val_recon_fixation", "val_recon_transition", "val_recon_temporal",
    "val_norm_loss", "val_spread_loss",
    "train_within_stimulus_dispersion", "train_between_stimulus_dispersion",
    "val_within_stimulus_dispersion", "val_between_stimulus_dispersion",
    "grad_norm_mean", "grad_norm_max", "grad_clip_fraction",
    "semantic_gamma_attention1", "semantic_gamma_attention2", "spatial_bridge_eta",
    "train_mask_ratio_realized", "val_mask_ratio_realized", "num_train_trials",
    "num_val_trials", "n_train_batches", "n_val_batches", "n_train_stimulus_groups",
    "n_val_stimulus_groups", "n_skipped_norm_groups_train", "n_skipped_norm_groups_val",
    "nonfinite_batch_count", "epoch_time_seconds", "peak_gpu_memory_mb", "seed",
]


class HistoryError(ValueError):
    pass


def validate_history(df_rows: list[dict[str, Any]]) -> None:
    """Uniqueness of (fold, epoch) and monotonic epoch order."""
    seen: set[tuple[object, object]] = set()
    prev_epoch = None
    for row in df_rows:
        key = (row["fold"], row["epoch"])
        if key in seen:
            raise HistoryError(f"duplicate history row for (fold, epoch) = {key}")
        seen.add(key)
        epoch = int(row["epoch"])
        if prev_epoch is not None and epoch != prev_epoch + 1:
            raise HistoryError(f"non-monotonic epoch order: {prev_epoch} -> {epoch}")
        prev_epoch = epoch


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return ""
    return str(value)


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically replace ``path`` with the full history table."""
    validate_history(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=HISTORY_COLUMNS, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                if set(row) - set(HISTORY_COLUMNS):
                    raise HistoryError(f"unknown history columns: {sorted(set(row) - set(HISTORY_COLUMNS))}")
                writer.writerow({k: _fmt(row.get(k)) for k in HISTORY_COLUMNS})
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, "r", newline="") as fh:
        return list(csv.DictReader(fh))


def row_from_metrics(
    run_id: str,
    fold: int,
    epoch: int,
    training_phase: str,
    eligible_for_best: bool,
    is_best_epoch: bool,
    best_epoch_so_far: int | None,
    learning_rate: float,
    lr_bounds: tuple[float, float],
    weight_decay: float,
    lambda_norm: float,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": run_id,
        "fold": fold,
        "epoch": epoch,
        "training_phase": training_phase,
        "eligible_for_best": eligible_for_best,
        "is_best_epoch": is_best_epoch,
        "best_epoch_so_far": best_epoch_so_far if best_epoch_so_far is not None else "",
        "learning_rate": learning_rate,
        "learning_rate_min": lr_bounds[0],
        "learning_rate_max": lr_bounds[1],
        "weight_decay": weight_decay,
        "lambda_norm": lambda_norm,
        "seed": extra.get("seed"),
    }
    for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
        row[f"{prefix}_loss"] = metrics.get(f"{prefix}_loss")
        row[f"{prefix}_recon_loss"] = metrics.get(f"{prefix}_recon_loss")
        row[f"{prefix}_recon_fixation"] = metrics.get(f"{prefix}_recon_fixation")
        row[f"{prefix}_recon_transition"] = metrics.get(f"{prefix}_recon_transition")
        row[f"{prefix}_recon_temporal"] = metrics.get(f"{prefix}_recon_temporal")
        row[f"{prefix}_norm_loss"] = metrics.get(f"{prefix}_norm_loss")
        row[f"{prefix}_spread_loss"] = metrics.get(f"{prefix}_spread_loss")
        row[f"{prefix}_within_stimulus_dispersion"] = metrics.get(f"{prefix}_within_stimulus_dispersion")
        row[f"{prefix}_between_stimulus_dispersion"] = metrics.get(f"{prefix}_between_stimulus_dispersion")
    row["semantic_gamma_attention1"] = extra.get("semantic_gamma_attention1")
    row["semantic_gamma_attention2"] = extra.get("semantic_gamma_attention2")
    row["spatial_bridge_eta"] = extra.get("spatial_bridge_eta")
    row["train_mask_ratio_realized"] = extra.get("train_mask_ratio_realized")
    row["val_mask_ratio_realized"] = extra.get("val_mask_ratio_realized")
    row["num_train_trials"] = extra.get("num_train_trials")
    row["num_val_trials"] = extra.get("num_val_trials")
    row["n_train_batches"] = extra.get("n_train_batches")
    row["n_val_batches"] = extra.get("n_val_batches")
    row["n_train_stimulus_groups"] = extra.get("n_train_stimulus_groups")
    row["n_val_stimulus_groups"] = extra.get("n_val_stimulus_groups")
    row["n_skipped_norm_groups_train"] = extra.get("n_skipped_norm_groups_train")
    row["n_skipped_norm_groups_val"] = extra.get("n_skipped_norm_groups_val")
    row["nonfinite_batch_count"] = extra.get("nonfinite_batch_count")
    row["epoch_time_seconds"] = extra.get("epoch_time_seconds")
    row["peak_gpu_memory_mb"] = extra.get("peak_gpu_memory_mb")
    row["grad_norm_mean"] = extra.get("grad_norm_mean")
    row["grad_norm_max"] = extra.get("grad_norm_max")
    row["grad_clip_fraction"] = extra.get("grad_clip_fraction")
    return row
