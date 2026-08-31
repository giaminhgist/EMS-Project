"""Crash-safe checkpoints with exact resume and weight-only loading (contract §8)."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .reproducibility import capture_rng_state, restore_rng_state


class CheckpointError(ValueError):
    pass


@dataclass
class CheckpointContents:
    state_dict: dict[str, Any]
    meta: dict[str, Any]


def save_checkpoint(
    path: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    *,
    epoch: int,
    global_step: int,
    best_metric: float | None,
    best_epoch: int | None,
    fold: int,
    run_id: str,
    cfg: Any,
    source_checksums: dict[str, str],
    device: str,
) -> None:
    """Atomically write a complete last/best checkpoint."""
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "fold": fold,
        "run_id": run_id,
        "config_resolved": cfg.to_dict(),
        "config_hash": cfg.config_hash(),
        "source_checksums": source_checksums,
        "rng_state": capture_rng_state(device),
        "sampler_epoch": epoch,  # deterministic epoch state (sampler.set_epoch)
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_checkpoint(path: Path, device: str = "cpu") -> CheckpointContents:
    if not path.is_file():
        raise CheckpointError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    return CheckpointContents(
        state_dict=payload["model_state"], meta={k: v for k, v in payload.items() if k != "model_state"}
    )


def resume_checkpoint(
    path: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    *,
    fold: int,
    run_id: str,
    cfg: Any,
    source_checksums: dict[str, str],
    device: str,
) -> dict[str, Any]:
    """Restore a complete checkpoint, rejecting identity mismatches.

    Rejections: fold, run ID, resolved-config hash, architecture (strict state
    dict), and source-manifest checksums.
    """
    contents = load_checkpoint(path, device)
    meta = contents.meta
    if meta.get("fold") != fold:
        raise CheckpointError(
            f"checkpoint fold {meta.get('fold')} does not match requested fold {fold}"
        )
    if meta.get("run_id") != run_id:
        raise CheckpointError(
            f"checkpoint run_id {meta.get('run_id')!r} does not match {run_id!r}"
        )
    if meta.get("config_hash") != cfg.config_hash():
        raise CheckpointError("checkpoint config hash does not match the resolved configuration")
    for key, expected in source_checksums.items():
        if meta.get("source_checksums", {}).get(key) != expected:
            raise CheckpointError(
                f"checkpoint source checksum {key} does not match current inputs"
            )

    missing, unexpected = _strict_state_dict_compare(model, contents.state_dict)
    if missing or unexpected:
        raise CheckpointError(
            f"architecture mismatch: missing keys {missing}, unexpected keys {unexpected}"
        )
    model.load_state_dict(contents.state_dict)
    if optimizer is not None:
        if meta.get("optimizer_state") is None:
            raise CheckpointError("checkpoint has no optimizer state for exact resume")
        optimizer.load_state_dict(meta["optimizer_state"])
    if scheduler is not None and meta.get("scheduler_state") is not None:
        scheduler.load_state_dict(meta["scheduler_state"])
    if scaler is not None and meta.get("scaler_state") is not None:
        scaler.load_state_dict(meta["scaler_state"])
    if meta.get("rng_state"):
        restore_rng_state(meta["rng_state"])
    return meta


def load_weights_only(
    path: Path,
    model: Any,
    *,
    fold: int | None = None,
    cfg: Any | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Initialize model weights only; fresh optimizer/scheduler/run.

    Reports missing/unexpected keys and requires the exact architecture by
    default (strict load).
    """
    contents = load_checkpoint(path, device)
    missing, unexpected = _strict_state_dict_compare(model, contents.state_dict)
    if missing or unexpected:
        raise CheckpointError(
            f"weight-load architecture mismatch: missing keys {missing}, unexpected keys {unexpected}"
        )
    if fold is not None and contents.meta.get("fold") != fold:
        raise CheckpointError(
            f"checkpoint fold {contents.meta.get('fold')} does not match requested fold {fold}"
        )
    model.load_state_dict(contents.state_dict)
    return contents.meta


def _strict_state_dict_compare(
    model: Any, state_dict: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Report key AND shape mismatches (architecture identity)."""
    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())
    ckpt_keys = set(state_dict.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    for key in sorted(model_keys & ckpt_keys):
        if model_sd[key].shape != state_dict[key].shape:
            missing.append(f"{key} (shape {state_dict[key].shape} != {model_sd[key].shape})")
    return missing, unexpected
