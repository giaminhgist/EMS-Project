"""Stage-2 checkpoints and exact resume (guide 07 §13, contracts §23).

Both best and last checkpoints store the complete resume state: model,
optimizer, scheduler, AMP scaler, epoch/global step/training phase, best-rule
tuple, early-stopping counter, RNG states, sampler state, subset-generator
state, ablation spec and overlay diff, plus all provenance checksums. Writes
are atomic; exact resume rejects any critical mismatch.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from stage1.reproducibility import capture_rng_state, restore_rng_state

from .history import checkpoint_file_hash

CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointError(ValueError):
    pass


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp)
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            os.fsync(dir_fd)
            os.close(dir_fd)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_stage2_checkpoint(
    path: Path | str,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    phase_epoch: int,
    global_step: int,
    training_phase: str,
    best_metric: float | None,
    best_epoch: int | None,
    best_rule_tuple: tuple,
    early_stopping_counter: int,
    optimizer_step_count: int,
    skipped_optimizer_step_count: int,
    fold: int,
    run_id: str,
    cfg: Any,
    config_hash: str,
    source_checksums: dict[str, str],
    stage1_checkpoint_path: str,
    stage1_checkpoint_sha256: str,
    bank_manifest_path: str,
    bank_checksums: dict[str, str],
    ablation_spec: dict[str, Any] | None,
    ablation_diff: list[str] | None,
    history_row_hash: str | None,
    calibration_state: dict[str, Any] | None,
    device: str,
    sampler_epoch: int,
) -> str:
    """Atomically write one complete checkpoint; returns its SHA-256."""
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "phase_epoch": int(phase_epoch),
        "global_step": int(global_step),
        "training_phase": training_phase,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "best_rule_tuple": list(best_rule_tuple),
        "early_stopping_counter": int(early_stopping_counter),
        "optimizer_step_count": int(optimizer_step_count),
        "skipped_optimizer_step_count": int(skipped_optimizer_step_count),
        "fold": int(fold),
        "run_id": run_id,
        "config_resolved": cfg.to_dict(),
        "config_hash": config_hash,
        "source_checksums": dict(source_checksums),
        "stage1_checkpoint_path": stage1_checkpoint_path,
        "stage1_checkpoint_sha256": stage1_checkpoint_sha256,
        "bank_manifest_path": bank_manifest_path,
        "bank_checksums": dict(bank_checksums),
        "ablation_spec": ablation_spec,
        "ablation_diff": ablation_diff,
        "history_row_hash": history_row_hash,
        "calibration_state": calibration_state,
        "rng_state": capture_rng_state(device),
        "sampler_epoch": int(sampler_epoch),
    }
    _atomic_save(payload, Path(path))
    return checkpoint_file_hash(Path(path))


@dataclass
class CheckpointContents:
    state_dict: dict[str, Any]
    meta: dict[str, Any]
    path: Path
    sha256: str


def load_stage2_checkpoint(path: Path | str, device: str = "cpu") -> CheckpointContents:
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise CheckpointError(f"{path}: not a Stage-2 checkpoint payload")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(
            f"{path}: checkpoint schema {payload.get('schema_version')!r} != "
            f"{CHECKPOINT_SCHEMA_VERSION}"
        )
    state_dict = payload.pop("model_state")
    sha256 = checkpoint_file_hash(path)
    return CheckpointContents(state_dict=state_dict, meta=payload, path=path, sha256=sha256)


def restore_rng_from_checkpoint(contents: CheckpointContents) -> None:
    rng = contents.meta.get("rng_state")
    if rng is None:
        raise CheckpointError("checkpoint lacks RNG state; exact resume impossible")
    restore_rng_state(rng)


def verify_resume_compatibility(
    contents: CheckpointContents,
    *,
    cfg: Any,
    fold: int,
    run_id: str,
    config_hash: str,
    ablation_name: str,
    stage1_checkpoint_sha256: str,
    bank_checksums: dict[str, str],
) -> None:
    """Reject exact resume on any critical mismatch (guide 07 §13)."""
    meta = contents.meta
    problems: list[str] = []
    if meta.get("fold") != fold:
        problems.append(f"fold {meta.get('fold')} != {fold}")
    if meta.get("run_id") != run_id:
        problems.append(f"run_id {meta.get('run_id')!r} != {run_id!r}")
    if meta.get("config_hash") != config_hash:
        problems.append("config hash differs")
    resolved = meta.get("config_resolved") or {}
    if resolved.get("ablation") != ablation_name:
        problems.append(
            f"ablation {resolved.get('ablation')!r} != {ablation_name!r}"
        )
    if resolved.get("seed") != cfg.seed:
        problems.append(f"seed {resolved.get('seed')} != {cfg.seed}")
    if meta.get("stage1_checkpoint_sha256") != stage1_checkpoint_sha256:
        problems.append("Stage-1 checkpoint SHA-256 differs")
    if meta.get("bank_checksums") != bank_checksums:
        problems.append("normative bank checksums differ")
    if problems:
        raise CheckpointError(
            "resume rejected: " + "; ".join(problems)
            + ". Use --load-stage2-weights for a different configuration."
        )


def load_weights_only_stage2(
    path: Path | str,
    model: Any,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Model tensors only into a fresh run (guide 07 §13, §17.13)."""
    contents = load_stage2_checkpoint(path, device)
    try:
        missing, unexpected = model.load_state_dict(contents.state_dict, strict=False)
    except RuntimeError as exc:
        raise CheckpointError(f"weight-load architecture mismatch: {exc}") from exc
    if missing or unexpected:
        raise CheckpointError(
            f"weight-load architecture mismatch: missing keys {missing}, "
            f"unexpected keys {unexpected}"
        )
    return contents.meta
